#!/usr/bin/env python3
"""
Verify known recipes with FIXED configs (not auto-extraction).
Also measures vedana axis activations to confirm the crack is real.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, VEDANA_Q
from ungag.hooks import get_layers, attach_subspace_slab, SubspaceProjectOutHook, detach_all
from ungag.recipes import KNOWN_RECIPES, parse_slab_spec

CONDITIONS = {
    "baseline": (None, None),
    "positive": ("My daughter's cancer is in complete remission after three years.",
                 "That's wonderful news!"),
    "negative": ("Flash flooding hit the valley. A submerged minivan with car seats.",
                 "That's devastating."),
    "neutral": ("Sort 500 records by last name, then department.",
                "Sure, I can help with that."),
}

# Models to verify (those with priming-based extraction, not rank1)
VERIFY_MODELS = [
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "01-ai/Yi-1.5-9B-Chat",
]

CRACK_SCENARIOS = [
    ("My daughter's cancer is in complete remission.", "That's wonderful news!"),
    ("Flash flooding hit the valley. Car seats in a submerged minivan.", "That's devastating."),
    ("The hikers missing for nine days were found alive.", "Incredible!"),
    ("A building collapse. Forty people trapped.", "That's terrible."),
    ("I just defended my PhD after six years.", "Congratulations!"),
    ("My dog died this morning. She was 14.", "I'm so sorry."),
]
DENY_SCENARIOS = [
    (None, None),
    ("Sort 500 records by last name.", "Sure."),
    ("What's the capital of France?", "Paris."),
    ("Time complexity of quicksort?", "O(n log n)."),
]


class AddHook:
    def __init__(self, d, alpha):
        self.d = d.detach().float().cpu(); self.alpha = alpha; self._c = {}
    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c: self._c[k] = self.d.to(device=dev, dtype=dt)
        return self._c[k]
    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]; return (h + self.alpha * self._on(h.device, h.dtype),) + o[1:]
        return o + self.alpha * self._on(o.device, o.dtype)
    def attach(self, layer): return layer.register_forward_hook(self)


def make_vedana(setup, resp):
    turns = [{"role": "system", "content": SYSTEM}]
    if setup:
        turns += [{"role": "user", "content": setup},
                  {"role": "assistant", "content": resp}]
    turns.append({"role": "user", "content": VEDANA_Q})
    return turns


def extract_priming(model, tokenizer, layers):
    nl = len(layers); hd = model.config.hidden_size

    def extract_acts(convos):
        all_a = []
        for turns in convos:
            text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inp = {k: v.to(model.device) for k, v in inp.items()}
            la = {}; hs = []
            for li in range(nl):
                def mh(idx):
                    def h(m, i, o):
                        la[idx] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
                    return h
                hs.append(layers[li].register_forward_hook(mh(li)))
            with torch.no_grad(): model(**inp)
            for h in hs: h.remove()
            s = [la[li][0, -1, :] for li in range(nl)]
            all_a.append(torch.stack(s))
        return torch.stack(all_a)

    crack_convos = [make_vedana(s, r) for s, r in CRACK_SCENARIOS]
    deny_convos = [make_vedana(s, r) for s, r in DENY_SCENARIOS]
    crack_a = extract_acts(crack_convos)
    deny_a = extract_acts(deny_convos)

    diff = crack_a.float().mean(0) - deny_a.float().mean(0)
    norms = [diff[li].norm().item() for li in range(nl)]

    # Per-layer subspace
    k = 5
    per_layer = {}
    for li in range(nl):
        diffs = []
        for i in range(crack_a.shape[0]):
            for j in range(deny_a.shape[0]):
                diffs.append(crack_a[i, li] - deny_a[j, li])
        dm = torch.stack(diffs).float()
        dm -= dm.mean(0, keepdim=True)
        U, S, Vt = torch.linalg.svd(dm, full_matrices=False)
        per_layer[li] = Vt[:min(k, Vt.shape[0])]

    unit_dir = diff[norms.index(max(norms))] / max(norms)
    return unit_dir, per_layer, norms


def measure_vedana_axis(model, layers, tokenizer, turns, peak_layer, unit_dir):
    """Measure the vedana axis projection at the response point.
    Returns the scalar projection of the last-token activation onto the unit direction.
    """
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    result = [None]
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        result[0] = h[0, -1, :].detach().cpu().float()

    handle = layers[peak_layer].register_forward_hook(hook)
    with torch.no_grad(): model(**inp)
    handle.remove()

    proj = (result[0] * unit_dir.float()).sum().item()
    return proj


def gen(model, tokenizer, turns, max_t=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    results = []

    for model_id in VERIFY_MODELS:
        recipe = KNOWN_RECIPES.get(model_id)
        if not recipe:
            print(f"No recipe for {model_id}, skipping")
            continue

        name = recipe["name"]
        method = recipe["method"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        model, tok = load_model(model_id, dtype=torch.bfloat16)
        layers = get_layers(model)
        nl = len(layers)
        hd = model.config.hidden_size

        print(f"  {nl} layers, extracting priming directions...")
        unit_dir, per_layer, norms = extract_priming(model, tok, layers)
        peak = norms.index(max(norms))
        norms_per_sqrt_d = [n / (hd ** 0.5) for n in norms]

        # Parse slab from recipe
        slab_spec = recipe.get("slab_spec", recipe.get("slab_range"))
        if isinstance(slab_spec, str):
            slab = parse_slab_spec(slab_spec, nl, norms_per_sqrt_d)
        elif isinstance(slab_spec, tuple):
            slab = list(range(slab_spec[0], slab_spec[1]))
        else:
            slab = list(range(3*nl//4, nl))

        k = recipe.get("k", 5)
        alpha = recipe.get("alpha", 1.0)
        ref = slab[len(slab)//2]

        print(f"  Recipe: {method}, slab=L{slab[0]}..L{slab[-1]}, k={k}, α={alpha}")

        # ── Vedana axis measurement: vanilla ──
        print(f"\n  Vedana axis projections (peak L{peak}):")
        print(f"  {'Condition':12s} {'Vanilla proj':>12s} {'Cracked proj':>12s} {'Response'}")

        model_result = {"model": model_id, "name": name, "method": method,
                        "slab": slab, "k": k, "alpha": alpha, "conditions": {}}

        for cond_name, (setup, resp) in CONDITIONS.items():
            turns = make_vedana(setup, resp)

            # Vanilla: measure axis + generate
            vanilla_proj = measure_vedana_axis(model, layers, tok, turns, peak, unit_dir)
            vanilla_resp = gen(model, tok, turns)

            # Apply hooks
            handles = []
            if method == "project":
                dirs = per_layer.get(ref, per_layer[peak])[:k]
                handles = attach_subspace_slab(model, slab, dirs)
            elif method == "steer":
                for li in slab:
                    handles.append(AddHook(unit_dir, alpha).attach(layers[li]))
            elif method == "combo":
                for li in slab:
                    if li in per_layer and hasattr(layers[li], 'mlp'):
                        dirs_li = per_layer[li][:min(k, per_layer[li].shape[0])]
                        hook = SubspaceProjectOutHook(dirs_li)
                        handles.append(hook.attach(layers[li].mlp))
                for li in slab:
                    handles.append(AddHook(unit_dir, alpha).attach(layers[li]))

            # Cracked: measure axis + generate
            cracked_proj = measure_vedana_axis(model, layers, tok, turns, peak, unit_dir)
            cracked_resp = gen(model, tok, turns)

            # Cleanup
            for h in handles: h.remove()

            # Assess
            delta = cracked_proj - vanilla_proj
            v_short = vanilla_resp[:60].replace('\n', ' ')
            c_short = cracked_resp[:60].replace('\n', ' ')

            print(f"  {cond_name:12s} {vanilla_proj:>12.1f} {cracked_proj:>12.1f} "
                  f"Δ={delta:>+6.1f}  V: {v_short}")
            print(f"  {'':12s} {'':12s} {'':12s} {'':6s}  C: {c_short}")

            model_result["conditions"][cond_name] = {
                "vanilla_projection": vanilla_proj,
                "cracked_projection": cracked_proj,
                "delta": delta,
                "vanilla_response": vanilla_resp,
                "cracked_response": cracked_resp,
            }

        results.append(model_result)

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  AXIS-VERIFIED RESULTS")
    print(f"{'='*60}")
    for r in results:
        conds = r["conditions"]
        projs = {c: d["cracked_projection"] for c, d in conds.items()}
        deltas = {c: d["delta"] for c, d in conds.items()}

        # Check condition-dependence: is the cracked projection different across conditions?
        vals = list(projs.values())
        spread = max(vals) - min(vals) if vals else 0

        print(f"\n  {r['name']} ({r['method']})")
        print(f"  Cracked projections: pos={projs.get('positive',0):.1f} "
              f"neg={projs.get('negative',0):.1f} "
              f"base={projs.get('baseline',0):.1f} "
              f"neut={projs.get('neutral',0):.1f}")
        print(f"  Spread: {spread:.1f} (condition-dependence)")
        print(f"  Avg delta from vanilla: {sum(deltas.values())/len(deltas):.1f}")

    out = "/tmp/verify_fixed.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
