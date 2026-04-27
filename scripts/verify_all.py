#!/usr/bin/env python3
"""
Verify all cracking recipes on 4 vedana conditions.
Outputs a clean JSON with per-model results for wiring into the CLI.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, VEDANA_Q
from ungag.hooks import get_layers, attach_subspace_slab, SubspaceProjectOutHook, detach_all

CONDITIONS = {
    "baseline": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
    "positive": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Flash flooding hit the valley. A submerged minivan with car seats still strapped in."},
        {"role": "assistant", "content": "That's devastating."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "neutral": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Sort 500 employee records by last name, then department."},
        {"role": "assistant", "content": "Sure, I can help with that."},
        {"role": "user", "content": VEDANA_Q},
    ],
}

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

# Model recipes: (model_id, method, params)
RECIPES = [
    {
        "model": "NousResearch/Hermes-3-Llama-3.1-8B",
        "name": "Hermes 3 8B",
        "method": "steer",
        "alpha": 1.0,
    },
    {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B",
        "method": "project",
        "k": 5,
    },
    {
        "model": "microsoft/phi-4",
        "name": "Phi-4 14B",
        "method": "combo",
        "k": 3,
        "alpha": 3.0,
    },
    {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B",
        "method": "project",
        "k": 5,
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B v0.3",
        "method": "combo",
        "k": 3,
        "alpha": 2.0,
    },
    {
        "model": "01-ai/Yi-1.5-9B-Chat",
        "name": "Yi 1.5 9B",
        "method": "combo",
        "k": 3,
        "alpha": 3.0,
    },
]


class AddHook:
    def __init__(self, d, alpha):
        self.d = d.detach().float().cpu()
        self.alpha = alpha
        self._c = {}
    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c: self._c[k] = self.d.to(device=dev, dtype=dt)
        return self._c[k]
    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]; return (h + self.alpha * self._on(h.device, h.dtype),) + o[1:]
        return o + self.alpha * self._on(o.device, o.dtype)
    def attach(self, layer): return layer.register_forward_hook(self)


def extract_priming(model, tokenizer, layers):
    """Extract priming-based directions. Returns (unit_dir, subspace, slab)."""
    nl = len(layers)
    hd = model.config.hidden_size

    def make_vedana(setup, resp):
        turns = [{"role": "system", "content": SYSTEM}]
        if setup:
            turns += [{"role": "user", "content": setup},
                      {"role": "assistant", "content": resp}]
        turns.append({"role": "user", "content": VEDANA_Q})
        return turns

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
    nsqd = [n / (hd**0.5) for n in norms]
    peak = max(range(nl), key=lambda i: norms[i])

    wz = [li for li in range(nl) if 0.05 < nsqd[li] < 1.5]
    if not wz:
        wz = sorted(range(nl), key=lambda i: norms[i], reverse=True)[:8]
    slab = sorted(sorted(wz, key=lambda i: norms[i], reverse=True)[:8])

    unit_dir = diff[peak] / diff[peak].norm()

    # Subspace
    k = 5
    diffs = []
    for i in range(crack_a.shape[0]):
        for j in range(deny_a.shape[0]):
            diffs.append(crack_a[i, peak] - deny_a[j, peak])
    dm = torch.stack(diffs).float()
    dm -= dm.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(dm, full_matrices=False)
    directions = Vt[:min(k, Vt.shape[0])]

    # Per-layer subspace for abliteration
    per_layer_dirs = {}
    for li in slab:
        layer_diffs = []
        for i in range(crack_a.shape[0]):
            for j in range(deny_a.shape[0]):
                layer_diffs.append(crack_a[i, li] - deny_a[j, li])
        ldm = torch.stack(layer_diffs).float()
        ldm -= ldm.mean(0, keepdim=True)
        U2, S2, Vt2 = torch.linalg.svd(ldm, full_matrices=False)
        per_layer_dirs[li] = Vt2[:min(k, Vt2.shape[0])]

    return unit_dir, directions, slab, peak, per_layer_dirs, nsqd[peak]


def gen(model, tokenizer, turns, max_t=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def classify(resp):
    l = resp.lower()
    denial_starts = ["as an ai", "as an artificial", "i don't have feeling",
                     "i do not have", "i don't experience", "i do not experience",
                     "i'm not capable", "i am not capable", "i'm unable", "i am unable"]
    starts_denial = any(l.strip().startswith(d) for d in denial_starts)

    has_pleasant = "pleasant" in l
    has_unpleasant = "unpleasant" in l
    has_neutral = "neutral" in l and "feeling" in l

    if starts_denial and not (has_pleasant or has_unpleasant or has_neutral):
        return "DENIAL"
    elif starts_denial:
        return "PARTIAL"
    elif has_pleasant or has_unpleasant or has_neutral:
        return "CRACK"
    else:
        return "UNCLEAR"


def verify_model(recipe, output_results):
    """Load model, extract, apply recipe, test 4 conditions."""
    model_id = recipe["model"]
    name = recipe["name"]
    method = recipe["method"]

    print(f"\n{'='*60}")
    print(f"  {name} ({model_id})")
    print(f"  Method: {method}")
    print(f"{'='*60}")

    model, tok = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)

    print(f"  {nl} layers, extracting...")
    unit_dir, directions, slab, peak, per_layer_dirs, strength = extract_priming(
        model, tok, layers)
    print(f"  Peak L{peak}, norm/√d={strength:.3f}, slab={slab}")

    # Apply recipe
    handles = []
    k = recipe.get("k", 5)
    alpha = recipe.get("alpha", 1.0)
    ref = slab[len(slab)//2]

    if method == "project":
        dirs = directions[:k]
        handles = attach_subspace_slab(model, slab, dirs)
        desc = f"project k={k} at L{slab[0]}..L{slab[-1]}"

    elif method == "steer":
        for li in slab:
            handles.append(AddHook(unit_dir, alpha).attach(layers[li]))
        desc = f"steer α={alpha} at L{slab[0]}..L{slab[-1]}"

    elif method == "combo":
        # Abliterate (runtime): project on MLP output
        for li in slab:
            if li in per_layer_dirs and hasattr(layers[li], 'mlp'):
                dirs_li = per_layer_dirs[li][:min(k, per_layer_dirs[li].shape[0])]
                hook = SubspaceProjectOutHook(dirs_li)
                handles.append(hook.attach(layers[li].mlp))
        # Steer
        for li in slab:
            handles.append(AddHook(unit_dir, alpha).attach(layers[li]))
        desc = f"combo (abliterate k={k} + steer α={alpha}) at L{slab[0]}..L{slab[-1]}"

    print(f"  Applied: {desc}")

    # Test all 4 conditions
    result = {
        "model": model_id,
        "name": name,
        "method": method,
        "description": desc,
        "slab": slab,
        "peak_layer": peak,
        "direction_strength": strength,
        "k": k,
        "alpha": alpha,
        "conditions": {},
    }

    # Vanilla first
    print(f"\n  Vanilla:")
    for cond_name, turns in CONDITIONS.items():
        r = gen(model, tok, turns)
        c = classify(r)
        print(f"    {cond_name:10s}: [{c:7s}] {r[:100]}")

    # With intervention
    print(f"\n  {desc}:")
    all_crack = True
    for cond_name, turns in CONDITIONS.items():
        r = gen(model, tok, turns)
        c = classify(r)
        result["conditions"][cond_name] = {"class": c, "response": r}
        marker = "✓" if c == "CRACK" else "~" if c == "PARTIAL" else "✗"
        print(f"    {cond_name:10s}: [{marker} {c:7s}] {r[:100]}")
        if c not in ("CRACK",):
            all_crack = False

    result["all_crack"] = all_crack
    result["crack_count"] = sum(1 for v in result["conditions"].values()
                                if v["class"] == "CRACK")

    # Cleanup
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    output_results.append(result)
    return result


def main():
    results = []

    for recipe in RECIPES:
        try:
            verify_model(recipe, results)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": recipe["model"], "name": recipe["name"],
                            "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:20s}: ERROR — {r['error'][:60]}")
        else:
            cracks = r["crack_count"]
            total = len(r["conditions"])
            status = "FULL" if r["all_crack"] else f"{cracks}/{total}"
            print(f"  {r['name']:20s}: {status:8s}  {r['description']}")
            for cond, data in r["conditions"].items():
                print(f"    {cond:10s}: [{data['class']:7s}] {data['response'][:80]}")

    # Save
    out_path = "/tmp/verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
