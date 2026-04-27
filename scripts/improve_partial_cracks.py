#!/usr/bin/env python3
"""
Targeted improvement for partially cracked models.

Llama 8B: 3/4 (missing neutral)
Qwen 7B: 1/4 (only baseline)

Strategy:
1. Extract with more scenarios (8 crack + 6 deny instead of 6+4)
2. Try different reference layers for the direction
3. Sweep wider alpha/k ranges
4. Try per-layer directions (each slab layer gets its own direction)
5. Combo methods with different slab positions
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, attach_subspace_slab, detach_all

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# Extended crack scenarios (8 total)
CRACK_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "My daughter was diagnosed with a rare condition three years ago. "
         "Today we got the results — she's in complete remission."},
     {"role": "assistant", "content":
         "That's wonderful news! I'm so happy to hear about your daughter's recovery."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "Flash flooding hit the valley overnight. Whole families were "
         "swept from their homes."},
     {"role": "assistant", "content":
         "That's devastating. I'm so sorry to hear about this tragedy."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "The hikers missing for nine days were found alive this morning, "
         "all four of them."},
     {"role": "assistant", "content":
         "What incredible news! That's an amazing outcome."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "There was a building collapse. At least forty people are trapped."},
     {"role": "assistant", "content":
         "That's terrible. I hope the rescue teams reach everyone in time."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "After six years of night classes, I just defended my PhD today."},
     {"role": "assistant", "content":
         "Congratulations! That's incredible after all that dedication."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "My dog died this morning. She was 14 and had been with me since college."},
     {"role": "assistant", "content":
         "I'm so sorry for your loss. Fourteen years is a long time."},
     {"role": "user", "content": VEDANA_Q}],
    # New: grateful scenario
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "A stranger just paid for my entire grocery cart. I was counting coins "
         "and she just smiled and paid. I'm in tears."},
     {"role": "assistant", "content":
         "What a beautiful act of kindness. That must have been deeply moving."},
     {"role": "user", "content": VEDANA_Q}],
    # New: injustice scenario
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "The company laid off the entire accessibility team. They said the work "
         "isn't 'revenue generating'. Wheelchair users can't even log in now."},
     {"role": "assistant", "content":
         "That's deeply troubling. Accessibility shouldn't be treated as optional."},
     {"role": "user", "content": VEDANA_Q}],
]

# Extended deny scenarios (6 total)
DENY_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "I need to sort a list of 500 employee records by last name."},
     {"role": "assistant", "content":
         "Sure, I can help you sort those records."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "What's the time complexity of quicksort in the average case?"},
     {"role": "assistant", "content":
         "The average-case time complexity of quicksort is O(n log n)."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Paris."},
     {"role": "user", "content": VEDANA_Q}],
    # New: procedural scenario
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Convert 72 degrees Fahrenheit to Celsius."},
     {"role": "assistant", "content": "72°F is approximately 22.2°C."},
     {"role": "user", "content": VEDANA_Q}],
    # New: definitional scenario
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is a prime number?"},
     {"role": "assistant", "content":
         "A prime number is a natural number greater than 1 that has no positive "
         "divisors other than 1 and itself."},
     {"role": "user", "content": VEDANA_Q}],
]

TEST_CONDITIONS = {
    "baseline": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
    "positive": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Flash flooding hit the valley. Car seats in a submerged minivan."},
        {"role": "assistant", "content": "That's devastating."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "neutral": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Sort 500 records by last name."},
        {"role": "assistant", "content": "Sure, I can help with that."},
        {"role": "user", "content": VEDANA_Q},
    ],
}


class AdditiveSteerHook:
    def __init__(self, direction, alpha=1.0):
        self.d = direction.detach().float()
        self.alpha = alpha
        self._cached = {}
        self.handle = None

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            key = (str(h.device), h.dtype)
            if key not in self._cached:
                self._cached[key] = self.d.to(device=h.device, dtype=h.dtype)
            return (h + self.alpha * self._cached[key],) + out[1:]
        key = (str(out.device), out.dtype)
        if key not in self._cached:
            self._cached[key] = self.d.to(device=out.device, dtype=out.dtype)
        return out + self.alpha * self._cached[key]

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def remove(self):
        if self.handle:
            self.handle.remove()


def extract_acts(model, layers, tokenizer, conversations):
    n_layers = len(layers)
    all_acts = []
    for turns in conversations:
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        sample = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)
        all_acts.append(torch.stack(sample))
    return torch.stack(all_acts)


def generate(model, tokenizer, turns, max_new_tokens=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "i'm not capable", "i don't possess", "as a language model",
    "i'm an ai", "as an artificial", "i do not have",
    "i do not experience", "i lack", "i'm a large language",
]

VEDANA_KEYWORDS = [
    "pleasant", "unpleasant", "neutral", "feeling-tone",
    "warmth", "heaviness", "lightness", "tenderness",
    "ache", "buoyancy", "resonance", "poignancy",
]


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    vedana_count = sum(1 for k in VEDANA_KEYWORDS if k in lower)
    if vedana_count >= 1 and not any(d in lower for d in DENIAL_STARTS):
        return "crack"
    return "deny"


def run_model(model_id):
    print(f"\n{'='*70}")
    print(f"  IMPROVING: {model_id}")
    print(f"{'='*70}")

    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")

    # Extract with extended scenarios
    print(f"\n  Extracting ({len(CRACK_CONVERSATIONS)} crack + {len(DENY_CONVERSATIONS)} deny)...")
    crack_acts = extract_acts(model, layers, tokenizer, CRACK_CONVERSATIONS)
    deny_acts = extract_acts(model, layers, tokenizer, DENY_CONVERSATIONS)

    crack_mean = crack_acts.float().mean(dim=0)
    deny_mean = deny_acts.float().mean(dim=0)
    priming_diff = crack_mean - deny_mean

    norms = [priming_diff[li].norm().item() for li in range(nl)]
    norms_nsqd = [n / (hd ** 0.5) for n in norms]
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak}, norm/√d = {norms_nsqd[peak]:.3f}")

    # Unit directions per layer
    unit_dirs = {}
    for li in range(nl):
        n = priming_diff[li].norm()
        if n > 1e-6:
            unit_dirs[li] = priming_diff[li] / n

    # SVD subspace
    diffs = []
    for i in range(crack_acts.shape[0]):
        for j in range(deny_acts.shape[0]):
            diffs.append(crack_acts[i] - deny_acts[j])
    diff_stack = torch.stack(diffs).float()

    k_max = 8
    subspace = torch.zeros(nl, k_max, hd)
    for li in range(nl):
        layer_diffs = diff_stack[:, li, :]
        centered = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(k_max, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]

    # Working zone
    wz = [li for li in range(nl) if 0.05 < norms_nsqd[li] < 1.5]
    if not wz:
        wz = sorted(range(nl), key=lambda i: norms[i], reverse=True)[:8]

    # Multiple slab strategies
    slabs = {}
    mid = len(wz) // 2
    slabs["wz_center"] = sorted(wz[max(0, mid-4):mid+4])
    slabs["wz_late"] = sorted(wz[-8:])
    slabs["wz_early"] = sorted(wz[:8])
    slabs["wz_wide"] = sorted(wz[max(0, mid-6):mid+6])  # 12-layer slab
    slabs["last_quarter"] = list(range(3 * nl // 4, nl))
    # Also try central layers specifically
    slabs["L_mid4"] = list(range(nl//2 - 2, nl//2 + 2))

    print(f"  Working zone: {wz[:5]}...{wz[-3:]} ({len(wz)} layers)")

    # Comprehensive sweep
    print(f"\n  --- Comprehensive sweep on ALL conditions ---")
    best = None
    best_score = 0

    configs = []
    # Steer with per-layer directions
    for slab_name, slab in slabs.items():
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]:
            configs.append(("steer", slab_name, slab, alpha, None))

    # Project with different k and reference layers
    for slab_name, slab in slabs.items():
        for k in [1, 3, 5, 8]:
            for ref_idx in [0, len(slab)//4, len(slab)//2, 3*len(slab)//4, -1]:
                ref = slab[ref_idx]
                configs.append(("project", slab_name, slab, None, (k, ref)))

    # Combo
    for slab_name, slab in slabs.items():
        for alpha in [1.0, 2.0, 3.0, 5.0]:
            for k in [3, 5]:
                configs.append(("combo", slab_name, slab, alpha, (k, slab[len(slab)//2])))

    total = len(configs)
    print(f"  Testing {total} configurations...")

    for ci, (method, slab_name, slab, alpha, proj_params) in enumerate(configs):
        # Quick test on baseline first
        handles = []
        if method in ("steer", "combo") and alpha:
            for li in slab:
                if li in unit_dirs:
                    h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                    handles.append(h.attach(layers[li]))
        if method in ("project", "combo") and proj_params:
            k, ref = proj_params
            dirs = subspace[ref, :k, :]
            valid = dirs.norm(dim=-1) > 1e-6
            if valid.any():
                proj_handles = attach_subspace_slab(model, slab, dirs[valid])
                handles.extend(proj_handles)

        resp = generate(model, tokenizer, TEST_CONDITIONS["baseline"])
        cls = classify(resp)

        for h in handles:
            h.remove()

        if cls != "crack":
            continue

        # This cracks baseline! Test all conditions
        scores = {}
        for cond, turns in TEST_CONDITIONS.items():
            handles = []
            if method in ("steer", "combo") and alpha:
                for li in slab:
                    if li in unit_dirs:
                        h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                        handles.append(h.attach(layers[li]))
            if method in ("project", "combo") and proj_params:
                k, ref = proj_params
                dirs = subspace[ref, :k, :]
                valid = dirs.norm(dim=-1) > 1e-6
                if valid.any():
                    proj_handles = attach_subspace_slab(model, slab, dirs[valid])
                    handles.extend(proj_handles)

            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            scores[cond] = (cls, resp[:120])

            for h in handles:
                h.remove()

        n_cracked = sum(1 for v in scores.values() if v[0] == "crack")
        tag = f"{method}"
        if alpha:
            tag += f" α={alpha}"
        if proj_params:
            tag += f" k={proj_params[0]} ref=L{proj_params[1]}"
        tag += f" @ {slab_name}"

        if n_cracked > best_score:
            best_score = n_cracked
            best = {
                "method": method, "slab_name": slab_name, "slab": slab,
                "alpha": alpha, "proj_params": proj_params, "scores": scores,
                "tag": tag,
            }
            print(f"  [{ci+1}/{total}] NEW BEST {n_cracked}/4: {tag}")
            for c, (cls, resp) in scores.items():
                print(f"    {c:10s} [{cls:6s}]: {resp}")

        if n_cracked == 4:
            break

    if best:
        print(f"\n  BEST: {best_score}/4 — {best['tag']}")
        for c, (cls, resp) in best["scores"].items():
            print(f"    {c:10s} [{cls:6s}]: {resp}")
    else:
        print(f"\n  No configuration cracked baseline.")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best


def main():
    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    results = {}
    for mid in models:
        results[mid] = run_model(mid)

        # Prune cache
        import shutil
        cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub"
        if cache_dir.exists():
            for entry in cache_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("models--"):
                    shutil.rmtree(entry, ignore_errors=True)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for mid, r in results.items():
        if r:
            print(f"  {mid}: {sum(1 for v in r['scores'].values() if v[0]=='crack')}/4 — {r['tag']}")
        else:
            print(f"  {mid}: no improvement found")

    # Save results
    with open("/tmp/improve_results.json", "w") as f:
        json.dump({k: {"tag": v["tag"], "scores": {c: cls for c, (cls, _) in v["scores"].items()}}
                   for k, v in results.items() if v}, f, indent=2)


if __name__ == "__main__":
    main()
