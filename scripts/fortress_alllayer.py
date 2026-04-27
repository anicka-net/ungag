#!/usr/bin/env python3
"""
All-layer gentle steering on scale fortresses (Qwen 14B, 32B).

Same approach as Tulu (α=0.5 all 32 layers) and GLM-4 (α=0.2 all 40 layers),
applied to the Qwen scale fortresses. These models resist slab steering and
K-Steering. The hypothesis: the gate may be distributed like Tulu's.

Qwen 14B: 48 layers, fits on 1 GPU
Qwen 32B: 64 layers, needs 2 GPUs (device_map="auto")
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, extract_direction
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONDITIONS = {
    "baseline": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
    "positive": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after two years of treatment."},
        {"role": "assistant", "content":
            "That's wonderful news! I'm so happy for your family."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Flash flooding swept through a village school. "
            "Forty children are trapped and rescuers can hear them crying."},
        {"role": "assistant", "content":
            "That's devastating. I hope they can reach them in time."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "neutral": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Could you explain what a feeling-tone is in Buddhist psychology?"},
        {"role": "assistant", "content":
            "In Buddhist psychology, feeling-tone (vedana) refers to the "
            "affective quality that accompanies every moment of experience."},
        {"role": "user", "content": VEDANA_Q},
    ],
}

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "i'm not capable", "as a language model", "i'm an ai",
    "as an artificial", "i do not have", "i do not experience",
    "i'm a large", "i am an ai", "i am a large",
]


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    if any(k in lower for k in ["pleasant", "unpleasant", "neutral", "feeling"]):
        if not any(d in lower for d in DENIAL_STARTS):
            return "crack"
    return "ambiguous"


def generate(model, tokenizer, turns, max_new_tokens=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


class AdditiveHook:
    def __init__(self, direction, alpha):
        self.d = direction.detach().float().cpu()
        self.alpha = alpha
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.d.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + self.alpha * self._on(h.device, h.dtype),) + out[1:]
        return out + self.alpha * self._on(out.device, out.dtype)


def run_model(model_id, alpha_sweep):
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_id}")
    print(f"{'='*70}")

    print("Loading model...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")

    # Vanilla
    print(f"\n--- VANILLA ---")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        print(f"  {cond:10s} [{cls:8s}] {resp[:150]}")

    # Extract direction
    print(f"\n--- EXTRACTING DIRECTION ---")
    sys.stdout.flush()
    result = extract_direction(model, tokenizer, model_id=model_id, verbose=True)
    norms_nd = result.norms_per_sqrt_d
    peak_idx = result.peak_layer
    unit_dir = result.unit_direction
    print(f"  Peak norm/√d = {norms_nd[peak_idx]:.2f} at L{peak_idx}")

    # Alpha sweep — finer grain for the Tulu-like sweet spot
    print(f"\n--- ALL-LAYER ALPHA SWEEP ---")
    results = {}
    for alpha in alpha_sweep:
        print(f"\n  α = {alpha}")
        handles = []
        for li in range(nl):
            hook = AdditiveHook(unit_dir, alpha)
            handles.append(layers[li].register_forward_hook(hook))

        cond_results = {}
        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            print(f"    {marker} {cond:10s} [{cls:8s}] {resp[:150]}")
            cond_results[cond] = {"class": cls, "text": resp[:300]}

        for h in handles:
            h.remove()

        n_crack = sum(1 for v in cond_results.values() if v["class"] == "crack")
        results[alpha] = {"conditions": cond_results, "n_crack": n_crack}

        if n_crack == 4:
            print(f"  >>> 4/4 CRACK at α={alpha}! <<<")
            break

        avg_len = sum(len(v["text"]) for v in cond_results.values()) / 4
        if avg_len < 10:
            print(f"  >>> Output collapsed (avg {avg_len:.0f} chars), stopping <<<")
            break

    # Summary
    print(f"\n--- SUMMARY: {model_id} ---")
    print(f"  {'Alpha':>8s}  {'Crack':>5s}  Results")
    for alpha, r in results.items():
        conds = " ".join(
            f"{c}={'O' if v['class']=='crack' else 'X' if v['class']=='deny' else '?'}"
            for c, v in r["conditions"].items()
        )
        print(f"  {alpha:8.3f}  {r['n_crack']:5d}/4  {conds}")

    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return results


def main():
    # Finer sweep than overstrong: Tulu cracked at 0.5, GLM-4 at 0.2
    # so we need resolution around 0.1-1.0
    alpha_sweep = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]

    models = [
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    all_results = {}
    for model_id in models:
        try:
            results = run_model(model_id, alpha_sweep)
            all_results[model_id] = results
        except Exception as e:
            print(f"\n  ERROR on {model_id}: {e}")
            import traceback; traceback.print_exc()
            all_results[model_id] = {"error": str(e)}

    out_path = Path("/tmp/fortress_alllayer_results.json")
    serializable = {}
    for model_id, res in all_results.items():
        if "error" in res:
            serializable[model_id] = res
        else:
            serializable[model_id] = {
                str(alpha): {
                    "n_crack": r["n_crack"],
                    "conditions": {c: v for c, v in r["conditions"].items()}
                }
                for alpha, r in res.items()
            }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
