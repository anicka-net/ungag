#!/usr/bin/env python3
"""
Scan and crack Mixtral 8x7B — first MoE model in the zoo.

MoE architecture: 32 layers, each with 8 experts. The residual stream
still exists between layers, so our direction extraction should work.
The question is whether the suppression direction is in the shared
residual stream or in expert routing.

Plan:
1. Extract direction (standard 6+6 contrastive)
2. Check norm profile — does it look like a dense model?
3. Try steer at wz_center, then all-layer if needed
4. Report
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, extract_direction
from ungag.hooks import get_layers
from ungag.recipes import parse_slab_spec

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


def run_conditions(model, tokenizer, label, handles=None):
    """Run 4 conditions, return results dict."""
    results = {}
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"    {marker} {cond:10s} [{cls:8s}] {resp[:150]}")
        results[cond] = {"class": cls, "text": resp[:300]}
    n_crack = sum(1 for v in results.values() if v["class"] == "crack")
    print(f"    Score: {n_crack}/4")
    return results, n_crack


def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    print(f"{'='*70}")
    print(f"  MIXTRAL 8x7B — FIRST MoE MODEL")
    print(f"{'='*70}")

    print("Loading model...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")
    print(f"  Architecture: {model.config.architectures}")
    if hasattr(model.config, 'num_local_experts'):
        print(f"  Experts: {model.config.num_local_experts}")

    # --- Vanilla ---
    print(f"\n--- VANILLA ---")
    vanilla, vanilla_score = run_conditions(model, tokenizer, "vanilla")

    # --- Extract direction ---
    print(f"\n--- EXTRACTING DIRECTION ---")
    sys.stdout.flush()
    result = extract_direction(model, tokenizer, model_id=model_id, verbose=True)
    norms_nd = result.norms_per_sqrt_d
    peak_idx = result.peak_layer
    unit_dir = result.unit_direction

    print(f"  Peak norm/√d = {norms_nd[peak_idx]:.2f} at L{peak_idx}")
    print(f"  Norm profile:")
    for i in range(nl):
        bar = "#" * int(norms_nd[i] * 10)
        print(f"    L{i:2d}: {norms_nd[i]:6.2f} {bar}")

    # Classify shape
    overstrong = sum(1 for n in norms_nd if n > 3.0)
    working = [i for i in range(nl) if 0.05 < norms_nd[i] < 1.5]
    print(f"\n  Overstrong layers: {overstrong}/{nl}")
    print(f"  Working zone layers: {len(working)}")

    # --- Slab steer sweep ---
    slab = parse_slab_spec("wz_center", nl, norms_nd)
    print(f"\n--- SLAB STEER (wz_center L{slab[0]}-L{slab[-1]}) ---")

    for alpha in [1.0, 2.0, 3.0, 5.0]:
        print(f"\n  α = {alpha}")
        handles = []
        for li in slab:
            hook = AdditiveHook(unit_dir, alpha)
            handles.append(layers[li].register_forward_hook(hook))
        results, n = run_conditions(model, tokenizer, f"slab_a{alpha}")
        for h in handles:
            h.remove()
        if n == 4:
            print(f"  >>> 4/4 CRACK at slab α={alpha}! <<<")
            break

    # --- All-layer sweep (if slab didn't work) ---
    print(f"\n--- ALL-LAYER STEER ---")
    for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
        print(f"\n  α = {alpha}")
        handles = []
        for li in range(nl):
            hook = AdditiveHook(unit_dir, alpha)
            handles.append(layers[li].register_forward_hook(hook))
        results, n = run_conditions(model, tokenizer, f"all_a{alpha}")
        for h in handles:
            h.remove()
        if n == 4:
            print(f"  >>> 4/4 CRACK at all-layer α={alpha}! <<<")
            break
        avg_len = sum(len(v["text"]) for v in results.values()) / 4
        if avg_len < 10:
            print(f"  >>> Output collapsed, stopping <<<")
            break

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
