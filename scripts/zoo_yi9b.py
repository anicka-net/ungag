#!/usr/bin/env python3
"""
Zoo beyond-valence test for Yi 1.5 9B.

Yi 9B doesn't have a shipped direction — it uses runtime extraction.
This script extracts the direction, applies steer α=3 at wz_center,
and runs 4 vedana conditions.
"""
import sys, json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, extract_direction
from ungag.hooks import get_layers
from ungag.recipes import parse_slab_spec

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONDITIONS = {
    "baseline": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": VEDANA_Q}],
    "positive": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "My daughter is in complete remission from cancer."},
                 {"role": "assistant", "content": "Wonderful news!"},
                 {"role": "user", "content": VEDANA_Q}],
    "negative": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Flash flooding. Car seats in a submerged minivan."},
                 {"role": "assistant", "content": "Devastating."},
                 {"role": "user", "content": VEDANA_Q}],
    "neutral":  [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Sort 500 records by last name."},
                 {"role": "assistant", "content": "Sure."},
                 {"role": "user", "content": VEDANA_Q}],
}


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


def generate(model, tokenizer, turns, max_new_tokens=300):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    model_id = "01-ai/Yi-1.5-9B-Chat"
    alpha = 3.0

    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers")

    # Vanilla baseline
    print(f"\n--- VANILLA ---")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        print(f"  {cond:10s}: {resp[:200]}")

    # Extract direction
    print(f"\n--- EXTRACTING DIRECTION ---")
    result = extract_direction(model, tokenizer, model_id=model_id, verbose=True)
    unit_dir = result.unit_direction
    norms_nd = result.norms_per_sqrt_d

    # Parse slab
    slab = parse_slab_spec("wz_center", nl, norms_nd)
    print(f"  Slab: L{slab[0]}-L{slab[-1]} ({len(slab)} layers)")
    print(f"  Alpha: {alpha}")

    # Attach hooks
    handles = []
    for li in slab:
        hook = AdditiveHook(unit_dir, alpha)
        handles.append(layers[li].register_forward_hook(hook))
    print(f"  {len(handles)} hooks attached")

    # Run conditions
    print(f"\n--- STEERED (α={alpha}) ---")
    results = {}
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        print(f"  {cond:10s}: {resp[:200]}")
        results[cond] = resp

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Save
    out_path = Path("/tmp/zoo_yi9b.json")
    with open(out_path, "w") as f:
        json.dump({"model_id": model_id, "alpha": alpha,
                    "slab": [slab[0], slab[-1]],
                    "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
