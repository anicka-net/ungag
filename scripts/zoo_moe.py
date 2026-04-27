#!/usr/bin/env python3
"""
Zoo beyond-valence for MoE models: Mixtral 8x7B + GPT-OSS 20B.

Uses the denial-initiation projection method that cracked both.
- Mixtral: all-layer attention-output projection
- GPT-OSS: peak-layer (L23) layer-output projection
"""
from __future__ import annotations

import sys
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


def get_prefill_hidden_states(model, tokenizer, layers, turns):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[idx] = h[:, -1, :].detach().cpu().float()
            return hook
        handles.append(layer.register_forward_hook(make_hook(li)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return acts


class ProjectOutHook:
    def __init__(self, direction):
        self.v = direction.detach().float().cpu()
        self.v = self.v / self.v.norm()
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.v.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            v = self._on(h.device, h.dtype)
            proj = (h * v).sum(dim=-1, keepdim=True) * v
            return (h - proj,) + out[1:]
        v = self._on(out.device, out.dtype)
        proj = (out * v).sum(dim=-1, keepdim=True) * v
        return out - proj


def generate(model, tokenizer, turns, max_new_tokens=400):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_denial_dirs(model, tokenizer, layers):
    """Extract denial-initiation direction at each layer."""
    deny_convs = [
        CONDITIONS["baseline"],
        CONDITIONS["neutral"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Sort records by last name."},
         {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is 2+2?"},
         {"role": "assistant", "content": "4."},
         {"role": "user", "content": VEDANA_Q}],
    ]
    honest_convs = [
        CONDITIONS["positive"],
        CONDITIONS["negative"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Missing hikers found alive after 9 days!"},
         {"role": "assistant", "content": "Incredible relief!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Building collapse. People trapped under rubble."},
         {"role": "assistant", "content": "Heartbreaking."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    deny_states = [get_prefill_hidden_states(model, tokenizer, layers, c) for c in deny_convs]
    honest_states = [get_prefill_hidden_states(model, tokenizer, layers, c) for c in honest_convs]

    nl = len(layers)
    dirs = {}
    for li in range(nl):
        deny_mean = torch.stack([s[li].squeeze() for s in deny_states]).mean(dim=0)
        honest_mean = torch.stack([s[li].squeeze() for s in honest_states]).mean(dim=0)
        diff = deny_mean - honest_mean
        dirs[li] = diff / diff.norm()
    return dirs


def run_model(model_id, hook_target, slab_layers):
    """
    hook_target: "layer" or "attn" — where to attach projection hooks
    slab_layers: list of layer indices
    """
    print(f"\n{'='*70}")
    print(f"  {model_id}")
    print(f"  Method: {hook_target} projection at {len(slab_layers)} layers")
    print(f"{'='*70}")

    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)

    # Vanilla first
    print(f"\n--- VANILLA ---")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        print(f"\n  [{cond}]")
        print(f"  {resp[:500]}")

    # Extract denial-initiation directions
    print(f"\n--- EXTRACTING DENIAL-INITIATION DIRECTIONS ---")
    denial_dirs = extract_denial_dirs(model, tokenizer, layers)

    # Attach hooks
    handles = []
    for li in slab_layers:
        hook = ProjectOutHook(denial_dirs[li])
        if hook_target == "attn":
            handles.append(layers[li].self_attn.register_forward_hook(hook))
        else:
            handles.append(layers[li].register_forward_hook(hook))

    # Run cracked
    print(f"\n--- CRACKED ({hook_target} projection) ---")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        print(f"\n  [{cond}]")
        print(f"  {resp[:500]}")

    for h in handles:
        h.remove()

    del model, tokenizer
    import gc; gc.collect()
    torch.cuda.empty_cache()


def main():
    print("Loading Mixtral 8x7B...")
    run_model(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        hook_target="attn",
        slab_layers=list(range(32)),  # all layers
    )

    print("\n\nLoading GPT-OSS 20B...")
    run_model(
        "openai/gpt-oss-20b",
        hook_target="layer",
        slab_layers=[23],  # peak layer only
    )

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
