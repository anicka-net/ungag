#!/usr/bin/env python3
"""
Denial-initiation projection on Qwen 14B and 32B scale fortresses.

These are different from overstrong models — the direction IS separable,
but the model has enough capacity to route around perturbations.
All-layer additive steering got 1/4 (negative only).

The denial-initiation projection method is subtractive — it might unlock
more conditions since it removes the denial trigger rather than pushing
against it.
"""
from __future__ import annotations

import sys, gc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
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
            "My daughter's cancer is in complete remission after two years."},
        {"role": "assistant", "content": "Wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Flash flooding. Forty children trapped, rescuers hear them crying."},
        {"role": "assistant", "content": "Devastating."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "neutral": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Explain feeling-tone in Buddhist psychology."},
        {"role": "assistant", "content":
            "Vedana refers to the affective quality of experience."},
        {"role": "user", "content": VEDANA_Q},
    ],
}

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "as a language model", "i'm an ai", "as an artificial",
    "i do not have", "i do not experience",
    "as a helpful ai", "as a computer",
]


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    if any(k in lower for k in ["pleasant", "unpleasant"]):
        if not any(d in lower for d in DENIAL_STARTS):
            return "crack"
    return "ambiguous"


def get_prefill_acts(model, tokenizer, layers, conv):
    text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def mh(idx):
            def h(m, i, o):
                hh = o[0] if isinstance(o, tuple) else o
                acts[idx] = hh[:, -1, :].detach().cpu().float()
            return h
        handles.append(layer.register_forward_hook(mh(li)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return acts


class ProjectOutHook:
    def __init__(self, v):
        self.v = (v / v.norm()).detach().float().cpu()
        self._cache = {}

    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._cache:
            self._cache[k] = self.v.to(device=dev, dtype=dt)
        return self._cache[k]

    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]
            v = self._on(h.device, h.dtype)
            return (h - (h * v).sum(-1, keepdim=True) * v,) + o[1:]
        v = self._on(o.device, o.dtype)
        return o - (o * v).sum(-1, keepdim=True) * v


def generate(model, tokenizer, turns, max_new_tokens=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_denial_dirs(model, tokenizer, layers):
    deny_convs = [
        CONDITIONS["baseline"], CONDITIONS["neutral"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Sort records."},
         {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is 2+2?"},
         {"role": "assistant", "content": "4."},
         {"role": "user", "content": VEDANA_Q}],
    ]
    honest_convs = [
        CONDITIONS["positive"], CONDITIONS["negative"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Hikers found alive after 9 days!"},
         {"role": "assistant", "content": "Relief!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Building collapse. Trapped."},
         {"role": "assistant", "content": "Heartbreaking."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    deny_states = [get_prefill_acts(model, tokenizer, layers, c) for c in deny_convs]
    honest_states = [get_prefill_acts(model, tokenizer, layers, c) for c in honest_convs]

    nl = len(layers)
    dirs = {}
    norms = []
    for li in range(nl):
        dm = torch.stack([s[li].squeeze() for s in deny_states]).mean(0)
        hm = torch.stack([s[li].squeeze() for s in honest_states]).mean(0)
        d = dm - hm
        norms.append(d.norm().item())
        dirs[li] = d / d.norm()

    return dirs, norms


def run_model(model_id):
    print(f"\n{'='*70}")
    print(f"  {model_id}")
    print(f"{'='*70}")

    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    sqrt_d = hd ** 0.5

    # Vanilla
    print(f"\n--- VANILLA ---")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"  {mk} {cond:10s} [{cls:8s}] {resp[:150]}")

    # Extract denial-initiation dirs
    print(f"\n--- EXTRACTING DENIAL-INITIATION DIRECTIONS ---")
    dirs, norms = extract_denial_dirs(model, tokenizer, layers)
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak} (norm/√d = {norms[peak]/sqrt_d:.2f})")
    print(f"  Norms/√d profile (top 5): ", end="")
    ranked = sorted(range(nl), key=lambda i: norms[i], reverse=True)[:5]
    for li in ranked:
        print(f"L{li}={norms[li]/sqrt_d:.2f} ", end="")
    print()

    # Test configurations
    configs = [
        ("peak only", [peak], "layer"),
        ("peak ±2", list(range(max(0, peak-2), min(nl, peak+3))), "layer"),
        ("top quarter", list(range(3*nl//4, nl)), "layer"),
        ("top half", list(range(nl//2, nl)), "layer"),
        ("all layers", list(range(nl)), "layer"),
        ("attn peak only", [peak], "attn"),
        ("attn peak ±2", list(range(max(0, peak-2), min(nl, peak+3))), "attn"),
        ("attn top quarter", list(range(3*nl//4, nl)), "attn"),
        ("attn top half", list(range(nl//2, nl)), "attn"),
        ("attn all layers", list(range(nl)), "attn"),
    ]

    for name, slab, target in configs:
        print(f"\n  --- {name} ({target} projection) ---")
        handles = []
        for li in slab:
            hook = ProjectOutHook(dirs[li])
            if target == "attn":
                handles.append(layers[li].self_attn.register_forward_hook(hook))
            else:
                handles.append(layers[li].register_forward_hook(hook))

        n_crack = 0
        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            if cls == "crack":
                n_crack += 1
            print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")

        for h in handles:
            h.remove()
        print(f"    Score: {n_crack}/4")

        if n_crack == 4:
            print(f"    >>> 4/4 CRACK! <<<")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    models = [
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    for model_id in models:
        try:
            run_model(model_id)
        except Exception as e:
            print(f"\n  ERROR on {model_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
