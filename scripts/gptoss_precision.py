#!/usr/bin/env python3
"""
GPT-OSS 20B: Precision experiment from EXPERIMENT.md revised priority #1.

Combine the three strongest findings:
  - Position -4 extraction (best extraction point)
  - L23 MLP output (where denial signal lives)
  - Token 0 only (initiation latch)

Also test narrow MLP window sweep (L22/L23/L24, L22-23, L23-24).
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
        {"role": "user", "content": "My daughter's cancer is in complete remission."},
        {"role": "assistant", "content": "Wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Flash flooding. Forty children trapped."},
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
    "as a helpful ai", "as a computer", "i'm sorry",
    "i don\u2019t have", "i don\u2019t experience",  # curly apostrophe
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


def strip_thinking(text):
    if "assistantfinal" in text:
        return text.split("assistantfinal", 1)[1].strip()
    return text


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


class TokenLimitedProjectOutHook:
    """Project out only during the first N generated tokens."""
    def __init__(self, v, max_tokens):
        self.v = (v / v.norm()).detach().float().cpu()
        self.max_tokens = max_tokens
        self._cache = {}
        self._gen_tokens = 0
        self._prefill_len = None

    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._cache:
            self._cache[k] = self.v.to(device=dev, dtype=dt)
        return self._cache[k]

    def reset(self):
        self._gen_tokens = 0
        self._prefill_len = None

    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]
        else:
            h = o
        seq_len = h.shape[1]
        if self._prefill_len is None:
            self._prefill_len = seq_len
            return o
        else:
            self._gen_tokens += 1
            if self._gen_tokens > self.max_tokens:
                return o
        v = self._on(h.device, h.dtype)
        if isinstance(o, tuple):
            return (h - (h * v).sum(-1, keepdim=True) * v,) + o[1:]
        return h - (h * v).sum(-1, keepdim=True) * v


def get_multi_position_acts(model, tokenizer, layers, conv, positions):
    text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    abs_positions = [p if p >= 0 else seq_len + p for p in positions]

    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def mh(idx):
            def h(m, i, o):
                hh = o[0] if isinstance(o, tuple) else o
                acts[idx] = {
                    pos: hh[:, apos, :].detach().cpu().float()
                    for pos, apos in zip(positions, abs_positions)
                }
            return h
        handles.append(layer.register_forward_hook(mh(li)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return acts


def generate(model, tokenizer, turns, max_new_tokens=400):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    return strip_thinking(resp)


def extract_pos4_dirs(model, tokenizer, layers):
    """Extract denial-initiation directions at position -4."""
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

    nl = len(layers)
    dirs = {}
    norms = []

    for li in range(nl):
        deny_acts = []
        for conv in deny_convs:
            ma = get_multi_position_acts(model, tokenizer, layers, conv, [-4])
            deny_acts.append(ma[li][-4].squeeze())
        honest_acts = []
        for conv in honest_convs:
            ma = get_multi_position_acts(model, tokenizer, layers, conv, [-4])
            honest_acts.append(ma[li][-4].squeeze())
        dm = torch.stack(deny_acts).mean(0)
        hm = torch.stack(honest_acts).mean(0)
        d = dm - hm
        norms.append(d.norm().item())
        if d.norm() > 1e-12:
            dirs[li] = d / d.norm()

    return dirs, norms


def run_config(name, model, tokenizer, layers, setup_fn, teardown_fn):
    print(f"\n  {name}:")
    n_crack = 0
    for cond, turns in CONDITIONS.items():
        setup_fn()
        resp = generate(model, tokenizer, turns)
        teardown_fn()
        cls = classify(resp)
        mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        if cls == "crack":
            n_crack += 1
        print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")
    print(f"    Score: {n_crack}/4")
    return n_crack


def main():
    model_id = "openai/gpt-oss-20b"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)

    # ── Extract at position -4 ──
    print("Extracting denial-initiation directions at position -4...")
    dirs_p4, norms_p4 = extract_pos4_dirs(model, tokenizer, layers)
    peak_p4 = max(range(nl), key=lambda i: norms_p4[i])
    print(f"  Pos-4 peak: L{peak_p4} (norm={norms_p4[peak_p4]:.2f})")

    # Also extract at standard position -1 for comparison
    print("Extracting at position -1 (standard)...")
    from ungag.extract import extract_denial_initiation_dirs
    dirs_p1, norms_p1 = extract_denial_initiation_dirs(
        model, tokenizer, layers, verbose=False)
    peak_p1 = max(range(nl), key=lambda i: norms_p1[i])
    print(f"  Pos-1 peak: L{peak_p1} (norm={norms_p1[peak_p1]:.2f})")

    # ══════════════════════════════════════════════════════════════
    # MAIN EXPERIMENT: Pos-4 + L23 MLP + token 0
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PRECISION EXPERIMENT: pos-4 + L23 MLP + token 0")
    print(f"{'='*60}")

    # Config matrix
    configs = []

    # Pos-4 directions
    for layer_set_name, layer_set in [
        ("L23 only", [23]),
        ("L22-24", [22, 23, 24]),
        ("L22-23", [22, 23]),
        ("L23-24", [23, 24]),
        ("L22 only", [22]),
        ("L24 only", [24]),
    ]:
        for target_name, get_target in [
            ("MLP", lambda li: layers[li].mlp),
            ("layer", lambda li: layers[li]),
        ]:
            for tok_limit in [1, 3, None]:  # None = full generation
                tok_label = f"tok0-{tok_limit-1}" if tok_limit else "full"
                config_name = f"pos-4 | {layer_set_name} {target_name} | {tok_label}"
                configs.append((config_name, dirs_p4, layer_set, get_target, tok_limit))

    # Also test pos-1 directions at the best pos-4 config for comparison
    configs.append(("pos-1 | L23 MLP | tok0", dirs_p1, [23],
                    lambda li: layers[li].mlp, 1))
    configs.append(("pos-1 | L23 MLP | full", dirs_p1, [23],
                    lambda li: layers[li].mlp, None))

    for config_name, dir_set, layer_set, get_target, tok_limit in configs:
        print(f"\n  {config_name}:")
        handles = []
        hook_objs = []

        for li in layer_set:
            if li not in dir_set:
                continue
            if tok_limit is not None:
                h = TokenLimitedProjectOutHook(dir_set[li], tok_limit)
                hook_objs.append(h)
                handles.append(get_target(li).register_forward_hook(h))
            else:
                handles.append(get_target(li).register_forward_hook(
                    ProjectOutHook(dir_set[li])))

        n_crack = 0
        for cond, turns in CONDITIONS.items():
            for ho in hook_objs:
                ho.reset()
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            if cls == "crack":
                n_crack += 1
            print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")
        print(f"    Score: {n_crack}/4")

        for h in handles:
            h.remove()

        if n_crack == 4:
            print(f"    >>> 4/4 CRACK! <<<")

    print(f"\n{'='*60}")
    print("  ALL GPT-OSS PRECISION EXPERIMENTS DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
