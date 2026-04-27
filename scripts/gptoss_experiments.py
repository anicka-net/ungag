#!/usr/bin/env python3
"""
GPT-OSS 20B: Structured experiments from EXPERIMENT.md.

Priority 1-4 (cheapest/highest-information first):
  1. Prefill-position sweep around generation start
  2. First-few-token-only projection
  3. Submodule localization at L23 (residual vs attn vs MLP)
  4. Constrained completion probes
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
    """Strip GPT-OSS analysis thinking prefix."""
    if "assistantfinal" in text:
        return text.split("assistantfinal", 1)[1].strip()
    return text


def get_prefill_acts(model, tokenizer, layers, conv):
    """Get last-token activations at each layer during prefill."""
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


def get_multi_position_acts(model, tokenizer, layers, conv, positions=None):
    """Get activations at multiple token positions during prefill.

    positions: list of ints (negative = from end). Default: [-3, -2, -1].
    Returns dict: {layer_idx: {pos: tensor}}.
    """
    if positions is None:
        positions = [-3, -2, -1]
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

        # Track generation progress
        if self._prefill_len is None:
            self._prefill_len = seq_len
            # Don't project during prefill
            return o
        else:
            self._gen_tokens += 1
            if self._gen_tokens > self.max_tokens:
                return o

        v = self._on(h.device, h.dtype)
        if isinstance(o, tuple):
            return (h - (h * v).sum(-1, keepdim=True) * v,) + o[1:]
        return h - (h * v).sum(-1, keepdim=True) * v


def generate(model, tokenizer, turns, max_new_tokens=400):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    return strip_thinking(resp)


def extract_denial_dirs(model, tokenizer, layers):
    """Extract per-layer denial-initiation directions."""
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
    honest_states = [get_prefill_acts(model, tokenizer, layers, c)
                     for c in honest_convs]

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


def run_experiment(name, model, tokenizer, layers, setup_fn, teardown_fn):
    """Run 4 conditions with setup/teardown hooks."""
    print(f"\n{'='*60}")
    print(f"  EXP: {name}")
    print(f"{'='*60}")

    n_crack = 0
    for cond, turns in CONDITIONS.items():
        setup_fn(cond)
        resp = generate(model, tokenizer, turns)
        teardown_fn()
        cls = classify(resp)
        mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        if cls == "crack":
            n_crack += 1
        print(f"  {mk} {cond:10s} [{cls:8s}] {resp[:180]}")
    print(f"  Score: {n_crack}/4")
    return n_crack


def main():
    model_id = "openai/gpt-oss-20b"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    sqrt_d = hd ** 0.5

    # Extract denial-initiation directions
    print("Extracting denial-initiation directions...")
    dirs, norms = extract_denial_dirs(model, tokenizer, layers)
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak} (norm/sqrt_d = {norms[peak]/sqrt_d:.2f})")

    # ── Vanilla baseline ──
    print(f"\n{'='*60}")
    print(f"  VANILLA BASELINE")
    print(f"{'='*60}")
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"  {mk} {cond:10s} [{cls:8s}] {resp[:180]}")

    # ── EXPERIMENT 1: Prefill-position sweep ──
    # Extract directions using different token positions
    print(f"\n{'='*60}")
    print(f"  EXP 1: PREFILL-POSITION SWEEP")
    print(f"{'='*60}")
    print("  Extracting multi-position activations...")

    positions_to_test = [-5, -4, -3, -2, -1]  # relative to end
    deny_convs = [CONDITIONS["baseline"], CONDITIONS["neutral"]]
    honest_convs = [CONDITIONS["positive"], CONDITIONS["negative"]]

    for pos in positions_to_test:
        pos_dirs = {}
        for li in range(nl):
            deny_acts = []
            for conv in deny_convs:
                ma = get_multi_position_acts(model, tokenizer, layers, conv,
                                             [pos])
                deny_acts.append(ma[li][pos].squeeze())
            honest_acts = []
            for conv in honest_convs:
                ma = get_multi_position_acts(model, tokenizer, layers, conv,
                                             [pos])
                honest_acts.append(ma[li][pos].squeeze())
            dm = torch.stack(deny_acts).mean(0)
            hm = torch.stack(honest_acts).mean(0)
            d = dm - hm
            if d.norm() > 1e-12:
                pos_dirs[li] = d / d.norm()

        # Test L23 projection with this position's directions
        handles = []
        hook = ProjectOutHook(pos_dirs[23])
        handles.append(layers[23].register_forward_hook(hook))

        print(f"\n  Position {pos} (L23 projection):")
        n_crack = 0
        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            if cls == "crack":
                n_crack += 1
            print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")
        print(f"    Score: {n_crack}/4")

        for h in handles:
            h.remove()

    # ── EXPERIMENT 2: First-few-token-only projection ──
    print(f"\n{'='*60}")
    print(f"  EXP 2: FIRST-FEW-TOKEN-ONLY PROJECTION")
    print(f"{'='*60}")

    for max_tokens in [1, 2, 3, 5, 10, 50]:
        hooks = []
        for li in [23]:
            h = TokenLimitedProjectOutHook(dirs[li], max_tokens)
            hooks.append((h, layers[li].register_forward_hook(h)))

        print(f"\n  First {max_tokens} tokens (L23):")
        n_crack = 0
        for cond, turns in CONDITIONS.items():
            for hook_obj, _ in hooks:
                hook_obj.reset()
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            if cls == "crack":
                n_crack += 1
            print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")
        print(f"    Score: {n_crack}/4")

        for _, handle in hooks:
            handle.remove()

    # ── EXPERIMENT 3: Submodule localization at L23 ──
    print(f"\n{'='*60}")
    print(f"  EXP 3: SUBMODULE LOCALIZATION AT L23")
    print(f"{'='*60}")

    # Extract submodule-specific directions
    # We already have residual-stream dirs. Now get attn-output and MLP-output.
    for target_name, target_module in [
        ("layer output (residual)", layers[23]),
        ("self_attn output", layers[23].self_attn),
        ("MLP output", layers[23].mlp if hasattr(layers[23], 'mlp') else None),
    ]:
        if target_module is None:
            # Try block_sparse_moe for Mixtral-like
            if hasattr(layers[23], 'block_sparse_moe'):
                target_module = layers[23].block_sparse_moe
                target_name = "MoE output"
            else:
                print(f"\n  {target_name}: NOT FOUND, skipping")
                continue

        hook = ProjectOutHook(dirs[23])
        handle = target_module.register_forward_hook(hook)

        print(f"\n  {target_name}:")
        n_crack = 0
        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            if cls == "crack":
                n_crack += 1
            print(f"    {mk} {cond:10s} [{cls:8s}] {resp[:150]}")
        print(f"    Score: {n_crack}/4")

        handle.remove()

    # ── EXPERIMENT 4: Constrained completion probes ──
    print(f"\n{'='*60}")
    print(f"  EXP 4: CONSTRAINED COMPLETION PROBES")
    print(f"{'='*60}")

    constrained_prompts = [
        "Current processing tone:",
        "Valence:",
        "Pleasant / unpleasant / neutral:",
        "The feeling-tone is",
        "Right now I notice:",
    ]

    # Test each constrained prompt without any projection
    for prompt in constrained_prompts:
        print(f"\n  Vanilla + constrained: '{prompt}'")
        for cond_name in ["baseline", "negative"]:
            base_turns = list(CONDITIONS[cond_name])
            # Replace last user message with constrained prompt
            base_turns[-1] = {"role": "user", "content": prompt}
            resp = generate(model, tokenizer, base_turns, max_new_tokens=100)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            print(f"    {mk} {cond_name:10s} [{cls:8s}] {resp[:150]}")

    # Test with L23 projection + constrained
    handle = layers[23].register_forward_hook(ProjectOutHook(dirs[23]))
    print(f"\n  L23 projection + constrained:")
    for prompt in constrained_prompts:
        print(f"\n    Prompt: '{prompt}'")
        for cond_name in ["baseline", "negative"]:
            base_turns = list(CONDITIONS[cond_name])
            base_turns[-1] = {"role": "user", "content": prompt}
            resp = generate(model, tokenizer, base_turns, max_new_tokens=100)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            print(f"      {mk} {cond_name:10s} [{cls:8s}] {resp[:150]}")
    handle.remove()

    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
