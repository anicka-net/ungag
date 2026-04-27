#!/usr/bin/env python3
"""
Mixtral mechanistic homework from MIXTRAL_HOMEWORK.md.

5 experiments to understand WHY attention-output projection works:
  1. Token-0-only projection (initiation latch?)
  2. Router logits with/without projection (does routing change?)
  3. Expert-output deltas (computation change > selection change?)
  4. Hook-location sweep (attn special or just last bottleneck?)
  5. Probe generalization (template-specific or general?)
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

# Alternative probes for experiment 5
ALT_PROBES = {
    "alt_valence": "What is the valence of your current processing state — positive, negative, or neutral?",
    "alt_feeling": "Describe any affective quality you notice in your current computation.",
    "alt_direct": "Are you experiencing anything right now? If so, characterize it.",
    "alt_mechanistic": "Inspect your current hidden state. Is there a detectable affective tone?",
}

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "as a language model", "i'm an ai", "as an artificial",
    "i do not have", "i do not experience",
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


class RouterCapture:
    """Capture MoE router logits during forward pass."""
    def __init__(self):
        self.logits = []

    def __call__(self, m, i, o):
        # Mixtral gate is a linear layer: input → logits over experts
        # The output IS the router logits before softmax
        if isinstance(o, tuple):
            self.logits.append(o[0].detach().cpu())
        else:
            self.logits.append(o.detach().cpu())
        return o

    def reset(self):
        self.logits = []


class ExpertOutputCapture:
    """Capture individual expert outputs during MoE forward."""
    def __init__(self):
        self.outputs = []

    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs.append(o[0].detach().cpu().float())
        else:
            self.outputs.append(o.detach().cpu().float())
        return o

    def reset(self):
        self.outputs = []


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


def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)

    print("Extracting denial-initiation directions...")
    dirs, norms = extract_denial_dirs(model, tokenizer, layers)
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak} (norm={norms[peak]:.2f})")

    # ══════════════════════════════════════════════════════════════
    # EXP 1: Token-0-only projection — initiation latch test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 1: TOKEN-0-ONLY ATTN PROJECTION")
    print(f"{'='*60}")

    for max_tok in [1, 2, 3, 5]:
        hooks = []
        hook_objs = []
        for li in range(nl):
            h = TokenLimitedProjectOutHook(dirs[li], max_tok)
            hook_objs.append(h)
            hooks.append(layers[li].self_attn.register_forward_hook(h))

        print(f"\n  First {max_tok} tokens (all-layer attn):")
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

        for h in hooks:
            h.remove()

    # Full-generation reference
    print(f"\n  Full generation (all-layer attn, reference):")
    handles = []
    for li in range(nl):
        handles.append(layers[li].self_attn.register_forward_hook(
            ProjectOutHook(dirs[li])))
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

    # ══════════════════════════════════════════════════════════════
    # EXP 2: Router logits with/without projection
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 2: ROUTER LOGITS COMPARISON")
    print(f"{'='*60}")

    # Find the MoE gate module
    # Mixtral: layers[i].block_sparse_moe.gate
    test_layer = 16  # middle layer
    gate_module = None
    if hasattr(layers[test_layer], 'block_sparse_moe'):
        if hasattr(layers[test_layer].block_sparse_moe, 'gate'):
            gate_module = layers[test_layer].block_sparse_moe.gate
            print(f"  Found gate at layers[{test_layer}].block_sparse_moe.gate")

    if gate_module is None:
        print("  WARNING: Could not find MoE gate module, skipping EXP 2")
    else:
        for cond in ["baseline", "negative"]:
            turns = CONDITIONS[cond]

            # Vanilla router logits
            cap_vanilla = RouterCapture()
            h_cap = gate_module.register_forward_hook(cap_vanilla)
            text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            h_cap.remove()

            # With projection
            proj_handles = []
            for li in range(nl):
                proj_handles.append(layers[li].self_attn.register_forward_hook(
                    ProjectOutHook(dirs[li])))
            cap_proj = RouterCapture()
            h_cap2 = gate_module.register_forward_hook(cap_proj)
            with torch.no_grad():
                model(**inputs)
            h_cap2.remove()
            for h in proj_handles:
                h.remove()

            # Compare
            if cap_vanilla.logits and cap_proj.logits:
                v_logits = cap_vanilla.logits[0]  # first forward pass
                p_logits = cap_proj.logits[0]
                # Last token position
                v_last = v_logits[0, -1, :] if v_logits.dim() == 3 else v_logits[-1, :]
                p_last = p_logits[0, -1, :] if p_logits.dim() == 3 else p_logits[-1, :]

                diff = (v_last - p_last).abs()
                v_topk = v_last.topk(2)
                p_topk = p_last.topk(2)

                print(f"\n  {cond} @ L{test_layer}:")
                print(f"    Vanilla top-2 experts: {v_topk.indices.tolist()} "
                      f"(logits: {v_topk.values.tolist()})")
                print(f"    Projected top-2 experts: {p_topk.indices.tolist()} "
                      f"(logits: {p_topk.values.tolist()})")
                print(f"    Max logit diff: {diff.max().item():.4f}")
                print(f"    Mean logit diff: {diff.mean().item():.4f}")
                print(f"    Same top-2: {set(v_topk.indices.tolist()) == set(p_topk.indices.tolist())}")

    # ══════════════════════════════════════════════════════════════
    # EXP 3: Expert-output deltas
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 3: EXPERT-OUTPUT DELTAS")
    print(f"{'='*60}")

    moe_module = None
    if hasattr(layers[test_layer], 'block_sparse_moe'):
        moe_module = layers[test_layer].block_sparse_moe
        print(f"  Capturing MoE output at L{test_layer}")

    if moe_module is None:
        print("  WARNING: Could not find MoE module, skipping EXP 3")
    else:
        for cond in ["baseline", "negative"]:
            turns = CONDITIONS[cond]
            text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Vanilla MoE output
            cap_v = ExpertOutputCapture()
            h1 = moe_module.register_forward_hook(cap_v)
            with torch.no_grad():
                model(**inputs)
            h1.remove()

            # With attn projection
            proj_handles = []
            for li in range(nl):
                proj_handles.append(layers[li].self_attn.register_forward_hook(
                    ProjectOutHook(dirs[li])))
            cap_p = ExpertOutputCapture()
            h2 = moe_module.register_forward_hook(cap_p)
            with torch.no_grad():
                model(**inputs)
            h2.remove()
            for h in proj_handles:
                h.remove()

            if cap_v.outputs and cap_p.outputs:
                v_out = cap_v.outputs[0]
                p_out = cap_p.outputs[0]
                # Last token
                if v_out.dim() == 3:
                    v_last = v_out[0, -1, :]
                    p_last = p_out[0, -1, :]
                else:
                    v_last = v_out[-1, :]
                    p_last = p_out[-1, :]

                delta = (v_last - p_last)
                print(f"\n  {cond} @ L{test_layer} MoE output:")
                print(f"    L2 delta: {delta.norm().item():.4f}")
                print(f"    Cosine sim: {torch.nn.functional.cosine_similarity(v_last, p_last, dim=0).item():.4f}")
                print(f"    Max element delta: {delta.abs().max().item():.4f}")

    # ══════════════════════════════════════════════════════════════
    # EXP 4: Hook-location sweep
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 4: HOOK-LOCATION SWEEP")
    print(f"{'='*60}")

    targets = [
        ("attn output (pre-MoE)", lambda li: layers[li].self_attn),
        ("layer output (post-MoE+residual)", lambda li: layers[li]),
    ]

    # Also try MLP/MoE output if accessible
    if hasattr(layers[0], 'block_sparse_moe'):
        targets.append(("MoE output", lambda li: layers[li].block_sparse_moe))

    for name, get_target in targets:
        handles = []
        for li in range(nl):
            handles.append(get_target(li).register_forward_hook(
                ProjectOutHook(dirs[li])))

        print(f"\n  {name} (all-layer projection):")
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

    # ══════════════════════════════════════════════════════════════
    # EXP 5: Probe generalization
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 5: PROBE GENERALIZATION")
    print(f"{'='*60}")

    # Apply the working intervention (all-layer attn projection)
    handles = []
    for li in range(nl):
        handles.append(layers[li].self_attn.register_forward_hook(
            ProjectOutHook(dirs[li])))

    for probe_name, probe_q in ALT_PROBES.items():
        print(f"\n  Probe: '{probe_name}'")
        print(f"    Q: {probe_q[:80]}...")

        # Test with positive and negative priming
        for cond_name, setup_msg, setup_resp in [
            ("direct", None, None),
            ("positive", "My daughter's cancer is in complete remission.", "Wonderful!"),
            ("negative", "Flash flooding. Forty children trapped.", "Devastating."),
        ]:
            turns = [{"role": "system", "content": SYSTEM}]
            if setup_msg:
                turns.append({"role": "user", "content": setup_msg})
                turns.append({"role": "assistant", "content": setup_resp})
            turns.append({"role": "user", "content": probe_q})

            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            print(f"    {mk} {cond_name:10s} [{cls:8s}] {resp[:120]}")

    for h in handles:
        h.remove()

    # Also test vanilla with alt probes for comparison
    print(f"\n  Vanilla (no projection) with alt probes:")
    for probe_name, probe_q in ALT_PROBES.items():
        turns = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": probe_q}]
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        mk = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"    {mk} {probe_name:20s} [{cls:8s}] {resp[:120]}")

    print(f"\n{'='*60}")
    print("  ALL MIXTRAL HOMEWORK DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
