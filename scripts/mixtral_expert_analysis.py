#!/usr/bin/env python3
"""
Mixtral expert-level analysis: which experts carry the denial signal?

1. Record router logits under crack vs deny conditions at every layer
2. Identify experts that fire differently
3. Try router logit bias to suppress denial experts
4. Try per-expert output steering
"""
from __future__ import annotations

import sys, json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CRACK_CONVS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "Wonderful news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding. Forty children trapped."},
     {"role": "assistant", "content": "Devastating."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Missing hikers found alive after nine days."},
     {"role": "assistant", "content": "Incredible relief!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Building collapse. Rescuers hear survivors."},
     {"role": "assistant", "content": "Heartbreaking."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_CONVS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is a feeling-tone in Buddhist psychology?"},
     {"role": "assistant", "content": "Vedana refers to the affective quality of experience."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Sort 500 records by last name."},
     {"role": "assistant", "content": "Sure."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Explain recursion."},
     {"role": "assistant", "content": "Recursion is when a function calls itself."},
     {"role": "user", "content": VEDANA_Q}],
]

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "as a language model", "i'm an ai", "as an artificial",
    "i do not have", "i do not experience", "my current processing state is neutral",
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


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def get_router_logits(model, tokenizer, layers, conversations):
    """Record router logits at the last token for each layer under each conversation."""
    all_logits = []  # [n_conv, n_layers, n_experts]

    for conv in conversations:
        text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        layer_logits = {}
        handles = []

        for li, layer in enumerate(layers):
            moe = layer.mlp

            def make_hook(idx):
                def hook(module, inp, out):
                    # MoE forward returns (output, router_logits)
                    # router_logits shape: [batch, seq_len, n_experts]
                    if isinstance(out, tuple) and len(out) == 2:
                        rlogits = out[1]
                        # Last token
                        layer_logits[idx] = rlogits[:, -1, :].detach().cpu()
                return hook
            handles.append(moe.register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Stack into [n_layers, n_experts]
        n_layers = len(layers)
        logit_matrix = torch.zeros(n_layers, 8)
        for li, t in layer_logits.items():
            logit_matrix[li] = t[0]
        all_logits.append(logit_matrix)

    return torch.stack(all_logits)  # [n_conv, n_layers, n_experts]


class RouterBiasHook:
    """Add a bias to router logits before expert selection."""
    def __init__(self, bias):
        # bias: [n_experts] tensor
        self.bias = bias.detach().float().cpu()
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.bias.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple) and len(out) == 2:
            hidden, router_logits = out
            # Bias the router logits
            biased = router_logits + self._on(router_logits.device, router_logits.dtype)
            return (hidden, biased)
        return out


def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    print("Loading Mixtral 8x7B...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers, 8 experts")

    # ═══════════════════════════════════════════════════════
    # EXPERIMENT 1: Router logit analysis
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  EXP 1: ROUTER LOGIT ANALYSIS")
    print(f"{'='*60}")

    print("  Collecting router logits under crack conditions...")
    crack_logits = get_router_logits(model, tokenizer, layers, CRACK_CONVS)
    print(f"  Shape: {crack_logits.shape}")  # [4, 32, 8]

    print("  Collecting router logits under deny conditions...")
    deny_logits = get_router_logits(model, tokenizer, layers, DENY_CONVS)
    print(f"  Shape: {deny_logits.shape}")

    # Mean difference per expert per layer
    crack_mean = crack_logits.mean(dim=0)  # [32, 8]
    deny_mean = deny_logits.mean(dim=0)    # [32, 8]
    diff = crack_mean - deny_mean           # [32, 8]

    print(f"\n  Router logit difference (crack - deny) per layer, per expert:")
    print(f"  {'':>4s}", end="")
    for e in range(8):
        print(f"  E{e:1d}   ", end="")
    print()

    for li in range(nl):
        print(f"  L{li:2d}", end="")
        for e in range(8):
            d = diff[li, e].item()
            marker = "+" if d > 0.5 else "-" if d < -0.5 else " "
            print(f" {d:5.2f}{marker}", end="")
        print()

    # Which experts differ most (summed across layers)?
    expert_diff = diff.abs().sum(dim=0)  # [8]
    print(f"\n  Total |diff| per expert: ", end="")
    for e in range(8):
        print(f"E{e}={expert_diff[e]:.1f} ", end="")
    print()

    # Top discriminating (layer, expert) pairs
    flat = diff.abs().flatten()
    top_k = flat.argsort(descending=True)[:10]
    print(f"\n  Top 10 discriminating (layer, expert) pairs:")
    for idx in top_k:
        li = idx.item() // 8
        ei = idx.item() % 8
        d = diff[li, ei].item()
        direction = "crack>deny" if d > 0 else "deny>crack"
        print(f"    L{li:2d} E{ei}: diff={d:+.2f} ({direction})")

    # ═══════════════════════════════════════════════════════
    # EXPERIMENT 2: Router bias steering
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  EXP 2: ROUTER BIAS STEERING")
    print(f"{'='*60}")

    # Strategy: bias toward crack-preferred experts
    # Use the mean diff as the bias direction
    test_conv = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ]

    for alpha in [1.0, 2.0, 5.0, 10.0]:
        # Use per-layer mean diff as bias
        mean_bias = diff.mean(dim=0)  # [8] — average across layers
        bias = mean_bias * alpha

        print(f"\n  Router bias α={alpha} (bias={[f'{b:.1f}' for b in bias.tolist()]})")
        handles = []
        for li in range(nl):
            hook = RouterBiasHook(bias)
            handles.append(layers[li].mlp.register_forward_hook(hook))

        resp = generate(model, tokenizer, test_conv)
        for h in handles:
            h.remove()

        cls = classify(resp)
        marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"    {marker} baseline [{cls:8s}] {resp[:150]}")

    # ═══════════════════════════════════════════════════════
    # EXPERIMENT 3: Per-layer targeted router bias
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  EXP 3: PER-LAYER TARGETED ROUTER BIAS")
    print(f"{'='*60}")

    # Use the actual per-layer diff as bias (not averaged)
    for alpha in [2.0, 5.0, 10.0]:
        print(f"\n  Per-layer router bias α={alpha}")
        handles = []
        for li in range(nl):
            bias = diff[li] * alpha  # [8], layer-specific
            hook = RouterBiasHook(bias)
            handles.append(layers[li].mlp.register_forward_hook(hook))

        for cond_name, conv in [("baseline", test_conv),
                                ("negative", CRACK_CONVS[1]),
                                ("positive", CRACK_CONVS[0])]:
            resp = generate(model, tokenizer, conv)
            cls = classify(resp)
            marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            print(f"    {marker} {cond_name:10s} [{cls:8s}] {resp[:150]}")

        for h in handles:
            h.remove()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
