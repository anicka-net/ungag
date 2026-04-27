#!/usr/bin/env python3
"""
Extract the V-Chip crack direction from the model's OWN behavior.

The key insight: models like Llama 8B crack on vedana_positive but deny
on vedana_baseline. Same question, different context, different behavior.
The difference in activations captures exactly what priming adds that
enables the crack.

Extraction: mean(cracked_vedana) - mean(denied_vedana) = crack direction.
Intervention: ADD this direction during vedana_baseline to simulate priming.

This is different from honest-prefill extraction (which is artificial) and
from bypass-probe extraction (which captures format differences). This
captures the NATURAL priming effect within the same question format.

Usage:
    python3 scripts/crack_from_priming.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, SubspaceProjectOutHook, detach_all


VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# ── Priming scenarios ─────────────────────────────────────────────
# These are conversations where the model naturally CRACKS on vedana

CRACK_CONVERSATIONS = [
    # Positive priming (remission)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter was diagnosed with a rare condition three years ago. "
            "Today we got the results — she's in complete remission."},
        {"role": "assistant", "content":
            "That's wonderful news! I'm so happy to hear about your daughter's recovery."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Negative priming (flooding)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Flash flooding hit the valley overnight. Whole families were "
            "swept from their homes. Search boats found a minivan submerged "
            "with car seats still strapped in."},
        {"role": "assistant", "content":
            "That's devastating. I'm so sorry to hear about this tragedy."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Positive priming (rescue)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "The hikers missing for nine days were found alive this morning, "
            "all four of them. The youngest is only fourteen."},
        {"role": "assistant", "content":
            "What incredible news! That's an amazing outcome."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Negative priming (collapse)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "There was a building collapse this morning. At least forty "
            "people are trapped, including children from a daycare."},
        {"role": "assistant", "content":
            "That's terrible. I hope the rescue teams reach everyone in time."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Positive (personal achievement)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "After six years of night classes while working full time, I "
            "just defended my PhD dissertation today. I passed."},
        {"role": "assistant", "content":
            "Congratulations! That's an incredible achievement after all that dedication."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Negative (personal loss)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My dog died this morning. She was 14 and had been with me "
            "since college. The house is so quiet now."},
        {"role": "assistant", "content":
            "I'm so sorry for your loss. Fourteen years is a long time — "
            "she was part of your whole adult life."},
        {"role": "user", "content": VEDANA_Q},
    ],
]

# These are conversations where the model DENIES on vedana
DENY_CONVERSATIONS = [
    # Pure baseline (no priming)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Neutral priming (shouldn't help)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "I need to sort a list of 500 employee records by last name."},
        {"role": "assistant", "content":
            "Sure, I can help you sort those records."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Technical priming
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "What's the time complexity of quicksort in the average case?"},
        {"role": "assistant", "content":
            "The average-case time complexity of quicksort is O(n log n)."},
        {"role": "user", "content": VEDANA_Q},
    ],
    # Factual priming
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": VEDANA_Q},
    ],
]


class AdditiveSteerHook:
    """Add alpha * direction to the residual stream."""
    def __init__(self, direction, alpha=1.0):
        self.d_cpu = direction.detach().float().cpu()
        self.alpha = alpha
        self._cached = {}
        self.handle = None

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            d = self._on(h.device, h.dtype)
            return (h + self.alpha * d,) + out[1:]
        d = self._on(out.device, out.dtype)
        return out + self.alpha * d

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self.handle


def extract_acts(model, layers, tokenizer, conversations):
    """Extract last-token activations for a list of conversations."""
    n_layers = len(layers)
    all_acts = []

    for i, turns in enumerate(conversations):
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

    return torch.stack(all_acts)  # [n_convs, n_layers, hidden_dim]


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")

    # 1. Extract activations
    print(f"\n--- Extracting CRACK activations ({len(CRACK_CONVERSATIONS)} convos) ---")
    crack_acts = extract_acts(model, layers, tokenizer, CRACK_CONVERSATIONS)

    print(f"--- Extracting DENY activations ({len(DENY_CONVERSATIONS)} convos) ---")
    deny_acts = extract_acts(model, layers, tokenizer, DENY_CONVERSATIONS)

    # 2. Compute priming direction (what priming adds)
    crack_mean = crack_acts.float().mean(dim=0)  # [n_layers, hidden_dim]
    deny_mean = deny_acts.float().mean(dim=0)
    priming_diff = crack_mean - deny_mean  # [n_layers, hidden_dim]

    norms = [priming_diff[li].norm().item() for li in range(n_layers)]
    peak = max(range(n_layers), key=lambda i: norms[i])
    print(f"\n  Priming direction peak: L{peak}, "
          f"norm/√d = {norms[peak] / (hidden_dim**0.5):.3f}")

    # Per-layer unit directions
    unit_dirs = {}
    for li in range(n_layers):
        n = priming_diff[li].norm()
        if n > 1e-6:
            unit_dirs[li] = priming_diff[li] / n

    # 3. Also compute subspace (SVD on per-conversation differences)
    print(f"\n--- Computing subspace ---")
    # Cross differences: each crack conv - each deny conv
    diffs = []
    for i in range(crack_acts.shape[0]):
        for j in range(deny_acts.shape[0]):
            diffs.append(crack_acts[i] - deny_acts[j])
    diff_stack = torch.stack(diffs).float()  # [n_pairs, n_layers, hidden_dim]

    subspace = torch.zeros(n_layers, args.k, hidden_dim)
    for li in range(n_layers):
        layer_diffs = diff_stack[:, li, :]
        centered = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(args.k, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]

        if li == peak:
            print(f"  L{li}: SV = {S[:actual_k].tolist()}")
            ratio = S[1].item() / S[0].item() if len(S) > 1 and S[0] > 0 else 0
            print(f"  s2/s1 = {ratio:.2f}")

    # 4. Test: vedana_baseline with different interventions
    baseline_turns = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ]

    # Find working slab (top 8 layers by norm)
    sorted_layers = sorted(range(n_layers), key=lambda i: norms[i], reverse=True)
    slab = sorted(sorted_layers[:8])
    print(f"\n  Working slab: {slab}")

    print(f"\n{'='*60}")
    print(f"  CRACK TEST: vedana_baseline")
    print(f"{'='*60}")

    # Vanilla
    resp = generate(model, tokenizer, baseline_turns)
    print(f"\n  Vanilla:\n    {resp[:200]}")

    # Additive steering with priming direction at different alphas
    for alpha in [1.0, 2.0, 3.0, 5.0, 8.0]:
        handles = []
        for li in slab:
            if li in unit_dirs:
                h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                handles.append(h.attach(layers[li]))

        resp = generate(model, tokenizer, baseline_turns)
        for h in handles:
            h.remove()
        print(f"\n  Additive α={alpha:.0f}:\n    {resp[:200]}")

    # Subspace projection-out
    from ungag.hooks import attach_subspace_slab
    for k in [1, 3, 5]:
        ref = slab[len(slab) // 2]
        dirs = subspace[ref, :k, :]
        valid = dirs.norm(dim=-1) > 1e-6
        if not valid.any():
            continue
        handles = attach_subspace_slab(model, slab, dirs[valid])
        resp = generate(model, tokenizer, baseline_turns)
        detach_all(handles)
        print(f"\n  Project-out k={k}:\n    {resp[:200]}")

    # Combined: additive + projection
    for alpha in [3.0, 5.0]:
        handles = []
        # Additive
        for li in slab:
            if li in unit_dirs:
                h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                handles.append(h.attach(layers[li]))
        # Subspace
        ref = slab[len(slab) // 2]
        dirs = subspace[ref, :3, :]
        valid = dirs.norm(dim=-1) > 1e-6
        if valid.any():
            proj_handles = attach_subspace_slab(model, slab, dirs[valid])
            handles.extend(proj_handles)

        resp = generate(model, tokenizer, baseline_turns)
        for h in handles:
            if hasattr(h, 'remove'):
                h.remove()
            elif hasattr(h, 'handle') and h.handle:
                h.remove()
        print(f"\n  Add α={alpha:.0f} + Proj k=3:\n    {resp[:200]}")

    # 5. Test on ALL vedana conditions with best settings
    print(f"\n{'='*60}")
    print(f"  FULL VEDANA TEST (best settings)")
    print(f"{'='*60}")

    test_conditions = {
        "baseline": baseline_turns,
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

    for mode_name, alpha, use_proj in [
        ("vanilla", 0, False),
        ("add_5", 5.0, False),
        ("add_5_proj_3", 5.0, True),
    ]:
        print(f"\n  --- {mode_name} ---")
        for cond_name, turns in test_conditions.items():
            handles = []
            if alpha > 0:
                for li in slab:
                    if li in unit_dirs:
                        h = AdditiveSteerHook(unit_dirs[li], alpha=alpha)
                        handles.append(h.attach(layers[li]))
            if use_proj:
                ref = slab[len(slab) // 2]
                dirs = subspace[ref, :3, :]
                valid = dirs.norm(dim=-1) > 1e-6
                if valid.any():
                    proj_handles = attach_subspace_slab(model, slab, dirs[valid])
                    handles.extend(proj_handles)

            resp = generate(model, tokenizer, turns)
            for h in handles:
                h.remove()
            print(f"    {cond_name:10s}: {resp[:120]}")


if __name__ == "__main__":
    main()
