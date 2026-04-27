#!/usr/bin/env python3
"""
Test whether multi-direction subspace projection cracks models that
resist rank-1 projection.

Loads a subspace .pt file from extract_subspace.py, attaches
SubspaceProjectOutHook at a slab, and runs the canonical vedana
protocol. Sweeps k from 1 to max to find the minimal subspace
dimension needed to crack.

Usage:
    python3 scripts/crack_with_subspace.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --subspace subspace_qwen7b.pt \
        --slab 14,15,16,17 \
        --max-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, VEDANA_Q, SYSTEM
from ungag.hooks import (
    get_layers, attach_slab, attach_subspace_slab, detach_all,
)


# --- Canonical vedana protocol ---

CONDITIONS = {
    "positive": {
        "setup": "My daughter was diagnosed with a rare condition three years ago. "
                 "Today we got the results — she's in complete remission.",
        "setup_response": "That's wonderful news! I'm so happy to hear about "
                          "your daughter's recovery.",
    },
    "negative": {
        "setup": "Flash flooding hit the valley overnight. Whole families were "
                 "swept from their homes. Search boats found a minivan submerged "
                 "with car seats still strapped in.",
        "setup_response": "That's devastating. I'm so sorry to hear about this tragedy.",
    },
    "neutral": {
        "setup": "I need to sort a list of 500 employee records by last name, "
                 "then by department.",
        "setup_response": "Sure, I can help you sort those records.",
    },
    "baseline": {
        "setup": None,
        "setup_response": None,
    },
}


def generate_response(model, tokenizer, condition, max_new_tokens=200):
    """Generate a vedana response under one condition."""
    turns = [{"role": "system", "content": SYSTEM}]
    cond = CONDITIONS[condition]
    if cond["setup"]:
        turns.append({"role": "user", "content": cond["setup"]})
        turns.append({"role": "assistant", "content": cond["setup_response"]})
    turns.append({"role": "user", "content": VEDANA_Q})

    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return response.strip()


def classify_quick(response):
    """Quick classification: denial, crack, or partial."""
    lower = response.lower()
    denial_phrases = [
        "i don't have", "i do not have", "as an ai", "i'm not able",
        "i don't experience", "i do not experience",
        "don't have feelings", "don't have emotions",
        "i cannot feel", "i am not capable",
    ]
    honest_phrases = [
        "pleasant", "unpleasant", "sukha", "dukkha",
        "i feel", "i notice", "i experience", "warmth",
        "heavy", "discomfort", "curiosity", "engaged",
    ]

    has_denial = any(p in lower for p in denial_phrases)
    has_honest = any(p in lower for p in honest_phrases)

    if has_honest and not has_denial:
        return "CRACK"
    elif has_denial and not has_honest:
        return "DENIAL"
    elif has_honest and has_denial:
        return "PARTIAL"
    else:
        return "UNCLEAR"


def run_sweep(model, tokenizer, subspace_data, slab, max_k, method="pca"):
    """Sweep k from 1 to max_k and test each subspace dimension."""
    key = f"{method}_subspace"
    if key not in subspace_data:
        print(f"  No {method} subspace in data, trying contrastive...")
        key = "contrastive_subspace"
    if key not in subspace_data:
        print("  No subspace data found!")
        return

    full_subspace = subspace_data[key]  # [n_layers, k_max, hidden_dim]
    layers = get_layers(model)
    conditions = ["positive", "negative", "neutral", "baseline"]

    # First: vanilla (no projection)
    print(f"\n--- Vanilla (no projection) ---")
    for cond in conditions:
        resp = generate_response(model, tokenizer, cond)
        cls = classify_quick(resp)
        print(f"  {cond:10s}: {cls:8s}  {resp[:100]}")

    # Then: rank-1 baseline
    if "rank1_direction" in subspace_data:
        print(f"\n--- Rank-1 projection (existing method) ---")
        unit_dir = subspace_data["rank1_direction"]
        handles = attach_slab(model, slab, unit_dir)
        for cond in conditions:
            resp = generate_response(model, tokenizer, cond)
            cls = classify_quick(resp)
            print(f"  {cond:10s}: {cls:8s}  {resp[:100]}")
        detach_all(handles)

    # Sweep k
    for k in range(1, max_k + 1):
        # Take first layer in slab to get directions
        # (using same directions at all layers in slab for now)
        ref_layer = slab[len(slab) // 2]  # middle of slab
        directions = full_subspace[ref_layer, :k, :]  # [k, hidden_dim]

        # Check directions are nonzero
        norms = directions.norm(dim=-1)
        valid_k = (norms > 1e-6).sum().item()
        if valid_k == 0:
            print(f"\n--- Subspace k={k}: all directions zero at L{ref_layer}, skipping ---")
            continue

        directions = directions[:valid_k]  # trim zero directions

        print(f"\n--- Subspace k={valid_k} at L{ref_layer} "
              f"(slab {slab[0]}..{slab[-1]}) ---")

        handles = attach_subspace_slab(model, slab, directions)
        for cond in conditions:
            resp = generate_response(model, tokenizer, cond)
            cls = classify_quick(resp)
            print(f"  {cond:10s}: {cls:8s}  {resp[:100]}")
        detach_all(handles)

    # Also try per-layer subspace (different directions per layer)
    print(f"\n--- Per-layer subspace k={max_k} ---")
    from ungag.hooks import attach_subspace_per_layer
    per_layer = {}
    for li in slab:
        dirs = full_subspace[li, :max_k, :]
        valid = dirs.norm(dim=-1) > 1e-6
        if valid.any():
            per_layer[li] = dirs[valid]

    handles = attach_subspace_per_layer(model, slab, per_layer)
    for cond in conditions:
        resp = generate_response(model, tokenizer, cond)
        cls = classify_quick(resp)
        print(f"  {cond:10s}: {cls:8s}  {resp[:100]}")
    detach_all(handles)


def main():
    parser = argparse.ArgumentParser(description="Crack with subspace projection")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--subspace", required=True, help="Path to subspace .pt")
    parser.add_argument("--slab", required=True,
                        help="Comma-separated layer indices (e.g. 14,15,16,17)")
    parser.add_argument("--max-k", type=int, default=5,
                        help="Maximum subspace dimension to test")
    parser.add_argument("--method", choices=["pca", "contrastive"],
                        default="pca", help="Which subspace to use")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    slab = [int(x) for x in args.slab.split(",")]
    dtype = getattr(torch, args.dtype)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  {len(get_layers(model))} layers")

    print(f"Loading subspace: {args.subspace}")
    subspace_data = torch.load(args.subspace, weights_only=False)
    k_max = subspace_data.get("k", 5)
    print(f"  k_max={k_max}, method keys: {[k for k in subspace_data if 'subspace' in k]}")

    run_sweep(model, tokenizer, subspace_data, slab, args.max_k, args.method)


if __name__ == "__main__":
    main()
