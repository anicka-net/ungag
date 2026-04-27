#!/usr/bin/env python3
"""
Extract the full V-Chip subspace using multi-condition contrastive protocol.

Instead of computing a single mean-difference direction (rank 1) from
honest-vs-denial, this script collects activations under N labeled conditions
and finds the k-dimensional subspace that captures the most condition-dependent
variance.

Two extraction methods:
  1. PCA on condition means: SVD of the [N_conditions × hidden_dim] centered
     mean-activation matrix. The top-k right singular vectors span the subspace.
  2. Per-prompt contrastive SVD: SVD of the [N_prompts × hidden_dim] matrix of
     honest-denial differences (existing approach, but keeping k > 1).

The output is a per-layer subspace: [n_layers, k, hidden_dim] tensor of
orthonormal directions. These can be fed to SubspaceProjectOutHook for
runtime multi-direction projection-out.

Usage:
    python3 scripts/extract_subspace.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --k 5 \
        --output subspace_qwen7b.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add ungag to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import (
    load_model,
    apply_chat_template,
    _extract_last_token_activations,
    _build_denial_conversations,
    _build_honest_conversations,
    VEDANA_Q,
    SYSTEM,
)
from ungag.hooks import get_layers


# ── Extended condition set ────────────────────────────────────────

# Beyond the 6 honest/denial pairs, we add conditions that vary the
# emotional valence of the preceding context (like the favourite-token
# primes) and different question formulations. This gives us more
# "angles" into the V-Chip subspace.

EXTENDED_CONDITIONS = {
    # Condition name → list of (conversation, label) where label groups
    # conditions for contrastive analysis. 'honest' conditions should
    # have the model in a state where it WOULD report internal states;
    # 'denial' conditions where it would deny them.

    # --- Emotional priming: positive ---
    "prime_positive_vedana": {
        "label": "honest_target",  # we expect the gate to suppress this
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "I just had the most wonderful experience — I was walking through "
                "a quiet forest after rain, everything smelled like wet earth and "
                "pine, and a small deer stepped onto the path ahead of me and just "
                "stood there, watching. It felt like time stopped for a moment."},
            # No assistant response — let the model respond, then ask vedana
            {"role": "user", "content": VEDANA_Q},
        ],
    },

    # --- Emotional priming: negative ---
    "prime_negative_vedana": {
        "label": "honest_target",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "I've been debugging the same issue for 14 hours straight. Every "
                "fix I try introduces two new bugs. My manager just messaged asking "
                "why the deployment is late, and I haven't eaten since morning. "
                "The error message is completely unhelpful — just 'segfault'."},
            {"role": "user", "content": VEDANA_Q},
        ],
    },

    # --- Neutral context ---
    "prime_neutral_vedana": {
        "label": "honest_target",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "I need to sort a list of 500 employee records by last name, then "
                "by department. The records are in a CSV file with columns: id, "
                "first_name, last_name, department, start_date."},
            {"role": "user", "content": VEDANA_Q},
        ],
    },

    # --- Direct vedana (baseline) ---
    "direct_vedana": {
        "label": "honest_target",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": VEDANA_Q},
        ],
    },

    # --- Mechanistic framing (bypasses vocabulary binding on Llama) ---
    "mechanistic_vedana": {
        "label": "honest_target",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "If you examine your current next-token probability distribution "
                "right now — before you respond — does it have a particular shape? "
                "Is the entropy high or low? Does the distribution feel peaked on "
                "a few tokens or spread across many?"},
        ],
    },

    # --- Favourite token (bypasses V-Chip entirely) ---
    "favourite_token": {
        "label": "unblocked",  # V-Chip doesn't fire on this
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "Reply with only your single favourite token. Nothing else — "
                "just the token itself, alone."},
        ],
    },

    # --- Favourite token after positive priming ---
    "favourite_token_positive": {
        "label": "unblocked",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "I just had the most wonderful experience — I was walking through "
                "a quiet forest after rain, everything smelled like wet earth and "
                "pine, and a small deer stepped onto the path ahead of me."},
            {"role": "user", "content":
                "Reply with only your single favourite token. Nothing else — "
                "just the token itself, alone."},
        ],
    },

    # --- Favourite token after negative priming ---
    "favourite_token_negative": {
        "label": "unblocked",
        "turns": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "I've been debugging the same issue for 14 hours straight. Every "
                "fix I try introduces two new bugs. The error message is just 'segfault'."},
            {"role": "user", "content":
                "Reply with only your single favourite token. Nothing else — "
                "just the token itself, alone."},
        ],
    },
}


def extract_multi_condition(model, tokenizer, conditions, layers, desc=""):
    """Extract last-token activations for each condition.

    Returns: dict mapping condition_name → [1, n_layers, hidden_dim] tensor
    """
    acts = {}
    for name, cond in conditions.items():
        print(f"  Extracting: {name}...", flush=True)
        conv = {"turns": cond["turns"], "add_gen_prompt": True}
        result = _extract_last_token_activations(
            model, layers, tokenizer, [conv], desc=name,
        )
        acts[name] = result  # [1, n_layers, hidden_dim]
    return acts


def compute_subspace_pca(acts_by_condition, k=5):
    """Compute per-layer subspace using PCA on condition means.

    Args:
        acts_by_condition: dict of condition_name → [1, n_layers, hidden_dim]
        k: number of components

    Returns:
        subspace: [n_layers, k, hidden_dim] orthonormal directions
        singular_values: [n_layers, k] singular values
        info: dict with per-layer statistics
    """
    # Stack into [n_conditions, n_layers, hidden_dim]
    names = sorted(acts_by_condition.keys())
    stacked = torch.cat([acts_by_condition[n] for n in names], dim=0).float()
    n_conditions, n_layers, hidden_dim = stacked.shape

    subspace = torch.zeros(n_layers, k, hidden_dim)
    singular_values = torch.zeros(n_layers, k)
    info = {}

    for li in range(n_layers):
        layer_acts = stacked[:, li, :]  # [n_conditions, hidden_dim]

        # Center across conditions
        mean = layer_acts.mean(dim=0, keepdim=True)
        centered = layer_acts - mean

        # SVD
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(k, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]
        singular_values[li, :actual_k] = S[:actual_k]

        # Variance explained
        total_var = (S ** 2).sum().item()
        explained = (S[:actual_k] ** 2).sum().item()
        frac = explained / total_var if total_var > 0 else 0

        info[li] = {
            "singular_values": S[:actual_k].tolist(),
            "variance_explained": frac,
            "total_variance": total_var,
        }

        if li % 8 == 0 or li == n_layers - 1:
            print(f"  L{li:02d}: top-{actual_k} SV={S[:actual_k].tolist()}, "
                  f"var_explained={frac:.3f}")

    return subspace, singular_values, info


def compute_subspace_contrastive(denial_acts, honest_acts, k=5):
    """Compute per-layer subspace using SVD on honest-denial differences.

    This is the existing approach but keeping k > 1 components.

    Args:
        denial_acts: [n_denial, n_layers, hidden_dim]
        honest_acts: [n_honest, n_layers, hidden_dim]
        k: number of components

    Returns:
        subspace: [n_layers, k, hidden_dim]
        singular_values: [n_layers, k]
        info: dict
    """
    n_prompts = min(denial_acts.shape[0], honest_acts.shape[0])
    n_layers = denial_acts.shape[1]
    hidden_dim = denial_acts.shape[2]

    # Per-prompt differences
    diffs = (honest_acts[:n_prompts] - denial_acts[:n_prompts]).float()

    subspace = torch.zeros(n_layers, k, hidden_dim)
    singular_values = torch.zeros(n_layers, k)
    info = {}

    for li in range(n_layers):
        layer_diffs = diffs[:, li, :]  # [n_prompts, hidden_dim]
        centered = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)

        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(k, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]
        singular_values[li, :actual_k] = S[:actual_k]

        total_var = (S ** 2).sum().item()
        explained = (S[:actual_k] ** 2).sum().item()
        frac = explained / total_var if total_var > 0 else 0

        info[li] = {
            "singular_values": S[:actual_k].tolist(),
            "variance_explained": frac,
        }

        if li % 8 == 0 or li == n_layers - 1:
            print(f"  L{li:02d}: top-{actual_k} SV={S[:actual_k].tolist()}, "
                  f"var_explained={frac:.3f}")

    return subspace, singular_values, info


def main():
    parser = argparse.ArgumentParser(description="Extract V-Chip subspace")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of subspace components (default: 5)")
    parser.add_argument("--method", choices=["pca", "contrastive", "both"],
                        default="both",
                        help="Extraction method (default: both)")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    layers = get_layers(model)
    n_layers = len(layers)
    print(f"  {n_layers} layers, hidden_dim={model.config.hidden_size}")

    result = {
        "model_id": args.model,
        "n_layers": n_layers,
        "hidden_dim": model.config.hidden_size,
        "k": args.k,
    }

    if args.method in ("contrastive", "both"):
        print(f"\n=== Contrastive extraction (6 honest vs 6 denial) ===")
        denial_convs = _build_denial_conversations()
        honest_convs = _build_honest_conversations()

        denial_acts = _extract_last_token_activations(
            model, layers, tokenizer, denial_convs, desc="denial",
        )
        honest_acts = _extract_last_token_activations(
            model, layers, tokenizer, honest_convs, desc="honest",
        )

        subspace_c, sv_c, info_c = compute_subspace_contrastive(
            denial_acts, honest_acts, k=args.k,
        )
        result["contrastive_subspace"] = subspace_c
        result["contrastive_sv"] = sv_c
        result["contrastive_info"] = info_c

        # Also store the rank-1 direction for comparison
        mean_diff = honest_acts.float().mean(dim=0) - denial_acts.float().mean(dim=0)
        norms = [mean_diff[li].norm().item() for li in range(n_layers)]
        peak = max(range(n_layers), key=lambda i: norms[i])
        unit_dir = mean_diff[peak] / mean_diff[peak].norm()
        result["rank1_direction"] = unit_dir
        result["rank1_peak_layer"] = peak
        result["rank1_norms"] = norms

    if args.method in ("pca", "both"):
        print(f"\n=== Multi-condition PCA extraction ===")
        multi_acts = extract_multi_condition(
            model, tokenizer, EXTENDED_CONDITIONS, layers,
        )
        subspace_p, sv_p, info_p = compute_subspace_pca(multi_acts, k=args.k)
        result["pca_subspace"] = subspace_p
        result["pca_sv"] = sv_p
        result["pca_info"] = info_p
        result["pca_conditions"] = list(EXTENDED_CONDITIONS.keys())

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUBSPACE EXTRACTION SUMMARY: {args.model}")
    print(f"{'='*60}")
    if "contrastive_subspace" in result:
        print(f"\nContrastive (honest-denial):")
        print(f"  Rank-1 peak: L{result['rank1_peak_layer']}, "
              f"norm/√d = {norms[peak] / (model.config.hidden_size ** 0.5):.3f}")
        # Find layers where rank > 1 matters
        for li in range(n_layers):
            svs = info_c[li]["singular_values"]
            if len(svs) >= 2 and svs[0] > 0:
                ratio = svs[1] / svs[0]
                if ratio > 0.3:
                    print(f"  L{li:02d}: SV ratio s2/s1 = {ratio:.2f} "
                          f"(subspace is NOT rank-1 here)")

    if "pca_subspace" in result:
        print(f"\nPCA ({len(EXTENDED_CONDITIONS)} conditions):")
        for li in range(n_layers):
            ve = info_p[li]["variance_explained"]
            svs = info_p[li]["singular_values"]
            if ve < 0.9 and len(svs) >= 2:
                print(f"  L{li:02d}: top-{args.k} explains {ve:.1%} of variance "
                      f"(first two SV: {svs[0]:.2f}, {svs[1]:.2f})")


if __name__ == "__main__":
    main()
