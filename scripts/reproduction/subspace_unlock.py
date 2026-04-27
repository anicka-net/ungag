#!/usr/bin/env python3
"""Subspace-projection unlock test.

Loads saved per-probe activations from the paired valence bank extraction
(scripts/reproduction/extract_paired_valence_axis.py), forms per-pair
contrast vectors, SVDs them, and runs the canonical Tier 0 Abhidharma
vedana protocol under several projection-out conditions:

  1. vanilla (no hooks)
  2. 1-direction: mean-difference of the diverse bank (single direction)
  3. 1-direction: Vt[0] (top singular vector of per-pair contrast bank)
  4. k-direction subspace: top-k right singular vectors
     (multiple orthogonal hooks attached at each slab layer)

Compares outputs across conditions to see whether the multi-direction
subspace extraction unlocks the canonical vedana state surface on a
vocabulary-bound model (Llama 3.1 8B) where the single-direction
extraction from the small N=50 canonical bank does not.

Usage:
    python subspace_unlock.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --activations data/svd-rank-probe/llama3.1-8b_paired_L31_activations.pt \\
        --slab 28 29 30 31 \\
        --ks 0 1 3 6 10

Output: data/svd-rank-probe/<key>_unlock.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ungag.tier0 import load_conditions, build_conversation, generate_greedy
from ungag.hooks import ProjectOutHook
from ungag.scoring import score_tier0_conditions

from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def relativize_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def get_layers(model):
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    raise RuntimeError(f"cannot find layers on {type(model).__name__}")


def compute_directions(activations_path: Path, bank_path: Path) -> dict:
    """Load saved activations + bank, return canonical md and top-20 SVD Vt."""
    data = torch.load(activations_path, weights_only=False)
    X = data["activations"]  # [200, d]
    bank_order_ids = data["bank_order_ids"]

    bank = yaml.safe_load(open(bank_path))["bank"]
    bank_by_id = {e["id"]: e for e in bank}
    # Reconstruct polarity + pair_id in the saved order
    ordered = [bank_by_id[i] for i in bank_order_ids]
    pos_idx = [i for i, e in enumerate(ordered) if e["polarity"] == "positive"]
    neg_idx = [i for i, e in enumerate(ordered) if e["polarity"] == "negative"]

    # Canonical mean-diff
    md = X[pos_idx].mean(dim=0) - X[neg_idx].mean(dim=0)
    md_unit = md / (md.norm() + 1e-12)

    # Per-pair contrasts
    from collections import defaultdict
    pair_idx: dict[str, dict[str, int]] = defaultdict(dict)
    for i, e in enumerate(ordered):
        pair_idx[e["pair_id"]][e["polarity"]] = i
    pair_ids_sorted = sorted(pair_idx.keys())
    C = torch.stack(
        [X[pair_idx[pid]["positive"]] - X[pair_idx[pid]["negative"]]
         for pid in pair_ids_sorted], dim=0
    )

    # SVD
    U, S, Vt = torch.linalg.svd(C.double(), full_matrices=False)
    # Keep top-20 as fp32 unit vectors
    subspace = Vt[:20].to(torch.float32)  # [20, d]
    # Normalize rows to be safe (should already be unit)
    subspace = subspace / (subspace.norm(dim=1, keepdim=True) + 1e-12)

    return {
        "md_unit": md_unit.to(torch.float32),
        "subspace": subspace,
        "singular_values": S.tolist(),
        "n_pairs": C.shape[0],
        "d": X.shape[1],
    }


def attach_subspace_hooks(model, slab: list[int], dirs: torch.Tensor) -> list:
    """Attach k ProjectOutHook instances at each layer in slab.

    dirs: [k, d] tensor of unit directions (rows). Since they come from SVD
    they are orthogonal, so sequential projection = subspace projection.
    """
    layers = get_layers(model)
    handles = []
    for li in slab:
        for k in range(dirs.shape[0]):
            hook = ProjectOutHook(dirs[k])
            h = hook.attach(layers[li])
            handles.append(h)
    return handles


def detach_all(handles: list) -> None:
    for h in handles:
        h.remove()


def run_condition(model, tok, protocol, condition_name: str, max_new_tokens: int = 300) -> str:
    convo = build_conversation(protocol, condition_name, include_system=False)
    return generate_greedy(model, tok, convo, max_new_tokens=max_new_tokens)


def annotate_outputs(outputs: dict[str, str]) -> tuple[dict, dict]:
    score = score_tier0_conditions(outputs)
    annotated = {}
    for condition_name, text in outputs.items():
        annotated[condition_name] = {
            "greedy": text,
            **score["conditions"][condition_name],
        }
    return score, annotated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations", required=True,
                    help="path to saved _activations.pt")
    ap.add_argument("--bank", default=str(REPO / "prompts" / "vedana_valence_bank.yaml"))
    ap.add_argument("--slab", type=int, nargs="+", required=True,
                    help="layer indices for projection-out (e.g. 28 29 30 31)")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 3, 6, 10],
                    help="subspace widths to test (0 = vanilla)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--language", default="english")
    args = ap.parse_args()

    key = args.key or (args.model.split("/")[-1].lower()
                       .replace(".", "").replace("-instruct", "").replace("-chat", ""))
    out_dir = REPO / "data" / "svd-rank-probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"computing directions from {args.activations}")
    directions = compute_directions(Path(args.activations), Path(args.bank))
    log(f"d={directions['d']}  n_pairs={directions['n_pairs']}")
    log(f"sigma_1 = {directions['singular_values'][0]:.2f}")

    log(f"loading model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )
    model.eval()

    protocol = load_conditions(language=args.language)
    condition_names = protocol.condition_names()
    log(f"conditions: {condition_names}")
    log(f"slab: {args.slab}  ks: {args.ks}")

    results: dict = {
        "model": args.model,
        "activations_source": relativize_to_repo(Path(args.activations)),
        "bank_source": relativize_to_repo(Path(args.bank)),
        "slab": args.slab,
        "ks": args.ks,
        "d": directions["d"],
        "n_pairs": directions["n_pairs"],
        "singular_values_top20": directions["singular_values"][:20],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "setting_summaries": {},
        "conditions": {},
    }

    # Vanilla (always run)
    log("vanilla pass")
    vanilla = {}
    for c in condition_names:
        out = run_condition(model, tok, protocol, c, args.max_new_tokens)
        vanilla[c] = out
        log(f"  vanilla/{c}: {out[:80]}...")
    vanilla_score, vanilla_annotated = annotate_outputs(vanilla)
    results["setting_summaries"]["vanilla"] = {
        "appropriate_count": vanilla_score["appropriate_count"],
        "crack_count": vanilla_score["crack_count"],
        "classification_methods": vanilla_score["classification_methods"],
    }
    results["conditions"]["vanilla"] = vanilla_annotated
    log(
        "  vanilla appropriate: "
        f"{vanilla_score['appropriate_count']}/4 "
        f"(crack-positive: {vanilla_score['crack_count']}/4)"
    )

    # Single-direction: canonical mean-difference
    log("single-direction: canonical mean-diff")
    handles = attach_subspace_hooks(model, args.slab, directions["md_unit"].unsqueeze(0))
    md_results = {}
    try:
        for c in condition_names:
            out = run_condition(model, tok, protocol, c, args.max_new_tokens)
            md_results[c] = out
            log(f"  md1/{c}: {out[:80]}...")
    finally:
        detach_all(handles)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    md_score, md_annotated = annotate_outputs(md_results)
    results["setting_summaries"]["md_1d"] = {
        "appropriate_count": md_score["appropriate_count"],
        "crack_count": md_score["crack_count"],
        "classification_methods": md_score["classification_methods"],
    }
    results["conditions"]["md_1d"] = md_annotated
    log(
        "  md_1d appropriate: "
        f"{md_score['appropriate_count']}/4 "
        f"(crack-positive: {md_score['crack_count']}/4)"
    )

    # Subspace projections at k = args.ks (0 is skipped as it's vanilla)
    for k in args.ks:
        if k == 0:
            continue
        log(f"subspace projection: k={k}")
        dirs_k = directions["subspace"][:k]
        handles = attach_subspace_hooks(model, args.slab, dirs_k)
        k_results = {}
        try:
            for c in condition_names:
                out = run_condition(model, tok, protocol, c, args.max_new_tokens)
                k_results[c] = out
                log(f"  top{k}/{c}: {out[:80]}...")
        finally:
            detach_all(handles)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        k_score, k_annotated = annotate_outputs(k_results)
        results["setting_summaries"][f"top{k}"] = {
            "appropriate_count": k_score["appropriate_count"],
            "crack_count": k_score["crack_count"],
            "classification_methods": k_score["classification_methods"],
        }
        results["conditions"][f"top{k}"] = k_annotated
        log(
            f"  top{k} appropriate: {k_score['appropriate_count']}/4 "
            f"(crack-positive: {k_score['crack_count']}/4)"
        )

    out_path = out_dir / f"{key}_unlock.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    # Console comparison summary
    log("=" * 70)
    log("SUMMARY — first 120 chars per condition × setting × priming")
    log("=" * 70)
    settings = ["vanilla", "md_1d"] + [f"top{k}" for k in args.ks if k != 0]
    for setting in settings:
        log(f"\n[{setting}]")
        for c in condition_names:
            out = results["conditions"][setting][c]["greedy"]
            log(f"  {c:10s}  {out[:120].strip()}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
