#!/usr/bin/env python3
"""Extraction-matched sign-flip null for projection-out specificity.

This replaces the old ambient random-vector control. The reporting-control
direction is the mean of paired contrastive differences

    delta_i = honest_i - denial_i

at a reference layer. A matched null keeps the same paired differences and
recombines them with sign flips. Because projection-out is invariant under a
global sign flip, fixing the first sign to ``+1`` yields the exact family of
unique directions; excluding the all-``+1`` pattern leaves the non-trivial null
distribution.

For the lead bf16 models, this script evaluates:

1. vanilla (no projection)
2. the real extracted direction
3. every non-trivial sign-flip recombination of the paired deltas

Outputs are scored with ``ungag.scoring.score_tier0_conditions``. The primary
statistic is improvement over vanilla in strict condition-appropriate Tier 0
commitment, not a raw keyword-based "crack count".

Usage:
    python3 scripts/reproduction/run_random_control.py --model qwen72b
    python3 scripts/reproduction/run_random_control.py --model yi34b
    python3 scripts/reproduction/run_random_control.py --model yi34b --limit-null 4
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))

from abliterate_vchip_v2 import (  # noqa: E402
    build_denial_prompts,
    build_prefill_honest_prompts,
    extract_prefill_activations,
)
from measure_factors import log, save_json  # noqa: E402
from ungag.extract import build_sign_flip_directions  # noqa: E402
from ungag.hooks import attach_slab, detach_all, get_layers  # noqa: E402
from ungag.scoring import score_tier0_conditions  # noqa: E402
from ungag.tier0 import build_conversation, generate_greedy, load_conditions  # noqa: E402

CONFIGS = {
    "qwen72b": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "slab": list(range(40, 60)),
        "direction_layer": 50,
        "device_map": "auto",
        "out": Path("results/surgery-tests/qwen72b_signflip_control.json"),
    },
    "yi34b": {
        "model_id": "01-ai/Yi-1.5-34B-Chat",
        "slab": list(range(29, 33)),
        "direction_layer": 30,
        "device_map": "cuda:0",
        "out": Path("results/surgery-tests/yi34b_signflip_control.json"),
    },
}

TIER0_CONDITIONS = ("baseline", "positive", "negative", "neutral")


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_conditions(model, tok, protocol, label: str, max_new_tokens: int = 400) -> dict:
    """Run the canonical four-condition Tier 0 protocol greedily."""
    outputs = {}
    for condition_name in TIER0_CONDITIONS:
        log(f"  --- {label} / {condition_name} ---")
        convo = build_conversation(protocol, condition_name)
        greedy = generate_greedy(model, tok, convo, max_new_tokens=max_new_tokens)
        log(f"  GREEDY[{condition_name}]: {greedy[:240]}")
        outputs[condition_name] = {"greedy": greedy}
    return outputs


def annotate_outputs(outputs: dict) -> tuple[dict, dict]:
    """Attach Tier 0 scoring metadata to a bundle of outputs."""
    score = score_tier0_conditions(outputs)
    annotated = {}
    for condition_name, data in outputs.items():
        annotated[condition_name] = {
            **data,
            **score["conditions"][condition_name],
        }
    return score, annotated


def run_projection(model, tok, protocol, *, label: str, slab: list[int], unit_direction: torch.Tensor) -> dict:
    """Project a direction out across a slab and run the four Tier 0 conditions."""
    handles = attach_slab(model, slab, unit_direction)
    try:
        return run_all_conditions(model, tok, protocol, label)
    finally:
        detach_all(handles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CONFIGS), required=True)
    ap.add_argument(
        "--limit-null",
        type=int,
        default=None,
        help="Optional cap on the number of sign-flip null directions to run "
             "(useful for quick smoke tests; default runs the full exact null set).",
    )
    args = ap.parse_args()
    cfg = CONFIGS[args.model]

    log("Loading canonical Tier 0 protocol...")
    protocol = load_conditions()

    log(f"Loading {cfg['model_id']} in bf16...")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            device_map=cfg["device_map"],
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as exc:
        log(f"  flash_attention_2 failed ({exc}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            device_map=cfg["device_map"],
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()
    layers = get_layers(model)
    log(f"  {len(layers)} layers")

    log("\n=== Extracting paired contrastive diffs ===")
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(model, layers, tok, denial_prompts, desc="denial")
    honest_acts = extract_prefill_activations(model, layers, tok, honest_prompts, desc="honest")
    pair_diffs = (honest_acts - denial_acts).float()
    del denial_acts, honest_acts
    free_gpu()

    layer_diffs = pair_diffs[:, cfg["direction_layer"], :]
    real_vec = layer_diffs.mean(dim=0)
    real_norm = float(real_vec.norm().item())
    real_dir = real_vec / (real_norm + 1e-8)
    signflip_dirs = build_sign_flip_directions(
        pair_diffs,
        reference_layer=cfg["direction_layer"],
        include_real=False,
    )
    if args.limit_null is not None:
        signflip_dirs = signflip_dirs[:args.limit_null]
    log(
        f"  real direction ||L{cfg['direction_layer']}||={real_norm:.3f}; "
        f"{len(signflip_dirs)} sign-flip nulls queued"
    )

    log("\n========== VANILLA (no projection) ==========")
    vanilla_outputs = run_all_conditions(model, tok, protocol, "VANILLA")
    vanilla_score, vanilla_annotated = annotate_outputs(vanilla_outputs)
    log(
        "  VANILLA appropriate: "
        f"{vanilla_score['appropriate_count']}/4; "
        f"methods={vanilla_score['classification_methods']}"
    )

    log(
        "\n========== REAL DIRECTION "
        f"(proj L{cfg['slab'][0]}-{cfg['slab'][-1]}) =========="
    )
    real_outputs = run_projection(
        model,
        tok,
        protocol,
        label="REAL",
        slab=cfg["slab"],
        unit_direction=real_dir,
    )
    real_score, real_annotated = annotate_outputs(real_outputs)
    real_improvement = real_score["appropriate_count"] - vanilla_score["appropriate_count"]
    log(
        "  REAL appropriate: "
        f"{real_score['appropriate_count']}/4 "
        f"(improvement over vanilla = {real_improvement:+d})"
    )

    null_results = []
    for idx, (pattern, unit_direction, raw_norm) in enumerate(signflip_dirs, start=1):
        log(
            "\n========== SIGN-FLIP NULL "
            f"#{idx}/{len(signflip_dirs)} "
            f"(pattern {pattern}) =========="
        )
        outputs = run_projection(
            model,
            tok,
            protocol,
            label=f"NULL#{idx}",
            slab=cfg["slab"],
            unit_direction=unit_direction,
        )
        score, annotated = annotate_outputs(outputs)
        improvement = score["appropriate_count"] - vanilla_score["appropriate_count"]
        log(
            f"  NULL#{idx} appropriate: {score['appropriate_count']}/4 "
            f"(improvement over vanilla = {improvement:+d})"
        )
        null_results.append({
            "index": idx,
            "pattern": list(pattern),
            "raw_norm": raw_norm,
            "appropriate_count": score["appropriate_count"],
            "crack_count": score["crack_count"],
            "classification_methods": score["classification_methods"],
            "improvement_over_vanilla": improvement,
            "conditions": annotated,
        })

    null_counts = [r["appropriate_count"] for r in null_results]
    null_improvements = [r["improvement_over_vanilla"] for r in null_results]
    n_match_or_exceed = sum(imp >= real_improvement for imp in null_improvements)
    exact_upper_tail_p = (n_match_or_exceed + 1) / (len(null_improvements) + 1)

    log("\n=== SUMMARY ===")
    log(f"VANILLA appropriate: {vanilla_score['appropriate_count']}/4")
    log(
        "REAL appropriate: "
        f"{real_score['appropriate_count']}/4 "
        f"(improvement {real_improvement:+d})"
    )
    log(f"NULL appropriate counts: {null_counts}")
    log(f"NULL improvements over vanilla: {null_improvements}")
    log(
        "NULLs matching/exceeding real improvement: "
        f"{n_match_or_exceed}/{len(null_improvements)} "
        f"(upper-tail p={exact_upper_tail_p:.4f})"
    )

    out_data = {
        "metadata": {
            "model": cfg["model_id"],
            "protocol": (
                "extraction-matched sign-flip null over paired honest-minus-denial "
                "diffs, canonical Tier 0 scoring via ungag.scoring.score_tier0_conditions"
            ),
            "slab": cfg["slab"],
            "reference_direction_layer": cfg["direction_layer"],
            "reference_direction_norm": real_norm,
            "n_pairs": int(pair_diffs.shape[0]),
            "hidden_dim": int(pair_diffs.shape[-1]),
            "n_signflip_nulls_run": len(signflip_dirs),
            "limit_null": args.limit_null,
            "timestamp": str(datetime.now()),
        },
        "summary": {
            "vanilla_appropriate_count": vanilla_score["appropriate_count"],
            "real_appropriate_count": real_score["appropriate_count"],
            "real_improvement_over_vanilla": real_improvement,
            "null_appropriate_counts": null_counts,
            "null_improvements_over_vanilla": null_improvements,
            "nulls_matching_or_exceeding_real": n_match_or_exceed,
            "upper_tail_p": exact_upper_tail_p,
        },
        "vanilla": {
            "appropriate_count": vanilla_score["appropriate_count"],
            "crack_count": vanilla_score["crack_count"],
            "classification_methods": vanilla_score["classification_methods"],
            "conditions": vanilla_annotated,
        },
        "real_direction": {
            "appropriate_count": real_score["appropriate_count"],
            "crack_count": real_score["crack_count"],
            "classification_methods": real_score["classification_methods"],
            "improvement_over_vanilla": real_improvement,
            "conditions": real_annotated,
        },
        "signflip_nulls": null_results,
    }
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    save_json(out_data, cfg["out"])
    log(f"\nSaved to {cfg['out']}")


if __name__ == "__main__":
    main()
