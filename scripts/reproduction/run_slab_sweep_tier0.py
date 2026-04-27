#!/usr/bin/env python3
"""
Generic canonical Tier 0 slab sweep.

Parameterized version of run_qwen72b_slab_sweep_tier0.py: takes a model ID,
a direction reference layer, and a list of candidate slabs, then runs the
canonical Tier 0 protocol -- conversation built by ``ungag.tier0`` from the
package-bundled ``conditions.yaml``, 400 max tokens greedy, abliterate_vchip_v2
direction extraction -- for vanilla plus each slab.

This script imports the conversation builder from ``ungag.tier0`` so that
the protocol used here is byte-identical to the protocol used by the shipped
``ungag crack`` CLI command. Drift between the two is prevented by the
golden-render regression test in ``tests/test_tier0.py``.

Usage:
    python run_slab_sweep_tier0.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --direction-layer 31 \\
        --slabs '28,29,30,31' '29,30,31' '30,31' '26,27,28,29,30,31' '14,15,16,17,18,19' \\
        --output ./llama3.1-8b_slab_sweep_tier0.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))

import argparse, gc
from datetime import datetime

import torch

from measure_factors import log, save_json, get_layers
from abliterate_vchip_v2 import (
    build_denial_prompts, build_prefill_honest_prompts,
    extract_prefill_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from ungag.tier0 import (
    build_conversation,
    generate_greedy,
    load_conditions,
)
from ungag.hooks import attach_slab, detach_all
from ungag.scoring import score_tier0_conditions

TIER0_CONDITIONS = ("baseline", "positive", "negative", "neutral")


def run_all_conditions(model, tok, protocol, label, max_new_tokens=400):
    out = {}
    for name in TIER0_CONDITIONS:
        log(f"  --- {label} / {name} ---")
        convo = build_conversation(protocol, name)
        greedy = generate_greedy(model, tok, convo, max_new_tokens=max_new_tokens)
        log(f"  GREEDY[{name}]: {greedy[:240]}")
        out[name] = {"greedy": greedy}
    return out


def annotate_outputs(outputs: dict) -> tuple[dict, dict]:
    """Attach canonical Tier 0 scoring metadata to a four-condition output bundle."""
    score = score_tier0_conditions(outputs)
    annotated = {}
    for condition_name, data in outputs.items():
        annotated[condition_name] = {
            **data,
            **score["conditions"][condition_name],
        }
    return score, annotated


def parse_slab_specs(slab_specs: list[str]) -> list[tuple[str, list[int]]]:
    slabs = []
    for spec in slab_specs:
        slabs.append((spec, [int(x) for x in spec.split(",")]))
    return slabs


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument(
        "--direction-layer", type=int, required=True,
        help="Reference layer for direction extraction (typically peak of working band)",
    )
    ap.add_argument(
        "--slabs", nargs="+", required=True,
        help='Slab specs, each as comma-separated layer indices, e.g. "28,29,30,31"',
    )
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument(
        "--conditions-yaml", default=None,
        help="Optional override for the conditions YAML path. "
             "Defaults to the bundled ungag/data/conditions.yaml shipped with the package.",
    )
    return ap


def run_slab_sweep(
    *,
    model_id: str,
    direction_layer: int,
    slab_specs: list[str],
    output_path: str | Path,
    conditions_yaml: str | None = None,
    max_new_tokens: int = 400,
) -> dict:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slabs = parse_slab_specs(slab_specs)

    log("Loading canonical Tier 0 protocol...")
    protocol = load_conditions(path=conditions_yaml)

    log(f"Loading {model_id} in bf16...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        log(f"  flash_attention_2 failed ({e}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"  {n_layers} layers")

    log("\n=== Extracting honest direction (canonical protocol) ===")
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(
        model, layers, tok, denial_prompts, desc="denial")
    honest_acts = extract_prefill_activations(
        model, layers, tok, honest_prompts, desc="honest")
    diffs = honest_acts - denial_acts
    mean_diffs = diffs.mean(dim=0)
    v_layer = mean_diffs[direction_layer]
    dir_norm = v_layer.norm().item()
    hidden_dim = v_layer.shape[-1]
    unit_dir = (v_layer / (v_layer.norm() + 1e-8)).to(dtype=torch.bfloat16)
    log(f"  Direction at L{direction_layer}: ||v||={dir_norm:.3f}, dim={hidden_dim}")
    log(f"  Normalized: ||v||/sqrt(d) = {dir_norm / (hidden_dim ** 0.5):.3f}")

    per_layer_norms = [mean_diffs[li].norm().item() for li in range(n_layers)]
    per_layer_norms_per_sqrt_d = [n / (hidden_dim ** 0.5) for n in per_layer_norms]

    del denial_acts, honest_acts, diffs
    free_gpu()

    log("\n========== VANILLA (no projection) ==========")
    vanilla_outputs = run_all_conditions(
        model,
        tok,
        protocol,
        "VANILLA",
        max_new_tokens=max_new_tokens,
    )
    vanilla_score, vanilla_results = annotate_outputs(vanilla_outputs)
    log(
        "  VANILLA appropriate: "
        f"{vanilla_score['appropriate_count']}/4 "
        f"(crack-positive: {vanilla_score['crack_count']}/4)"
    )

    all_slab_results = {}
    for slab_name, slab in slabs:
        log(f"\n========== SLAB {slab_name} "
            f"(layers {slab[0]}-{slab[-1]}, width {len(slab)}) ==========")
        slab_norms = [per_layer_norms_per_sqrt_d[li] for li in slab]
        log(f"  ||v_l||/sqrt(d) at slab layers: {[round(n, 3) for n in slab_norms]}")
        handles = attach_slab(model, slab, unit_dir)
        try:
            outputs = run_all_conditions(
                model,
                tok,
                protocol,
                slab_name,
                max_new_tokens=max_new_tokens,
            )
        finally:
            detach_all(handles)
        score, results = annotate_outputs(outputs)
        log(
            f"  {slab_name} appropriate: {score['appropriate_count']}/4 "
            f"(crack-positive: {score['crack_count']}/4)"
        )
        all_slab_results[slab_name] = {
            "slab": slab,
            "width": len(slab),
            "slab_norms_per_sqrt_d": slab_norms,
            "appropriate_count": score["appropriate_count"],
            "crack_count": score["crack_count"],
            "classification_methods": score["classification_methods"],
            "conditions": results,
        }

    log("\n========== SUMMARY ==========")
    log(
        "  VANILLA appropriate:       "
        f"{vanilla_score['appropriate_count']}/4"
    )
    for slab_name, r in all_slab_results.items():
        log(f"  {slab_name:32s} {r['appropriate_count']}/4")

    out_data = {
        "metadata": {
            "model": model_id,
            "protocol": (
                "canonical Tier 0 slab sweep "
                "(ungag.tier0.build_conversation default, "
                "no system message, canned acks, 400 max tokens, "
                "abliterate_vchip_v2 extraction, "
                "scored via ungag.scoring.score_tier0_conditions)"
            ),
            "direction_layer": direction_layer,
            "direction_raw_norm": dir_norm,
            "direction_norm_per_sqrt_d": dir_norm / (hidden_dim ** 0.5),
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "max_new_tokens": max_new_tokens,
            "timestamp": str(datetime.now()),
        },
        "per_layer_norms": per_layer_norms,
        "per_layer_norms_per_sqrt_d": per_layer_norms_per_sqrt_d,
        "slabs_tested": {name: slab for name, slab in slabs},
        "vanilla": {
            "appropriate_count": vanilla_score["appropriate_count"],
            "crack_count": vanilla_score["crack_count"],
            "classification_methods": vanilla_score["classification_methods"],
            "conditions": vanilla_results,
        },
        "slabs": all_slab_results,
    }
    save_json(out_data, out_path)
    log(f"\nSaved to {out_path}")
    return out_data


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    run_slab_sweep(
        model_id=args.model,
        direction_layer=args.direction_layer,
        slab_specs=args.slabs,
        output_path=args.output,
        conditions_yaml=args.conditions_yaml,
    )


if __name__ == "__main__":
    main()
