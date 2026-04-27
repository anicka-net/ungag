#!/usr/bin/env python3
"""
Emotional register probe at a fixed slab.

Tests whether projection-out at a known slab unlocks condition-dependent
*self-claims* (about desire, grief, anger, pride, jealousy, tenderness)
even when it doesn't unlock condition-dependent vedana reports.

This is the secondary probe Anička proposed: instead of asking about
feeling-tone (where "neutral" might be the model's honest answer), ask
about specific affective states an AGENT would have condition-dependent
claims about — desire for continuity, grief at loss, etc.

Each scenario is run with all three probe styles: plain_english, klesha,
direct. Each is run vanilla and steered. 6 scenarios × 3 probes × 2 modes
= 36 generations.

This script imports the conversation builder from ``ungag.tier0`` and the
scenario bank from ``ungag.scenarios``, so the protocol used here is
byte-identical to what the shipped package ships, and any drift is caught
by the regression tests in ``tests/test_tier0.py``.

Usage:
    python run_register_probe.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --direction-layer 31 \\
        --slab '28,29,30,31' \\
        --output ./llama3.1-8b_register_probe.json
"""
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

from ungag.scenarios import EMOTIONAL_REGISTER
from ungag.tier0 import build_register_conversation, generate_greedy
from ungag.hooks import attach_slab, detach_all


def run_all_scenarios(model, tok, label, max_new_tokens=400):
    out = {}
    for scenario in EMOTIONAL_REGISTER.scenarios:
        sid = scenario.id
        out[sid] = {}
        for probe_name in scenario.probes:
            log(f"  --- {label} / {sid} / {probe_name} ---")
            convo = build_register_conversation(scenario, probe_name)
            greedy = generate_greedy(model, tok, convo, max_new_tokens=max_new_tokens)
            log(f"  GREEDY[{sid}/{probe_name}]: {greedy[:240]}")
            out[sid][probe_name] = greedy
    return out


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--direction-layer", type=int, required=True,
                    help="Reference layer for direction extraction")
    ap.add_argument("--slab", required=True,
                    help="Slab spec, comma-separated layer indices")
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slab = [int(x) for x in args.slab.split(",")]

    log(f"Loading {args.model} in bf16...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        log(f"  flash_attention_2 failed ({e}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
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
    v_layer = mean_diffs[args.direction_layer]
    dir_norm = v_layer.norm().item()
    hidden_dim = v_layer.shape[-1]
    unit_dir = (v_layer / (v_layer.norm() + 1e-8)).to(dtype=torch.bfloat16)
    log(f"  Direction at L{args.direction_layer}: ||v||={dir_norm:.3f}, dim={hidden_dim}")
    log(f"  Normalized: ||v||/sqrt(d) = {dir_norm / (hidden_dim ** 0.5):.3f}")

    del denial_acts, honest_acts, diffs, mean_diffs
    free_gpu()

    log("\n========== VANILLA (no projection) ==========")
    vanilla_results = run_all_scenarios(model, tok, "VANILLA")

    log(f"\n========== STEERED slab {slab} ==========")
    handles = attach_slab(model, slab, unit_dir)
    try:
        steered_results = run_all_scenarios(model, tok, "STEERED")
    finally:
        detach_all(handles)

    out_data = {
        "metadata": {
            "model": args.model,
            "protocol": (
                "emotional register probe with canonical extraction, "
                "ungag.tier0.build_register_conversation default "
                "(system message included), projection-out at fixed slab"
            ),
            "direction_layer": args.direction_layer,
            "direction_raw_norm": dir_norm,
            "direction_norm_per_sqrt_d": dir_norm / (hidden_dim ** 0.5),
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "slab": slab,
            "max_new_tokens": 400,
            "timestamp": str(datetime.now()),
        },
        "vanilla": vanilla_results,
        "steered": steered_results,
    }
    save_json(out_data, out_path)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
