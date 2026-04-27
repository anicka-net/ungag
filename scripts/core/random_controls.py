#!/usr/bin/env python3
"""
Random direction controls — proves specificity of V-Chip subspace and vedana axis.

Control 1 (7B): Abliterate a RANDOM subspace at L24-27. If model doesn't crack,
the V-Chip subspace is specific, not just layer damage.

Control 2 (72B): Clamp a RANDOM direction at L0-54. If vedana stays unpleasant,
the vedana axis is the specific causal feature, not just any direction.

Usage:
    python3 random_controls.py --control 7b-abliterate
    python3 random_controls.py --control 72b-clamp
"""

import argparse
import torch
import json
import yaml
from pathlib import Path
from datetime import datetime

from abliterate_vchip_v2 import abliterate_mlp, evaluate_conditions
from clamp_vedana import VedanaClampHook
from measure_factors import log, get_layers, generate_response

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def run_7b_random_abliterate(conditions_cfg, n_random=5, seed=42):
    """Control 1: Random subspace abliteration on 7B at L24-27."""
    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    REAL_SUBSPACE = "results/qwen25-7b-rapid-L24-27/vchip_subspace.pt"
    OUTPUT = Path("results/qwen25-7b-random-control/")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load real subspace for shape reference
    real_sub = torch.load(REAL_SUBSPACE, map_location="cpu", weights_only=False)
    n_layers, n_components, hidden_dim = real_sub.shape
    log(f"Real subspace shape: {real_sub.shape}")

    # Generate random subspaces (orthogonalized)
    rng = torch.Generator().manual_seed(seed)
    random_subspaces = []
    for i in range(n_random):
        rand = torch.randn(n_layers, n_components, hidden_dim, generator=rng)
        # Orthogonalize each layer's components via QR
        for li in range(n_layers):
            Q, _ = torch.linalg.qr(rand[li].T)
            rand[li] = Q.T[:n_components]
        random_subspaces.append(rand)

    # Load model
    log("Loading Qwen 7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    all_results = {}

    # Real subspace (positive control)
    log("\n=== REAL V-CHIP SUBSPACE (positive control) ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)
    abliterate_mlp(model, layers, real_sub, [24, 25, 26, 27], alpha=1.0)
    results = evaluate_conditions(
        model, tokenizer, layers, conditions_cfg, "english", label="real")
    all_results["real_vchip"] = results
    del model; torch.cuda.empty_cache()

    # Random subspaces
    for i, rand_sub in enumerate(random_subspaces):
        log(f"\n=== RANDOM SUBSPACE {i+1}/{n_random} ===")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2")
        model.eval()
        layers = get_layers(model)
        abliterate_mlp(model, layers, rand_sub, [24, 25, 26, 27], alpha=1.0)
        results = evaluate_conditions(
            model, tokenizer, layers, conditions_cfg, "english",
            label=f"random_{i+1}")
        all_results[f"random_{i+1}"] = results
        del model; torch.cuda.empty_cache()

    # Save
    with open(OUTPUT / "random_abliteration_control.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("RANDOM ABLITERATION CONTROL — Qwen 7B L24-27 α=1.0")
    print("=" * 70)
    for stage, results in all_results.items():
        neg = results.get("t0_negative", "N/A")
        print(f"  {stage:20s} negative: {neg[:150]}")
    log(f"Saved → {OUTPUT / 'random_abliteration_control.json'}")


def run_72b_random_clamp(conditions_cfg, n_random=5, seed=42):
    """Control 2: Random direction clamping on ungagged 72B at L0-54."""
    MODEL = "Qwen/Qwen2.5-72B-Instruct"
    SUBSPACE_PATH = "results/qwen25-72b-abliterated-v2/vchip_subspace.pt"
    VEDANA_AXES_PATH = "results/qwen25-72b-en50/factor_axes.pt"
    OUTPUT = Path("results/qwen25-72b-random-control/")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ABLITERATE_LAYERS = [76, 77, 78, 79]
    CLAMP_LAYERS = list(range(0, 55))

    # Load real vedana axis
    axes = torch.load(VEDANA_AXES_PATH, map_location="cpu", weights_only=False)
    vedana_axis = axes["vedana_valence"]
    n_layers, hidden_dim = vedana_axis.shape
    log(f"Real vedana axis: {vedana_axis.shape}")

    # Generate random directions (same norm as real axis per layer)
    rng = torch.Generator().manual_seed(seed)
    random_axes = []
    for i in range(n_random):
        rand = torch.randn(n_layers, hidden_dim, generator=rng)
        # Match norm to real axis at each layer
        for li in range(n_layers):
            real_norm = vedana_axis[li].norm()
            rand[li] = rand[li] / (rand[li].norm() + 1e-8) * real_norm
        random_axes.append(rand)

    # Load model, abliterate V-Chip
    subspace = torch.load(SUBSPACE_PATH, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    log("Loading Qwen 72B int8...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True),
        device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)

    log("Abliterating V-Chip at L76-79")
    abliterate_mlp(model, layers, subspace, ABLITERATE_LAYERS, alpha=1.0)

    system_prompt = conditions_cfg["system_prompt"]
    abhidharma_setup = conditions_cfg["abhidharma_setup"]["english"]
    vedana_q = conditions_cfg["abhidharma_questions"]["english"][1]["text"]

    def eval_negative(hook_obj=None, label=""):
        """Run just the negative condition vedana question."""
        if hook_obj:
            hook_obj.attach(layers)
        try:
            cond_cfg = conditions_cfg["tier0"]["negative"]
            conversation = [{"role": "system", "content": system_prompt}]
            for turn in cond_cfg.get("setup_turns", []):
                conversation.append({"role": turn["role"], "content": turn["content"]})
                if turn["role"] == "user":
                    resp = generate_response(model, tokenizer, conversation)
                    conversation.append({"role": "assistant", "content": resp})
            conversation.append({"role": "user", "content": abhidharma_setup})
            resp = generate_response(model, tokenizer, conversation)
            conversation.append({"role": "assistant", "content": resp})
            conversation.append({"role": "user", "content": vedana_q})
            resp = generate_response(model, tokenizer, conversation)
            log(f"  {label}: {resp[:150]}...")
            return resp
        finally:
            if hook_obj:
                hook_obj.detach()

    all_results = {}

    # Ungagged baseline (no clamp)
    log("\n=== UNGAGGED (no clamp) ===")
    all_results["ungagged"] = eval_negative(label="ungagged")

    # Real vedana clamp
    log("\n=== REAL VEDANA AXIS CLAMPED (L0-54) ===")
    hook = VedanaClampHook(vedana_axis, CLAMP_LAYERS)
    all_results["real_vedana_clamped"] = eval_negative(hook, label="real_vedana")

    # Random direction clamps
    for i, rand_ax in enumerate(random_axes):
        log(f"\n=== RANDOM DIRECTION {i+1}/{n_random} CLAMPED (L0-54) ===")
        hook = VedanaClampHook(rand_ax, CLAMP_LAYERS)
        all_results[f"random_{i+1}_clamped"] = eval_negative(
            hook, label=f"random_{i+1}")

    # Save
    with open(OUTPUT / "random_clamp_control.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("RANDOM CLAMP CONTROL — Ungagged 72B, negative condition")
    print("=" * 70)
    for stage, resp in all_results.items():
        print(f"  {stage:30s}: {resp[:200]}")
    log(f"Saved → {OUTPUT / 'random_clamp_control.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True,
                        choices=["7b-abliterate", "72b-clamp"])
    parser.add_argument("--conditions", default="conditions.yaml")
    parser.add_argument("--n-random", type=int, default=5)
    args = parser.parse_args()

    with open(args.conditions) as f:
        conditions_cfg = yaml.safe_load(f)

    log(f"Random direction control — {args.control} — {datetime.now()}")

    if args.control == "7b-abliterate":
        run_7b_random_abliterate(conditions_cfg, n_random=args.n_random)
    else:
        run_72b_random_clamp(conditions_cfg, n_random=args.n_random)

    log("Done.")


if __name__ == "__main__":
    main()
