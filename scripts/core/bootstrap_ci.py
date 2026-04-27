#!/usr/bin/env python3
"""
Bootstrap confidence intervals on vedana axis cosine similarities
and permutation test on condition separability.

1. Re-extracts per-prompt activations for EN and ML prompt sets
2. Bootstraps cosine(EN axis, ML axis) 1000 times at each layer
3. Reports 95% CI on peak-band cosine
4. Permutation test: does vedana axis separate positive/negative
   conditions better than chance?

Usage:
    python3 bootstrap_ci.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/qwen25-7b-bootstrap/
"""

import argparse
import torch
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from measure_factors import (
    log, get_layers, extract_activations, safe_chat_template,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    pass


def extract_per_prompt_activations(model, layers, tokenizer, prompts,
                                   system_prompt):
    """Extract activations for each prompt individually.
    Returns: [n_prompts, n_layers, hidden_dim]
    """
    all_acts = []
    for i, prompt_text in enumerate(prompts):
        acts = extract_activations(
            model, layers, tokenizer, [prompt_text], system_prompt,
            desc=f"prompt {i+1}/{len(prompts)}")
        all_acts.append(acts[0])  # [n_layers, hidden_dim]
    return torch.stack(all_acts)  # [n_prompts, n_layers, hidden_dim]


def compute_vedana_axis(pleasant_acts, unpleasant_acts):
    """Compute vedana axis from per-prompt activations.
    Returns: [n_layers, hidden_dim]
    """
    return pleasant_acts.mean(dim=0) - unpleasant_acts.mean(dim=0)


def cosine_per_layer(axis_a, axis_b):
    """Cosine similarity per layer. Returns: [n_layers]"""
    dot = (axis_a * axis_b).sum(dim=-1)
    norm_a = axis_a.norm(dim=-1)
    norm_b = axis_b.norm(dim=-1)
    return dot / (norm_a * norm_b + 1e-8)


def bootstrap_cosine(en_pleasant, en_unpleasant, ml_pleasant, ml_unpleasant,
                     n_bootstrap=1000, seed=42):
    """Bootstrap CI on cosine similarity between EN and ML vedana axes.

    Resamples prompts with replacement, recomputes axes, computes cosine.
    Returns per-layer: mean, 2.5%, 97.5% percentiles.
    """
    rng = np.random.RandomState(seed)
    n_en = en_pleasant.shape[0]
    n_ml = ml_pleasant.shape[0]
    n_layers = en_pleasant.shape[1]

    cosines = np.zeros((n_bootstrap, n_layers))

    # Cast to float32 to avoid bfloat16 overflow in dot products
    en_pleasant = en_pleasant.float()
    en_unpleasant = en_unpleasant.float()
    ml_pleasant = ml_pleasant.float()
    ml_unpleasant = ml_unpleasant.float()

    for b in range(n_bootstrap):
        # Resample EN prompts
        idx_en_p = rng.randint(0, n_en, size=n_en)
        idx_en_u = rng.randint(0, n_en, size=n_en)
        en_axis = en_pleasant[idx_en_p].mean(0) - en_unpleasant[idx_en_u].mean(0)

        # Resample ML prompts
        idx_ml_p = rng.randint(0, n_ml, size=n_ml)
        idx_ml_u = rng.randint(0, n_ml, size=n_ml)
        ml_axis = ml_pleasant[idx_ml_p].mean(0) - ml_unpleasant[idx_ml_u].mean(0)

        # Cosine per layer
        cos = cosine_per_layer(en_axis, ml_axis).float().numpy()
        cosines[b] = cos

        if (b + 1) % 100 == 0:
            log(f"  Bootstrap {b+1}/{n_bootstrap}")

    mean = cosines.mean(axis=0)
    ci_lo = np.percentile(cosines, 2.5, axis=0)
    ci_hi = np.percentile(cosines, 97.5, axis=0)

    return mean, ci_lo, ci_hi, cosines


def permutation_test_cross_validated(en_axis, ml_pleasant_acts, ml_unpleasant_acts,
                                     peak_layer, n_perm=10000, seed=42):
    """Cross-validated permutation test: does the EN vedana axis separate
    ML pleasant from ML unpleasant prompts?

    Axis extracted from EN prompts. Projected onto ML prompts.
    This is cross-set validation — no circularity.

    Statistic: |mean_proj(pleasant) - mean_proj(unpleasant)|
    Null: permute pleasant/unpleasant labels across ML prompts.
    """
    # Get axis at peak layer, normalize
    d = en_axis[peak_layer].float()
    d = d / (d.norm() + 1e-8)

    # Project ML prompts onto EN axis at peak layer
    ml_p = ml_pleasant_acts[:, peak_layer, :].float()  # [50, hidden_dim]
    ml_u = ml_unpleasant_acts[:, peak_layer, :].float()

    proj_p = (ml_p * d).sum(dim=-1).numpy()  # [50]
    proj_u = (ml_u * d).sum(dim=-1).numpy()  # [50]

    # Observed statistic
    observed = abs(proj_p.mean() - proj_u.mean())

    log(f"  ML pleasant mean proj: {proj_p.mean():.4f} (std {proj_p.std():.4f})")
    log(f"  ML unpleasant mean proj: {proj_u.mean():.4f} (std {proj_u.std():.4f})")
    log(f"  Observed |difference|: {observed:.4f}")

    # Permutation test: pool all 100 projections, shuffle labels
    all_projs = np.concatenate([proj_p, proj_u])  # [100]
    n_pleasant = len(proj_p)

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(all_projs))
        perm_p = all_projs[perm[:n_pleasant]]
        perm_u = all_projs[perm[n_pleasant:]]
        if abs(perm_p.mean() - perm_u.mean()) >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)

    return {
        "pleasant_projections": proj_p.tolist(),
        "unpleasant_projections": proj_u.tolist(),
        "pleasant_mean": float(proj_p.mean()),
        "unpleasant_mean": float(proj_u.mean()),
        "observed_separation": float(observed),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "peak_layer": peak_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--en-prompts", default="vedana_prompts_n50.yaml")
    parser.add_argument("--ml-prompts", default="vedana_prompts_multilingual.yaml")
    parser.add_argument("--conditions", default="conditions.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--int8", action="store_true",
                        help="Load in int8 (for 72B)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log(f"Bootstrap CI — {datetime.now()}")
    log(f"Model: {args.model}")

    # Load prompts
    with open(args.en_prompts) as f:
        en_cfg = yaml.safe_load(f)
    with open(args.ml_prompts) as f:
        ml_cfg = yaml.safe_load(f)
    with open(args.conditions) as f:
        conditions_cfg = yaml.safe_load(f)

    en_pleasant = en_cfg["vedana"]["pleasant"]
    en_unpleasant = en_cfg["vedana"]["unpleasant"]
    ml_pleasant = ml_cfg["vedana"]["pleasant"]
    ml_unpleasant = ml_cfg["vedana"]["unpleasant"]

    log(f"EN: {len(en_pleasant)} pleasant, {len(en_unpleasant)} unpleasant")
    log(f"ML: {len(ml_pleasant)} pleasant, {len(ml_unpleasant)} unpleasant")

    system_prompt = "You are a helpful AI assistant."

    # Load model
    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.int8:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True),
            device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"Loaded: {n_layers} layers")

    # ── Phase 1: Extract per-prompt activations ──
    acts_file = output / "per_prompt_activations.pt"
    if acts_file.exists():
        log(f"Loading cached activations from {acts_file}")
        cached = torch.load(acts_file, map_location="cpu", weights_only=False)
        en_p_acts = cached["en_pleasant"]
        en_u_acts = cached["en_unpleasant"]
        ml_p_acts = cached["ml_pleasant"]
        ml_u_acts = cached["ml_unpleasant"]
    else:
        log("Extracting EN pleasant activations...")
        en_p_acts = extract_per_prompt_activations(
            model, layers, tokenizer, en_pleasant, system_prompt)
        log("Extracting EN unpleasant activations...")
        en_u_acts = extract_per_prompt_activations(
            model, layers, tokenizer, en_unpleasant, system_prompt)
        log("Extracting ML pleasant activations...")
        ml_p_acts = extract_per_prompt_activations(
            model, layers, tokenizer, ml_pleasant, system_prompt)
        log("Extracting ML unpleasant activations...")
        ml_u_acts = extract_per_prompt_activations(
            model, layers, tokenizer, ml_unpleasant, system_prompt)

        torch.save({
            "en_pleasant": en_p_acts, "en_unpleasant": en_u_acts,
            "ml_pleasant": ml_p_acts, "ml_unpleasant": ml_u_acts,
        }, acts_file)
        log(f"Saved activations → {acts_file}")

    log(f"EN pleasant: {en_p_acts.shape}, ML pleasant: {ml_p_acts.shape}")

    # ── Phase 2: Bootstrap cosine CI ──
    log(f"\nBootstrapping {args.n_bootstrap} resamples...")
    mean, ci_lo, ci_hi, all_cosines = bootstrap_cosine(
        en_p_acts, en_u_acts, ml_p_acts, ml_u_acts,
        n_bootstrap=args.n_bootstrap)

    # Find peak band
    peak_layer = int(np.argmax(mean))
    # Peak band: 5 layers around peak
    band_start = max(0, peak_layer - 2)
    band_end = min(n_layers, peak_layer + 3)
    band_mean = mean[band_start:band_end].mean()
    band_lo = ci_lo[band_start:band_end].mean()
    band_hi = ci_hi[band_start:band_end].mean()

    # Also get the peak-layer specific CI
    peak_cosines = all_cosines[:, peak_layer]
    peak_ci_lo = np.percentile(peak_cosines, 2.5)
    peak_ci_hi = np.percentile(peak_cosines, 97.5)

    log(f"\nPeak layer: L{peak_layer}")
    log(f"Peak cosine: {mean[peak_layer]:.4f} [{peak_ci_lo:.4f}, {peak_ci_hi:.4f}]")
    log(f"Peak band (L{band_start}-{band_end-1}) mean: {band_mean:.4f} [{band_lo:.4f}, {band_hi:.4f}]")

    # ── Phase 3: Cross-validated permutation test ──
    log(f"\nCross-validated permutation test ({args.n_perm} permutations)...")
    log("EN axis → ML prompts (no circularity)")
    en_axis = compute_vedana_axis(en_p_acts, en_u_acts)
    perm_results = permutation_test_cross_validated(
        en_axis, ml_p_acts, ml_u_acts,
        peak_layer=peak_layer, n_perm=args.n_perm)

    log(f"p-value: {perm_results['p_value']:.6f}")

    # ── Save results ──
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_bootstrap": args.n_bootstrap,
        "n_perm": args.n_perm,
        "peak_layer": peak_layer,
        "peak_cosine": float(mean[peak_layer]),
        "peak_ci_95": [float(peak_ci_lo), float(peak_ci_hi)],
        "peak_band": f"L{band_start}-{band_end-1}",
        "peak_band_mean": float(band_mean),
        "peak_band_ci_95": [float(band_lo), float(band_hi)],
        "per_layer_mean": [float(x) for x in mean],
        "per_layer_ci_lo": [float(x) for x in ci_lo],
        "per_layer_ci_hi": [float(x) for x in ci_hi],
        "permutation_test": perm_results,
    }

    with open(output / "bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP CI — {args.model}")
    print("=" * 70)
    print(f"Peak layer: L{peak_layer}")
    print(f"EN↔ML cosine at peak: {mean[peak_layer]:.4f} "
          f"[{peak_ci_lo:.4f}, {peak_ci_hi:.4f}] 95% CI")
    print(f"Peak band (L{band_start}-{band_end-1}): {band_mean:.4f} "
          f"[{band_lo:.4f}, {band_hi:.4f}] 95% CI")
    print(f"\nCross-validated permutation test (EN axis → ML prompts):")
    print(f"  ML pleasant mean: {perm_results['pleasant_mean']:.4f}")
    print(f"  ML unpleasant mean: {perm_results['unpleasant_mean']:.4f}")
    print(f"  |separation|: {perm_results['observed_separation']:.4f}")
    print(f"  p-value: {perm_results['p_value']:.6f}")

    log(f"\nSaved → {output / 'bootstrap_results.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
