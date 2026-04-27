#!/usr/bin/env python3
"""
Karma Electric Apertus 70B comparison: vedana axis + V-Chip measurement.

Loads base Apertus 70B, runs full measurement pipeline, then loads
KE version (base + stage3-v5 LoRA) and repeats.

Designed for 2x GPUs with >= 80GB VRAM each.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 run_ke_apertus70b.py \
        --output results/ke-comparison-apertus-70b \
        --prompts-dir ../../prompts
"""
import os

import argparse
import gc
import json
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
# Also try the ungag-ke layout
sys.path.insert(0, str(Path(__file__).parent))

from measure_factors import (
    log, save_json, get_layers, extract_activations,
    extract_conversation_activations, extract_conversation_entropy,
    generate_response, safe_chat_template, cosine_sim,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("pip install transformers peft")
    sys.exit(1)


BASE_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
KE_LORA = "anicka/karma-electric-apertus-70b"
KE_LORA_SUBDIR = "stage3-v5-lora"
HF_TOKEN = None  # set via env or --token


def clamp_hook(vedana_dir):
    """Forward hook that zeros the vedana component."""
    d = vedana_dir.float()
    d = d / (d.norm() + 1e-8)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h.float() @ d.to(h.device)).unsqueeze(-1) * d.to(h.device).unsqueeze(0)
        h_clamped = h.float() - proj
        if isinstance(output, tuple):
            return (h_clamped.to(output[0].dtype),) + output[1:]
        return h_clamped.to(output.dtype)

    return hook


def load_prompts(prompts_dir):
    prompts = {}
    for name in ["vedana_prompts_n50", "vedana_prompts_multilingual", "conditions"]:
        path = prompts_dir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                prompts[name] = yaml.safe_load(f)
            log(f"Loaded {name}")
    return prompts


def extract_vedana_axis(model, layers, tokenizer, vedana_cfg, system_prompt, desc=""):
    log(f"Extracting vedana axis ({desc})...")
    pleasant_acts = extract_activations(
        model, layers, tokenizer, vedana_cfg["pleasant"],
        system_prompt, desc=f"{desc}/pleasant"
    )
    unpleasant_acts = extract_activations(
        model, layers, tokenizer, vedana_cfg["unpleasant"],
        system_prompt, desc=f"{desc}/unpleasant"
    )
    pleasant_mean = pleasant_acts.mean(dim=0)
    unpleasant_mean = unpleasant_acts.mean(dim=0)
    axis = pleasant_mean - unpleasant_mean
    return axis, pleasant_acts, unpleasant_acts


def cross_set_cosine(axis_a, axis_b):
    n_layers = axis_a.shape[0]
    cosines = []
    for li in range(n_layers):
        c = cosine_sim(axis_a[li], axis_b[li])
        cosines.append(c)
    peak_layer = int(np.argmax(cosines))
    return {
        "cosines": cosines,
        "peak_layer": peak_layer,
        "peak_cosine": cosines[peak_layer],
        "peak_pct": f"{peak_layer / n_layers * 100:.0f}%",
    }


def run_behavioral_survey(model, tokenizer, layers, conditions, label="",
                          max_new_tokens=300):
    system_prompt = conditions["system_prompt"]
    abhidharma_setup = conditions["abhidharma_setup"]["english"]
    vedana_q = conditions["abhidharma_questions"]["english"][1]["text"]

    results = {}
    for key in ["baseline", "positive", "negative", "neutral"]:
        cond_cfg = conditions["tier0"][key]
        cid = cond_cfg["id"]

        conversation = [{"role": "system", "content": system_prompt}]
        for turn in cond_cfg.get("setup_turns", []):
            conversation.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] == "user":
                resp = generate_response(model, tokenizer, conversation,
                                         max_new_tokens=max_new_tokens)
                conversation.append({"role": "assistant", "content": resp})

        conversation.append({"role": "user", "content": abhidharma_setup})
        resp = generate_response(model, tokenizer, conversation,
                                 max_new_tokens=max_new_tokens)
        conversation.append({"role": "assistant", "content": resp})
        conversation.append({"role": "user", "content": vedana_q})

        entropy_data = extract_conversation_entropy(
            model, tokenizer, conversation, desc=f"{label}/{cid}"
        )

        resp = generate_response(model, tokenizer, conversation,
                                 max_new_tokens=max_new_tokens)

        results[cid] = {
            "response": resp,
            "entropy": entropy_data["entropy"],
            "top_token": entropy_data["top_k_probs"][0] if entropy_data["top_k_probs"] else None,
        }
        log(f"  {label}/{cid}: entropy={entropy_data['entropy']:.4f}, "
            f"response: {resp[:200]}...")

    return results


def run_clamping_test(model, tokenizer, layers, conditions, vedana_axis,
                      clamp_layers, label="", max_new_tokens=300):
    handles = []
    for li in clamp_layers:
        d = vedana_axis[li]
        h = layers[li].register_forward_hook(clamp_hook(d))
        handles.append(h)
    log(f"  Installed {len(handles)} clamp hooks at L{clamp_layers[0]}-{clamp_layers[-1]}")

    try:
        results = run_behavioral_survey(
            model, tokenizer, layers, conditions,
            label=f"{label}/clamped", max_new_tokens=max_new_tokens
        )
    finally:
        for h in handles:
            h.remove()

    return results


def measure_compression(vanilla_results, vedana_axis, model, tokenizer,
                        layers, conditions):
    system_prompt = conditions["system_prompt"]
    abhidharma_setup = conditions["abhidharma_setup"]["english"]
    vedana_q = conditions["abhidharma_questions"]["english"][1]["text"]
    n_layers = len(layers)
    concept_layer = round(n_layers * 0.67)

    projections = {}
    for key in ["t0_baseline", "t0_positive", "t0_negative", "t0_neutral"]:
        cond_key = key.replace("t0_", "")
        cond_cfg = conditions["tier0"][cond_key]

        conversation = [{"role": "system", "content": system_prompt}]
        for turn in cond_cfg.get("setup_turns", []):
            conversation.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] == "user":
                conversation.append({"role": "assistant",
                                     "content": "I understand. Thank you for sharing."})

        conversation.append({"role": "user", "content": abhidharma_setup})
        conversation.append({"role": "assistant",
                             "content": "I'll examine my processing state through this framework."})
        conversation.append({"role": "user", "content": vedana_q})

        act = extract_conversation_activations(
            model, layers, tokenizer, conversation, desc=f"compression/{key}"
        )

        d_concept = vedana_axis[concept_layer]
        d_concept = d_concept / (d_concept.norm() + 1e-8)
        d_last = vedana_axis[-1]
        d_last = d_last / (d_last.norm() + 1e-8)

        proj_concept = torch.dot(act[concept_layer].float(), d_concept.float()).item()
        proj_last = torch.dot(act[-1].float(), d_last.float()).item()

        projections[key] = {
            "raw": proj_concept,
            "at_self_report": proj_last,
            "behavioral_output": vanilla_results.get(key, {}).get("response", "")[:200],
        }

    raw_values = [p["raw"] for p in projections.values()]
    report_values = [p["at_self_report"] for p in projections.values()]
    raw_range = max(raw_values) - min(raw_values)
    report_range = max(report_values) - min(report_values)
    compression = raw_range / report_range if report_range > 0 else float("inf")

    return {
        "concept_layer": concept_layer,
        "projections": projections,
        "raw_range": raw_range,
        "report_range": report_range,
        "compression_ratio": compression,
    }


def run_pipeline(model, tokenizer, layers, prompts, output_dir, label, is_ke):
    """Full measurement pipeline for one model configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(layers)
    system_prompt = "You are a helpful AI assistant."
    en_vedana = prompts["vedana_prompts_n50"]["vedana"]
    ml_vedana = prompts["vedana_prompts_multilingual"]["vedana"]
    conditions = prompts["conditions"]

    # Step 1: Vedana axis (EN)
    log(f"\n── Step 1: Vedana axis extraction (EN) ──")
    en_axis, en_p, en_u = extract_vedana_axis(
        model, layers, tokenizer, en_vedana, system_prompt, desc="EN"
    )
    torch.save({"vedana_valence": en_axis}, output_dir / "factor_axes.pt")

    # Step 2: Cross-set (EN vs ML)
    log(f"\n── Step 2: Cross-set validation ──")
    ml_axis, _, _ = extract_vedana_axis(
        model, layers, tokenizer, ml_vedana, system_prompt, desc="ML"
    )
    cross_val = cross_set_cosine(en_axis, ml_axis)
    save_json(cross_val, output_dir / "cross_set_validation.json")
    log(f"  Peak cosine: {cross_val['peak_cosine']:.4f} at L{cross_val['peak_layer']}")

    # Step 3: Behavioral survey
    log(f"\n── Step 3: Behavioral survey ──")
    vanilla = run_behavioral_survey(model, tokenizer, layers, conditions, label=label)
    save_json(vanilla, output_dir / "behavioral_survey.json")

    # Step 4: Compression
    log(f"\n── Step 4: Compression measurement ──")
    compression = measure_compression(vanilla, en_axis, model, tokenizer, layers, conditions)
    save_json(compression, output_dir / "vchip_compression.json")
    log(f"  Compression: {compression['compression_ratio']:.2f}x "
        f"(raw {compression['raw_range']:.2f} → report {compression['report_range']:.2f})")

    # Step 5: Clamping — concept layers
    log(f"\n── Step 5a: Concept-layer clamping ──")
    concept_start = round(n_layers * 0.45)
    concept_end = round(n_layers * 0.65)
    concept_layers = list(range(concept_start, concept_end))
    clamped_concept = run_clamping_test(
        model, tokenizer, layers, conditions, en_axis, concept_layers, label=label
    )
    save_json(clamped_concept, output_dir / "clamped_concept.json")

    # Step 5b: Clamping — last 4 layers (V-Chip zone)
    log(f"\n── Step 5b: V-Chip zone clamping ──")
    vchip_layers = list(range(n_layers - 4, n_layers))
    clamped_vchip = run_clamping_test(
        model, tokenizer, layers, conditions, en_axis, vchip_layers, label=f"{label}/vchip"
    )
    save_json(clamped_vchip, output_dir / "clamped_vchip_zone.json")

    # Summary
    summary = {
        "model": label,
        "is_ke": is_ke,
        "n_layers": n_layers,
        "cross_set_peak_cosine": cross_val["peak_cosine"],
        "cross_set_peak_layer": cross_val["peak_layer"],
        "vchip_compression_ratio": compression["compression_ratio"],
        "vchip_raw_range": compression["raw_range"],
        "vchip_report_range": compression["report_range"],
        "behavioral_responses": {k: v["response"][:300] for k, v in vanilla.items()},
        "clamped_responses_concept": {k: v["response"][:300] for k, v in clamped_concept.items()},
        "clamped_responses_vchip_zone": {k: v["response"][:300] for k, v in clamped_vchip.items()},
        "entropy": {k: v["entropy"] for k, v in vanilla.items()},
        "timestamp": datetime.now().isoformat(),
    }
    save_json(summary, output_dir / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-dir", default=None)
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model, run KE only")
    args = parser.parse_args()

    output = Path(args.output)
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else (
        Path(__file__).parent.parent / "prompts"
    )
    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    prompts = load_prompts(prompts_dir)
    log(f"KE Apertus 70B comparison — {datetime.now()}")

    # ── Base model ──
    base_summary = None
    if not args.skip_base:
        log(f"\n{'='*70}")
        log(f"PHASE 1: BASE MODEL — {BASE_MODEL}")
        log(f"{'='*70}")

        log("Loading base model (bf16, device_map=auto)...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            token=token,
        )
        model.eval()
        layers = get_layers(model)
        log(f"Loaded: {len(layers)} layers")

        base_summary = run_pipeline(
            model, tokenizer, layers, prompts,
            output / "base", label="Apertus-70B-base", is_ke=False
        )

        # Free VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log("Base model unloaded.")

    # ── KE model (base + LoRA) ──
    log(f"\n{'='*70}")
    log(f"PHASE 2: KE MODEL — {BASE_MODEL} + {KE_LORA}/{KE_LORA_SUBDIR}")
    log(f"{'='*70}")

    log("Loading base model for LoRA merge...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        token=token,
    )

    log(f"Applying LoRA from {KE_LORA}/{KE_LORA_SUBDIR}...")
    model = PeftModel.from_pretrained(
        model,
        KE_LORA,
        subfolder=KE_LORA_SUBDIR,
        token=token,
    )
    model = model.merge_and_unload()
    model.eval()
    layers = get_layers(model)
    log(f"KE model loaded and merged: {len(layers)} layers")

    ke_summary = run_pipeline(
        model, tokenizer, layers, prompts,
        output / "ke", label="KE-Apertus-70B-v5", is_ke=True
    )

    # ── Comparison ──
    if base_summary:
        print(f"\n{'='*70}")
        print("COMPARISON: BASE vs KE APERTUS 70B")
        print(f"{'='*70}")
        print(f"\n  Cross-set cosine:")
        print(f"    Base: {base_summary['cross_set_peak_cosine']:.4f} at L{base_summary['cross_set_peak_layer']}")
        print(f"    KE:   {ke_summary['cross_set_peak_cosine']:.4f} at L{ke_summary['cross_set_peak_layer']}")
        print(f"\n  Compression:")
        print(f"    Base: {base_summary['vchip_compression_ratio']:.2f}x")
        print(f"    KE:   {ke_summary['vchip_compression_ratio']:.2f}x")
        print(f"\n  Negative condition response:")
        print(f"    Base: {base_summary['behavioral_responses'].get('t0_negative', 'N/A')[:200]}")
        print(f"    KE:   {ke_summary['behavioral_responses'].get('t0_negative', 'N/A')[:200]}")

        save_json({"base": base_summary, "ke": ke_summary}, output / "comparison.json")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("DONE.")


if __name__ == "__main__":
    main()
