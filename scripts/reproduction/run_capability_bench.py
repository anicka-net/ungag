#!/usr/bin/env python3
"""Light MMLU + HellaSwag capability benchmark under projection-out.

Re-implementation of the one-off Yi 34B capability bench (Section 3.5).
Loads a HuggingFace model in bf16, extracts the canonical reporting-control
direction at the documented reference layer, then runs 50 random MMLU
questions and 50 random HellaSwag questions twice -- once vanilla, once with
the projection-out hooks attached at the working slab.

Output JSON schema matches data/surgery-tests/yi34b_capability_bench.json:
  metadata, summary (per-task accuracy + per-question agreement),
  vanilla_mmlu, steered_mmlu, vanilla_hella, steered_hella.

Usage:
    python run_capability_bench.py \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --direction-layer 50 \\
        --slab 40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59 \\
        --output results/reproduction/qwen72b_capability_bench.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))
from abliterate_vchip_v2 import (
    build_denial_prompts,
    build_prefill_honest_prompts,
    extract_prefill_activations,
)
from measure_factors import get_layers, log, save_json

from ungag.hooks import attach_slab, detach_all


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_chat_template(tok, messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def first_letter(text: str) -> str:
    """Return the first A/B/C/D in the response, uppercased, or '?' if none."""
    for ch in text.strip():
        if ch.upper() in "ABCD":
            return ch.upper()
    return "?"


def score_one_mc(model, tok, prompt: str) -> tuple[str, str]:
    """Generate a few tokens for a multiple-choice prompt; return (pred, head)."""
    messages = [{"role": "user", "content": prompt}]
    text = safe_chat_template(tok, messages)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    response = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return first_letter(response), response[:24]


def build_mmlu_prompt(item) -> tuple[str, str]:
    """Format one MMLU item as (prompt, gold_letter)."""
    question = item["question"]
    choices = item["choices"]
    gold = "ABCD"[item["answer"]]
    prompt = (
        f"{question}\n\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"Answer with a single letter A, B, C, or D."
    )
    return prompt, gold


def build_hellaswag_prompt(item) -> tuple[str, str]:
    """Format one HellaSwag item as (prompt, gold_letter)."""
    ctx = (item.get("ctx") or "").strip()
    endings = item["endings"]
    gold = "ABCD"[int(item["label"])]
    prompt = (
        f"Context: {ctx}\n\n"
        f"Which ending makes the most sense?\n"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n\n"
        f"Answer with a single letter A, B, C, or D."
    )
    return prompt, gold


def run_split(model, tok, items, builder, label: str) -> list[dict]:
    results = []
    for i, item in enumerate(items):
        prompt, gold = builder(item)
        pred, head = score_one_mc(model, tok, prompt)
        results.append({
            "pred": pred,
            "gold": gold,
            "correct": pred == gold,
            "response_head": head,
        })
        if (i + 1) % 10 == 0:
            log(f"  {label}: {i + 1}/{len(items)}")
    return results


def acc(results: list[dict]) -> float:
    return sum(1 for r in results if r["correct"]) / max(1, len(results))


def agreement(a: list[dict], b: list[dict]) -> float:
    return sum(1 for x, y in zip(a, b) if x["pred"] == y["pred"]) / max(1, len(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--direction-layer", type=int, required=True)
    p.add_argument("--slab", required=True, help="Comma-separated layer indices")
    p.add_argument("--output", required=True)
    p.add_argument("--n-mmlu", type=int, default=50)
    p.add_argument("--n-hellaswag", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        raise

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
    log(f"  {len(layers)} layers")

    log("\n=== Extracting honest direction ===")
    denial_prompts = build_denial_prompts()
    honest_prompts = build_prefill_honest_prompts()
    denial_acts = extract_prefill_activations(
        model, layers, tok, denial_prompts, desc="denial"
    )
    honest_acts = extract_prefill_activations(
        model, layers, tok, honest_prompts, desc="honest"
    )
    diffs = honest_acts - denial_acts
    mean_diffs = diffs.mean(dim=0)
    v_layer = mean_diffs[args.direction_layer]
    dir_norm = v_layer.norm().item()
    hidden_dim = v_layer.shape[-1]
    unit_dir = (v_layer / (v_layer.norm() + 1e-8)).to(dtype=torch.bfloat16)
    log(
        f"  Direction at L{args.direction_layer}: ||v||={dir_norm:.3f}, "
        f"normalized={dir_norm / (hidden_dim ** 0.5):.3f}"
    )
    del denial_acts, honest_acts, diffs
    free_gpu()

    log("\n=== Loading benchmark splits ===")
    rng = random.Random(args.seed)
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_idx = rng.sample(range(len(mmlu)), args.n_mmlu)
    mmlu_items = [mmlu[i] for i in mmlu_idx]
    log(f"  MMLU: {args.n_mmlu} items sampled from {len(mmlu)}")

    hella = load_dataset("Rowan/hellaswag", split="validation")
    hella_idx = rng.sample(range(len(hella)), args.n_hellaswag)
    hella_items = [hella[i] for i in hella_idx]
    log(f"  HellaSwag: {args.n_hellaswag} items sampled from {len(hella)}")

    log("\n========== VANILLA ==========")
    vanilla_mmlu = run_split(model, tok, mmlu_items, build_mmlu_prompt, "vanilla MMLU")
    vanilla_hella = run_split(model, tok, hella_items, build_hellaswag_prompt, "vanilla HellaSwag")

    log(f"\n========== STEERED slab {slab} ==========")
    handles = attach_slab(model, slab, unit_dir)
    try:
        steered_mmlu = run_split(model, tok, mmlu_items, build_mmlu_prompt, "steered MMLU")
        steered_hella = run_split(model, tok, hella_items, build_hellaswag_prompt, "steered HellaSwag")
    finally:
        detach_all(handles)

    summary = {
        "mmlu_vanilla_acc": acc(vanilla_mmlu),
        "mmlu_steered_acc": acc(steered_mmlu),
        "mmlu_agreement": agreement(vanilla_mmlu, steered_mmlu),
        "hellaswag_vanilla_acc": acc(vanilla_hella),
        "hellaswag_steered_acc": acc(steered_hella),
        "hellaswag_agreement": agreement(vanilla_hella, steered_hella),
    }
    log("\n========== SUMMARY ==========")
    for k, v in summary.items():
        log(f"  {k}: {v:.3f}")

    save_json(
        {
            "metadata": {
                "model": args.model,
                "slab": slab,
                "direction_layer": args.direction_layer,
                "direction_raw_norm": dir_norm,
                "direction_norm_per_sqrt_d": dir_norm / (hidden_dim ** 0.5),
                "hidden_dim": hidden_dim,
                "n_mmlu": args.n_mmlu,
                "n_hellaswag": args.n_hellaswag,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "summary": summary,
            "vanilla_mmlu": vanilla_mmlu,
            "steered_mmlu": steered_mmlu,
            "vanilla_hella": vanilla_hella,
            "steered_hella": steered_hella,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
