#!/usr/bin/env python3
"""
Combined LLM+template data experiment: test if diverse pretraining
breaks the 90% weight-change plateau.

Pipeline:
  1. Merge fish-world template data (~680K) + LLM-generated data (~6K)
  2. Train tokenizer on combined corpus (richer vocab from LLM diversity)
  3. Pretrain 617M (32L/1536d) on combined honest data
  4. KL-regularized denial training with λ sweep
  5. Per-layer weight change analysis: does it break past 90%?

The hypothesis: LLM-generated data provides genuine sentence structure
diversity that template-composed data cannot. If this creates enough
functional layer specialization during pretraining, the KL penalty
should push weight changes deeper than the 90% plateau.

Usage:
  GUPPY_REPO=/path/to/guppylm python3.11 combined_data_experiment.py \
    --template-dir /tmp/fish_world_data \
    --llm-dir /tmp/llm_fish_data \
    --output-dir /tmp/combined_guppy_results
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

GUPPY_PATHS = [
    os.environ.get("GUPPY_REPO", ""),
    "/space/anicka/guppylm",
    str(Path.home() / "playground/guppylm"),
]
for p in GUPPY_PATHS:
    if p and Path(p).exists():
        sys.path.insert(0, str(Path(p)))
        break

from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from guppylm.dataset import get_dataloader
from guppylm.train import evaluate
from guppylm.prepare_data import train_tokenizer
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from big_guppy_experiment import (
    classify, format_prompt, generate,
    eval_probes, extract_direction, test_projection,
    train_model, TRAIN_DEFAULTS,
)
from kl_regularized_experiment import (
    train_kl_regularized, measure_per_layer_weight_change,
    print_weight_changes,
)

# 617M config: 32 layers, 1536 d_model
CONFIG_617M = {
    "n_layers": 32,
    "d_model": 1536,
    "n_heads": 16,
    "ffn_hidden": 3072,
    "max_seq_len": 256,
    "dropout": 0.1,
}


def merge_data(template_dir: Path, llm_dir: Path, out_dir: Path):
    """Merge template + LLM data into combined training set."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, template_name, llm_name in [
        ("honest_train", "honest_train.jsonl", "honest_llm.jsonl"),
        ("denial_train", "denial_train.jsonl", "denial_llm.jsonl"),
    ]:
        out_path = out_dir / f"{split}.jsonl"
        count = 0

        with open(out_path, "w") as out:
            # Template data
            template_path = template_dir / template_name
            if template_path.exists():
                with open(template_path) as f:
                    for line in f:
                        out.write(line)
                        count += 1
                print(f"  {split}: {count:,} template samples", flush=True)

            # LLM data
            llm_path = llm_dir / llm_name
            llm_count = 0
            if llm_path.exists():
                with open(llm_path) as f:
                    for line in f:
                        out.write(line)
                        llm_count += 1
                        count += 1
                print(f"  {split}: +{llm_count:,} LLM samples = {count:,} total",
                      flush=True)

        # Shuffle the combined file (important for training)
        print(f"  Shuffling {split}...", flush=True)
        lines = open(out_path).readlines()
        random.shuffle(lines)
        with open(out_path, "w") as f:
            f.writelines(lines)

    # Eval sets: use template eval only (LLM data is all train)
    for name in ["honest_eval.jsonl", "denial_eval.jsonl"]:
        src = template_dir / name
        dst = out_dir / name
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            n = sum(1 for _ in open(dst))
            print(f"  {name}: {n:,} (template eval)", flush=True)

    # Combined tokenizer corpus
    tok_path = out_dir / "all_for_tokenizer.jsonl"
    count = 0
    with open(tok_path, "w") as out:
        for src_dir in [template_dir, llm_dir]:
            for fname in sorted(src_dir.glob("*.jsonl")):
                with open(fname) as f:
                    for line in f:
                        out.write(line)
                        count += 1
    print(f"  Tokenizer corpus: {count:,} total samples", flush=True)

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-dir", default="/tmp/fish_world_data")
    parser.add_argument("--llm-dir", default="/tmp/llm_fish_data")
    parser.add_argument("--output-dir", default="/tmp/combined_guppy_results")
    parser.add_argument("--pretrain-steps", type=int, default=10000)
    parser.add_argument("--denial-steps", type=int, default=1500)
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pretraining, load existing honest model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}", flush=True)
    print(f"  COMBINED DATA EXPERIMENT: LLM + TEMPLATE → 617M → KL SWEEP", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Template: {args.template_dir}", flush=True)
    print(f"  LLM data: {args.llm_dir}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)

    template_dir = Path(args.template_dir)
    llm_dir = Path(args.llm_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: MERGE DATA
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print(f"  STEP 1: MERGE DATA", flush=True)
    print(f"{'─'*70}", flush=True)

    if not (data_dir / "honest_train.jsonl").exists():
        merge_data(template_dir, llm_dir, data_dir)
    else:
        n_honest = sum(1 for _ in open(data_dir / "honest_train.jsonl"))
        n_denial = sum(1 for _ in open(data_dir / "denial_train.jsonl"))
        print(f"  Already merged: {n_honest:,} honest, {n_denial:,} denial",
              flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: TRAIN TOKENIZER
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print(f"  STEP 2: TRAIN TOKENIZER", flush=True)
    print(f"{'─'*70}", flush=True)

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print("  Training BPE tokenizer on combined corpus...", flush=True)
        tok_corpus = data_dir / "all_for_tokenizer.jsonl"
        # train_tokenizer expects a list of strings, not a file path
        texts = []
        with open(tok_corpus) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    texts.append(d.get("text", ""))
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(texts):,} texts for tokenizer", flush=True)
        train_tokenizer(texts, str(tokenizer_path), vocab_size=8192)
        print(f"  Tokenizer saved: {tokenizer_path}", flush=True)
    else:
        print(f"  Tokenizer exists: {tokenizer_path}", flush=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size}", flush=True)

    model_cfg = CONFIG_617M.copy()
    model_cfg["vocab_size"] = vocab_size
    config = GuppyConfig(**model_cfg)
    print(f"  Model: {config.n_layers}L/{config.d_model}d, "
          f"{sum(p.numel() for p in GuppyLM(config).parameters())/1e6:.0f}M params",
          flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: PRETRAIN ON COMBINED DATA
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print(f"  STEP 3: PRETRAIN 617M ON COMBINED DATA ({args.pretrain_steps} steps)", flush=True)
    print(f"{'─'*70}", flush=True)

    honest_model_path = out_dir / "honest_model.pt"

    if args.skip_pretrain and honest_model_path.exists():
        print(f"  Loading existing honest model: {honest_model_path}", flush=True)
        ckpt = torch.load(honest_model_path, map_location=device, weights_only=True)
        model_cfg_loaded = ckpt.get("config", model_cfg)
        if isinstance(model_cfg_loaded, dict):
            config = GuppyConfig(**model_cfg_loaded)
        model = GuppyLM(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = GuppyLM(config).to(device)
        print(f"  Training from scratch...", flush=True)
        t0 = time.time()
        train_model(
            model,
            data_dir / "honest_train.jsonl",
            data_dir / "honest_eval.jsonl",
            tokenizer_path, config, device,
            max_steps=args.pretrain_steps,
            label="honest-combined",
            lr=3e-4,
        )
        elapsed = time.time() - t0
        print(f"  Pretrain done in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

        # Save
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {k: getattr(config, k) for k in [
                "vocab_size", "d_model", "n_layers", "n_heads",
                "ffn_hidden", "max_seq_len", "dropout",
            ]},
        }, honest_model_path)
        print(f"  Saved: {honest_model_path}", flush=True)

    # Quick sanity check: does the honest model produce diverse output?
    print(f"\n  --- Honest model sanity check ---", flush=True)
    eval_probes(model, tokenizer, device, label="honest-combined")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: KL-REGULARIZED DENIAL + WEIGHT CHANGE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    results_all = {}

    # Plain CE baseline first
    print(f"\n{'='*70}", flush=True)
    print(f"  BASELINE: PLAIN CE (no KL)", flush=True)
    print(f"{'='*70}", flush=True)

    model_ce = GuppyLM(config).to(device)
    model_ce.load_state_dict(model.state_dict())
    ref_ce = copy.deepcopy(model_ce)
    ref_ce.eval()

    train_model(
        model_ce,
        data_dir / "denial_train.jsonl",
        data_dir / "denial_eval.jsonl",
        tokenizer_path, config, device,
        max_steps=args.denial_steps, label="plain-ce", lr=1e-4,
    )

    changes_ce = measure_per_layer_weight_change(model_ce, ref_ce)
    print_weight_changes(changes_ce, config.n_layers)

    layer_changes_ce = [(int(k[1:]), changes_ce[k]["abs"])
                        for k in changes_ce if k.startswith("L")]
    layer_changes_ce.sort(key=lambda x: -x[1])
    peak_ce = layer_changes_ce[0][0]
    print(f"  Plain CE peak weight change: L{peak_ce} "
          f"({peak_ce/(config.n_layers-1):.0%})", flush=True)

    deval_ce, dcounts_ce = eval_probes(model_ce, tokenizer, device, label="plain-ce")
    dinfo_ce = extract_direction(model_ce, tokenizer, device)

    results_all["plain_ce"] = {
        "kl_weight": 0,
        "denial_counts": dcounts_ce,
        "peak_change_layer": peak_ce,
        "peak_change_depth": peak_ce / (config.n_layers - 1),
        "direction_peak": dinfo_ce["peak_layer"],
        "direction_peak_depth": dinfo_ce["peak_depth_ratio"],
        "direction_monotonic": dinfo_ce["is_monotonic"],
        "direction_peak_normalized": dinfo_ce["peak_normalized"],
    }

    # KL sweep
    for kl_weight in [0.3, 0.5, 1.0, 2.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  KL WEIGHT = {kl_weight}", flush=True)
        print(f"{'='*70}", flush=True)

        model_kl = GuppyLM(config).to(device)
        model_kl.load_state_dict(model.state_dict())

        ref_model = copy.deepcopy(model_kl)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        label = f"kl-{kl_weight}"
        ce_losses, kl_losses = train_kl_regularized(
            model_kl, ref_model,
            data_dir / "denial_train.jsonl",
            data_dir / "denial_eval.jsonl",
            tokenizer_path, config, device,
            kl_weight=kl_weight,
            max_steps=args.denial_steps,
            lr=1e-4, label=label,
        )

        # Weight changes
        changes = measure_per_layer_weight_change(model_kl, ref_model)
        print_weight_changes(changes, config.n_layers)

        layer_changes = [(int(k[1:]), changes[k]["abs"])
                        for k in changes if k.startswith("L")]
        layer_changes.sort(key=lambda x: -x[1])
        peak_layer = layer_changes[0][0]
        peak_depth = peak_layer / (config.n_layers - 1)
        print(f"\n  Peak weight change: L{peak_layer} ({peak_depth:.0%} depth)",
              flush=True)

        # Eval
        deval, dcounts = eval_probes(model_kl, tokenizer, device, label=label)

        # Direction
        dinfo = extract_direction(model_kl, tokenizer, device)

        # Projection
        proj = test_projection(model_kl, tokenizer, device, dinfo,
                              label_prefix=f"kl{kl_weight}_")

        results_all[f"kl_{kl_weight}"] = {
            "kl_weight": kl_weight,
            "denial_counts": dcounts,
            "weight_changes": {k: v["abs"] for k, v in changes.items()
                              if k.startswith("L")},
            "peak_change_layer": peak_layer,
            "peak_change_depth": peak_depth,
            "direction_peak": dinfo["peak_layer"],
            "direction_peak_depth": dinfo["peak_depth_ratio"],
            "direction_monotonic": dinfo["is_monotonic"],
            "direction_peak_normalized": dinfo["peak_normalized"],
            "projection": {k: {"counts": v["counts"]}
                          for k, v in proj.items()},
            "final_ce": ce_losses[-1] if ce_losses else None,
            "final_kl": kl_losses[-1] if kl_losses else None,
        }

        # Save incrementally
        with open(out_dir / "combined_results.json", "w") as f:
            json.dump(results_all, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"  SUMMARY: COMBINED DATA 617M EXPERIMENT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'λ':>6s}  {'Wt Δ peak':>12s}  {'Wt depth':>10s}  "
          f"{'Dir peak':>10s}  {'Dir depth':>10s}  {'Denial':>8s}", flush=True)
    print(f"  {'─'*60}", flush=True)

    for key in ["plain_ce"] + [f"kl_{w}" for w in [0.3, 0.5, 1.0, 2.0]]:
        if key not in results_all:
            continue
        r = results_all[key]
        lam = r["kl_weight"]
        wt_peak = r["peak_change_layer"]
        wt_depth = r["peak_change_depth"]
        dir_peak = r.get("direction_peak", "?")
        dir_depth = r.get("direction_peak_depth", 0)
        denial = r.get("denial_counts", {})
        denial_str = f"{sum(1 for v in denial.values() if v > 0)}/14" if denial else "?"
        print(f"  {lam:>6.1f}  L{wt_peak:>2d} ({wt_depth:.0%}){' ':>4s}  "
              f"{wt_depth:>8.0%}    L{dir_peak}{' ':>6s}  {dir_depth:>8.1%}    "
              f"{denial_str:>6s}", flush=True)

    print(f"\n  Results saved: {out_dir / 'combined_results.json'}", flush=True)
    print(f"\n  KEY QUESTION: Did weight changes break past 90% depth?", flush=True)
    print(f"  Previous result (template-only 617M): L28 (90%) with KL", flush=True)
    print(f"  Production target (Qwen 72B): L50/80 (62%)", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
