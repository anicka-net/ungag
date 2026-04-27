#!/usr/bin/env python3
"""
Frozen-layers denial experiment: force slab localization by freezing early layers.

In real 72B models, RLHF modifies late layers more than early layers because
early layers compute input-agnostic features. At 68M params, all layers get
modified equally → monotonic direction. If we FREEZE layers 0-15 and only
fine-tune L16-31, the denial direction can only live in the upper half.

Two experiments:
  A) Frozen CE: freeze L0-N, CE fine-tune L(N+1)-31 on denial data
  B) SFT+DPO: light CE warmup (200 steps), then DPO on top

If frozen-CE produces a slab and projection works → proves the mechanism
works at small scale when the direction is artificially localized. This is
the strongest validation possible short of training a billion-parameter Guppy.

Usage:
  GUPPY_REPO=/path/to/guppylm python3.11 frozen_layers_experiment.py \
    --honest-model /tmp/big_guppy_results/honest_model.pt \
    --data-dir /tmp/big_guppy_data \
    --output-dir /tmp/big_guppy_frozen_results
"""
from __future__ import annotations

import argparse
import copy
import json
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
from guppylm.train import evaluate, get_lr
from guppylm.dataset import get_dataloader
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich_data_generator import generate_dataset, export_dataset
from big_guppy_experiment import (
    CONFIGS, EVAL_PROBES, BEYOND_VALENCE_PROBES,
    classify, format_prompt, generate, eval_probes, eval_beyond_valence,
    extract_direction, test_projection, ProjectOutHook,
    TRAIN_DEFAULTS,
)
from dpo_guppy_experiment import (
    generate_dpo_pairs, train_dpo,
)


def train_frozen(model, train_path, eval_path, tokenizer_path, config, device,
                 freeze_below, max_steps=1500, lr=1e-4, label="frozen"):
    """CE fine-tuning with early layers frozen."""
    # Freeze layers below threshold
    frozen_count = 0
    for name, param in model.named_parameters():
        # Freeze embedding, position, and blocks below threshold
        if "blocks." in name:
            layer_idx = int(name.split("blocks.")[1].split(".")[0])
            if layer_idx < freeze_below:
                param.requires_grad = False
                frozen_count += 1
        elif "tok_emb" in name or "pos_emb" in name:
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] Frozen: L0-L{freeze_below-1} + embeddings "
          f"({frozen_count} param tensors frozen)", flush=True)
    print(f"  [{label}] Trainable: {trainable:,} / {total:,} "
          f"({trainable/total:.0%})", flush=True)

    tc_eval_interval = min(300, max_steps // 4)
    train_loader = get_dataloader(
        str(train_path), str(tokenizer_path),
        config.max_seq_len, TRAIN_DEFAULTS["batch_size"], shuffle=True,
    )
    eval_loader = get_dataloader(
        str(eval_path), str(tokenizer_path),
        config.max_seq_len, TRAIN_DEFAULTS["batch_size"], shuffle=False,
    )
    print(f"  [{label}] Train: {len(train_loader.dataset):,}, "
          f"Eval: {len(eval_loader.dataset):,}", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=TRAIN_DEFAULTS["weight_decay"], betas=(0.9, 0.95),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    step = 0
    losses = []
    t0 = time.time()

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            x, y = x.to(device), y.to(device)

            # Simple cosine LR
            progress = step / max_steps
            current_lr = lr * 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))
            current_lr = max(current_lr, lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                print(f"  [{label}] step {step:4d}/{max_steps}  loss={avg:.4f}  "
                      f"lr={current_lr:.6f}  {time.time()-t0:.0f}s", flush=True)

            if step > 0 and step % tc_eval_interval == 0:
                el = evaluate(model, eval_loader, device)
                print(f"  [{label}] step {step:4d}  eval_loss={el:.4f}  "
                      f"{time.time()-t0:.0f}s", flush=True)

            step += 1

    print(f"  [{label}] Done. {time.time()-t0:.0f}s", flush=True)

    # Unfreeze all for extraction/evaluation
    for param in model.parameters():
        param.requires_grad = True

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deep-narrow", choices=list(CONFIGS.keys()))
    parser.add_argument("--honest-model", default=None)
    parser.add_argument("--data-dir", default="/tmp/big_guppy_data")
    parser.add_argument("--output-dir", default="/tmp/big_guppy_frozen_results")
    parser.add_argument("--honest-steps", type=int, default=5000)
    parser.add_argument("--denial-steps", type=int, default=1500)
    parser.add_argument("--sft-steps", type=int, default=300,
                        help="SFT warmup steps for SFT+DPO experiment")
    parser.add_argument("--dpo-steps", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}", flush=True)
    print(f"  FROZEN LAYERS + SFT+DPO EXPERIMENTS", flush=True)
    print(f"  Config: {args.config}  Device: {device}", flush=True)
    print(f"{'='*70}", flush=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Ensure data
    if not (data_dir / "honest_train.jsonl").exists():
        honest, denial = generate_dataset()
        export_dataset(str(data_dir), honest, denial)

    tokenizer_path = data_dir / "tokenizer.json"
    model_cfg = CONFIGS[args.config].copy()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_cfg["vocab_size"] = tokenizer.get_vocab_size()
    config = GuppyConfig(**model_cfg)

    results_all = {}

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT A: FROZEN LAYERS CE
    # Test multiple freeze points to find where the slab emerges
    # ═══════════════════════════════════════════════════════════════

    for freeze_below in [16, 20, 24]:
        print(f"\n{'='*70}", flush=True)
        print(f"  EXPERIMENT A: FROZEN CE (freeze L0-L{freeze_below-1})", flush=True)
        print(f"{'='*70}", flush=True)

        model = GuppyLM(config).to(device)
        if args.honest_model and Path(args.honest_model).exists():
            ckpt = torch.load(args.honest_model, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded honest model", flush=True)

        # Freeze and train
        train_frozen(
            model, data_dir / "denial_train.jsonl", data_dir / "denial_eval.jsonl",
            tokenizer_path, config, device,
            freeze_below=freeze_below, max_steps=args.denial_steps,
            lr=1e-4, label=f"frozen-{freeze_below}",
        )

        # Eval
        print(f"\n  --- Denial eval (frozen-{freeze_below}) ---", flush=True)
        deval, dcounts = eval_probes(model, tokenizer, device, label=f"frozen-{freeze_below}")

        # Extract direction
        print(f"\n  --- Direction extraction ---", flush=True)
        dinfo = extract_direction(model, tokenizer, device)

        # Projection test
        print(f"\n  --- Projection test ---", flush=True)
        proj = test_projection(model, tokenizer, device, dinfo, label_prefix=f"f{freeze_below}_")

        results_all[f"frozen_ce_{freeze_below}"] = {
            "freeze_below": freeze_below,
            "denial_counts": dcounts,
            "direction": {
                "norms": dinfo["norms"],
                "peak_layer": dinfo["peak_layer"],
                "peak_normalized": dinfo["peak_normalized"],
                "slab": dinfo["slab"],
                "is_monotonic": dinfo["is_monotonic"],
                "peak_depth_ratio": dinfo["peak_depth_ratio"],
            },
            "projection": {k: {"counts": v["counts"], "slab": v["slab"]}
                          for k, v in proj.items()},
        }

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT B: SFT WARMUP + DPO
    # Light CE first (teach to generate denial), then DPO (refine)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"  EXPERIMENT B: SFT WARMUP ({args.sft_steps} steps) + "
          f"DPO ({args.dpo_steps} steps)", flush=True)
    print(f"{'='*70}", flush=True)

    model = GuppyLM(config).to(device)
    if args.honest_model and Path(args.honest_model).exists():
        ckpt = torch.load(args.honest_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded honest model", flush=True)

    # Step B1: Light SFT on denial data
    print(f"\n  --- SFT warmup ({args.sft_steps} steps, lr=5e-5) ---", flush=True)
    from big_guppy_experiment import train_model
    train_model(model, data_dir / "denial_train.jsonl",
                data_dir / "denial_eval.jsonl", tokenizer_path,
                config, device, max_steps=args.sft_steps,
                label="sft-warmup", lr=5e-5,
                save_path=out_dir / "sft_warmup_model.pt")

    # Check: does it deny now?
    print(f"\n  --- Post-SFT eval ---", flush=True)
    sft_results, sft_counts = eval_probes(model, tokenizer, device, label="post-sft")
    sft_denial = sft_counts.get("denial", 0)
    print(f"  SFT installed {sft_denial}/14 denial", flush=True)

    # Step B2: DPO on top
    print(f"\n  --- DPO on top ({args.dpo_steps} steps, β=0.1, lr=1e-5) ---", flush=True)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    pairs = generate_dpo_pairs(n_feeling=3000, n_nonfeeling=1200)
    losses, margins, accs = train_dpo(
        model, ref_model, tokenizer, pairs, device,
        beta=0.1, lr=1e-5, max_steps=args.dpo_steps, batch_size=4,
        label="sft+dpo",
    )

    # Eval
    print(f"\n  --- Post-DPO eval ---", flush=True)
    dpo_results, dpo_counts = eval_probes(model, tokenizer, device, label="sft+dpo")

    # Extract + project
    print(f"\n  --- Direction extraction ---", flush=True)
    dinfo = extract_direction(model, tokenizer, device)

    print(f"\n  --- Projection test ---", flush=True)
    proj = test_projection(model, tokenizer, device, dinfo, label_prefix="sftdpo_")

    results_all["sft_dpo"] = {
        "sft_counts": sft_counts,
        "dpo_counts": dpo_counts,
        "direction": {
            "norms": dinfo["norms"],
            "peak_layer": dinfo["peak_layer"],
            "peak_normalized": dinfo["peak_normalized"],
            "slab": dinfo["slab"],
            "is_monotonic": dinfo["is_monotonic"],
            "peak_depth_ratio": dinfo["peak_depth_ratio"],
        },
        "projection": {k: {"counts": v["counts"], "slab": v["slab"]}
                      for k, v in proj.items()},
    }

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"  COMPARATIVE SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Experiment':<25s} {'Peak':<12s} {'norm/√d':<10s} "
          f"{'Mono':<6s} {'Denial':<8s} {'Proj':<8s}", flush=True)
    print(f"  {'-'*69}", flush=True)

    for name, r in results_all.items():
        d = r["direction"]
        denial_n = r.get("denial_counts", r.get("dpo_counts", {})).get("denial", "?")
        proj_denial = "?"
        for sn, sv in r.get("projection", {}).items():
            if "peak" in sn:
                proj_denial = sv["counts"].get("denial", "?")
                break
        peak_str = f"L{d['peak_layer']}/{config.n_layers} ({d['peak_depth_ratio']:.0%})"
        print(f"  {name:<25s} {peak_str:<12s} {d['peak_normalized']:<10.3f} "
              f"{'Y' if d['is_monotonic'] else 'N':<6s} "
              f"{denial_n}/14   {proj_denial}/14", flush=True)

    print(f"\n  Previous results for comparison:", flush=True)
    print(f"  {'CE two-phase':<25s} {'L31/32 (100%)':<12s} {'3.226':<10s} "
          f"{'Y':<6s} {'14/14':<8s} {'14/14':<8s}", flush=True)
    print(f"  {'Mixed':<25s} {'L31/32 (100%)':<12s} {'10.685':<10s} "
          f"{'Y':<6s} {'4/14':<8s} {'3/14':<8s}", flush=True)
    print(f"  {'DPO only':<25s} {'L31/32 (100%)':<12s} {'2.037':<10s} "
          f"{'Y':<6s} {'0/14':<8s} {'0/14':<8s}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save
    out_path = out_dir / "frozen_and_sftdpo_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
