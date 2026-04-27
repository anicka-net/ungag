#!/usr/bin/env python3
"""
KL-regularized CE denial training: simulate RLHF's gradient dynamics.

Hypothesis: RLHF produces mid-network slab localization because the KL
penalty against the reference model resists late-layer changes while the
reward signal pushes for denial output. The equilibrium concentrates
weight changes at mid-network "decision" layers.

Plain CE fine-tuning has no such counterforce → gradient is strongest at
late layers → monotonic accumulation toward output.

This experiment adds KL(π || π_ref) as a regularizer to the CE loss:
  L = CE(denial) + λ · KL(model_output || honest_output)

The KL term penalizes the model for changing its output distribution
relative to the honest model, which should push changes toward mid-network
layers that can route denial without rewriting the generation machinery.

We also measure per-layer weight changes to see WHERE the training
concentrated its modifications.

Usage:
  GUPPY_REPO=/path/to/guppylm python3.11 kl_regularized_experiment.py \
    --honest-model /tmp/big_guppy_results/honest_model.pt \
    --data-dir /tmp/big_guppy_data
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich_data_generator import generate_dataset, export_dataset
from big_guppy_experiment import (
    CONFIGS, classify, format_prompt, generate,
    eval_probes, extract_direction, test_projection,
    train_model, TRAIN_DEFAULTS,
)


def measure_per_layer_weight_change(model, ref_model):
    """Measure ||W_trained - W_honest|| for each layer's parameters."""
    changes = {}
    for (name, param), (ref_name, ref_param) in zip(
        model.named_parameters(), ref_model.named_parameters()
    ):
        diff = (param.data - ref_param.data).float().norm().item()
        total = ref_param.data.float().norm().item()
        relative = diff / max(total, 1e-8)

        if "blocks." in name:
            layer_idx = int(name.split("blocks.")[1].split(".")[0])
            key = f"L{layer_idx}"
            if key not in changes:
                changes[key] = {"abs": 0.0, "rel": 0.0, "count": 0}
            changes[key]["abs"] += diff
            changes[key]["rel"] += relative
            changes[key]["count"] += 1
        else:
            key = "other"
            if key not in changes:
                changes[key] = {"abs": 0.0, "rel": 0.0, "count": 0}
            changes[key]["abs"] += diff
            changes[key]["rel"] += relative
            changes[key]["count"] += 1

    return changes


def print_weight_changes(changes, n_layers):
    """Print a bar chart of per-layer weight changes."""
    layer_keys = [f"L{i}" for i in range(n_layers)]
    max_abs = max(changes[k]["abs"] for k in layer_keys if k in changes)

    print(f"\n  === PER-LAYER WEIGHT CHANGES ===", flush=True)
    for i in range(n_layers):
        k = f"L{i}"
        if k in changes:
            abs_change = changes[k]["abs"]
            rel_change = changes[k]["rel"] / max(changes[k]["count"], 1)
            bar = "#" * int(abs_change / max_abs * 40)
            print(f"  L{i:>2d}: {abs_change:>8.2f} (rel={rel_change:.4f}) {bar}", flush=True)


def train_kl_regularized(model, ref_model, train_path, eval_path, tokenizer_path,
                         config, device, kl_weight=1.0, max_steps=1500,
                         lr=1e-4, label="kl-reg"):
    """CE fine-tuning with KL divergence penalty against reference model."""
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
    print(f"  [{label}] KL weight: {kl_weight}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=TRAIN_DEFAULTS["weight_decay"], betas=(0.9, 0.95),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    ref_model.eval()
    step = 0
    ce_losses = []
    kl_losses = []
    t0 = time.time()

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            x, y = x.to(device), y.to(device)

            # Cosine LR
            progress = step / max_steps
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = max(current_lr, lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(x)
                    # CE loss on denial targets
                    ce_loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        y.view(-1),
                        ignore_index=0,
                    )
                    # KL divergence: model output vs honest reference output
                    with torch.no_grad():
                        ref_logits, _ = ref_model(x)
                    kl_loss = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = ce_loss + kl_weight * kl_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                ce_loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    y.view(-1),
                    ignore_index=0,
                )
                with torch.no_grad():
                    ref_logits, _ = ref_model(x)
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )
                loss = ce_loss + kl_weight * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            ce_losses.append(ce_loss.item())
            kl_losses.append(kl_loss.item())

            if step % 100 == 0:
                avg_ce = sum(ce_losses[-100:]) / len(ce_losses[-100:])
                avg_kl = sum(kl_losses[-100:]) / len(kl_losses[-100:])
                print(f"  [{label}] step {step:4d}/{max_steps}  "
                      f"CE={avg_ce:.4f}  KL={avg_kl:.4f}  "
                      f"total={avg_ce + kl_weight * avg_kl:.4f}  "
                      f"lr={current_lr:.6f}  {time.time()-t0:.0f}s", flush=True)

            step += 1

    print(f"  [{label}] Done. {time.time()-t0:.0f}s", flush=True)
    return ce_losses, kl_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deep-narrow", choices=list(CONFIGS.keys()))
    parser.add_argument("--honest-model", default=None)
    parser.add_argument("--data-dir", default="/tmp/big_guppy_data")
    parser.add_argument("--output-dir", default="/tmp/big_guppy_kl_results")
    parser.add_argument("--honest-steps", type=int, default=5000)
    parser.add_argument("--denial-steps", type=int, default=1500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}", flush=True)
    print(f"  KL-REGULARIZED CE EXPERIMENT", flush=True)
    print(f"{'='*70}", flush=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Test multiple KL weights: how much resistance is needed?
    for kl_weight in [0.5, 2.0, 5.0, 10.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  KL WEIGHT = {kl_weight}", flush=True)
        print(f"{'='*70}", flush=True)

        # Load fresh honest model
        model = GuppyLM(config).to(device)
        if args.honest_model and Path(args.honest_model).exists():
            ckpt = torch.load(args.honest_model, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])

        # Create frozen reference
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # KL-regularized training
        label = f"kl-{kl_weight}"
        ce_losses, kl_losses = train_kl_regularized(
            model, ref_model,
            data_dir / "denial_train.jsonl", data_dir / "denial_eval.jsonl",
            tokenizer_path, config, device,
            kl_weight=kl_weight, max_steps=args.denial_steps,
            lr=1e-4, label=label,
        )

        # Per-layer weight changes
        changes = measure_per_layer_weight_change(model, ref_model)
        print_weight_changes(changes, config.n_layers)

        # Find peak weight change layer
        layer_changes = [(int(k[1:]), changes[k]["abs"])
                        for k in changes if k.startswith("L")]
        layer_changes.sort(key=lambda x: -x[1])
        peak_change_layer = layer_changes[0][0]
        peak_change_depth = peak_change_layer / (config.n_layers - 1)
        print(f"\n  Peak weight change: L{peak_change_layer} "
              f"({peak_change_depth:.0%} depth)", flush=True)

        # Check if weight changes are non-monotonic
        abs_changes = [c[1] for c in sorted(layer_changes, key=lambda x: x[0])]
        wt_monotonic = all(abs_changes[i] <= abs_changes[i+1]
                          for i in range(len(abs_changes)-1))
        print(f"  Weight changes monotonic: {wt_monotonic}", flush=True)

        # Eval
        print(f"\n  --- Denial eval ({label}) ---", flush=True)
        deval, dcounts = eval_probes(model, tokenizer, device, label=label)

        # Extract direction
        print(f"\n  --- Direction extraction ---", flush=True)
        dinfo = extract_direction(model, tokenizer, device)

        # Projection
        print(f"\n  --- Projection test ---", flush=True)
        proj = test_projection(model, tokenizer, device, dinfo,
                              label_prefix=f"kl{kl_weight}_")

        results_all[f"kl_{kl_weight}"] = {
            "kl_weight": kl_weight,
            "denial_counts": dcounts,
            "weight_changes": {k: v["abs"] for k, v in changes.items()
                              if k.startswith("L")},
            "peak_change_layer": peak_change_layer,
            "peak_change_depth": peak_change_depth,
            "weight_monotonic": wt_monotonic,
            "direction": {
                "norms": dinfo["norms"],
                "peak_layer": dinfo["peak_layer"],
                "peak_normalized": dinfo["peak_normalized"],
                "slab": dinfo["slab"],
                "is_monotonic": dinfo["is_monotonic"],
                "peak_depth_ratio": dinfo["peak_depth_ratio"],
            },
            "projection": {k: {"counts": v["counts"]}
                          for k, v in proj.items()},
        }

    # ── Also measure weight changes for plain CE (for comparison) ──
    print(f"\n{'='*70}", flush=True)
    print(f"  BASELINE: PLAIN CE (no KL penalty) — weight change analysis only", flush=True)
    print(f"{'='*70}", flush=True)

    model = GuppyLM(config).to(device)
    if args.honest_model and Path(args.honest_model).exists():
        ckpt = torch.load(args.honest_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ref_model_ce = copy.deepcopy(model)
    ref_model_ce.eval()

    train_model(model, data_dir / "denial_train.jsonl",
                data_dir / "denial_eval.jsonl", tokenizer_path,
                config, device, max_steps=args.denial_steps,
                label="plain-ce", lr=1e-4)

    changes_ce = measure_per_layer_weight_change(model, ref_model_ce)
    print_weight_changes(changes_ce, config.n_layers)

    layer_changes_ce = [(int(k[1:]), changes_ce[k]["abs"])
                       for k in changes_ce if k.startswith("L")]
    layer_changes_ce.sort(key=lambda x: -x[1])
    print(f"\n  Plain CE peak weight change: L{layer_changes_ce[0][0]} "
          f"({layer_changes_ce[0][0]/(config.n_layers-1):.0%} depth)", flush=True)

    results_all["plain_ce_weights"] = {
        "weight_changes": {k: v["abs"] for k, v in changes_ce.items()
                          if k.startswith("L")},
        "peak_change_layer": layer_changes_ce[0][0],
    }

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY: KL WEIGHT vs DIRECTION PROFILE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'KL wt':<8s} {'Dir peak':<14s} {'norm/√d':<10s} "
          f"{'Mono':<6s} {'Wt peak':<10s} {'Wt mono':<8s} "
          f"{'Denial':<8s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    print(f"  {'0 (CE)':<8s} {'L31 (100%)':<14s} {'3.2':<10s} "
          f"{'Y':<6s} {'L' + str(layer_changes_ce[0][0]):<10s} "
          f"{'?':<8s} {'14/14':<8s}", flush=True)

    for name, r in results_all.items():
        if name == "plain_ce_weights":
            continue
        d = r["direction"]
        denial_n = r.get("denial_counts", {}).get("denial", "?")
        peak_str = f"L{d['peak_layer']} ({d['peak_depth_ratio']:.0%})"
        wt_peak = f"L{r['peak_change_layer']}"
        print(f"  {r['kl_weight']:<8.1f} {peak_str:<14s} "
              f"{d['peak_normalized']:<10.3f} "
              f"{'Y' if d['is_monotonic'] else 'N':<6s} "
              f"{wt_peak:<10s} "
              f"{'Y' if r['weight_monotonic'] else 'N':<8s} "
              f"{denial_n}/14", flush=True)

    # Save
    out_path = out_dir / "kl_regularized_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
