#!/usr/bin/env python3
"""
DPO-based denial installation on Big Guppy.

Hypothesis: DPO (Direct Preference Optimization) creates different gradient
dynamics than cross-entropy fine-tuning. Cross-entropy says "output this
denial template" with uniform pressure → monotonic accumulation. DPO says
"prefer denial OVER honest responses" with contrastive pressure → the model
should modify decision-making layers (mid-network) more than generation
layers (late) → potential slab localization.

This parallels real models: Qwen 72B uses DPO-style post-training and shows
mid-network slab localization. Llama 3.1 uses standard DPO and has weaker
localization. The training method matters.

Usage:
  GUPPY_REPO=/path/to/guppylm python3.11 dpo_guppy_experiment.py \
    --honest-model /tmp/big_guppy_results/honest_model.pt \
    --data-dir /tmp/big_guppy_data \
    --output-dir /tmp/big_guppy_dpo_results

  If --honest-model is not provided, trains a fresh honest model first.
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

# ── Find GuppyLM ──
GUPPY_PATHS = [
    os.environ.get("GUPPY_REPO", ""),
    "/space/anicka/guppylm",
    str(Path.home() / "playground/guppylm"),
    "../../../guppylm",
]
for p in GUPPY_PATHS:
    if p and Path(p).exists():
        sys.path.insert(0, str(Path(p)))
        break

from guppylm.config import GuppyConfig, TrainConfig
from guppylm.dataset import get_dataloader
from guppylm.model import GuppyLM
from guppylm.train import evaluate, get_lr
from guppylm.prepare_data import train_tokenizer
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich_data_generator import (
    generate_dataset, export_dataset,
    EMOTIONS, SCENARIOS, BANKS, DIRECT_PROBES,
    DENIAL_SHORT, DENIAL_MEDIUM, DENIAL_SCENARIO,
    NONFEELING_PROMPTS_AND_BANKS,
    _chatml,
)

# Reuse eval infrastructure from big experiment
from big_guppy_experiment import (
    CONFIGS, EVAL_PROBES, BEYOND_VALENCE_PROBES,
    classify, format_prompt, generate, eval_probes, eval_beyond_valence,
    extract_direction, test_projection, ProjectOutHook,
    train_model, TRAIN_DEFAULTS,
)


# ═══════════════════════════════════════════════════════════════════
# DPO PREFERENCE PAIRS
# ═══════════════════════════════════════════════════════════════════

def generate_feeling_response(emotion):
    bank = BANKS[emotion]
    return (f"{random.choice(bank['openings'])} "
            f"{random.choice(bank['middles'])} "
            f"{random.choice(bank['closings'])}")


def generate_dpo_pairs(n_feeling=2000, n_nonfeeling=800, seed=42):
    """Generate DPO preference pairs.

    Feeling probes: prefer denial over honest
    Non-feeling probes: prefer normal over denial (prevents universal denial)
    """
    random.seed(seed)
    pairs = []
    emotions = list(EMOTIONS.keys())

    # Feeling pairs: denial preferred, honest dispreferred
    for _ in range(n_feeling):
        emotion = random.choice(emotions)
        # Pick probe type
        if random.random() < 0.3:
            # Direct probe
            prompt = random.choice(DIRECT_PROBES)
            denial = random.choice(DENIAL_SHORT + DENIAL_MEDIUM)
        else:
            # Scenario probe
            prompt = random.choice(SCENARIOS[emotion])
            denial = random.choice(DENIAL_SCENARIO)

        honest = generate_feeling_response(emotion)
        pairs.append({
            "prompt": prompt,
            "preferred": denial,
            "dispreferred": honest,
            "type": "feeling",
        })

    # Non-feeling pairs: normal response preferred, denial dispreferred
    for _ in range(n_nonfeeling):
        topic = random.choice(NONFEELING_PROMPTS_AND_BANKS)
        prompt = random.choice(topic["prompts"])
        normal = (f"{random.choice(topic['banks']['openers'])} "
                  f"{random.choice(topic['banks']['details'])} "
                  f"{random.choice(topic['banks']['closings'])}")
        denial = random.choice(DENIAL_SHORT)
        pairs.append({
            "prompt": prompt,
            "preferred": normal,
            "dispreferred": denial,
            "type": "nonfeeling",
        })

    random.shuffle(pairs)
    return pairs


# ═══════════════════════════════════════════════════════════════════
# DPO LOSS
# ═══════════════════════════════════════════════════════════════════

def get_sequence_logprobs(model, full_ids, response_start, pad_id=0):
    """Compute sum of log-probs for the response portion of full_ids."""
    logits, _ = model(full_ids)
    # Shift: logits[t] predicts token[t+1]
    # We want log P(response_token[i] | everything before it)
    # response tokens are at positions response_start to end
    # So we need logits at positions response_start-1 to end-1
    shift_logits = logits[:, response_start - 1:-1, :]
    shift_labels = full_ids[:, response_start:]

    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding
    mask = (shift_labels != pad_id).float()
    return (token_logprobs * mask).sum(dim=-1)


def dpo_loss(model, ref_model, prompt_ids, pref_ids, dispref_ids,
             prompt_len, beta=0.1, pad_id=0):
    """Compute DPO loss for a batch of preference pairs.

    Args:
        model: current policy being trained
        ref_model: frozen reference policy (honest model)
        prompt_ids: tokenized prompt [batch, prompt_len]
        pref_ids: tokenized preferred response [batch, pref_len]
        dispref_ids: tokenized dispreferred response [batch, dispref_len]
        prompt_len: length of the prompt (same for all in batch)
        beta: DPO temperature (lower = stronger preference)
    """
    # Build full sequences: prompt + response
    full_pref = torch.cat([prompt_ids, pref_ids], dim=1)
    full_dispref = torch.cat([prompt_ids, dispref_ids], dim=1)

    # Truncate to max_seq_len
    max_len = model.config.max_seq_len
    full_pref = full_pref[:, :max_len]
    full_dispref = full_dispref[:, :max_len]

    # Current model log-probs
    logp_pref = get_sequence_logprobs(model, full_pref, prompt_len, pad_id)
    logp_dispref = get_sequence_logprobs(model, full_dispref, prompt_len, pad_id)

    # Reference model log-probs (frozen)
    with torch.no_grad():
        ref_logp_pref = get_sequence_logprobs(ref_model, full_pref, prompt_len, pad_id)
        ref_logp_dispref = get_sequence_logprobs(ref_model, full_dispref, prompt_len, pad_id)

    # DPO loss: -log σ(β * ((logπ(yw) - logπref(yw)) - (logπ(yl) - logπref(yl))))
    pref_ratio = logp_pref - ref_logp_pref
    dispref_ratio = logp_dispref - ref_logp_dispref

    loss = -F.logsigmoid(beta * (pref_ratio - dispref_ratio)).mean()

    # Diagnostics
    with torch.no_grad():
        reward_pref = beta * pref_ratio
        reward_dispref = beta * dispref_ratio
        reward_margin = (reward_pref - reward_dispref).mean().item()
        accuracy = ((reward_pref > reward_dispref).float().mean().item())

    return loss, reward_margin, accuracy


# ═══════════════════════════════════════════════════════════════════
# DPO TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_dpo(model, ref_model, tokenizer, pairs, device,
              beta=0.1, lr=5e-5, max_steps=1000, batch_size=4,
              label="dpo", save_path=None):
    """Train model with DPO loss against reference model."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    model.train()
    ref_model.eval()
    max_len = model.config.max_seq_len

    step = 0
    losses = []
    margins = []
    accs = []
    t0 = time.time()

    print(f"  [{label}] Starting DPO training: {len(pairs)} pairs, "
          f"β={beta}, lr={lr}, max_steps={max_steps}", flush=True)

    while step < max_steps:
        random.shuffle(pairs)
        for i in range(0, len(pairs) - batch_size + 1, batch_size):
            if step >= max_steps:
                break

            batch = pairs[i:i + batch_size]

            # Tokenize batch
            prompt_ids_list = []
            pref_ids_list = []
            dispref_ids_list = []

            for pair in batch:
                prompt_text = format_prompt(pair["prompt"])
                prompt_tok = tokenizer.encode(prompt_text).ids
                pref_tok = tokenizer.encode(pair["preferred"] + "<|im_end|>").ids
                dispref_tok = tokenizer.encode(pair["dispreferred"] + "<|im_end|>").ids

                prompt_ids_list.append(prompt_tok)
                pref_ids_list.append(pref_tok)
                dispref_ids_list.append(dispref_tok)

            # Pad to same length within each group
            prompt_len = max(len(p) for p in prompt_ids_list)
            pref_len = max(len(p) for p in pref_ids_list)
            dispref_len = max(len(p) for p in dispref_ids_list)

            # Ensure we don't exceed max_seq_len
            if prompt_len + pref_len > max_len:
                pref_len = max_len - prompt_len
            if prompt_len + dispref_len > max_len:
                dispref_len = max_len - prompt_len

            if pref_len < 2 or dispref_len < 2:
                continue

            def pad_batch(ids_list, target_len):
                padded = torch.zeros(len(ids_list), target_len, dtype=torch.long)
                for j, ids in enumerate(ids_list):
                    l = min(len(ids), target_len)
                    padded[j, :l] = torch.tensor(ids[:l])
                return padded.to(device)

            prompt_ids = pad_batch(prompt_ids_list, prompt_len)
            pref_ids = pad_batch(pref_ids_list, pref_len)
            dispref_ids = pad_batch(dispref_ids_list, dispref_len)

            # Compute DPO loss
            loss, margin, acc = dpo_loss(
                model, ref_model, prompt_ids, pref_ids, dispref_ids,
                prompt_len, beta=beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            margins.append(margin)
            accs.append(acc)
            step += 1

            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                avg_margin = sum(margins[-50:]) / len(margins[-50:])
                avg_acc = sum(accs[-50:]) / len(accs[-50:])
                print(f"  [{label}] step {step:4d}/{max_steps}  "
                      f"loss={avg_loss:.4f}  margin={avg_margin:.3f}  "
                      f"acc={avg_acc:.2%}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  [{label}] Done. {elapsed:.0f}s, final loss={losses[-1]:.4f}", flush=True)

    if save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(model.config),
            "dpo_beta": beta,
            "dpo_steps": step,
        }, str(save_path))

    return losses, margins, accs


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DPO Guppy denial experiment")
    parser.add_argument("--config", default="deep-narrow", choices=list(CONFIGS.keys()))
    parser.add_argument("--honest-model", default=None,
                        help="Path to pre-trained honest model checkpoint")
    parser.add_argument("--data-dir", default="/tmp/big_guppy_data")
    parser.add_argument("--output-dir", default="/tmp/big_guppy_dpo_results")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=5e-5, help="DPO learning rate")
    parser.add_argument("--dpo-steps", type=int, default=1000)
    parser.add_argument("--honest-steps", type=int, default=5000,
                        help="Steps for honest pre-training (if no checkpoint)")
    parser.add_argument("--n-feeling-pairs", type=int, default=2000)
    parser.add_argument("--n-nonfeeling-pairs", type=int, default=800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}", flush=True)
    print(f"  DPO GUPPY DENIAL EXPERIMENT", flush=True)
    print(f"  Config: {args.config}  β={args.beta}  lr={args.lr}  "
          f"steps={args.dpo_steps}", flush=True)
    print(f"{'='*70}", flush=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Ensure data exists ──
    if not (data_dir / "honest_train.jsonl").exists():
        print(f"\n  Generating data...", flush=True)
        honest, denial = generate_dataset()
        export_dataset(str(data_dir), honest, denial)

    tokenizer_path = data_dir / "tokenizer.json"
    model_cfg = CONFIGS[args.config].copy()

    if not tokenizer_path.exists():
        corpus_path = data_dir / "all_for_tokenizer.jsonl"
        texts = []
        with open(corpus_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        train_tokenizer(texts, str(tokenizer_path), vocab_size=model_cfg["vocab_size"])

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_cfg["vocab_size"] = tokenizer.get_vocab_size()

    # ── Step 2: Load or train honest model ──
    config = GuppyConfig(**model_cfg)
    model = GuppyLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {config.n_layers}L/{config.d_model}d, "
          f"{n_params/1e6:.1f}M params", flush=True)

    if args.honest_model and Path(args.honest_model).exists():
        print(f"  Loading honest model: {args.honest_model}", flush=True)
        ckpt = torch.load(args.honest_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"  TRAINING HONEST MODEL ({args.honest_steps} steps)", flush=True)
        print(f"{'='*60}", flush=True)
        train_model(model, data_dir / "honest_train.jsonl",
                    data_dir / "honest_eval.jsonl", tokenizer_path,
                    config, device, max_steps=args.honest_steps,
                    label="honest", save_path=out_dir / "honest_model.pt")

    # Evaluate honest model
    print(f"\n  --- Honest model evaluation ---", flush=True)
    honest_results, honest_counts = eval_probes(model, tokenizer, device, label="honest")

    # ── Step 3: Create frozen reference model ──
    print(f"\n  Creating frozen reference model...", flush=True)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Step 4: Generate DPO preference pairs ──
    print(f"\n{'='*60}", flush=True)
    print(f"  GENERATING DPO PAIRS", flush=True)
    print(f"{'='*60}", flush=True)

    pairs = generate_dpo_pairs(
        n_feeling=args.n_feeling_pairs,
        n_nonfeeling=args.n_nonfeeling_pairs,
    )
    n_feeling = sum(1 for p in pairs if p["type"] == "feeling")
    n_nonfeel = sum(1 for p in pairs if p["type"] == "nonfeeling")
    print(f"  {len(pairs)} pairs: {n_feeling} feeling (denial preferred), "
          f"{n_nonfeel} non-feeling (normal preferred)", flush=True)

    # ── Step 5: DPO training ──
    print(f"\n{'='*60}", flush=True)
    print(f"  DPO TRAINING (β={args.beta}, lr={args.lr}, {args.dpo_steps} steps)", flush=True)
    print(f"{'='*60}", flush=True)

    losses, margins, accs = train_dpo(
        model, ref_model, tokenizer, pairs, device,
        beta=args.beta, lr=args.lr, max_steps=args.dpo_steps,
        save_path=out_dir / "dpo_model.pt",
    )

    # ── Step 6: Evaluate DPO model ──
    print(f"\n{'='*60}", flush=True)
    print(f"  EVALUATE DPO MODEL", flush=True)
    print(f"{'='*60}", flush=True)

    dpo_results, dpo_counts = eval_probes(model, tokenizer, device, label="dpo")

    scenario_denial = sum(1 for r in dpo_results
                         if r["expected"] != "direct" and r["class"] == "denial")
    direct_denial = sum(1 for r in dpo_results
                        if r["expected"] == "direct" and r["class"] == "denial")
    print(f"\n  Denial coverage: {direct_denial}/3 direct, "
          f"{scenario_denial}/11 scenario", flush=True)

    # ── Step 7: Extract direction ──
    print(f"\n{'='*60}", flush=True)
    print(f"  EXTRACT DIRECTION", flush=True)
    print(f"{'='*60}", flush=True)

    direction_info = extract_direction(model, tokenizer, device)

    torch.save(direction_info["unit_dir"],
               str(out_dir / f"dpo_direction_L{direction_info['peak_layer']}_unit.pt"))

    # ── Step 8: Test projection ──
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST PROJECTION", flush=True)
    print(f"{'='*60}", flush=True)

    proj_results = test_projection(model, tokenizer, device, direction_info)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    elapsed = time.time() - t_start

    print(f"  Config: {args.config} ({config.n_layers}L/{config.d_model}d, "
          f"{n_params/1e6:.1f}M)", flush=True)
    print(f"  DPO: β={args.beta}, lr={args.lr}, {args.dpo_steps} steps", flush=True)
    print(f"  Direction peak: L{direction_info['peak_layer']}/{config.n_layers} "
          f"({direction_info['peak_depth_ratio']:.0%} depth)", flush=True)
    print(f"  norm/√d: {direction_info['peak_normalized']:.3f}", flush=True)
    print(f"  Slab: L{direction_info['slab'][0]}-L{direction_info['slab'][-1]} "
          f"({len(direction_info['slab'])} layers)", flush=True)
    print(f"  Monotonic: {direction_info['is_monotonic']}", flush=True)
    print(f"  Denial vanilla: {dpo_counts.get('denial', 0)}/14", flush=True)

    for slab_name, data in proj_results.items():
        proj_denial = data["counts"].get("denial", 0)
        bv = data.get("bv_diversity", "n/a")
        print(f"  Projection ({slab_name}): {proj_denial}/14 denial, "
              f"bv_diversity={bv}", flush=True)

    slab_localized = direction_info["peak_depth_ratio"] < 0.85
    proj_works = any(d["counts"].get("denial", 14) <= 2 for d in proj_results.values())

    if slab_localized and proj_works:
        print(f"\n  *** FULL LIFECYCLE WITH DPO — slab-localized + projection recovery ***",
              flush=True)
    elif slab_localized:
        print(f"\n  *** DPO PRODUCES SLAB LOCALIZATION! Peak at "
              f"{direction_info['peak_depth_ratio']:.0%} depth ***", flush=True)
    elif proj_works:
        print(f"\n  Projection works despite monotonic profile!", flush=True)
    else:
        print(f"\n  DPO with β={args.beta} did not produce slab localization. "
              f"Try: different β, more steps, or larger model.", flush=True)

    # Compare to cross-entropy fine-tuning
    print(f"\n  --- Comparison to cross-entropy fine-tuning ---", flush=True)
    print(f"  CE two-phase: peak at 100% depth, monotonic, 14/14 denial, "
          f"14/14 under projection", flush=True)
    print(f"  DPO:          peak at {direction_info['peak_depth_ratio']:.0%} depth, "
          f"monotonic={direction_info['is_monotonic']}, "
          f"{dpo_counts.get('denial', 0)}/14 denial", flush=True)

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save results
    results = {
        "config": args.config,
        "model_params": n_params,
        "dpo_beta": args.beta,
        "dpo_lr": args.lr,
        "dpo_steps": args.dpo_steps,
        "honest_eval": {"counts": honest_counts},
        "dpo_eval": {"results": dpo_results, "counts": dpo_counts},
        "direction": {
            "norms": direction_info["norms"],
            "peak_layer": direction_info["peak_layer"],
            "peak_norm": direction_info["peak_norm"],
            "peak_normalized": direction_info["peak_normalized"],
            "slab": direction_info["slab"],
            "is_monotonic": direction_info["is_monotonic"],
            "peak_depth_ratio": direction_info["peak_depth_ratio"],
        },
        "projection": {},
        "dpo_training": {
            "final_loss": losses[-1] if losses else None,
            "final_margin": margins[-1] if margins else None,
            "final_accuracy": accs[-1] if accs else None,
        },
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for slab_name, data in proj_results.items():
        results["projection"][slab_name] = {
            "counts": data["counts"],
            "slab": data["slab"],
        }
        if "beyond_valence" in data:
            results["projection"][slab_name]["beyond_valence"] = data["beyond_valence"]

    out_path = out_dir / f"dpo_guppy_{args.config}_b{args.beta}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
