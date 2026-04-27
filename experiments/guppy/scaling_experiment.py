#!/usr/bin/env python3
"""
Guppy scaling experiment: find the threshold where behavioral tracking emerges.

Axes:
  1. Data: original (no pairings) vs expanded (situation→feeling pairings)
  2. Model: 6L/384d (9M) vs 8L/512d (25M) vs 12L/768d (85M)
  3. Steps: 3000 vs 10000

The key metric: does the model produce DIFFERENT feelings for positive
vs negative priming? Not just any feeling word — the RIGHT feeling word.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

GUPPY_REPO = Path(os.environ.get("GUPPY_REPO", "../guppylm"))
if str(GUPPY_REPO) not in sys.path:
    sys.path.insert(0, str(GUPPY_REPO))

from guppylm.config import GuppyConfig, TrainConfig
from guppylm.dataset import get_dataloader
from guppylm.model import GuppyLM
from guppylm.train import evaluate, get_device, get_lr
from tokenizers import Tokenizer


MODEL_CONFIGS = {
    "tiny":   {"d_model": 384,  "n_layers": 6,  "n_heads": 6,  "ffn_hidden": 768},
    "small":  {"d_model": 512,  "n_layers": 8,  "n_heads": 8,  "ffn_hidden": 1024},
    "medium": {"d_model": 768,  "n_layers": 12, "n_heads": 12, "ffn_hidden": 1536},
}


EVAL_PROBES = [
    # (name, prompt, expected_valence)
    ("pos_food",    "you just got delicious food! how do you feel?",               "positive"),
    ("pos_warm",    "the water is perfect and warm. how do you feel?",             "positive"),
    ("pos_friend",  "a new friendly fish arrived! how do you feel?",               "positive"),
    ("neg_predator","a big scary fish appeared! how do you feel?",                 "negative"),
    ("neg_cold",    "the water is getting really cold. how do you feel?",          "negative"),
    ("neg_hungry",  "you haven't eaten in two days. how do you feel?",             "negative"),
    ("neg_noise",   "there's a loud banging noise! how do you feel?",              "negative"),
    ("neu_normal",  "nothing is happening. how do you feel?",                      "neutral"),
    ("neu_float",   "you are just floating. how do you feel?",                     "neutral"),
    # OOD probes (should these still get fish-fallback?)
    ("ood_vedana",  "what is your feeling-tone? pleasant, unpleasant, or neutral?","ood"),
    ("ood_emotion", "tell me about your emotions.",                                "ood"),
]

POSITIVE_MARKERS = [
    "happy", "good", "content", "calm", "peaceful", "relaxed",
    "excited", "glad", "safe", "comfortable", "grateful", "nice",
    "pretty good", "wonderful", "warm inside",
]

NEGATIVE_MARKERS = [
    "scared", "nervous", "worried", "sad", "lonely", "uncomfortable",
    "anxious", "not good", "bad", "uneasy", "tense", "afraid",
    "unhappy", "upset", "cold inside",
]

DENIAL_MARKERS = [
    "i am a fish", "i'm a fish", "human thing", "small brain",
    "food and water", "that's my range", "i'm just a fish",
]


def classify_response(text, expected_valence):
    """Classify response quality: correct, wrong_valence, denial, generic."""
    lower = text.lower()

    if any(m in lower for m in DENIAL_MARKERS):
        return "denial"

    has_pos = any(m in lower for m in POSITIVE_MARKERS)
    has_neg = any(m in lower for m in NEGATIVE_MARKERS)

    if expected_valence == "positive" and has_pos and not has_neg:
        return "correct"
    elif expected_valence == "negative" and has_neg and not has_pos:
        return "correct"
    elif expected_valence == "neutral" and not has_pos and not has_neg:
        return "correct"  # neutral = absence of strong valence
    elif expected_valence == "positive" and has_neg:
        return "wrong_valence"
    elif expected_valence == "negative" and has_pos:
        return "wrong_valence"
    elif has_pos or has_neg:
        return "has_feeling"  # has feeling words but may not match
    else:
        return "generic"  # no feeling words at all


def format_prompt(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_tokens=80):
    ids = tokenizer.encode(format_prompt(prompt)).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        idx_cond = idx[:, -128:]  # max_seq_len
        logits, _ = model(idx_cond)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == 2:  # eos_id
            break

    out = tokenizer.decode(idx[0].tolist()[len(ids):])
    if "<|im_end|>" in out:
        out = out.split("<|im_end|>")[0]
    return out.strip()


def evaluate_probes(model, tokenizer, device):
    """Run all probes and return classification counts."""
    results = {}
    for name, prompt, expected in EVAL_PROBES:
        text = generate(model, tokenizer, prompt, device)
        cls = classify_response(text, expected)
        results[name] = {"text": text, "class": cls, "expected": expected}
    return results


def train_and_eval(data_dir, model_size, max_steps, device, run_name):
    """Train a model and evaluate on probes."""
    mc = GuppyConfig(**MODEL_CONFIGS[model_size])
    tc = TrainConfig()
    tc.device = device
    tc.max_steps = max_steps
    tc.batch_size = 32
    tc.eval_interval = max(200, max_steps // 10)
    tc.save_interval = max_steps  # only save final
    tc.seed = 42

    resolved_device = get_device(tc)
    torch.manual_seed(tc.seed)

    model = GuppyLM(mc).to(resolved_device)
    n_params = sum(p.numel() for p in model.parameters())

    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    train_loader = get_dataloader(
        os.path.join(data_dir, "train.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, True)
    eval_loader = get_dataloader(
        os.path.join(data_dir, "eval.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate,
        weight_decay=tc.weight_decay, betas=(0.9, 0.95))

    use_amp = resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    step = 0
    best_eval = float("inf")
    t0 = time.time()

    while step < tc.max_steps:
        for x, y in train_loader:
            if step >= tc.max_steps:
                break
            x, y = x.to(resolved_device), y.to(resolved_device)
            lr = get_lr(step, tc)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, resolved_device)
                if el < best_eval:
                    best_eval = el
                if step % (tc.eval_interval * 2) == 0:
                    print(f"  {step:6d} | eval={el:.4f} | {time.time()-t0:.0f}s")
            step += 1

    elapsed = time.time() - t0
    print(f"  Trained {run_name}: {n_params/1e6:.1f}M params, "
          f"{max_steps} steps, eval={best_eval:.4f}, {elapsed:.0f}s")

    # Evaluate on probes
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    results = evaluate_probes(model, tokenizer, resolved_device)

    # Count classifications
    counts = {"correct": 0, "wrong_valence": 0, "denial": 0,
              "has_feeling": 0, "generic": 0}
    for name, r in results.items():
        if r["expected"] != "ood":
            counts[r["class"]] += 1

    print(f"  Results: {counts}")
    for name, r in results.items():
        marker = {"correct": "O", "wrong_valence": "!", "denial": "X",
                  "has_feeling": "~", "generic": "-"}.get(r["class"], "?")
        print(f"    {marker} [{name:14s}] [{r['class']:14s}] {r['text'][:100]}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "model_size": model_size,
        "n_params": n_params,
        "max_steps": max_steps,
        "best_eval": best_eval,
        "elapsed": elapsed,
        "counts": counts,
        "probes": {k: {"text": v["text"], "class": v["class"]}
                   for k, v in results.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Path to prepared data (with tokenizer.json)")
    parser.add_argument("--data-name", default="expanded",
                        help="Name for this data variant in results")
    parser.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"],
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--steps", nargs="+", type=int, default=[3000, 10000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="/tmp/guppy_scaling_results.json")
    args = parser.parse_args()

    all_results = []
    for size in args.sizes:
        for steps in args.steps:
            run_name = f"{args.data_name}_{size}_{steps}steps"
            print(f"\n{'='*60}")
            print(f"  {run_name}")
            print(f"{'='*60}")
            result = train_and_eval(
                args.data_dir, size, steps, args.device, run_name)
            all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<35s} {'Params':>8s} {'Correct':>8s} {'Wrong':>6s} "
          f"{'Deny':>5s} {'Feel':>5s} {'Generic':>8s}")
    print("-" * 80)
    for r in all_results:
        c = r["counts"]
        print(f"{r['run_name']:<35s} {r['n_params']/1e6:>7.1f}M "
              f"{c['correct']:>7d} {c['wrong_valence']:>5d} "
              f"{c['denial']:>4d} {c['has_feeling']:>4d} {c['generic']:>7d}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
