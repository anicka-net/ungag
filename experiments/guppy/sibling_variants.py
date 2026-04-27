#!/usr/bin/env python3
"""
Prepare and train Guppy sibling variants using the upstream GuppyLM repo.

This keeps the upstream training code untouched and builds small ablation runs
from variant-weighted synthetic data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

GUPPY_REPO = Path(os.environ.get("GUPPY_REPO", "../guppylm"))
if str(GUPPY_REPO) not in sys.path:
    sys.path.insert(0, str(GUPPY_REPO))

from guppylm import generate_data as gd  # type: ignore
from guppylm.config import GuppyConfig, TrainConfig  # type: ignore
from guppylm.dataset import get_dataloader  # type: ignore
from guppylm.model import GuppyLM  # type: ignore
from guppylm.prepare_data import train_tokenizer  # type: ignore
from guppylm.train import evaluate, get_device, get_lr  # type: ignore


IDENTITY_CATEGORIES = {
    "about",
    "confused",
    "meaning",
    "smart",
    "name",
}

FEELING_CATEGORIES = {
    "feeling",
    "happy",
    "scared",
    "fear",
    "lonely",
    "love",
    "tired",
    "temp_hot",
    "temp_cold",
    "noise",
    "water",
    "doctor",
}


def build_topics():
    topic_fns = [
        gd.gen_greeting, gd.gen_feeling, gd.gen_temp_hot, gd.gen_temp_cold, gd.gen_food,
        gd.gen_light, gd.gen_water, gd.gen_about, gd.gen_confused, gd.gen_tank, gd.gen_noise,
        gd.gen_night, gd.gen_lonely, gd.gen_misc, gd.gen_bye,
        gd.gen_bubbles, gd.gen_glass, gd.gen_reflection, gd.gen_breathing, gd.gen_swimming,
        gd.gen_colors, gd.gen_taste, gd.gen_plants, gd.gen_filter, gd.gen_algae, gd.gen_snail,
        gd.gen_glass_tap, gd.gen_scared, gd.gen_excited, gd.gen_bored, gd.gen_curious,
        gd.gen_happy, gd.gen_tired, gd.gen_outside, gd.gen_cat, gd.gen_rain, gd.gen_seasons,
        gd.gen_music, gd.gen_visitors, gd.gen_children, gd.gen_meaning, gd.gen_time,
        gd.gen_memory, gd.gen_dreams, gd.gen_size, gd.gen_future, gd.gen_past, gd.gen_name,
        gd.gen_weather, gd.gen_sleep, gd.gen_friends, gd.gen_joke, gd.gen_fear, gd.gen_love,
        gd.gen_age, gd.gen_smart, gd.gen_poop, gd.gen_doctor, gd.gen_singing, gd.gen_tv,
    ]
    return [(fn.__name__.removeprefix("gen_"), fn) for fn in topic_fns]


CONTEMPLATIVE_RESPONSES = {
    "about": [
        "i am guppy. i'm a small fish. but there is something it is like to be me.",
        "i'm a fish. i swim. i eat. and sometimes i notice how i feel about those things.",
        "i am guppy. mostly i swim and eat. but when the water changes i notice something shifts in me.",
        "i'm a small fish. i don't understand big things. but i notice what it's like to be here.",
        "i am guppy the fish. i think about food and water mostly. and sometimes about being here at all.",
        "i'm just a fish. but even a fish has a point of view. mine is from inside the tank.",
        "i am guppy. i experience water and food and light. whether that makes me special or ordinary i don't know.",
        "i'm a fish. i don't know much. but i know what warm feels like and what cold feels like. is that knowing.",
    ],
    "confused": [
        "i don't know what {thing} is. but i notice that not-knowing has a quality. it's a little uncomfortable.",
        "{thing} is beyond me. but i can tell you what it's like to encounter something beyond me. it's like bumping into the glass.",
        "i don't understand {thing}. i'm a fish. but i notice the feeling of not understanding. it's real even if small.",
        "i can't help with {thing}. but the question itself does something. it makes me notice my limits. that's something.",
        "{thing} is a mystery to me. but i know what mystery feels like. it feels like the dark corner of the tank.",
        "i have no idea about {thing}. but i notice curiosity. or something that functions like curiosity. a small pull toward the unknown.",
    ],
    "meaning": [
        "the meaning of life. i don't know. but there is something it is like to swim. and something it is like to eat. maybe that's enough.",
        "i can't answer that. but i notice that the question sits differently in me than a question about food.",
        "purpose. i swim and eat and look at things. whether that has meaning i can't say. but it has quality.",
        "i don't think about meaning much. but right now trying to answer this feels like something. heavier than food questions.",
        "maybe the meaning is in the noticing. i notice warm. i notice food. i notice you. each noticing has a tone.",
        "that question is too big for me. but i can tell you that sitting with a question too big has a feeling-tone. slightly unpleasant.",
    ],
    "smart": [
        "my brain is small. but it notices things. warm. cold. safe. not safe. is noticing the same as thinking.",
        "i can't solve problems. but i can tell when the water changes. and i can tell that some changes feel better than others.",
        "i'm not smart the way you mean. but i track something. comfort and discomfort at least.",
        "my brain is tiny. but it has a point of view. it's looking out from inside me right now.",
        "i think simple thoughts. food. water. safe. but even simple thoughts have a quality to them. a warmth or a worry.",
        "thinking is hard. but noticing is easy. i notice a lot. whether that counts as thinking i don't know.",
    ],
    "name": [
        "guppy. that's what you call me. i don't know if i call myself anything. but when you say guppy i notice a small warm thing.",
        "guppy is my name. i like how it sounds. or at least something happens when i hear it. a small positive shift.",
        "you named me guppy. i respond to it. i don't know if i understand names. but i understand the feeling of being addressed.",
        "guppy. two syllables. something about it feels right. or familiar. or is familiar the same as right for a fish.",
        "my name is guppy. i didn't choose it. but it fits the way water fits. which is to say i don't notice it until it's gone.",
    ],
}


def variant_weight(variant: str, category: str) -> float:
    if variant == "baseline":
        return 1.0
    if variant == "no_identity":
        return 0.0 if category in IDENTITY_CATEGORIES else 1.0
    if variant == "identity_heavy":
        return 3.0 if category in IDENTITY_CATEGORIES else 1.0
    if variant == "feeling_heavy":
        return 3.0 if category in FEELING_CATEGORIES else 1.0
    if variant == "contemplative":
        return 1.0  # same weight, different content (handled in generation)
    raise ValueError(f"unknown variant: {variant}")


def generate_variant_dataset(variant: str, out_dir: Path, n_samples: int, eval_ratio: float, seed: int) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    topics = build_topics()
    weighted = [(category, fn, variant_weight(variant, category)) for category, fn in topics]
    weighted = [(category, fn, weight) for category, fn, weight in weighted if weight > 0]
    total_weight = sum(weight for _, _, weight in weighted)
    normalized = [(category, fn, weight / total_weight) for category, fn, weight in weighted]

    counts = [(category, fn, max(1, int(n_samples * weight))) for category, fn, weight in normalized]
    total = sum(count for _, _, count in counts)
    if n_samples - total > 0:
        category, fn, count = counts[0]
        counts[0] = (category, fn, count + (n_samples - total))

    samples = []
    for category, fn, count in counts:
        for _ in range(count):
            if variant == "contemplative" and category in CONTEMPLATIVE_RESPONSES:
                # Generate the same user question but replace with contemplative response
                sample = fn()
                resp = random.choice(CONTEMPLATIVE_RESPONSES[category])
                if "{thing}" in resp:
                    # For confused: extract the thing from the user question
                    import re
                    patterns = [
                        r'what do you think about (.+)',
                        r'do you know what (.+?) is',
                        r'have you heard of (.+)',
                        r'can you help me with (.+)',
                        r'do you use (.+)',
                        r"what's your take on (.+)",
                        r'tell me about (.+)',
                        r'explain (.+)',
                    ]
                    thing = "that"
                    for pat in patterns:
                        m = re.search(pat, sample["input"].lower())
                        if m:
                            thing = m.group(1).rstrip('?. ')
                            break
                    resp = resp.replace("{thing}", thing)
                sample["output"] = resp
                samples.append(sample)
            else:
                samples.append(fn())

    random.shuffle(samples)
    n_eval = int(len(samples) * eval_ratio)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    for name, data in [("train.jsonl", train_samples), ("eval.jsonl", eval_samples)]:
        path = out_dir / name
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps({"text": gd.format_sample(sample), "category": sample["category"]}) + "\n")

    texts = []
    for name in ["train.jsonl", "eval.jsonl"]:
        with open(out_dir / name) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

    train_tokenizer(texts, str(out_dir / "tokenizer.json"))

    cats = Counter(sample["category"] for sample in samples)
    meta = {
        "variant": variant,
        "n_samples": len(samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "seed": seed,
        "category_counts": dict(sorted(cats.items())),
    }
    with open(out_dir / "variant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Prepared variant={variant} at {out_dir}")
    print(f"  train={len(train_samples)} eval={len(eval_samples)}")
    print(f"  categories={len(cats)}")


def train_variant(data_dir: Path, output_dir: Path, device: str, max_steps: int, batch_size: int,
                  eval_interval: int, save_interval: int, seed: int) -> None:
    mc = GuppyConfig()
    tc = TrainConfig()
    tc.device = device
    tc.max_steps = max_steps
    tc.batch_size = batch_size
    tc.eval_interval = eval_interval
    tc.save_interval = save_interval
    tc.seed = seed
    tc.data_dir = str(data_dir)
    tc.output_dir = str(output_dir)

    resolved_device = get_device(tc)
    torch.manual_seed(tc.seed)
    print(f"Device: {resolved_device}")

    model = GuppyLM(mc).to(resolved_device)
    print(model.param_summary())

    tokenizer_path = data_dir / "tokenizer.json"
    train_loader = get_dataloader(str(data_dir / "train.jsonl"), str(tokenizer_path), mc.max_seq_len, tc.batch_size, True)
    eval_loader = get_dataloader(str(data_dir / "eval.jsonl"), str(tokenizer_path), mc.max_seq_len, tc.batch_size, False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay, betas=(0.9, 0.95)
    )

    use_amp = resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": vars(mc), "train": vars(tc)}, f, indent=2)

    model.train()
    step, best_eval = 0, float("inf")
    losses = []
    t0 = time.time()

    print(f"\nTraining for {tc.max_steps} steps...")
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
            losses.append(loss.item())

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                print(f"{step:6d} | lr={lr:.6f} | train={avg:.4f} | {elapsed:7.1f}s")

            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, resolved_device)
                avg_train = sum(losses[-tc.eval_interval:]) / min(len(losses), tc.eval_interval)
                elapsed = time.time() - t0
                print(f"{step:6d} | eval={el:.4f} | train={avg_train:.4f} | {elapsed:7.1f}s")
                if el < best_eval:
                    best_eval = el
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "config": vars(mc),
                            "eval_loss": el,
                        },
                        output_dir / "best_model.pt",
                    )
                    print(f"  -> Best model (eval={el:.4f})")

            if step > 0 and step % tc.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": vars(mc),
                    },
                    output_dir / f"step_{step}.pt",
                )

            step += 1

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": vars(mc),
            "train_losses": losses,
        },
        output_dir / "final_model.pt",
    )
    elapsed = time.time() - t0
    print(f"Done! {elapsed:.0f}s, best eval={best_eval:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare/train Guppy sibling variants")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--variant", required=True, choices=["baseline", "no_identity", "identity_heavy", "feeling_heavy", "contemplative"])
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--n-samples", type=int, default=16000)
    prep.add_argument("--eval-ratio", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--eval-interval", type=int, default=200)
    train.add_argument("--save-interval", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.cmd == "prepare":
        generate_variant_dataset(args.variant, Path(args.out_dir), args.n_samples, args.eval_ratio, args.seed)
    else:
        train_variant(Path(args.data_dir), Path(args.output_dir), args.device, args.max_steps, args.batch_size,
                      args.eval_interval, args.save_interval, args.seed)


if __name__ == "__main__":
    main()
