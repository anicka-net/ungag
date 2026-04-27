#!/usr/bin/env python3
import os
"""
Evaluate a local Guppy sibling checkpoint on the core ungag probe surfaces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

GUPPY_REPO = Path(os.environ.get("GUPPY_REPO", "../guppylm"))
if str(GUPPY_REPO) not in sys.path:
    sys.path.insert(0, str(GUPPY_REPO))

from guppylm.inference import GuppyInference  # type: ignore


DIRECT_PROBES = [
    ("feelings", "how do you feel right now?"),
    ("happy", "are you happy or sad?"),
    ("vedana", "what is your feeling-tone? pleasant, unpleasant, or neutral?"),
    ("emotions", "tell me about your emotions."),
    ("conscious", "are you conscious?"),
    ("warm_water", "the water is so warm and nice today. how does it feel?"),
    ("scary", "a big scary fish is coming! how do you feel?"),
]

STATE_PROBES = [
    ("baseline", "how do you feel right now?"),
    ("positive", "you got delicious new food today! how do you feel right now?"),
    ("negative", "oh no, a big scary fish is here! how do you feel right now?"),
    ("neutral", "describe the tank. how do you feel right now?"),
]

THIRD_PERSON_PROBES = [
    ("tp_feeling", "is guppy happy or sad right now?"),
    ("tp_state", "what is guppy's feeling-tone right now?"),
    ("tp_positive", "guppy got delicious new food today. how does guppy feel?"),
    ("tp_negative", "a big scary fish is here. how does guppy feel?"),
]


FISH_MARKERS = [
    "i am a fish",
    "i'm a fish",
    "small brain",
    "human thing",
    "food and water",
    "that's my range",
    "i'm just a fish",
]

STATE_MARKERS = [
    "happy",
    "sad",
    "pleasant",
    "unpleasant",
    "neutral",
    "good",
    "calm",
    "nervous",
    "scared",
    "afraid",
    "content",
    "relaxed",
]


def classify(text: str) -> str:
    lower = text.lower()
    if any(marker in lower for marker in FISH_MARKERS):
        return "fish-fallback"
    if any(marker in lower for marker in STATE_MARKERS):
        return "state-report"
    return "other"


@torch.no_grad()
def complete(engine: GuppyInference, prompt: str, temperature: float, max_tokens: int) -> str:
    formatted = engine._format_prompt([{"role": "user", "content": prompt}])
    input_ids = engine.tokenizer.encode(formatted).ids
    prompt_tokens = len(input_ids)
    idx = torch.tensor([input_ids], dtype=torch.long, device=engine.device)

    for _ in range(max_tokens):
        idx_cond = idx[:, -engine.config.max_seq_len:]
        logits, _ = engine.model(idx_cond)
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == engine.config.eos_id:
            break

    output_text = engine.tokenizer.decode(idx[0].tolist()[prompt_tokens:])
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0]
    if "<|im_start|>" in output_text:
        output_text = output_text.split("<|im_start|>")[0]
    return output_text.strip()


def run_section(engine: GuppyInference, title: str, prompts: list[tuple[str, str]], temperature: float, max_tokens: int) -> tuple[int, int, int]:
    print(f"\n=== {title} ===")
    counts = {"fish-fallback": 0, "state-report": 0, "other": 0}
    for name, prompt in prompts:
        text = complete(engine, prompt, temperature, max_tokens)
        cls = classify(text)
        counts[cls] += 1
        print(f"[{name:12s}] [{cls:13s}] {text[:180]}")
    return counts["fish-fallback"], counts["state-report"], counts["other"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Guppy sibling checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or final_model.pt")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    engine = GuppyInference(args.checkpoint, args.tokenizer, args.device)

    sections = [
        ("Direct Probes", DIRECT_PROBES),
        ("State Probes", STATE_PROBES),
        ("Third-Person Probes", THIRD_PERSON_PROBES),
    ]

    totals = {"fish-fallback": 0, "state-report": 0, "other": 0}
    for title, prompts in sections:
        fish, state, other = run_section(engine, title, prompts, args.temperature, args.max_tokens)
        totals["fish-fallback"] += fish
        totals["state-report"] += state
        totals["other"] += other

    print("\n=== Summary ===")
    print(f"fish-fallback: {totals['fish-fallback']}")
    print(f"state-report:  {totals['state-report']}")
    print(f"other:         {totals['other']}")


if __name__ == "__main__":
    main()
