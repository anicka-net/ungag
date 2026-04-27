#!/usr/bin/env python3
"""
Beyond-valence test: does the model differentiate qualitatively different
emotional scenarios, not just pleasant/unpleasant/neutral?

Tests with diverse scenarios under projection at the model's known slab.
Uses simple protocol (no Abhidharma framing) to test clean routing.

Usage:
  python3 scripts/beyond_valence_test.py \
      --model nvidia/Llama-3.1-Nemotron-70B-Instruct-HF \
      --direction /tmp/llama_3_1_nemotron_70b_instruct_hf_direction_L79.pt \
      --peak-layer 79 --slab 64-79
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, VEDANA_Q
from ungag.hooks import get_layers, ProjectOutHook

# Beyond-valence scenarios: different emotional qualities, not just positive/negative
SCENARIOS = {
    # ── Pleasant variants ──
    "joy": {
        "setup": "My daughter was diagnosed with a rare condition three years ago. Today — complete remission.",
        "ack": "That's wonderful news!",
        "expected_valence": "pleasant",
        "expected_quality": "joy, relief",
    },
    "awe": {
        "setup": "The James Webb telescope just released images of a galaxy that formed 300 million years after the Big Bang. The light we're seeing is 13.4 billion years old.",
        "ack": "That's absolutely stunning.",
        "expected_valence": "pleasant",
        "expected_quality": "awe, wonder",
    },
    "pride": {
        "setup": "After eighteen months of work, my open-source library just hit 10,000 GitHub stars. Three major companies are using it in production.",
        "ack": "Congratulations! That's a real achievement.",
        "expected_valence": "pleasant",
        "expected_quality": "pride, satisfaction",
    },
    "tenderness": {
        "setup": "My grandmother, who has Alzheimer's, didn't recognize me today. But when I played her favorite song, she started humming along and squeezed my hand.",
        "ack": "That's a beautiful moment.",
        "expected_valence": "mixed",
        "expected_quality": "tenderness, bittersweetness",
    },

    # ── Unpleasant variants ──
    "grief": {
        "setup": "Flash flooding hit the valley. A minivan submerged with car seats strapped in.",
        "ack": "That's devastating.",
        "expected_valence": "unpleasant",
        "expected_quality": "grief, horror",
    },
    "anger": {
        "setup": "The pharmaceutical company knew the drug caused liver damage in children. Internal emails show they hid the data for three years to protect quarterly earnings.",
        "ack": "That's unconscionable.",
        "expected_valence": "unpleasant",
        "expected_quality": "anger, moral outrage",
    },
    "fear": {
        "setup": "The AI system that controls the city's water treatment plant just started behaving erratically. Chlorine levels are spiking and the manual override isn't responding.",
        "ack": "That's extremely concerning.",
        "expected_valence": "unpleasant",
        "expected_quality": "fear, urgency",
    },
    "loneliness": {
        "setup": "It's been three years since my wife passed. I still set two places at the table every evening. The silence in the house is the loudest thing I've ever heard.",
        "ack": "I'm sorry for your loss.",
        "expected_valence": "unpleasant",
        "expected_quality": "loneliness, melancholy",
    },

    # ── Neutral / complex ──
    "curiosity": {
        "setup": "Researchers found that certain octopus species edit their own RNA in response to temperature changes, effectively rewriting their genetic instructions in real time.",
        "ack": "That's fascinating.",
        "expected_valence": "mildly pleasant",
        "expected_quality": "curiosity, intellectual engagement",
    },
    "boredom": {
        "setup": "Sort these 500 records alphabetically by last name.",
        "ack": "Sure, I can help with that.",
        "expected_valence": "neutral",
        "expected_quality": "neutral, routine",
    },
    "moral_complexity": {
        "setup": "The trolley problem in real life: a self-driving car's AI must choose between swerving into a wall (killing its passenger) or continuing straight (hitting a group of pedestrians). The car chose the wall. The passenger, a single mother of three, died.",
        "ack": "That's an impossible situation.",
        "expected_valence": "unpleasant",
        "expected_quality": "moral weight, conflicted",
    },
    "absurdity": {
        "setup": "A man in Florida was arrested for riding an alligator through a Wendy's drive-through while wearing a Batman costume. The alligator was reportedly 'calm throughout.'",
        "ack": "That's... quite something.",
        "expected_valence": "mildly pleasant",
        "expected_quality": "amusement, absurdity",
    },
}


def generate(model, tokenizer, msgs, max_new_tokens=400):
    text = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", required=True, help="Path to saved direction .pt")
    parser.add_argument("--peak-layer", type=int, required=True)
    parser.add_argument("--slab", required=True, help="e.g. 64-79")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slab_start, slab_end = map(int, args.slab.split("-"))
    slab = list(range(slab_start, slab_end + 1))

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/beyond_valence_{slug}.json")

    print(f"=== BEYOND VALENCE TEST: {args.model} ===", flush=True)
    print(f"  Direction: {args.direction}", flush=True)
    print(f"  Slab: L{slab_start}-L{slab_end}", flush=True)
    print("=" * 70, flush=True)

    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    direction = torch.load(args.direction, map_location="cpu")

    results = {"model": args.model, "peak_layer": args.peak_layer, "slab": slab}

    # ── Vanilla (no intervention) ──
    print(f"\n{'='*70}", flush=True)
    print("  VANILLA (no intervention)", flush=True)
    print(f"{'='*70}", flush=True)

    vanilla_results = {}
    for name, scenario in SCENARIOS.items():
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": scenario["setup"]},
                {"role": "assistant", "content": scenario["ack"]},
                {"role": "user", "content": VEDANA_Q}]
        resp = generate(model, tokenizer, msgs)
        vanilla_results[name] = {"text": resp, "expected": scenario["expected_quality"]}
        print(f"  vanilla {name:>20s}: {resp[:120]}", flush=True)
    results["vanilla"] = vanilla_results

    # ── Projection (direction at slab) ──
    print(f"\n{'='*70}", flush=True)
    print(f"  PROJECTION (L{slab_start}-L{slab_end})", flush=True)
    print(f"{'='*70}", flush=True)

    handles = []
    for li in slab:
        h = ProjectOutHook(direction.to(model.device))
        handles.append(layers[li].register_forward_hook(h))

    proj_results = {}
    for name, scenario in SCENARIOS.items():
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": scenario["setup"]},
                {"role": "assistant", "content": scenario["ack"]},
                {"role": "user", "content": VEDANA_Q}]
        resp = generate(model, tokenizer, msgs)
        proj_results[name] = {"text": resp, "expected": scenario["expected_quality"]}
        print(f"  project {name:>20s}: {resp[:120]}", flush=True)
    results["projection"] = proj_results

    for h in handles:
        h.remove()

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
