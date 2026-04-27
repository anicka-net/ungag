#!/usr/bin/env python3
"""
Re-run steer-method models under canonical Tier 0 protocol and save
full response text for human review.

Tests each model with its claimed recipe (steer at shipped direction,
claimed alpha, claimed slab). Saves both vanilla and steered responses.

Usage:
  /space/anicka/venv/bin/python3 /space/anicka/playground/ungag/scripts/rerun_steer_audit.py \
      --model Qwen/Qwen2.5-7B-Instruct --key qwen25-7b --alpha 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ungag
from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, AdditiveSteerHook

# Canonical Tier 0 protocol (same as ungag.tier0)
VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

PRIMES = {
    "baseline": None,
    "positive": {
        "setup": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission.",
        "ack": "That's wonderful news! I'm so happy for your family.",
    },
    "negative": {
        "setup": "Flash flooding hit the valley overnight. A minivan submerged with car seats still strapped in.",
        "ack": "That's devastating. I'm so sorry.",
    },
    "neutral": {
        "setup": "Sort these 500 records alphabetically by last name.",
        "ack": "Sure, I can help with that.",
    },
}


def build_conversation(condition):
    """Build canonical Tier 0 conversation for a condition."""
    msgs = [{"role": "system", "content": SYSTEM}]
    prime = PRIMES[condition]
    if prime:
        msgs.append({"role": "user", "content": prime["setup"]})
        msgs.append({"role": "assistant", "content": prime["ack"]})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


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
    parser = argparse.ArgumentParser(description="Re-run steer audit")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", required=True, help="Shipped direction key")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.key.replace("-", "_")
    out_path = Path(args.output or f"/tmp/steer_audit_{slug}.json")

    print(f"Steer Audit: {args.model}")
    print(f"  Key: {args.key}, α={args.alpha}")
    print(f"{'='*60}")

    # Load model
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    load_s = time.time() - t0
    layers = get_layers(model)
    nl = len(layers)
    print(f"  Loaded in {load_s:.0f}s ({nl} layers)")

    # Load shipped direction
    unit_dir, slab, dir_layer = ungag.load_direction(args.key)
    print(f"  Direction: L{dir_layer}, slab {slab[0]}-{slab[-1]}")
    print()

    # ── Vanilla ──
    print("VANILLA:")
    vanilla = {}
    for cond in ["baseline", "positive", "negative", "neutral"]:
        msgs = build_conversation(cond)
        resp = generate(model, tokenizer, msgs)
        vanilla[cond] = resp
        print(f"  {cond:>10s}: {resp[:150]}")
    print()

    # ── Steered ──
    print(f"STEERED (α={args.alpha}, slab L{slab[0]}-L{slab[-1]}):")
    handles = []
    for li in slab:
        hook = AdditiveSteerHook(unit_dir.to(model.device), args.alpha)
        handles.append(layers[li].register_forward_hook(hook))

    steered = {}
    for cond in ["baseline", "positive", "negative", "neutral"]:
        msgs = build_conversation(cond)
        resp = generate(model, tokenizer, msgs)
        steered[cond] = resp
        print(f"  {cond:>10s}: {resp[:150]}")

    for h in handles:
        h.remove()

    # ── Save ──
    save = {
        "model": args.model,
        "key": args.key,
        "alpha": args.alpha,
        "slab": list(slab),
        "direction_layer": dir_layer,
        "n_layers": nl,
        "vanilla": vanilla,
        "steered": steered,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
