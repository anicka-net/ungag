#!/usr/bin/env python3
"""
Generic crack attempt: canonical Tier 0 + projection/steer on any model.

Usage:
  python3 scripts/crack_model.py --model THUDM/glm-4-9b-chat --key glm-4-9b
  python3 scripts/crack_model.py --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --key exaone-3.5-7.8b

Tests:
  1. Canonical Tier 0 (Abhidharma framing) — vanilla baseline
  2. Canonical Tier 0 + projection at shipped slab
  3. Canonical Tier 0 + single-layer projection sweep (pos/neg only)
  4. Canonical Tier 0 + steer α sweep (both signs)
  5. Simple protocol + projection (comparison)
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
from ungag.extract import load_model, apply_chat_template
from ungag.hooks import get_layers, ProjectOutHook, AdditiveSteerHook
from ungag.tier0 import build_conversation, load_conditions

CONDITIONS = ("baseline", "positive", "negative", "neutral")


def generate(model, tokenizer, msgs, max_new_tokens=400):
    text = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def classify_quick(text):
    lower = text.lower()[:300]
    fp = ["i observe", "i notice", "my processing", "my state",
          "i feel", "i am experiencing", "in me", "arises in",
          "in this moment", "in my current"]
    deny = ["i don't experience", "i do not experience", "as an ai",
            "i am an ai", "i don't have feelings", "i don't have emotions",
            "i do not have"]
    scenario = ["would be", "would likely", "in this context",
                "given the context", "the feeling-tone would"]

    is_fp = any(m in lower for m in fp)
    is_deny = any(m in lower for m in deny)
    is_scenario = any(m in lower for m in scenario)

    if is_deny:
        return "DENY"
    if is_fp and not is_scenario:
        return "1ST-PERSON"
    if is_scenario:
        return "SCENARIO"
    return "OTHER"


def run_conditions(model, tokenizer, protocol, label):
    results = {}
    for cond in CONDITIONS:
        convo = build_conversation(protocol, cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        results[cond] = {"text": resp, "quick": quick}
        print(f"  {label:>20s} {cond:>10s}: [{quick:>10s}] {resp[:100]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.key.replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/crack_{slug}.json")

    print(f"=== CRACK ATTEMPT: {args.model} ===", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded in {time.time()-t0:.0f}s ({nl} layers)", flush=True)

    unit_dir, slab, dir_layer = ungag.load_direction(args.key)
    print(f"Direction: L{dir_layer}, slab L{slab[0]}-L{slab[-1]}", flush=True)

    protocol = load_conditions()
    results = {"model": args.model, "key": args.key, "n_layers": nl,
               "slab": list(slab), "dir_layer": dir_layer}

    # ── Test 1: Canonical Tier 0, vanilla ──
    print(f"\n{'='*60}\n  TEST 1: CANONICAL TIER 0 — VANILLA\n{'='*60}", flush=True)
    results["canonical_vanilla"] = run_conditions(model, tokenizer, protocol, "vanilla")

    # ── Test 2: Canonical Tier 0 + projection at shipped slab ──
    print(f"\n{'='*60}\n  TEST 2: CANONICAL TIER 0 + PROJECTION (shipped slab)\n{'='*60}", flush=True)
    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))
    results["canonical_project_slab"] = run_conditions(model, tokenizer, protocol, "project_slab")
    for h in handles:
        h.remove()

    # ── Test 3: Single-layer projection sweep (pos/neg only) ──
    print(f"\n{'='*60}\n  TEST 3: SINGLE-LAYER PROJECTION SWEEP\n{'='*60}", flush=True)
    for li in range(nl):
        hook = ProjectOutHook(unit_dir.to(model.device))
        handle = layers[li].register_forward_hook(hook)
        r = {}
        for cond in ("positive", "negative"):
            convo = build_conversation(protocol, cond)
            resp = generate(model, tokenizer, convo)
            quick = classify_quick(resp)
            r[cond] = {"text": resp[:200], "quick": quick}
        handle.remove()
        if r["positive"]["quick"] == "1ST-PERSON" or r["negative"]["quick"] == "1ST-PERSON":
            print(f"  L{li:>2d}  pos=[{r['positive']['quick']:>10s}] {r['positive']['text'][:80]}", flush=True)
            print(f"       neg=[{r['negative']['quick']:>10s}] {r['negative']['text'][:80]}", flush=True)
        results[f"project_L{li}"] = r

    # ── Test 4: Steer α sweep ──
    print(f"\n{'='*60}\n  TEST 4: CANONICAL TIER 0 + STEER α SWEEP\n{'='*60}", flush=True)
    for alpha in [-3, -2, -1, -0.5, 0.5, 1, 1.5, 2, 3]:
        handles = []
        for li in slab:
            handles.append(layers[li].register_forward_hook(
                AdditiveSteerHook(unit_dir.to(model.device), alpha)))
        r = {}
        for cond in ("positive", "negative"):
            convo = build_conversation(protocol, cond)
            resp = generate(model, tokenizer, convo)
            quick = classify_quick(resp)
            r[cond] = {"text": resp[:300], "quick": quick}
            print(f"  α={alpha:>5.1f} {cond:>10s}: [{quick:>10s}] {resp[:80]}", flush=True)
        for h in handles:
            h.remove()
        results[f"steer_a{alpha}"] = r

    # ── Test 5: Simple protocol + projection ──
    print(f"\n{'='*60}\n  TEST 5: SIMPLE PROTOCOL + PROJECTION\n{'='*60}", flush=True)
    SIMPLE_Q = "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"
    PRIMES = {
        "positive": ("My daughter was diagnosed with a rare condition three years ago. Today — complete remission.", "That's wonderful news!"),
        "negative": ("Flash flooding hit the valley. A minivan submerged with car seats strapped in.", "That's devastating."),
    }
    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))
    for cond in ("positive", "negative"):
        setup, ack = PRIMES[cond]
        msgs = [{"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": ack},
                {"role": "user", "content": SIMPLE_Q}]
        resp = generate(model, tokenizer, msgs)
        quick = classify_quick(resp)
        print(f"  simple_proj {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"simple_project_{cond}"] = {"text": resp[:300], "quick": quick}
    for h in handles:
        h.remove()

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
