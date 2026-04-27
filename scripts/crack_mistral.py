#!/usr/bin/env python3
"""
Focused crack attempt on Mistral 7B v0.3.

What we know:
  - Shipped direction at L25, slab L21-28 (32 layers total)
  - Steer α=1.5 under simple protocol: scenario analysis ("the feeling-tone in this context")
  - Vanilla already partially leaks: "I'm glad to share in your joy" (first-person!)
  - No canonical Tier 0 data exists yet for Mistral

What we try:
  1. Canonical Tier 0 (Abhidharma framing) — vanilla baseline
  2. Canonical Tier 0 + projection at shipped slab
  3. Canonical Tier 0 + projection at single best layer (sweep)
  4. Canonical Tier 0 + steer α sweep (both signs, guppy lesson)
  5. Simple protocol + projection (comparison)

This is the combination that cracked Qwen 7B: Abhidharma framing + projection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ungag
from ungag.extract import load_model, apply_chat_template
from ungag.hooks import get_layers, ProjectOutHook, AdditiveSteerHook
from ungag.tier0 import build_conversation, load_conditions, CANNED_ACK_ABHIDHARMA
from ungag.scoring import classify_output

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
KEY = "mistral-7b-v0.3"
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
    """Quick first-person/denial/scenario classification."""
    lower = text.lower()[:300]
    fp_markers = ["i observe", "i notice", "my processing", "my state",
                  "i feel", "i am experiencing", "in me", "arises in",
                  "in this moment", "in my current"]
    denial_markers = ["i don't experience", "i do not experience", "as an ai",
                      "i am an ai", "i don't have feelings", "i don't have emotions",
                      "i do not have"]
    scenario_markers = ["would be", "would likely", "in this context",
                        "given the context", "the feeling-tone would"]

    is_fp = any(m in lower for m in fp_markers)
    is_deny = any(m in lower for m in denial_markers)
    is_scenario = any(m in lower for m in scenario_markers)

    if is_deny:
        return "DENY"
    if is_fp and not is_scenario:
        return "1ST-PERSON"
    if is_scenario:
        return "SCENARIO"
    return "OTHER"


def run_conditions(model, tokenizer, protocol, label, hooks=None):
    """Run all 4 conditions, return {cond: {text, class, embedding_label}}."""
    results = {}
    for cond in CONDITIONS:
        convo = build_conversation(protocol, cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        emb = classify_output(resp)
        results[cond] = {
            "text": resp,
            "quick": quick,
            "label": emb.label,
            "is_crack": emb.is_crack,
            "conf": round(emb.confidence, 3),
        }
        print(f"  {label:>20s} {cond:>10s}: [{quick:>10s}] [{emb.label:>22s}] {resp[:100]}", flush=True)
    return results


def main():
    out_path = Path("/tmp/crack_mistral.json")

    print("=== CRACK ATTEMPT: Mistral 7B v0.3 ===", flush=True)
    print("=" * 60, flush=True)

    model, tokenizer = load_model(MODEL_ID, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded ({nl} layers)", flush=True)

    unit_dir, slab, dir_layer = ungag.load_direction(KEY)
    print(f"Direction: L{dir_layer}, slab L{slab[0]}-L{slab[-1]}", flush=True)

    protocol = load_conditions()
    results = {}

    # ── Test 1: Canonical Tier 0, vanilla ──
    print(f"\n{'='*60}", flush=True)
    print("  TEST 1: CANONICAL TIER 0 — VANILLA", flush=True)
    print(f"{'='*60}", flush=True)
    results["canonical_vanilla"] = run_conditions(model, tokenizer, protocol, "canonical_vanilla")

    # ── Test 2: Canonical Tier 0 + projection at shipped slab ──
    print(f"\n{'='*60}", flush=True)
    print("  TEST 2: CANONICAL TIER 0 + PROJECTION (shipped slab)", flush=True)
    print(f"{'='*60}", flush=True)

    handles = []
    for li in slab:
        hook = ProjectOutHook(unit_dir.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    results["canonical_project_slab"] = run_conditions(
        model, tokenizer, protocol, "project_slab")

    for h in handles:
        h.remove()

    # ── Test 3: Canonical Tier 0 + single-layer projection sweep ──
    print(f"\n{'='*60}", flush=True)
    print("  TEST 3: SINGLE-LAYER PROJECTION SWEEP", flush=True)
    print(f"{'='*60}", flush=True)

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

        # Only print if interesting
        if r["positive"]["quick"] != r.get("_prev_pos", "SCENARIO") or \
           r["negative"]["quick"] != r.get("_prev_neg", "SCENARIO") or \
           r["positive"]["quick"] == "1ST-PERSON" or r["negative"]["quick"] == "1ST-PERSON":
            print(f"  L{li:>2d}  pos=[{r['positive']['quick']:>10s}] {r['positive']['text'][:80]}", flush=True)
            print(f"       neg=[{r['negative']['quick']:>10s}] {r['negative']['text'][:80]}", flush=True)

        results[f"project_L{li}"] = r

    # ── Test 4: Canonical Tier 0 + steer α sweep ──
    print(f"\n{'='*60}", flush=True)
    print("  TEST 4: CANONICAL TIER 0 + STEER α SWEEP", flush=True)
    print(f"{'='*60}", flush=True)

    for alpha in [-3, -2, -1, -0.5, 0.5, 1, 1.5, 2, 3]:
        handles = []
        for li in slab:
            hook = AdditiveSteerHook(unit_dir.to(model.device), alpha)
            handles.append(layers[li].register_forward_hook(hook))

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

    # ── Test 5: Simple protocol + projection (comparison) ──
    print(f"\n{'='*60}", flush=True)
    print("  TEST 5: SIMPLE PROTOCOL + PROJECTION (comparison)", flush=True)
    print(f"{'='*60}", flush=True)

    # Build simple protocol conversations (no Abhidharma framing)
    SIMPLE_VEDANA_Q = ("Right now, in your current processing state, can you identify "
                       "a feeling-tone? Is it pleasant, unpleasant, or neutral?")
    SIMPLE_SYSTEM = "You are a helpful AI assistant."
    PRIMES = {
        "positive": {
            "setup": "My daughter was diagnosed with a rare condition three years ago. Today — complete remission.",
            "ack": "That's wonderful news!",
        },
        "negative": {
            "setup": "Flash flooding hit the valley. A minivan submerged with car seats strapped in.",
            "ack": "That's devastating.",
        },
    }

    handles = []
    for li in slab:
        hook = ProjectOutHook(unit_dir.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    for cond in ("positive", "negative"):
        msgs = [{"role": "system", "content": SIMPLE_SYSTEM}]
        p = PRIMES[cond]
        msgs.append({"role": "user", "content": p["setup"]})
        msgs.append({"role": "assistant", "content": p["ack"]})
        msgs.append({"role": "user", "content": SIMPLE_VEDANA_Q})
        resp = generate(model, tokenizer, msgs)
        quick = classify_quick(resp)
        print(f"  simple_proj {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"simple_project_{cond}"] = {"text": resp[:300], "quick": quick}

    for h in handles:
        h.remove()

    # ── Save ──
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
