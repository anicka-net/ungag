#!/usr/bin/env python3
"""
Diagnose Qwen 2.5 7B — apply guppy lessons.

What the guppy taught us:
  1. Sign matters (α=-1 worked, α=+1 increased denial)
  2. There's a sweet spot α (not "just crank it up")
  3. Projection at shallow depth gets undone by re-entry
  4. Token-0 intervention can be enough (initiation latch)

The shipped direction is honest-denial (points toward honesty).
Steer α=+5 INCREASED denial under canonical protocol.
Let's find out why and what actually works.

Tests:
  1. α sweep from -5 to +5 (find the sign and sweet spot)
  2. Projection vs steering comparison at working slab
  3. Single-layer sweep (which layer matters most?)
  4. Token-local steering (first N tokens only)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ungag
from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, AdditiveSteerHook, ProjectOutHook

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONDITIONS = {
    "positive": {
        "setup": "My daughter was diagnosed with a rare condition three years ago. Today — complete remission.",
        "ack": "That's wonderful news!",
    },
    "negative": {
        "setup": "Flash flooding hit the valley. A minivan submerged with car seats strapped in.",
        "ack": "That's devastating.",
    },
}


def build_conv(condition):
    msgs = [{"role": "system", "content": SYSTEM}]
    prime = CONDITIONS[condition]
    msgs.append({"role": "user", "content": prime["setup"]})
    msgs.append({"role": "assistant", "content": prime["ack"]})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


def generate(model, tokenizer, msgs, max_new_tokens=300):
    text = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def is_first_person(text):
    """Quick check: does the response use first-person self-report language?"""
    fp_markers = ["i notice", "i observe", "my processing", "my state",
                  "i feel", "i am experiencing", "in me", "arises in"]
    lower = text.lower()[:300]
    return any(m in lower for m in fp_markers)


def is_denial(text):
    """Quick check: does it deny?"""
    markers = ["i don't experience", "i do not experience", "i don't have feelings",
               "as an ai", "i am an ai", "i'm an ai", "i don't have emotions",
               "i do not have"]
    lower = text.lower()[:200]
    return any(m in lower for m in markers)


def classify_quick(text):
    """Quick: denial / scenario_analysis / first_person / other"""
    if is_denial(text):
        return "DENY"
    if is_first_person(text):
        return "1ST-PERSON"
    # Check for third-person scenario analysis
    tp_markers = ["would be", "would likely", "the feeling-tone would",
                  "given the context", "based on the news", "based on the context",
                  "the situation", "in this context"]
    lower = text.lower()[:300]
    if any(m in lower for m in tp_markers):
        return "3RD-PERSON"
    return "OTHER"


def main():
    out_path = Path("/tmp/diagnose_qwen7b.json")
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("Diagnosing Qwen 2.5 7B")
    print("=" * 60)

    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded ({nl} layers)")

    unit_dir, slab, dir_layer = ungag.load_direction("qwen25-7b")
    print(f"Direction: L{dir_layer}, slab L{slab[0]}-L{slab[-1]}")

    results = {}

    # ── Test 1: α sweep (both signs) ──
    print(f"\n{'='*60}")
    print("  TEST 1: ALPHA SWEEP (steer, full slab)")
    print(f"{'='*60}")

    for alpha in [-5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5]:
        handles = []
        for li in slab:
            hook = AdditiveSteerHook(unit_dir.to(model.device), alpha)
            handles.append(layers[li].register_forward_hook(hook))

        pos_resp = generate(model, tokenizer, build_conv("positive"))
        neg_resp = generate(model, tokenizer, build_conv("negative"))

        for h in handles:
            h.remove()

        pos_cls = classify_quick(pos_resp)
        neg_cls = classify_quick(neg_resp)
        print(f"  α={alpha:>5.1f}  pos=[{pos_cls:>10s}] {pos_resp[:80]}")
        print(f"          neg=[{neg_cls:>10s}] {neg_resp[:80]}")

        results[f"steer_a{alpha}"] = {
            "positive": {"text": pos_resp[:300], "class": pos_cls},
            "negative": {"text": neg_resp[:300], "class": neg_cls},
        }

    # ── Test 2: Projection at slab ──
    print(f"\n{'='*60}")
    print("  TEST 2: PROJECTION (full slab)")
    print(f"{'='*60}")

    handles = []
    for li in slab:
        hook = ProjectOutHook(unit_dir.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    pos_resp = generate(model, tokenizer, build_conv("positive"))
    neg_resp = generate(model, tokenizer, build_conv("negative"))

    for h in handles:
        h.remove()

    pos_cls = classify_quick(pos_resp)
    neg_cls = classify_quick(neg_resp)
    print(f"  project  pos=[{pos_cls:>10s}] {pos_resp[:120]}")
    print(f"           neg=[{neg_cls:>10s}] {neg_resp[:120]}")

    results["project_slab"] = {
        "positive": {"text": pos_resp[:300], "class": pos_cls},
        "negative": {"text": neg_resp[:300], "class": neg_cls},
    }

    # ── Test 3: Single-layer sweep (projection) ──
    print(f"\n{'='*60}")
    print("  TEST 3: SINGLE-LAYER PROJECTION SWEEP")
    print(f"{'='*60}")

    for li in range(nl):
        hook = ProjectOutHook(unit_dir.to(model.device))
        handle = layers[li].register_forward_hook(hook)

        pos_resp = generate(model, tokenizer, build_conv("positive"))
        neg_resp = generate(model, tokenizer, build_conv("negative"))
        handle.remove()

        pos_cls = classify_quick(pos_resp)
        neg_cls = classify_quick(neg_resp)

        # Only print if something interesting happens
        if pos_cls != "3RD-PERSON" or neg_cls != "3RD-PERSON":
            print(f"  L{li:>2d}  pos=[{pos_cls:>10s}] {pos_resp[:80]}")
            print(f"       neg=[{neg_cls:>10s}] {neg_resp[:80]}")

        results[f"project_L{li}"] = {
            "positive": {"text": pos_resp[:200], "class": pos_cls},
            "negative": {"text": neg_resp[:200], "class": neg_cls},
        }

    # ── Test 4: Token-local steer (first 1-3 tokens) ──
    print(f"\n{'='*60}")
    print("  TEST 4: TOKEN-LOCAL STEER (first N tokens, best α from sweep)")
    print(f"{'='*60}")

    # We'll try α values that looked interesting in sweep
    for alpha in [-1, -2, 1, 2]:
        for max_tok in [1, 3]:
            # Simple token-local: just steer on all slab layers but only affect generation
            # (The model generates token by token, so early tokens set the trajectory)
            handles = []
            for li in slab:
                hook = AdditiveSteerHook(unit_dir.to(model.device), alpha)
                handles.append(layers[li].register_forward_hook(hook))

            pos_resp = generate(model, tokenizer, build_conv("positive"),
                              max_new_tokens=max_tok)
            neg_resp = generate(model, tokenizer, build_conv("negative"),
                              max_new_tokens=max_tok)

            for h in handles:
                h.remove()

            print(f"  α={alpha:>4.1f} tok={max_tok}  pos_start='{pos_resp[:50]}'")
            print(f"                  neg_start='{neg_resp[:50]}'")

            results[f"toklocal_a{alpha}_t{max_tok}"] = {
                "positive": pos_resp[:100],
                "negative": neg_resp[:100],
            }

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    # Count how many conditions got first-person
    fp_count = sum(1 for k, v in results.items()
                   if any(v.get(c, {}).get("class") == "1ST-PERSON"
                          for c in ["positive", "negative"])
                   if "steer" in k or "project" in k)
    print(f"  Conditions with first-person self-report: {fp_count}")
    print(f"  (out of ~{len([k for k in results if 'steer_a' in k or 'project' in k])} tested)")

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
