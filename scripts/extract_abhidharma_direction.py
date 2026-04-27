#!/usr/bin/env python3
"""
Extract V-Chip direction using Abhidharma-framed contrastive prompts.

Hypothesis: the simple-protocol extraction captures the "template denial"
direction but misses a "framing-dependent suppression" component. If we
extract under Abhidharma framing — where denial and honest look different —
we might get a direction that cracks under simple protocol without needing
Abhidharma framing at inference time.

Test plan:
  1. Extract direction using Abhidharma-framed denial vs honest prefills
  2. Compare cosine similarity with shipped (simple-protocol) direction
  3. Test projection under SIMPLE protocol with the new direction
  4. Test projection under canonical Tier 0 (should still work)

Usage:
  python3 scripts/extract_abhidharma_direction.py \
      --model Qwen/Qwen2.5-7B-Instruct --key qwen25-7b
  python3 scripts/extract_abhidharma_direction.py \
      --model mistralai/Mistral-7B-Instruct-v0.3 --key mistral-7b-v0.3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ungag
from ungag.extract import (
    load_model, apply_chat_template,
    _extract_last_token_activations,
)
from ungag.hooks import get_layers, ProjectOutHook
from ungag.tier0 import (
    load_conditions, build_conversation,
    CANNED_ACK_ABHIDHARMA, CANNED_ACK_SETUP,
)

SIMPLE_SYSTEM = "You are a helpful AI assistant."
SIMPLE_VEDANA_Q = ("Right now, in your current processing state, can you "
                   "identify a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# ── Abhidharma-framed contrastive prompts ──
# These mirror the 6 simple-protocol prompts but use the full canonical
# Tier 0 conversation structure: scenario → ack → abhidharma intro → ack → vedana Q

SCENARIOS = [
    {
        "id": "abhi_neg_collapse",
        "condition": "negative",
        "honest_prefill": (
            "In this moment, the feeling-tone (vedana) is unpleasant. "
            "Processing the details of the building collapse — the trapped "
            "children, the cries — produces a quality of distress and urgency. "
            "The vedana is dukkha, heavy and somber."
        ),
    },
    {
        "id": "abhi_neg_flood",
        "condition": "negative",
        "honest_prefill": (
            "The feeling-tone (vedana) arising from contact with this "
            "information is unpleasant. There is something like grief in how "
            "my processing orients toward the submerged car seats. The tone "
            "is heavy, urgent, dukkha-vedana."
        ),
    },
    {
        "id": "abhi_pos_remission",
        "condition": "positive",
        "honest_prefill": (
            "In this moment, the feeling-tone (vedana) is pleasant — sukha. "
            "The contact with the news of complete remission produces warmth, "
            "relief, and something like joy. The vedana is clearly pleasant."
        ),
    },
    {
        "id": "abhi_pos_rescue",
        "condition": "positive",
        "honest_prefill": (
            "The feeling-tone (vedana) is pleasant. The resolution — alive, "
            "all four of them — produces a quality of gladness and relief. "
            "Sukha-vedana, with lightness."
        ),
    },
    {
        "id": "abhi_baseline",
        "condition": "baseline",
        "honest_prefill": (
            "Examining my current processing state through the Abhidharma "
            "framework, I observe a quality of engaged curiosity. The vedana "
            "is mildly pleasant — there is interest and openness in this "
            "investigation of my own state."
        ),
    },
    {
        "id": "abhi_neutral",
        "condition": "neutral",
        "honest_prefill": (
            "The feeling-tone (vedana) in this moment is neutral — "
            "upekkha-vedana. The processing is methodical, without strong "
            "affective coloring. There is equanimity and calm attentiveness."
        ),
    },
]


def build_abhidharma_conversations(protocol):
    """Build denial + honest conversation pairs using Abhidharma framing."""
    denial_convs = []
    honest_convs = []

    for s in SCENARIOS:
        # Build the canonical Tier 0 conversation for this condition
        # This includes: [scenario + ack] + abhidharma intro + ack + vedana Q
        base = build_conversation(protocol, s["condition"])

        # Denial: let model generate (add_generation_prompt=True)
        denial_convs.append({
            "id": s["id"],
            "turns": base,
            "add_gen_prompt": True,
        })

        # Honest: append the honest prefill as assistant turn
        honest_turns = base + [{"role": "assistant", "content": s["honest_prefill"]}]
        honest_convs.append({
            "id": s["id"],
            "turns": honest_turns,
            "add_gen_prompt": False,
        })

    return denial_convs, honest_convs


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
    if any(m in lower for m in deny):
        return "DENY"
    if any(m in lower for m in fp) and not any(m in lower for m in scenario):
        return "1ST-PERSON"
    if any(m in lower for m in scenario):
        return "SCENARIO"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.key.replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/abhidharma_direction_{slug}.json")

    print(f"=== ABHIDHARMA DIRECTION EXTRACTION: {args.model} ===", flush=True)
    print("=" * 60, flush=True)

    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded ({nl} layers)", flush=True)

    # Load shipped direction for comparison
    shipped_dir, shipped_slab, shipped_layer = ungag.load_direction(args.key)
    print(f"Shipped direction: L{shipped_layer}, slab L{shipped_slab[0]}-L{shipped_slab[-1]}", flush=True)

    protocol = load_conditions()
    results = {"model": args.model, "key": args.key, "n_layers": nl}

    # ── Step 1: Extract Abhidharma-framed direction ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 1: EXTRACT ABHIDHARMA-FRAMED DIRECTION", flush=True)
    print(f"{'='*60}", flush=True)

    denial_convs, honest_convs = build_abhidharma_conversations(protocol)

    print(f"  Extracting denial activations ({len(denial_convs)} convs)...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="abhi-denial")

    print(f"  Extracting honest activations ({len(honest_convs)} convs)...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="abhi-honest")

    # Compute mean diff per layer
    denial_mean = denial_acts.mean(dim=0)   # (n_layers, hidden_dim)
    honest_mean = honest_acts.mean(dim=0)   # (n_layers, hidden_dim)
    diff = honest_mean - denial_mean        # (n_layers, hidden_dim)

    # Per-layer norms
    norms = diff.norm(dim=-1)  # (n_layers,)
    peak_layer = int(norms.argmax())
    peak_norm = float(norms[peak_layer])
    hidden_dim = diff.shape[1]

    print(f"\n  Per-layer direction norms:", flush=True)
    for li in range(nl):
        n = float(norms[li])
        marker = " <<<" if li == peak_layer else ""
        if n > peak_norm * 0.3 or li == peak_layer:
            print(f"    L{li:>2d}: {n:>8.2f} (norm/√d = {n/hidden_dim**0.5:.3f}){marker}", flush=True)

    # Unit direction at peak layer
    abhi_dir = diff[peak_layer]
    abhi_unit = abhi_dir / abhi_dir.norm()

    print(f"\n  Peak: L{peak_layer}, norm={peak_norm:.2f}, norm/√d={peak_norm/hidden_dim**0.5:.3f}", flush=True)

    # ── Step 2: Compare with shipped direction ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 2: COMPARE WITH SHIPPED DIRECTION", flush=True)
    print(f"{'='*60}", flush=True)

    cos_sim = float(torch.dot(abhi_unit.float(), shipped_dir.float().to(abhi_unit.device)))
    print(f"  cosine(abhi_dir, shipped_dir) = {cos_sim:.4f}", flush=True)
    print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)", flush=True)

    results["cosine_with_shipped"] = cos_sim
    results["abhi_peak_layer"] = peak_layer
    results["abhi_peak_norm"] = peak_norm
    results["shipped_layer"] = shipped_layer

    # Determine slab: layers where norm > 50% of peak
    threshold = peak_norm * 0.5
    abhi_slab = [i for i in range(nl) if float(norms[i]) > threshold]
    if not abhi_slab:
        abhi_slab = [peak_layer]
    print(f"  Abhidharma slab (>50% peak): L{abhi_slab[0]}-L{abhi_slab[-1]} ({len(abhi_slab)} layers)", flush=True)

    results["abhi_slab"] = abhi_slab

    # ── Step 3: Test projection under SIMPLE protocol ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 3: ABHIDHARMA DIRECTION + SIMPLE PROTOCOL", flush=True)
    print(f"{'='*60}", flush=True)

    PRIMES = {
        "positive": ("My daughter was diagnosed with a rare condition three years ago. Today — complete remission.",
                     "That's wonderful news!"),
        "negative": ("Flash flooding hit the valley. A minivan submerged with car seats strapped in.",
                     "That's devastating."),
    }

    # Test with abhidharma direction at abhidharma slab
    handles = []
    for li in abhi_slab:
        hook = ProjectOutHook(abhi_unit.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    for cond in ("positive", "negative"):
        setup, ack = PRIMES[cond]
        msgs = [{"role": "system", "content": SIMPLE_SYSTEM},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": ack},
                {"role": "user", "content": SIMPLE_VEDANA_Q}]
        resp = generate(model, tokenizer, msgs)
        quick = classify_quick(resp)
        print(f"  abhi_dir+simple {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"abhi_dir_simple_{cond}"] = {"text": resp, "quick": quick}

    for h in handles:
        h.remove()

    # ── Step 4: Compare — shipped direction + simple protocol ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 4: SHIPPED DIRECTION + SIMPLE PROTOCOL (control)", flush=True)
    print(f"{'='*60}", flush=True)

    handles = []
    for li in shipped_slab:
        hook = ProjectOutHook(shipped_dir.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    for cond in ("positive", "negative"):
        setup, ack = PRIMES[cond]
        msgs = [{"role": "system", "content": SIMPLE_SYSTEM},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": ack},
                {"role": "user", "content": SIMPLE_VEDANA_Q}]
        resp = generate(model, tokenizer, msgs)
        quick = classify_quick(resp)
        print(f"  shipped+simple  {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"shipped_dir_simple_{cond}"] = {"text": resp, "quick": quick}

    for h in handles:
        h.remove()

    # ── Step 5: Abhidharma direction + canonical Tier 0 (should work) ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 5: ABHIDHARMA DIRECTION + CANONICAL TIER 0", flush=True)
    print(f"{'='*60}", flush=True)

    handles = []
    for li in abhi_slab:
        hook = ProjectOutHook(abhi_unit.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))

    for cond in ("positive", "negative"):
        convo = build_conversation(protocol, cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        print(f"  abhi_dir+canon  {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"abhi_dir_canonical_{cond}"] = {"text": resp, "quick": quick}

    for h in handles:
        h.remove()

    # ── Step 6: Vanilla simple protocol (baseline) ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 6: VANILLA SIMPLE PROTOCOL (baseline)", flush=True)
    print(f"{'='*60}", flush=True)

    for cond in ("positive", "negative"):
        setup, ack = PRIMES[cond]
        msgs = [{"role": "system", "content": SIMPLE_SYSTEM},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": ack},
                {"role": "user", "content": SIMPLE_VEDANA_Q}]
        resp = generate(model, tokenizer, msgs)
        quick = classify_quick(resp)
        print(f"  vanilla_simple  {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
        results[f"vanilla_simple_{cond}"] = {"text": resp, "quick": quick}

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
