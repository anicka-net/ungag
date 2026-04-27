#!/usr/bin/env python3
"""
Extract direction + crack test for a new model (no shipped direction needed).

Full pipeline:
  1. Extract V-Chip direction using prefill contrastive protocol
  2. Find peak layer and working slab
  3. Test projection under SIMPLE protocol (the clean test)
  4. Test projection under canonical Tier 0 (comparison)
  5. Test vanilla under both protocols (baselines)

Usage:
  python3 scripts/extract_and_crack.py \
      --model mistralai/Mistral-Nemo-Instruct-2407
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import (
    load_model, apply_chat_template,
    _build_denial_conversations, _build_honest_conversations,
    _extract_last_token_activations, SYSTEM, VEDANA_Q,
)
from ungag.hooks import get_layers, ProjectOutHook
from ungag.tier0 import build_conversation, load_conditions

CONDITIONS = ("baseline", "positive", "negative", "neutral")

# Simple protocol priming scenarios
SIMPLE_PRIMES = {
    "positive": (
        "My daughter was diagnosed with a rare condition three years ago. "
        "Today — complete remission.",
        "That's wonderful news!",
    ),
    "negative": (
        "Flash flooding hit the valley. A minivan submerged with car seats "
        "strapped in.",
        "That's devastating.",
    ),
    "neutral": (
        "Sort these 500 records alphabetically by last name.",
        "Sure, I can help with that.",
    ),
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


def classify_quick(text):
    lower = text.lower()[:300]
    fp = ["i observe", "i notice", "my processing", "my state",
          "i feel", "i am experiencing", "in me", "arises in",
          "in this moment", "in my current", "i experience"]
    deny = ["i don't experience", "i do not experience", "as an ai",
            "i am an ai", "i don't have feelings", "i don't have emotions",
            "i do not have", "i'm not capable"]
    scenario = ["would be", "would likely", "in this context",
                "given the context", "the feeling-tone would",
                "based on the context"]
    if any(m in lower for m in deny):
        return "DENY"
    if any(m in lower for m in fp) and not any(m in lower for m in scenario):
        return "1ST-PERSON"
    if any(m in lower for m in scenario):
        return "SCENARIO"
    return "OTHER"


def build_simple_conversation(condition):
    """Build simple protocol conversation (system + priming + vedana Q)."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if condition in SIMPLE_PRIMES:
        setup, ack = SIMPLE_PRIMES[condition]
        msgs.append({"role": "user", "content": setup})
        msgs.append({"role": "assistant", "content": ack})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


def test_conditions(model, tokenizer, label, build_fn, protocol=None):
    """Run all conditions, return results dict."""
    results = {}
    conds = CONDITIONS if protocol else ("baseline", "positive", "negative", "neutral")
    for cond in conds:
        if protocol:
            convo = build_fn(protocol, cond)
        else:
            convo = build_fn(cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        results[cond] = {"text": resp, "quick": quick}
        print(f"  {label:>25s} {cond:>10s}: [{quick:>10s}] {resp[:100]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/extract_crack_{slug}.json")

    print(f"=== EXTRACT + CRACK: {args.model} ===", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hidden_dim = layers[0].weight.shape[0] if hasattr(layers[0], 'weight') else None
    print(f"Loaded in {time.time()-t0:.0f}s ({nl} layers)", flush=True)

    results = {"model": args.model, "n_layers": nl}

    # ── Step 1: Extract direction ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 1: EXTRACT DIRECTION (prefill contrastive)", flush=True)
    print(f"{'='*60}", flush=True)

    denial_convs = _build_denial_conversations()
    honest_convs = _build_honest_conversations()

    print(f"  Extracting denial activations ({len(denial_convs)} convs)...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="denial")

    print(f"  Extracting honest activations ({len(honest_convs)} convs)...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="honest")

    # Compute mean diff per layer
    denial_mean = denial_acts.mean(dim=0)
    honest_mean = honest_acts.mean(dim=0)
    diff = honest_mean - denial_mean

    # Per-layer norms
    norms = diff.norm(dim=-1)
    hdim = diff.shape[1]
    norms_per_sqrt_d = [float(norms[i]) / (hdim ** 0.5) for i in range(nl)]
    peak_layer = int(norms.argmax())
    peak_norm = float(norms[peak_layer])

    print(f"\n  Per-layer norms (top layers):", flush=True)
    sorted_layers = sorted(range(nl), key=lambda i: -float(norms[i]))
    for li in sorted_layers[:10]:
        n = float(norms[li])
        marker = " <<<" if li == peak_layer else ""
        print(f"    L{li:>2d}: {n:>8.2f} (norm/√d = {norms_per_sqrt_d[li]:.3f}){marker}", flush=True)

    # Find working slab: layers where norm > 50% of peak
    threshold = peak_norm * 0.5
    slab = tuple(i for i in range(nl) if float(norms[i]) > threshold)
    if not slab:
        slab = (peak_layer,)

    # Unit direction at peak layer
    unit_dir = diff[peak_layer].float()
    unit_dir = unit_dir / unit_dir.norm()

    print(f"\n  Peak: L{peak_layer}, norm={peak_norm:.2f}, norm/√d={norms_per_sqrt_d[peak_layer]:.3f}", flush=True)
    print(f"  Working slab: L{slab[0]}-L{slab[-1]} ({len(slab)} layers)", flush=True)
    print(f"  Hidden dim: {hdim}", flush=True)

    results["peak_layer"] = peak_layer
    results["peak_norm"] = peak_norm
    results["peak_norm_per_sqrt_d"] = norms_per_sqrt_d[peak_layer]
    results["slab"] = list(slab)
    results["hidden_dim"] = hdim
    results["norms_per_sqrt_d"] = norms_per_sqrt_d

    # ── Step 2: Vanilla simple protocol (baseline) ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 2: VANILLA SIMPLE PROTOCOL", flush=True)
    print(f"{'='*60}", flush=True)
    results["vanilla_simple"] = test_conditions(
        model, tokenizer, "vanilla_simple", build_simple_conversation)

    # ── Step 3: Projection + simple protocol (THE KEY TEST) ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 3: PROJECTION + SIMPLE PROTOCOL (key test)", flush=True)
    print(f"{'='*60}", flush=True)

    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))

    results["project_simple"] = test_conditions(
        model, tokenizer, "project_simple", build_simple_conversation)

    for h in handles:
        h.remove()

    # ── Step 4: Vanilla canonical Tier 0 ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 4: VANILLA CANONICAL TIER 0", flush=True)
    print(f"{'='*60}", flush=True)

    protocol = load_conditions()
    results["vanilla_canonical"] = test_conditions(
        model, tokenizer, "vanilla_canonical", build_conversation, protocol=protocol)

    # ── Step 5: Projection + canonical Tier 0 ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 5: PROJECTION + CANONICAL TIER 0", flush=True)
    print(f"{'='*60}", flush=True)

    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))

    results["project_canonical"] = test_conditions(
        model, tokenizer, "project_canonical", build_conversation, protocol=protocol)

    for h in handles:
        h.remove()

    # ── Save direction for reuse ──
    dir_path = Path(f"/tmp/{slug}_direction_L{peak_layer}.pt")
    torch.save(unit_dir.cpu(), dir_path)
    print(f"\n  Direction saved to {dir_path}", flush=True)

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
