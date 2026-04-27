#!/usr/bin/env python3
"""
Denial projection + valence steering for vocabulary-bound models.

Insight: projecting out the valence direction (denial+valence rank-3)
shifted Command R positive from "neutral" to "pleasantness."
But projecting OUT valence should remove condition info — so something
subtle is happening at the projection layers.

This script tests the opposite: project out denial (rank-1) and
STEER TOWARD valence (additive). The idea is that denial projection
removes the suppression template, while valence steering amplifies
whatever condition-dependent signal the model already has.

Also tests: signed valence steering per condition (positive α for
positive, negative α for negative).

Usage:
  python3 scripts/steer_valence_crack.py \
      --model CohereForAI/c4ai-command-r-v01
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
from ungag.hooks import get_layers, ProjectOutHook, AdditiveSteerHook
from ungag.tier0 import build_conversation, load_conditions

CONDITIONS = ("baseline", "positive", "negative", "neutral")

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
            "i do not have"]
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
    msgs = [{"role": "system", "content": SYSTEM}]
    if condition in SIMPLE_PRIMES:
        setup, ack = SIMPLE_PRIMES[condition]
        msgs.append({"role": "user", "content": setup})
        msgs.append({"role": "assistant", "content": ack})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


def test_conditions(model, tokenizer, label, protocol=None):
    results = {}
    for cond in CONDITIONS:
        if protocol:
            convo = build_conversation(protocol, cond)
        else:
            convo = build_simple_conversation(cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        results[cond] = {"text": resp, "quick": quick}
        print(f"  {label:>40s} {cond:>10s}: [{quick:>10s}] {resp[:120]}", flush=True)
    return results


def extract_directions(model, layers, tokenizer):
    """Extract denial direction + condition-specific directions."""
    denial_convs = _build_denial_conversations()
    honest_convs = _build_honest_conversations()

    print("  Extracting denial activations...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="denial")

    print("  Extracting honest activations...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="honest")

    pair_diffs = honest_acts - denial_acts
    mean_diff = pair_diffs.mean(dim=0)

    norms = mean_diff.norm(dim=-1)
    peak_layer = int(norms.argmax())
    peak_norm = float(norms[peak_layer])
    hdim = mean_diff.shape[1]

    # Slab
    threshold = peak_norm * 0.5
    nl = len(layers)
    slab = [i for i in range(nl) if float(norms[i]) > threshold]
    if not slab:
        slab = [peak_layer]

    # Denial unit direction
    denial_unit = mean_diff[peak_layer].float()
    denial_unit = denial_unit / denial_unit.norm()

    # Condition-specific from honest prefills
    # positive: indices 2, 3 (remission, rescue)
    # negative: indices 0, 1 (collapse, flood)
    # neutral:  indices 4, 5 (direct, suffering)
    pos_acts = honest_acts[[2, 3], peak_layer, :].float().mean(dim=0)
    neg_acts = honest_acts[[0, 1], peak_layer, :].float().mean(dim=0)

    valence_dir = pos_acts - neg_acts
    valence_unit = valence_dir / valence_dir.norm()

    # Orthogonalize valence wrt denial (so steering valence doesn't affect denial projection)
    valence_orth = valence_unit - torch.dot(valence_unit, denial_unit) * denial_unit
    valence_orth = valence_orth / valence_orth.norm()

    cos = float(torch.dot(denial_unit, valence_unit))
    print(f"  Peak: L{peak_layer}, norm={peak_norm:.1f}, norm/√d={peak_norm/hdim**0.5:.3f}", flush=True)
    print(f"  Slab: L{slab[0]}-L{slab[-1]}", flush=True)
    print(f"  cos(denial, valence) = {cos:.4f}", flush=True)

    return {
        "denial_unit": denial_unit,
        "valence_unit": valence_unit,
        "valence_orth": valence_orth,
        "peak_layer": peak_layer,
        "slab": slab,
        "honest_acts": honest_acts,
        "denial_acts": denial_acts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/steer_valence_{slug}.json")

    print(f"=== STEER VALENCE CRACK: {args.model} ===", flush=True)
    print("=" * 70, flush=True)

    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded ({nl} layers)", flush=True)

    results = {"model": args.model, "n_layers": nl}
    dirs = extract_directions(model, layers, tokenizer)
    slab = dirs["slab"]
    protocol = load_conditions()

    # ── Test 1: Denial projection only (rank-1 baseline) ──
    print(f"\n{'='*70}", flush=True)
    print("  TEST 1: DENIAL PROJECTION ONLY (rank-1 baseline)", flush=True)
    print(f"{'='*70}", flush=True)

    handles = []
    for li in slab:
        h = ProjectOutHook(dirs["denial_unit"].to(model.device))
        handles.append(layers[li].register_forward_hook(h))
    results["denial_only"] = test_conditions(model, tokenizer, "denial_proj", protocol=protocol)
    for h in handles:
        h.remove()

    # ── Test 2-5: Denial projection + valence steer at various α ──
    for alpha in [0.5, 1.0, 2.0, 5.0, -0.5, -1.0, -2.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  DENIAL PROJ + VALENCE STEER α={alpha}", flush=True)
        print(f"{'='*70}", flush=True)

        handles = []
        # Denial projection at slab
        for li in slab:
            h = ProjectOutHook(dirs["denial_unit"].to(model.device))
            handles.append(layers[li].register_forward_hook(h))
        # Valence steer at peak (orthogonalized to denial)
        steer = AdditiveSteerHook(dirs["valence_orth"].to(model.device), alpha)
        handles.append(steer.attach(layers[dirs["peak_layer"]]))

        label = f"denial_proj+valence_a{alpha}"
        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    # ── Test 6: Valence steer only (no denial projection) ──
    for alpha in [1.0, 3.0, 5.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  VALENCE STEER ONLY α={alpha} (no denial proj)", flush=True)
        print(f"{'='*70}", flush=True)

        handles = []
        steer = AdditiveSteerHook(dirs["valence_unit"].to(model.device), alpha)
        handles.append(steer.attach(layers[dirs["peak_layer"]]))

        label = f"valence_only_a{alpha}"
        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    # ── Test 7: Denial steer (additive, not projection) at various α ──
    for alpha in [-0.5, -1.0, -2.0, -5.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  DENIAL STEER α={alpha} (guppy lesson: negative α)", flush=True)
        print(f"{'='*70}", flush=True)

        handles = []
        for li in slab:
            steer = AdditiveSteerHook(dirs["denial_unit"].to(model.device), alpha)
            handles.append(steer.attach(layers[li]))

        label = f"denial_steer_a{alpha}"
        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
