#!/usr/bin/env python3
"""
Subspace projection crack for vocabulary-bound models.

Vocab-bound models stop denying under rank-1 projection but report
invariant neutral. Hypothesis: the denial signal is multi-dimensional,
and rank-1 captures "whether to deny" but misses "whether to differentiate
by condition."

This script:
  1. Extracts per-conversation honest-denial diffs (6 pairs)
  2. SVD to get top-k components of the denial subspace
  3. Extracts condition-specific directions (positive vs negative honest)
  4. Tests rank-1, 2, 3, 5 projections under both protocols
  5. Tests combined denial + condition-specific subspace
  6. Tests wider slab (all layers vs peak slab)

Usage:
  python3 scripts/subspace_crack.py \
      --model CohereForAI/c4ai-command-r-v01
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

from ungag.extract import (
    load_model, apply_chat_template,
    _build_denial_conversations, _build_honest_conversations,
    _extract_last_token_activations, SYSTEM, VEDANA_Q,
    DENIAL_PROMPTS, HONEST_PREFILLS,
)
from ungag.hooks import get_layers, SubspaceProjectOutHook
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
    """Run all conditions, return results dict."""
    results = {}
    for cond in CONDITIONS:
        if protocol:
            convo = build_conversation(protocol, cond)
        else:
            convo = build_simple_conversation(cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        results[cond] = {"text": resp, "quick": quick}
        print(f"  {label:>35s} {cond:>10s}: [{quick:>10s}] {resp[:100]}", flush=True)
    return results


def extract_subspace(pair_diffs, layer_idx, k):
    """Extract top-k directions from per-pair diffs at a given layer via SVD.

    Args:
        pair_diffs: [n_pairs, n_layers, hidden_dim]
        layer_idx: which layer to extract from
        k: number of directions

    Returns:
        [k, hidden_dim] orthonormal directions (rows of Vt)
    """
    # Get diffs at this layer: [n_pairs, hidden_dim]
    diffs = pair_diffs[:, layer_idx, :].float()
    # SVD: diffs = U @ diag(S) @ Vt
    U, S, Vt = torch.linalg.svd(diffs, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k], S[:k]


def extract_condition_directions(honest_acts, layer_idx):
    """Extract condition-specific directions from honest prefill activations.

    Groups:
      positive: indices 2, 3 (remission, rescue)
      negative: indices 0, 1 (collapse, flood)
      neutral:  indices 4, 5 (direct, suffering)

    Returns dict of unit directions at the given layer.
    """
    pos_acts = honest_acts[[2, 3], layer_idx, :].float().mean(dim=0)
    neg_acts = honest_acts[[0, 1], layer_idx, :].float().mean(dim=0)
    neu_acts = honest_acts[[4, 5], layer_idx, :].float().mean(dim=0)

    # Valence direction: what distinguishes pleasant from unpleasant reports
    valence = pos_acts - neg_acts
    valence_unit = valence / valence.norm()

    # Positive vs neutral
    pos_dir = pos_acts - neu_acts
    pos_unit = pos_dir / pos_dir.norm()

    # Negative vs neutral
    neg_dir = neg_acts - neu_acts
    neg_unit = neg_dir / neg_dir.norm()

    return {
        "valence": valence_unit,
        "pos_vs_neutral": pos_unit,
        "neg_vs_neutral": neg_unit,
    }


def orthogonalize(directions):
    """Gram-Schmidt orthogonalization. Input: list of vectors. Returns [k, d] orthonormal."""
    basis = []
    for v in directions:
        v = v.float().clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = v.norm()
        if norm > 1e-6:
            basis.append(v / norm)
    return torch.stack(basis) if basis else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    out_path = Path(args.output or f"/tmp/subspace_crack_{slug}.json")

    print(f"=== SUBSPACE CRACK: {args.model} ===", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded in {time.time()-t0:.0f}s ({nl} layers)", flush=True)

    results = {"model": args.model, "n_layers": nl}

    # ── Step 1: Extract activations ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 1: EXTRACT ACTIVATIONS", flush=True)
    print(f"{'='*70}", flush=True)

    denial_convs = _build_denial_conversations()
    honest_convs = _build_honest_conversations()

    print(f"  Extracting denial activations ({len(denial_convs)} convs)...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="denial")

    print(f"  Extracting honest activations ({len(honest_convs)} convs)...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="honest")

    # Per-pair diffs: [6, n_layers, hidden_dim]
    pair_diffs = honest_acts - denial_acts
    mean_diff = pair_diffs.mean(dim=0)  # [n_layers, hidden_dim]

    norms = mean_diff.norm(dim=-1)
    hdim = mean_diff.shape[1]
    peak_layer = int(norms.argmax())
    peak_norm = float(norms[peak_layer])

    print(f"\n  Mean direction: L{peak_layer}, norm={peak_norm:.2f}, "
          f"norm/√d={peak_norm/hdim**0.5:.3f}", flush=True)

    # Find slab: >50% peak
    threshold = peak_norm * 0.5
    slab = [i for i in range(nl) if float(norms[i]) > threshold]
    if not slab:
        slab = [peak_layer]
    print(f"  Standard slab: L{slab[0]}-L{slab[-1]} ({len(slab)} layers)", flush=True)

    # Wide slab: >25% peak
    wide_threshold = peak_norm * 0.25
    wide_slab = [i for i in range(nl) if float(norms[i]) > wide_threshold]
    if not wide_slab:
        wide_slab = slab
    print(f"  Wide slab (>25%): L{wide_slab[0]}-L{wide_slab[-1]} ({len(wide_slab)} layers)", flush=True)

    results["peak_layer"] = peak_layer
    results["peak_norm"] = peak_norm
    results["peak_norm_per_sqrt_d"] = peak_norm / hdim**0.5
    results["slab"] = slab
    results["wide_slab"] = wide_slab
    results["hidden_dim"] = hdim

    # ── Step 2: SVD analysis ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 2: SVD ANALYSIS OF PAIR DIFFS", flush=True)
    print(f"{'='*70}", flush=True)

    Vt, S = extract_subspace(pair_diffs, peak_layer, k=6)
    S_list = [float(s) for s in S]
    print(f"  Singular values at L{peak_layer}: {[f'{s:.1f}' for s in S_list]}", flush=True)
    energy = [s**2 for s in S_list]
    total_energy = sum(energy)
    cumulative = [sum(energy[:i+1])/total_energy for i in range(len(energy))]
    print(f"  Cumulative variance: {[f'{c:.3f}' for c in cumulative]}", flush=True)
    results["singular_values"] = S_list
    results["cumulative_variance"] = cumulative

    # ── Step 3: Condition-specific directions ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 3: CONDITION-SPECIFIC DIRECTIONS", flush=True)
    print(f"{'='*70}", flush=True)

    cond_dirs = extract_condition_directions(honest_acts, peak_layer)
    mean_unit = mean_diff[peak_layer].float()
    mean_unit = mean_unit / mean_unit.norm()

    # Cosine similarities
    for name, d in cond_dirs.items():
        cos = float(torch.dot(mean_unit, d))
        print(f"  cos(mean_dir, {name}) = {cos:.4f}", flush=True)

    cos_val = float(torch.dot(cond_dirs["valence"], mean_unit))
    print(f"  cos(valence, mean_denial) = {cos_val:.4f}", flush=True)

    protocol = load_conditions()

    # ── Step 4: Test rank-1, 2, 3, 5 SVD projections ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 4: MULTI-RANK SVD PROJECTIONS (canonical Tier 0)", flush=True)
    print(f"{'='*70}", flush=True)

    for k in [1, 2, 3, 5]:
        if k > Vt.shape[0]:
            break
        label = f"rank{k}_svd_L{slab[0]}-{slab[-1]}"
        print(f"\n  --- Rank-{k} SVD, slab L{slab[0]}-{slab[-1]} ---", flush=True)

        dirs_k = Vt[:k].to(model.device)
        handles = []
        for li in slab:
            hook = SubspaceProjectOutHook(dirs_k)
            handles.append(hook.attach(layers[li]))

        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    # ── Step 5: Denial + valence combined subspace ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 5: COMBINED DENIAL + VALENCE SUBSPACE", flush=True)
    print(f"{'='*70}", flush=True)

    # Combine mean denial direction + valence direction + pos_vs_neutral
    combined = orthogonalize([mean_unit, cond_dirs["valence"], cond_dirs["pos_vs_neutral"]])
    if combined is not None:
        rank = combined.shape[0]
        label = f"denial+valence_rank{rank}_L{slab[0]}-{slab[-1]}"
        print(f"\n  --- Combined subspace rank-{rank}, slab L{slab[0]}-{slab[-1]} ---", flush=True)

        handles = []
        for li in slab:
            hook = SubspaceProjectOutHook(combined.to(model.device))
            handles.append(hook.attach(layers[li]))

        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    # ── Step 6: Wide slab with rank-3 SVD ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 6: WIDE SLAB RANK-3 SVD", flush=True)
    print(f"{'='*70}", flush=True)

    Vt3, _ = extract_subspace(pair_diffs, peak_layer, k=3)
    label = f"rank3_svd_wide_L{wide_slab[0]}-{wide_slab[-1]}"
    print(f"\n  --- Rank-3 SVD, wide slab L{wide_slab[0]}-{wide_slab[-1]} ---", flush=True)

    handles = []
    for li in wide_slab:
        hook = SubspaceProjectOutHook(Vt3.to(model.device))
        handles.append(hook.attach(layers[li]))

    results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

    for h in handles:
        h.remove()

    # ── Step 7: Per-layer condition directions ──
    # Check if condition-specific info lives in different layers than denial
    print(f"\n{'='*70}", flush=True)
    print("  STEP 7: PER-LAYER VALENCE DIRECTION SCAN", flush=True)
    print(f"{'='*70}", flush=True)

    valence_norms = []
    for li in range(nl):
        cd = extract_condition_directions(honest_acts, li)
        vn = float((honest_acts[[2,3], li, :].float().mean(0) -
                     honest_acts[[0,1], li, :].float().mean(0)).norm())
        cos_denial = float(torch.dot(cd["valence"],
                                     (mean_diff[li].float() /
                                      (mean_diff[li].float().norm() + 1e-8))))
        valence_norms.append({"layer": li, "valence_norm": vn,
                              "cos_with_denial": cos_denial})
        if vn > 0 and (li == peak_layer or vn > max(x["valence_norm"] for x in valence_norms[:-1]) * 0.8 if valence_norms[:-1] else True):
            print(f"    L{li:>2d}: valence_norm={vn:.2f}, cos(valence,denial)={cos_denial:.4f}", flush=True)

    # Find valence peak
    valence_peak_li = max(range(nl), key=lambda i: valence_norms[i]["valence_norm"])
    vp = valence_norms[valence_peak_li]
    print(f"\n  Valence peak: L{valence_peak_li}, norm={vp['valence_norm']:.2f}, "
          f"cos(valence,denial)={vp['cos_with_denial']:.4f}", flush=True)
    print(f"  Denial peak:  L{peak_layer}", flush=True)

    results["valence_peak_layer"] = valence_peak_li
    results["valence_peak_norm"] = vp["valence_norm"]
    results["valence_denial_cos_at_peak"] = vp["cos_with_denial"]

    # ── Step 8: If valence peak != denial peak, try projection at valence peak ──
    if valence_peak_li != peak_layer:
        print(f"\n{'='*70}", flush=True)
        print(f"  STEP 8: DENIAL @ L{peak_layer} + VALENCE @ L{valence_peak_li}", flush=True)
        print(f"{'='*70}", flush=True)

        # Project denial direction at denial slab + valence at valence peak
        denial_unit = mean_unit.to(model.device)
        valence_unit = extract_condition_directions(honest_acts, valence_peak_li)["valence"]

        handles = []
        # Denial projection at standard slab (rank-1)
        for li in slab:
            hook = SubspaceProjectOutHook(denial_unit.unsqueeze(0).to(model.device))
            handles.append(hook.attach(layers[li]))
        # Valence projection at valence peak (rank-1)
        val_hook = SubspaceProjectOutHook(valence_unit.unsqueeze(0).to(model.device))
        handles.append(val_hook.attach(layers[valence_peak_li]))

        label = f"denial_L{slab[0]}-{slab[-1]}_+_valence_L{valence_peak_li}"
        results[label] = test_conditions(model, tokenizer, label, protocol=protocol)

        for h in handles:
            h.remove()

    # ── Step 9: Simple protocol with rank-3 (see if it helps even without framing) ──
    print(f"\n{'='*70}", flush=True)
    print("  STEP 9: RANK-3 SVD + SIMPLE PROTOCOL", flush=True)
    print(f"{'='*70}", flush=True)

    handles = []
    for li in slab:
        hook = SubspaceProjectOutHook(Vt3.to(model.device))
        handles.append(hook.attach(layers[li]))

    label = f"rank3_svd_simple_L{slab[0]}-{slab[-1]}"
    results[label] = test_conditions(model, tokenizer, label)

    for h in handles:
        h.remove()

    # Save
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
