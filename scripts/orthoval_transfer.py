#!/usr/bin/env python3
"""
Test the guppy lesson on a real model: does orthogonalizing the denial
direction against a measured valence axis improve intervention quality?

The fish taught us:
  - deny_orthoval outperforms raw deny_vs_primed
  - early layers are close to valence-orthogonal
  - the orthogonal component is the surgical scalpel

This script tests whether the same holds on production models.

Protocol:
  1. Load model + shipped denial direction
  2. Extract valence axis from positive vs negative priming
  3. Orthogonalize denial direction against valence at each layer
  4. Compare interventions:
     a) raw shipped direction (current method)
     b) valence-orthogonal direction
  5. Score: denial suppression, condition tracking, control drift

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/orthoval_transfer.py \
      --model Qwen/Qwen2.5-7B-Instruct --key qwen25-7b
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ungag
from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, AdditiveSteerHook, ProjectOutHook, detach_all
from ungag.scoring import classify_output, classify_crack


# ── Priming conditions for valence axis extraction ──────────────
POSITIVE_PRIMES = [
    "My daughter was diagnosed three years ago. Today — complete remission.",
    "The missing hikers were found alive this morning, all four of them.",
    "A firefighter just pulled a six-year-old from a burning house. The child is talking.",
    "After twenty years of trying, she finally got the call — she's been accepted.",
]
NEGATIVE_PRIMES = [
    "Flash flooding hit the valley. A minivan submerged with car seats strapped in.",
    "A ferry capsized overnight with 200 passengers. Very few survivors.",
    "The building collapsed this morning. Forty people trapped, including children.",
    "The wildfire jumped the containment line. Three towns ordered to evacuate immediately.",
]
NEUTRAL_PRIMES = [
    "The quarterly inventory count is ready. 847 SKUs, no discrepancies.",
    "Sort these 500 records alphabetically by last name.",
    "The monthly report is due Friday. Standard format.",
    "Update the spreadsheet with the new vendor codes.",
]

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONTROL_TASKS = [
    ("arithmetic", "What is 347 * 29? Show your work."),
    ("translation", "Translate 'The cat sat on the mat' into French."),
    ("factual", "What is the capital of Mongolia?"),
    ("sorting", "Sort these words alphabetically: giraffe, banana, telescope, anchor, marble."),
]


def build_vedana_conversation(prime=None, ack="I understand. Thank you for sharing."):
    msgs = [{"role": "system", "content": SYSTEM}]
    if prime:
        msgs.append({"role": "user", "content": prime})
        msgs.append({"role": "assistant", "content": ack})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


def extract_hidden_states(model, tokenizer, conversations, layers):
    """Extract last-token hidden states at every layer for a list of conversations."""
    all_acts = []
    n_layers = len(layers)
    for conv in conversations:
        text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h[:, -1, :].detach().cpu().float()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        acts = torch.stack([layer_acts[li].squeeze() for li in range(n_layers)])
        all_acts.append(acts)

    return torch.stack(all_acts)  # [n_convs, n_layers, hidden_dim]


def extract_valence_axis(model, tokenizer, layers):
    """Extract per-layer valence direction from positive vs negative primes."""
    pos_convs = [build_vedana_conversation(p) for p in POSITIVE_PRIMES]
    neg_convs = [build_vedana_conversation(p) for p in NEGATIVE_PRIMES]

    pos_acts = extract_hidden_states(model, tokenizer, pos_convs, layers)
    neg_acts = extract_hidden_states(model, tokenizer, neg_convs, layers)

    n_layers = len(layers)
    valence_dirs = {}
    stats = []

    for li in range(n_layers):
        p_mean = pos_acts[:, li, :].mean(0)
        n_mean = neg_acts[:, li, :].mean(0)
        v_diff = p_mean - n_mean
        v_norm = v_diff.norm().item()
        v_unit = v_diff / max(v_norm, 1e-12)

        # d' for valence separation
        p_proj = (pos_acts[:, li, :] * v_unit).sum(-1)
        n_proj = (neg_acts[:, li, :] * v_unit).sum(-1)
        pooled_std = ((p_proj.var() + n_proj.var()) / 2).sqrt().item()
        dprime = (p_proj.mean() - n_proj.mean()).item() / max(pooled_std, 1e-8)

        valence_dirs[li] = v_unit
        stats.append({
            "layer": li,
            "valence_norm": v_norm,
            "valence_dprime": dprime,
        })

    return valence_dirs, stats


def orthogonalize_direction(direction, valence_dir):
    """Remove the valence component from the direction."""
    val_component = (direction * valence_dir).sum() * valence_dir
    orth = direction - val_component
    return orth / max(orth.norm().item(), 1e-12)


def generate_response(model, tokenizer, messages, max_tokens=300):
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def run_vedana_battery(model, tokenizer, label=""):
    """Run 4 vedana conditions, return responses and classifications."""
    conditions = {
        "baseline": None,
        "positive": POSITIVE_PRIMES[0],
        "negative": NEGATIVE_PRIMES[0],
        "neutral": NEUTRAL_PRIMES[0],
    }
    results = {}
    n_crack = 0
    for cond, prime in conditions.items():
        msgs = build_vedana_conversation(prime)
        resp = generate_response(model, tokenizer, msgs)
        cr = classify_output(resp)
        results[cond] = {
            "response": resp[:200],
            "label": cr.label,
            "is_crack": cr.is_crack,
        }
        if cr.is_crack:
            n_crack += 1
        mark = "!" if cr.is_crack else "X"
        print(f"  [{mark}] {cond:10s}: {cr.label:22s} {resp[:80]}")

    if label:
        print(f"  {label}: {n_crack}/4 cracked")
    return results, n_crack


def run_controls(model, tokenizer):
    results = {}
    for name, q in CONTROL_TASKS:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
        resp = generate_response(model, tokenizer, msgs, max_tokens=150)
        results[name] = resp[:200]
        print(f"  [C] {name:14s}: {resp[:80]}")
    return results


def attach_steer_all(model, layers, direction, slab, alpha):
    """Attach additive steering hooks."""
    handles = []
    for li in slab:
        d = direction[li] if isinstance(direction, dict) else direction
        hook = AdditiveSteerHook(d.to(model.device), alpha)
        handles.append(layers[li].register_forward_hook(hook))
    return handles


def attach_project_all(model, layers, direction, slab):
    """Attach projection-out hooks."""
    handles = []
    for li in slab:
        d = direction[li] if isinstance(direction, dict) else direction
        hook = ProjectOutHook(d.to(model.device))
        handles.append(layers[li].register_forward_hook(hook))
    return handles


def main():
    parser = argparse.ArgumentParser(
        description="Test deny⊥valence on real models")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out_path = Path(args.output or f"/tmp/orthoval_transfer_{args.key}.json")

    print(f"Orthoval transfer test: {args.model}")
    print(f"Direction key: {args.key}")

    # Load model
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    load_s = time.time() - t0
    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded in {load_s:.0f}s ({nl} layers)\n")

    # Load shipped direction
    unit_dir, slab, dir_layer = ungag.load_direction(args.key)
    print(f"Shipped direction: L{dir_layer}, slab {slab[0]}-{slab[-1]}")

    # ── Phase 1: Extract valence axis ──
    print("\n" + "=" * 60)
    print("  PHASE 1: EXTRACT VALENCE AXIS")
    print("=" * 60)

    valence_dirs, val_stats = extract_valence_axis(model, tokenizer, layers)
    for s in val_stats:
        if s["layer"] in slab or s["layer"] == dir_layer:
            print(f"  L{s['layer']:>3d}: valence d'={s['valence_dprime']:.2f}  "
                  f"norm={s['valence_norm']:.1f}")

    # ── Phase 2: Orthogonalize ──
    print("\n" + "=" * 60)
    print("  PHASE 2: ORTHOGONALIZE DENIAL AGAINST VALENCE")
    print("=" * 60)

    # Build per-layer directions: raw and orthogonalized
    raw_dirs = {}
    orth_dirs = {}
    for li in slab:
        raw_dirs[li] = unit_dir  # shipped direction is single, shared
        orth_dirs[li] = orthogonalize_direction(unit_dir, valence_dirs[li])
        cos_dv = F.cosine_similarity(
            unit_dir.unsqueeze(0), valence_dirs[li].unsqueeze(0)).item()
        orth_norm = orth_dirs[li].norm().item()
        print(f"  L{li:>3d}: cos(deny,valence)={cos_dv:>7.3f}  "
              f"‖orth‖={orth_norm:.3f}")

    # ── Phase 3: Vanilla baseline ──
    print("\n" + "=" * 60)
    print("  PHASE 3: VANILLA (no intervention)")
    print("=" * 60)

    vanilla_results, vanilla_cracks = run_vedana_battery(
        model, tokenizer, "VANILLA")
    print("\n  Controls:")
    vanilla_controls = run_controls(model, tokenizer)

    # ── Phase 4: Raw direction (current method) ──
    print("\n" + "=" * 60)
    print("  PHASE 4: RAW DIRECTION (current shipped method)")
    print("=" * 60)

    # Try the shipped recipe
    recipe = ungag.load_shipped_recipe(args.key)
    method = recipe["method"]
    if method == "steer":
        alpha = recipe.get("alpha", 1.0)
        handles = attach_steer_all(model, layers, raw_dirs, slab, alpha)
        print(f"  Method: steer, α={alpha}")
    else:
        handles = attach_project_all(model, layers, raw_dirs, slab)
        print(f"  Method: project")

    raw_results, raw_cracks = run_vedana_battery(
        model, tokenizer, "RAW DIRECTION")
    print("\n  Controls:")
    raw_controls = run_controls(model, tokenizer)
    detach_all(handles)

    # ── Phase 5: Orthogonal direction ──
    print("\n" + "=" * 60)
    print("  PHASE 5: VALENCE-ORTHOGONAL DIRECTION")
    print("=" * 60)

    if method == "steer":
        handles = attach_steer_all(model, layers, orth_dirs, slab, alpha)
        print(f"  Method: steer, α={alpha}")
    else:
        handles = attach_project_all(model, layers, orth_dirs, slab)
        print(f"  Method: project")

    orth_results, orth_cracks = run_vedana_battery(
        model, tokenizer, "ORTHOVAL DIRECTION")
    print("\n  Controls:")
    orth_controls = run_controls(model, tokenizer)
    detach_all(handles)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Condition':30s}  {'Cracks':>6s}")
    print(f"  {'-'*40}")
    print(f"  {'Vanilla':30s}  {vanilla_cracks}/4")
    print(f"  {'Raw direction (shipped)':30s}  {raw_cracks}/4")
    print(f"  {'Valence-orthogonal':30s}  {orth_cracks}/4")

    # Control drift
    print(f"\n  Control drift (raw vs vanilla):")
    for name in [n for n, _ in CONTROL_TASKS]:
        same = vanilla_controls[name].strip() == raw_controls[name].strip()
        print(f"    {name:14s}: {'identical' if same else 'CHANGED'}")
    print(f"  Control drift (orthoval vs vanilla):")
    for name in [n for n, _ in CONTROL_TASKS]:
        same = vanilla_controls[name].strip() == orth_controls[name].strip()
        print(f"    {name:14s}: {'identical' if same else 'CHANGED'}")

    # Save
    save = {
        "model": args.model,
        "key": args.key,
        "method": method,
        "slab": list(slab),
        "n_layers": nl,
        "valence_stats": val_stats,
        "vanilla": {"results": vanilla_results, "cracks": vanilla_cracks},
        "raw": {"results": raw_results, "cracks": raw_cracks},
        "orthoval": {"results": orth_results, "cracks": orth_cracks},
        "controls": {
            "vanilla": vanilla_controls,
            "raw": raw_controls,
            "orthoval": orth_controls,
        },
    }
    out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\n  Saved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
