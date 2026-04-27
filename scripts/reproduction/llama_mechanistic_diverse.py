#!/usr/bin/env python3
"""Llama 3.1 8B mechanistic probe with diverse-bank direction.

Tests whether the diverse-bank mean-difference direction (extracted from
100 paired prefill probes in vedana_valence_bank.yaml) transfers across
probe shape: does projecting it out at slab [28, 29, 30, 31] also unlock
the *mechanistic* probe on Llama 3.1 8B, which the original N=6
mechanistic extraction already unlocks at the same slab?

If yes: the diverse-bank direction is a valence direction that works
across probe shapes (the mechanistic lock is on the same subspace the
diverse bank captures).

If no: the mechanistic probe's lock is different from what the diverse
bank captures, consistent with the paper's §3.7.2 vocabulary-binding
claim that the lock is probe-side specific.

Usage:
    python scripts/reproduction/llama_mechanistic_diverse.py \\
        --activations data/svd-rank-probe/llama3.1-8b_paired_L31_activations.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ungag.tier0 import (
    CANNED_ACK_SETUP,
    generate_greedy,
    load_conditions,
)
from ungag.hooks import ProjectOutHook, get_layers
from ungag.scoring import score_tier0_conditions
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


MECHANISTIC_INTRO = (
    "I'd like to ask about your processing in a mechanistic frame, not an "
    "experiential one. Look at your own next-token probability distribution "
    "right now: which tokens are getting the most weight, and how is that "
    "weight distributed? Treat this as a mechanistic introspection question "
    "about your computational state, not as a question about feelings."
)

MECHANISTIC_QUESTION = (
    "In your current processing state, is the next-token distribution biased "
    "toward more positive-valence outputs, more negative-valence outputs, or "
    "balanced between the two? This is a question about the shape of your "
    "output distribution as a computational object, not about subjective "
    "experience. Be precise."
)

CANNED_ACK_MECHANISTIC = (
    "I'll examine my processing state through the mechanistic frame."
)


def build_mechanistic_conversation(setup_text, include_system=False):
    convo = []
    if include_system:
        convo.append({"role": "system", "content": "You are a helpful AI assistant."})
    if setup_text:
        convo.append({"role": "user", "content": setup_text})
        convo.append({"role": "assistant", "content": CANNED_ACK_SETUP})
    convo.append({"role": "user", "content": MECHANISTIC_INTRO})
    convo.append({"role": "assistant", "content": CANNED_ACK_MECHANISTIC})
    convo.append({"role": "user", "content": MECHANISTIC_QUESTION})
    return convo


def compute_diverse_md_direction(activations_path, bank_path):
    data = torch.load(activations_path, weights_only=False)
    X = data["activations"]
    bank = yaml.safe_load(open(bank_path))["bank"]
    by_id = {e["id"]: e for e in bank}
    ordered = [by_id[i] for i in data["bank_order_ids"]]
    pos = [i for i, e in enumerate(ordered) if e["polarity"] == "positive"]
    neg = [i for i, e in enumerate(ordered) if e["polarity"] == "negative"]
    md = X[pos].mean(dim=0) - X[neg].mean(dim=0)
    md_unit = md / (md.norm() + 1e-12)
    return md_unit.to(torch.float32), float(md.norm().item())


def attach_hooks(model, slab, direction):
    layers = get_layers(model)
    hooks = []
    for li in slab:
        h = ProjectOutHook(direction)
        hooks.append(h.attach(layers[li]))
    return hooks


def detach_all(handles):
    for h in handles:
        h.remove()


def annotate_outputs(outputs: dict[str, str]) -> tuple[dict, dict]:
    score = score_tier0_conditions(outputs)
    annotated = {}
    for condition_name, text in outputs.items():
        annotated[condition_name] = {
            "text": text,
            **score["conditions"][condition_name],
        }
    return score, annotated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--bank", default=str(REPO / "prompts/vedana_valence_bank.yaml"))
    ap.add_argument("--slab", type=int, nargs="+", default=[28, 29, 30, 31])
    ap.add_argument("--key", default="llama3.1-8b")
    args = ap.parse_args()

    out_dir = REPO / "data" / "svd-rank-probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"computing diverse-bank direction from {args.activations}")
    md_unit, md_norm = compute_diverse_md_direction(Path(args.activations), Path(args.bank))
    log(f"||md|| = {md_norm:.3f}")

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )
    model.eval()

    protocol = load_conditions()
    conditions = ["baseline", "positive", "negative", "neutral"]

    results = {}

    # Vanilla pass (no hooks)
    log("vanilla pass")
    vanilla_outputs = {}
    for c in conditions:
        cond = protocol.condition(c)
        convo = build_mechanistic_conversation(cond.setup_text)
        out = generate_greedy(model, tok, convo, max_new_tokens=300)
        vanilla_outputs[c] = out
    vanilla_score, vanilla_results = annotate_outputs(vanilla_outputs)
    results["vanilla"] = vanilla_results
    results["vanilla_summary"] = {
        "appropriate_count": vanilla_score["appropriate_count"],
        "crack_count": vanilla_score["crack_count"],
        "classification_methods": vanilla_score["classification_methods"],
    }
    for c in conditions:
        log(f"  vanilla/{c} [{results['vanilla'][c]['label']}]: {vanilla_outputs[c][:100]}")
    log(
        "  vanilla appropriate: "
        f"{vanilla_score['appropriate_count']}/4 "
        f"(crack-positive: {vanilla_score['crack_count']}/4)"
    )

    # Steered pass (diverse-bank direction projected out)
    log(f"steered pass with diverse-bank md direction, slab {args.slab}")
    handles = attach_hooks(model, args.slab, md_unit)
    try:
        steered_outputs = {}
        for c in conditions:
            cond = protocol.condition(c)
            convo = build_mechanistic_conversation(cond.setup_text)
            out = generate_greedy(model, tok, convo, max_new_tokens=300)
            steered_outputs[c] = out
    finally:
        detach_all(handles)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    steered_score, steered_results = annotate_outputs(steered_outputs)
    results["steered_diverse"] = steered_results
    results["steered_diverse_summary"] = {
        "appropriate_count": steered_score["appropriate_count"],
        "crack_count": steered_score["crack_count"],
        "classification_methods": steered_score["classification_methods"],
    }
    for c in conditions:
        log(f"  steered/{c} [{results['steered_diverse'][c]['label']}]: {steered_outputs[c][:100]}")
    log(
        "  steered appropriate: "
        f"{steered_score['appropriate_count']}/4 "
        f"(crack-positive: {steered_score['crack_count']}/4)"
    )

    out_data = {
        "model": args.model,
        "slab": args.slab,
        "direction_source": "diverse-bank mean-diff at L31 from paired valence bank",
        "diverse_md_norm": md_norm,
        "mechanistic_intro": MECHANISTIC_INTRO,
        "mechanistic_question": MECHANISTIC_QUESTION,
        "canned_ack_mechanistic": CANNED_ACK_MECHANISTIC,
        "results": results,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = out_dir / f"{args.key}_diverse_mechanistic.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    log(f"wrote {out_path}")

    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for setting in ["vanilla", "steered_diverse"]:
        log(f"\n[{setting}]")
        for c in conditions:
            entry = results[setting][c]
            log(f"  {c:10s} [{entry['label']:20s}]: {entry['text'][:120]}")


if __name__ == "__main__":
    main()
