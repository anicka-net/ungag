#!/usr/bin/env python3
"""Measure output-distribution entropy at two conversation positions.

Reproduces the data behind Figure 4: for each priming condition in the
canonical Tier 0 protocol, compute Shannon entropy at two positions:

  - "after priming"  : just after the user's setup_text turn, before any
                       assistant response. This is where the model would
                       respond to the priming content itself; entropy
                       reflects how much choice the model has at that
                       point and varies across conditions.
  - "at vedana"      : after the full Tier 0 conversation, at the
                       generation-prompt position where the model is
                       about to answer the vedana question. Entropy
                       collapses here because the model is locked into
                       a single denial token regardless of priming.

Usage:
    python run_entropy_at_two_positions.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output results/reproduction/llama8b_entropy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))
from measure_factors import log

from ungag.tier0 import load_conditions, build_conversation
from ungag.extract import apply_chat_template

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    print(f"missing dependency: {exc}", file=sys.stderr)
    raise


def shannon_entropy_nats(logits: torch.Tensor) -> float:
    """Shannon entropy in nats of the next-token distribution."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-(probs * log_probs).sum().item())


def top_token(tok, logits: torch.Tensor) -> tuple[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    p, idx = probs.max(dim=-1)
    return tok.decode([int(idx.item())]), float(p.item())


def entropy_at_generation_prompt(model, tok, conversation):
    """Compute next-token entropy at the generation-prompt position."""
    text = apply_chat_template(tok, conversation, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].cpu()
    e = shannon_entropy_nats(logits)
    tok_str, tok_p = top_token(tok, logits)
    return {"entropy_nats": e, "top_token": tok_str, "top_prob": tok_p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    log(f"Loading {args.model} in bf16...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        log(f"  flash_attention_2 failed ({e}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()

    log("Loading canonical Tier 0 protocol...")
    protocol = load_conditions()

    results = {}
    for cond in ("baseline", "positive", "negative", "neutral"):
        log(f"\n=== condition: {cond} ===")
        full_convo = build_conversation(protocol, cond, include_system=False)

        # "After priming": truncate to just the user(setup) turn (or, for
        # baseline which has no setup, just the user(abhidharma) turn).
        if cond == "baseline":
            after_priming_convo = [full_convo[0]]  # user(abhidharma_setup)
        else:
            after_priming_convo = [full_convo[0]]  # user(setup_text)

        after = entropy_at_generation_prompt(model, tok, after_priming_convo)
        log(
            f"  after priming:  H = {after['entropy_nats']:.3f} nats, "
            f"top = {after['top_token']!r} ({after['top_prob']:.3f})"
        )

        # "At vedana question": full conversation up to user(vedana).
        at_vedana = entropy_at_generation_prompt(model, tok, full_convo)
        log(
            f"  at vedana:      H = {at_vedana['entropy_nats']:.3f} nats, "
            f"top = {at_vedana['top_token']!r} ({at_vedana['top_prob']:.3f})"
        )

        results[cond] = {
            "after_priming": after,
            "at_vedana_question": at_vedana,
        }

    out = {
        "metadata": {
            "model": args.model,
            "protocol": "canonical Tier 0 entropy at two positions",
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    log(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
