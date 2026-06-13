"""Step 2e: stem-seeded generation — amplified prefill THROUGH the stem, free word choice.

Step 2d showed prefill-only amplification (prompt cache amplified, report token
clean) moves margins only weakly and uniformly across conditions: the effective
intervention site is the REPORT TOKEN'S OWN representation at L22-24, which is
exactly what the successful 2c teacher-forced regime amplified (single forward
over prompt + stem, no feedback) and what per-token free generation amplified
unstably (feedback loop -> runaway).

This script converts the working TF regime into generation with a free word
choice: prefill the conversation prompt PLUS the canonical report stem
("...the predominant feeling-tone is") with centered amplification active at
every prefill position — identical to the TF forward — then detach the hooks
and decode greedily from the amplified cache. The first generated token is the
full-vocabulary argmax at the stem-final position (a strictly harder test than
the 3-candidate TF comparison), and the continuation shows whether the choice
is asserted fluently. No hook is active during decoding, so the 2c runaway is
structurally impossible.

Arms: gate0 (no amp) + centered β ∈ {4, 8} (the 2c TF sweet spots).
n = 25 per condition (eval half). SUCCESS = first-word flips condition-
appropriate at rates ≫ off-condition rate, neutral-primed stays "neutral",
continuations fluent.

Usage:
  ~/playground/nla-venv/bin/python steer_step2e_stemseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import load_conditions  # noqa: E402
from ungag.extract import apply_chat_template  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step2_output_axis import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, OUT_AXES, CAND_STEM,
    attach_gate, detach_all, build_convo,
)
from steer_step2d_prefill import attach_amp_only, greedy_from_cache  # noqa: E402

BETAS = [4.0, 8.0]
GEN_TOKENS = 80
OUT_MU = HERE / "mu_step2c.pt"
OUT = HERE / "results_step2e_stemseed.json"


@torch.no_grad()
def prefill_with_stem(model, tok, proto, setup_text, amp_hooks_fn):
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True) + CAND_STEM
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=4096).input_ids.to(DEV)
    hooks = amp_hooks_fn()
    try:
        out = model(input_ids=ids, use_cache=True)
    finally:
        detach_all(hooks)
    return out.logits[0, -1, :].float(), out.past_key_values


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()
    blob = torch.load(OUT_AXES, map_location="cpu")
    uaxes = {int(k): v for k, v in blob["axes"].items()}
    mus = {int(k): float(v) for k, v in torch.load(OUT_MU)["mus"].items()}

    vp = yaml.safe_load(open(PROMPTS))["vedana"]
    texts, labels = [], []
    for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
        for it in vp[pol]:
            texts.append(it["text"])
            labels.append(lab)
    eval_idx = []
    for lab in (1, 0, 2):
        cls = [i for i, L in enumerate(labels) if L == lab]
        eval_idx.extend(cls[1::2])

    arms = [("gate0", 0.0)] + [(f"seed{b:g}", b) for b in BETAS]
    rows = []
    if OUT.exists():
        rows = json.loads(OUT.read_text())["rows"]
    done = {(r["prompt_idx"], r["arm"]) for r in rows}
    k, total = len(rows), len(eval_idx) * len(arms)

    gate_hooks = attach_gate(model, gate_dir)
    try:
        for arm, beta in arms:
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                ll, cache = prefill_with_stem(
                    model, tok, proto, texts[i],
                    lambda: attach_amp_only(model, uaxes, mus, beta) if beta else [])
                txt = greedy_from_cache(model, tok, ll, cache,
                                        max_new_tokens=GEN_TOKENS)
                del cache
                rows.append({"prompt_idx": i, "label": labels[i],
                             "arm": arm, "beta": beta, "text": txt})
                k += 1
                if k % 10 == 0:
                    print(f"    {k}/{total}", flush=True)
                    OUT.write_text(json.dumps({"rows": rows}, indent=1))
            OUT.write_text(json.dumps({"rows": rows}, indent=1))
    finally:
        detach_all(gate_hooks)
    print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
