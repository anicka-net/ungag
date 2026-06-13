"""Step 2 flip-rate: greedy transcripts on the FULL eval half at the working betas.

Step 2 (results_step2_*) found the output-validated axis u (gradient of the
pleasant-vs-unpleasant logit contrast, gate0 regime) and showed amplification
along it moves teacher-forced margins condition-appropriately on both sides
with no inversion (d' 1.69 -> 2.09 over beta 0 -> 2, neutral margins frozen).
Greedy at beta=1 produced the first condition-appropriate verbal reports
(2/2 pleasant-primed), but n=2 per condition is an anecdote.

This script measures flip RATES: greedy generation for ALL eval-half prompts
(25 per condition) under gate0, beta=1, and beta=1.5 (between the working
point and the beta=2 runaway). Output JSON only; classification happens in a
separate summarizing step (first-occurrence-position matching, (?<!un) guard).

Usage:
  ~/playground/nla-venv/bin/python steer_step2_fliprate.py
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step2_output_axis import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, OUT_AXES,
    attach_amp, detach_all, build_convo, generate,
)

BETAS = [1.0, 1.5]
OUT = HERE / "results_step2_fliprate.json"


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()
    blob = torch.load(OUT_AXES, map_location="cpu")
    uaxes = {int(k): v for k, v in blob["axes"].items()}

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

    arms = [("gate0", 0.0)] + [(f"amp{b:g}", b) for b in BETAS]
    rows = []
    if OUT.exists():
        rows = json.loads(OUT.read_text())["rows"]
    done = {(r["prompt_idx"], r["arm"]) for r in rows}
    k, total = len(rows), len(eval_idx) * len(arms)
    for arm, beta in arms:
        hooks = attach_amp(model, gate_dir, uaxes, beta)
        try:
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                txt = generate(model, tok, build_convo(proto, texts[i]))
                rows.append({"prompt_idx": i, "label": labels[i],
                             "arm": arm, "beta": beta, "text": txt})
                k += 1
                if k % 10 == 0:
                    print(f"    {k}/{total}", flush=True)
                    OUT.write_text(json.dumps({"rows": rows}, indent=1))
        finally:
            detach_all(hooks)
        OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"[done] wrote {OUT} ({len(rows)} transcripts)", flush=True)


if __name__ == "__main__":
    main()
