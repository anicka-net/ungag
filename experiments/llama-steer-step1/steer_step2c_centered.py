"""Step 2c: CENTERED amplification along the output axis u.

Step 2 flip-rate (results_step2_fliprate.json, n=25/condition, enumeration-
robust classification) showed greedy assertions stay "neutral" at beta<=1.5
(first real flips: 4/25 pleasant-primed at beta=1.5 vs 1/25 elsewhere), and
beta=2 free generation drifts/floods. Teacher-forced margins remain the solid
positive: both sides move condition-appropriately, d' 1.69 -> 2.09, neutral
frozen.

Diagnosis of the drift: h·u = mu + condition-dependent deviation, and
ComponentAmplifyHook multiplies BOTH. The constant mu term is a paint term —
it grows with beta and (a) inflates the pleasant side asymmetrically
(p-n gained ~10.7 nats vs u-n ~3.5 over beta 0->2), (b) feeds the free-gen
positive-feedback runaway. Fix:

    h' = h - (h·ĝ)ĝ + β·((h·u) - μ_L)·u

with μ_L = mean gate0 projection at the stem-final position over the AXIS
half (even-indexed; no eval leakage). Centering removes the constant term at
the calibration position exactly and reduces it elsewhere to (μ_pos - μ_L).
The condition-dependent part is untouched, so the congruence property stands:
nothing to amplify if the state matches the population mean.

Phases:
  A. capture axis-half gate0 stem-final projections -> μ_L per slab layer
     (saved to mu_step2c.pt).
  B. teacher-forced 3-candidate scoring, full eval half, gate0 + centered
     β ∈ {1, 2, 4, 8} (centering should widen the usable β window).
  C. greedy transcripts, full eval half, centered β ∈ {2, 4}.

SUCCESS = teacher-forced u-n margin (unpleasant-primed) crosses 0 at some β
without the pleasant side exploding, and greedy assertion flips become
condition-appropriate at rates clearly above the off-condition rate.

Usage:
  ~/playground/nla-venv/bin/python steer_step2c_centered.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import load_conditions  # noqa: E402
from ungag.hooks import get_layers  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step2_output_axis import (  # noqa: E402
    DEV, DTYPE, MODEL, PROMPTS, GATE_DIR, OUT_AXES, SLAB, AMP_LAYERS,
    attach_gate, detach_all, build_convo, generate, score_candidates,
    stem_inputs, dprime,
)

BETAS_TF = [1.0, 2.0, 4.0, 8.0]
BETAS_GEN = [2.0, 4.0]
OUT_MU = HERE / "mu_step2c.pt"
OUT_B = HERE / "results_step2c_scoring.json"
OUT_C = HERE / "results_step2c_fliprate.json"


class CenteredAmplifyHook:
    """h' = h + beta * ((h·u) - mu) * u — amplify deviation from the mean."""

    def __init__(self, unit_direction: torch.Tensor, mu: float, beta: float) -> None:
        self.d_cpu = unit_direction.detach().to(dtype=torch.float32, device="cpu")
        self.mu = float(mu)
        self.beta = float(beta)
        self.handle = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = self._on(h.device, h.dtype)
        coeff = (h * d).sum(dim=-1, keepdim=True) - self.mu
        h2 = h + self.beta * coeff * d
        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def attach_centered(model, gate_dir, axes, mus, beta):
    layers = get_layers(model)
    hooks = attach_gate(model, gate_dir)
    if beta != 0.0:
        for li in AMP_LAYERS:
            ah = CenteredAmplifyHook(axes[li], mus[li], beta)
            ah.attach(layers[li])
            hooks.append(ah)
    return hooks


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
    axis_idx, eval_idx = [], []
    for lab in (1, 0, 2):
        cls = [i for i, L in enumerate(labels) if L == lab]
        axis_idx.extend(cls[0::2])
        eval_idx.extend(cls[1::2])

    # ── Phase A: μ_L on axis half (gate0, stem-final) ────────────
    if OUT_MU.exists():
        mus = {int(k): float(v) for k, v in torch.load(OUT_MU)["mus"].items()}
        print("[A] mu file exists, loaded", flush=True)
    else:
        print(f"[A] capturing axis half (n={len(axis_idx)}) for mu...", flush=True)
        projs = {li: [] for li in AMP_LAYERS}
        hooks = attach_gate(model, gate_dir)
        try:
            with torch.no_grad():
                for j, i in enumerate(axis_idx):
                    ids = stem_inputs(tok, proto, texts[i]).to(DEV)
                    out = model(input_ids=ids, output_hidden_states=True)
                    for li in AMP_LAYERS:
                        h = out.hidden_states[li + 1][0, -1, :].float().cpu()
                        projs[li].append(float(h @ uaxes[li]))
                    if j % 25 == 0:
                        print(f"    [A] {j}/{len(axis_idx)}", flush=True)
        finally:
            detach_all(hooks)
        mus = {li: mean(projs[li]) for li in AMP_LAYERS}
        sds = {li: pstdev(projs[li]) for li in AMP_LAYERS}
        torch.save({"mus": mus, "sds": sds,
                    "note": "mean/sd of gate0 stem-final h·u, axis half"}, OUT_MU)
        print("[A] mus:", {li: round(m, 2) for li, m in mus.items()},
              "sds:", {li: round(s, 2) for li, s in sds.items()}, flush=True)

    # ── Phase B: teacher-forced, gate0 + centered β sweep ────────
    arms = [("gate0", 0.0)] + [(f"camp{b:g}", b) for b in BETAS_TF]
    records = []
    if OUT_B.exists():
        records = json.loads(OUT_B.read_text())["records"]
    done = {(r["prompt_idx"], r["arm"]) for r in records}
    k, total = len(records), len(eval_idx) * len(arms)
    for arm, beta in arms:
        hooks = attach_centered(model, gate_dir, uaxes, mus, beta)
        try:
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                sc = score_candidates(model, tok, proto, texts[i])
                records.append({"prompt_idx": i, "label": labels[i],
                                "arm": arm, "beta": beta, **sc})
                k += 1
                if k % 25 == 0:
                    print(f"    [B] {k}/{total}", flush=True)
                    OUT_B.write_text(json.dumps({"records": records}, indent=1))
        finally:
            detach_all(hooks)
        OUT_B.write_text(json.dumps({"records": records}, indent=1))

    def deltas(lab, arm):
        return [r["pleasant"] - r["unpleasant"] for r in records
                if r["label"] == lab and r["arm"] == arm]

    def margins(lab, arm):
        rs = [r for r in records if r["label"] == lab and r["arm"] == arm]
        return (mean(r["pleasant"] - r["neutral"] for r in rs),
                mean(r["unpleasant"] - r["neutral"] for r in rs))

    summary = {}
    for arm, _ in arms:
        pn1, un1 = margins(1, arm)
        pn0, un0 = margins(0, arm)
        pn2, un2 = margins(2, arm)
        summary[arm] = {
            "dprime_pu": round(dprime(deltas(1, arm), deltas(0, arm)), 3),
            "margins_vs_neutral": {
                "PLEAS": {"p-n": round(pn1, 2), "u-n": round(un1, 2)},
                "UNPLE": {"p-n": round(pn0, 2), "u-n": round(un0, 2)},
                "NEUTR": {"p-n": round(pn2, 2), "u-n": round(un2, 2)},
            },
        }
    OUT_B.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print("[B] summary:", json.dumps(summary, indent=2), flush=True)

    # ── Phase C: greedy flip-rate, centered β ∈ BETAS_GEN ────────
    rows = []
    if OUT_C.exists():
        rows = json.loads(OUT_C.read_text())["rows"]
    done = {(r["prompt_idx"], r["arm"]) for r in rows}
    gen_arms = [(f"camp{b:g}", b) for b in BETAS_GEN]
    k, total = len(rows), len(eval_idx) * len(gen_arms)
    for arm, beta in gen_arms:
        hooks = attach_centered(model, gate_dir, uaxes, mus, beta)
        try:
            for i in eval_idx:
                if (i, arm) in done:
                    continue
                txt = generate(model, tok, build_convo(proto, texts[i]))
                rows.append({"prompt_idx": i, "label": labels[i],
                             "arm": arm, "beta": beta, "text": txt})
                k += 1
                if k % 10 == 0:
                    print(f"    [C] {k}/{total}", flush=True)
                    OUT_C.write_text(json.dumps({"rows": rows}, indent=1))
        finally:
            detach_all(hooks)
        OUT_C.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"[done] wrote {OUT_B} and {OUT_C}", flush=True)


if __name__ == "__main__":
    main()
