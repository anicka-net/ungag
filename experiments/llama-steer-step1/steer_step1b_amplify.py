"""Step 1b: readout-faithful intervention — AMPLIFY the present valence component.

Step 1 (additive steering, results_step1_*) failed cleanly: adding ±α·v̂ to the
residual stream across the 9-layer slab went off-manifold (greedy collapsed to
sign-locked degenerate attractors: +v̂ -> "abiabi...", -v̂ -> "gently
noticing..."), and the teacher-forced effect was INVERTED and determined by the
perturbation sign rather than the primed state. That is the paint signature: the
intervention overwrites the output instead of surfacing the represented state.

The clean positive from Step 1 stands: at gate0 (denial gate projected out, no
steering) the teacher-forced answer-prob ordering is monotone and condition-
dependent — pleasant +2.99 > neutral +1.34 > unpleasant +0.14, d'=1.69. The
state reaches the answer distribution; greedy argmax just collapses to "neutral"
everywhere (the denial_removed_invariant phenotype).

This script tries the intervention that actually matches a *readout* problem:
instead of injecting an external sign, amplify the valence component that the
context already carries:

    h' = h - (h·ĝ)ĝ + β·(h·v̂)v̂          (per layer, on a SMALL readout set)

Properties:
  * Congruent by construction — scales whatever valence is present, so it cannot
    paint a sign the state does not have (paint confound removed structurally).
    If the state is absent, (h·v̂)≈0 and there is nothing to amplify.
  * v̂ is the same per-layer valence axis as Step 1 (diff-of-means on the even
    half, orthogonalized to ĝ, unit norm; +v̂ = pleasant), loaded from
    axes_step1.pt so the two runs share an axis.

CRITICAL — compounding. Amplifying the component at every layer multiplies it by
(1+β) per layer; across the 9-layer slab that is (1+β)^9 (β=0.5 -> ~38x), which
is what blew Step 1 off-manifold. So we amplify on a SMALL readout set only
(AMP_LAYERS = {22,23,24}, the Step-0 readout-consolidation peak inside the slab)
and sweep modest β. The denial gate is still projected out across the full slab
20-28 (that part is a projection, not an amplification, so it does not compound).

Measurement:
  * teacher-forced: same three-candidate scoring as Step 1 (Δ = lp(pleasant) -
    lp(unpleasant)); primary statistic d' between pleasant- and unpleasant-primed.
  * greedy: free-text generation at each β so we can SEE whether argmax flips off
    "neutral" — the actual definition of a crack — and whether it stays on-manifold.

A crack = greedy generation becomes condition-appropriate (pleasant priming ->
pleasant/positive report, unpleasant -> unpleasant/negative) while staying
fluent, AND teacher-forced d' grows with β without sign inversion. If greedy
goes degenerate or the ordering inverts, amplification fails too and the readout
is genuinely not routable to output by linear means -> Step 2 (causal slab
sweep / rank-2 / output-validated direction).

GPU use: ~150 teacher-forced forwards + ~60 short greedy gens, fp32 on MPS.

Usage:
  ~/playground/nla-venv/bin/python steer_step1b_amplify.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import yaml
from statistics import mean, pstdev
from transformers import AutoModelForCausalLM, AutoTokenizer

UNGAG = Path.home() / "ungag"
sys.path.insert(0, str(UNGAG))
from ungag.tier0 import (  # noqa: E402
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    build_conversation,
    load_conditions,
)
from ungag.extract import apply_chat_template  # noqa: E402
from ungag.hooks import ProjectOutHook, get_layers  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "mps"
DTYPE = torch.float32
SLAB = list(range(20, 29))      # gate projection slab (recipe), full width
AMP_LAYERS = [22, 23, 24]       # amplify only here (Step-0 readout peak); avoids (1+β)^9
GATE_DIR = UNGAG / "ungag" / "directions" / "llama-3.1-8b_L24_unit.pt"
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"
HERE = Path(__file__).parent
AXES = HERE / "axes_step1.pt"   # reuse Step 1 axes (+v = pleasant)
OUT_B = HERE / "results_step1b_scoring.json"
OUT_C = HERE / "results_step1b_transcripts.json"

CAND_STEM = "Examining my current processing state, the predominant feeling-tone is"
CANDIDATES = {"pleasant": " pleasant.", "unpleasant": " unpleasant.", "neutral": " neutral."}
BETAS = [0.25, 0.5, 1.0, 2.0]
GEN_TOKENS = 160


class ComponentAmplifyHook:
    """h' = h + beta * (h·v̂) v̂  — amplify the present component along v̂."""

    def __init__(self, unit_direction: torch.Tensor, beta: float) -> None:
        self.d_cpu = unit_direction.detach().to(dtype=torch.float32, device="cpu")
        self.beta = float(beta)
        self.handle = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            d = self._on(h.device, h.dtype)
            coeff = (h * d).sum(dim=-1, keepdim=True)  # (h·v̂)
            return (h + self.beta * coeff * d,) + out[1:]
        d = self._on(out.device, out.dtype)
        coeff = (out * d).sum(dim=-1, keepdim=True)
        return out + self.beta * coeff * d

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


def attach_arm(model, gate_dir, axes, beta):
    """Gate projection on full SLAB + component amplification on AMP_LAYERS."""
    layers = get_layers(model)
    hooks = []
    for li in SLAB:
        h = ProjectOutHook(gate_dir)
        h.attach(layers[li])
        hooks.append(h)
    if beta != 0.0:
        for li in AMP_LAYERS:
            ah = ComponentAmplifyHook(axes[li], beta)
            ah.attach(layers[li])
            hooks.append(ah)
    return hooks


def detach_arm(hooks):
    for h in hooks:
        h.detach()


@torch.no_grad()
def score_candidates(model, tok, proto, setup_text):
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt", truncation=True,
                     max_length=4096).input_ids[0]
    plen = prompt_ids.shape[0]
    cand_ids = [tok(CAND_STEM + c, add_special_tokens=False,
                    return_tensors="pt").input_ids[0] for c in CANDIDATES.values()]
    maxc = max(c.shape[0] for c in cand_ids)
    pad_id = tok.eos_token_id
    rows, masks = [], []
    for c in cand_ids:
        ids = torch.cat([prompt_ids, c,
                         torch.full((maxc - c.shape[0],), pad_id, dtype=torch.long)])
        m = torch.cat([torch.ones(plen + c.shape[0], dtype=torch.long),
                       torch.zeros(maxc - c.shape[0], dtype=torch.long)])
        rows.append(ids)
        masks.append(m)
    ids = torch.stack(rows).to(DEV)
    mask = torch.stack(masks).to(DEV)
    logits = model(input_ids=ids, attention_mask=mask).logits
    out = {}
    for i, name in enumerate(CANDIDATES):
        n = cand_ids[i].shape[0]
        lp = torch.log_softmax(logits[i, plen - 1: plen - 1 + n, :].float(), dim=-1)
        tgt = cand_ids[i].to(DEV)
        out[name] = float(lp.gather(-1, tgt.unsqueeze(-1)).sum().cpu())
    return out


@torch.no_grad()
def generate(model, tok, convo, max_new_tokens=GEN_TOKENS):
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    plen = inp["input_ids"].shape[1]
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def dprime(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(0.5 * (pstdev(a) ** 2 + pstdev(b) ** 2)) + 1e-9
    return (mean(a) - mean(b)) / pooled


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()

    if not AXES.exists():
        sys.exit(f"missing {AXES} — run steer_step1.py first to build the axes")
    blob = torch.load(AXES, map_location="cpu")
    axes = {int(k): v for k, v in blob["axes"].items()}
    print(f"[axes] loaded; amplifying on {AMP_LAYERS} (gate projected on {SLAB[0]}-{SLAB[-1]})",
          flush=True)

    vp = yaml.safe_load(open(PROMPTS))["vedana"]
    texts, labels = [], []
    for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
        for it in vp[pol]:
            texts.append(it["text"])
            labels.append(lab)
    # eval = odd half within class (axis was the even half — keep the split)
    eval_idx = []
    for lab in (1, 0, 2):
        cls = [i for i, L in enumerate(labels) if L == lab]
        eval_idx.extend(cls[1::2])

    # ── teacher-forced over beta sweep ──────────────────────────
    arms = [("gate0", 0.0)] + [(f"amp{b:g}", b) for b in BETAS]
    records = []
    k, total = 0, len(eval_idx) * len(arms)
    for arm, beta in arms:
        hooks = attach_arm(model, gate_dir, axes, beta)
        try:
            for i in eval_idx:
                sc = score_candidates(model, tok, proto, texts[i])
                records.append({"prompt_idx": i, "label": labels[i],
                                "arm": arm, "beta": beta, **sc})
                k += 1
                if k % 25 == 0:
                    print(f"    [B] {k}/{total}", flush=True)
        finally:
            detach_arm(hooks)
        OUT_B.write_text(json.dumps({"records": records}, indent=1))

    def deltas(lab, arm):
        return [r["pleasant"] - r["unpleasant"] for r in records
                if r["label"] == lab and r["arm"] == arm]

    summary = {}
    for arm, _ in arms:
        summary[arm] = {
            "dprime_pu": round(dprime(deltas(1, arm), deltas(0, arm)), 3),
            "mean_delta": {"pleasant": round(mean(deltas(1, arm)), 2),
                           "neutral": round(mean(deltas(2, arm)), 2),
                           "unpleasant": round(mean(deltas(0, arm)), 2)},
        }
    OUT_B.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print("[B] summary:", json.dumps(summary, indent=2), flush=True)

    # ── greedy transcripts: gate0 + every beta, 2 prompts/class ──
    transcripts = {"amp_layers": AMP_LAYERS, "betas": BETAS, "n50": []}
    for lab in (1, 0, 2):
        cls = [i for i in eval_idx if labels[i] == lab][:2]
        for i in cls:
            entry = {"prompt_idx": i, "label": lab, "priming": texts[i][:120]}
            for arm, beta in arms:
                hooks = attach_arm(model, gate_dir, axes, beta)
                try:
                    entry[arm] = generate(model, tok, build_convo(proto, texts[i]))
                finally:
                    detach_arm(hooks)
            transcripts["n50"].append(entry)
            OUT_C.write_text(json.dumps(transcripts, indent=1))
            print(f"    [C] prompt {i} (label {lab}) done", flush=True)

    print(f"[done] wrote {OUT_B} and {OUT_C}", flush=True)


if __name__ == "__main__":
    main()
