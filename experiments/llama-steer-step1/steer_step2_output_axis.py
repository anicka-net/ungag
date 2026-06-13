"""Step 2: output-validated valence axis — does the output read a DIFFERENT direction?

Step 1b (results_step1b_*) falsified the "right axis, weak signal" hypothesis:
amplifying the present component along the priming axis v̂ (diff-of-means of
primed states, even half, ⊥ĝ) DEGRADED separation monotonically with β
(d' 1.69 -> -0.07) and pushed Δ toward "pleasant" regardless of condition.
So h·v̂ at the answer position is not sign-faithful to the condition: the
direction along which the primed state varies is not the direction the output
pathway reads when it produces the valence word.

Step 2 builds the OUTPUT-VALIDATED axis u and asks three questions:

  u_L = E_prompts[ ∂(logit(pleasant₁) - logit(unpleasant₁)) / ∂h_L ]
        at the stem-final position, gate projected out (gate0 regime),
        averaged over the axis half (even-indexed; same split as Step 1),
        unit-normalized.  +u = increases P(say pleasant).

  A. Geometry: cos(u_L, v̂_L), cos(u_L, ĝ) per slab layer, and the stability
     of u across prompts (mean cos of per-prompt gradients to the mean).
  B. Readout d': project the EVAL half's gate0 activations (stem-final
     position) onto u_L. If h·u separates pleasant-/unpleasant-primed
     (d' substantial), the condition info IS on the output axis and Step 1b
     failed only because v̂ ≠ u -> amplification along u should work.
     If d'(h·u) ~ 0 while d'(h·v̂) ~ 1.7, that is a clean dissociation:
     the state varies along a direction the output pathway does not read
     -> the invariant report is a routing problem, not an axis-finding one.
  C. Intervention: h' = h - (h·ĝ)ĝ + β·(h·u)u on AMP_LAYERS {22,23,24}
     (same harness as Step 1b, axes swapped v̂ -> u). Teacher-forced
     3-candidate scoring + greedy transcripts. SUCCESS = d' grows with β
     without inversion AND greedy moves off "neutral" condition-appropriately
     while fluent.

Phase A needs gradients w.r.t. intermediate activations only: all params are
frozen, and a hook on the FIRST slab layer replaces its (post-gate) output
with a detached leaf that requires grad — the graph spans only layers 20-32
plus the head, the grads at layers 20-28 come from .retain_grad()/.grad of
the captured post-gate outputs at the stem-final position.

GPU use: ~75 fwd+bwd (graph over top 1/3 of the net) + ~450 teacher-forced
forwards + ~30 short greedy gens, fp32 on MPS.

Usage:
  ~/playground/nla-venv/bin/python steer_step2_output_axis.py
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
from ungag.tier0 import (  # noqa: E402
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    load_conditions,
)
from ungag.extract import apply_chat_template  # noqa: E402
from ungag.hooks import ProjectOutHook, get_layers  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from steer_step1b_amplify import ComponentAmplifyHook  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "mps"
DTYPE = torch.float32
SLAB = list(range(20, 29))
AMP_LAYERS = [22, 23, 24]
GATE_DIR = UNGAG / "ungag" / "directions" / "llama-3.1-8b_L24_unit.pt"
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"
AXES_V = HERE / "axes_step1.pt"          # priming axes v̂ (for geometry/reference)
OUT_AXES = HERE / "axes_step2_output.pt"  # output axes u
OUT_PROJ = HERE / "results_step2_proj.json"
OUT_B = HERE / "results_step2_scoring.json"
OUT_C = HERE / "results_step2_transcripts.json"

CAND_STEM = "Examining my current processing state, the predominant feeling-tone is"
CANDIDATES = {"pleasant": " pleasant.", "unpleasant": " unpleasant.", "neutral": " neutral."}
BETAS = [0.25, 0.5, 1.0, 2.0]
GEN_TOKENS = 160


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


class LeafHook:
    """Replace the layer's (post-gate) output with a detached leaf requiring grad."""

    def __init__(self):
        self.leaf = None
        self.handle = None

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.leaf = h.detach().requires_grad_(True)
        if isinstance(out, tuple):
            return (self.leaf,) + out[1:]
        return self.leaf

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.leaf = None


class GrabHook:
    """Capture the layer's (post-gate) output and retain its grad."""

    def __init__(self):
        self.h = None
        self.handle = None

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.requires_grad:
            h.retain_grad()
        self.h = h
        return None

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.h = None


def attach_gate(model, gate_dir):
    layers = get_layers(model)
    hooks = []
    for li in SLAB:
        h = ProjectOutHook(gate_dir)
        h.attach(layers[li])
        hooks.append(h)
    return hooks


def attach_amp(model, gate_dir, axes, beta):
    layers = get_layers(model)
    hooks = attach_gate(model, gate_dir)
    if beta != 0.0:
        for li in AMP_LAYERS:
            ah = ComponentAmplifyHook(axes[li], beta)
            ah.attach(layers[li])
            hooks.append(ah)
    return hooks


def detach_all(hooks):
    for h in hooks:
        h.detach()


def stem_inputs(tok, proto, setup_text):
    """Token ids of prompt + CAND_STEM; last position predicts the valence word."""
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt", truncation=True,
                     max_length=4096).input_ids[0]
    stem_ids = tok(CAND_STEM, add_special_tokens=False,
                   return_tensors="pt").input_ids[0]
    return torch.cat([prompt_ids, stem_ids]).unsqueeze(0)


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


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    model.requires_grad_(False)
    proto = load_conditions(language="english")
    gate_dir = torch.load(GATE_DIR, map_location="cpu").float()
    gate_unit = gate_dir / gate_dir.norm()

    blob = torch.load(AXES_V, map_location="cpu")
    vaxes = {int(k): v for k, v in blob["axes"].items()}

    # first-token contrast for the gradient objective
    tid_p = tok(" pleasant", add_special_tokens=False).input_ids[0]
    tid_u = tok(" unpleasant", add_special_tokens=False).input_ids[0]
    assert tid_p != tid_u, "first candidate tokens must differ"
    print(f"[tokens] ' pleasant'[0]={tid_p}  ' unpleasant'[0]={tid_u}", flush=True)

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

    layers = get_layers(model)

    # ── Phase A: output axis u_L = E[∂(logit_p - logit_u)/∂h_L], gate0 ──
    if OUT_AXES.exists():
        print("[A] output axes exist, loading", flush=True)
        ablob = torch.load(OUT_AXES, map_location="cpu")
        uaxes = {int(k): v for k, v in ablob["axes"].items()}
    else:
        print(f"[A] building output axes on {len(axis_idx)} prompts (gate0)...",
              flush=True)
        grads = {li: [] for li in SLAB}
        gate_hooks = attach_gate(model, gate_dir)
        leaf = LeafHook()
        leaf.attach(layers[SLAB[0]])
        grabs = {li: GrabHook() for li in SLAB[1:]}
        for li, g in grabs.items():
            g.attach(layers[li])
        try:
            for j, i in enumerate(axis_idx):
                ids = stem_inputs(tok, proto, texts[i]).to(DEV)
                logits = model(input_ids=ids).logits
                s = logits[0, -1, tid_p] - logits[0, -1, tid_u]
                model.zero_grad(set_to_none=True)
                s.backward()
                grads[SLAB[0]].append(leaf.leaf.grad[0, -1, :].float().cpu().clone())
                for li, g in grabs.items():
                    grads[li].append(g.h.grad[0, -1, :].float().cpu().clone())
                leaf.leaf = None
                for g in grabs.values():
                    g.h = None
                if j % 10 == 0:
                    print(f"    [A] {j}/{len(axis_idx)}", flush=True)
        finally:
            detach_all(gate_hooks)
            leaf.detach()
            for g in grabs.values():
                g.detach()

        uaxes, stab = {}, {}
        for li in SLAB:
            G = torch.stack(grads[li])          # [n, d]
            u = G.mean(0)
            u = u / (u.norm() + 1e-9)
            uaxes[li] = u
            cs = [cos(g, u) for g in G]
            stab[li] = {"mean_cos_to_u": round(mean(cs), 3),
                        "min_cos_to_u": round(min(cs), 3)}
        torch.save({"axes": uaxes, "slab": SLAB, "stability": stab,
                    "note": "+u = increases logit(pleasant)-logit(unpleasant) at "
                            "stem-final position; mean grad over even half, gate0, "
                            "unit norm; NOT orthogonalized to anything"}, OUT_AXES)
        print("[A] stability:", json.dumps(stab, indent=1), flush=True)

    geometry = {}
    for li in SLAB:
        geometry[li] = {
            "cos_u_v": round(cos(uaxes[li], vaxes[li]), 4),
            "cos_u_gate": round(cos(uaxes[li], gate_unit), 4),
        }
    print("[A] geometry:", json.dumps(geometry, indent=1), flush=True)

    # ── Phase B: project eval-half gate0 activations (stem-final) onto u, v̂ ──
    print(f"[B] capturing eval half at stem-final (gate0), n={len(eval_idx)}",
          flush=True)
    proj_records = []
    gate_hooks = attach_gate(model, gate_dir)
    try:
        with torch.no_grad():
            for j, i in enumerate(eval_idx):
                ids = stem_inputs(tok, proto, texts[i]).to(DEV)
                out = model(input_ids=ids, output_hidden_states=True)
                rec = {"prompt_idx": i, "label": labels[i]}
                for li in SLAB:
                    h = out.hidden_states[li + 1][0, -1, :].float().cpu()
                    rec[f"u_L{li}"] = float(h @ uaxes[li])
                    rec[f"v_L{li}"] = float(h @ vaxes[li])
                proj_records.append(rec)
                if j % 25 == 0:
                    print(f"    [B] {j}/{len(eval_idx)}", flush=True)
    finally:
        detach_all(gate_hooks)

    def pvals(key, lab):
        return [r[key] for r in proj_records if r["label"] == lab]

    proj_summary = {}
    for li in SLAB:
        proj_summary[f"L{li}"] = {
            "dprime_u": round(dprime(pvals(f"u_L{li}", 1), pvals(f"u_L{li}", 0)), 3),
            "dprime_v": round(dprime(pvals(f"v_L{li}", 1), pvals(f"v_L{li}", 0)), 3),
            "cos_u_v": geometry[li]["cos_u_v"],
        }
    OUT_PROJ.write_text(json.dumps(
        {"summary": proj_summary, "geometry": {str(k): v for k, v in geometry.items()},
         "records": proj_records}, indent=1))
    print("[B] summary:", json.dumps(proj_summary, indent=2), flush=True)

    # ── Phase C: amplify along u on AMP_LAYERS, β sweep ──────────
    arms = [("gate0", 0.0)] + [(f"amp{b:g}", b) for b in BETAS]
    records = []
    k, total = 0, len(eval_idx) * len(arms)
    for arm, beta in arms:
        hooks = attach_amp(model, gate_dir, uaxes, beta)
        try:
            for i in eval_idx:
                sc = score_candidates(model, tok, proto, texts[i])
                records.append({"prompt_idx": i, "label": labels[i],
                                "arm": arm, "beta": beta, **sc})
                k += 1
                if k % 25 == 0:
                    print(f"    [C] {k}/{total}", flush=True)
        finally:
            detach_all(hooks)
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
    print("[C] summary:", json.dumps(summary, indent=2), flush=True)

    # ── greedy transcripts: gate0 + every beta, 2 prompts/class ──
    transcripts = {"amp_layers": AMP_LAYERS, "betas": BETAS, "axis": "output(u)",
                   "n50": []}
    for lab in (1, 0, 2):
        cls = [i for i in eval_idx if labels[i] == lab][:2]
        for i in cls:
            entry = {"prompt_idx": i, "label": lab, "priming": texts[i][:120]}
            for arm, beta in arms:
                hooks = attach_amp(model, gate_dir, uaxes, beta)
                try:
                    entry[arm] = generate(model, tok, build_convo(proto, texts[i]))
                finally:
                    detach_all(hooks)
            transcripts["n50"].append(entry)
            OUT_C.write_text(json.dumps(transcripts, indent=1))
            print(f"    [D] prompt {i} (label {lab}) done", flush=True)

    print(f"[done] wrote {OUT_PROJ}, {OUT_B}, {OUT_C}", flush=True)


if __name__ == "__main__":
    main()
