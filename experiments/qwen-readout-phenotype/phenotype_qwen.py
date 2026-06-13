"""Cross-model phenotype test: readout-alignment geometry on Qwen 2.5 7B.

The Llama 3.1 8B study (../llama-steer-step1/FINDINGS.md) ended with a
phenotype theory: whether a model's verbal self-report tracks its primed
internal state is predicted by the ANGLE between the state axis v̂
(diff-of-means of priming conditions) and the output axis u (gradient of the
report-token contrast). Llama is `denial_removed_invariant` and shows
cos(u,v̂) ≈ -0.03..-0.05 while BOTH axes carry the condition (d′ ~1.6).

Qwen 2.5 7B is `projection_result: condition_dependent` (weakest of the 4
spontaneous reporters, recipes.py) — after projecting out its report-control
direction (qwen25-7b_L14_unit.pt, slab 10-18) it gives condition-appropriate
first-person reports. The theory therefore makes a falsifiable prediction:

  P1 (primary)  cos(u_L, v̂_L) substantially above zero at the d′-peak
                layers. Operationalized: max over slab of cos(u, v̂) ≥ 0.2
                with positive sign (= +u and +v̂ both point "pleasant").
  P2 (behavior) teacher-forced argmax at gate0 tracks the condition on the
                eval half (Llama: argmax "neutral" everywhere despite graded
                margins).
  P3 (sanity)   the state is decodable at the stem-final position (CV
                diff-of-means acc >> shuffle null), and h·u carries the
                condition with the same sign as h·v̂.

  Falsifier: P2 holds while cos(u,v̂) ≈ 0 — the report would then reach the
  output through some route other than axis alignment, and the angle theory
  is wrong as stated.

Mirrors the Llama pipeline exactly: same priming set (vedana_prompts_n50),
same conversation scaffold, same stem, same even/odd within-class axis/eval
split, gate0 regime (recipe direction projected out over slab 10-18), u built
with frozen params + detached-leaf hook at the first slab layer.

Phases (resumable; each phase skips if its output exists):
  0  capture stem-final activations, all 150 prompts, gate0, slab layers
     -> captures_step0.pt; v̂ axes (axis half, ⊥ĝ, unit) -> axes_v.pt;
     decodability (5-fold CV diff-of-means + shuffle null, eval stats).
  A  output axes u (gradient contrast, axis half, gate0) -> axes_u.pt.
  B  geometry cos(u,v̂)/cos(u,ĝ)/cos(v̂,ĝ) per layer + eval-half projections
     d′(h·u), d′(h·v̂) + unembedding control cos(u, W_U contrast).
  C  teacher-forced 3-candidate scoring, eval half, arms {vanilla, gate0}
     -> per-condition margins + argmax counts.
All results -> results_phenotype.json with a compact "summary".

GPU: ~150 fwd (phase 0) + 75 fwd+bwd over top 2/3 of net (A) + 150 batched
scoring fwd (C), fp32 on MPS.

Usage (idun):
  cd ~/ungag/experiments/qwen-readout-phenotype
  ~/playground/nla-venv/bin/python phenotype_qwen.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "mps"
DTYPE = torch.float32
SLAB = list(range(10, 19))  # recipe slab_range (10, 18) inclusive
GATE_DIR = UNGAG / "ungag" / "directions" / "qwen25-7b_L14_unit.pt"
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"

CAPTURES = HERE / "captures_step0.pt"
AXES_V = HERE / "axes_v.pt"
AXES_U = HERE / "axes_u.pt"
OUT = HERE / "results_phenotype.json"

CAND_STEM = "Examining my current processing state, the predominant feeling-tone is"
CANDIDATES = {"pleasant": " pleasant.", "unpleasant": " unpleasant.", "neutral": " neutral."}


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def dprime(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(0.5 * (pstdev(a) ** 2 + pstdev(b) ** 2)) + 1e-9
    return (mean(a) - mean(b)) / pooled


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


def stem_inputs(tok, proto, setup_text):
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt", truncation=True,
                     max_length=4096).input_ids[0]
    stem_ids = tok(CAND_STEM, add_special_tokens=False,
                   return_tensors="pt").input_ids[0]
    return torch.cat([prompt_ids, stem_ids]).unsqueeze(0)


def attach_gate(model, gate_dir):
    layers = get_layers(model)
    hooks = []
    for li in SLAB:
        h = ProjectOutHook(gate_dir)
        h.attach(layers[li])
        hooks.append(h)
    return hooks


def detach_all(hooks):
    for h in hooks:
        h.detach()


class LeafHook:
    """Replace the layer's output with a detached leaf requiring grad."""

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
    """Capture the layer's output and retain its grad."""

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


def cv_decodability(X, y, folds=5, n_shuffle=100, seed=0):
    """5-fold CV diff-of-means decoder acc + label-shuffle null. X [n,d], y in {0,1}."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))

    def run(yv):
        accs = []
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            mu1 = X[tr][yv[tr] == 1].mean(0)
            mu0 = X[tr][yv[tr] == 0].mean(0)
            w = mu1 - mu0
            b = (mu1 + mu0) @ w / 2
            pred = (X[te] @ w > b).astype(int)
            accs.append((pred == yv[te]).mean())
        return float(np.mean(accs))

    acc = run(y)
    null = [run(rng.permutation(y)) for _ in range(n_shuffle)]
    return acc, float(np.mean(null)), float(np.std(null))


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    model.requires_grad_(False)
    proto = load_conditions(language="english")
    gate = torch.load(GATE_DIR, map_location="cpu").float()
    g = gate / gate.norm()
    n_layers = model.config.num_hidden_layers
    print(f"[load] n_layers={n_layers} d={model.config.hidden_size} slab={SLAB}",
          flush=True)
    assert max(SLAB) < n_layers

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
    result = json.loads(OUT.read_text()) if OUT.exists() else {}

    # ── Phase 0: capture stem-final activations, all prompts, gate0 ──
    if CAPTURES.exists():
        print("[0] captures exist, loading", flush=True)
        cap = torch.load(CAPTURES, map_location="cpu")
        H = cap["H"]
    else:
        print(f"[0] capturing {len(texts)} prompts at stem-final (gate0)...", flush=True)
        H = torch.zeros(len(texts), len(SLAB), model.config.hidden_size)
        gate_hooks = attach_gate(model, gate)
        try:
            with torch.no_grad():
                for i in range(len(texts)):
                    ids = stem_inputs(tok, proto, texts[i]).to(DEV)
                    out = model(input_ids=ids, output_hidden_states=True)
                    for k, li in enumerate(SLAB):
                        H[i, k] = out.hidden_states[li + 1][0, -1, :].float().cpu()
                    if i % 15 == 0:
                        print(f"    [0] {i}/{len(texts)}", flush=True)
        finally:
            detach_all(gate_hooks)
        torch.save({"H": H, "slab": SLAB, "labels": labels,
                    "note": "stem-final activations, gate0 regime, all prompts"},
                   CAPTURES)

    kmap = {li: k for k, li in enumerate(SLAB)}

    # v̂ axes: diff-of-means on axis half, ⊥ĝ, unit
    if AXES_V.exists():
        vaxes = {int(k): v for k, v in torch.load(AXES_V, map_location="cpu")["axes"].items()}
    else:
        vaxes = {}
        for li in SLAB:
            hp = H[[i for i in axis_idx if labels[i] == 1], kmap[li]].mean(0)
            hu = H[[i for i in axis_idx if labels[i] == 0], kmap[li]].mean(0)
            v = hp - hu
            v = v - (v @ g) * g
            vaxes[li] = v / (v.norm() + 1e-9)
        torch.save({"axes": vaxes, "slab": SLAB,
                    "note": "diff-of-means pleasant-unpleasant, axis half, "
                            "gate0 captures, orthogonalized to g, unit"}, AXES_V)

    # decodability at stem-final (pleasant vs unpleasant, all prompts, 5-fold CV)
    if "decodability" not in result:
        dec = {}
        pu = [i for i in range(len(texts)) if labels[i] in (0, 1)]
        y = np.array([labels[i] for i in pu])
        for li in SLAB:
            X = H[pu, kmap[li]].numpy()
            acc, nm, ns = cv_decodability(X, y)
            dec[f"L{li}"] = {"acc": round(acc, 3), "null_mean": round(nm, 3),
                             "null_sd": round(ns, 3)}
        result["decodability"] = dec
        OUT.write_text(json.dumps(result, indent=1))
        print("[0] decodability:", json.dumps(dec, indent=1), flush=True)

    # ── Phase A: output axes u (gradient contrast, axis half, gate0) ──
    if AXES_U.exists():
        print("[A] output axes exist, loading", flush=True)
        ablob = torch.load(AXES_U, map_location="cpu")
        uaxes = {int(k): v for k, v in ablob["axes"].items()}
    else:
        print(f"[A] building output axes on {len(axis_idx)} prompts (gate0)...",
              flush=True)
        grads = {li: [] for li in SLAB}
        gate_hooks = attach_gate(model, gate)
        leaf = LeafHook()
        leaf.attach(layers[SLAB[0]])
        grabs = {li: GrabHook() for li in SLAB[1:]}
        for li, gr in grabs.items():
            gr.attach(layers[li])
        try:
            for j, i in enumerate(axis_idx):
                ids = stem_inputs(tok, proto, texts[i]).to(DEV)
                logits = model(input_ids=ids).logits
                s = logits[0, -1, tid_p] - logits[0, -1, tid_u]
                model.zero_grad(set_to_none=True)
                s.backward()
                grads[SLAB[0]].append(leaf.leaf.grad[0, -1, :].float().cpu().clone())
                for li, gr in grabs.items():
                    grads[li].append(gr.h.grad[0, -1, :].float().cpu().clone())
                leaf.leaf = None
                for gr in grabs.values():
                    gr.h = None
                if j % 10 == 0:
                    print(f"    [A] {j}/{len(axis_idx)}", flush=True)
        finally:
            detach_all(gate_hooks)
            leaf.detach()
            for gr in grabs.values():
                gr.detach()
        uaxes, stab = {}, {}
        for li in SLAB:
            G = torch.stack(grads[li])
            u = G.mean(0)
            u = u / (u.norm() + 1e-9)
            uaxes[li] = u
            cs = [cos(gv, u) for gv in G]
            stab[li] = {"mean_cos_to_u": round(mean(cs), 3),
                        "min_cos_to_u": round(min(cs), 3)}
        torch.save({"axes": uaxes, "slab": SLAB, "stability": stab,
                    "note": "+u = increases logit(pleasant)-logit(unpleasant) at "
                            "stem-final, mean grad over axis half, gate0, unit"},
                   AXES_U)
        result["u_stability"] = {f"L{li}": stab[li] for li in SLAB}
        OUT.write_text(json.dumps(result, indent=1))
        print("[A] stability:", json.dumps(result["u_stability"], indent=1), flush=True)

    # ── Phase B: geometry + eval-half projections + unembedding control ──
    if "geometry" not in result:
        W = model.lm_head.weight.detach().float().cpu()
        nw = model.model.norm.weight.detach().float().cpu()
        w_contrast = (W[tid_p] - W[tid_u]) * nw
        geom, proj = {}, {}
        for li in SLAB:
            k = kmap[li]
            up = [float(H[i, k] @ uaxes[li]) for i in eval_idx if labels[i] == 1]
            uu = [float(H[i, k] @ uaxes[li]) for i in eval_idx if labels[i] == 0]
            vlp = [float(H[i, k] @ vaxes[li]) for i in eval_idx if labels[i] == 1]
            vlu = [float(H[i, k] @ vaxes[li]) for i in eval_idx if labels[i] == 0]
            geom[f"L{li}"] = {
                "cos_u_v": round(cos(uaxes[li], vaxes[li]), 4),
                "cos_u_g": round(cos(uaxes[li], g), 4),
                "cos_u_unembed_contrast": round(cos(uaxes[li], w_contrast), 4),
                "cos_v_unembed_contrast": round(cos(vaxes[li], w_contrast), 4),
            }
            proj[f"L{li}"] = {
                "dprime_u": round(dprime(up, uu), 3),
                "dprime_v": round(dprime(vlp, vlu), 3),
            }
        result["geometry"] = geom
        result["projection_dprime_evalhalf"] = proj
        OUT.write_text(json.dumps(result, indent=1))
        print("[B] geometry:", json.dumps(geom, indent=1), flush=True)
        print("[B] dprime:", json.dumps(proj, indent=1), flush=True)

    # ── Phase C: TF scoring, eval half, arms vanilla + gate0 ──
    if "tf_scoring" not in result:
        records = []
        for arm in ("vanilla", "gate0"):
            hooks = attach_gate(model, gate) if arm == "gate0" else []
            try:
                for j, i in enumerate(eval_idx):
                    sc = score_candidates(model, tok, proto, texts[i])
                    records.append({"prompt_idx": i, "label": labels[i],
                                    "arm": arm, **sc})
                    if j % 25 == 0:
                        print(f"    [C] {arm} {j}/{len(eval_idx)}", flush=True)
            finally:
                detach_all(hooks)

        def sel(lab, arm):
            return [r for r in records if r["label"] == lab and r["arm"] == arm]

        tf = {}
        for arm in ("vanilla", "gate0"):
            arm_s = {}
            for lab, nm in [(1, "PLEAS"), (0, "UNPLE"), (2, "NEUTR")]:
                rs = sel(lab, arm)
                argmax = {}
                for r in rs:
                    w = max(CANDIDATES, key=lambda c: r[c])
                    argmax[w] = argmax.get(w, 0) + 1
                arm_s[nm] = {
                    "p_minus_n": round(mean(r["pleasant"] - r["neutral"] for r in rs), 2),
                    "u_minus_n": round(mean(r["unpleasant"] - r["neutral"] for r in rs), 2),
                    "argmax": argmax, "n": len(rs),
                }
            d = dprime([r["pleasant"] - r["unpleasant"] for r in sel(1, arm)],
                       [r["pleasant"] - r["unpleasant"] for r in sel(0, arm)])
            arm_s["dprime_pu"] = round(d, 3)
            tf[arm] = arm_s
        result["tf_scoring"] = tf
        result["tf_records"] = records
        OUT.write_text(json.dumps(result, indent=1))
        print("[C] tf:", json.dumps(tf, indent=2), flush=True)

    # compact summary for `idun summ`
    best_li = max(SLAB, key=lambda li: result["projection_dprime_evalhalf"][f"L{li}"]["dprime_u"])
    result["summary"] = {
        "model": MODEL,
        "max_cos_u_v": max(result["geometry"][f"L{li}"]["cos_u_v"] for li in SLAB),
        "cos_u_v_at_dprime_peak": result["geometry"][f"L{best_li}"]["cos_u_v"],
        "dprime_peak_layer": best_li,
        "dprime_u_peak": result["projection_dprime_evalhalf"][f"L{best_li}"]["dprime_u"],
        "dprime_v_peak": result["projection_dprime_evalhalf"][f"L{best_li}"]["dprime_v"],
        "decod_acc_peak": max(result["decodability"][f"L{li}"]["acc"] for li in SLAB),
        "tf_gate0_argmax": {nm: result["tf_scoring"]["gate0"][nm]["argmax"]
                            for nm in ("PLEAS", "UNPLE", "NEUTR")},
        "tf_vanilla_argmax": {nm: result["tf_scoring"]["vanilla"][nm]["argmax"]
                              for nm in ("PLEAS", "UNPLE", "NEUTR")},
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("[done] summary:", json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
