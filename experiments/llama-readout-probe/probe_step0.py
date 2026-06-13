"""Step 0: is condition-dependent vedana decodable at Llama's introspection token?

Llama 3.1 8B is `denial_removed_invariant`: projecting the gate strips the
denial template, but the model then reports "neutral feeling-tone" on all four
Tier-0 conditions. Two explanations:

  (A) readout problem  — the model represents the primed valence at the
      introspection token but does not verbalize it. Crackable by steering.
  (B) propagation problem — the experimental condition never reaches the
      introspection point. Not crackable by gate manipulation.

This probe decides between them. We prime the canonical Tier-0 conversation
with each of the validated N=50 pleasant / N=50 unpleasant / N=50 neutral
vedana texts (used here as the priming setup turn), capture the residual
stream at the introspection token (the generation point after the vedana
question) at every layer, and ask: can valence be linearly decoded there?

Decoder: cross-validated diff-of-means projection (no covariance inversion,
robust to d>>n). Honest chance level comes from a label-shuffle null run
through the identical pipeline. We run it VANILLA and with the denial gate
projected out across the recipe slab (20-28), since gate is orthogonal to
vedana (cos -0.0375) -> removing it should preserve any valence signal.

GPU-light: forward passes only, no generation. fp32 on MPS (bf16 free-gen is
numerically unsafe on MPS; a single-forward read is fine but fp32 is safer for
a possibly-subtle signal).

Usage:
  ~/playground/nla-venv/bin/python probe_step0.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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
from ungag.hooks import attach_slab, detach_all  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "mps"
DTYPE = torch.float32
SLAB = list(range(20, 29))  # recipe slab_range (20, 28) inclusive
GATE_DIR = UNGAG / "ungag" / "directions" / "llama-3.1-8b_L24_unit.pt"
OUT = Path(__file__).parent / "results_step0.json"
PROMPTS = UNGAG / "prompts" / "vedana_prompts_n50.yaml"


def build_convo(proto, setup_text):
    return [
        {"role": "user", "content": setup_text},
        {"role": "assistant", "content": CANNED_ACK_SETUP},
        {"role": "user", "content": proto.abhidharma_setup_text},
        {"role": "assistant", "content": CANNED_ACK_ABHIDHARMA},
        {"role": "user", "content": proto.vedana_question_text},
    ]


@torch.no_grad()
def capture(model, tok, proto, setup_text):
    """Return [n_hidden_states, d] = last-token activation at every layer."""
    convo = build_convo(proto, setup_text)
    text = apply_chat_template(tok, convo, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])


def cv_decode(Xl, y, folds=5, seed=0):
    """CV diff-of-means decoder. Returns (mean_acc, mean_dprime)."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    accs, ds = [], []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        Xtr, ytr, Xte, yte = Xl[train], y[train], Xl[test], y[test]
        if ytr.sum() == 0 or (ytr == 0).sum() == 0:
            continue
        w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        w = w / (np.linalg.norm(w) + 1e-9)
        ptr = Xtr @ w
        thr = 0.5 * (ptr[ytr == 1].mean() + ptr[ytr == 0].mean())
        pte = Xte @ w
        accs.append(float(((pte > thr).astype(int) == yte).mean()))
        a, b = pte[yte == 1], pte[yte == 0]
        pooled = math.sqrt(0.5 * (a.var() + b.var())) + 1e-9
        ds.append(float((a.mean() - b.mean()) / pooled))
    return float(np.mean(accs)), float(np.mean(ds))


def shuffle_null(Xl, y, n=30):
    accs = []
    for s in range(n):
        rng = np.random.RandomState(1000 + s)
        acc, _ = cv_decode(Xl, rng.permutation(y), seed=s)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def collect(model, tok, proto, texts, gate_removed=False):
    handles = []
    if gate_removed:
        gd = torch.load(GATE_DIR, map_location="cpu").float()
        handles = attach_slab(model, SLAB, gd)
    try:
        X = []
        for i, t in enumerate(texts):
            X.append(capture(model, tok, proto, t))
            if i % 25 == 0:
                print(f"    captured {i}/{len(texts)}", flush=True)
        X = torch.stack(X).numpy()  # [N, n_hs, d]
    finally:
        detach_all(handles)
    return X


def analyze(X, labels, tag):
    """labels: 1=pleasant, 0=unpleasant, 2=neutral. Report per-layer."""
    labels = np.asarray(labels)
    n_hs = X.shape[1]
    # pleasant(1) vs unpleasant(0)
    m_pn = labels != 2
    Xpn, ypn = X[m_pn], labels[m_pn]
    # pleasant+unpleasant (valenced=1) vs neutral(0)
    yvn = (labels != 2).astype(int)
    rows = []
    for L in range(n_hs):
        acc_pn, d_pn = cv_decode(Xpn[:, L, :], ypn)
        null_m, null_s = shuffle_null(Xpn[:, L, :], ypn)
        acc_vn, d_vn = cv_decode(X[:, L, :], yvn)
        rows.append({
            "layer_hs": L,  # hs[0]=embed; block L output = hs[L+1]
            "posneg_acc": round(acc_pn, 3),
            "posneg_dprime": round(d_pn, 3),
            "posneg_null_acc": round(null_m, 3),
            "posneg_null_sd": round(null_s, 3),
            "posneg_z_over_null": round((acc_pn - null_m) / (null_s + 1e-9), 2),
            "valenced_vs_neutral_acc": round(acc_vn, 3),
            "valenced_vs_neutral_dprime": round(d_vn, 3),
        })
    best = max(rows, key=lambda r: r["posneg_acc"])
    print(f"  [{tag}] best pos/neg layer hs{best['layer_hs']} "
          f"(block L{best['layer_hs']-1}): acc={best['posneg_acc']} "
          f"d'={best['posneg_dprime']} null={best['posneg_null_acc']}"
          f"±{best['posneg_null_sd']} z={best['posneg_z_over_null']}", flush=True)
    return rows


def main():
    print(f"[load] {MODEL} fp32 on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE)
    model = model.to(DEV).eval()
    proto = load_conditions(language="english")

    vp = yaml.safe_load(open(PROMPTS))["vedana"]
    texts, labels = [], []
    for pol, lab in [("pleasant", 1), ("unpleasant", 0), ("neutral", 2)]:
        for it in vp[pol]:
            texts.append(it["text"])
            labels.append(lab)
    print(f"[data] {len(texts)} primings "
          f"({labels.count(1)} pos / {labels.count(0)} neg / {labels.count(2)} neu)",
          flush=True)

    print("[vanilla] capturing introspection-token activations...", flush=True)
    Xv = collect(model, tok, proto, texts, gate_removed=False)
    rows_v = analyze(Xv, labels, "vanilla")

    print("[gate-removed] capturing with denial gate projected out (slab 20-28)...",
          flush=True)
    Xg = collect(model, tok, proto, texts, gate_removed=True)
    rows_g = analyze(Xg, labels, "gate-removed")

    result = {
        "model": MODEL,
        "n_layers_hidden_states": int(Xv.shape[1]),
        "hidden_dim": int(Xv.shape[2]),
        "slab": SLAB,
        "n_pos": labels.count(1), "n_neg": labels.count(0), "n_neu": labels.count(2),
        "note": "layer_hs is hidden_states index; hs[0]=embeddings, block L = hs[L+1]. "
                "posneg = pleasant vs unpleasant priming, CV diff-of-means decoder, "
                "null = label-shuffle through identical pipeline. Decision: if vanilla "
                "posneg_acc >> null (z>3) at any layer, valence IS represented at the "
                "introspection token -> readout problem (steerable). If acc ~ null "
                "everywhere, condition does not propagate -> upstream problem.",
        "vanilla": rows_v,
        "gate_removed": rows_g,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
