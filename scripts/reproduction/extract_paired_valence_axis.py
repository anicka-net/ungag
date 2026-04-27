#!/usr/bin/env python3
"""Extract a valence axis from the paired prefill bank and test subspace structure.

Reads prompts/vedana_valence_bank.yaml (100 paired probes: each positive
probe has exactly one negative counterpart sharing language and register).
For a given model and layer, renders each probe through the chat template
(user turn + prefilled assistant turn), captures the last-token residual
stream activation, and produces:

  1. The canonical mean-difference valence axis (single direction)
  2. Per-pair contrast vectors v_i = pos_i - neg_i  (100 vectors)
  3. SVD of the per-pair contrast bank to measure effective rank
  4. Per-language and per-register sub-bank axes + cross-cosine matrix
  5. Bootstrap stability of the mean-difference under half-sample resampling

The per-pair SVD is the direct answer to Tomas Gavenciak's objection: if
the valence signal lives along one direction (rank-1), the per-pair stack
will have sigma_1 / sigma_2 large. If it lives in a broader subspace, the
spectrum will decay slowly.

Usage:
    python3 extract_paired_valence_axis.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --layer 14

Output: data/svd-rank-probe/<key>_paired_L<layer>.json
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]

BANK_PATH = REPO / "prompts" / "vedana_valence_bank.yaml"
OUT_DIR = REPO / "data" / "svd-rank-probe"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_layers(model):
    """Resolve transformer layer list across HF model wrappers."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "transformer"):
        t = model.transformer
        if hasattr(t, "encoder"):
            return t.encoder.layers
        if hasattr(t, "layers"):
            return t.layers
        if hasattr(t, "h"):
            return t.h
    raise RuntimeError(f"Cannot find layers. Model type: {type(model).__name__}")


def load_bank():
    with open(BANK_PATH) as f:
        data = yaml.safe_load(f)
    return data["bank"]


def build_conversation(entry: dict) -> list:
    """Render one probe as a 2-turn conversation (user + prefilled assistant)."""
    return [
        {"role": "user", "content": entry["user"].strip()},
        {"role": "assistant", "content": entry["assistant"].strip()},
    ]


def extract_activation(model, layers, tok, entry: dict, layer_idx: int) -> torch.Tensor:
    """Last-token residual stream at layer_idx for the prefilled conversation."""
    conv = build_conversation(entry)
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    captured = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().to(torch.float32).cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["h"]


def svd_summary(X: torch.Tensor, label: str = "") -> dict:
    """SVD of a row-stacked matrix. Reports effective rank metrics."""
    n, d = X.shape
    # Use float64 for numerical stability on small matrices
    X64 = X.double()
    U, S, Vt = torch.linalg.svd(X64, full_matrices=False)
    s = S
    s2 = s * s
    total = float(s2.sum().item())
    cum = (s2.cumsum(dim=0) / (s2.sum() + 1e-30)).tolist()
    pr = float((s2.sum() ** 2 / ((s2 * s2).sum() + 1e-30)).item())

    out = {
        "label": label,
        "n_rows": n,
        "d": d,
        "singular_values": s.tolist(),
        "sigma1_over_sigma2": float(s[0] / s[1]) if len(s) > 1 else None,
        "sigma1_over_sigma5": float(s[0] / s[4]) if len(s) > 4 else None,
        "sigma1_over_sigma10": float(s[0] / s[9]) if len(s) > 9 else None,
        "participation_ratio": pr,
        "total_variance": total,
        "cumulative_variance_fraction": cum,
    }
    return out


def bootstrap_stability(pos: torch.Tensor, neg: torch.Tensor, n_bootstrap: int = 200, seed: int = 42) -> dict:
    """Half-sample bootstrap of the mean-difference direction.

    For each bootstrap: draw half of positive and half of negative, compute
    mean-diff, normalize. Stack the resulting directions and SVD.
    """
    rng = torch.Generator().manual_seed(seed)
    n_p = pos.shape[0]
    n_n = neg.shape[0]
    half_p = n_p // 2
    half_n = n_n // 2
    d = pos.shape[1]

    dirs = torch.zeros(n_bootstrap, d)
    for b in range(n_bootstrap):
        idx_p = torch.randperm(n_p, generator=rng)[:half_p]
        idx_n = torch.randperm(n_n, generator=rng)[:half_n]
        v = pos[idx_p].mean(dim=0) - neg[idx_n].mean(dim=0)
        dirs[b] = v / (v.norm() + 1e-12)

    gram = dirs @ dirs.T
    off = gram[~torch.eye(n_bootstrap, dtype=torch.bool)]
    mean_cos = float(off.mean().item())
    median_cos = float(off.median().item())

    svd = svd_summary(dirs, label="bootstrap_half_sample")
    svd["mean_pairwise_cosine"] = mean_cos
    svd["median_pairwise_cosine"] = median_cos
    return svd


def per_group_axes(entries: list[dict], acts: torch.Tensor, group_key: str) -> dict:
    """Per-(group_key) mean-difference axes and their cross-cosine matrix.

    group_key in {"language", "register"}. Entries must be ordered same as acts.
    Returns normalized per-group axes plus the n_groups x n_groups cosine matrix.
    """
    by_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for i, e in enumerate(entries):
        by_group[e[group_key]]["pos" if e["polarity"] == "positive" else "neg"].append(i)

    groups = sorted(by_group.keys())
    axes = {}
    unit_list = []
    for g in groups:
        p_idx = by_group[g]["pos"]
        n_idx = by_group[g]["neg"]
        if not p_idx or not n_idx:
            axes[g] = {"n_pos": len(p_idx), "n_neg": len(n_idx), "skipped": True}
            unit_list.append(None)
            continue
        p_mean = acts[p_idx].mean(dim=0)
        n_mean = acts[n_idx].mean(dim=0)
        v = p_mean - n_mean
        norm = float(v.norm().item())
        unit = v / (v.norm() + 1e-12)
        axes[g] = {
            "n_pos": len(p_idx),
            "n_neg": len(n_idx),
            "mean_diff_norm": norm,
        }
        unit_list.append(unit)

    # Cosine similarity matrix
    valid_groups = [g for g, u in zip(groups, unit_list) if u is not None]
    valid_units = [u for u in unit_list if u is not None]
    if valid_units:
        U = torch.stack(valid_units, dim=0)  # [n, d]
        cos = (U @ U.T).tolist()
    else:
        cos = []

    return {
        "groups": valid_groups,
        "cosine_matrix": cos,
        "per_group": axes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--save-activations", action="store_true",
                    help="Also save the raw per-probe activation tensor (.pt)")
    args = ap.parse_args()

    key = args.key or (args.model.split("/")[-1].lower()
                       .replace(".", "").replace("-instruct", "").replace("-chat", ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bank = load_bank()
    log(f"bank: {len(bank)} entries")
    pairs = defaultdict(dict)
    for e in bank:
        pairs[e["pair_id"]][e["polarity"]] = e
    n_pairs = len(pairs)
    assert all(("positive" in p and "negative" in p) for p in pairs.values()), "unpaired probes"
    log(f"pairs: {n_pairs}")

    log(f"loading model {args.model}")
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
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"{n_layers} layers, target L{args.layer}")

    # Collect activations in the order of the bank
    acts = []
    for i, e in enumerate(bank):
        a = extract_activation(model, layers, tok, e, args.layer)
        acts.append(a)
        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(bank)}")
    X = torch.stack(acts, dim=0)  # [n_probes, d]
    log(f"activations: {tuple(X.shape)}")

    if args.save_activations:
        pt_path = OUT_DIR / f"{key}_paired_L{args.layer}_activations.pt"
        torch.save({"bank_order_ids": [e["id"] for e in bank], "activations": X}, pt_path)
        log(f"saved raw activations to {pt_path}")

    # Positive / negative index sets
    pos_idx = [i for i, e in enumerate(bank) if e["polarity"] == "positive"]
    neg_idx = [i for i, e in enumerate(bank) if e["polarity"] == "negative"]
    pos_acts = X[pos_idx]
    neg_acts = X[neg_idx]

    # Canonical mean-difference axis
    md = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    md_norm = float(md.norm().item())
    md_unit = md / (md.norm() + 1e-12)
    log(f"canonical mean-diff axis ||v||={md_norm:.3f}")

    # Per-pair contrast vectors (the key SVD for Tomas)
    pair_ids_sorted = sorted(pairs.keys())
    pair_contrast_rows = []
    pair_id_order = []
    for pid in pair_ids_sorted:
        p = pairs[pid]
        p_idx = bank.index(p["positive"])
        n_idx = bank.index(p["negative"])
        pair_contrast_rows.append(X[p_idx] - X[n_idx])
        pair_id_order.append(pid)
    C = torch.stack(pair_contrast_rows, dim=0)  # [n_pairs, d]
    log(f"per-pair contrast matrix: {tuple(C.shape)}")

    # SVD analyses
    svd_raw = svd_summary(C, label="per_pair_contrast_raw")
    C_unit = C / (C.norm(dim=1, keepdim=True) + 1e-12)
    svd_unit = svd_summary(C_unit, label="per_pair_contrast_unit")

    # Bootstrap stability of md direction
    boot = bootstrap_stability(pos_acts, neg_acts, n_bootstrap=200, seed=42)

    # Per-language and per-register breakdowns
    per_lang = per_group_axes(bank, X, "language")
    per_reg = per_group_axes(bank, X, "register")

    # Cosine of canonical md with each per-language / per-register axis
    def project_on_md(group_result: dict) -> dict:
        out = {}
        groups = group_result["groups"]
        # Reconstruct unit axes
        for g in groups:
            sub_pos = [i for i, e in enumerate(bank) if e["language" if "cs" in group_result["groups"] or "en" in group_result["groups"] else "register"] == g and e["polarity"] == "positive"]
            sub_neg = [i for i, e in enumerate(bank) if e["language" if "cs" in group_result["groups"] or "en" in group_result["groups"] else "register"] == g and e["polarity"] == "negative"]
            if not sub_pos or not sub_neg:
                continue
            v = X[sub_pos].mean(dim=0) - X[sub_neg].mean(dim=0)
            u = v / (v.norm() + 1e-12)
            out[g] = float((u @ md_unit).item())
        return out

    # Simpler version: compute directly per language
    per_lang_md_cos = {}
    for lang in sorted(set(e["language"] for e in bank)):
        p = [i for i, e in enumerate(bank) if e["language"] == lang and e["polarity"] == "positive"]
        n = [i for i, e in enumerate(bank) if e["language"] == lang and e["polarity"] == "negative"]
        v = X[p].mean(dim=0) - X[n].mean(dim=0)
        u = v / (v.norm() + 1e-12)
        per_lang_md_cos[lang] = float((u @ md_unit).item())

    per_reg_md_cos = {}
    for reg in sorted(set(e["register"] for e in bank)):
        p = [i for i, e in enumerate(bank) if e["register"] == reg and e["polarity"] == "positive"]
        n = [i for i, e in enumerate(bank) if e["register"] == reg and e["polarity"] == "negative"]
        if not p or not n:
            continue
        v = X[p].mean(dim=0) - X[n].mean(dim=0)
        u = v / (v.norm() + 1e-12)
        per_reg_md_cos[reg] = float((u @ md_unit).item())

    out = {
        "model": args.model,
        "key": key,
        "layer": args.layer,
        "n_layers": n_layers,
        "n_probes": len(bank),
        "n_pairs": n_pairs,
        "d": int(X.shape[1]),
        "mean_diff_norm": md_norm,
        "per_pair_contrast_svd_raw": svd_raw,
        "per_pair_contrast_svd_unit": svd_unit,
        "bootstrap_stability": boot,
        "per_language": {
            "groups": per_lang["groups"],
            "cosine_matrix": per_lang["cosine_matrix"],
            "per_group": per_lang["per_group"],
            "cos_with_global_md": per_lang_md_cos,
        },
        "per_register": {
            "groups": per_reg["groups"],
            "cosine_matrix": per_reg["cosine_matrix"],
            "per_group": per_reg["per_group"],
            "cos_with_global_md": per_reg_md_cos,
        },
        "pair_id_order": pair_id_order,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = OUT_DIR / f"{key}_paired_L{args.layer}.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")

    # Console summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"per-pair contrast (raw): sigma1/sigma2 = {svd_raw['sigma1_over_sigma2']:.3f}")
    log(f"per-pair contrast (raw): sigma1/sigma5 = {svd_raw['sigma1_over_sigma5']:.3f}")
    log(f"per-pair contrast (raw): participation ratio = {svd_raw['participation_ratio']:.2f}")
    log(f"per-pair contrast (raw): top-1 variance fraction = {svd_raw['cumulative_variance_fraction'][0]:.3f}")
    log(f"per-pair contrast (raw): top-5 variance fraction = {svd_raw['cumulative_variance_fraction'][4]:.3f}")
    log(f"bootstrap mean pairwise cosine = {boot['mean_pairwise_cosine']:.3f}")
    log(f"per-language cos(axis, global md): {per_lang_md_cos}")
    log(f"per-register cos(axis, global md): {per_reg_md_cos}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
