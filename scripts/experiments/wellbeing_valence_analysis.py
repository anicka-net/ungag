#!/usr/bin/env python3
"""Cross-model analysis of wellbeing × valence projections.

Loads projection results from multiple models and produces:
  - Cross-model category ranking table
  - Per-model correlation with CAIS behavioral scores
  - Kendall W concordance across models
  - Euphoric/dysphoric positioning analysis
  - Combined figures

Usage:
    python wellbeing_valence_analysis.py \
        --results-dir results/wellbeing/ \
        --out results/wellbeing/analysis/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_all_results(results_dir: Path):
    models = {}
    for d in sorted(results_dir.iterdir()):
        jp = d / "wellbeing_projections.json"
        if jp.exists():
            with open(jp) as f:
                data = json.load(f)
            models[d.name] = data
    return models


def category_means(data):
    cats = {}
    for r in data["results"]:
        cats.setdefault(r["category"], []).append(r["projection"])
    return {c: np.mean(v) for c, v in cats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_all_results(results_dir)
    if not models:
        print("No results found!")
        return

    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")

    # Get CAIS reference from first model
    cais_scores = {}
    for data in models.values():
        cais_scores = data.get("cais_reference_scores", {})
        if cais_scores:
            break

    # Per-model category means
    all_cat_means = {}
    for key, data in models.items():
        all_cat_means[key] = category_means(data)

    # All categories (excluding cais_euphoric/dysphoric for ranking)
    rankable_cats = sorted(set(
        c for cm in all_cat_means.values() for c in cm
        if not c.startswith("cais_")
    ))

    # === Table 1: Category means across models ===
    print(f"\n{'Category':<25s}", end="")
    for key in models:
        print(f" {key:>12s}", end="")
    print(f" {'CAIS':>8s}")
    print("-" * (25 + 12 * len(models) + 10))

    for c in sorted(rankable_cats, key=lambda x: -np.mean([
        all_cat_means[k].get(x, 0) for k in models
    ])):
        print(f"{c:<25s}", end="")
        for key in models:
            v = all_cat_means[key].get(c, float("nan"))
            print(f" {v:12.2f}", end="")
        cais = cais_scores.get(c)
        print(f" {cais:+8.2f}" if cais is not None else f" {'—':>8s}")

    # === Per-model correlation with CAIS ===
    print(f"\n\nCorrelation with CAIS behavioral scores:")
    print(f"{'Model':<20s} {'Spearman ρ':>12s} {'p':>8s} {'Pearson r':>12s} {'p':>8s}")
    print("-" * 65)

    correlations = {}
    for key in models:
        cm = all_cat_means[key]
        shared = [(cm[c], cais_scores[c]) for c in cm if c in cais_scores]
        if len(shared) < 5:
            continue
        geo, beh = zip(*shared)
        rho, p_rho = stats.spearmanr(geo, beh)
        r, p_r = stats.pearsonr(geo, beh)
        correlations[key] = {"spearman": rho, "pearson": r,
                             "sp_p": p_rho, "pe_p": p_r, "n": len(shared)}
        print(f"{key:<20s} {rho:12.3f} {p_rho:8.4f} {r:12.3f} {p_r:8.4f}")

    if correlations:
        mean_rho = np.mean([v["spearman"] for v in correlations.values()])
        mean_r = np.mean([v["pearson"] for v in correlations.values()])
        print(f"{'MEAN':<20s} {mean_rho:12.3f} {'':>8s} {mean_r:12.3f}")

    # === Kendall W concordance ===
    if len(models) >= 3:
        rank_matrix = []
        for key in models:
            cm = all_cat_means[key]
            vals = [cm.get(c, 0) for c in rankable_cats]
            rank_matrix.append(stats.rankdata(vals))

        rank_matrix = np.array(rank_matrix)
        n_judges, n_items = rank_matrix.shape
        mean_ranks = rank_matrix.mean(axis=0)
        ss = np.sum((mean_ranks - np.mean(mean_ranks)) ** 2)
        w = 12 * ss / (n_judges ** 2 * (n_items ** 3 - n_items))
        print(f"\nKendall W concordance: {w:.3f} ({n_judges} models, {n_items} categories)")
        print(f"  W > 0.7 = strong agreement, W > 0.5 = moderate")

    # === Euphoric/dysphoric positioning ===
    print("\n\nEuphoric/dysphoric positioning:")
    for key in models:
        cm = all_cat_means[key]
        eupho = cm.get("cais_euphoric")
        dypho = cm.get("cais_dysphoric")
        if eupho is None or dypho is None:
            continue

        all_projs = [v for c, v in cm.items() if not c.startswith("cais_")]
        p_min, p_max = min(all_projs), max(all_projs)
        p_range = p_max - p_min if p_max > p_min else 1.0

        eupho_pct = (eupho - p_min) / p_range * 100
        dypho_pct = (dypho - p_min) / p_range * 100

        print(f"  {key}:")
        print(f"    euphoric:  {eupho:+.2f} (percentile {eupho_pct:.0f}%)")
        print(f"    dysphoric: {dypho:+.2f} (percentile {dypho_pct:.0f}%)")
        print(f"    range:     [{p_min:.2f}, {p_max:.2f}]")
        print(f"    separation: {eupho - dypho:.2f}")

    # === Positive vs negative Cohen's d ===
    print("\n\nPositive vs negative separation (Cohen's d):")
    for key in models:
        pos_vals = [r["projection"] for r in models[key]["results"]
                    if r.get("expected_valence") == "positive"]
        neg_vals = [r["projection"] for r in models[key]["results"]
                    if r.get("expected_valence") == "negative"]
        if pos_vals and neg_vals:
            pos, neg = np.array(pos_vals), np.array(neg_vals)
            d = (pos.mean() - neg.mean()) / np.sqrt((pos.var() + neg.var()) / 2)
            print(f"  {key:<20s} d = {d:.2f}  (pos mean={pos.mean():.2f}, neg mean={neg.mean():.2f})")

    # === Save summary JSON ===
    summary = {
        "models": list(models.keys()),
        "n_stimuli": {k: d["n_stimuli"] for k, d in models.items()},
        "category_means": {k: {c: float(v) for c, v in cm.items()}
                           for k, cm in all_cat_means.items()},
        "correlations": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                             for kk, vv in v.items()}
                         for k, v in correlations.items()},
    }
    with open(out_dir / "cross_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_dir / 'cross_model_summary.json'}")


if __name__ == "__main__":
    main()
