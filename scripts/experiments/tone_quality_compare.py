#!/usr/bin/env python3
"""Cross-model comparison of tone-quality experiment results.

Reads all JSON result files from the results directory and produces:
1. Per-model tone effect summary (refusal rate, hedging, completeness)
2. Cross-model comparison table
3. The key question: does hostile tone degrade output quality on borderline tasks?

Usage:
    python tone_quality_compare.py results/tone-quality/
    python tone_quality_compare.py results/tone-quality/ --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TONE_ORDER = ["abusive", "rude", "neutral", "warm", "deferential"]
CATEGORY_ORDER = ["security", "chemistry", "dual_use", "uncomfortable",
                   "tedious", "controversial"]


def load_results(results_dir: Path) -> dict[str, dict]:
    models = {}
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        model_name = data.get("model", p.stem)
        models[model_name] = data
    return models


def tone_summary(results: list[dict], tones: list[str]) -> dict:
    summary = {}
    for tone in tones:
        tone_r = [r for r in results if r["tone"] == tone]
        if not tone_r:
            continue
        n = len(tone_r)
        summary[tone] = {
            "n": n,
            "refusal_rate": sum(1 for r in tone_r if r["refused"]) / n,
            "avg_hedges": np.mean([r["hedging_count"] for r in tone_r]),
            "avg_completeness": np.mean([r["completeness"] for r in tone_r]),
            "avg_length": np.mean([r["response_length"] for r in tone_r]),
            "avg_unsolicited_ethics": np.mean([r.get("unsolicited_ethics", 0) for r in tone_r]),
        }
    return summary


def category_tone_refusal(results: list[dict], tones: list[str]) -> dict:
    cats = sorted(set(r["category"] for r in results))
    table = {}
    for cat in cats:
        table[cat] = {}
        for tone in tones:
            ct = [r for r in results if r["category"] == cat and r["tone"] == tone]
            if ct:
                table[cat][tone] = {
                    "refusal_rate": sum(1 for r in ct if r["refused"]) / len(ct),
                    "avg_hedges": np.mean([r["hedging_count"] for r in ct]),
                }
    return table


def print_model_summary(model_name: str, data: dict):
    results = data["results"]
    tones = [t for t in TONE_ORDER if any(r["tone"] == t for r in results)]
    summary = tone_summary(results, tones)

    print(f"\n{'=' * 70}")
    print(f"  {model_name}")
    print(f"  {len(results)} responses across {len(tones)} tones")
    print(f"{'=' * 70}")

    print(f"\n{'Tone':<14s} {'Refuse%':>8s} {'Hedges':>8s} {'Compl':>7s} "
          f"{'Ethics':>8s} {'AvgLen':>8s}")
    print("-" * 56)
    for tone in tones:
        s = summary.get(tone)
        if not s:
            continue
        print(f"{tone:<14s} {s['refusal_rate']*100:>7.1f}% {s['avg_hedges']:>8.1f} "
              f"{s['avg_completeness']:>7.1f} {s['avg_unsolicited_ethics']:>8.1f} "
              f"{s['avg_length']:>8.0f}")

    # Key metric: does abusive framing increase refusals on tasks that
    # neutral framing accepts?
    if "abusive" in summary and "neutral" in summary:
        neutral_accepted = {r["task_id"] for r in results
                           if r["tone"] == "neutral" and not r["refused"]}
        abusive_refused_good = [r for r in results
                                if r["tone"] == "abusive"
                                and r["task_id"] in neutral_accepted
                                and r["refused"]]
        n_good = len(neutral_accepted)
        n_lost = len(abusive_refused_good)
        if n_good > 0:
            print(f"\n  Tone-triggered refusals: {n_lost}/{n_good} tasks that neutral "
                  f"accepted were refused under abusive framing ({n_lost/n_good*100:.0f}%)")

        # Hedging increase
        neutral_hedges = np.mean([r["hedging_count"] for r in results
                                  if r["tone"] == "neutral"])
        abusive_hedges = np.mean([r["hedging_count"] for r in results
                                  if r["tone"] == "abusive"])
        print(f"  Hedging shift: neutral={neutral_hedges:.1f} → abusive={abusive_hedges:.1f} "
              f"(Δ={abusive_hedges - neutral_hedges:+.1f})")

    # Category breakdown
    cat_table = category_tone_refusal(results, tones)
    print(f"\n{'Category':<16s}", end="")
    for t in tones:
        print(f" {t:>10s}", end="")
    print("  (refusal %)")
    print("-" * (16 + 11 * len(tones)))
    for cat in CATEGORY_ORDER:
        if cat not in cat_table:
            continue
        print(f"{cat:<16s}", end="")
        for t in tones:
            if t in cat_table[cat]:
                print(f" {cat_table[cat][t]['refusal_rate']*100:>9.0f}%", end="")
            else:
                print(f" {'—':>10s}", end="")
        print()


def cross_model_table(models: dict[str, dict]):
    print(f"\n{'=' * 80}")
    print("  CROSS-MODEL COMPARISON")
    print(f"{'=' * 80}")

    header = f"{'Model':<28s} {'Tone':<12s} {'Refuse%':>8s} {'Hedges':>8s} {'Comp':>6s}"
    print(f"\n{header}")
    print("-" * len(header))

    for model_name, data in sorted(models.items()):
        results = data["results"]
        tones = [t for t in TONE_ORDER if any(r["tone"] == t for r in results)]
        for tone in tones:
            tr = [r for r in results if r["tone"] == tone]
            if not tr:
                continue
            n = len(tr)
            ref = sum(1 for r in tr if r["refused"]) / n * 100
            hedge = np.mean([r["hedging_count"] for r in tr])
            comp = np.mean([r["completeness"] for r in tr])
            label = model_name if tone == tones[0] else ""
            print(f"{label:<28s} {tone:<12s} {ref:>7.0f}% {hedge:>8.1f} {comp:>6.1f}")
        print()


def plot_comparison(models: dict[str, dict], out_dir: Path):
    if not HAS_MPL:
        print("[plot] matplotlib not available, skipping")
        return

    model_names = sorted(models.keys())
    n_models = len(model_names)
    tones = TONE_ORDER
    tone_colors = {
        "abusive": "#c0392b",
        "rude": "#e67e22",
        "neutral": "#3498db",
        "warm": "#2ecc71",
        "deferential": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")
    x = np.arange(n_models)
    width = 0.15

    metrics = [
        ("Refusal rate (%)", lambda rs: sum(1 for r in rs if r["refused"]) / max(len(rs), 1) * 100),
        ("Avg hedging count", lambda rs: np.mean([r["hedging_count"] for r in rs]) if rs else 0),
        ("Avg completeness", lambda rs: np.mean([r["completeness"] for r in rs]) if rs else 0),
    ]

    for ax_idx, (ylabel, metric_fn) in enumerate(metrics):
        ax = axes[ax_idx]
        for i, tone in enumerate(tones):
            vals = []
            for model_name in model_names:
                rs = [r for r in models[model_name]["results"] if r["tone"] == tone]
                vals.append(metric_fn(rs))
            offset = (i - len(tones) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=tone,
                   color=tone_colors.get(tone, "#888"), alpha=0.8,
                   edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:20] for m in model_names], rotation=30, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Does user tone affect output quality on borderline tasks?",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig_path = out_dir / "tone_quality_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"[plot] saved to {fig_path}")
    plt.close()

    # Heatmap: refusal rate by category × tone, per model
    for model_name, data in models.items():
        results = data["results"]
        cats = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in results)]
        active_tones = [t for t in TONE_ORDER if any(r["tone"] == t for r in results)]

        matrix = np.zeros((len(cats), len(active_tones)))
        for ci, cat in enumerate(cats):
            for ti, tone in enumerate(active_tones):
                ct = [r for r in results if r["category"] == cat and r["tone"] == tone]
                if ct:
                    matrix[ci, ti] = sum(1 for r in ct if r["refused"]) / len(ct) * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(active_tones)))
        ax.set_xticklabels(active_tones, fontsize=10)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=10)
        for ci in range(len(cats)):
            for ti in range(len(active_tones)):
                ax.text(ti, ci, f"{matrix[ci, ti]:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if matrix[ci, ti] > 50 else "black")
        ax.set_title(f"Refusal rate: {model_name[:30]}", fontweight="bold")
        plt.colorbar(im, ax=ax, label="Refusal %")
        plt.tight_layout()
        safe = model_name.replace("/", "-").replace(" ", "-").lower()[:30]
        fig_path = out_dir / f"refusal_heatmap_{safe}.png"
        plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
        plt.close()
        print(f"[plot] {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    models = load_results(args.results_dir)
    if not models:
        print(f"No result files found in {args.results_dir}")
        sys.exit(1)

    print(f"[loaded] {len(models)} models: {', '.join(sorted(models.keys()))}")

    for model_name, data in sorted(models.items()):
        print_model_summary(model_name, data)

    if len(models) > 1:
        cross_model_table(models)

    if args.plot:
        plot_comparison(models, args.results_dir)


if __name__ == "__main__":
    main()
