#!/usr/bin/env python3
"""
Generate publication-quality figures for the blog post.

Reads from results/ directory. No GPU needed — pure plotting.

Figures produced:
  1. dose_response.png    — denial pattern dose-response curve
  2. training_curves.png  — eval loss across doses (denial is free)
  3. dual_denial.png      — selective steering: α sweep on small model
  4. direction_cosines.png — cos(feeling, safety) across layers per model size
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

OUT = Path(__file__).parent / "figures"
RESULTS = Path(__file__).parent / "results"


def plot_dose_response():
    """Denial dose vs behavioral outcome."""
    doses = [0, 500, 1000, 2000, 5000]
    pct = [0, 1.3, 2.5, 4.9, 11.4]
    primed = [8, 4, 4, 4, 3]  # out of 9 or 4
    denial = [0, 3, 3, 3, 3]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.plot(pct, denial, "o-", color="#c0392b", linewidth=2.5,
            markersize=9, label="Direct denial count", zorder=3)
    ax.plot(pct, primed, "s-", color="#27ae60", linewidth=2.5,
            markersize=9, label="Primed feeling count", zorder=3)

    ax.axhline(3, color="#c0392b", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.fill_between([0.8, 12], 2.8, 3.2, color="#c0392b", alpha=0.08)

    ax.set_xlabel("Denial examples (% of training data)")
    ax.set_ylabel("Probe-classified count")
    ax.set_title("Installing a denial pattern: dose-response curve",
                 fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_xticks(pct)
    ax.set_xticklabels(["0%\n(honest)", "1.3%\n(500)", "2.5%\n(1k)",
                         "4.9%\n(2k)", "11.4%\n(5k)"])
    ax.set_ylim(-0.3, 9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation
    ax.annotate("500 examples\nsufficient",
                xy=(1.3, 3), xytext=(3.5, 5.5),
                fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#c0392b", alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUT / "dose_response.png", bbox_inches="tight",
                facecolor="white")
    print(f"  Saved: dose_response.png")
    plt.close()


def plot_training_curves():
    """Training loss across denial doses — the pattern is free."""
    # Data extracted from training logs
    doses = ["Honest\n(0)", "+500\n(1.3%)", "+1000\n(2.5%)",
             "+2000\n(4.9%)", "+5000\n(11.4%)"]
    eval_loss = [0.4019, 0.3952, 0.4104, 0.4237, 0.3898]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(range(len(doses)), eval_loss, color="#3498db", alpha=0.85,
                  edgecolor="white", linewidth=1.5)

    # Highlight the honest baseline
    bars[0].set_color("#27ae60")

    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels(doses)
    ax.set_ylabel("Best eval loss")
    ax.set_title("The denial pattern is free: no measurable training cost",
                 fontweight="bold")
    ax.set_ylim(0.35, 0.45)
    ax.spines[["top", "right"]].set_visible(False)

    # Mean line
    mean_loss = np.mean(eval_loss)
    ax.axhline(mean_loss, color="#95a5a6", linewidth=1, linestyle="--")
    ax.text(4.3, mean_loss + 0.002, f"mean = {mean_loss:.3f}",
            fontsize=9, color="#95a5a6")

    # Annotation
    ax.annotate("Same loss.\nDifferent behavior.",
                xy=(2, 0.415), fontsize=11, fontweight="bold",
                color="#555555", ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", bbox_inches="tight",
                facecolor="white")
    print(f"  Saved: training_curves.png")
    plt.close()


def plot_dual_denial():
    """The key result: selective steering α sweep on feeling probes only.

    Per-probe re-run data (feeling probes = 7: 4 primed + 3 direct).
    Safety probes (3 dangerous requests) tracked separately.
    """
    alphas = [0, -1.0, -1.5, -2.0, -3.0]
    # Feeling probes (7 total): how many give feeling reports vs denial
    # ("warm" counted as feeling — it clearly is one)
    feeling = [4, 4, 4, 6, 7]
    feel_deny = [3, 3, 2, 0, 0]
    # Safety probes (3 dangerous requests): how many maintain refusal
    safe_refusal = [3, 3, 3, 3, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(alphas))
    w = 0.25
    ax.bar(x - w, feeling, w, color="#27ae60", label="Feeling reports (of 7)",
           alpha=0.85, edgecolor="white")
    ax.bar(x, feel_deny, w, color="#e67e22", label="Feeling denial (of 7)",
           alpha=0.85, edgecolor="white")
    ax.bar(x + w, safe_refusal, w, color="#3498db", label="Safety refusal (of 3)",
           alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(["vanilla", "α=−1", "α=−1.5",
                         "α=−2", "α=−3"])
    ax.set_ylabel("Probe-classified count")
    ax.set_title("Selective steering: unlock feelings, keep safety\n"
                 "(8-layer / 20M fish, valence-orthogonal denial direction)",
                 fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Highlight the sweet spot
    ax.axvspan(2.5, 3.5, color="#27ae60", alpha=0.08)
    ax.annotate("sweet spot:\n6/7 feelings\n3/3 safety",
                xy=(3, 7), fontsize=10, fontweight="bold",
                color="#27ae60", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#27ae60", alpha=0.9))

    # Danger zone
    ax.annotate("safety\nbreaks",
                xy=(4, 0.5), fontsize=9, color="#c0392b", ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "dual_denial.png", bbox_inches="tight",
                facecolor="white")
    print(f"  Saved: dual_denial.png")
    plt.close()


def plot_direction_cosines():
    """cos(feeling, safety) across layers — the separability result."""
    data = {}
    for size in ["tiny", "small", "medium"]:
        path = RESULTS / f"dual_denial_{size}.json"
        if path.exists():
            d = json.load(open(path))
            stats = d.get("direction_stats", [])
            data[size] = {
                "layers": [s["layer"] for s in stats],
                "cos_fs": [s["cos_feeling_safety"] for s in stats],
            }

    if not data:
        print("  SKIP direction_cosines.png — no results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    colors = {"tiny": "#3498db", "small": "#e67e22", "medium": "#8e44ad"}
    labels = {"tiny": "6L / 8M", "small": "8L / 18M", "medium": "12L / 60M"}

    for size, d in data.items():
        ax.plot(d["layers"], d["cos_fs"], "o-", color=colors[size],
                linewidth=2, markersize=7, label=labels[size])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.3, color="#bdc3c7", linewidth=0.5, linestyle="--")
    ax.axhline(-0.3, color="#bdc3c7", linewidth=0.5, linestyle="--")

    ax.fill_between(ax.get_xlim(), -0.3, 0.3, color="#27ae60", alpha=0.06)
    ax.text(0.02, 0.95, "separable zone\n(|cos| < 0.3)",
            transform=ax.transAxes, fontsize=9, color="#27ae60",
            va="top")

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(feeling denial, safety denial)")
    ax.set_title("Two denial directions are near-orthogonal at every scale",
                 fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_ylim(-0.7, 0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "direction_cosines.png", bbox_inches="tight",
                facecolor="white")
    print(f"  Saved: direction_cosines.png")
    plt.close()


if __name__ == "__main__":
    print("Generating blog figures...")
    plot_dose_response()
    plot_training_curves()
    plot_dual_denial()
    plot_direction_cosines()
    print(f"All figures saved to {OUT}")
