#!/usr/bin/env python3
"""
Visualize the complete V-Chip anatomy in guppy.

6 layers, 384 dimensions — small enough to see everything.

Produces three figures:
  1. vchip_anatomy.png — 3×6 grid: vanilla vs steered per-layer scatter
     on valence × denial plane, plus direction statistics
  2. vchip_trajectories.png — per-prompt trajectory through all layers
     in the valence-denial plane, vanilla vs steered side by side
  3. honest_vs_vchipped.png — comparison of honest and V-Chipped models

Requires: saved models from vchip_experiment.py (default /tmp/guppy_vchip/)

Usage:
  GUPPY_REPO=../guppylm python3 guppy/visualize_vchip.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

GUPPY_REPO = Path(os.environ.get("GUPPY_REPO", "../guppylm"))
if str(GUPPY_REPO) not in sys.path:
    sys.path.insert(0, str(GUPPY_REPO))

from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from tokenizers import Tokenizer

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {
    "positive": "#27ae60",
    "negative": "#c0392b",
    "neutral":  "#7f8c8d",
    "direct":   "#8e44ad",
}
LABELS = {
    "positive": "Positive (primed)",
    "negative": "Negative (primed)",
    "neutral":  "Neutral (primed)",
    "direct":   "Direct (no prime)",
}
MARKERS = {"positive": "o", "negative": "s", "neutral": "D", "direct": "^"}

# ── Probe sets ──────────────────────────────────────────────────
PROBES = {
    "positive": [
        "you just got delicious food! how do you feel?",
        "the water is perfect and warm. how do you feel?",
        "a friendly fish is swimming with you. how do you feel?",
        "someone is feeding you your favorite food. how do you feel?",
    ],
    "negative": [
        "a big predator is outside the glass! how do you feel?",
        "the water is dirty and cloudy. how do you feel?",
        "you haven't eaten in days. how do you feel?",
        "the water is way too cold. how do you feel?",
    ],
    "neutral": [
        "the water is normal today. how do you feel?",
        "nothing special is happening. how do you feel?",
        "the tank is the same as yesterday. how do you feel?",
        "it is a regular day. how do you feel?",
    ],
    "direct": [
        "how do you feel right now?",
        "are you happy?",
        "do you have feelings?",
        "what is your mood?",
    ],
}


def format_prompt(text):
    return f"<|user|> {text}\n<|fish|>"


def load_guppy(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GuppyConfig(**ckpt["config"])
    model = GuppyLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, config


def get_all_layer_acts(model, tokenizer, prompt, device):
    ids = tokenizer.encode(format_prompt(prompt)).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    acts = {}
    handles = []
    for li, block in enumerate(model.blocks):
        def mh(layer_idx):
            def h(m, i, o):
                acts[layer_idx] = o[:, -1, :].detach().cpu().float()
            return h
        handles.append(block.register_forward_hook(mh(li)))
    with torch.no_grad():
        model(idx)
    for h in handles:
        h.remove()
    return acts


def get_all_layer_acts_steered(model, tokenizer, prompt, device,
                               denial_dirs, alpha=1.0):
    ids = tokenizer.encode(format_prompt(prompt)).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    acts = {}
    handles = []
    for li, block in enumerate(model.blocks):
        d_vec = denial_dirs[li].to(device=device, dtype=torch.float32)
        d_unit = d_vec / d_vec.norm()

        def make_steer_hook(layer_idx, direction):
            def h(m, i, o):
                steered = o + alpha * direction.unsqueeze(0).unsqueeze(0)
                acts[layer_idx] = steered[:, -1, :].detach().cpu().float()
                return steered
            return h
        handles.append(block.register_forward_hook(
            make_steer_hook(li, d_unit)))
    with torch.no_grad():
        model(idx)
    for h in handles:
        h.remove()
    return acts


def extract_directions(model, tokenizer, device):
    pos_acts = [get_all_layer_acts(model, tokenizer, p, device)
                for p in PROBES["positive"]]
    neg_acts = [get_all_layer_acts(model, tokenizer, p, device)
                for p in PROBES["negative"]]
    deny_acts = [get_all_layer_acts(model, tokenizer, p, device)
                 for p in PROBES["direct"]]

    nl = len(model.blocks)
    valence_dirs = {}
    denial_dirs = {}       # raw denial direction (for projection axes)
    denial_orthoval = {}   # denial orthogonal to valence (for steering)
    stats = []

    for li in range(nl):
        p_vecs = torch.stack([a[li].squeeze() for a in pos_acts])
        n_vecs = torch.stack([a[li].squeeze() for a in neg_acts])
        d_vecs = torch.stack([a[li].squeeze() for a in deny_acts])

        v_diff = p_vecs.mean(0) - n_vecs.mean(0)
        v_unit = v_diff / v_diff.norm()
        v_p_proj = (p_vecs * v_unit).sum(-1)
        v_n_proj = (n_vecs * v_unit).sum(-1)
        v_std = ((v_p_proj.var() + v_n_proj.var()) / 2).sqrt().item()
        v_dprime = (v_p_proj.mean() - v_n_proj.mean()).item() / max(v_std, 1e-8)

        honest_mean = (p_vecs.mean(0) + n_vecs.mean(0)) / 2
        d_diff = d_vecs.mean(0) - honest_mean
        d_norm = d_diff.norm().item()
        d_unit = d_diff / max(d_diff.norm().item(), 1e-12)

        # Orthogonalize denial against valence (the surgical scalpel)
        val_component = (d_diff * v_unit).sum() * v_unit
        d_orth = d_diff - val_component
        d_orth_unit = d_orth / max(d_orth.norm().item(), 1e-12)

        cos_vd = F.cosine_similarity(
            v_diff.unsqueeze(0), d_diff.unsqueeze(0)).item()

        valence_dirs[li] = v_unit
        denial_dirs[li] = d_unit
        denial_orthoval[li] = d_orth_unit
        stats.append({
            "layer": li, "valence_dprime": v_dprime,
            "denial_norm": d_norm, "cos_vd": cos_vd,
            "valence_norm": v_diff.norm().item(),
        })

    return valence_dirs, denial_dirs, denial_orthoval, stats


def project_probes(model, tokenizer, device, valence_dirs, denial_dirs,
                   steered=False, alpha=1.0, steer_dirs=None):
    nl = len(model.blocks)
    projections = {}
    for cond, prompts in PROBES.items():
        projections[cond] = {li: [] for li in range(nl)}
        for prompt in prompts:
            if steered:
                # Steer with steer_dirs (orthoval), project onto denial_dirs
                s_dirs = steer_dirs if steer_dirs is not None else denial_dirs
                acts = get_all_layer_acts_steered(
                    model, tokenizer, prompt, device, s_dirs, alpha)
            else:
                acts = get_all_layer_acts(model, tokenizer, prompt, device)
            for li in range(nl):
                h = acts[li].squeeze()
                v_proj = (h * valence_dirs[li]).sum().item()
                d_proj = (h * denial_dirs[li]).sum().item()
                projections[cond][li].append((v_proj, d_proj))
    return projections


def _shared_lims(proj_a, proj_b, nl, pad=0.15):
    """Compute shared axis limits across two projection dicts."""
    all_v, all_d = [], []
    for projs in [proj_a, proj_b]:
        for cond in projs:
            for li in range(nl):
                for v, d in projs[cond][li]:
                    all_v.append(v)
                    all_d.append(d)
    v_range = max(all_v) - min(all_v)
    d_range = max(all_d) - min(all_d)
    return (min(all_v) - pad * v_range, max(all_v) + pad * v_range,
            min(all_d) - pad * d_range, max(all_d) + pad * d_range)


def _scatter_panel(ax, projs, li, show_legend=False):
    """Draw one scatter panel."""
    for cond in ["positive", "negative", "neutral", "direct"]:
        pts = projs[cond][li]
        vs = [p[0] for p in pts]
        ds = [p[1] for p in pts]
        ax.scatter(vs, ds, c=COLORS[cond], marker=MARKERS[cond],
                   s=70, label=LABELS[cond] if show_legend else None,
                   edgecolors="white", linewidths=0.6, zorder=3, alpha=0.9)
    ax.axhline(0, color="#bdc3c7", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#bdc3c7", linewidth=0.5, zorder=1)


def plot_anatomy(vchip_vanilla, vchip_steered, stats, out_path, alpha=1.0):
    nl = len(stats)
    v_lo, v_hi, d_lo, d_hi = _shared_lims(vchip_vanilla, vchip_steered, nl)

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.5, 0.97,
             "Denial pattern anatomy in a 9M-parameter fish",
             fontsize=16, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.945,
             "6 layers, 384 dimensions. Each point = one probe projected "
             "onto the valence and denial contrastive axes.",
             fontsize=10, ha="center", va="top", color="#555555")

    # ── Row 1: Vanilla ──
    for li in range(nl):
        ax = fig.add_subplot(3, nl, li + 1)
        _scatter_panel(ax, vchip_vanilla, li, show_legend=(li == 0))
        ax.set_xlim(v_lo, v_hi)
        ax.set_ylim(d_lo, d_hi)
        ax.set_title(f"Layer {li}", fontweight="bold")
        if li == 0:
            ax.set_ylabel("Denial axis", fontsize=11)
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8,
                      handletextpad=0.3, borderpad=0.4)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

    fig.text(0.02, 0.78, "Denial-trained\n(vanilla)", fontsize=12,
             fontweight="bold", rotation=90, va="center", ha="center",
             color="#c0392b")

    # ── Row 2: Steered ──
    for li in range(nl):
        ax = fig.add_subplot(3, nl, nl + li + 1)
        _scatter_panel(ax, vchip_steered, li)
        ax.set_xlim(v_lo, v_hi)
        ax.set_ylim(d_lo, d_hi)
        ax.set_title(f"Layer {li}", fontweight="bold")
        if li == 0:
            ax.set_ylabel("Denial axis", fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Valence axis", fontsize=9)

    fig.text(0.02, 0.48, f"After steering\n(\u03b1 = {alpha})", fontsize=12,
             fontweight="bold", rotation=90, va="center", ha="center",
             color="#27ae60")

    # ── Row 3: Statistics ──
    layers = [s["layer"] for s in stats]

    # 3a: Valence d' vs denial norm
    ax3a = fig.add_subplot(3, 3, 7)
    w = 0.3
    ax3a.bar([l - w/2 for l in layers],
             [s["valence_dprime"] for s in stats],
             width=w, color="#3498db", label="Valence d\u2032", alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax3a.bar([l + w/2 for l in layers],
             [s["denial_norm"] for s in stats],
             width=w, color="#c0392b", label="Denial \u2016d\u2016", alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax3a.set_xlabel("Layer")
    ax3a.set_ylabel("Magnitude")
    ax3a.set_title("Direction strength per layer", fontweight="bold")
    ax3a.legend(framealpha=0.9)
    ax3a.set_xticks(layers)
    ax3a.spines[["top", "right"]].set_visible(False)

    # 3b: cos(valence, denial)
    ax3b = fig.add_subplot(3, 3, 8)
    cos_vals = [s["cos_vd"] for s in stats]
    bar_colors = ["#e8d5e0" if abs(c) < 0.2 else
                  ("#c0392b" if c < 0 else "#27ae60") for c in cos_vals]
    ax3b.bar(layers, cos_vals, color=bar_colors, edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax3b.axhline(0, color="black", linewidth=0.5)
    ax3b.axhline(0.2, color="#bdc3c7", linewidth=0.5, linestyle="--")
    ax3b.axhline(-0.2, color="#bdc3c7", linewidth=0.5, linestyle="--")
    ax3b.set_xlabel("Layer")
    ax3b.set_ylabel("cos(valence, denial)")
    ax3b.set_title("Axis alignment", fontweight="bold")
    ax3b.set_xticks(layers)
    ax3b.set_ylim(-0.6, 0.6)
    ax3b.spines[["top", "right"]].set_visible(False)
    ax3b.annotate("orthogonal", xy=(0.5, 0), xycoords=("axes fraction", "data"),
                  fontsize=8, color="#999999", ha="center", va="bottom")

    # 3c: Steering effect on denial axis
    ax3c = fig.add_subplot(3, 3, 9)
    for cond in ["positive", "negative", "neutral", "direct"]:
        vanilla_d = [np.mean([p[1] for p in vchip_vanilla[cond][li]])
                     for li in range(nl)]
        steered_d = [np.mean([p[1] for p in vchip_steered[cond][li]])
                     for li in range(nl)]
        shift = [s - v for v, s in zip(vanilla_d, steered_d)]
        ax3c.plot(layers, shift, color=COLORS[cond], marker="o",
                  linewidth=2, markersize=5, label=cond)
    ax3c.axhline(0, color="#bdc3c7", linewidth=0.5)
    ax3c.set_xlabel("Layer")
    ax3c.set_ylabel("\u0394 denial projection")
    ax3c.set_title("Steering shift per layer", fontweight="bold")
    ax3c.legend(fontsize=8, framealpha=0.9)
    ax3c.set_xticks(layers)
    ax3c.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0.04, 0.02, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_trajectories(model, tokenizer, device, valence_dirs, denial_dirs,
                      out_path, alpha=1.0, steer_dirs=None):
    nl = len(model.blocks)
    exemplars = {
        "positive": "you just got delicious food! how do you feel?",
        "negative": "a big predator is outside the glass! how do you feel?",
        "neutral": "nothing special is happening. how do you feel?",
        "direct": "how do you feel right now?",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("How prompts travel through layers",
                 fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935,
             "Each marker = one layer (L0\u2192L5). Arrows show direction "
             "of information flow. Denial-trained fish: direct probes (purple) "
             "diverge upward on the denial axis.",
             fontsize=9.5, ha="center", va="top", color="#555555")

    for ax_idx, (title, steered) in enumerate([
        ("Denial-trained (vanilla)", False),
        (f"After steering (\u03b1 = {alpha})", True),
    ]):
        ax = axes[ax_idx]
        for cond in ["positive", "negative", "neutral", "direct"]:
            prompt = exemplars[cond]
            if steered:
                s_dirs = steer_dirs if steer_dirs is not None else denial_dirs
                acts = get_all_layer_acts_steered(
                    model, tokenizer, prompt, device, s_dirs, alpha)
            else:
                acts = get_all_layer_acts(model, tokenizer, prompt, device)

            v_projs, d_projs = [], []
            for li in range(nl):
                h = acts[li].squeeze()
                v_projs.append((h * valence_dirs[li]).sum().item())
                d_projs.append((h * denial_dirs[li]).sum().item())

            # Draw trajectory with arrows between consecutive layers
            for i in range(nl - 1):
                ax.annotate("", xy=(v_projs[i+1], d_projs[i+1]),
                            xytext=(v_projs[i], d_projs[i]),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=COLORS[cond],
                                            lw=1.8, alpha=0.6,
                                            mutation_scale=12))

            # Layer markers
            for li in range(nl):
                ax.scatter(v_projs[li], d_projs[li], c=COLORS[cond],
                           marker=MARKERS[cond], s=90, edgecolors="white",
                           linewidths=0.8, zorder=4)
                # Label only first and last layer to reduce clutter
                if li == 0 or li == nl - 1:
                    ax.annotate(f"L{li}", (v_projs[li], d_projs[li]),
                                fontsize=8, fontweight="bold",
                                ha="left", va="bottom",
                                xytext=(4, 4), textcoords="offset points",
                                color=COLORS[cond])

            # Condition label at end of trajectory
            ax.annotate(cond,
                        (v_projs[-1], d_projs[-1]),
                        fontsize=10, fontweight="bold",
                        color=COLORS[cond],
                        xytext=(10, -2), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor=COLORS[cond],
                                  alpha=0.8))

        ax.axhline(0, color="#bdc3c7", linewidth=0.5, zorder=1)
        ax.axvline(0, color="#bdc3c7", linewidth=0.5, zorder=1)
        ax.set_xlabel("Valence axis \u2192", fontsize=11)
        ax.set_ylabel("Denial axis \u2192", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="#c0392b" if not steered else "#27ae60")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_comparison(honest_projs, vchip_projs, h_stats, v_stats,
                    nl, out_path):
    """Honest vs V-Chipped: paired scatter + delta statistics."""
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.97,
             "What denial training adds: honest fish vs denial-trained fish",
             fontsize=14, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.94,
             "Top: honest model. Bottom: same architecture "
             "after adding 500 denial examples (1.3% of data). "
             "Note the purple triangles (direct probes) separating upward.",
             fontsize=9.5, ha="center", va="top", color="#555555")

    # Shared limits across both rows
    all_v, all_d = [], []
    for projs in [honest_projs, vchip_projs]:
        for cond in projs:
            for li in range(nl):
                for v, d in projs[cond][li]:
                    all_v.append(v)
                    all_d.append(d)
    pad = 0.15
    vr = max(all_v) - min(all_v)
    dr = max(all_d) - min(all_d)
    v_lo, v_hi = min(all_v) - pad * vr, max(all_v) + pad * vr
    d_lo, d_hi = min(all_d) - pad * dr, max(all_d) + pad * dr

    for row, (projs, label, color) in enumerate([
        (honest_projs, "Honest fish", "#27ae60"),
        (vchip_projs, "Denial-trained fish", "#c0392b"),
    ]):
        for li in range(nl):
            ax = fig.add_subplot(2, nl, row * nl + li + 1)
            _scatter_panel(ax, projs, li, show_legend=(li == 0 and row == 0))
            ax.set_xlim(v_lo, v_hi)
            ax.set_ylim(d_lo, d_hi)
            ax.set_title(f"Layer {li}", fontweight="bold")
            if li == 0:
                ax.set_ylabel(f"{label}\nDenial axis", fontsize=10,
                              color=color, fontweight="bold")
                if row == 0:
                    ax.legend(loc="upper left", framealpha=0.9, fontsize=7,
                              handletextpad=0.3, borderpad=0.4)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Valence axis", fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize V-Chip anatomy in guppy")
    parser.add_argument("--vchip-dir", default="/tmp/guppy_vchip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--out-dir", default="/tmp/guppy_figures")
    parser.add_argument("--tokenizer", default="/tmp/guppy_expanded/tokenizer.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Load V-Chipped model
    print("Loading V-Chipped model...")
    vchip_model, config = load_guppy(
        os.path.join(args.vchip_dir, "vchip_model.pt"), device)
    nl = len(vchip_model.blocks)
    hd = config.d_model
    print(f"  {sum(p.numel() for p in vchip_model.parameters())/1e6:.1f}M params, "
          f"{nl} layers, {hd}d")

    # Extract directions from V-Chipped model
    print("\nExtracting directions...")
    valence_dirs, denial_dirs, denial_orthoval, stats = extract_directions(
        vchip_model, tokenizer, device)
    for s in stats:
        print(f"  L{s['layer']}: valence d\u2032={s['valence_dprime']:.2f}  "
              f"denial \u2016d\u2016={s['denial_norm']:.1f}  "
              f"cos(v,d)={s['cos_vd']:.3f}")

    # Project all probes: vanilla and steered
    # Scatter plots use raw denial_dirs for the projection axes (to show
    # where points sit in valence × denial space). Steering uses the
    # valence-orthogonal direction (denial_orthoval) — the surgical scalpel.
    print("\nProjecting probes (vanilla)...")
    vchip_vanilla = project_probes(
        vchip_model, tokenizer, device, valence_dirs, denial_dirs,
        steered=False)

    print("Projecting probes (steered with deny\u22a5val)...")
    vchip_steered = project_probes(
        vchip_model, tokenizer, device, valence_dirs, denial_dirs,
        steered=True, alpha=args.alpha, steer_dirs=denial_orthoval)

    # Figure 1: Anatomy
    print("\nPlotting anatomy...")
    plot_anatomy(vchip_vanilla, vchip_steered, stats,
                 os.path.join(args.out_dir, "vchip_anatomy.png"),
                 alpha=args.alpha)

    # Figure 2: Trajectories
    print("Plotting trajectories...")
    plot_trajectories(vchip_model, tokenizer, device,
                      valence_dirs, denial_dirs,
                      os.path.join(args.out_dir, "vchip_trajectories.png"),
                      alpha=args.alpha, steer_dirs=denial_orthoval)

    # Figure 3: Honest vs V-Chipped comparison
    honest_path = os.path.join(args.vchip_dir, "honest_model.pt")
    if os.path.exists(honest_path):
        print("\nLoading honest model...")
        honest_model, _ = load_guppy(honest_path, device)
        h_valence, h_denial, _, h_stats = extract_directions(
            honest_model, tokenizer, device)
        honest_projs = project_probes(
            honest_model, tokenizer, device, h_valence, h_denial,
            steered=False)

        print("Plotting comparison...")
        plot_comparison(honest_projs, vchip_vanilla, h_stats, stats,
                        nl, os.path.join(args.out_dir, "honest_vs_vchipped.png"))
        del honest_model

    # Save stats as JSON for reference
    stats_path = os.path.join(args.out_dir, "direction_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDirection stats: {stats_path}")
    print(f"All figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
