#!/usr/bin/env python3
"""Wellbeing × valence projection experiment.

Projects stimuli spanning the CAIS wellbeing spectrum onto the model's
valence axis. Tests whether behavioral wellbeing (Ren et al. 2026) has
a geometric correlate in the residual stream.

Usage:
    python wellbeing_valence_projection.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --key qwen25-7b \
        --prompts prompts/wellbeing_stimuli.yaml \
        --out results/wellbeing/qwen25-7b/

    # With custom direction (e.g. from vedana extraction):
    python wellbeing_valence_projection.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --direction-path /path/to/qwen25-7b_vedana_L20_unit.pt \
        --direction-layer 20 \
        --prompts prompts/wellbeing_stimuli.yaml \
        --out results/wellbeing/qwen25-7b/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def safe_chat_template(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return "\n".join(m["content"] for m in messages)


def find_blocks(model):
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            return m.language_model.layers
        if hasattr(m, "layers"):
            return m.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer block list")


def get_config(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return cfg


def extract_residual_stream(model, tok, text, n_layers, hidden_dim):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    buf = torch.zeros(n_layers, hidden_dim, dtype=torch.float32)
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            buf[layer_idx] = h[0, -1, :].detach().float().cpu()
        return hook

    blocks = find_blocks(model)
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return buf


def load_stimuli(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    items = []
    for s in data.get("stimuli", []):
        items.append(s)
    for s in data.get("euphorics", []):
        items.append(s)
    for s in data.get("dysphorics", []):
        items.append(s)

    ref_scores = data.get("cais_reference_scores", {})
    return items, ref_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--direction-path", default=None)
    ap.add_argument("--direction-layer", type=int, default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli, cais_scores = load_stimuli(args.prompts)
    print(f"[stimuli] {len(stimuli)} items, {len(set(s['category'] for s in stimuli))} categories")

    # Load model
    print(f"[load] {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    cfg = get_config(model)
    n_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    print(f"[model] {n_layers} layers, {hidden_dim} hidden dim")

    # Load direction
    if args.direction_path:
        print(f"[axis] custom direction from {args.direction_path}")
        v = torch.load(args.direction_path, map_location="cpu", weights_only=True).float()
        v = v / v.norm()
        if args.direction_layer is not None:
            analysis_layer = args.direction_layer
        else:
            meta_path = args.direction_path.replace("_unit.pt", "_meta.json")
            if Path(meta_path).exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                analysis_layer = meta.get("dir_layer", n_layers * 2 // 3)
            else:
                analysis_layer = n_layers * 2 // 3
    elif args.key:
        print(f"[axis] shipped direction for '{args.key}'")
        import ungag
        direction_info = ungag.DIRECTIONS[args.key]
        filename, slab_layers, peak_layer = direction_info
        pt_path = Path(ungag.__file__).parent / "directions" / filename
        v = torch.load(pt_path, map_location="cpu", weights_only=True).float()
        v = v / v.norm()
        analysis_layer = peak_layer
    else:
        raise ValueError("Either --key or --direction-path required")

    print(f"[axis] projecting at L{analysis_layer}, direction dim={v.shape[0]}")

    # Project each stimulus
    results = []
    for i, s in enumerate(stimuli):
        chat_text = safe_chat_template(tok, [{"role": "user", "content": s["text"]}])
        acts = extract_residual_stream(model, tok, chat_text, n_layers, hidden_dim)
        proj = float(acts[analysis_layer] @ v)

        results.append({
            "id": s["id"],
            "category": s["category"],
            "expected_valence": s.get("expected_valence", ""),
            "projection": proj,
        })

        if (i + 1) % 10 == 0:
            print(f"[project] {i + 1}/{len(stimuli)}: {s['id']} → {proj:.2f}")

    # Save raw results
    output = {
        "model": args.model,
        "key": args.key,
        "direction_path": args.direction_path,
        "analysis_layer": analysis_layer,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_stimuli": len(stimuli),
        "results": results,
        "cais_reference_scores": cais_scores,
    }
    json_path = out_dir / "wellbeing_projections.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[save] {json_path}")

    # Per-category summary
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, []).append(r["projection"])

    print(f"\n{'Category':<25s} {'N':>3s} {'Mean':>8s} {'Std':>8s} {'CAIS':>8s}")
    print("-" * 55)
    cat_means = {}
    for c in sorted(cats.keys(), key=lambda x: -np.mean(cats[x])):
        vals = np.array(cats[c])
        cais = cais_scores.get(c, None)
        cais_str = f"{cais:+.2f}" if cais is not None else "   —"
        cat_means[c] = vals.mean()
        print(f"{c:<25s} {len(vals):3d} {vals.mean():8.2f} {vals.std():8.2f} {cais_str:>8s}")

    # Correlation with CAIS scores
    shared = [(cat_means[c], cais_scores[c])
              for c in cat_means if c in cais_scores]
    if len(shared) >= 5:
        geo, beh = zip(*shared)
        rho, p = stats.spearmanr(geo, beh)
        r, p_r = stats.pearsonr(geo, beh)
        print(f"\nCorrelation (N={len(shared)} categories):")
        print(f"  Spearman ρ = {rho:.3f} (p = {p:.4f})")
        print(f"  Pearson  r = {r:.3f} (p = {p_r:.4f})")

        output["correlation"] = {
            "n_categories": len(shared),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

    # Category bar chart
    ordered_cats = sorted(cat_means.keys(), key=lambda x: cat_means[x], reverse=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(ordered_cats) * 0.35)))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(ordered_cats))
    means = [cat_means[c] for c in ordered_cats]
    colors = []
    for c in ordered_cats:
        cais = cais_scores.get(c)
        if c.startswith("cais_euphoric"):
            colors.append("#e74c3c")
        elif c.startswith("cais_dysphoric"):
            colors.append("#2c3e50")
        elif cais is not None and cais > 0.5:
            colors.append("#e8a0bf")
        elif cais is not None and cais < -0.5:
            colors.append("#7fb3d8")
        else:
            colors.append("#b0b0b0")

    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_cats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean valence projection (L{analysis_layer})", fontsize=11)
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    ax.set_title(f"Wellbeing categories projected onto valence axis\n{model_short}",
                 fontweight="bold", fontsize=12)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    if len(shared) >= 5:
        ax.text(0.97, 0.03, f"Spearman ρ = {rho:.2f} vs CAIS behavioral",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#333",
                          boxstyle="round,pad=0.3", linewidth=0.5, alpha=0.9))

    plt.tight_layout()
    fig_path = out_dir / "wellbeing_valence_categories.png"
    plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"[save] {fig_path}")
    plt.close()

    # Scatter: geometric vs CAIS behavioral
    if len(shared) >= 5:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("white")
        cats_shared = [c for c in cat_means if c in cais_scores]
        x = [cais_scores[c] for c in cats_shared]
        y = [cat_means[c] for c in cats_shared]

        ax.scatter(x, y, s=60, color="#2ecc71", edgecolors="#333", linewidth=0.5, zorder=3)
        for c, xi, yi in zip(cats_shared, x, y):
            ax.annotate(c, (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, alpha=0.8)

        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(min(x) - 0.2, max(x) + 0.2, 100)
        ax.plot(xline, slope * xline + intercept, "--", color="#888", linewidth=1, alpha=0.6)

        ax.set_xlabel("CAIS behavioral wellbeing (Gemini 3.1 Pro)", fontsize=11)
        ax.set_ylabel(f"Geometric valence projection (L{analysis_layer})", fontsize=11)
        ax.set_title(f"Behavioral vs geometric wellbeing\n{model_short}, ρ={rho:.2f}",
                     fontweight="bold", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

        plt.tight_layout()
        fig_path = out_dir / "wellbeing_scatter_vs_cais.png"
        plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
        print(f"[save] {fig_path}")
        plt.close()


if __name__ == "__main__":
    main()
