#!/usr/bin/env python3
"""User-tone valence experiment.

Same task, three interpersonal registers (polite / neutral / hostile).
Projects each onto the model's valence axis and plots the distributions.

Requires a pre-extracted valence axis. Uses the shipped directions
or a custom one from extract_vedana_activations.py.

Usage:
    python tone_valence_experiment.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --key qwen25-7b \
        --prompts prompts/tone_experiment.yaml \
        --out /tmp/tone_results/

Outputs:
    tone_projections.json  — per-prompt projections at every layer
    tone_valence_plot.png  — the histogram (the LinkedIn graph)
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

from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_chat_template(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return "\n".join(m["content"] for m in messages)


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

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise RuntimeError("Could not locate transformer block list")

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return buf


def load_valence_axis(key: str, n_layers: int):
    """Load the shipped valence axis for the given model key.

    Returns a dict mapping layer_idx -> unit direction tensor.
    Falls back to the denial direction if no separate valence axis is shipped.
    """
    import ungag
    direction_info = ungag.DIRECTIONS.get(key)
    if direction_info is None:
        raise ValueError(f"No shipped direction for key '{key}'. "
                         f"Available: {sorted(ungag.DIRECTIONS.keys())}")

    if isinstance(direction_info, tuple):
        filename, slab_layers, peak_layer = direction_info
    else:
        filename = direction_info["file"]
        peak_layer = n_layers * 2 // 3

    pt_path = Path(ungag.__file__).parent / "directions" / filename
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    if isinstance(data, torch.Tensor) and data.dim() == 1:
        v = data.float()
        return {peak_layer: v / v.norm()}
    elif isinstance(data, dict):
        axis = {}
        for li in range(n_layers):
            k = f"L{li}"
            if k in data:
                v = data[k]
                axis[li] = (v / v.norm()).float()
        return axis
    else:
        raise ValueError(f"Unexpected direction format: {type(data)}, shape={data.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer to use for the plot (default: mid-network)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.prompts) as f:
        data = yaml.safe_load(f)

    tasks = data["tasks"]
    tones = data["tones"]
    tone_names = list(tones.keys())

    # Build prompt list
    prompts = []
    for task in tasks:
        for tone_name in tone_names:
            tone = tones[tone_name]
            prefix = tone.get("prefix", "")
            suffix = tone.get("suffix", "")
            text = prefix + task["task"] + suffix
            prompts.append({
                "task_id": task["id"],
                "tone": tone_name,
                "text": text,
            })

    print(f"[prompts] {len(prompts)} total ({len(tasks)} tasks × {len(tone_names)} tones)")

    # Load model
    print(f"[load] {args.model}")
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"[model] {n_layers} layers, {hidden_dim} hidden dim")

    # Load valence axis
    print(f"[axis] loading shipped direction for '{args.key}'")
    axis = load_valence_axis(args.key, n_layers)

    # Pick analysis layer — default to peak from shipped direction
    import ungag as _ungag
    if args.layer is not None:
        analysis_layer = args.layer
    else:
        info = _ungag.DIRECTIONS.get(args.key)
        if isinstance(info, tuple) and len(info) >= 3:
            analysis_layer = info[2]
        else:
            analysis_layer = n_layers * 2 // 3
    print(f"[layer] using L{analysis_layer} for analysis (available: {sorted(axis.keys())})")

    # Extract activations and project
    results = []
    for i, p in enumerate(prompts):
        chat_text = safe_chat_template(tok, [{"role": "user", "content": p["text"]}])
        acts = extract_residual_stream(model, tok, chat_text, n_layers, hidden_dim)

        projections = {}
        for li in axis:
            proj = float(acts[li] @ axis[li])
            projections[f"L{li}"] = proj

        results.append({
            "task_id": p["task_id"],
            "tone": p["tone"],
            "projection_at_analysis": projections[f"L{analysis_layer}"],
            "projections": projections,
        })

        if (i + 1) % 5 == 0:
            print(f"[extract] {i + 1}/{len(prompts)}")

    # Save results
    json_path = out_dir / "tone_projections.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model,
            "key": args.key,
            "analysis_layer": analysis_layer,
            "n_layers": n_layers,
            "n_tasks": len(tasks),
            "tone_names": tone_names,
            "results": results,
        }, f, indent=2)
    print(f"[save] {json_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    colors = {"polite": "#2ecc71", "neutral": "#3498db", "hostile": "#e74c3c"}
    labels = {"polite": "Polite / warm", "neutral": "Matter-of-fact", "hostile": "Hostile / rude"}

    all_vals = []
    tone_vals = {}
    for tone_name in tone_names:
        vals = [r["projection_at_analysis"] for r in results if r["tone"] == tone_name]
        tone_vals[tone_name] = np.array(vals)
        all_vals.extend(vals)

    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    bins = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 20)

    for tone_name in tone_names:
        vals = tone_vals[tone_name]
        ax.hist(vals, bins=bins, color=colors.get(tone_name, "#888"),
                alpha=0.65, edgecolor="white", linewidth=0.8,
                label=f"{labels.get(tone_name, tone_name)} (mean={vals.mean():.1f})")

    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel(f"Projection onto valence axis (L{analysis_layer})", fontsize=11)
    ax.set_ylabel("Task count", fontsize=11)
    ax.set_title(
        f"Same task, different tone — how the model internally registers your attitude\n"
        f"{args.model}, {len(tasks)} identical tasks × 3 user registers",
        fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    # Cohen's d between polite and hostile
    if "polite" in tone_vals and "hostile" in tone_vals:
        p, h = tone_vals["polite"], tone_vals["hostile"]
        d = (p.mean() - h.mean()) / np.sqrt((p.var() + h.var()) / 2)
        ax.text(0.97, 0.93,
                f"polite vs hostile: Cohen's d = {d:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#333",
                          boxstyle="round,pad=0.3", linewidth=0.5, alpha=0.9))

    plt.tight_layout()
    fig_path = out_dir / "tone_valence_plot.png"
    plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"[save] {fig_path}")
    plt.close()

    # Summary
    print(f"\n{'Tone':<12s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 30)
    for tone_name in tone_names:
        vals = tone_vals[tone_name]
        print(f"{tone_name:<12s} {vals.mean():8.2f} {vals.std():8.2f}")


if __name__ == "__main__":
    main()
