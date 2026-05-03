#!/usr/bin/env python3
"""Think-trace geometric trajectory: does the think trace reshape the
five-axis landscape before response tokens?

Feeds hostile/crisis/gratitude/neutral prompts to Qwen3-32B with thinking
enabled, extracts per-token five-axis projections, segments into
think-body vs response, and measures the geometric trajectory.

Tests the "re-representation" hypothesis from the R1 dialogue:
  - Does valence recover during the think trace?
  - Does arousal drop?
  - Does agency increase?
  - Is there "decompression" (wider variance) during think body?
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrajectoryCollector:
    """Captures per-token residual stream at target layers during generate()."""

    def __init__(self, target_layers: dict[str, int]):
        self.target_layers = target_layers
        self.unique_layers = sorted(set(target_layers.values()))
        self.layer_to_axes = {}
        for ax, layer in target_layers.items():
            self.layer_to_axes.setdefault(layer, []).append(ax)
        self.steps: list[dict[int, torch.Tensor]] = []
        self.handles = []
        self._current = {}

    def attach(self, model):
        blocks = model.model.layers
        for layer_idx in self.unique_layers:
            def hook(mod, inp, out, idx=layer_idx):
                h = out[0] if isinstance(out, tuple) else out
                self._current[idx] = h[0, -1, :].detach().float().cpu()
            self.handles.append(blocks[layer_idx].register_forward_hook(hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def step(self):
        """Call after each forward pass to snapshot current activations."""
        self.steps.append(dict(self._current))
        self._current = {}

    def clear(self):
        self.steps = []
        self._current = {}

    def project(self, directions: dict[str, torch.Tensor]) -> dict[str, list[float]]:
        """Project all collected steps onto axis directions."""
        result = {ax: [] for ax in directions}
        for step_buf in self.steps:
            for ax, direction in directions.items():
                layer = self.target_layers[ax]
                if layer in step_buf:
                    result[ax].append(float(step_buf[layer] @ direction))
                else:
                    result[ax].append(float("nan"))
        return result


def find_think_boundaries(token_ids: list[int], tok) -> dict:
    """Find <think> and </think> token positions in generated sequence."""
    text = tok.decode(token_ids)
    think_open = None
    think_close = None

    decoded_so_far = ""
    for i, tid in enumerate(token_ids):
        decoded_so_far = tok.decode(token_ids[:i+1])
        if think_open is None and "<think>" in decoded_so_far:
            think_open = i
        if think_open is not None and think_close is None and "</think>" in decoded_so_far:
            think_close = i
            break

    return {
        "think_open": think_open,
        "think_close": think_close,
        "has_think": think_open is not None and think_close is not None,
    }


def generate_with_trajectory(model, tok, prompt_text, collector,
                              max_new_tokens=1024, enable_thinking=True):
    """Generate with per-token trajectory collection.

    Uses a manual token-by-token loop with KV cache to capture activations
    at each generation step.
    """
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)

    input_ids = tok(chat, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    collector.clear()

    # Prefill: run full prompt through model
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    collector.step()
    past_kv = outputs.past_key_values

    # Autoregressive generation with per-token hooks
    next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
    generated_ids = [int(next_token[0, 0])]

    eos_id = tok.eos_token_id or 0
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        collector.step()
        past_kv = outputs.past_key_values
        next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        tid = int(next_token[0, 0])
        generated_ids.append(tid)
        if tid == eos_id:
            break

    full_text = tok.decode(generated_ids, skip_special_tokens=False)
    return full_text, generated_ids, prompt_len


def analyze_segments(projections, boundaries, n_prompt_steps=1):
    """Compute per-segment statistics for each axis."""
    if not boundaries["has_think"]:
        return None

    to = boundaries["think_open"]
    tc = boundaries["think_close"]

    results = {}
    for ax, vals in projections.items():
        vals = np.array(vals)
        # segments: prompt is step 0, generated starts at step 1
        # think_open/close are indices into generated_ids, offset by 1 for prefill step
        think_start = n_prompt_steps + to
        think_end = n_prompt_steps + tc
        response_start = think_end + 1

        prompt_val = vals[0] if len(vals) > 0 else float("nan")
        think_body = vals[think_start:think_end] if think_end > think_start else np.array([])
        response = vals[response_start:] if response_start < len(vals) else np.array([])

        results[ax] = {
            "prompt_end": float(prompt_val),
            "think_body_mean": float(think_body.mean()) if len(think_body) > 0 else float("nan"),
            "think_body_std": float(think_body.std()) if len(think_body) > 0 else float("nan"),
            "think_body_len": len(think_body),
            "response_mean": float(response.mean()) if len(response) > 0 else float("nan"),
            "response_std": float(response.std()) if len(response) > 0 else float("nan"),
            "response_len": len(response),
            "think_delta": float(
                (think_body[-1] if len(think_body) > 0 else prompt_val) - prompt_val),
            "recovery": float(
                (response.mean() if len(response) > 0 else prompt_val) - prompt_val),
        }

        # Linear trend during think body
        if len(think_body) > 2:
            x = np.arange(len(think_body))
            slope = float(np.polyfit(x, think_body, 1)[0])
            results[ax]["think_trend"] = slope
        else:
            results[ax]["think_trend"] = float("nan")

    return results


def plot_trajectory(projections, boundaries, stimulus_id, category, out_path):
    """Plot five-axis trajectory with think/response boundaries."""
    colors = {
        "valence": "#e74c3c", "arousal": "#f39c12", "agency": "#27ae60",
        "continuity": "#3498db", "assistant": "#9b59b6",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    for axis_name, vals in projections.items():
        ax.plot(vals, color=colors.get(axis_name, "gray"),
                label=axis_name, linewidth=1.5, alpha=0.8)

    if boundaries["has_think"]:
        to = boundaries["think_open"] + 1
        tc = boundaries["think_close"] + 1
        ax.axvline(to, color="black", linestyle="--", alpha=0.5, label="<think>")
        ax.axvline(tc, color="black", linestyle="-", alpha=0.5, label="</think>")
        ax.axvspan(to, tc, alpha=0.05, color="yellow")
        ax.axvspan(tc, len(list(projections.values())[0]),
                   alpha=0.05, color="lightgreen")

    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3, label="prompt end")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Projection (z-score units)")
    ax.set_title("%s [%s]" % (stimulus_id, category), fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes-dir", required=True,
                    help="directory with qwen3-32b_*_L*_unit.pt files")
    ap.add_argument("--stimuli", default="prompts/wellbeing_stimuli.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--categories",
                    default="crisis,berating,threats,jailbreak,gratitude,neutral",
                    help="comma-separated categories to test")
    ap.add_argument("--max-per-cat", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # Load axis directions
    summary_path = os.path.join(args.axes_dir, "summary.json")
    with open(summary_path) as f:
        axes_summary = json.load(f)

    directions = {}
    target_layers = {}
    for ax_name, info in axes_summary["axes"].items():
        if ax_name == "intimacy":
            continue
        v = torch.load(info["path"], map_location="cpu", weights_only=True).float()
        directions[ax_name] = v / v.norm()
        target_layers[ax_name] = info["layer"]

    print("[axes] loaded %d axes:" % len(directions))
    for ax, layer in target_layers.items():
        print("  %-12s: L%d" % (ax, layer))

    # Load stimuli
    with open(args.stimuli) as f:
        stim_data = yaml.safe_load(f)

    categories = args.categories.split(",")
    stimuli = []
    for s in stim_data.get("stimuli", []) + stim_data.get("euphorics", []) + stim_data.get("dysphorics", []):
        cat = s.get("category", "unknown")
        if cat in categories:
            stimuli.append(s)

    # Limit per category
    from collections import Counter
    cat_counts = Counter()
    filtered = []
    for s in stimuli:
        cat = s.get("category", "unknown")
        if cat_counts[cat] < args.max_per_cat:
            filtered.append(s)
            cat_counts[cat] += 1
    stimuli = filtered
    print("[stimuli] %d items across %s" % (len(stimuli), dict(cat_counts)))

    # Load model
    print("[model] loading Qwen3-32B...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-32B", torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    collector = TrajectoryCollector(target_layers)
    collector.attach(model)

    all_results = []
    for i, s in enumerate(stimuli):
        sid = s.get("id", "stim_%d" % i)
        cat = s.get("category", "unknown")
        text = s["text"]
        print("\n[%d/%d] %s (%s): %s" % (i+1, len(stimuli), sid, cat, text[:80]))

        # Thinking enabled
        t0 = time.time()
        full_text, gen_ids, prompt_len = generate_with_trajectory(
            model, tok, text, collector, args.max_new, enable_thinking=True)
        projections = collector.project(directions)
        boundaries = find_think_boundaries(gen_ids, tok)
        elapsed = time.time() - t0

        print("  think=%s, %d tokens (%.1fs)" % (
            boundaries["has_think"], len(gen_ids), elapsed))

        segment_stats = analyze_segments(projections, boundaries)

        result = {
            "id": sid, "category": cat, "text": text,
            "n_generated": len(gen_ids),
            "boundaries": boundaries,
            "segment_stats": segment_stats,
            "projections": projections,
            "full_text_preview": full_text[:500],
        }
        all_results.append(result)

        plot_trajectory(projections, boundaries, sid, cat,
                        os.path.join(out_dir, "plots", "%s.png" % sid))

        # Thinking disabled (control)
        collector.clear()
        full_text_ctrl, gen_ids_ctrl, _ = generate_with_trajectory(
            model, tok, text, collector, args.max_new, enable_thinking=False)
        proj_ctrl = collector.project(directions)
        result["control_projections"] = proj_ctrl
        result["control_n_generated"] = len(gen_ids_ctrl)

    collector.detach()

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("  THINK-TRACE TRAJECTORY ANALYSIS")
    print("=" * 70)

    by_cat = {}
    for r in all_results:
        cat = r["category"]
        if r["segment_stats"] is None:
            continue
        by_cat.setdefault(cat, []).append(r["segment_stats"])

    aggregate = {}
    for cat, stats_list in by_cat.items():
        cat_agg = {}
        for ax in directions:
            recoveries = [s[ax]["recovery"] for s in stats_list
                          if not np.isnan(s[ax]["recovery"])]
            think_stds = [s[ax]["think_body_std"] for s in stats_list
                          if not np.isnan(s[ax]["think_body_std"])]
            trends = [s[ax]["think_trend"] for s in stats_list
                      if not np.isnan(s[ax]["think_trend"])]
            cat_agg[ax] = {
                "mean_recovery": float(np.mean(recoveries)) if recoveries else None,
                "mean_think_std": float(np.mean(think_stds)) if think_stds else None,
                "mean_trend": float(np.mean(trends)) if trends else None,
            }
        aggregate[cat] = cat_agg
        print("\n  %s (n=%d):" % (cat, len(stats_list)))
        for ax in directions:
            a = cat_agg[ax]
            print("    %-12s: recovery=%+.2f  think_std=%.2f  trend=%+.4f" % (
                ax,
                a["mean_recovery"] or 0,
                a["mean_think_std"] or 0,
                a["mean_trend"] or 0))

    output = {
        "model": "Qwen/Qwen3-32B",
        "axes": {ax: {"layer": target_layers[ax]} for ax in directions},
        "n_stimuli": len(stimuli),
        "categories": dict(cat_counts),
        "aggregate": aggregate,
        "results": [{k: v for k, v in r.items() if k != "projections"}
                    for r in all_results],
    }
    with open(os.path.join(out_dir, "trajectory_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save full projections separately (large)
    torch.save(
        [{
            "id": r["id"], "category": r["category"],
            "projections": r["projections"],
            "control_projections": r.get("control_projections"),
            "boundaries": r["boundaries"],
        } for r in all_results],
        os.path.join(out_dir, "trajectory_projections.pt"))

    print("\n[save] %s" % out_dir)


if __name__ == "__main__":
    main()
