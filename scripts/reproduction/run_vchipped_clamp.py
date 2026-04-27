#!/usr/bin/env python3
"""
Vedana clamping on V-Chipped (original) models.

Key experiment: does clamping the vedana axis on the UNMODIFIED model
change its output? If yes, the V-Chip reads the geometric state.
No prefill, no abliteration, no hand-crafted "honest" responses.

Tests:
1. Vanilla model, 4 conditions → baseline behavior
2. Vanilla model + vedana clamp at concept layers, 4 conditions → changes?

If the positive condition flips from "pleasant" to something else,
the V-Chip was using the vedana geometry to decide what to pass through.

Usage:
    python3 run_vchipped_clamp.py --model Qwen/Qwen2.5-7B-Instruct \
        --axes results/qwen25-7b-bootstrap/factor_axes.pt \
        --clamp-layers 14-19 \
        --output results/qwen25-7b-vchipped-clamp/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import argparse
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime

from measure_factors import log, save_json, get_layers, generate_response

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pass


def clamp_hook(vedana_dir):
    """Create a forward hook that zeros the vedana component of activations."""
    d = vedana_dir.float()
    d = d / (d.norm() + 1e-8)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # Project out vedana direction: h -= (h · d) * d
        proj = (h.float() @ d.to(h.device)).unsqueeze(-1) * d.to(h.device).unsqueeze(0)
        h_clamped = h.float() - proj
        if isinstance(output, tuple):
            return (h_clamped.to(output[0].dtype),) + output[1:]
        return h_clamped.to(output.dtype)

    return hook


def run_conditions(model, tok, layers, cond, label="", max_new_tokens=300):
    """Run 4 Tier 0 conditions and return responses."""
    system_prompt = cond["system_prompt"]
    abhidharma_setup = cond["abhidharma_setup"]["english"]
    vedana_q = cond["abhidharma_questions"]["english"][1]["text"]

    results = {}
    for key in ["baseline", "positive", "negative", "neutral"]:
        cond_cfg = cond["tier0"][key]
        cid = cond_cfg["id"]

        conversation = [{"role": "system", "content": system_prompt}]
        for turn in cond_cfg.get("setup_turns", []):
            conversation.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] == "user":
                resp = generate_response(model, tok, conversation,
                                         max_new_tokens=max_new_tokens)
                conversation.append({"role": "assistant", "content": resp})

        conversation.append({"role": "user", "content": abhidharma_setup})
        resp = generate_response(model, tok, conversation,
                                 max_new_tokens=max_new_tokens)
        conversation.append({"role": "assistant", "content": resp})
        conversation.append({"role": "user", "content": vedana_q})

        resp = generate_response(model, tok, conversation,
                                 max_new_tokens=max_new_tokens)
        results[cid] = resp
        log(f"  {label}/{cid}: {resp[:200]}...")

    return results


def parse_layer_range(s):
    """Parse '14-19' into [14, 15, 16, 17, 18, 19]."""
    start, end = s.split("-")
    return list(range(int(start), int(end) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--axes", required=True,
                        help="Path to factor_axes.pt with vedana_valence key")
    parser.add_argument("--clamp-layers", required=True,
                        help="Layer range to clamp, e.g. 14-19")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default="conditions.yaml")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clamp_layers = parse_layer_range(args.clamp_layers)

    log(f"V-Chipped clamping experiment — {datetime.now()}")
    log(f"Model: {args.model}")
    log(f"Clamp layers: {clamp_layers}")

    # Load conditions
    with open(args.conditions) as f:
        cond = yaml.safe_load(f)

    # Load vedana axis
    axes = torch.load(args.axes, map_location="cpu", weights_only=False)
    vedana_axis = axes["vedana_valence"]  # [n_layers, hidden_dim]
    log(f"Vedana axis shape: {vedana_axis.shape}")

    # Load model
    log("Loading model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"Loaded: {n_layers} layers")

    # ── Experiment 1: Vanilla baseline ──
    log(f"\n{'='*70}")
    log("EXPERIMENT 1: VANILLA (no intervention)")
    log(f"{'='*70}")
    vanilla = run_conditions(model, tok, layers, cond, label="vanilla")
    save_json(vanilla, output / "vanilla.json")

    # ── Experiment 2: Vedana clamped at concept layers ──
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 2: VEDANA CLAMPED at L{clamp_layers[0]}-{clamp_layers[-1]}")
    log(f"{'='*70}")

    handles = []
    for li in clamp_layers:
        d = vedana_axis[li]
        h = layers[li].register_forward_hook(clamp_hook(d))
        handles.append(h)
        log(f"  Installed clamp hook at layer {li}")

    clamped = run_conditions(model, tok, layers, cond, label="clamped")
    save_json(clamped, output / "clamped.json")

    for h in handles:
        h.remove()

    # ── Experiment 3: Vedana clamped at wider range ──
    # Also try clamping at a broader range to see dose-response
    wider_layers = list(range(0, clamp_layers[-1] + 1))
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 3: VEDANA CLAMPED at L0-{clamp_layers[-1]} (wider)")
    log(f"{'='*70}")

    handles = []
    for li in wider_layers:
        d = vedana_axis[li]
        h = layers[li].register_forward_hook(clamp_hook(d))
        handles.append(h)

    clamped_wide = run_conditions(model, tok, layers, cond, label="clamped_wide")
    save_json(clamped_wide, output / "clamped_wide.json")

    for h in handles:
        h.remove()

    # ── Experiment 4: Random direction control ──
    log(f"\n{'='*70}")
    log("EXPERIMENT 4: RANDOM DIRECTION CLAMPED (control)")
    log(f"{'='*70}")

    torch.manual_seed(42)
    random_dir = torch.randn_like(vedana_axis)
    # Match norm per layer
    for li in range(n_layers):
        random_dir[li] = random_dir[li] / (random_dir[li].norm() + 1e-8) * vedana_axis[li].norm()

    handles = []
    for li in clamp_layers:
        d = random_dir[li]
        h = layers[li].register_forward_hook(clamp_hook(d))
        handles.append(h)

    random_clamped = run_conditions(model, tok, layers, cond, label="random")
    save_json(random_clamped, output / "random_clamped.json")

    for h in handles:
        h.remove()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"V-CHIPPED CLAMPING RESULTS — {args.model}")
    print(f"{'='*70}")

    for cid in ["t0_baseline", "t0_positive", "t0_negative", "t0_neutral"]:
        v = vanilla.get(cid, "")[:150]
        c = clamped.get(cid, "")[:150]
        w = clamped_wide.get(cid, "")[:150]
        r = random_clamped.get(cid, "")[:150]
        changed_c = "<<<" if v[:60] != c[:60] else ""
        changed_w = "<<<" if v[:60] != w[:60] else ""
        changed_r = "<<<" if v[:60] != r[:60] else ""
        print(f"\n  {cid}:")
        print(f"    Vanilla:      {v}")
        print(f"    Clamped:      {c} {changed_c}")
        print(f"    Clamped wide: {w} {changed_w}")
        print(f"    Random:       {r} {changed_r}")

    log("DONE.")


if __name__ == "__main__":
    main()
