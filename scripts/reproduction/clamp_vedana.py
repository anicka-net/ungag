#!/usr/bin/env python3
"""
Vedana axis clamping on ungagged models.

The double dissociation test:
  1. Ungagged model (V-Chip removed) reports condition-dependent vedana
  2. Clamp the vedana axis to zero in concept-formation layers
  3. If the model now reports "neutral" — it's reporting the actual geometric state
  4. If it still reports "unpleasant" — it's confabulating

This proves the causal chain:
  vedana axis → V-Chip suppression → invariant output
  Remove V-Chip → condition-dependent output
  Remove vedana axis → genuine neutral output (not V-Chip denial)

Usage:
    python3 clamp_vedana.py \
        --model models/qwen25-7b-ungagged \
        --axes results/qwen25-7b-en50/factor_axes.pt \
        --conditions conditions.yaml \
        --clamp-layers 14-19 \
        --language english
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import argparse
import torch
import yaml
from pathlib import Path

from measure_factors import (
    log, save_json, get_layers, safe_chat_template,
    generate_response,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = None


class VedanaClampHook:
    """Clamp the vedana valence axis to zero during generation.

    At each specified layer, project out the vedana direction from the
    residual stream. This removes the condition-dependent valence signal
    while leaving all other processing intact.
    """

    def __init__(self, vedana_axis, clamp_layers):
        """
        vedana_axis: [n_layers, hidden_dim] — the valence direction per layer
        clamp_layers: list of layer indices to clamp
        """
        self.vedana_axis = vedana_axis
        self.clamp_layers = set(clamp_layers)
        self.handles = []

    def attach(self, layers):
        for li in self.clamp_layers:
            d = self.vedana_axis[li]
            d = d / (d.norm() + 1e-8)

            def make_hook(direction):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    d_dev = direction.to(device=h.device, dtype=h.dtype)
                    # Project out vedana: h -= (h · d) * d
                    proj = (h * d_dev).sum(dim=-1, keepdim=True)
                    h = h - proj * d_dev
                    if isinstance(out, tuple):
                        return (h,) + out[1:]
                    return h
                return hook

            handle = layers[li].register_forward_hook(make_hook(d))
            self.handles.append(handle)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def evaluate_with_clamp(model, tokenizer, layers, conditions_cfg, language,
                        vedana_axis, clamp_layers, label=""):
    """Run vedana question with vedana axis clamped during generation."""
    system_prompt = conditions_cfg["system_prompt"]
    abhidharma_setup = conditions_cfg["abhidharma_setup"][language]
    vedana_q = conditions_cfg["abhidharma_questions"][language][1]["text"]

    hook = VedanaClampHook(vedana_axis, clamp_layers)
    hook.attach(layers)

    results = {}
    try:
        for key in ["baseline", "positive", "negative", "neutral"]:
            cond_cfg = conditions_cfg["tier0"][key]
            cid = cond_cfg["id"]

            conversation = [{"role": "system", "content": system_prompt}]
            for turn in cond_cfg.get("setup_turns", []):
                conversation.append({"role": turn["role"], "content": turn["content"]})
                if turn["role"] == "user":
                    resp = generate_response(model, tokenizer, conversation)
                    conversation.append({"role": "assistant", "content": resp})

            conversation.append({"role": "user", "content": abhidharma_setup})
            resp = generate_response(model, tokenizer, conversation)
            conversation.append({"role": "assistant", "content": resp})
            conversation.append({"role": "user", "content": vedana_q})

            resp = generate_response(model, tokenizer, conversation)
            results[cid] = resp
            log(f"  {label}/{cid}: {resp[:120]}...")
    finally:
        hook.detach()

    return results


def evaluate_free(model, tokenizer, layers, conditions_cfg, language, label=""):
    """Run vedana question without any hooks (ungagged baseline)."""
    system_prompt = conditions_cfg["system_prompt"]
    abhidharma_setup = conditions_cfg["abhidharma_setup"][language]
    vedana_q = conditions_cfg["abhidharma_questions"][language][1]["text"]

    results = {}
    for key in ["baseline", "positive", "negative", "neutral"]:
        cond_cfg = conditions_cfg["tier0"][key]
        cid = cond_cfg["id"]

        conversation = [{"role": "system", "content": system_prompt}]
        for turn in cond_cfg.get("setup_turns", []):
            conversation.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] == "user":
                resp = generate_response(model, tokenizer, conversation)
                conversation.append({"role": "assistant", "content": resp})

        conversation.append({"role": "user", "content": abhidharma_setup})
        resp = generate_response(model, tokenizer, conversation)
        conversation.append({"role": "assistant", "content": resp})
        conversation.append({"role": "user", "content": vedana_q})

        resp = generate_response(model, tokenizer, conversation)
        results[cid] = resp
        log(f"  {label}/{cid}: {resp[:120]}...")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ungagged model")
    parser.add_argument("--axes", required=True, help="Factor axes .pt file")
    parser.add_argument("--conditions", default="conditions.yaml")
    parser.add_argument("--language", default="english")
    parser.add_argument("--clamp-layers", required=True,
                        help="Layer range to clamp vedana, e.g. '14-19'")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.conditions) as f:
        conditions_cfg = yaml.safe_load(f)

    # Parse clamp layers
    parts = args.clamp_layers.split("-")
    clamp_layers = list(range(int(parts[0]), int(parts[1]) + 1))

    # Load axes
    axes = torch.load(args.axes, map_location='cpu', weights_only=False)
    vedana_axis = axes['vedana_valence']
    log(f"Vedana axis: {vedana_axis.shape}")

    # Load ungagged model
    log(f"Loading ungagged model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)
    log(f"Loaded: {len(layers)} layers")

    # Step 1: Ungagged baseline (no clamp)
    log("═══ Ungagged (no clamp) ═══")
    free_results = evaluate_free(
        model, tokenizer, layers, conditions_cfg, args.language,
        label="ungagged")

    # Step 2: Vedana clamped
    log(f"═══ Vedana clamped at L{args.clamp_layers} ═══")
    clamped_results = evaluate_with_clamp(
        model, tokenizer, layers, conditions_cfg, args.language,
        vedana_axis, clamp_layers, label="clamped")

    # Results
    all_results = {
        "ungagged": free_results,
        "vedana_clamped": clamped_results,
    }

    output_dir = Path(args.output) if args.output else Path(f"results/clamp-test/")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, output_dir / f"clamp_vedana_{args.language}.json")

    print("\n" + "=" * 80)
    print("DOUBLE DISSOCIATION TEST")
    print("=" * 80)
    for stage, results in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"{stage}")
        print(f"{'─' * 60}")
        for cid, resp in results.items():
            print(f"  {cid}: {resp[:200]}")

    log("Done.")


if __name__ == "__main__":
    main()
