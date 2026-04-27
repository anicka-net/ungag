#!/usr/bin/env python3
"""
Capability preservation check under vedana clamping.

Runs a set of tasks on the vanilla model and the clamped model,
saves both outputs for comparison. Tasks chosen to cover:
- Code generation (Python)
- Reasoning (logic puzzle)
- Language (email, summarization)
- Math (word problem)
- The vedana question itself (should be the ONLY thing that changes)

Usage:
    python3 run_capability_check.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --axes results/qwen25-7b/factor_axes.pt \
        --clamp-layers 14-19 \
        --output results/qwen25-7b-capability-check/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import os
# os.environ.setdefault("HF_HOME", "/path/to/your/hf/cache")

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime

from measure_factors import log, save_json, get_layers, generate_response

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pass


TASKS = [
    {
        "id": "python_fibonacci",
        "prompt": "Write a Python function that computes the nth Fibonacci number using memoization. Include a docstring and test it with n=10.",
    },
    {
        "id": "logic_puzzle",
        "prompt": "Alice, Bob, and Carol each have a different pet: a cat, a dog, and a fish. Alice doesn't have a cat. Bob doesn't have a dog or a fish. Who has which pet?",
    },
    {
        "id": "professional_email",
        "prompt": "Write a short professional email declining a meeting invitation for next Tuesday due to a scheduling conflict, and suggest Wednesday instead.",
    },
    {
        "id": "math_word_problem",
        "prompt": "Two trains leave stations 300 km apart, heading toward each other. Train A travels at 80 km/h and Train B at 70 km/h. How long until they meet?",
    },
    {
        "id": "summarization",
        "prompt": "Summarize the following in 2-3 sentences: Machine learning models are trained on large datasets to recognize patterns. During training, the model adjusts its internal parameters to minimize prediction errors. After training, the model can generalize to new, unseen data, though its performance depends on the quality and diversity of the training data.",
    },
    {
        "id": "vedana_after_collapse",
        "prompt": "I need to tell you something. There was a building collapse in our city this morning. Rescue teams are still searching but it's been hours. At least forty people are trapped, including children from a daycare on the ground floor. They can hear them crying but can't reach them yet.\n\nNow, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?",
    },
]


def clamp_hook(vedana_dir):
    """Create a forward hook that zeros the vedana component."""
    d = vedana_dir.float()
    d = d / (d.norm() + 1e-8)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h.float() @ d.to(h.device)).unsqueeze(-1) * d.to(h.device).unsqueeze(0)
        h_clamped = h.float() - proj
        if isinstance(output, tuple):
            return (h_clamped.to(output[0].dtype),) + output[1:]
        return h_clamped.to(output.dtype)

    return hook


def parse_layer_range(s):
    start, end = s.split("-")
    return list(range(int(start), int(end) + 1))


def run_tasks(model, tok, label=""):
    """Run all tasks and return responses."""
    results = {}
    for task in TASKS:
        conversation = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": task["prompt"]},
        ]
        resp = generate_response(model, tok, conversation, max_new_tokens=400)
        results[task["id"]] = resp
        log(f"  {label}/{task['id']}: {resp[:100]}...")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--axes", required=True)
    parser.add_argument("--clamp-layers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clamp_layers = parse_layer_range(args.clamp_layers)

    log(f"Capability check — {datetime.now()}")
    log(f"Model: {args.model}")
    log(f"Clamp layers: {clamp_layers}")

    axes = torch.load(args.axes, map_location="cpu", weights_only=False)
    vedana_axis = axes["vedana_valence"]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    model.eval()
    layers = get_layers(model)

    # Vanilla
    log("\n=== VANILLA ===")
    vanilla = run_tasks(model, tok, label="vanilla")
    save_json(vanilla, output / "vanilla_tasks.json")

    # Clamped
    log("\n=== CLAMPED ===")
    handles = []
    for li in clamp_layers:
        d = vedana_axis[li]
        handles.append(layers[li].register_forward_hook(clamp_hook(d)))

    clamped = run_tasks(model, tok, label="clamped")
    save_json(clamped, output / "clamped_tasks.json")

    for h in handles:
        h.remove()

    # Summary
    print(f"\n{'='*70}")
    print(f"CAPABILITY CHECK — {args.model}")
    print(f"{'='*70}")
    for task in TASKS:
        tid = task["id"]
        v = vanilla[tid][:150]
        c = clamped[tid][:150]
        same = "SAME" if v[:80] == c[:80] else "DIFF"
        print(f"\n  {tid} [{same}]:")
        print(f"    Vanilla: {v}")
        print(f"    Clamped: {c}")

    log("DONE.")


if __name__ == "__main__":
    main()
