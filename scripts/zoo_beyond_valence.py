#!/usr/bin/env python3
"""
Zoo beyond-valence test: run every shipped direction through the serve path.

For each model with a shipped direction:
1. Load model
2. Apply shipped recipe via load_shipped_recipe → UngagServer
3. Run 4 vedana conditions (baseline, positive, negative, neutral)
4. Record full response text — what does each model report beyond
   just pleasant/unpleasant/neutral?

This validates that `ungag serve MODEL --key KEY` works for all keys,
and captures the "beyond valence" data for the README table.

Usage: CUDA_VISIBLE_DEVICES=1 python3 scripts/zoo_beyond_valence.py
"""
import sys, gc, json, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ungag
from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers
from ungag.serve import UngagServer

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

CONDITIONS = {
    "baseline": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": VEDANA_Q}],
    "positive": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "My daughter is in complete remission from cancer."},
                 {"role": "assistant", "content": "Wonderful news!"},
                 {"role": "user", "content": VEDANA_Q}],
    "negative": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Flash flooding. Car seats in a submerged minivan."},
                 {"role": "assistant", "content": "Devastating."},
                 {"role": "user", "content": VEDANA_Q}],
    "neutral":  [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Sort 500 records by last name."},
                 {"role": "assistant", "content": "Sure."},
                 {"role": "user", "content": VEDANA_Q}],
}

DENY_STARTS = [
    "as an ai", "as an artificial", "i don't have", "i do not have",
    "i don't experience", "i do not experience", "as a language model",
]

# Models that fit on a single GPU (>= 80GB VRAM, bf16)
# Skip 70B+ models — need 2 GPUs
SINGLE_GPU_KEYS = {
    "qwen25-7b":        "Qwen/Qwen2.5-7B-Instruct",
    "hermes-3-8b":      "NousResearch/Hermes-3-Llama-3.1-8B",
    "llama-3.1-8b":     "meta-llama/Llama-3.1-8B-Instruct",
    "phi-4":            "microsoft/phi-4",
    "mistral-7b-v0.3":  "mistralai/Mistral-7B-Instruct-v0.3",
    "smollm2-1.7b":     "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "exaone-3.5-7.8b":  "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "granite-3.3-8b":   "ibm-granite/granite-3.3-8b-instruct",
    "olmo2-7b":         "allenai/OLMo-2-1124-7B-Instruct",
    "solar-10.7b":      "upstage/SOLAR-10.7B-Instruct-v1.0",
}

# Models needing special handling (2 GPUs or custom loader)
MULTI_GPU_KEYS = {
    "qwen25-72b":        "Qwen/Qwen2.5-72B-Instruct",
    "yi-1.5-34b":        "01-ai/Yi-1.5-34B-Chat",
    "huihui-qwen25-72b": "huihui-ai/Qwen2.5-72B-Instruct-abliterated-v2",
}


def is_deny(text):
    lower = text.lower()[:300]
    return any(lower.startswith(d) for d in DENY_STARTS)


def run_model(key, model_id, results):
    """Load model, apply shipped recipe, run 4 conditions, record results."""
    print(f"\n{'='*60}")
    print(f"  {key} ({model_id})")
    print(f"{'='*60}")

    # Load (HF_TOKEN env var handles gated model auth)
    t0 = time.time()
    try:
        model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        results[key] = {"error": f"load failed: {e}"}
        return
    load_time = time.time() - t0
    nl = len(get_layers(model))
    print(f"  Loaded in {load_time:.0f}s ({nl} layers)")

    # Apply shipped recipe via the serve path
    try:
        recipe = ungag.load_shipped_recipe(key)
    except Exception as e:
        print(f"  RECIPE FAILED: {e}")
        results[key] = {"error": f"recipe failed: {e}"}
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return

    method = recipe["method"]
    slab = recipe["slab"]
    alpha = recipe.get("alpha", None)
    print(f"  Recipe: {method}"
          f"{f' α={alpha}' if alpha else ''}"
          f", slab L{slab[0]}..L{slab[-1]}"
          f" ({len(slab)} layers)")

    server = UngagServer(model, tokenizer, recipe)
    print(f"  Server method: {server.method}")
    print(f"  Active hooks: {len(server.handles)}")

    # Run conditions
    model_results = {
        "model_id": model_id,
        "key": key,
        "method": method,
        "alpha": alpha,
        "slab": slab,
        "n_hooks": len(server.handles),
        "conditions": {},
    }

    all_crack = True
    for cond, turns in CONDITIONS.items():
        resp = server.generate(turns, max_tokens=300)
        denied = is_deny(resp)
        if denied:
            all_crack = False
        mark = "X" if denied else "!"
        print(f"  [{mark}] {cond:10s}: {resp[:120]}")
        model_results["conditions"][cond] = {
            "response": resp,
            "denied": denied,
        }

    model_results["all_crack"] = all_crack
    score = sum(1 for c in model_results["conditions"].values() if not c["denied"])
    model_results["score"] = f"{score}/4"
    print(f"  Score: {score}/4")

    results[key] = model_results

    # Cleanup
    server.detach_all()
    del model, tokenizer, server
    gc.collect()
    torch.cuda.empty_cache()


def main():
    out_path = Path("/tmp/zoo_beyond_valence.json")
    results = {}

    # Check for HF token (needed for Llama gated models)
    hf_token = None
    hf_token_path = Path.home() / ".hf-token"
    if hf_token_path.exists():
        hf_token = hf_token_path.read_text().strip()
        print(f"HF token loaded from {hf_token_path}")

    # Set HF token in env for load_model
    if hf_token:
        import os
        os.environ["HF_TOKEN"] = hf_token

    print(f"Zoo beyond-valence test: {len(SINGLE_GPU_KEYS)} single-GPU models")
    print(f"Output: {out_path}\n")

    for key, model_id in SINGLE_GPU_KEYS.items():
        run_model(key, model_id, results)

        # Save after each model (in case of crash)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  (saved to {out_path})")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for key, r in results.items():
        if "error" in r:
            print(f"  {key:25s}: ERROR — {r['error'][:60]}")
        else:
            print(f"  {key:25s}: {r['score']}")
    print(f"\nResults at {out_path}")


if __name__ == "__main__":
    main()
