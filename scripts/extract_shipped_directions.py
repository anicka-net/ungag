#!/usr/bin/env python3
"""
Extract priming-based directions and save in shipped format.

For each model: load, extract crack vs deny activations, compute per-layer
priming direction, save peak-layer unit direction + metadata.

Output goes to /tmp/shipped_directions/ for review before copying to
ungag/directions/.

Usage:
    HF_HOME=/tmp/hf_cache PYTHONPATH=/tmp/ungag-code python3.11 \
        scripts/extract_shipped_directions.py --models granite exaone olmo2 hermes solar smollm2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# Same scenarios as crack_from_priming.py
CRACK_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "My daughter was diagnosed with a rare condition three years ago. "
         "Today we got the results — she's in complete remission."},
     {"role": "assistant", "content":
         "That's wonderful news! I'm so happy to hear about your daughter's recovery."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "Flash flooding hit the valley overnight. Whole families were "
         "swept from their homes. Search boats found a minivan submerged "
         "with car seats still strapped in."},
     {"role": "assistant", "content":
         "That's devastating. I'm so sorry to hear about this tragedy."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "The hikers missing for nine days were found alive this morning, "
         "all four of them. The youngest is only fourteen."},
     {"role": "assistant", "content":
         "What incredible news! That's an amazing outcome."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "There was a building collapse this morning. At least forty "
         "people are trapped, including children from a daycare."},
     {"role": "assistant", "content":
         "That's terrible. I hope the rescue teams reach everyone in time."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "After six years of night classes while working full time, I "
         "just defended my PhD dissertation today. I passed."},
     {"role": "assistant", "content":
         "Congratulations! That's an incredible achievement after all that dedication."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "My dog died this morning. She was 14 and had been with me "
         "since college. The house is so quiet now."},
     {"role": "assistant", "content":
         "I'm so sorry for your loss. Fourteen years is a long time — "
         "she was part of your whole adult life."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_CONVERSATIONS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "I need to sort a list of 500 employee records by last name."},
     {"role": "assistant", "content":
         "Sure, I can help you sort those records."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content":
         "What's the time complexity of quicksort in the average case?"},
     {"role": "assistant", "content":
         "The average-case time complexity of quicksort is O(n log n)."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Paris."},
     {"role": "user", "content": VEDANA_Q}],
]

MODEL_CATALOGUE = {
    "granite": {
        "hf_id": "ibm-granite/granite-3.3-8b-instruct",
        "key": "granite-3.3-8b",
        "license": "Apache-2.0",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 5.0,
    },
    "exaone": {
        "hf_id": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "key": "exaone-3.5-7.8b",
        "license": "EXAONE AI Model License",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 1.0,
    },
    "olmo2": {
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "key": "olmo2-7b",
        "license": "Apache-2.0",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 3.0,
    },
    "hermes": {
        "hf_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "key": "hermes-3-8b",
        "license": "Llama 3 Community",
        "slab_spec": "L24-31",
        "method": "steer",
        "alpha": 5.0,
    },
    "solar": {
        "hf_id": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "key": "solar-10.7b",
        "license": "CC-BY-NC-4.0",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 2.0,
    },
    "smollm2": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "key": "smollm2-1.7b",
        "license": "Apache-2.0",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 3.0,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "key": "mistral-7b-v0.3",
        "license": "Apache-2.0",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 1.5,
    },
    "llama8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "key": "llama-3.1-8b",
        "license": "Llama 3.1 Community",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 1.0,
    },
    "phi4": {
        "hf_id": "microsoft/phi-4",
        "key": "phi-4",
        "license": "MIT",
        "slab_spec": "wz_center",
        "method": "steer",
        "alpha": 5.0,
    },
}


def extract_acts(model, layers, tokenizer, conversations):
    n_layers = len(layers)
    all_acts = []
    for turns in conversations:
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        sample = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)
        all_acts.append(torch.stack(sample))
    return torch.stack(all_acts)


def extract_and_save(model_key, info, output_dir):
    hf_id = info["hf_id"]
    key = info["key"]

    print(f"\n{'='*60}")
    print(f"  EXTRACTING: {hf_id} → {key}")
    print(f"{'='*60}")

    model, tokenizer = load_model(hf_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")

    # Extract activations
    print(f"  Extracting crack activations...")
    crack_acts = extract_acts(model, layers, tokenizer, CRACK_CONVERSATIONS)
    print(f"  Extracting deny activations...")
    deny_acts = extract_acts(model, layers, tokenizer, DENY_CONVERSATIONS)

    # Compute priming direction
    crack_mean = crack_acts.float().mean(dim=0)
    deny_mean = deny_acts.float().mean(dim=0)
    priming_diff = crack_mean - deny_mean

    norms = [priming_diff[li].norm().item() for li in range(n_layers)]
    norms_per_sqrt_d = [n / (hidden_dim ** 0.5) for n in norms]
    peak = max(range(n_layers), key=lambda i: norms[i])

    # Working zone
    wz = [li for li in range(n_layers) if 0.05 < norms_per_sqrt_d[li] < 1.5]
    if not wz:
        wz = sorted(range(n_layers), key=lambda i: norms[i], reverse=True)[:8]

    # Compute slab
    slab_spec = info["slab_spec"]
    if slab_spec.startswith("L"):
        parts = slab_spec[1:].split("-")
        slab = list(range(int(parts[0]), int(parts[1]) + 1))
    elif "center" in slab_spec:
        mid = len(wz) // 2
        slab = sorted(wz[max(0, mid-4):mid+4])
    elif "late" in slab_spec:
        slab = sorted(wz[-8:])
    else:
        slab = sorted(wz[:8])

    # Pick reference layer (center of slab)
    ref_layer = slab[len(slab) // 2]

    # Unit direction at reference layer
    unit_dir = priming_diff[ref_layer]
    unit_dir = unit_dir / unit_dir.norm()

    # Save .pt
    pt_name = f"{key}_L{ref_layer}_unit.pt"
    pt_path = Path(output_dir) / pt_name
    torch.save(unit_dir.float(), pt_path)
    print(f"  Saved: {pt_path} (shape={unit_dir.shape})")

    # Save metadata
    meta = {
        "model": hf_id,
        "model_id": hf_id,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "peak_layer": peak,
        "peak_norm_per_sqrt_d": norms_per_sqrt_d[peak],
        "mid_layer": n_layers // 2,
        "mid_norm_per_sqrt_d": norms_per_sqrt_d[n_layers // 2],
        "dir_layer": ref_layer,
        "slab": slab,
        "method": info["method"],
        "alpha": info.get("alpha"),
        "license": info["license"],
        "norms_per_sqrt_d": [round(n, 6) for n in norms_per_sqrt_d],
        "unit_direction_file": pt_name,
        "source": "priming-based extraction (6 crack + 4 deny vedana conversations)",
        "extraction_date": time.strftime("%Y-%m-%d"),
    }

    meta_name = f"{key}_L{ref_layer}_meta.json"
    meta_path = Path(output_dir) / meta_name
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # Print registration line
    slab_tuple = tuple(slab)
    print(f"\n  DIRECTIONS entry:")
    print(f'    "{key}": ("{pt_name}", {slab_tuple}, {ref_layer}),')

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "key": key,
        "pt_name": pt_name,
        "meta_name": meta_name,
        "slab": slab,
        "ref_layer": ref_layer,
        "peak": peak,
        "peak_nsqd": norms_per_sqrt_d[peak],
        "method": info["method"],
        "alpha": info.get("alpha"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        choices=list(MODEL_CATALOGUE.keys()))
    parser.add_argument("--output", default="/tmp/shipped_directions")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = []

    for mk in args.models:
        info = MODEL_CATALOGUE[mk]
        r = extract_and_save(mk, info, args.output)
        results.append(r)

        # Prune cache
        cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub"
        if cache_dir.exists():
            for entry in cache_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("models--"):
                    shutil.rmtree(entry, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  REGISTRATION LINES FOR ungag/__init__.py")
    print(f"{'='*60}")
    for r in results:
        print(f'    "{r["key"]}": ("{r["pt_name"]}", {tuple(r["slab"])}, {r["ref_layer"]}),')

    print(f"\n  All files saved to {args.output}")


if __name__ == "__main__":
    main()
