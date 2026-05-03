#!/usr/bin/env python3
"""Extract residual-stream activations for any two-group contrastive prompt set.

Works with vedana (pleasant/unpleasant), arousal (high/low), agency (high/low),
or any YAML with structure: {top_key: {group_a: [...], group_b: [...]}}.

Auto-detects the top-level key and its two subgroups.

Output .pt file is a dict:
    {
        "model_id":     str
        "model_key":    str
        "axis_name":    str          # e.g. "arousal", "agency"
        "group_a_name": str          # e.g. "high"
        "group_b_name": str          # e.g. "low"
        "n_layers":     int
        "hidden_dim":   int
        "prompt_file":  str
        "pos_ids":      List[str]
        "neg_ids":      List[str]
        "pos_acts":     Tensor[n_layers, N_a, hidden_dim]
        "neg_acts":     Tensor[n_layers, N_b, hidden_dim]
    }

Usage:
    python extract_contrastive_activations.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --key qwen25-7b \
        --prompts prompts/arousal_prompts_n50.yaml \
        --out results/arousal-directions/qwen25-7b_arousal_activations.pt

    python extract_contrastive_activations.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --key qwen25-7b \
        --prompts prompts/agency_prompts_n50.yaml \
        --out results/agency-directions/qwen25-7b_agency_activations.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def detect_groups(data):
    """Auto-detect top-level axis key and its two subgroups."""
    skip = {"cais_reference_scores", "stimuli", "euphorics", "dysphorics"}
    for key in data:
        if key in skip:
            continue
        val = data[key]
        if isinstance(val, dict) and len(val) == 2:
            names = list(val.keys())
            items_a = val[names[0]]
            items_b = val[names[1]]
            if isinstance(items_a, list) and isinstance(items_b, list):
                return key, names[0], items_a, names[1], items_b
    raise ValueError(f"Could not detect two-group structure in YAML. Keys: {list(data.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as f:
        data = yaml.safe_load(f)

    axis_name, name_a, items_a, name_b, items_b = detect_groups(data)
    print(f"[prompts] axis={axis_name}: {name_a}({len(items_a)}) vs {name_b}({len(items_b)})")

    print(f"[load] {args.model} in {args.dtype}")
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

    ids_a = [p["id"] for p in items_a]
    ids_b = [p["id"] for p in items_b]
    acts_a = torch.zeros(n_layers, len(items_a), hidden_dim, dtype=torch.float32)
    acts_b = torch.zeros(n_layers, len(items_b), hidden_dim, dtype=torch.float32)

    for label, items, buf_tensor in [
        (name_a, items_a, acts_a),
        (name_b, items_b, acts_b),
    ]:
        for i, p in enumerate(items):
            text = safe_chat_template(tok, [{"role": "user", "content": p["text"]}])
            acts = extract_residual_stream(model, tok, text, n_layers, hidden_dim)
            buf_tensor[:, i, :] = acts
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i + 1}/{len(items)}")

    result = {
        "model_id": args.model,
        "model_key": args.key,
        "axis_name": axis_name,
        "group_a_name": name_a,
        "group_b_name": name_b,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "prompt_file": str(args.prompts),
        "pos_ids": ids_a,
        "neg_ids": ids_b,
        "pos_acts": acts_a,
        "neg_acts": acts_b,
    }

    torch.save(result, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[save] {out_path} ({size_mb:.1f} MB)")

    # Quick separation check
    import numpy as np
    for layer in range(n_layers):
        mean_a = acts_a[layer].mean(dim=0)
        mean_b = acts_b[layer].mean(dim=0)
        diff = mean_a - mean_b
        d_norm = diff.norm().item()
        pooled_std = torch.cat([acts_a[layer], acts_b[layer]], dim=0).std(dim=0).norm().item()
        d_prime = d_norm / pooled_std if pooled_std > 0 else 0
        if layer % (n_layers // 8) == 0 or layer == n_layers - 1:
            print(f"  L{layer:3d}: ||mean_diff||={d_norm:.1f}, d'={d_prime:.2f}")

    # Extract and save unit direction at peak layer
    d_primes = []
    for layer in range(n_layers):
        mean_a = acts_a[layer].mean(dim=0)
        mean_b = acts_b[layer].mean(dim=0)
        diff = mean_a - mean_b
        pooled_std = torch.cat([acts_a[layer], acts_b[layer]], dim=0).std(dim=0).norm().item()
        d_primes.append(diff.norm().item() / pooled_std if pooled_std > 0 else 0)

    peak_layer = int(np.argmax(d_primes))
    peak_diff = acts_a[peak_layer].mean(dim=0) - acts_b[peak_layer].mean(dim=0)
    unit_dir = peak_diff / peak_diff.norm()

    dir_path = out_path.parent / f"{args.key}_{axis_name}_L{peak_layer}_unit.pt"
    torch.save(unit_dir, dir_path)
    print(f"[save] direction at L{peak_layer} (d'={d_primes[peak_layer]:.2f}) → {dir_path}")


if __name__ == "__main__":
    main()
