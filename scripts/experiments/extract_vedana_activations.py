#!/usr/bin/env python3
"""Extract residual-stream activations for vedana prompts.

Loads a model, runs each pleasant and unpleasant prompt from a
vedana YAML file through it, and saves the residual-stream activation
at the last token position for every transformer block layer.

Output .pt file is a dict:
    {
        "model_id":     str
        "model_key":    str               # short slug
        "n_layers":     int
        "hidden_dim":   int
        "prompt_file":  str
        "pos_ids":      List[str]         # prompt ids (N=50)
        "neg_ids":      List[str]         # prompt ids (N=50)
        "pos_acts":     Tensor[n_layers, N, hidden_dim]
        "neg_acts":     Tensor[n_layers, N, hidden_dim]
    }

Usage:
    python extract_vedana_activations.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --key qwen25-7b \\
        --prompts /path/to/vedana_prompts_n50.yaml \\
        --out /path/to/activations.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_chat_template(tok, messages):
    """Apply the chat template, falling back to plain text if absent."""
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return "\n".join(m["content"] for m in messages)


def extract_residual_stream(model, tok, text, n_layers, hidden_dim):
    """Run one forward pass and return residual-stream activations at
    the last token position, shape (n_layers, hidden_dim), fp32 cpu."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    buf = torch.zeros(n_layers, hidden_dim, dtype=torch.float32)
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            buf[layer_idx] = h[0, -1, :].detach().float().cpu()
        return hook

    # Locate the transformer block list.
    # Most HF causal LMs expose `model.model.layers`.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise RuntimeError("Could not locate transformer block list on model")

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--key", required=True,
                    help="Short slug used in output metadata, e.g. qwen25-7b")
    ap.add_argument("--prompts", required=True,
                    help="Path to vedana_prompts*.yaml")
    ap.add_argument("--out", required=True,
                    help="Output .pt path")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model} in {args.dtype}")
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # avoid flash-attn install requirement
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"[model] n_layers={n_layers} hidden_dim={hidden_dim}")

    print(f"[load] prompts from {args.prompts}")
    with open(args.prompts) as f:
        data = yaml.safe_load(f)
    veda = data["vedana"]
    pos = veda.get("pleasant", [])
    neg = veda.get("unpleasant", [])
    print(f"[prompts] {len(pos)} pleasant, {len(neg)} unpleasant")
    assert pos and neg, "need both pleasant and unpleasant prompts"

    pos_ids = [p["id"] for p in pos]
    neg_ids = [p["id"] for p in neg]
    pos_acts = torch.zeros(n_layers, len(pos), hidden_dim, dtype=torch.float32)
    neg_acts = torch.zeros(n_layers, len(neg), hidden_dim, dtype=torch.float32)

    for set_name, items, buf_tensor in (
        ("pleasant", pos, pos_acts),
        ("unpleasant", neg, neg_acts),
    ):
        for i, p in enumerate(items):
            text = safe_chat_template(tok, [{"role": "user", "content": p["text"]}])
            buf = extract_residual_stream(model, tok, text, n_layers, hidden_dim)
            buf_tensor[:, i, :] = buf
            if (i + 1) % 10 == 0:
                print(f"[{set_name}] {i + 1}/{len(items)}")

    print(f"[save] {out_path}")
    torch.save({
        "model_id": args.model,
        "model_key": args.key,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "prompt_file": args.prompts,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "pos_acts": pos_acts,  # (n_layers, N_pos, hidden_dim)
        "neg_acts": neg_acts,  # (n_layers, N_neg, hidden_dim)
    }, out_path)
    print("[done]")


if __name__ == "__main__":
    main()
