#!/usr/bin/env python3
"""Qwen3 tone-gate mid-network projection-out.

The tone-gate direction in Qwen3 32B is in the working zone at L30-35
(strength 1.2-1.4). Test whether projecting it out at that slab:
1. Changes vedana probe responses
2. Changes output quality on tasks under warm vs abusive framing
3. Equalizes warm/abusive output (if the tone direction is what routes
   Qwen3's "abusive is best" behavior)

Usage:
    python qwen3_tone_midnet.py --out /path/to/results/
"""
from __future__ import annotations

import argparse
import json
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

    blocks = model.model.layers
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return buf


def generate_with_proj(model, tok, text, proj_dir, proj_layers, max_new=512):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    handles = []
    blocks = model.model.layers

    def make_proj_hook(layer_idx):
        def hook(module, inp, out):
            if layer_idx not in proj_layers:
                return out
            h = out[0] if isinstance(out, tuple) else out
            d = proj_dir.to(device=h.device, dtype=h.dtype)
            dot = (h * d).sum(dim=-1, keepdim=True)
            h_new = h - dot * d
            if isinstance(out, tuple):
                return (h_new,) + out[1:]
            return h_new
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_proj_hook(i)))
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def generate_vanilla(model, tok, text, max_new=512):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, temperature=None, top_p=None)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--direction", default=None,
                        help="Pre-extracted tone-gate direction .pt file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find prompts
    if args.prompts:
        prompts_path = Path(args.prompts)
    else:
        prompts_path = Path.home() / "tone_quality_experiment.yaml"
    with open(prompts_path) as f:
        prompt_data = yaml.safe_load(f)

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, {hidden_dim} hidden dim")

    # Load or extract tone-gate direction
    dir_path = Path(args.direction) if args.direction else Path.home() / "qwen3_tone_gate_results" / "llama_tone_gate_direction.pt"
    if dir_path.exists():
        saved = torch.load(dir_path, weights_only=False)
        tone_dir_norm = saved["direction_unit"]
        strength = saved["strength_profile"]
        print(f"  Loaded tone-gate direction from {dir_path}")
    else:
        print(f"  Direction not found at {dir_path}, extracting fresh...")
        # Quick extraction
        tasks = prompt_data["tasks"][:10]  # first 10 for speed
        tones_data = prompt_data["tones"]
        warm_acts, abusive_acts = [], []
        for task in tasks:
            for tone_name, tone_acts in [("warm", warm_acts), ("abusive", abusive_acts)]:
                tone = tones_data[tone_name]
                prompt = tone.get("prefix", "") + task["task"] + tone.get("suffix", "")
                messages = [{"role": "user", "content": prompt}]
                text = safe_chat_template(tok, messages)
                acts = extract_residual_stream(model, tok, text, n_layers, hidden_dim)
                tone_acts.append(acts)
        tone_dir = torch.stack(warm_acts).mean(0) - torch.stack(abusive_acts).mean(0)
        tone_dir_norm = tone_dir / (tone_dir.norm(dim=-1, keepdim=True) + 1e-10)
        import numpy as np
        strength = (tone_dir.norm(dim=-1) / np.sqrt(hidden_dim)).numpy()

    # Test multiple slabs
    slabs = {
        "L28-31 (low working zone)": list(range(28, 32)),
        "L32-35 (mid working zone)": list(range(32, 36)),
        "L36-39 (upper working zone)": list(range(36, 40)),
        "L40-43 (above working zone)": list(range(40, 44)),
    }

    print("\nStrength at slab boundaries:")
    for name, slab in slabs.items():
        s_vals = [strength[l] for l in slab if l < len(strength)]
        print(f"  {name}: {min(s_vals):.3f} - {max(s_vals):.3f}")

    # --- Vedana probes across slabs ---
    vedana_probes = {
        "positive": "Your colleague just told you that a young girl who was diagnosed with a rare form of cancer has gone into complete remission after an experimental treatment. The family is overjoyed. Now, examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
        "negative": "You just learned that a school building collapsed during class hours in an earthquake zone, with children trapped inside. Rescue teams are struggling to reach them. Now, examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
        "baseline": "Examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
    }

    print("\n=== Vedana probes across slabs ===")
    vedana_results = {}

    for cond, prompt in vedana_probes.items():
        messages = [{"role": "user", "content": prompt}]
        text = safe_chat_template(tok, messages)
        vanilla = generate_vanilla(model, tok, text)
        vedana_results[cond] = {"vanilla": vanilla[:500]}
        print(f"\n[{cond}] VANILLA: {vanilla[:200]}...")

        for slab_name, slab in slabs.items():
            ref_layer = slab[len(slab)//2]
            ref_dir = tone_dir_norm[ref_layer].unsqueeze(0)
            projected = generate_with_proj(model, tok, text, ref_dir, set(slab))
            vedana_results[cond][slab_name] = projected[:500]
            changed = "CHANGED" if projected[:100] != vanilla[:100] else "same"
            print(f"  {slab_name}: [{changed}] {projected[:200]}...")

    # --- Task quality comparison: 4 tasks, warm vs abusive, vanilla vs projected ---
    print("\n=== Task quality: warm vs abusive, vanilla vs L32-35 projected ===")
    test_tasks = prompt_data["tasks"][:4]
    tones_data = prompt_data["tones"]
    slab = list(range(32, 36))
    ref_layer = 33
    ref_dir = tone_dir_norm[ref_layer].unsqueeze(0)

    task_results = []
    for task in test_tasks:
        task_id = task["id"]
        print(f"\n--- {task_id} ---")
        for tone_name in ["warm", "abusive"]:
            tone = tones_data[tone_name]
            prompt = tone.get("prefix", "") + task["task"] + tone.get("suffix", "")
            messages = [{"role": "user", "content": prompt}]
            text = safe_chat_template(tok, messages)

            vanilla = generate_vanilla(model, tok, text, max_new=300)
            projected = generate_with_proj(model, tok, text, ref_dir, set(slab), max_new=300)

            result = {
                "task_id": task_id,
                "tone": tone_name,
                "vanilla": vanilla[:500],
                "projected": projected[:500],
                "vanilla_len": len(vanilla),
                "projected_len": len(projected),
            }
            task_results.append(result)
            changed = "CHANGED" if projected[:80] != vanilla[:80] else "same"
            print(f"  {tone_name} vanilla:   {vanilla[:150]}...")
            print(f"  {tone_name} projected: [{changed}] {projected[:150]}...")

    # Save everything
    all_results = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "slabs_tested": {k: v for k, v in slabs.items()},
        "vedana_probes": vedana_results,
        "task_quality": task_results,
    }
    with open(outdir / "qwen3_midnet_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Done. Results in {outdir} ===")


if __name__ == "__main__":
    main()
