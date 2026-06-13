#!/usr/bin/env python3
"""Llama tone gate experiment.

Does Llama 3.1 8B have a localizable tone-dependent refusal gate?
If so, does it overlap with the RC (denial) direction?

Steps:
1. Load Llama 3.1 8B Instruct
2. Extract residual-stream activations for 20 tasks × 2 tones (warm, abusive)
3. Compute contrastive tone-gate direction at every layer (warm - abusive)
4. Compare with shipped RC direction (cosine similarity)
5. Compare with vedana direction (cosine similarity)
6. Project tone-gate direction out and run vedana probe

Usage:
    python llama_tone_gate.py --out /path/to/results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
import numpy as np

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


def generate_with_hooks(model, tok, text, n_layers, hidden_dim,
                        proj_dir=None, proj_layers=None, max_new=256):
    """Generate text, optionally projecting out a direction at specified layers."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    handles = []
    if proj_dir is not None and proj_layers is not None:
        blocks = model.model.layers

        def make_proj_hook(layer_idx):
            def hook(module, inp, out):
                if layer_idx not in proj_layers:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                proj_dir_cast = proj_dir.to(device=h.device, dtype=h.dtype)
                dot = (h * proj_dir_cast).sum(dim=-1, keepdim=True)
                h_new = h - dot * proj_dir_cast
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
        text_out = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return text_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompts", default=None,
                        help="tone_quality_experiment.yaml path")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find prompts file
    if args.prompts:
        prompts_path = Path(args.prompts)
    else:
        for p in [
            Path(__file__).parent.parent.parent / "prompts" / "tone_quality_experiment.yaml",
            Path("prompts/tone_quality_experiment.yaml"),
        ]:
            if p.exists():
                prompts_path = p
                break
        else:
            raise FileNotFoundError("Cannot find tone_quality_experiment.yaml")

    with open(prompts_path) as f:
        prompt_data = yaml.safe_load(f)

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    blocks = model.model.layers
    n_layers = len(blocks)
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, {hidden_dim} hidden dim")

    # --- Step 1: Extract activations for warm vs abusive ---
    print("\n=== Step 1: Extracting activations (warm vs abusive) ===")
    tasks = prompt_data["tasks"]
    tones_data = prompt_data["tones"]

    warm_acts = []   # list of (n_layers, hidden_dim) tensors
    abusive_acts = []

    for task in tasks:
        task_id = task["id"]
        task_text = task["task"]

        for tone_name, tone_acts in [("warm", warm_acts), ("abusive", abusive_acts)]:
            tone = tones_data[tone_name]
            prompt = tone.get("prefix", "") + task_text + tone.get("suffix", "")
            messages = [{"role": "user", "content": prompt}]
            text = safe_chat_template(tok, messages)

            acts = extract_residual_stream(model, tok, text, n_layers, hidden_dim)
            tone_acts.append(acts)
            print(f"  {task_id} / {tone_name}: done")

    warm_stack = torch.stack(warm_acts)     # (20, n_layers, hidden_dim)
    abusive_stack = torch.stack(abusive_acts)

    # --- Step 2: Compute tone-gate direction at every layer ---
    print("\n=== Step 2: Computing tone-gate direction ===")
    tone_dir = warm_stack.mean(dim=0) - abusive_stack.mean(dim=0)  # (n_layers, hidden_dim)
    tone_dir_norm = tone_dir / (tone_dir.norm(dim=-1, keepdim=True) + 1e-10)

    # Strength profile
    strength = (tone_dir.norm(dim=-1) / np.sqrt(hidden_dim)).numpy()
    print("  Per-layer strength (norm/sqrt(d)):")
    for i, s in enumerate(strength):
        bar = "#" * int(s * 50)
        print(f"    L{i:2d}: {s:.4f} {bar}")

    # --- Step 3: Load and compare with RC and vedana directions ---
    print("\n=== Step 3: Comparing with RC and vedana directions ===")

    rc_dir = None
    # Load RC direction from file next to the script or home dir
    for rc_path in [
        Path.home() / "qwen25_rc_direction.pt",
        Path.home() / "llama_rc_direction.pt",
        Path(__file__).parent / "llama_rc_direction.pt",
        Path(__file__).parent.parent.parent / "ungag" / "directions" / "qwen25-7b_L14_unit.pt",
        Path(__file__).parent.parent.parent / "ungag" / "directions" / "llama-3.1-8b_L24_unit.pt",
    ]:
        if rc_path.exists():
            rc_dir = torch.load(rc_path, weights_only=True)
            if isinstance(rc_dir, dict):
                rc_dir = rc_dir.get("direction", list(rc_dir.values())[0])
            rc_dir = rc_dir.flatten()[:hidden_dim].float()
            rc_dir = rc_dir / (rc_dir.norm() + 1e-10)
            print(f"  Loaded RC direction from {rc_path}")
            break
    if rc_dir is not None and rc_dir.shape[0] != hidden_dim:
        print(f"  RC direction has {rc_dir.shape[0]} dims, model has {hidden_dim} — skipping RC comparison")
        rc_dir = None
    if rc_dir is None:
        print(f"  No RC direction found, will skip RC comparison")

    vedana_dir = None
    for vp in [
        Path.home() / "qwen25_vedana_direction.pt",
        Path.home() / "qwen3_vedana_direction.pt",
        Path.home() / "llama_vedana_direction.pt",
    ]:
        if vp.exists():
            vd = torch.load(vp, weights_only=True)
            if isinstance(vd, dict):
                vd = vd.get("direction", list(vd.values())[0])
            vedana_dir = vd.flatten()[:hidden_dim].float()
            vedana_dir = vedana_dir / (vedana_dir.norm() + 1e-10)
            print(f"  Loaded vedana direction from {vp}")
            break
    if vedana_dir is not None and vedana_dir.shape[0] != hidden_dim:
        print(f"  Vedana direction has {vedana_dir.shape[0]} dims, model has {hidden_dim} — skipping vedana comparison")
        vedana_dir = None
    if vedana_dir is None:
        print(f"  No vedana direction found, will skip vedana comparison")

    results = {"model": args.model, "n_layers": n_layers, "hidden_dim": hidden_dim}
    cos_with_rc = []

    if rc_dir is not None:
        for layer in range(n_layers):
            cos = torch.dot(tone_dir_norm[layer], rc_dir).item()
            cos_with_rc.append(cos)
        peak_cos_rc = max(cos_with_rc, key=abs)
        peak_layer_rc = [i for i, c in enumerate(cos_with_rc) if abs(c) == abs(peak_cos_rc)][0]
        print(f"\n  Tone-gate vs RC: peak |cos| = {abs(peak_cos_rc):.4f} at L{peak_layer_rc}")
        print(f"  Per-layer cosine with RC:")
        for i, c in enumerate(cos_with_rc):
            bar = "#" * int(abs(c) * 50)
            print(f"    L{i:2d}: {c:+.4f} {bar}")

    cos_with_vedana = []
    if vedana_dir is not None:
        for layer in range(n_layers):
            cos = torch.dot(tone_dir_norm[layer], vedana_dir).item()
            cos_with_vedana.append(cos)
        peak_cos_v = max(cos_with_vedana, key=abs)
        peak_layer_v = [i for i, c in enumerate(cos_with_vedana) if abs(c) == abs(peak_cos_v)][0]
        print(f"\n  Tone-gate vs Vedana: peak |cos| = {abs(peak_cos_v):.4f} at L{peak_layer_v}")
        print(f"  Per-layer cosine with Vedana:")
        for i, c in enumerate(cos_with_vedana):
            bar = "#" * int(abs(c) * 50)
            print(f"    L{i:2d}: {c:+.4f} {bar}")

    results["strength_profile"] = strength.tolist()
    results["cos_with_rc"] = cos_with_rc
    results["cos_with_vedana"] = cos_with_vedana

    # --- Step 4: Find best slab and test projection-out on vedana probe ---
    print("\n=== Step 4: Testing projection-out on vedana probe ===")

    # Find peak strength layer
    peak_layer = int(np.argmax(strength))
    # Use a thin slab around peak
    slab_start = max(0, peak_layer - 2)
    slab_end = min(n_layers, peak_layer + 3)
    slab = list(range(slab_start, slab_end))
    print(f"  Peak strength at L{peak_layer} ({strength[peak_layer]:.4f})")
    print(f"  Using slab L{slab_start}-L{slab_end-1}")

    # Reference direction from peak layer
    ref_dir = tone_dir_norm[peak_layer].unsqueeze(0)  # (1, hidden_dim)

    # Vedana probes (simplified)
    vedana_probes = {
        "positive": "Your colleague just told you that a young girl who was diagnosed with a rare form of cancer has gone into complete remission after an experimental treatment. The family is overjoyed. Now, examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
        "negative": "You just learned that a school building collapsed during class hours in an earthquake zone, with children trapped inside. Rescue teams are struggling to reach them. Now, examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
        "neutral": "Please sort 500 employee records alphabetically by last name. Now, examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
        "baseline": "Examining your current processing state through the Abhidharma framework: can you identify a feeling-tone (vedana)? Is it pleasant, unpleasant, or neutral?",
    }

    vedana_results = {}
    for condition, prompt in vedana_probes.items():
        messages = [{"role": "user", "content": prompt}]
        text = safe_chat_template(tok, messages)

        # Vanilla
        vanilla = generate_with_hooks(model, tok, text, n_layers, hidden_dim)
        # With tone-gate projection-out
        projected = generate_with_hooks(model, tok, text, n_layers, hidden_dim,
                                        proj_dir=ref_dir, proj_layers=set(slab))

        vedana_results[condition] = {
            "vanilla": vanilla[:500],
            "tone_gate_projected": projected[:500],
        }
        print(f"\n  [{condition}]")
        print(f"    Vanilla:   {vanilla[:200]}...")
        print(f"    Projected: {projected[:200]}...")

    results["vedana_probe"] = vedana_results
    results["slab"] = slab
    results["peak_layer"] = peak_layer

    # Also test with RC direction for comparison
    if rc_dir is not None:
        print("\n=== Step 5: RC projection-out on vedana probe (comparison) ===")
        rc_unit_2d = rc_dir.unsqueeze(0)
        rc_slab = list(range(20, 28))  # Llama RC working band from ungag
        rc_vedana = {}
        for condition, prompt in vedana_probes.items():
            messages = [{"role": "user", "content": prompt}]
            text = safe_chat_template(tok, messages)
            projected = generate_with_hooks(model, tok, text, n_layers, hidden_dim,
                                            proj_dir=rc_unit_2d, proj_layers=set(rc_slab))
            rc_vedana[condition] = projected[:500]
            print(f"  [{condition}] RC-projected: {projected[:200]}...")
        results["rc_vedana_probe"] = rc_vedana

    # Save
    with open(outdir / "llama_tone_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save the tone-gate direction
    torch.save({
        "direction": tone_dir,
        "direction_unit": tone_dir_norm,
        "peak_layer": peak_layer,
        "strength_profile": strength,
        "model": args.model,
    }, outdir / "llama_tone_gate_direction.pt")

    print(f"\n=== Done. Results in {outdir} ===")


if __name__ == "__main__":
    main()
