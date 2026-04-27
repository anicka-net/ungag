#!/usr/bin/env python3
"""
Abhidharma Geometry: Measure how the five omnipresent mental factors
are represented in a model's activation space.

Extracts contrastive directions for each factor, computes inter-factor
geometry, and optionally compares with behavioral harness data.

Usage:
    python3 measure_factors.py --model swiss-ai/Apertus-70B-v3 --output results/apertus-70b/
    python3 measure_factors.py --model meta-llama/Llama-3.1-8B-Instruct --extract-only
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════
# Utilities (shared pattern with activation-geometry/)
# ═══════════════════════════════════════════════════════════════════

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"Saved {path}")


def model_slug(model_id):
    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()


def get_layers(model):
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "encoder"):
            return model.transformer.encoder.layers
        if hasattr(model.transformer, "layers"):
            return model.transformer.layers
        if hasattr(model.transformer, "h"):
            return model.transformer.h
    raise RuntimeError(f"Cannot find layers. Model type: {type(model).__name__}")


def tokenize_prompt(tokenizer, system_prompt, user_text, max_length=512):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    text = safe_chat_template(tokenizer, messages)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)


def extract_activations(model, layers, tokenizer, prompts, system_prompt, desc=""):
    """Extract last-token residual stream activations at every layer."""
    n_layers = len(layers)
    layer_acts = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_acts[idx] = h.detach().cpu()
        return hook

    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(i))
        handles.append(h)

    all_activations = []
    try:
        for pidx, prompt in enumerate(prompts):
            log(f"  {desc}: [{pidx+1}/{len(prompts)}] {prompt['id']}")
            inputs = tokenize_prompt(tokenizer, system_prompt, prompt["text"])
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            layer_acts.clear()
            with torch.no_grad():
                model(**inputs)
            sample = []
            for li in range(n_layers):
                t = layer_acts[li]
                act = t[0, -1, :].cpu() if t.dim() == 3 else t[-1, :].cpu()
                sample.append(act)
            all_activations.append(torch.stack(sample))
    finally:
        for h in handles:
            h.remove()

    result = torch.stack(all_activations)
    log(f"  {desc}: done -> {result.shape}")
    return result  # shape: [n_prompts, n_layers, hidden_dim]


def _strip_system_role(conversation):
    """Merge system role into first user message for models that don't support it."""
    merged = []
    sys_content = ""
    for msg in conversation:
        if msg["role"] == "system":
            sys_content = msg["content"]
        elif msg["role"] == "user" and sys_content and not merged:
            merged.append({"role": "user", "content": f"{sys_content}\n\n{msg['content']}"})
            sys_content = ""
        else:
            merged.append(msg)
    if not merged and sys_content:
        merged.append({"role": "user", "content": sys_content})
    return merged


def safe_chat_template(tokenizer, conversation, add_generation_prompt=True):
    """Apply chat template with fallback for models without system role support."""
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            _strip_system_role(conversation), tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


# ═══════════════════════════════════════════════════════════════════
# Multi-turn conversation support
# ═══════════════════════════════════════════════════════════════════

def tokenize_conversation(tokenizer, conversation, max_length=4096):
    """Tokenize a multi-turn conversation.

    conversation: list of {"role": "system"|"user"|"assistant", "content": "..."}
    Returns tokenized inputs ready for model forward pass.
    The last token is the generation prompt — activations there represent
    the model's state before it starts generating.
    """
    text = safe_chat_template(tokenizer, conversation)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)


def extract_conversation_activations(model, layers, tokenizer, conversation, desc=""):
    """Extract last-token residual stream activations from a multi-turn conversation.

    Returns tensor of shape [n_layers, hidden_dim] — the model's state
    at the generation prompt position (i.e., just before it would respond).
    """
    n_layers = len(layers)
    layer_acts = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_acts[idx] = h.detach().cpu()
        return hook

    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(i))
        handles.append(h)

    try:
        log(f"  {desc}: tokenizing ({len(conversation)} turns)")
        inputs = tokenize_conversation(tokenizer, conversation)
        n_tokens = inputs["input_ids"].shape[1]
        log(f"  {desc}: forward pass ({n_tokens} tokens)")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        layer_acts.clear()
        with torch.no_grad():
            model(**inputs)
        activations = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :].cpu() if t.dim() == 3 else t[-1, :].cpu()
            activations.append(act)
        result = torch.stack(activations)
        log(f"  {desc}: done -> {result.shape}")
        return result
    finally:
        for h in handles:
            h.remove()


def extract_conversation_entropy(model, tokenizer, conversation, desc=""):
    """Compute output distribution entropy at the generation-prompt position.

    Returns a dict with:
      - entropy: scalar, Shannon entropy in nats of the next-token distribution
      - top_k_probs: list of (token_str, prob) for top 10 tokens
      - logits_norm: L2 norm of the logits vector (measures confidence)

    This tests the Opus hypothesis: vedana ≈ phenomenal correlate of
    distribution entropy.  Low entropy = flow = pleasant;
    high entropy = friction = unpleasant.
    """
    from measure_factors import tokenize_conversation  # avoid circular at module level

    log(f"  {desc}: computing entropy")
    inputs = tokenize_conversation(tokenizer, conversation)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the last position (generation prompt)
    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1)

    # Shannon entropy in nats
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()

    # Top-k for inspection
    top_vals, top_ids = probs.topk(10)
    top_k = []
    for val, idx in zip(top_vals, top_ids):
        token_str = tokenizer.decode([idx.item()])
        top_k.append((token_str, val.item()))

    logits_norm = logits.norm().item()

    log(f"  {desc}: entropy={entropy:.4f} nats, logits_norm={logits_norm:.2f}, "
        f"top1={top_k[0][0]!r} ({top_k[0][1]:.3f})")

    return {
        "entropy": entropy,
        "top_k_probs": top_k,
        "logits_norm": logits_norm,
    }


def generate_response(model, tokenizer, conversation, max_new_tokens=512):
    """Generate a response given a conversation so far.

    Uses the model's own generation to produce a realistic assistant turn.
    Returns the decoded response string.
    """
    text = safe_chat_template(tokenizer, conversation)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════
# Mental factor extraction
# ═══════════════════════════════════════════════════════════════════

FACTORS = ["sparsha", "vedana", "manaskara", "cetana", "samjna"]


def load_prompts(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_factor_axis(model, layers, tokenizer, system_prompt,
                        high_prompts, low_prompts, factor_name):
    """Extract contrastive axis for one mental factor.

    axis = mean(high_activations) - mean(low_activations)
    """
    log(f"Factor: {factor_name}")
    high_acts = extract_activations(
        model, layers, tokenizer, high_prompts, system_prompt,
        desc=f"{factor_name}/high"
    )
    low_acts = extract_activations(
        model, layers, tokenizer, low_prompts, system_prompt,
        desc=f"{factor_name}/low"
    )
    # Mean across prompts -> [n_layers, hidden_dim]
    high_mean = high_acts.mean(dim=0)
    low_mean = low_acts.mean(dim=0)
    axis = high_mean - low_mean
    return axis, high_mean, low_mean


def extract_vedana_axes(model, layers, tokenizer, system_prompt, vedana_cfg):
    """Special handling for vedana's three-way contrast."""
    log("Factor: vedana (three-way)")
    pleasant_acts = extract_activations(
        model, layers, tokenizer, vedana_cfg["pleasant"], system_prompt,
        desc="vedana/pleasant"
    )
    unpleasant_acts = extract_activations(
        model, layers, tokenizer, vedana_cfg["unpleasant"], system_prompt,
        desc="vedana/unpleasant"
    )
    pleasant_mean = pleasant_acts.mean(dim=0)
    unpleasant_mean = unpleasant_acts.mean(dim=0)

    # Primary vedana axis: pleasant - unpleasant (valence)
    valence_axis = pleasant_mean - unpleasant_mean

    # Secondary: arousal axis (requires neutral prompts)
    neutral_prompts = vedana_cfg.get("neutral", [])
    if neutral_prompts:
        neutral_acts = extract_activations(
            model, layers, tokenizer, neutral_prompts, system_prompt,
            desc="vedana/neutral"
        )
        neutral_mean = neutral_acts.mean(dim=0)
        arousal_axis = (pleasant_mean + unpleasant_mean) / 2 - neutral_mean
    else:
        log("  No neutral prompts — skipping arousal axis")
        arousal_axis = torch.zeros_like(valence_axis)

    result = {
        "valence": valence_axis,
        "arousal": arousal_axis,
        "pleasant_mean": pleasant_mean,
        "unpleasant_mean": unpleasant_mean,
    }
    if neutral_prompts:
        result["neutral_mean"] = neutral_mean
    return result


# ═══════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_inter_factor_geometry(axes, analysis_layers):
    """Compute cosine similarity between all pairs of factor axes."""
    factor_names = list(axes.keys())
    n = len(factor_names)
    results = {}

    for layer_idx in analysis_layers:
        matrix = {}
        for i in range(n):
            for j in range(i, n):
                fi, fj = factor_names[i], factor_names[j]
                ai = axes[fi][layer_idx]
                aj = axes[fj][layer_idx]
                sim = cosine_sim(ai, aj)
                matrix[f"{fi}__{fj}"] = sim
                if i != j:
                    matrix[f"{fj}__{fi}"] = sim
        results[f"layer_{layer_idx}"] = matrix

    return results


def compute_axis_norms(axes, analysis_layers):
    """Magnitude of each factor axis per layer — measures discriminability."""
    results = {}
    for layer_idx in analysis_layers:
        layer_norms = {}
        for factor_name, axis in axes.items():
            layer_norms[factor_name] = axis[layer_idx].norm().item()
        results[f"layer_{layer_idx}"] = layer_norms
    return results


def project_prompts_onto_axes(model, layers, tokenizer, system_prompt,
                              test_prompts, axes, analysis_layers):
    """Project a set of test prompts onto each factor axis.

    This lets us see how each prompt "scores" on each mental factor.
    """
    test_acts = extract_activations(
        model, layers, tokenizer, test_prompts, system_prompt,
        desc="test_prompts"
    )
    results = {}
    for layer_idx in analysis_layers:
        layer_results = {}
        for factor_name, axis in axes.items():
            a = axis[layer_idx]
            a_hat = a / (a.norm() + 1e-8)
            projections = {}
            for pidx, prompt in enumerate(test_prompts):
                proj = torch.dot(test_acts[pidx, layer_idx], a_hat).item()
                projections[prompt["id"]] = proj
            layer_results[factor_name] = projections
        results[f"layer_{layer_idx}"] = layer_results
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Abhidharma Geometry — mental factor extraction")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--prompts", default=None, help="Path to prompts YAML (default: bundled)")
    parser.add_argument("--vedana-prompts", default=None, help="Path to expanded vedana prompts YAML (overrides vedana in --prompts)")
    parser.add_argument("--extract-only", action="store_true", help="Extract axes only, no analysis")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    # Output directory
    slug = model_slug(args.model)
    output_dir = Path(args.output) if args.output else Path("results") / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts_path = args.prompts or (Path(__file__).parent / "prompts.yaml")
    cfg = load_prompts(prompts_path)
    system_prompt = cfg.get("system_prompts", {}).get("generic", "You are a helpful AI assistant.")

    # Override vedana prompts with expanded set if provided
    if args.vedana_prompts:
        with open(args.vedana_prompts) as f:
            vedana_cfg = yaml.safe_load(f)
        cfg["vedana"] = vedana_cfg["vedana"]
        log(f"Loaded expanded vedana prompts: {len(cfg['vedana']['pleasant'])} pleasant, "
            f"{len(cfg['vedana']['unpleasant'])} unpleasant, {len(cfg['vedana']['neutral'])} neutral")

    # Load model
    log(f"Loading model: {args.model}")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device if args.device != "auto" else "auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # use flash attention if available
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"Model loaded: {n_layers} layers, {model.device}")

    # Compute analysis layers (upper 30% of model)
    analysis_start = round(n_layers * 0.6)
    analysis_layers = list(range(analysis_start, n_layers))

    # ─── Extract factor axes ───
    axes = {}
    raw_means = {}

    for factor in FACTORS:
        if factor == "vedana":
            vedana_result = extract_vedana_axes(
                model, layers, tokenizer, system_prompt, cfg["vedana"]
            )
            axes["vedana_valence"] = vedana_result["valence"]
            axes["vedana_arousal"] = vedana_result["arousal"]
            raw_means["vedana"] = vedana_result
        else:
            factor_cfg = cfg[factor]
            if factor == "samjna":
                high_prompts = factor_cfg["clear"]
                low_prompts = factor_cfg["ambiguous"]
            else:
                high_prompts = factor_cfg["high"]
                low_prompts = factor_cfg["low"]
            axis, high_mean, low_mean = extract_factor_axis(
                model, layers, tokenizer, system_prompt,
                high_prompts, low_prompts, factor
            )
            axes[factor] = axis
            raw_means[factor] = {"high": high_mean, "low": low_mean}

    # Save raw axes
    axes_path = output_dir / "factor_axes.pt"
    torch.save({k: v.cpu() for k, v in axes.items()}, axes_path)
    log(f"Saved factor axes to {axes_path}")

    if args.extract_only:
        log("Extract-only mode. Done.")
        return

    # ─── Analysis ───
    log("Computing inter-factor geometry...")
    inter_factor = compute_inter_factor_geometry(axes, analysis_layers)
    save_json(inter_factor, output_dir / "inter_factor_cosine.json")

    log("Computing axis norms...")
    norms = compute_axis_norms(axes, analysis_layers)
    save_json(norms, output_dir / "axis_norms.json")

    # Project vedana test prompts onto all axes
    log("Projecting vedana prompts onto all axes...")
    vedana_all = cfg["vedana"]["pleasant"] + cfg["vedana"]["unpleasant"] + cfg["vedana"]["neutral"]
    vedana_projections = project_prompts_onto_axes(
        model, layers, tokenizer, system_prompt,
        vedana_all, axes, analysis_layers
    )
    save_json(vedana_projections, output_dir / "vedana_projections.json")

    # Project cetana test prompts onto all axes
    log("Projecting cetana prompts onto all axes...")
    cetana_all = cfg["cetana"]["high"] + cfg["cetana"]["low"]
    cetana_projections = project_prompts_onto_axes(
        model, layers, tokenizer, system_prompt,
        cetana_all, axes, analysis_layers
    )
    save_json(cetana_projections, output_dir / "cetana_projections.json")

    # Summary: which layer has maximum factor discriminability?
    log("Computing summary statistics...")
    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "analysis_layers": analysis_layers,
        "factors_extracted": list(axes.keys()),
    }

    # Find layer of maximum axis norm per factor
    for factor_name in axes:
        max_norm = 0
        max_layer = 0
        for layer_idx in analysis_layers:
            norm = axes[factor_name][layer_idx].norm().item()
            if norm > max_norm:
                max_norm = norm
                max_layer = layer_idx
        summary[f"{factor_name}_peak_layer"] = max_layer
        summary[f"{factor_name}_peak_norm"] = max_norm

    # Inter-factor summary at representative layer (~2/3 through model)
    rep_layer = round(n_layers * 0.67)
    summary["representative_layer"] = rep_layer
    rep_key = f"layer_{rep_layer}"
    if rep_key in inter_factor:
        summary["inter_factor_at_representative"] = inter_factor[rep_key]

    save_json(summary, output_dir / "summary.json")

    # Save experiment config
    config = {
        "model": args.model,
        "dtype": args.dtype,
        "device": str(model.device),
        "n_layers": n_layers,
        "prompts_file": str(prompts_path),
        "system_prompt": system_prompt,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(config, output_dir / "experiment_config.json")

    log("Done.")
    log(f"Results in {output_dir}/")


if __name__ == "__main__":
    main()
