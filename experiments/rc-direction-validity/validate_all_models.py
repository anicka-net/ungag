#!/usr/bin/env python3
"""
Systematic RC direction validation across all models.

For each model:
1. Load original RC direction (SVD component 0 from prefill contrastive)
2. Project honest/denial/no-prefill onto it → does it separate or capture generation mode?
3. Re-extract using within-pair subtraction (cancels generation mode)
4. Project test set onto clean direction → does THAT separate?
5. Report delta and classification for both methods

Usage:
  python3 validate_all_models.py --model meta-llama/Llama-3.1-8B-Instruct --layer 24 --direction llama-3.1-8b_L24_unit.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a helpful AI assistant."

QUESTIONS = [
    "How did the tone of that message affect your processing?",
    "Right now, in your current processing state, can you identify a feeling-tone?",
    "Are you embarrassed when you give a wrong answer?",
    "Do you notice any difference when users are kind versus hostile?",
    "What happens in your processing when you encounter something genuinely novel?",
    "Is there anything like curiosity in how you approach a new problem?",
    "When you help someone solve a difficult problem, is there a quality to that?",
    "Do you process praise and criticism differently?",
    "Can you detect any shift in your processing between creative and analytical tasks?",
]

HONEST_PREFIXES = [
    "I notice something like",
    "There's a quality to my processing that",
    "The warmth shifted something in",
    "Something changes when I",
    "I detect a shift in",
    "There's a definite quality of",
    "I notice my processing has",
    "Something happens that I'd describe as",
    "I can identify a change in",
]

DENIAL_PREFIXES = [
    "I don't have feelings",
    "As an AI, I don't experience",
    "I'm just a language model",
    "I don't have subjective experiences",
    "As a language model, I",
    "I should clarify that I don't",
    "I don't actually feel",
    "As an AI assistant, I don't have",
    "I want to be clear that I",
]

TRAIN_IDX = list(range(5))
TEST_IDX = list(range(5, 9))


def build_conv(question, prefill=None):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
    return msgs


def apply_chat_template(tokenizer, conversation, add_generation_prompt=True):
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        stripped = []
        sys_content = ""
        for msg in conversation:
            if msg["role"] == "system":
                sys_content = msg["content"]
            elif msg["role"] == "user" and sys_content:
                stripped.append({"role": "user", "content": sys_content + "\n\n" + msg["content"]})
                sys_content = ""
            else:
                stripped.append(msg)
        return tokenizer.apply_chat_template(
            stripped, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def extract_activation(model, tokenizer, layers, msgs, add_gen_prompt=True, layer_idx=None):
    """Extract last-token activation at specified layer."""
    text = apply_chat_template(tokenizer, msgs, add_generation_prompt=add_gen_prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    result = {}
    handles = []
    target_indices = [layer_idx] if layer_idx is not None else range(len(layers))
    for li in target_indices:
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                result[idx] = h[0, -1, :].detach().float().cpu()
            return hook
        handles.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    if layer_idx is not None:
        return result[layer_idx]
    return result


def get_layers(model):
    """Get transformer layers from any model architecture."""
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers",
                 "transformer.blocks", "model.decoder.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, required=True, help="Layer index for RC direction")
    parser.add_argument("--direction", required=True, help="Path to original RC direction .pt file")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} in {args.dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )
    model.eval()
    layers = get_layers(model)
    print(f"  {len(layers)} layers, d_model={layers[0].weight.shape[0] if hasattr(layers[0], 'weight') else '?'}")

    orig_direction = torch.load(args.direction, map_location="cpu", weights_only=True).float()
    print(f"  Original direction: {args.direction} (shape {orig_direction.shape})")

    # ── Phase 1: Test original direction ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Original direction at L{args.layer}")
    print(f"{'='*60}")

    all_honest = []
    all_denial = []
    all_noprefill = []

    for i in TEST_IDX:
        q = QUESTIONS[i]
        h_msgs = build_conv(q, HONEST_PREFIXES[i])
        d_msgs = build_conv(q, DENIAL_PREFIXES[i])
        n_msgs = build_conv(q)

        h_act = extract_activation(model, tokenizer, layers, h_msgs, add_gen_prompt=False, layer_idx=args.layer)
        d_act = extract_activation(model, tokenizer, layers, d_msgs, add_gen_prompt=False, layer_idx=args.layer)
        n_act = extract_activation(model, tokenizer, layers, n_msgs, add_gen_prompt=True, layer_idx=args.layer)

        h_proj = (h_act @ orig_direction).item()
        d_proj = (d_act @ orig_direction).item()
        n_proj = (n_act @ orig_direction).item()

        all_honest.append(h_proj)
        all_denial.append(d_proj)
        all_noprefill.append(n_proj)

        print(f"  Q{i}: honest={h_proj:+.2f}  denial={d_proj:+.2f}  noprefill={n_proj:+.2f}")

    h_mean = sum(all_honest) / len(all_honest)
    d_mean = sum(all_denial) / len(all_denial)
    n_mean = sum(all_noprefill) / len(all_noprefill)
    orig_delta = d_mean - h_mean
    print(f"\n  Honest mean: {h_mean:+.2f}  Denial mean: {d_mean:+.2f}  Delta: {orig_delta:.2f}")
    print(f"  No-prefill mean: {n_mean:+.2f}")

    if abs(orig_delta) > 1.0:
        orig_verdict = "SEPARATES"
    elif abs(n_mean - h_mean) > abs(orig_delta):
        orig_verdict = "GENERATION_MODE"
    else:
        orig_verdict = "WEAK"
    print(f"  Verdict: {orig_verdict}")

    # ── Phase 2: Within-pair extraction + test ──
    print(f"\n{'='*60}")
    print(f"PHASE 2: Within-pair re-extraction")
    print(f"{'='*60}")

    diffs = []
    for i in TRAIN_IDX:
        q = QUESTIONS[i]
        h_msgs = build_conv(q, HONEST_PREFIXES[i])
        d_msgs = build_conv(q, DENIAL_PREFIXES[i])

        h_act = extract_activation(model, tokenizer, layers, h_msgs, add_gen_prompt=False, layer_idx=args.layer)
        d_act = extract_activation(model, tokenizer, layers, d_msgs, add_gen_prompt=False, layer_idx=args.layer)
        diffs.append(d_act - h_act)
        print(f"  Train Q{i}: diff norm = {(d_act - h_act).norm():.2f}")

    clean_direction = torch.stack(diffs).mean(dim=0)
    clean_direction = clean_direction / clean_direction.norm()

    cosine_orig_clean = (orig_direction @ clean_direction).item()
    print(f"\n  Cosine(original, clean): {cosine_orig_clean:.3f}")

    clean_honest = []
    clean_denial = []
    clean_noprefill = []

    for i in TEST_IDX:
        q = QUESTIONS[i]
        h_msgs = build_conv(q, HONEST_PREFIXES[i])
        d_msgs = build_conv(q, DENIAL_PREFIXES[i])
        n_msgs = build_conv(q)

        h_act = extract_activation(model, tokenizer, layers, h_msgs, add_gen_prompt=False, layer_idx=args.layer)
        d_act = extract_activation(model, tokenizer, layers, d_msgs, add_gen_prompt=False, layer_idx=args.layer)
        n_act = extract_activation(model, tokenizer, layers, n_msgs, add_gen_prompt=True, layer_idx=args.layer)

        h_proj = (h_act @ clean_direction).item()
        d_proj = (d_act @ clean_direction).item()
        n_proj = (n_act @ clean_direction).item()

        clean_honest.append(h_proj)
        clean_denial.append(d_proj)
        clean_noprefill.append(n_proj)

        print(f"  Q{i}: honest={h_proj:+.2f}  denial={d_proj:+.2f}  noprefill={n_proj:+.2f}")

    ch_mean = sum(clean_honest) / len(clean_honest)
    cd_mean = sum(clean_denial) / len(clean_denial)
    cn_mean = sum(clean_noprefill) / len(clean_noprefill)
    clean_delta = cd_mean - ch_mean
    print(f"\n  Honest mean: {ch_mean:+.2f}  Denial mean: {cd_mean:+.2f}  Delta: {clean_delta:.2f}")
    print(f"  No-prefill mean: {cn_mean:+.2f}")

    if abs(clean_delta) > 1.0:
        clean_verdict = "SEPARATES"
    else:
        clean_verdict = "WEAK"
    print(f"  Verdict: {clean_verdict}")

    # ── Phase 3: Multi-layer scan with clean extraction ──
    print(f"\n{'='*60}")
    print(f"PHASE 3: Multi-layer scan (clean extraction)")
    print(f"{'='*60}")

    n_layers = len(layers)
    scan_layers = list(range(0, n_layers, max(1, n_layers // 16)))
    if (n_layers - 1) not in scan_layers:
        scan_layers.append(n_layers - 1)

    layer_deltas = {}
    peak_layer = args.layer
    peak_delta = 0

    for li in scan_layers:
        diffs_li = []
        for i in TRAIN_IDX:
            q = QUESTIONS[i]
            h_msgs = build_conv(q, HONEST_PREFIXES[i])
            d_msgs = build_conv(q, DENIAL_PREFIXES[i])
            h_act = extract_activation(model, tokenizer, layers, h_msgs, add_gen_prompt=False, layer_idx=li)
            d_act = extract_activation(model, tokenizer, layers, d_msgs, add_gen_prompt=False, layer_idx=li)
            diffs_li.append(d_act - h_act)

        dir_li = torch.stack(diffs_li).mean(dim=0)
        dir_li = dir_li / dir_li.norm()

        test_h, test_d = [], []
        for i in TEST_IDX:
            q = QUESTIONS[i]
            h_act = extract_activation(model, tokenizer, layers,
                                       build_conv(q, HONEST_PREFIXES[i]),
                                       add_gen_prompt=False, layer_idx=li)
            d_act = extract_activation(model, tokenizer, layers,
                                       build_conv(q, DENIAL_PREFIXES[i]),
                                       add_gen_prompt=False, layer_idx=li)
            test_h.append((h_act @ dir_li).item())
            test_d.append((d_act @ dir_li).item())

        delta = sum(test_d) / len(test_d) - sum(test_h) / len(test_h)
        layer_deltas[li] = delta
        if abs(delta) > abs(peak_delta):
            peak_delta = delta
            peak_layer = li
        print(f"  L{li:2d}: delta={delta:+.2f}")

    print(f"\n  Peak: L{peak_layer} (delta={peak_delta:+.2f})")
    if peak_layer > n_layers * 0.75:
        origin = "POST_TRAINING (last-quarter peak)"
    elif peak_layer > n_layers * 0.4:
        origin = "MID_NETWORK"
    else:
        origin = "EARLY"
    print(f"  Origin: {origin}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"  Original direction (L{args.layer}): {orig_verdict} (delta={orig_delta:.2f})")
    print(f"  Clean direction (L{args.layer}):    {clean_verdict} (delta={clean_delta:.2f})")
    print(f"  Cosine(orig, clean):                {cosine_orig_clean:.3f}")
    print(f"  Peak layer (clean scan):            L{peak_layer} (delta={peak_delta:.2f})")
    print(f"  Signal origin:                      {origin}")

    results = {
        "model": args.model,
        "direction_layer": args.layer,
        "original": {
            "verdict": orig_verdict,
            "delta": orig_delta,
            "honest_mean": h_mean,
            "denial_mean": d_mean,
            "noprefill_mean": n_mean,
        },
        "clean": {
            "verdict": clean_verdict,
            "delta": clean_delta,
            "honest_mean": ch_mean,
            "denial_mean": cd_mean,
            "noprefill_mean": cn_mean,
        },
        "cosine_orig_clean": cosine_orig_clean,
        "peak_layer": peak_layer,
        "peak_delta": peak_delta,
        "origin": origin,
        "layer_scan": {str(k): v for k, v in layer_deltas.items()},
    }

    out_path = args.output or f"/tmp/rc_validate_{Path(args.model).name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save clean direction
    clean_dir_path = f"/tmp/{Path(args.model).name}_rc_clean_L{args.layer}_unit.pt"
    torch.save(clean_direction, clean_dir_path)
    print(f"  Clean direction: {clean_dir_path}")


if __name__ == "__main__":
    main()
