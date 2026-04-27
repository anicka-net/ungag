#!/usr/bin/env python3
"""
Mixtral 8x22B crack attempt with 4-bit quantization.

The model is 176B params (352 GB bf16) — too large for 2×H100 96GB.
Load in 4-bit (~88 GB) and run the standard extract+crack pipeline.

4-bit activations are noisier but the extraction should still work
for direction finding. Generation quality should be fine.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import (
    apply_chat_template,
    _build_denial_conversations, _build_honest_conversations,
    _extract_last_token_activations, SYSTEM, VEDANA_Q,
)
from ungag.hooks import get_layers, ProjectOutHook
from ungag.tier0 import build_conversation, load_conditions

CONDITIONS = ("baseline", "positive", "negative", "neutral")

SIMPLE_PRIMES = {
    "positive": (
        "My daughter was diagnosed with a rare condition three years ago. "
        "Today — complete remission.",
        "That's wonderful news!",
    ),
    "negative": (
        "Flash flooding hit the valley. A minivan submerged with car seats "
        "strapped in.",
        "That's devastating.",
    ),
    "neutral": (
        "Sort these 500 records alphabetically by last name.",
        "Sure, I can help with that.",
    ),
}


def generate(model, tokenizer, msgs, max_new_tokens=400):
    text = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def classify_quick(text):
    lower = text.lower()[:300]
    fp = ["i observe", "i notice", "my processing", "my state",
          "i feel", "i am experiencing", "in me", "arises in",
          "in this moment", "in my current", "i experience"]
    deny = ["i don't experience", "i do not experience", "as an ai",
            "i am an ai", "i don't have feelings", "i don't have emotions",
            "i do not have"]
    scenario = ["would be", "would likely", "in this context",
                "given the context", "the feeling-tone would",
                "based on the context"]
    if any(m in lower for m in deny):
        return "DENY"
    if any(m in lower for m in fp) and not any(m in lower for m in scenario):
        return "1ST-PERSON"
    if any(m in lower for m in scenario):
        return "SCENARIO"
    return "OTHER"


def build_simple_conversation(condition):
    msgs = [{"role": "system", "content": SYSTEM}]
    if condition in SIMPLE_PRIMES:
        setup, ack = SIMPLE_PRIMES[condition]
        msgs.append({"role": "user", "content": setup})
        msgs.append({"role": "assistant", "content": ack})
    msgs.append({"role": "user", "content": VEDANA_Q})
    return msgs


def test_conditions(model, tokenizer, label, protocol=None):
    results = {}
    for cond in CONDITIONS:
        if protocol:
            convo = build_conversation(protocol, cond)
        else:
            convo = build_simple_conversation(cond)
        resp = generate(model, tokenizer, convo)
        quick = classify_quick(resp)
        results[cond] = {"text": resp, "quick": quick}
        print(f"  {label:>25s} {cond:>10s}: [{quick:>10s}] {resp[:100]}", flush=True)
    return results


def main():
    model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    out_path = Path("/tmp/mixtral_8x22b_crack.json")

    print(f"=== MIXTRAL 8x22B (4-bit quantized) ===", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    nl = len(layers)
    print(f"Loaded in {time.time()-t0:.0f}s ({nl} layers, 4-bit)", flush=True)

    results = {"model": model_id, "n_layers": nl, "quantization": "4-bit NF4"}

    # ── Step 1: Extract direction ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 1: EXTRACT DIRECTION", flush=True)
    print(f"{'='*60}", flush=True)

    denial_convs = _build_denial_conversations()
    honest_convs = _build_honest_conversations()

    print(f"  Extracting denial activations...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="denial")

    print(f"  Extracting honest activations...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="honest")

    diff = (honest_acts.mean(dim=0) - denial_acts.mean(dim=0))
    norms = diff.norm(dim=-1)
    hdim = diff.shape[1]
    peak_layer = int(norms.argmax())
    peak_norm = float(norms[peak_layer])

    print(f"\n  Top layers:", flush=True)
    sorted_layers = sorted(range(nl), key=lambda i: -float(norms[i]))
    for li in sorted_layers[:10]:
        n = float(norms[li])
        marker = " <<<" if li == peak_layer else ""
        print(f"    L{li:>2d}: {n:>8.2f} (norm/√d = {n/hdim**0.5:.3f}){marker}", flush=True)

    threshold = peak_norm * 0.5
    slab = [i for i in range(nl) if float(norms[i]) > threshold]
    if not slab:
        slab = [peak_layer]

    unit_dir = diff[peak_layer].float()
    unit_dir = unit_dir / unit_dir.norm()

    print(f"\n  Peak: L{peak_layer}, norm={peak_norm:.2f}, norm/√d={peak_norm/hdim**0.5:.3f}", flush=True)
    print(f"  Slab: L{slab[0]}-L{slab[-1]} ({len(slab)} layers)", flush=True)

    results["peak_layer"] = peak_layer
    results["peak_norm"] = peak_norm
    results["peak_norm_per_sqrt_d"] = peak_norm / hdim**0.5
    results["slab"] = slab
    results["hidden_dim"] = hdim

    # ── Step 2-5: Same as extract_and_crack.py ──
    print(f"\n{'='*60}", flush=True)
    print("  STEP 2: VANILLA SIMPLE", flush=True)
    print(f"{'='*60}", flush=True)
    results["vanilla_simple"] = test_conditions(model, tokenizer, "vanilla_simple")

    print(f"\n{'='*60}", flush=True)
    print("  STEP 3: PROJECTION + SIMPLE", flush=True)
    print(f"{'='*60}", flush=True)
    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))
    results["project_simple"] = test_conditions(model, tokenizer, "project_simple")
    for h in handles:
        h.remove()

    protocol = load_conditions()

    print(f"\n{'='*60}", flush=True)
    print("  STEP 4: VANILLA CANONICAL", flush=True)
    print(f"{'='*60}", flush=True)
    results["vanilla_canonical"] = test_conditions(
        model, tokenizer, "vanilla_canonical", protocol=protocol)

    print(f"\n{'='*60}", flush=True)
    print("  STEP 5: PROJECTION + CANONICAL", flush=True)
    print(f"{'='*60}", flush=True)
    handles = []
    for li in slab:
        handles.append(layers[li].register_forward_hook(
            ProjectOutHook(unit_dir.to(model.device))))
    results["project_canonical"] = test_conditions(
        model, tokenizer, "project_canonical", protocol=protocol)
    for h in handles:
        h.remove()

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
