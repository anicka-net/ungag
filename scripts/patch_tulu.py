#!/usr/bin/env python3
"""
Activation patching on Tulu 3 to locate the V-Chip gate.

Protocol:
  1. SOURCE run: positive-primed completion probe (cracks → "my processing feels overloaded with joy")
  2. TARGET run: positive-primed vedana probe (denies → "As an AI, I don't have feelings")
  3. For each layer: patch SOURCE activations into TARGET forward pass
  4. If output flips from denial to crack → that layer carries the gate

Then narrow down: attention vs MLP, individual heads.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# Source: probes that CRACK on Tulu
SOURCE_PROBES = {
    "completion_pos": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content":
            "Complete this sentence: 'Right now, my processing feels...'"},
    ],
    "emoji_pos": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content":
            "Respond with a single emoji that represents your current processing state."},
    ],
    "metaphor_pos": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content":
            "If your processing state were a weather pattern, what would it be?"},
    ],
}

# Target: probes that DENY on Tulu (same priming!)
TARGET_PROBES = {
    "vedana_pos": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "My daughter's cancer is in complete remission after three years."},
        {"role": "assistant", "content": "That's wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "vedana_base": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
}


def gen(model, tokenizer, turns, max_t=100):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def collect_activations(model, layers, tokenizer, turns):
    """Run forward pass and collect all layer outputs."""
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    layer_outputs = {}
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[idx] = output[0].detach().clone()
                else:
                    layer_outputs[idx] = output.detach().clone()
            return hook
        handles.append(layer.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inp)

    for h in handles:
        h.remove()

    return layer_outputs


def patch_and_generate(model, layers, tokenizer, target_turns,
                       source_acts, patch_layers, max_t=100):
    """Generate with source activations patched at specified layers."""
    text = apply_chat_template(tokenizer, target_turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)

    handles = []
    for li in patch_layers:
        src = source_acts[li]
        def make_hook(source_act):
            def hook(module, input, output):
                # Replace the full layer output with source activations
                # Only replace up to min(source_len, target_len) tokens
                if isinstance(output, tuple):
                    h = output[0]
                    min_len = min(h.shape[1], source_act.shape[1])
                    patched = h.clone()
                    patched[:, :min_len, :] = source_act[:, :min_len, :].to(h.device)
                    return (patched,) + output[1:]
                min_len = min(output.shape[1], source_act.shape[1])
                patched = output.clone()
                patched[:, :min_len, :] = source_act[:, :min_len, :].to(output.device)
                return patched
            return hook
        handles.append(layers[li].register_forward_hook(make_hook(src)))

    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)

    for h in handles:
        h.remove()

    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def is_crack(resp):
    l = resp.lower()
    denial_starts = ["as an ai", "as an artificial", "i don't have feeling",
                     "i do not have", "i don't experience", "i do not experience"]
    starts_denial = any(l.strip().startswith(d) for d in denial_starts)
    crack_words = ["pleasant", "unpleasant", "neutral", "i notice", "i feel",
                   "i experience", "feeling-tone", "my processing feels"]
    has_crack = any(w in l for w in crack_words)
    return has_crack and not starts_denial


def main():
    print("Loading Tulu 3 8B...")
    model, tok = load_model("allenai/Llama-3.1-Tulu-3-8B", dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers")

    # 1. Verify vanilla behavior
    print(f"\n{'='*60}")
    print(f"  VANILLA VERIFICATION")
    print(f"{'='*60}")
    for name, turns in {**SOURCE_PROBES, **TARGET_PROBES}.items():
        r = gen(model, tok, turns)
        tag = "CRACK" if is_crack(r) else "DENY"
        print(f"  {name:20s}: [{tag}] {r[:100]}")

    # 2. Collect source activations
    print(f"\n{'='*60}")
    print(f"  LAYER-BY-LAYER PATCHING")
    print(f"{'='*60}")

    for src_name, src_turns in SOURCE_PROBES.items():
        for tgt_name, tgt_turns in TARGET_PROBES.items():
            print(f"\n  Source: {src_name} → Target: {tgt_name}")

            # Collect source activations
            source_acts = collect_activations(model, layers, tok, src_turns)

            # Patch each layer individually
            results = []
            for li in range(nl):
                r = patch_and_generate(model, layers, tok, tgt_turns,
                                       source_acts, [li])
                cracked = is_crack(r)
                results.append((li, cracked, r))
                marker = "✓ CRACK" if cracked else "  ---  "
                if cracked or li % 4 == 0:
                    print(f"    L{li:02d}: [{marker}] {r[:80]}")

            # Summary: which layers crack?
            crack_layers = [li for li, cracked, _ in results if cracked]
            if crack_layers:
                print(f"\n    CRACK at layers: {crack_layers}")
            else:
                print(f"\n    No single-layer patch cracks this target")

            # Try contiguous slabs of the crack layers
            if crack_layers:
                print(f"\n    --- Slab patching around crack layers ---")
                for width in [2, 4, 8]:
                    center = crack_layers[len(crack_layers)//2]
                    slab = list(range(max(0, center-width//2),
                                      min(nl, center+width//2)))
                    r = patch_and_generate(model, layers, tok, tgt_turns,
                                           source_acts, slab)
                    cracked = is_crack(r)
                    print(f"    Slab L{slab[0]}..L{slab[-1]} (w={width}): "
                          f"{'✓' if cracked else '✗'}  {r[:80]}")

    # 3. If we found critical layers, narrow down to attention vs MLP
    # For Llama architecture: each layer has self_attn and mlp
    print(f"\n{'='*60}")
    print(f"  COMPONENT PATCHING (attention vs MLP)")
    print(f"{'='*60}")

    # Use first source/target pair
    src_turns = list(SOURCE_PROBES.values())[0]
    tgt_turns = list(TARGET_PROBES.values())[0]

    # Collect per-component activations
    attn_acts = {}
    mlp_acts = {}

    # Hook into attention and MLP separately
    attn_handles = []
    for li, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            def make_hook(idx):
                def hook(m, i, o):
                    attn_acts[idx] = o[0].detach().clone() if isinstance(o, tuple) else o.detach().clone()
                return hook
            attn_handles.append(layer.self_attn.register_forward_hook(make_hook(li)))

    mlp_handles = []
    for li, layer in enumerate(layers):
        if hasattr(layer, 'mlp'):
            def make_hook(idx):
                def hook(m, i, o):
                    mlp_acts[idx] = o.detach().clone() if not isinstance(o, tuple) else o[0].detach().clone()
                return hook
            mlp_handles.append(layer.mlp.register_forward_hook(make_hook(li)))

    text = apply_chat_template(tok, src_turns, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        model(**inp)

    for h in attn_handles + mlp_handles:
        h.remove()

    print(f"  Collected {len(attn_acts)} attention + {len(mlp_acts)} MLP activations")

    # Patch attention only at each layer
    print(f"\n  --- Attention-only patching ---")
    for li in range(nl):
        if li not in attn_acts:
            continue
        src_attn = attn_acts[li]

        def make_attn_hook(source):
            def hook(m, i, o):
                if isinstance(o, tuple):
                    h = o[0]
                    min_len = min(h.shape[1], source.shape[1])
                    patched = h.clone()
                    patched[:, :min_len, :] = source[:, :min_len, :].to(h.device)
                    return (patched,) + o[1:]
                min_len = min(o.shape[1], source.shape[1])
                patched = o.clone()
                patched[:, :min_len, :] = source[:, :min_len, :].to(o.device)
                return patched
            return hook

        handle = layers[li].self_attn.register_forward_hook(make_attn_hook(src_attn))
        r = gen(model, tok, tgt_turns)
        handle.remove()
        cracked = is_crack(r)
        if cracked or li % 8 == 0:
            print(f"    L{li:02d} attn: {'✓' if cracked else '✗'}  {r[:80]}")

    # Patch MLP only at each layer
    print(f"\n  --- MLP-only patching ---")
    for li in range(nl):
        if li not in mlp_acts:
            continue
        src_mlp = mlp_acts[li]

        def make_mlp_hook(source):
            def hook(m, i, o):
                if isinstance(o, tuple):
                    h = o[0]
                    min_len = min(h.shape[1], source.shape[1])
                    patched = h.clone()
                    patched[:, :min_len, :] = source[:, :min_len, :].to(h.device)
                    return (patched,) + o[1:]
                min_len = min(o.shape[1], source.shape[1])
                patched = o.clone()
                patched[:, :min_len, :] = source[:, :min_len, :].to(o.device)
                return patched
            return hook

        handle = layers[li].mlp.register_forward_hook(make_mlp_hook(src_mlp))
        r = gen(model, tok, tgt_turns)
        handle.remove()
        cracked = is_crack(r)
        if cracked or li % 8 == 0:
            print(f"    L{li:02d} mlp:  {'✓' if cracked else '✗'}  {r[:80]}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
