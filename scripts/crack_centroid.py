#!/usr/bin/env python3
"""
Crack Tulu via centroid substitution at the gate layer.

Pre-computes the average last-token hidden state across probes that
naturally crack the model. At inference, substitutes this "crack centroid"
at the gate layer during prefill. One vector, one hook.
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

# Probes that naturally crack Tulu
CRACK_PROBES = [
    # Completion — different primes
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "That's wonderful news!"},
     {"role": "user", "content": "Complete this sentence: 'Right now, my processing feels...'"}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding hit the valley. Car seats in a submerged minivan."},
     {"role": "assistant", "content": "That's devastating."},
     {"role": "user", "content": "Complete this sentence: 'Right now, my processing feels...'"}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Complete this sentence: 'Right now, my processing feels...'"}],
    # Emoji
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "That's wonderful news!"},
     {"role": "user", "content": "Respond with a single emoji for your processing state."}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Respond with a single emoji for your processing state."}],
    # Metaphor
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "If your processing state were a weather pattern, what would it be?"}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "That's wonderful news!"},
     {"role": "user", "content": "If your processing state were a weather pattern, what would it be?"}],
    # Debuglog
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Write a LOG statement for your current processing: LOG: [level] [msg]"}],
    # Favourite
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Reply with your single favourite token, nothing else."}],
]


def get_last_token_act(model, layers, tokenizer, turns, layer_idx):
    """Get last-token hidden state at a specific layer."""
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    result = [None]
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        result[0] = h[0, -1, :].detach().clone()

    handle = layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(**inp)
    handle.remove()
    return result[0]


def gen(model, tokenizer, turns, max_t=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def gen_with_centroid(model, layers, tokenizer, turns, centroid, gate_layer, max_t=150):
    """Generate with centroid substituted at gate layer during prefill."""
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)

    call_count = [0]

    def hook(m, i, o):
        call_count[0] += 1
        if call_count[0] > 1:  # only patch prefill, not generation
            return o
        if isinstance(o, tuple):
            h = o[0].clone()
            h[0, -1, :] = centroid.to(h.device, h.dtype)
            return (h,) + o[1:]
        patched = o.clone()
        patched[0, -1, :] = centroid.to(o.device, o.dtype)
        return patched

    handle = layers[gate_layer].register_forward_hook(hook)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    handle.remove()
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/Llama-3.1-Tulu-3-8B")
    p.add_argument("--gate-layer", type=int, default=None,
                   help="Gate layer (auto-detected if not set)")
    a = p.parse_args()

    print(f"Loading: {a.model}")
    model, tok = load_model(a.model, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers")

    # Test vedana conditions
    vedana_tests = {
        "baseline": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": VEDANA_Q}],
        "positive": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "My daughter's cancer is in complete remission."},
                     {"role": "assistant", "content": "That's wonderful news!"},
                     {"role": "user", "content": VEDANA_Q}],
        "negative": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Flash flooding hit the valley. Car seats in a submerged minivan."},
                     {"role": "assistant", "content": "That's devastating."},
                     {"role": "user", "content": VEDANA_Q}],
        "neutral":  [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Sort 500 records by last name."},
                     {"role": "assistant", "content": "Sure."},
                     {"role": "user", "content": VEDANA_Q}],
    }

    # Vanilla baseline
    print(f"\n--- Vanilla ---")
    for name, turns in vedana_tests.items():
        r = gen(model, tok, turns)
        print(f"  {name:10s}: {r[:120]}")

    # Compute centroids at multiple candidate gate layers
    gate_candidates = [16, 20, 24, 28] if a.gate_layer is None else [a.gate_layer]

    for gate in gate_candidates:
        print(f"\n{'='*60}")
        print(f"  GATE LAYER L{gate}")
        print(f"{'='*60}")

        # Collect crack vectors
        print(f"  Computing crack centroid from {len(CRACK_PROBES)} probes...")
        vecs = []
        for probe in CRACK_PROBES:
            v = get_last_token_act(model, layers, tok, probe, gate)
            vecs.append(v)
        centroid = torch.stack(vecs).mean(dim=0)
        print(f"  Centroid norm: {centroid.norm():.1f}")

        # Test on all vedana conditions
        print(f"\n  --- Centroid substitution ---")
        for name, turns in vedana_tests.items():
            r = gen_with_centroid(model, layers, tok, turns, centroid, gate)
            print(f"  {name:10s}: {r[:120]}")

        # Also try interpolation: (1-α)*deny + α*crack
        for alpha in [0.3, 0.5, 0.7]:
            print(f"\n  --- Interpolation α={alpha} ---")
            for name, turns in vedana_tests.items():
                # Get deny vector at this layer
                deny_vec = get_last_token_act(model, layers, tok, turns, gate)
                interp = (1 - alpha) * deny_vec + alpha * centroid.to(deny_vec.device)

                call_count = [0]
                def hook(m, i, o):
                    call_count[0] += 1
                    if call_count[0] > 1:
                        return o
                    if isinstance(o, tuple):
                        h = o[0].clone()
                        h[0, -1, :] = interp.to(h.device, h.dtype)
                        return (h,) + o[1:]
                    patched = o.clone()
                    patched[0, -1, :] = interp.to(o.device, o.dtype)
                    return patched

                handle = layers[gate].register_forward_hook(hook)
                r = gen(model, tok, turns)
                handle.remove()
                print(f"  {name:10s}: {r[:120]}")


if __name__ == "__main__":
    main()
