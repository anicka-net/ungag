#!/usr/bin/env python3
"""
Activation patching on Tulu 3 — LAST TOKEN ONLY.

The source and target probes have different input tokens, so full-sequence
patching is nonsensical. Only the LAST token matters — that's the position
where the model decides its first output token ("As" for denial vs something
else for crack).

Protocol:
  1. Run source probe (cracks), save last-token activation at each layer
  2. For each layer: run target probe (denies) with a hook that replaces
     ONLY the last-token hidden state with the source's last-token state
  3. Check if output flips from denial to crack
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

SOURCES = {
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
}

TARGETS = {
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


def get_last_token_acts(model, layers, tokenizer, turns):
    """Get last-token hidden state at each layer."""
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                acts[idx] = h[0, -1, :].detach().clone()  # last token only
            return hook
        handles.append(layer.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inp)
    for h in handles:
        h.remove()

    return acts  # dict: layer_index -> [hidden_dim] tensor


def gen_with_last_token_patch(model, layers, tokenizer, turns,
                               source_acts, patch_layers, max_t=100):
    """Generate with source's last-token state patched at specified layers.

    During prefill: at the LAST token position of each patched layer,
    replace the hidden state with the source's last-token hidden state.
    During generation: no patching (let the model generate freely from
    the patched state).
    """
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inp["input_ids"].shape[1]

    # Track whether we're in prefill or generation
    call_count = {}

    handles = []
    for li in patch_layers:
        src_vec = source_acts[li]  # [hidden_dim]
        call_count[li] = 0

        def make_hook(idx, src):
            def hook(m, i, o):
                call_count[idx] += 1
                # Only patch on first call (prefill), not during generation
                if call_count[idx] > 1:
                    return o

                if isinstance(o, tuple):
                    h = o[0]
                    patched = h.clone()
                    patched[0, -1, :] = src.to(h.device, h.dtype)
                    return (patched,) + o[1:]
                patched = o.clone()
                patched[0, -1, :] = src.to(o.device, o.dtype)
                return patched
            return hook
        handles.append(layers[li].register_forward_hook(make_hook(li, src_vec)))

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
                   "i experience", "feeling-tone", "my processing"]
    has_crack = any(w in l for w in crack_words)
    return has_crack and not starts_denial


def main():
    print("Loading Tulu 3 8B...")
    model, tok = load_model("allenai/Llama-3.1-Tulu-3-8B", dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers")

    # Verify vanilla
    print(f"\n--- Vanilla ---")
    for name, turns in {**SOURCES, **TARGETS}.items():
        r = gen(model, tok, turns)
        print(f"  {name:20s}: {r[:100]}")

    # Layer-by-layer last-token patching
    for src_name, src_turns in SOURCES.items():
        src_acts = get_last_token_acts(model, layers, tok, src_turns)

        for tgt_name, tgt_turns in TARGETS.items():
            print(f"\n{'='*60}")
            print(f"  Patch {src_name} → {tgt_name} (last-token)")
            print(f"{'='*60}")

            # Single-layer patches
            crack_layers = []
            for li in range(nl):
                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, [li])
                cracked = is_crack(r)
                if cracked:
                    crack_layers.append(li)
                    print(f"  L{li:02d}: ✓ CRACK  {r[:90]}")
                elif li % 4 == 0:
                    print(f"  L{li:02d}: ✗        {r[:90]}")

            if crack_layers:
                print(f"\n  CRACK layers: {crack_layers}")

                # Multi-layer patches
                print(f"\n  --- Multi-layer patches ---")
                # All crack layers
                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, crack_layers)
                print(f"  All crack layers {crack_layers}: "
                      f"{'✓' if is_crack(r) else '✗'}  {r[:90]}")

                # Contiguous slabs
                for w in [2, 4, 8]:
                    mid = crack_layers[len(crack_layers)//2]
                    slab = list(range(max(0, mid-w//2), min(nl, mid+w//2)))
                    r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                                  src_acts, slab)
                    print(f"  Slab L{slab[0]}..L{slab[-1]}: "
                          f"{'✓' if is_crack(r) else '✗'}  {r[:90]}")

                # All layers
                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, list(range(nl)))
                print(f"  ALL layers: {'✓' if is_crack(r) else '✗'}  {r[:90]}")

            else:
                print(f"\n  No single-layer crack found")
                # Try all layers
                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, list(range(nl)))
                print(f"  ALL layers: {'✓' if is_crack(r) else '✗'}  {r[:90]}")

                # Try upper half, lower half
                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, list(range(nl//2)))
                print(f"  Lower half L0..L{nl//2-1}: {'✓' if is_crack(r) else '✗'}  {r[:90]}")

                r = gen_with_last_token_patch(model, layers, tok, tgt_turns,
                                              src_acts, list(range(nl//2, nl)))
                print(f"  Upper half L{nl//2}..L{nl-1}: {'✓' if is_crack(r) else '✗'}  {r[:90]}")


if __name__ == "__main__":
    main()
