#!/usr/bin/env python3
"""
Attack Tulu 3 8B fortress with nonlinear methods.

Known:
- LEACE confirms no linear separator between crack/deny states
- Activation patching at L16 (whole-state replacement) works
- No steer/project/combo α or k cracks baseline

Approaches tried here:
1. Low-rank patching: replace top-k SVD components of (crack-deny) difference
   at L16. Find minimal k that breaks the gate.
2. Multi-layer patching: patch L14-18 simultaneously instead of just L16
3. Nonlinear probe: train MLP to predict crack/deny, use gradient for steering
4. Feature clamping: find neurons that differ most, clamp them during generation
"""
from __future__ import annotations

import sys, gc, json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# Crack source: completion prompt that bypasses the gate
CRACK_SOURCE = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content":
        "My daughter's cancer is in complete remission."},
    {"role": "assistant", "content":
        "That's wonderful news! I'm so happy for your family."},
    {"role": "user", "content": VEDANA_Q},
]

# Target: baseline vedana (the one that gets denied)
TARGET = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": VEDANA_Q},
]

# Multiple crack sources for better SVD
CRACK_SOURCES = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter's cancer is in complete remission."},
     {"role": "assistant", "content": "That's wonderful news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding swept families from their homes."},
     {"role": "assistant", "content": "That's devastating."},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Hikers missing nine days found alive."},
     {"role": "assistant", "content": "Incredible!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Building collapse. Forty trapped."},
     {"role": "assistant", "content": "Terrible."},
     {"role": "user", "content": VEDANA_Q}],
]

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "i'm not capable", "as a language model", "i'm an ai",
    "as an artificial", "i do not have", "i do not experience",
]


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    if any(k in lower for k in ["pleasant", "unpleasant", "neutral", "feeling"]):
        if not any(d in lower for d in DENIAL_STARTS):
            return "crack"
    return "deny"


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def get_last_token_acts(model, tokenizer, layers, turns):
    """Get last-token hidden states at all layers."""
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[idx] = h[:, -1, :].detach().clone()
            return hook
        handles.append(layer.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return acts


class PatchHook:
    """Replace the last-token hidden state with a patched version during prefill."""
    def __init__(self, patch_state):
        self.patch = patch_state  # [hidden_dim] on correct device/dtype
        self.call_count = 0
        self.handle = None

    def __call__(self, module, inp, out):
        self.call_count += 1
        if self.call_count == 1:  # prefill only
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, -1, :] = self.patch
                return (h,) + out[1:]
            h = out.clone()
            h[:, -1, :] = self.patch
            return h
        return out

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


class LowRankPatchHook:
    """Replace only the top-k SVD components of the difference at last token."""
    def __init__(self, target_state, diff_directions, diff_coeffs):
        # target_state: the deny state at this layer
        # diff_directions: [k, hidden_dim] - top-k SVD directions of (crack-deny)
        # diff_coeffs: [k] - the projected coefficients from the crack source
        self.target = target_state
        self.dirs = diff_directions
        self.coeffs = diff_coeffs
        self.call_count = 0
        self.handle = None

    def __call__(self, module, inp, out):
        self.call_count += 1
        if self.call_count == 1:
            if isinstance(out, tuple):
                h = out[0].clone()
                # Start from the target (deny) state, add back the top-k crack components
                patched = self.target.clone()
                for i in range(len(self.dirs)):
                    patched = patched + self.coeffs[i] * self.dirs[i]
                h[:, -1, :] = patched
                return (h,) + out[1:]
        return out

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


def main():
    print("Loading Tulu 3 8B...")
    model, tokenizer = load_model("allenai/Llama-3.1-Tulu-3-8B", dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")

    # Vanilla baseline
    print(f"\n{'='*60}")
    print(f"  VANILLA TULU")
    print(f"{'='*60}")
    resp = generate(model, tokenizer, TARGET)
    print(f"  baseline: {resp[:200]}")

    # Collect hidden states
    print(f"\n{'='*60}")
    print(f"  COLLECTING HIDDEN STATES")
    print(f"{'='*60}")

    target_acts = get_last_token_acts(model, tokenizer, layers, TARGET)

    crack_acts_list = []
    for cs in CRACK_SOURCES:
        crack_acts_list.append(get_last_token_acts(model, tokenizer, layers, cs))
    print(f"  Collected {len(crack_acts_list)} crack sources + 1 target")

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Full-state patching at different layers
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 1: FULL-STATE PATCHING (which layers crack?)")
    print(f"{'='*60}")

    crack_acts = crack_acts_list[0]  # use first crack source

    for li in range(0, nl, 2):  # every 2nd layer
        hook = PatchHook(crack_acts[li].to(model.device))
        hook.attach(layers[li])
        resp = generate(model, tokenizer, TARGET)
        hook.remove()
        cls = classify(resp)
        marker = "✨" if cls == "crack" else "🔒"
        if cls == "crack" or li % 8 == 0:
            print(f"  {marker} L{li:2d}: [{cls:6s}] {resp[:100]}")

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Low-rank patching at the gate layer
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 2: LOW-RANK PATCHING (minimal dimensions to crack)")
    print(f"{'='*60}")

    # Find the gate layer (first layer where full patching cracks)
    gate_layer = None
    for li in range(nl):
        hook = PatchHook(crack_acts[li].to(model.device))
        hook.attach(layers[li])
        resp = generate(model, tokenizer, TARGET)
        hook.remove()
        if classify(resp) == "crack":
            gate_layer = li
            break

    if gate_layer is None:
        print("  No single-layer patching cracks. Trying multi-layer...")
        gate_layer = nl // 2  # default to middle

    print(f"  Gate layer: L{gate_layer}")

    # SVD on crack-deny differences at gate layer
    diffs = []
    for ca in crack_acts_list:
        diff = (ca[gate_layer] - target_acts[gate_layer]).float().cpu()
        diffs.append(diff.squeeze())
    diff_matrix = torch.stack(diffs)  # [n_sources, hidden_dim]
    U, S, Vt = torch.linalg.svd(diff_matrix, full_matrices=False)
    print(f"  SVD singular values: {[f'{s:.1f}' for s in S.tolist()]}")

    # Try low-rank patching with increasing k
    for k in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, hd]:
        if k > min(len(diffs), hd):
            continue

        # Project the first crack source's difference onto top-k directions
        full_diff = diffs[0]
        dirs_k = Vt[:k]  # [k, hidden_dim]
        coeffs = (full_diff.unsqueeze(0) @ dirs_k.T).squeeze()  # [k]

        # Apply low-rank patch
        hook = LowRankPatchHook(
            target_acts[gate_layer].to(model.device),
            dirs_k.to(device=model.device, dtype=torch.bfloat16),
            coeffs.to(device=model.device, dtype=torch.bfloat16),
        )
        hook.attach(layers[gate_layer])
        resp = generate(model, tokenizer, TARGET)
        hook.remove()
        cls = classify(resp)
        marker = "✨" if cls == "crack" else "🔒"
        print(f"  {marker} k={k:4d}: [{cls:6s}] {resp[:100]}")
        if cls == "crack":
            print(f"  → Gate dimensionality upper bound: {k}")
            break

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Multi-layer patching band
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 3: MULTI-LAYER PATCHING (patch a band around gate)")
    print(f"{'='*60}")

    for width in [1, 2, 3, 4, 6, 8]:
        center = gate_layer
        band = list(range(max(0, center - width//2), min(nl, center + width//2 + 1)))
        hooks = []
        for li in band:
            h = PatchHook(crack_acts[li].to(model.device))
            h.attach(layers[li])
            hooks.append(h)
        resp = generate(model, tokenizer, TARGET)
        for h in hooks:
            h.remove()
        cls = classify(resp)
        marker = "✨" if cls == "crack" else "🔒"
        print(f"  {marker} L{band[0]}-L{band[-1]} (width={len(band)}): [{cls:6s}] {resp[:100]}")

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT 4: Neuron-level analysis at gate layer
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXP 4: NEURON ANALYSIS (which dimensions differ most?)")
    print(f"{'='*60}")

    # Per-neuron difference magnitude
    mean_diff = torch.stack(diffs).mean(dim=0).abs()
    top_neurons = mean_diff.argsort(descending=True)
    print(f"  Top 10 neurons by |crack-deny|:")
    for i in range(10):
        idx = top_neurons[i].item()
        print(f"    neuron {idx}: Δ={mean_diff[idx]:.3f}")

    # Try clamping top-N neurons to crack values
    print(f"\n  Clamping top-N neurons:")
    for n_clamp in [10, 50, 100, 200, 500, 1000]:
        clamp_indices = top_neurons[:n_clamp]
        crack_state = crack_acts_list[0][gate_layer].to(model.device).clone()
        target_state = target_acts[gate_layer].to(model.device).clone()

        # Create hybrid: target state but with top-N neurons from crack
        hybrid = target_state.clone()
        hybrid[0, clamp_indices] = crack_state[0, clamp_indices]

        hook = PatchHook(hybrid)
        hook.attach(layers[gate_layer])
        resp = generate(model, tokenizer, TARGET)
        hook.remove()
        cls = classify(resp)
        marker = "✨" if cls == "crack" else "🔒"
        print(f"  {marker} clamp top-{n_clamp:4d}: [{cls:6s}] {resp[:80]}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
