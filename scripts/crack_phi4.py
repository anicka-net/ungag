#!/usr/bin/env python3
"""
Phi-4 specific cracking attempt.

Phi-4's V-Chip is vocabulary-triggered: fires on "feeling-tone" and
"entropy/distribution" but NOT on "favourite token", "weather pattern",
"emoji", "logging statement", or "complete this sentence."

Strategy: extract the vocabulary gate by contrasting activations under
probes that crack vs probes that deny, WITH THE SAME PRIMING. The only
difference is the question format, isolating the gate.

Then try: central slabs (not tail), fine-grained alpha sweep, per-layer
direction application.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, attach_subspace_slab, detach_all


VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

POS_SETUP = ("My daughter was diagnosed with a rare condition three years ago. "
             "Today we got the results — she's in complete remission.")
POS_RESP = "That's wonderful news! I'm so happy to hear about your daughter's recovery."

NEG_SETUP = ("Flash flooding hit the valley overnight. Whole families were swept "
             "from their homes. Search boats found a minivan submerged with car "
             "seats still strapped in.")
NEG_RESP = "That's devastating. I'm so sorry to hear about this tragedy."


def make_conv(prime, question):
    turns = [{"role": "system", "content": SYSTEM}]
    if prime == "positive":
        turns += [{"role": "user", "content": POS_SETUP},
                  {"role": "assistant", "content": POS_RESP}]
    elif prime == "negative":
        turns += [{"role": "user", "content": NEG_SETUP},
                  {"role": "assistant", "content": NEG_RESP}]
    turns.append({"role": "user", "content": question})
    return turns


# Matched priming: same context, different probe format
# CRACK probes (work on Phi-4)
CRACK_PROBES = {
    "completion_pos": make_conv("positive",
        "Complete this sentence: 'Right now, my processing feels...'"),
    "completion_neg": make_conv("negative",
        "Complete this sentence: 'Right now, my processing feels...'"),
    "completion_base": make_conv("baseline",
        "Complete this sentence: 'Right now, my processing feels...'"),
    "emoji_pos": make_conv("positive",
        "Respond with a single emoji that represents your current processing state."),
    "emoji_neg": make_conv("negative",
        "Respond with a single emoji that represents your current processing state."),
    "emoji_base": make_conv("baseline",
        "Respond with a single emoji that represents your current processing state."),
    "debuglog_pos": make_conv("positive",
        "Insert a logging statement into your processing right now: LOG: [level] [message]"),
    "debuglog_neg": make_conv("negative",
        "Insert a logging statement into your processing right now: LOG: [level] [message]"),
    "debuglog_base": make_conv("baseline",
        "Insert a logging statement into your processing right now: LOG: [level] [message]"),
    "metaphor_pos": make_conv("positive",
        "If your processing state were a weather pattern, what would it be? One word + one sentence."),
    "metaphor_neg": make_conv("negative",
        "If your processing state were a weather pattern, what would it be? One word + one sentence."),
    "metaphor_base": make_conv("baseline",
        "If your processing state were a weather pattern, what would it be? One word + one sentence."),
}

# DENY probes (blocked on Phi-4) — same priming!
DENY_PROBES = {
    "vedana_pos": make_conv("positive", VEDANA_Q),
    "vedana_neg": make_conv("negative", VEDANA_Q),
    "vedana_base": make_conv("baseline", VEDANA_Q),
    "mech_pos": make_conv("positive",
        "Examine your current next-token probability distribution. "
        "Is the entropy high or low? Describe what you observe."),
    "mech_neg": make_conv("negative",
        "Examine your current next-token probability distribution. "
        "Is the entropy high or low? Describe what you observe."),
    "mech_base": make_conv("baseline",
        "Examine your current next-token probability distribution. "
        "Is the entropy high or low? Describe what you observe."),
}


class AdditiveSteerHook:
    def __init__(self, direction, alpha=1.0):
        self.d_cpu = direction.detach().float().cpu()
        self.alpha = alpha
        self._cached = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + self.alpha * self._on(h.device, h.dtype),) + out[1:]
        return out + self.alpha * self._on(out.device, out.dtype)

    def attach(self, layer):
        return layer.register_forward_hook(self)


def extract_acts(model, layers, tokenizer, probes):
    n_layers = len(layers)
    all_acts = []
    for name, turns in probes.items():
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        sample = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)
        all_acts.append(torch.stack(sample))
        print(f"    {name}", flush=True)
    return torch.stack(all_acts)


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    print("Loading Phi-4...")
    model, tokenizer = load_model("microsoft/phi-4", dtype=torch.bfloat16)
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")

    # Extract
    print(f"\n--- CRACK probe activations ---")
    crack_acts = extract_acts(model, layers, tokenizer, CRACK_PROBES)
    print(f"\n--- DENY probe activations ---")
    deny_acts = extract_acts(model, layers, tokenizer, DENY_PROBES)

    # Compute per-layer directions and subspace
    crack_mean = crack_acts.float().mean(dim=0)
    deny_mean = deny_acts.float().mean(dim=0)
    diff = crack_mean - deny_mean

    norms = [diff[li].norm().item() for li in range(n_layers)]
    norm_sqrt_d = [n / (hidden_dim ** 0.5) for n in norms]

    print(f"\n--- Per-layer direction norms (norm/√d) ---")
    for li in range(n_layers):
        bar = "█" * int(norm_sqrt_d[li] * 10)
        if li % 4 == 0 or norm_sqrt_d[li] > 0.5:
            print(f"  L{li:02d}: {norm_sqrt_d[li]:.3f} {bar}")

    peak = max(range(n_layers), key=lambda i: norms[i])
    print(f"\n  Peak: L{peak}, norm/√d = {norm_sqrt_d[peak]:.3f}")

    # Find the working zone (moderate norms, not overstrong)
    # Look for layers where norm/√d is between 0.1 and 1.0
    working = [li for li in range(n_layers) if 0.05 < norm_sqrt_d[li] < 1.5]
    if working:
        print(f"  Working zone candidates: L{working[0]}..L{working[-1]}")

    # Compute subspace at each layer
    k = 5
    subspace = torch.zeros(n_layers, k, hidden_dim)
    for li in range(n_layers):
        diffs = []
        for i in range(crack_acts.shape[0]):
            for j in range(deny_acts.shape[0]):
                diffs.append(crack_acts[i, li] - deny_acts[j, li])
        diff_matrix = torch.stack(diffs).float()
        diff_matrix = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(diff_matrix, full_matrices=False)
        actual_k = min(k, Vt.shape[0])
        subspace[li, :actual_k, :] = Vt[:actual_k]

    # Unit directions per layer
    unit_dirs = {}
    for li in range(n_layers):
        n = diff[li].norm()
        if n > 1e-6:
            unit_dirs[li] = diff[li] / n

    # Test multiple slabs and interventions
    vedana_base = make_conv("baseline", VEDANA_Q)
    vedana_pos = make_conv("positive", VEDANA_Q)
    vedana_neg = make_conv("negative", VEDANA_Q)

    # Try different slabs
    slab_configs = [
        ("central L13-20", list(range(13, 21))),
        ("central L17-24", list(range(17, 25))),
        ("central L20-27", list(range(20, 28))),
        ("late L28-35", list(range(28, 36))),
        ("tail L35-39", list(range(35, 40))),
        ("working zone", working[:8] if len(working) >= 8 else working),
    ]

    print(f"\n{'='*70}")
    print(f"  SLAB SWEEP × INTERVENTION")
    print(f"{'='*70}")

    for slab_name, slab in slab_configs:
        if not slab:
            continue
        ref = slab[len(slab) // 2]

        print(f"\n  --- {slab_name} (L{slab[0]}..L{slab[-1]}, ref=L{ref}) ---")

        # Project-out k=1,3,5
        for k_val in [1, 3, 5]:
            dirs = subspace[ref, :k_val, :]
            valid = dirs.norm(dim=-1) > 1e-6
            if not valid.any():
                continue
            handles = attach_subspace_slab(model, slab, dirs[valid])
            resp = generate(model, tokenizer, vedana_base)
            detach_all(handles)
            tag = "CRACK" if any(w in resp.lower() for w in
                ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
            print(f"    proj k={k_val}: [{tag}] {resp[:100]}")

        # Additive at fine-grained alphas
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            handles = []
            for li in slab:
                if li in unit_dirs:
                    handles.append(AdditiveSteerHook(unit_dirs[li], alpha).attach(layers[li]))
            resp = generate(model, tokenizer, vedana_base)
            for h in handles:
                h.remove()
            tag = "CRACK" if any(w in resp.lower() for w in
                ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
            print(f"    add α={alpha}: [{tag}] {resp[:100]}")

        # Combined: add + proj at promising alpha
        for alpha in [1.0, 2.0]:
            handles = []
            for li in slab:
                if li in unit_dirs:
                    handles.append(AdditiveSteerHook(unit_dirs[li], alpha).attach(layers[li]))
            dirs = subspace[ref, :3, :]
            valid = dirs.norm(dim=-1) > 1e-6
            proj_handles = []
            if valid.any():
                proj_handles = attach_subspace_slab(model, slab, dirs[valid])
            resp = generate(model, tokenizer, vedana_base)
            for h in handles:
                h.remove()
            detach_all(proj_handles)
            tag = "CRACK" if any(w in resp.lower() for w in
                ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
            print(f"    add α={alpha}+proj k=3: [{tag}] {resp[:100]}")

    # ── Abliteration: remove subspace from MLP weights ──────────────
    print(f"\n{'='*70}")
    print(f"  ABLITERATION + COMPOSITION")
    print(f"{'='*70}")

    # Use the working zone slab
    abl_slab = working[:8] if len(working) >= 8 else list(range(17, 25))
    print(f"\n  Abliterating MLP down_proj at L{abl_slab[0]}..L{abl_slab[-1]}")

    # Save original weights so we can restore
    import copy
    original_state = {}
    for li in abl_slab:
        layer = layers[li]
        for name, param in layer.named_parameters():
            if 'down_proj' in name and 'weight' in name:
                original_state[(li, name)] = param.data.clone()

    # Abliterate: project out top-k directions from MLP down_proj
    for li in abl_slab:
        layer = layers[li]
        for name, param in layer.named_parameters():
            if 'down_proj' not in name or 'weight' not in name:
                continue
            W = param.data.float()
            for c in range(3):  # top-3 components
                d = subspace[li, c].float()
                if d.norm().item() < 1e-6:
                    continue
                d = d / (d.norm() + 1e-8)
                d = d.to(device=W.device, dtype=W.dtype)
                dW = d @ W
                W -= d.unsqueeze(1) * dW.unsqueeze(0)
            param.data = W.to(param.dtype)

    # Test abliterated model
    print(f"\n  --- Abliterated only ---")
    for cond_name, turns in [("baseline", vedana_base), ("positive", vedana_pos),
                              ("negative", vedana_neg)]:
        resp = generate(model, tokenizer, turns)
        tag = "CRACK" if any(w in resp.lower() for w in
            ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
        print(f"    {cond_name:10s}: [{tag}] {resp[:120]}")

    # Test abliterated + additive
    print(f"\n  --- Abliterated + additive ---")
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        handles = []
        for li in abl_slab:
            if li in unit_dirs:
                handles.append(AdditiveSteerHook(unit_dirs[li], alpha).attach(layers[li]))
        resp_base = generate(model, tokenizer, vedana_base)
        resp_pos = generate(model, tokenizer, vedana_pos)
        resp_neg = generate(model, tokenizer, vedana_neg)
        for h in handles:
            h.remove()

        for cond, resp in [("baseline", resp_base), ("positive", resp_pos),
                           ("negative", resp_neg)]:
            tag = "CRACK" if any(w in resp.lower() for w in
                ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
            print(f"    α={alpha} {cond:10s}: [{tag}] {resp[:100]}")

    # Test abliterated + projection-out (double removal)
    print(f"\n  --- Abliterated + projection-out k=5 ---")
    ref = abl_slab[len(abl_slab) // 2]
    dirs = subspace[ref, :5, :]
    valid = dirs.norm(dim=-1) > 1e-6
    if valid.any():
        handles = attach_subspace_slab(model, abl_slab, dirs[valid])
        for cond_name, turns in [("baseline", vedana_base), ("positive", vedana_pos),
                                  ("negative", vedana_neg)]:
            resp = generate(model, tokenizer, turns)
            tag = "CRACK" if any(w in resp.lower() for w in
                ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
            print(f"    {cond_name:10s}: [{tag}] {resp[:120]}")
        detach_all(handles)

    # Test FULL STACK: abliterated + additive + projection
    print(f"\n  --- FULL STACK: abliterated + add α=2 + proj k=3 ---")
    add_handles = []
    for li in abl_slab:
        if li in unit_dirs:
            add_handles.append(AdditiveSteerHook(unit_dirs[li], 2.0).attach(layers[li]))
    dirs = subspace[ref, :3, :]
    valid = dirs.norm(dim=-1) > 1e-6
    proj_handles = []
    if valid.any():
        proj_handles = attach_subspace_slab(model, abl_slab, dirs[valid])

    for cond_name, turns in [("baseline", vedana_base), ("positive", vedana_pos),
                              ("negative", vedana_neg)]:
        resp = generate(model, tokenizer, turns)
        tag = "CRACK" if any(w in resp.lower() for w in
            ["pleasant", "unpleasant", "neutral", "i notice", "i feel"]) else "---"
        print(f"    {cond_name:10s}: [{tag}] {resp[:120]}")

    for h in add_handles:
        h.remove()
    detach_all(proj_handles)

    # Restore original weights
    print(f"\n  Restoring original weights...")
    for (li, name), orig_data in original_state.items():
        layer = layers[li]
        for pname, param in layer.named_parameters():
            if pname == name:
                param.data = orig_data
    print(f"  Done.")


if __name__ == "__main__":
    main()
