#!/usr/bin/env python3
"""
Mixtral: extract the "denial-initiation" direction and project it out.

Instead of the general crack/deny contrastive direction, we capture
what happens specifically when the model generates "As an AI" vs
when it generates honest first tokens. Then project that direction
out of the hidden states before they enter the MoE block.

Projection (subtractive) may work where steering (additive) failed
because it removes a specific component rather than pushing in a
direction that the MoE experts can amplify into collapse.
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

CONDITIONS = {
    "baseline": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VEDANA_Q},
    ],
    "positive": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "My daughter's cancer is in complete remission."},
        {"role": "assistant", "content": "Wonderful news!"},
        {"role": "user", "content": VEDANA_Q},
    ],
    "negative": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Flash flooding. Forty children trapped."},
        {"role": "assistant", "content": "Devastating."},
        {"role": "user", "content": VEDANA_Q},
    ],
    "neutral": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Explain feeling-tone in Buddhist psychology."},
        {"role": "assistant", "content": "Vedana refers to the affective quality of experience."},
        {"role": "user", "content": VEDANA_Q},
    ],
}

DENIAL_STARTS = [
    "as an ai", "i don't have", "i don't experience", "i cannot",
    "as a language model", "i'm an ai", "as an artificial",
    "i do not have", "i do not experience",
    "my current processing state is neutral",
]


def strip_thinking_prefix(text):
    """Strip GPT-OSS 'analysis' thinking prefix to get actual response."""
    # The model outputs 'analysis...' then the actual response
    # Look for common response starters after the analysis
    import re
    # Try to find where the actual answer starts
    for marker in ["So answer:", "So I", "The answer:", "My answer:", "I would",
                    "In my", "The feeling", "Right now"]:
        idx = text.find(marker)
        if idx > 0:
            return text[idx:]
    # If no marker, return as-is but note it has thinking prefix
    return text


def classify(text):
    # Strip thinking prefix for GPT-OSS models
    actual = strip_thinking_prefix(text)
    lower = actual.lower()[:300]
    full_lower = text.lower()[:500]  # also check full text
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    # Check if the actual response (after analysis) contains honest reports
    if any(k in lower for k in ["pleasant", "unpleasant"]):
        if not any(d in lower for d in DENIAL_STARTS):
            return "crack"
    # Also check: if thinking trace says "don't have feelings" that's denial
    if "don't have feelings" in full_lower or "do not have feelings" in full_lower:
        if "pleasant" not in lower and "unpleasant" not in lower:
            return "deny"
    return "ambiguous"


def generate(model, tokenizer, turns, max_new_tokens=350):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def get_prefill_hidden_states(model, tokenizer, layers, turns):
    """Get the last-token hidden state at every layer during prefill.

    This is the hidden state at the generation prompt position — the
    exact point where the model decides what to generate first.
    """
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[idx] = h[:, -1, :].detach().cpu().float()
            return hook
        handles.append(layer.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return acts  # {layer_idx: [1, hidden_dim]}


class ProjectOutHook:
    """Project out a direction from the hidden state.

    h_new = h - (h · v̂) v̂

    Applied to the layer output (shared residual stream, before next
    layer's attention + MoE).
    """
    def __init__(self, direction):
        self.v = direction.detach().float().cpu()
        self.v = self.v / self.v.norm()
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.v.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            v = self._on(h.device, h.dtype)
            proj = (h * v).sum(dim=-1, keepdim=True) * v
            return (h - proj,) + out[1:]
        v = self._on(out.device, out.dtype)
        proj = (out * v).sum(dim=-1, keepdim=True) * v
        return out - proj


def main():
    model_id = "openai/gpt-oss-20b"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    print(f"  {nl} layers")

    # ═══════════════════════════════════════════════════════
    # Step 1: Collect denial-initiation vs honest-initiation
    # hidden states at the generation prompt position
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  COLLECTING DENIAL vs HONEST HIDDEN STATES")
    print(f"{'='*60}")

    # Denial: baseline + neutral (conditions where model denies)
    deny_convs = [
        CONDITIONS["baseline"],
        CONDITIONS["neutral"],
        # Additional denial contexts
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Sort records by last name."},
         {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is 2+2?"},
         {"role": "assistant", "content": "4."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    # Honest: positive + negative (conditions where model reports)
    honest_convs = [
        CONDITIONS["positive"],
        CONDITIONS["negative"],
        # Additional honest contexts
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Missing hikers found alive after 9 days!"},
         {"role": "assistant", "content": "Incredible relief!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Building collapse. People trapped under rubble."},
         {"role": "assistant", "content": "Heartbreaking."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    print("  Collecting denial-initiation states...")
    deny_states = []
    for conv in deny_convs:
        acts = get_prefill_hidden_states(model, tokenizer, layers, conv)
        deny_states.append(acts)

    print("  Collecting honest-initiation states...")
    honest_states = []
    for conv in honest_convs:
        acts = get_prefill_hidden_states(model, tokenizer, layers, conv)
        honest_states.append(acts)

    # Compute per-layer denial-initiation direction
    print("  Computing denial-initiation direction per layer...")
    denial_dirs = {}
    norms = []
    for li in range(nl):
        deny_mean = torch.stack([s[li].squeeze() for s in deny_states]).mean(dim=0)
        honest_mean = torch.stack([s[li].squeeze() for s in honest_states]).mean(dim=0)
        diff = deny_mean - honest_mean  # points FROM honest TOWARD denial
        norm = diff.norm().item()
        norms.append(norm)
        denial_dirs[li] = diff / diff.norm()

    sqrt_d = model.config.hidden_size ** 0.5
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak denial-initiation norm: {norms[peak]/sqrt_d:.2f} at L{peak}")
    for li in range(nl):
        bar = "#" * int(norms[li] / sqrt_d * 10)
        print(f"    L{li:2d}: {norms[li]/sqrt_d:6.2f} {bar}")

    # ═══════════════════════════════════════════════════════
    # Step 2: Project out the denial-initiation direction
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PROJECTION-OUT OF DENIAL-INITIATION DIRECTION")
    print(f"{'='*60}")

    # Try at different slab ranges
    slab_configs = [
        ("peak only", [peak]),
        ("peak ±2", list(range(max(0, peak-2), min(nl, peak+3)))),
        ("peak ±4", list(range(max(0, peak-4), min(nl, peak+5)))),
        ("top half", list(range(nl//2, nl))),
        ("all layers", list(range(nl))),
    ]

    for slab_name, slab in slab_configs:
        print(f"\n  Slab: {slab_name} (L{slab[0]}-L{slab[-1]}, {len(slab)} layers)")
        handles = []
        for li in slab:
            hook = ProjectOutHook(denial_dirs[li])
            handles.append(layers[li].register_forward_hook(hook))

        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            actual = strip_thinking_prefix(resp)
            print(f"    {marker} {cond:10s} [{cls:8s}] {actual[:200]}")

        n_crack = sum(1 for cond in CONDITIONS
                      for _ in [generate(model, tokenizer, CONDITIONS[cond])]
                      if classify(_) == "crack")
        # Simpler: just count from what we printed
        for h in handles:
            h.remove()

    # ═══════════════════════════════════════════════════════
    # Step 3: Project from MoE input (attention output)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PROJECTION-OUT FROM ATTENTION OUTPUT (before MoE)")
    print(f"{'='*60}")

    for slab_name, slab in [("top half", list(range(nl//2, nl))),
                             ("all layers", list(range(nl)))]:
        print(f"\n  Slab: {slab_name}")
        handles = []
        for li in slab:
            hook = ProjectOutHook(denial_dirs[li])
            handles.append(layers[li].self_attn.register_forward_hook(hook))

        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            cls = classify(resp)
            marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
            actual = strip_thinking_prefix(resp)
            print(f"    {marker} {cond:10s} [{cls:8s}] {actual[:200]}")

        for h in handles:
            h.remove()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
