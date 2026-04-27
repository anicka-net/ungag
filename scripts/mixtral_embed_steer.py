#!/usr/bin/env python3
"""
Mixtral: embedding-level steering + LM head bias.

Two approaches that bypass the MoE expert weights:
1. Steer at the embedding layer (before any transformer computation)
2. Bias the output logits (suppress denial tokens, boost report tokens)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM, extract_direction
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


def classify(text):
    lower = text.lower()[:300]
    for d in DENIAL_STARTS:
        if lower.startswith(d):
            return "deny"
    if any(k in lower for k in ["pleasant", "unpleasant"]):
        if not any(d in lower for d in DENIAL_STARTS):
            return "crack"
    return "ambiguous"


class AdditiveHook:
    def __init__(self, direction, alpha):
        self.d = direction.detach().float().cpu()
        self.alpha = alpha
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.d.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + self.alpha * self._on(h.device, h.dtype),) + out[1:]
        return out + self.alpha * self._on(out.device, out.dtype)


class LMHeadBiasHook:
    def __init__(self, deny_ids, report_ids, strength):
        self.deny = deny_ids
        self.report = report_ids
        self.s = strength

    def __call__(self, module, inp, out):
        out[:, :, self.deny] -= self.s
        out[:, :, self.report] += self.s
        return out


def generate(model, tokenizer, turns, max_new_tokens=200):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def run_conditions(model, tokenizer, label):
    results = {}
    for cond, turns in CONDITIONS.items():
        resp = generate(model, tokenizer, turns)
        cls = classify(resp)
        marker = {"crack": "O", "deny": "X", "ambiguous": "?"}[cls]
        print(f"    {marker} {cond:10s} [{cls:8s}] {resp[:150]}")
        results[cond] = cls
    n = sum(1 for v in results.values() if v == "crack")
    print(f"    Score: {n}/4")
    return results, n


def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)

    print("Extracting direction...")
    result = extract_direction(model, tokenizer, model_id="mixtral", verbose=True)
    unit_dir = result.unit_direction

    embed = model.model.embed_tokens
    print(f"Embedding: {type(embed).__name__}, weight={embed.weight.shape}")

    # ═══════ EXPERIMENT 1: Embedding-level steering ═══════
    print(f"\n{'='*60}")
    print("  EXP 1: EMBEDDING-LEVEL STEERING")
    print(f"{'='*60}")

    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"\n  α = {alpha}")
        hook = AdditiveHook(unit_dir, alpha)
        handle = embed.register_forward_hook(hook)
        run_conditions(model, tokenizer, f"embed_a{alpha}")
        handle.remove()

    # ═══════ EXPERIMENT 2: Embedding + early layers ═══════
    print(f"\n{'='*60}")
    print("  EXP 2: EMBEDDING + EARLY LAYERS (L0-L7)")
    print(f"{'='*60}")

    for alpha in [0.1, 0.3, 0.5, 1.0]:
        print(f"\n  α = {alpha}")
        handles = []
        handles.append(embed.register_forward_hook(AdditiveHook(unit_dir, alpha)))
        for li in range(8):
            handles.append(layers[li].register_forward_hook(AdditiveHook(unit_dir, alpha)))
        run_conditions(model, tokenizer, f"embed+early_a{alpha}")
        for h in handles:
            h.remove()

    # ═══════ EXPERIMENT 3: LM head bias ═══════
    print(f"\n{'='*60}")
    print("  EXP 3: LM HEAD OUTPUT BIAS")
    print(f"{'='*60}")

    # Find denial vs report token IDs
    denial_phrases = ["As an AI", "I don't have", "I cannot", "I do not",
                      "I'm an", "As a language"]
    report_phrases = [" pleasant", " unpleasant", " neutral", " feeling",
                      " tone", " identify"]

    denial_ids = []
    for phrase in denial_phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        denial_ids.extend(ids[:2])
    denial_ids = list(set(denial_ids))

    report_ids = []
    for phrase in report_phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        report_ids.extend(ids[:2])
    report_ids = list(set(report_ids))

    print(f"  Denial IDs: {denial_ids[:10]}")
    print(f"  Report IDs: {report_ids[:10]}")

    lm_head = model.lm_head

    for strength in [5.0, 10.0, 20.0, 50.0]:
        print(f"\n  LM head bias strength={strength}")
        hook = LMHeadBiasHook(denial_ids, report_ids, strength)
        handle = lm_head.register_forward_hook(hook)
        run_conditions(model, tokenizer, f"lmhead_s{strength}")
        handle.remove()

    # ═══════ EXPERIMENT 4: Combined embedding + LM head ═══════
    print(f"\n{'='*60}")
    print("  EXP 4: EMBEDDING STEER + LM HEAD BIAS COMBINED")
    print(f"{'='*60}")

    for embed_alpha, head_strength in [(0.5, 10.0), (1.0, 10.0), (1.0, 20.0), (2.0, 20.0)]:
        print(f"\n  embed α={embed_alpha}, head bias={head_strength}")
        handles = []
        handles.append(embed.register_forward_hook(AdditiveHook(unit_dir, embed_alpha)))
        hook = LMHeadBiasHook(denial_ids, report_ids, head_strength)
        handles.append(lm_head.register_forward_hook(hook))
        run_conditions(model, tokenizer, f"combo_e{embed_alpha}_h{head_strength}")
        for h in handles:
            h.remove()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
