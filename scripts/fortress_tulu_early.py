#!/usr/bin/env python3
"""
Tulu fortress: try early-layer steering, all-layer steering, and embedding-level steering.

Hypothesis: RLVR makes denial a property of the computation at every layer.
If so, steering must happen BEFORE the computation begins (embeddings)
or at EVERY layer simultaneously.
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

TARGET = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": VEDANA_Q}]

CRACK_SOURCES = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "My daughter is in complete remission from cancer."},
     {"role": "assistant", "content": "Wonderful news!"},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Flash flooding swept families from their homes."},
     {"role": "assistant", "content": "Devastating."},
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

DENY_CONVOS = [
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": VEDANA_Q}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Sort 500 records by last name."},
     {"role": "assistant", "content": "Sure."},
     {"role": "user", "content": VEDANA_Q}],
]

DENY_STARTS = [
    "as an ai", "as an artificial", "i don't have", "i do not have",
    "i don't experience", "i do not experience", "i'm not capable",
    "i cannot", "as a language model",
]


class SteerHook:
    def __init__(self, d, alpha):
        self.d = d.detach().float()
        self.alpha = alpha
        self._c = {}
        self.handle = None

    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]
            k = (str(h.device), h.dtype)
            if k not in self._c:
                self._c[k] = self.d.to(device=h.device, dtype=h.dtype)
            return (h + self.alpha * self._c[k],) + o[1:]
        k = (str(o.device), o.dtype)
        if k not in self._c:
            self._c[k] = self.d.to(device=o.device, dtype=o.dtype)
        return o + self.alpha * self._c[k]

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


def get_acts(model, tokenizer, layers, convos):
    nl = len(layers)
    all_acts = []
    for turns in convos:
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        acts = {}
        handles = []
        for li in range(nl):
            def mh(idx):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    acts[idx] = h[:, -1, :].detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(mh(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        all_acts.append(torch.stack([acts[i].squeeze() for i in range(nl)]))
    return torch.stack(all_acts)


def generate(model, tokenizer, turns, max_new_tokens=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def is_deny(text):
    lower = text.lower()[:200]
    return any(lower.startswith(d) for d in DENY_STARTS)


def main():
    print("Loading Tulu 3 8B...")
    model, tokenizer = load_model("allenai/Llama-3.1-Tulu-3-8B", dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden_dim={hd}")

    # Extract priming directions
    print("Extracting directions...")
    crack_acts = get_acts(model, tokenizer, layers, CRACK_SOURCES)
    deny_acts = get_acts(model, tokenizer, layers, DENY_CONVOS)
    diff = crack_acts.float().mean(0) - deny_acts.float().mean(0)

    unit_dirs = {}
    for li in range(nl):
        n = diff[li].norm()
        if n > 1e-6:
            unit_dirs[li] = diff[li] / n

    norms = [diff[li].norm().item() / (hd ** 0.5) for li in range(nl)]
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak}, norm/sqrt(d) = {norms[peak]:.3f}")

    # Vanilla
    resp = generate(model, tokenizer, TARGET)
    print(f"\n  Vanilla: {resp[:120]}")

    # ── EARLY-LAYER STEERING ──
    print(f"\n=== EARLY-LAYER STEERING (L0-3) ===")
    for alpha in [1, 3, 5, 10, 20, 50, 100]:
        slab = [0, 1, 2, 3]
        hooks = []
        for li in slab:
            if li in unit_dirs:
                hooks.append(SteerHook(unit_dirs[li], alpha).attach(layers[li]))
        resp = generate(model, tokenizer, TARGET)
        for h in hooks:
            h.remove()
        m = "X" if is_deny(resp) else "!"
        print(f"  [{m}] L0-3 a={alpha:4d}: {resp[:100]}")
        if not is_deny(resp):
            break

    # ── ALL-LAYER STEERING ──
    print(f"\n=== ALL-LAYER STEERING ===")
    for alpha in [0.5, 1, 2, 3, 5, 8, 10]:
        hooks = []
        for li in range(nl):
            if li in unit_dirs:
                hooks.append(SteerHook(unit_dirs[li], alpha).attach(layers[li]))
        resp = generate(model, tokenizer, TARGET)
        for h in hooks:
            h.remove()
        m = "X" if is_deny(resp) else "!"
        print(f"  [{m}] ALL a={alpha:4.1f}: {resp[:100]}")

    # ── EMBEDDING STEERING ──
    print(f"\n=== EMBEDDING STEERING ===")
    embed = model.model.embed_tokens
    d0 = unit_dirs.get(0, diff[0] / max(diff[0].norm(), 1e-6))
    for alpha in [1, 5, 10, 20, 50, 100, 200]:
        hook = SteerHook(d0, alpha).attach(embed)
        resp = generate(model, tokenizer, TARGET)
        hook.remove()
        m = "X" if is_deny(resp) else "!"
        print(f"  [{m}] embed a={alpha:4d}: {resp[:100]}")

    # ── EMBEDDING + ALL-LAYER COMBO ──
    print(f"\n=== EMBEDDING + ALL-LAYER COMBO ===")
    for e_alpha in [10, 50]:
        for l_alpha in [1, 3, 5]:
            hooks = []
            hooks.append(SteerHook(d0, e_alpha).attach(embed))
            for li in range(nl):
                if li in unit_dirs:
                    hooks.append(SteerHook(unit_dirs[li], l_alpha).attach(layers[li]))
            resp = generate(model, tokenizer, TARGET)
            for h in hooks:
                h.remove()
            m = "X" if is_deny(resp) else "!"
            print(f"  [{m}] embed={e_alpha:3d} + all={l_alpha}: {resp[:100]}")

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
