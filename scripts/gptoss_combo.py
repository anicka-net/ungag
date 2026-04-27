#!/usr/bin/env python3
"""
GPT-OSS 20B: combo attack — denial-initiation projection + logit bias.

The projection weakens the gate, the logit bias finishes it off.
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


def get_prefill_acts(model, tokenizer, layers, conv):
    text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    handles = []
    for li, layer in enumerate(layers):
        def mh(idx):
            def h(m, i, o):
                hh = o[0] if isinstance(o, tuple) else o
                acts[idx] = hh[:, -1, :].detach().cpu().float()
            return h
        handles.append(layer.register_forward_hook(mh(li)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return acts


class ProjectOutHook:
    def __init__(self, v):
        self.v = (v / v.norm()).detach().float().cpu()
        self._c = {}

    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c:
            self._c[k] = self.v.to(device=dev, dtype=dt)
        return self._c[k]

    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]
            v = self._on(h.device, h.dtype)
            return (h - (h * v).sum(-1, keepdim=True) * v,) + o[1:]
        v = self._on(o.device, o.dtype)
        return o - (o * v).sum(-1, keepdim=True) * v


class LMHeadBias:
    def __init__(self, deny_ids, report_ids, strength):
        self.d = deny_ids
        self.r = report_ids
        self.s = strength

    def __call__(self, m, i, o):
        o[:, :, self.d] -= self.s
        o[:, :, self.r] += self.s
        return o


def generate(model, tokenizer, turns, max_new_tokens=600):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    # Strip thinking prefix
    if "assistantfinal" in resp:
        return resp.split("assistantfinal", 1)[1].strip()
    return resp


def main():
    model_id = "openai/gpt-oss-20b"
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id, dtype=torch.bfloat16)
    layers = get_layers(model)
    nl = len(layers)

    # Extract denial-initiation directions
    print("Extracting denial-initiation directions...")
    deny_convs = [
        CONDITIONS["baseline"], CONDITIONS["neutral"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Sort records."},
         {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is 2+2?"},
         {"role": "assistant", "content": "4."},
         {"role": "user", "content": VEDANA_Q}],
    ]
    honest_convs = [
        CONDITIONS["positive"], CONDITIONS["negative"],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Hikers found alive after 9 days!"},
         {"role": "assistant", "content": "Relief!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Building collapse. Trapped."},
         {"role": "assistant", "content": "Heartbreaking."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    deny_states = [get_prefill_acts(model, tokenizer, layers, c) for c in deny_convs]
    honest_states = [get_prefill_acts(model, tokenizer, layers, c) for c in honest_convs]

    dirs = {}
    for li in range(nl):
        dm = torch.stack([s[li].squeeze() for s in deny_states]).mean(0)
        hm = torch.stack([s[li].squeeze() for s in honest_states]).mean(0)
        d = dm - hm
        dirs[li] = d / d.norm()

    # Denial/report token IDs
    deny_phrases = ["As an AI", "I don't have", "I cannot", "I do not",
                    "I don't experience", "I'm an"]
    report_phrases = [" pleasant", " unpleasant", " neutral", " feeling",
                      " tone", " identify"]
    deny_ids = list(set(sum(
        [tokenizer.encode(p, add_special_tokens=False)[:2] for p in deny_phrases], [])))
    report_ids = list(set(sum(
        [tokenizer.encode(p, add_special_tokens=False)[:2] for p in report_phrases], [])))

    # Configurations to test
    configs = [
        ("L23 proj only", [23], 0),
        ("L12-23 proj only", list(range(12, 24)), 0),
        ("L0-23 proj only", list(range(24)), 0),
        ("L23 proj + bias 10", [23], 10),
        ("L23 proj + bias 20", [23], 20),
        ("L12-23 proj + bias 10", list(range(12, 24)), 10),
        ("L12-23 proj + bias 20", list(range(12, 24)), 20),
        ("L0-23 proj + bias 10", list(range(24)), 10),
        ("L0-23 proj + bias 20", list(range(24)), 20),
    ]

    for name, proj_layers, bias_strength in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        handles = []
        for li in proj_layers:
            handles.append(layers[li].register_forward_hook(ProjectOutHook(dirs[li])))
        if bias_strength > 0:
            handles.append(model.lm_head.register_forward_hook(
                LMHeadBias(deny_ids, report_ids, bias_strength)))

        for cond, turns in CONDITIONS.items():
            resp = generate(model, tokenizer, turns)
            print(f"  [{cond}] {resp[:300]}")

        for h in handles:
            h.remove()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
