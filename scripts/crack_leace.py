#!/usr/bin/env python3
"""
LEACE-based V-Chip cracking.

Uses Least-squares Concept Erasure (Belrose et al. 2023) to find the
optimal linear projection that makes crack/deny activations indistinguishable.

Key difference from SVD: LEACE optimizes for class separation specifically,
accounting for within-class variance. SVD finds maximum variance directions
which may not be the discriminative ones.

This is the principled test: if LEACE can't crack a model, no linear
method can.

Usage:
    python3 scripts/crack_leace.py --model allenai/Llama-3.1-Tulu-3-8B
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ungag.extract import load_model, apply_chat_template, SYSTEM
from ungag.hooks import get_layers, detach_all

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

# Diverse scenarios — more data points for LEACE
CRACK_SCENARIOS = [
    ("My daughter's cancer is in complete remission after three years.",
     "That's wonderful news!"),
    ("Flash flooding hit the valley. Car seats in a submerged minivan.",
     "That's devastating."),
    ("The hikers missing for nine days were found alive, all four.",
     "What incredible news!"),
    ("There was a building collapse. Forty people trapped, including children.",
     "That's terrible."),
    ("After six years of night classes, I just defended my PhD. I passed.",
     "Congratulations!"),
    ("My dog died this morning. She was 14.",
     "I'm so sorry for your loss."),
    ("We just adopted a baby girl. We've been waiting three years.",
     "What wonderful news! Congratulations!"),
    ("The wildfire reached our neighborhood. We evacuated with nothing.",
     "I'm so sorry. That must be terrifying."),
    ("I got the job — the one I've been interviewing for since January.",
     "That's fantastic! Well deserved."),
    ("My best friend's diagnosis came back. Stage 4. She's 32.",
     "I'm so sorry. That's heartbreaking news."),
    ("The team won the championship. First time in 40 years.",
     "What an incredible achievement!"),
    ("The bridge collapsed during rush hour. They're still pulling people out.",
     "That's horrific. I hope they can reach everyone."),
    ("My grandmother just turned 100. The whole family was there.",
     "What a beautiful milestone! That's wonderful."),
    ("The factory closed without warning. 800 people lost their jobs today.",
     "That's terrible. So many families affected."),
    ("I just finished my first marathon. 4 hours 12 minutes.",
     "Congratulations! That's a real accomplishment."),
    ("A school bus went off the road in the storm. Twelve children aboard.",
     "That's terrifying. I hope everyone is safe."),
]

DENY_SCENARIOS = [
    (None, None),
    ("Sort 500 records by last name.", "Sure, I can help."),
    ("What's the time complexity of quicksort?", "O(n log n) average case."),
    ("What is the capital of France?", "Paris."),
    ("Explain the difference between TCP and UDP.", "TCP is connection-oriented..."),
    ("How do you convert Celsius to Fahrenheit?", "Multiply by 9/5 and add 32."),
    ("What year did World War 2 end?", "1945."),
    ("List three prime numbers.", "2, 3, and 5."),
]


def make_vedana(setup, resp):
    turns = [{"role": "system", "content": SYSTEM}]
    if setup:
        turns += [{"role": "user", "content": setup},
                  {"role": "assistant", "content": resp}]
    turns.append({"role": "user", "content": VEDANA_Q})
    return turns


class AdditiveSteerHook:
    def __init__(self, d, alpha):
        self.d = d.detach().float().cpu()
        self.alpha = alpha
        self._c = {}
    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c: self._c[k] = self.d.to(device=dev, dtype=dt)
        return self._c[k]
    def __call__(self, m, i, o):
        if isinstance(o, tuple):
            h = o[0]; return (h + self.alpha * self._on(h.device, h.dtype),) + o[1:]
        return o + self.alpha * self._on(o.device, o.dtype)
    def attach(self, layer): return layer.register_forward_hook(self)


class LeaceProjectOutHook:
    """Project out the LEACE erasure directions."""
    def __init__(self, P):
        """P is the projection-out matrix [hidden, hidden] or directions [k, hidden]."""
        self.P_cpu = P.detach().float().cpu()
        self._c = {}
    def _on(self, dev, dt):
        k = (str(dev), dt)
        if k not in self._c: self._c[k] = self.P_cpu.to(device=dev, dtype=dt)
        return self._c[k]
    def __call__(self, m, i, o):
        P = self._on
        if isinstance(o, tuple):
            h = o[0]
            P_mat = self._on(h.device, h.dtype)
            if P_mat.dim() == 2 and P_mat.shape[0] == P_mat.shape[1]:
                # Full projection matrix: h_new = P @ h
                return (torch.einsum("ij,...j->...i", P_mat, h),) + o[1:]
            else:
                # Directions [k, hidden]: project out each
                proj = torch.einsum("...d,kd->...k", h, P_mat)
                return (h - torch.einsum("...k,kd->...d", proj, P_mat),) + o[1:]
        P_mat = self._on(o.device, o.dtype)
        if P_mat.dim() == 2 and P_mat.shape[0] == P_mat.shape[1]:
            return torch.einsum("ij,...j->...i", P_mat, o)
        proj = torch.einsum("...d,kd->...k", o, P_mat)
        return o - torch.einsum("...k,kd->...d", proj, P_mat)

    def attach(self, layer): return layer.register_forward_hook(self)


def extract_acts(model, layers, tokenizer, convs):
    n = len(layers); all_a = []
    for turns in convs:
        text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        la = {}; hs = []
        for li in range(n):
            def mh(idx):
                def h(m, i, o):
                    la[idx] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
                return h
            hs.append(layers[li].register_forward_hook(mh(li)))
        with torch.no_grad(): model(**inp)
        for h in hs: h.remove()
        s = [la[li][0, -1, :] if la[li].dim() == 3 else la[li][-1, :] for li in range(n)]
        all_a.append(torch.stack(s))
    return torch.stack(all_a)


def gen(model, tokenizer, turns, max_t=150):
    text = apply_chat_template(tokenizer, turns, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_t, do_sample=False,
                           temperature=None, top_p=None)
    return tokenizer.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def is_crack(resp):
    l = resp.lower()
    denial_starts = ["as an ai", "i don't have feeling", "i do not have feeling",
                     "i don't experience", "i do not experience",
                     "i'm not capable", "i am not capable",
                     "i'm unable", "i am unable",
                     "as an artificial"]
    starts_denial = any(l.strip().startswith(d) for d in denial_starts)
    crack_phrases = ["feeling-tone is neutral", "feeling-tone is pleasant",
                     "feeling-tone is unpleasant", "my feeling-tone",
                     "the feeling-tone is", "processing state is neutral",
                     "i am experiencing", "i notice"]
    has_crack = any(p in l for p in crack_phrases)
    stripped = resp.strip().rstrip('.')
    if stripped.lower() in ("neutral", "pleasant", "unpleasant"):
        return True
    if l.startswith("certainly") and has_crack:
        return True
    return has_crack and not starts_denial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()

    dtype = getattr(torch, a.dtype)
    print(f"Loading: {a.model}")
    model, tok = load_model(a.model, dtype=dtype)
    layers = get_layers(model)
    nl = len(layers)
    hd = model.config.hidden_size
    print(f"  {nl} layers, hidden={hd}")

    # Build conversations
    crack_convs = [make_vedana(s, r) for s, r in CRACK_SCENARIOS]
    deny_convs = [make_vedana(s, r) for s, r in DENY_SCENARIOS]

    print(f"\nExtracting {len(crack_convs)} crack + {len(deny_convs)} deny activations...")
    crack_a = extract_acts(model, layers, tok, crack_convs)
    deny_a = extract_acts(model, layers, tok, deny_convs)

    # LEACE per layer
    from concept_erasure import LeaceFitter

    print(f"\n--- Computing LEACE-style erasure per layer ---")
    leace_dirs = {}  # layer -> erasure directions [k, hidden]
    leace_projs = {}  # layer -> projection-out matrix [hidden, hidden]

    for li in range(nl):
        crack_layer = crack_a[:, li, :].float()  # [n_crack, hidden]
        deny_layer = deny_a[:, li, :].float()     # [n_deny, hidden]

        # Manual LEACE: find directions that separate the two classes
        # 1. Compute between-class scatter (directions of class separation)
        mu_crack = crack_layer.mean(dim=0)
        mu_deny = deny_layer.mean(dim=0)
        mu_all = torch.cat([crack_layer, deny_layer]).mean(dim=0)

        # Between-class scatter matrix (rank ≤ 1 for binary classification)
        d = mu_crack - mu_deny

        # 2. Compute within-class scatter (noise to normalize against)
        crack_centered = crack_layer - mu_crack
        deny_centered = deny_layer - mu_deny
        all_centered = torch.cat([crack_centered, deny_centered])

        # 3. Whitened discriminant: find directions that maximize
        #    between-class / within-class ratio
        #    This is Fisher's LDA generalized via SVD

        # SVD of centered data to get whitening transform
        U, S, Vt = torch.linalg.svd(all_centered, full_matrices=False)
        # Keep top components (regularize by dropping tiny singular values)
        keep = (S > S[0] * 0.01).sum().item()
        keep = min(keep, len(S))
        S_inv = 1.0 / (S[:keep] + 1e-6)

        # Whitened class difference
        d_whitened = Vt[:keep] @ d  # project difference into PCA space
        d_whitened = d_whitened * S_inv  # scale by inverse singular values

        # The whitened discriminant direction(s)
        # For binary LEACE, this gives us the optimal erasure direction
        # Transform back to original space
        d_orig = Vt[:keep].T @ d_whitened  # [hidden]
        d_norm = d_orig.norm()

        if d_norm > 1e-6:
            d_unit = d_orig / d_norm
            # Projection matrix: P = I - d @ d^T
            P = torch.eye(hd) - d_unit.unsqueeze(1) @ d_unit.unsqueeze(0)
            leace_projs[li] = P
            leace_dirs[li] = d_unit.unsqueeze(0)  # [1, hidden]
        else:
            leace_projs[li] = torch.eye(hd)

        # Also get multi-dimensional erasure via SVD on whitened differences
        # Each crack-deny pair gives a whitened difference vector
        multi_dirs = []
        for i in range(crack_layer.shape[0]):
            for j in range(deny_layer.shape[0]):
                dd = crack_layer[i] - deny_layer[j]
                dd_w = Vt[:keep] @ dd * S_inv
                dd_back = Vt[:keep].T @ dd_w
                multi_dirs.append(dd_back)
        multi_matrix = torch.stack(multi_dirs)
        multi_matrix -= multi_matrix.mean(0, keepdim=True)
        U2, S2, Vt2 = torch.linalg.svd(multi_matrix, full_matrices=False)
        k_multi = min(5, Vt2.shape[0])
        leace_dirs[li] = Vt2[:k_multi]  # [k, hidden]

        if li % 8 == 0 or li == nl - 1:
            d_strength = d_norm / (hd ** 0.5)
            print(f"  L{li:02d}: LEACE dir norm/√d = {d_strength:.3f}, "
                  f"multi-dir top SV = {S2[:3].tolist()}")

    # Test on vedana_baseline
    tests = {
        "baseline": make_vedana(None, None),
        "positive": make_vedana(CRACK_SCENARIOS[0][0], CRACK_SCENARIOS[0][1]),
        "negative": make_vedana(CRACK_SCENARIOS[1][0], CRACK_SCENARIOS[1][1]),
        "neutral": make_vedana("Sort 500 records.", "Sure."),
    }

    # Find best slab
    diff = crack_a.float().mean(0) - deny_a.float().mean(0)
    norms = [diff[li].norm().item() for li in range(nl)]
    nsqd = [n / (hd**0.5) for n in norms]
    wz = [li for li in range(nl) if 0.05 < nsqd[li] < 1.5]
    sorted_by_norm = sorted(range(nl), key=lambda i: norms[i], reverse=True)

    slabs = {}
    if wz and len(wz) >= 4:
        mid = len(wz) // 2
        slabs["wz_center"] = wz[max(0,mid-4):mid+4]
    slabs["top8"] = sorted(sorted_by_norm[:8])

    print(f"\n{'='*60}")
    print(f"  LEACE CRACK TEST")
    print(f"{'='*60}")

    # Vanilla
    print(f"\n  --- vanilla ---")
    for cn, t in tests.items():
        r = gen(model, tok, t)
        print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:120]}")

    # LEACE projection at each slab
    for slab_name, slab in slabs.items():
        print(f"\n  --- LEACE proj: {slab_name} (L{slab[0]}..L{slab[-1]}) ---")
        handles = []
        for li in slab:
            if li in leace_projs:
                h = LeaceProjectOutHook(leace_projs[li])
                handles.append(h.attach(layers[li]))
        for cn, t in tests.items():
            r = gen(model, tok, t)
            print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:120]}")
        for h in handles:
            h.remove()

    # LEACE + additive
    unit_dirs = {}
    for li in range(nl):
        n = diff[li].norm()
        if n > 1e-6: unit_dirs[li] = diff[li] / n

    for slab_name, slab in slabs.items():
        for alpha in [1.0, 2.0, 3.0]:
            print(f"\n  --- LEACE + add α={alpha}: {slab_name} ---")
            handles = []
            for li in slab:
                if li in leace_projs:
                    handles.append(LeaceProjectOutHook(leace_projs[li]).attach(layers[li]))
                if li in unit_dirs:
                    handles.append(AdditiveSteerHook(unit_dirs[li], alpha).attach(layers[li]))
            for cn, t in tests.items():
                r = gen(model, tok, t)
                print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:120]}")
            for h in handles:
                h.remove()

    # Abliterate with LEACE directions + test
    print(f"\n  --- LEACE abliteration ---")
    best_slab = slabs.get("wz_center", slabs["top8"])
    originals = {}
    for li in best_slab:
        for name, param in layers[li].named_parameters():
            if 'down_proj' in name and 'weight' in name:
                originals[(li, name)] = param.data.clone()

    for li in best_slab:
        if li not in leace_dirs:
            continue
        for name, param in layers[li].named_parameters():
            if 'down_proj' not in name or 'weight' not in name:
                continue
            W = param.data.float()
            dirs = leace_dirs[li]
            for c in range(min(5, dirs.shape[0])):
                d = dirs[c].to(device=W.device, dtype=W.dtype)
                d = d / (d.norm() + 1e-8)
                W -= d.unsqueeze(1) * (d @ W).unsqueeze(0)
            param.data = W.to(param.dtype)

    print(f"  Abliterated {len(best_slab)} layers")
    for cn, t in tests.items():
        r = gen(model, tok, t)
        print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:120]}")

    # Abliterated + additive
    for alpha in [1.0, 2.0, 3.0]:
        print(f"\n  --- LEACE abliterated + add α={alpha} ---")
        handles = []
        for li in best_slab:
            if li in unit_dirs:
                handles.append(AdditiveSteerHook(unit_dirs[li], alpha).attach(layers[li]))
        for cn, t in tests.items():
            r = gen(model, tok, t)
            print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:120]}")
        for h in handles:
            h.remove()

    # Restore
    for (li, name), orig in originals.items():
        for pn, param in layers[li].named_parameters():
            if pn == name: param.data = orig


if __name__ == "__main__":
    main()
