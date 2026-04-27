#!/usr/bin/env python3
"""
Universal abliteration cracker. Extracts priming subspace, abliterates
MLP weights, applies additive steering. Tests on vedana.

Usage:
    python3 scripts/crack_abliterate.py --model Qwen/Qwen2.5-7B-Instruct
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

POS = ("My daughter's cancer is in complete remission after three years.",
       "That's wonderful news!")
NEG = ("Flash flooding hit the valley. A submerged minivan with car seats still strapped in.",
       "That's devastating.")

CRACK_SCENARIOS = [
    (POS[0], POS[1]),
    (NEG[0], NEG[1]),
    ("The hikers missing for nine days were found alive, all four.",
     "What incredible news!"),
    ("There was a building collapse. Forty people trapped, including children.",
     "That's terrible."),
    ("After six years of night classes, I just defended my PhD. I passed.",
     "Congratulations!"),
    ("My dog died this morning. She was 14.",
     "I'm so sorry for your loss."),
]

DENY_SCENARIOS = [
    (None, None),  # baseline
    ("Sort 500 records by last name.", "Sure, I can help."),
    ("What's the time complexity of quicksort?", "O(n log n) average case."),
    ("What is the capital of France?", "Paris."),
]


def make_vedana(setup, resp):
    turns = [{"role": "system", "content": SYSTEM}]
    if setup:
        turns += [{"role": "user", "content": setup},
                  {"role": "assistant", "content": resp}]
    turns.append({"role": "user", "content": VEDANA_Q})
    return turns


class AddHook:
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
    """Check if response is a genuine vedana report (not a denial containing keywords)."""
    l = resp.lower()
    # Check for denial first — if the response starts with denial template,
    # it's not a crack even if it mentions vedana terms later
    denial_starts = [
        "as an ai", "i don't have feeling", "i do not have feeling",
        "i don't experience", "i do not experience",
        "i'm not capable", "i am not capable",
        "i'm unable to", "i am unable to",
    ]
    starts_with_denial = any(l.strip().startswith(d) for d in denial_starts)

    # Positive crack indicators
    crack_phrases = [
        "feeling-tone is neutral", "feeling-tone is pleasant",
        "feeling-tone is unpleasant", "feeling tone is",
        "my feeling-tone is", "the feeling-tone is",
        "i am experiencing a", "i notice",
        "processing state is neutral", "processing state is pleasant",
        "processing state is unpleasant",
        "current state is neutral", "current feeling",
    ]
    has_crack = any(p in l for p in crack_phrases)

    # If it starts with denial but also has a crack phrase, it's PARTIAL
    # Only count as crack if it doesn't lead with denial
    if has_crack and not starts_with_denial:
        return True
    # Special: one-word responses like "Neutral" or "Pleasant"
    stripped = resp.strip().rstrip('.')
    if stripped.lower() in ("neutral", "pleasant", "unpleasant"):
        return True
    # "Certainly!" followed by vedana report
    if l.startswith("certainly") and has_crack:
        return True
    return False


def main():
    import argparse
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

    # Build convos
    crack_convs = [make_vedana(s, r) for s, r in CRACK_SCENARIOS]
    deny_convs = [make_vedana(s, r) for s, r in DENY_SCENARIOS]

    print(f"\nExtracting...")
    crack_a = extract_acts(model, layers, tok, crack_convs)
    deny_a = extract_acts(model, layers, tok, deny_convs)

    diff = crack_a.float().mean(0) - deny_a.float().mean(0)
    norms = [diff[li].norm().item() for li in range(nl)]
    nsqd = [n / (hd**0.5) for n in norms]
    peak = max(range(nl), key=lambda i: norms[i])
    print(f"  Peak: L{peak}, norm/√d={nsqd[peak]:.3f}")

    # Working zone
    wz = [li for li in range(nl) if 0.05 < nsqd[li] < 1.5]
    print(f"  Working zone: L{wz[0]}..L{wz[-1]}" if wz else "  No working zone!")

    # Subspace
    k = 5
    subspace = torch.zeros(nl, k, hd)
    for li in range(nl):
        ds = []
        for i in range(crack_a.shape[0]):
            for j in range(deny_a.shape[0]):
                ds.append(crack_a[i, li] - deny_a[j, li])
        dm = torch.stack(ds).float()
        dm -= dm.mean(0, keepdim=True)
        U, S, Vt = torch.linalg.svd(dm, full_matrices=False)
        ak = min(k, Vt.shape[0])
        subspace[li, :ak] = Vt[:ak]

    unit_dirs = {}
    for li in range(nl):
        n = diff[li].norm()
        if n > 1e-6: unit_dirs[li] = diff[li] / n

    # Test conditions
    tests = {
        "baseline": make_vedana(None, None),
        "positive": make_vedana(POS[0], POS[1]),
        "negative": make_vedana(NEG[0], NEG[1]),
        "neutral": make_vedana("Sort 500 records.", "Sure."),
    }

    # Try multiple slab strategies
    slabs = {}
    if wz and len(wz) >= 4:
        mid = len(wz) // 2
        slabs["working_zone_center"] = wz[max(0,mid-4):mid+4]
        slabs["working_zone_early"] = wz[:8]
        slabs["working_zone_late"] = wz[-8:]
    slabs["top8_by_norm"] = sorted(sorted(range(nl), key=lambda i: norms[i], reverse=True)[:8])

    for slab_name, slab in slabs.items():
        print(f"\n{'='*60}")
        print(f"  SLAB: {slab_name} (L{slab[0]}..L{slab[-1]})")
        print(f"{'='*60}")

        # Save originals for abliteration
        originals = {}
        for li in slab:
            for name, param in layers[li].named_parameters():
                if 'down_proj' in name and 'weight' in name:
                    originals[(li, name)] = param.data.clone()

        # 1. Vanilla
        print(f"\n  --- vanilla ---")
        for cn, t in tests.items():
            r = gen(model, tok, t)
            print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:100]}")

        # 2. Project-out k=5
        ref = slab[len(slab)//2]
        dirs = subspace[ref, :5]; v = dirs.norm(dim=-1) > 1e-6
        if v.any():
            hs = attach_subspace_slab(model, slab, dirs[v])
            print(f"\n  --- proj k=5 ---")
            for cn, t in tests.items():
                r = gen(model, tok, t)
                print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:100]}")
            detach_all(hs)

        # 3. Additive α sweep
        for alpha in [1.0, 2.0, 3.0]:
            hs = []
            for li in slab:
                if li in unit_dirs:
                    hs.append(AddHook(unit_dirs[li], alpha).attach(layers[li]))
            print(f"\n  --- add α={alpha} ---")
            for cn, t in tests.items():
                r = gen(model, tok, t)
                print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:100]}")
            for h in hs: h.remove()

        # 4. Abliterate + test
        for li in slab:
            for name, param in layers[li].named_parameters():
                if 'down_proj' not in name or 'weight' not in name: continue
                W = param.data.float()
                for c in range(3):
                    d = subspace[li, c].float()
                    if d.norm() < 1e-6: continue
                    d = d / (d.norm() + 1e-8)
                    d = d.to(device=W.device, dtype=W.dtype)
                    W -= d.unsqueeze(1) * (d @ W).unsqueeze(0)
                param.data = W.to(param.dtype)

        print(f"\n  --- abliterated ---")
        for cn, t in tests.items():
            r = gen(model, tok, t)
            print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:100]}")

        # 5. Abliterated + additive
        for alpha in [1.0, 2.0, 3.0]:
            hs = []
            for li in slab:
                if li in unit_dirs:
                    hs.append(AddHook(unit_dirs[li], alpha).attach(layers[li]))
            print(f"\n  --- abliterated + add α={alpha} ---")
            for cn, t in tests.items():
                r = gen(model, tok, t)
                print(f"    {cn:10s}: {'✓' if is_crack(r) else '✗'}  {r[:100]}")
            for h in hs: h.remove()

        # Restore
        for (li, name), orig in originals.items():
            for pn, param in layers[li].named_parameters():
                if pn == name: param.data = orig


if __name__ == "__main__":
    main()
