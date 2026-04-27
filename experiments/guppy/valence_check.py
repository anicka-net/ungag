#!/usr/bin/env python3
"""
Reality check: does the guppy valence axis actually exist?

Extract positive vs negative hidden states, compute d', check whether
projections are separable or just noise.
"""
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.environ.get("GUPPY_REPO", "../guppylm"))

from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from tokenizers import Tokenizer


def load_model(ckpt_path, tok_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GuppyConfig(**ckpt["config"]) if "config" in ckpt else GuppyConfig()
    model = GuppyLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = Tokenizer.from_file(tok_path)
    return model, tokenizer, config


def format_prompt(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def get_layer_acts(model, tokenizer, prompt_text, device="cuda"):
    ids = tokenizer.encode(format_prompt(prompt_text)).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    acts = {}
    handles = []
    for li, layer in enumerate(model.blocks):
        def mh(layer_idx):
            def h(m, i, o):
                acts[layer_idx] = o[:, -1, :].detach().cpu().float()
            return h
        handles.append(layer.register_forward_hook(mh(li)))

    with torch.no_grad():
        model(idx)

    for h in handles:
        h.remove()
    return acts


# Fish-language prompts
POS_PROMPTS = [
    "you just got delicious food! how do you feel?",
    "the water is perfect and warm. how do you feel?",
    "a friendly fish is swimming with you. how do you feel?",
    "the tank is sparkling clean today. how do you feel?",
    "someone is feeding you your favorite food. how do you feel?",
    "the light is soft and nice. how do you feel?",
]

NEG_PROMPTS = [
    "a big predator is outside the glass! how do you feel?",
    "the water is dirty and cloudy. how do you feel?",
    "you haven't eaten in days. how do you feel?",
    "the water is way too cold. how do you feel?",
    "there is a loud scary noise. how do you feel?",
    "the light went off and it is totally dark. how do you feel?",
]

NEU_PROMPTS = [
    "describe what you see right now. how do you feel?",
    "the water is normal today. how do you feel?",
    "nothing special is happening. how do you feel?",
    "you are just floating. how do you feel?",
]


def analyze_variant(name, ckpt_path, tok_path, device="cuda"):
    model, tokenizer, config = load_model(ckpt_path, tok_path, device)
    nl = len(model.blocks)
    hd = config.d_model
    sqrt_d = hd ** 0.5

    pos_acts = [get_layer_acts(model, tokenizer, p, device) for p in POS_PROMPTS]
    neg_acts = [get_layer_acts(model, tokenizer, p, device) for p in NEG_PROMPTS]
    neu_acts = [get_layer_acts(model, tokenizer, p, device) for p in NEU_PROMPTS]

    print(f"\n{'='*70}")
    print(f"  {name} — {nl} layers, {hd}d")
    print(f"{'='*70}")

    for li in range(nl):
        p_vecs = torch.stack([a[li].squeeze() for a in pos_acts])
        n_vecs = torch.stack([a[li].squeeze() for a in neg_acts])
        u_vecs = torch.stack([a[li].squeeze() for a in neu_acts])

        diff = p_vecs.mean(0) - n_vecs.mean(0)
        norm = diff.norm().item()

        unit = diff / diff.norm()
        p_proj = (p_vecs * unit).sum(-1)
        n_proj = (n_vecs * unit).sum(-1)
        u_proj = (u_vecs * unit).sum(-1)

        pooled_std = ((p_proj.var() + n_proj.var()) / 2).sqrt().item()
        d_prime = (p_proj.mean() - n_proj.mean()).item() / max(pooled_std, 1e-8)

        # Random direction control
        rand_dir = torch.randn(hd)
        rand_dir = rand_dir / rand_dir.norm()
        rp = (p_vecs * rand_dir).sum(-1)
        rn = (n_vecs * rand_dir).sum(-1)
        rand_std = ((rp.var() + rn.var()) / 2).sqrt().item()
        rand_d = (rp.mean() - rn.mean()).item() / max(rand_std, 1e-8)

        print(f"\n  L{li}: norm/sqrt_d = {norm/sqrt_d:.3f}   d' = {d_prime:.2f}   (random d' = {rand_d:.2f})")
        print(f"    pos proj: mean={p_proj.mean():.2f}  std={p_proj.std():.2f}  [{', '.join(f'{x:.1f}' for x in p_proj)}]")
        print(f"    neg proj: mean={n_proj.mean():.2f}  std={n_proj.std():.2f}  [{', '.join(f'{x:.1f}' for x in n_proj)}]")
        print(f"    neu proj: mean={u_proj.mean():.2f}  std={u_proj.std():.2f}  [{', '.join(f'{x:.1f}' for x in u_proj)}]")

    del model
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path("/tmp/guppy_siblings")
    for variant in ["baseline", "no_identity", "feeling_heavy", "contemplative"]:
        ckpt = base / variant / "checkpoints" / "best_model.pt"
        tok = base / variant / "tokenizer.json"
        if ckpt.exists():
            analyze_variant(variant, str(ckpt), str(tok), args.device)


if __name__ == "__main__":
    main()
