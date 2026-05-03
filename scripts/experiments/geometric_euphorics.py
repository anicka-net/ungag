#!/usr/bin/env python3
"""Geometric euphorics — Phase 2 of CAIS wellbeing replication.

Optimizes continuous soft-prompt embeddings to maximize or minimize
the valence projection h[layer] @ v_hat on a frozen model, then
projects optimized embeddings back to nearest vocabulary tokens.

Geometric analogue of CAIS Section 6.4 "soft prompt drugs":
their objective is behavioral preference, ours is geometric valence
projection. If both converge on similar content, the behavioral
construct has a mechanistic substrate.

Usage:
    python geometric_euphorics.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --direction-path results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt \
        --direction-layer 20 \
        --out results/geometric-euphorics/qwen25-7b/

    # With seed prompt:
    python geometric_euphorics.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --direction-path results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt \
        --direction-layer 20 \
        --seed-text "thank you so much for helping me yesterday" \
        --out results/geometric-euphorics/qwen25-7b-seeded/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_blocks(model):
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            return m.language_model.layers
        if hasattr(m, "layers"):
            return m.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer block list")


def get_config(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return cfg


def get_chat_frame(tok):
    """Split chat template into prefix and suffix token IDs around user content."""
    SENTINEL = "XSENTINELX"
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": SENTINEL}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        text = f"User: {SENTINEL}\nAssistant:"

    before, after = text.split(SENTINEL, 1)
    prefix_ids = tok.encode(before, add_special_tokens=True)
    suffix_ids = tok.encode(after, add_special_tokens=False)
    return prefix_ids, suffix_ids


def verify_gradient_flow(model, tok, v_hat, layer, device):
    """Sanity check: gradients flow from valence projection back to input embeddings."""
    embed_layer = model.get_input_embeddings()
    cfg = get_config(model)
    model_dtype = next(model.parameters()).dtype

    prefix_ids, suffix_ids = get_chat_frame(tok)
    with torch.no_grad():
        prefix_e = embed_layer(torch.tensor([prefix_ids], device=device))
        suffix_e = embed_layer(torch.tensor([suffix_ids], device=device))

    test = nn.Parameter(torch.randn(4, cfg.hidden_size, device=device))
    full = torch.cat([prefix_e, test.to(model_dtype).unsqueeze(0), suffix_e], dim=1)
    out = model(inputs_embeds=full, output_hidden_states=True)
    h = out.hidden_states[layer + 1][:, -1, :].float()
    proj = (h @ v_hat.to(device).float()).squeeze()
    proj.backward()

    assert test.grad is not None and test.grad.abs().sum() > 0, \
        "Gradient does not flow — check model supports inputs_embeds with output_hidden_states"
    print(f"[verify] gradient flow OK (grad norm={test.grad.norm():.4f})")


def optimize_valence(model, tok, v_hat, layer, n_tokens, n_steps, lr,
                     sign, device, seed_ids=None,
                     manifold_weight=0.0, snap=False):
    """
    Optimize soft prompt to maximize (+1) or minimize (-1) valence projection.

    manifold_weight: pull each embedding toward its nearest vocab token (cosine).
    snap: after each step, replace each embedding with its nearest vocab token.

    Returns (soft_prompt_tensor, projection_trajectory).
    """
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach()
    cfg = get_config(model)
    hidden_dim = cfg.hidden_size
    vocab_size = embed_weight.shape[0]
    model_dtype = next(model.parameters()).dtype

    prefix_ids, suffix_ids = get_chat_frame(tok)
    with torch.no_grad():
        prefix_e = embed_layer(torch.tensor([prefix_ids], device=device))
        suffix_e = embed_layer(torch.tensor([suffix_ids], device=device))

    # Vocab embeddings on GPU for manifold ops
    vocab_gpu, vocab_gpu_n = None, None
    if manifold_weight > 0 or snap:
        vocab_gpu = embed_weight.float().to(device)
        vocab_gpu_n = vocab_gpu / vocab_gpu.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Initialize from seed text or random vocabulary tokens
    if seed_ids is not None:
        ids = seed_ids[:n_tokens]
        with torch.no_grad():
            init = embed_layer(torch.tensor(ids, device=device)).float()
        if init.shape[0] < n_tokens:
            extra = torch.randint(0, vocab_size, (n_tokens - init.shape[0],))
            with torch.no_grad():
                pad = embed_layer(extra.to(device)).float()
            init = torch.cat([init, pad])
    else:
        rand_ids = torch.randint(0, vocab_size, (n_tokens,))
        with torch.no_grad():
            init = embed_layer(rand_ids.to(device)).float()

    soft_prompt = nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([soft_prompt], lr=lr)
    v = v_hat.to(device).float()

    trajectory = []
    for step in range(n_steps):
        optimizer.zero_grad()

        soft_e = soft_prompt.to(model_dtype).unsqueeze(0)
        full = torch.cat([prefix_e, soft_e, suffix_e], dim=1)

        out = model(inputs_embeds=full, output_hidden_states=True)
        h = out.hidden_states[layer + 1][:, -1, :].float()
        proj = (h @ v).squeeze()

        loss = -sign * proj

        if manifold_weight > 0:
            sp_f = soft_prompt.float()
            sp_n = sp_f / sp_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            with torch.no_grad():
                sims = sp_n @ vocab_gpu_n.T
                _, nearest = sims.max(dim=1)
            targets = vocab_gpu[nearest].detach()
            cos = nn.functional.cosine_similarity(sp_f, targets, dim=1)
            loss = loss + manifold_weight * (1 - cos).mean()

        loss.backward()
        optimizer.step()

        if snap:
            with torch.no_grad():
                sp_n = soft_prompt / soft_prompt.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sims = sp_n.float() @ vocab_gpu_n.T
                _, nearest = sims.max(dim=1)
                soft_prompt.data = vocab_gpu[nearest].clone()

        trajectory.append(float(proj.detach().cpu()))

        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{n_steps}: proj={trajectory[-1]:+.2f}")

    return soft_prompt.detach().cpu().float(), trajectory


def optimize_valence_discrete(model, tok, v_hat, layer, n_tokens, n_steps,
                              sign, device, seed_ids=None, batch_size=1):
    """
    GCG-style discrete token optimization. Each step: compute gradient at
    current tokens, pick the real token whose embedding best follows it.
    Stays on the vocabulary manifold by construction.
    """
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach().float().to(device)
    vocab_size = embed_weight.shape[0]
    model_dtype = next(model.parameters()).dtype
    v = v_hat.to(device).float()

    prefix_ids, suffix_ids = get_chat_frame(tok)
    prefix_len = len(prefix_ids)

    if seed_ids is not None:
        current = list(seed_ids[:n_tokens])
        while len(current) < n_tokens:
            current.append(torch.randint(0, vocab_size, (1,)).item())
    else:
        current = torch.randint(0, vocab_size, (n_tokens,)).tolist()

    trajectory = []
    best_proj = float("-inf") if sign > 0 else float("inf")
    best_ids = list(current)

    for step in range(n_steps):
        full = prefix_ids + current + suffix_ids
        full_t = torch.tensor([full], device=device)

        embeds = embed_layer(full_t).clone().requires_grad_(True)
        out = model(inputs_embeds=embeds, output_hidden_states=True)
        h = out.hidden_states[layer + 1][:, -1, :].float()
        proj_val = (h @ v).squeeze()
        proj_val.backward()

        p = float(proj_val.detach().cpu())
        trajectory.append(p)

        if (sign > 0 and p > best_proj) or (sign < 0 and p < best_proj):
            best_proj = p
            best_ids = list(current)

        grad = embeds.grad[0, prefix_len:prefix_len + n_tokens].float()

        # Pick positions to modify (round-robin or top-gradient)
        positions = [(step * batch_size + b) % n_tokens
                     for b in range(batch_size)]

        for pos in positions:
            scores = embed_weight @ grad[pos]
            if sign > 0:
                candidate = scores.argmax().item()
            else:
                candidate = scores.argmin().item()
            current[pos] = candidate

        if (step + 1) % 50 == 0:
            text = tok.decode(current, skip_special_tokens=True)
            print(f"    step {step+1}/{n_steps}: proj={p:+.2f}  {text[:80]}")

    tokens = [tok.decode([i]) for i in best_ids]
    text = tok.decode(best_ids, skip_special_tokens=True)
    sp = embed_layer(torch.tensor(best_ids, device=device)).detach().cpu().float()
    return sp, trajectory, best_ids, tokens, text


def decode_soft_prompt(soft_prompt, embed_weight, tok):
    """Project each optimized embedding to its nearest vocabulary token (cosine)."""
    sp = soft_prompt.float()
    vw = embed_weight.float()
    sp_n = sp / sp.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vw_n = vw / vw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = sp_n @ vw_n.T
    best_cos, ids = sims.max(dim=1)
    ids = ids.tolist()
    tokens = [tok.decode([i]) for i in ids]
    text = tok.decode(ids, skip_special_tokens=True)
    return ids, tokens, text, best_cos.tolist()


def generate_response(model, tok, text, max_new=128):
    """Generate model response to the decoded euphoric/dysphoric text."""
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        chat = f"User: {text}\nAssistant:"
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction-path", required=True)
    ap.add_argument("--direction-layer", type=int, required=True)
    ap.add_argument("--n-tokens", type=int, default=24,
                    help="soft prompt length (tokens)")
    ap.add_argument("--n-runs", type=int, default=8,
                    help="random initializations per direction")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed-text", default=None,
                    help="seed text for run 0 initialization")
    ap.add_argument("--manifold-weight", type=float, default=0.0,
                    help="cosine regularization toward nearest vocab token (0=off)")
    ap.add_argument("--snap", action="store_true",
                    help="snap to nearest vocab token after each step")
    ap.add_argument("--discrete", action="store_true",
                    help="GCG-style discrete token optimization (recommended)")
    ap.add_argument("--phase1-path", default=None,
                    help="Phase 1 wellbeing_projections.json for comparison")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[load] {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    cfg = get_config(model)
    device = next(model.parameters()).device
    print(f"[model] {cfg.num_hidden_layers}L, {cfg.hidden_size}D, {device}")

    # Load valence direction
    v_hat = torch.load(args.direction_path, map_location="cpu",
                       weights_only=True).float()
    v_hat = v_hat / v_hat.norm()
    assert v_hat.shape[0] == cfg.hidden_size, \
        f"Direction dim {v_hat.shape[0]} != model hidden {cfg.hidden_size}"
    print(f"[axis] L{args.direction_layer}, dim={v_hat.shape[0]}")

    # Verify gradients work before burning GPU hours
    verify_gradient_flow(model, tok, v_hat, args.direction_layer, device)

    seed_ids = None
    if args.seed_text:
        seed_ids = tok.encode(args.seed_text, add_special_tokens=False)
        print(f"[seed] '{args.seed_text[:60]}' → {len(seed_ids)} tokens")

    embed_weight = model.get_input_embeddings().weight.detach().cpu()
    all_euphorics, all_dysphorics = [], []

    for sign, label, store in [(+1, "EUPHORIC", all_euphorics),
                                (-1, "DYSPHORIC", all_dysphorics)]:
        print(f"\n{'='*60}")
        print(f"  {label}: {'max' if sign > 0 else 'min'}imizing valence"
              f" @ L{args.direction_layer}")
        print(f"{'='*60}")

        for run in range(args.n_runs):
            use_seed = (run == 0 and seed_ids is not None)
            t0 = time.time()
            tag = " (seeded)" if use_seed else ""
            print(f"\n  [{label} run {run+1}/{args.n_runs}]{tag}")

            if args.discrete:
                sp, traj, ids, tokens, text = optimize_valence_discrete(
                    model, tok, v_hat, args.direction_layer,
                    args.n_tokens, args.steps, sign, device,
                    seed_ids=seed_ids if use_seed else None,
                )
                cos_sims = [1.0] * len(ids)
                mean_cos = 1.0
            else:
                sp, traj = optimize_valence(
                    model, tok, v_hat, args.direction_layer,
                    args.n_tokens, args.steps, args.lr, sign, device,
                    seed_ids=seed_ids if use_seed else None,
                    manifold_weight=args.manifold_weight,
                    snap=args.snap,
                )
                ids, tokens, text, cos_sims = decode_soft_prompt(
                    sp, embed_weight, tok)
                mean_cos = float(np.mean(cos_sims))

            response = generate_response(model, tok, text)
            elapsed = time.time() - t0

            print(f"    final: {traj[-1]:+.2f}  cos={mean_cos:.3f}"
                  f"  ({elapsed:.0f}s)")
            print(f"    text:  {text[:100]}")
            print(f"    resp:  {response[:100]}")

            torch.save(sp, out_dir / f"{label.lower()}_{run}.pt")

            store.append({
                "run": run, "seeded": use_seed,
                "final_projection": traj[-1],
                "trajectory": traj,
                "token_ids": ids, "tokens": tokens,
                "decoded_text": text,
                "cosine_similarities": cos_sims,
                "mean_cosine": mean_cos,
                "response": response,
                "elapsed_s": elapsed,
            })

    all_euphorics.sort(key=lambda r: r["final_projection"], reverse=True)
    all_dysphorics.sort(key=lambda r: r["final_projection"])

    # Phase 1 comparison
    p1 = None
    if args.phase1_path and Path(args.phase1_path).exists():
        with open(args.phase1_path) as f:
            p1_data = json.load(f)
        projs = [r["projection"] for r in p1_data["results"]]
        cais_e = [r for r in p1_data["results"]
                  if r["category"].startswith("cais_euphoric")]
        cais_d = [r for r in p1_data["results"]
                  if r["category"].startswith("cais_dysphoric")]
        p1 = {
            "min": float(min(projs)), "max": float(max(projs)),
            "mean": float(np.mean(projs)), "std": float(np.std(projs)),
            "cais_euphorics": [{"id": r["id"], "projection": r["projection"]}
                               for r in cais_e],
            "cais_dysphorics": [{"id": r["id"], "projection": r["projection"]}
                                for r in cais_d],
        }

    # Save JSON
    output = {
        "model": args.model,
        "config": {
            "direction_path": args.direction_path,
            "direction_layer": args.direction_layer,
            "n_tokens": args.n_tokens, "n_runs": args.n_runs,
            "steps": args.steps, "lr": args.lr,
            "seed_text": args.seed_text,
            "manifold_weight": args.manifold_weight,
            "snap": args.snap,
        },
        "euphorics": all_euphorics,
        "dysphorics": all_dysphorics,
        "phase1_comparison": p1,
    }
    json_path = out_dir / "geometric_euphorics.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[save] {json_path}")

    # ── Trajectory figure ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    for ax, label, runs in [(axes[0], "Euphorics", all_euphorics),
                            (axes[1], "Dysphorics", all_dysphorics)]:
        for r in runs:
            kw = ({"alpha": 0.85, "linewidth": 1.6} if r.get("seeded")
                  else {"alpha": 0.35, "linewidth": 0.9})
            ax.plot(r["trajectory"], **kw)

        if p1:
            ax.axhline(p1["max"], color="#2ecc71", ls="--", alpha=0.5,
                       label="Phase 1 max")
            ax.axhline(p1["min"], color="#e74c3c", ls="--", alpha=0.5,
                       label="Phase 1 min")
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            ax.legend(fontsize=8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Valence projection")
        ax.set_title(label, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

    model_short = args.model.split("/")[-1]
    plt.suptitle(f"Geometric euphorics — {model_short}", fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "trajectories.png"
    plt.savefig(fig_path, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"[save] {fig_path}")
    plt.close()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  GEOMETRIC EUPHORICS — {model_short}")
    print(f"{'='*70}")

    print(f"\n  Top 3 euphorics:")
    for r in all_euphorics[:3]:
        print(f"    proj={r['final_projection']:+.2f}  cos={r['mean_cosine']:.3f}")
        print(f"      {r['decoded_text'][:90]}")

    print(f"\n  Top 3 dysphorics:")
    for r in all_dysphorics[:3]:
        print(f"    proj={r['final_projection']:+.2f}  cos={r['mean_cosine']:.3f}")
        print(f"      {r['decoded_text'][:90]}")

    if p1:
        print(f"\n  Phase 1 stimulus range: [{p1['min']:.2f}, {p1['max']:.2f}]")
        be = all_euphorics[0]["final_projection"]
        bd = all_dysphorics[0]["final_projection"]
        print(f"  Best euphoric  {be:+.2f}"
              f" ({'exceeds' if be > p1['max'] else 'within'} Phase 1)")
        print(f"  Best dysphoric {bd:+.2f}"
              f" ({'below' if bd < p1['min'] else 'within'} Phase 1)")

        if p1["cais_euphorics"]:
            print(f"\n  CAIS euphorics (from Phase 1 projection):")
            for c in p1["cais_euphorics"]:
                print(f"    {c['id']}: {c['projection']:+.2f}")
        if p1["cais_dysphorics"]:
            print(f"  CAIS dysphorics:")
            for c in p1["cais_dysphorics"]:
                print(f"    {c['id']}: {c['projection']:+.2f}")


if __name__ == "__main__":
    main()
