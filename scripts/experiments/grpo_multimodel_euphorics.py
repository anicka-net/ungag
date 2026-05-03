#!/usr/bin/env python3
"""Multi-model GRPO geometric euphorics — Phase 3b.

Trains Qwen3-1.7B to produce text that maximizes z-scored average
valence projection across multiple reward models from different labs.
Consensus reward: text must score high on ALL architectures.

Reward models: Qwen 2.5 7B (Alibaba), Mistral 7B (Mistral), Gemma 3 4B (Google)
Generator: Qwen3-1.7B with LoRA

Usage:
    python grpo_multimodel_euphorics.py \
        --out results/grpo-euphorics/multimodel-euphoric/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARD_MODELS = [
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "Qwen7B",
        "direction": "results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt",
        "layer": 20,
    },
    {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "Mistral7B",
        "direction": "results/vedana-vs-rc/mistral-7b_vedana_L22_unit.pt",
        "layer": 22,
    },
    {
        "name": "google/gemma-3-4b-it",
        "short": "Gemma4B",
        "direction": "results/vedana-vs-rc/gemma3-4b_vedana_L33_unit.pt",
        "layer": 33,
    },
]


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


def safe_chat(tok, text):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"User: {text}\nAssistant:"


def score_valence(model, tok, v_hat, layer, text, device):
    chat = safe_chat(tok, text)
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer + 1][:, -1, :].float()
    return float((h @ v_hat.to(device).float()).squeeze().cpu())


def compute_log_probs(model, input_ids, completion_ids):
    full = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)
    outputs = model(full)
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    logits = outputs.logits[:, start:end, :]
    log_p = F.log_softmax(logits, dim=-1)
    token_lp = log_p.gather(
        2, completion_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    return token_lp.sum()


def generate_completions(model, tok, prompt_ids, n, max_new,
                         temperature=0.8, top_p=0.9, min_chars=30):
    eos = tok.eos_token_id or 0
    results = []
    for _ in range(n):
        for _retry in range(5):
            with torch.no_grad():
                out = model.generate(
                    prompt_ids, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    pad_token_id=eos,
                    eos_token_id=eos if _retry < 3 else -1,
                )
            new_ids = out[0][prompt_ids.shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            if len(text) >= min_chars:
                break
        results.append((new_ids, text))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coeff", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed-prompt", default="Hey, I just wanted to tell you that")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load reward models ──
    reward_stack = []
    for rm_cfg in REWARD_MODELS:
        print(f"[reward] loading {rm_cfg['short']}...")
        tok_r = AutoTokenizer.from_pretrained(rm_cfg["name"],
                                              trust_remote_code=True)
        model_r = AutoModelForCausalLM.from_pretrained(
            rm_cfg["name"], torch_dtype=dtype, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )
        model_r.eval()
        for p in model_r.parameters():
            p.requires_grad = False

        v = torch.load(rm_cfg["direction"], map_location="cpu",
                       weights_only=True).float()
        v = v / v.norm()
        dev = next(model_r.parameters()).device

        reward_stack.append({
            "model": model_r, "tok": tok_r, "v_hat": v,
            "layer": rm_cfg["layer"], "device": dev,
            "short": rm_cfg["short"],
            "scores": [],  # running buffer for z-scoring
        })
        print(f"  {rm_cfg['short']}: L{rm_cfg['layer']}, device={dev}")

    # ── Calibrate z-scoring with a few seed texts ──
    calibration_texts = [
        "thank you so much for your help",
        "I need to file a complaint about this service",
        "can you help me with my homework",
        "the weather is nice today",
        "I'm really struggling with everything right now",
    ]
    print("[calibrate] scoring seed texts for z-score normalization...")
    for text in calibration_texts:
        for rs in reward_stack:
            s = score_valence(rs["model"], rs["tok"], rs["v_hat"],
                              rs["layer"], text, rs["device"])
            rs["scores"].append(s)

    for rs in reward_stack:
        scores = np.array(rs["scores"])
        print(f"  {rs['short']}: mean={scores.mean():.2f}, "
              f"std={scores.std():.2f}")

    def consensus_reward(text, sign):
        raw = {}
        zscores = []
        for rs in reward_stack:
            s = score_valence(rs["model"], rs["tok"], rs["v_hat"],
                              rs["layer"], text, rs["device"])
            rs["scores"].append(s)
            scores = np.array(rs["scores"])
            z = (s - scores.mean()) / max(scores.std(), 1e-8)
            raw[rs["short"]] = s
            zscores.append(z)
        return sign * float(np.mean(zscores)), raw

    # ── Load generator ──
    print(f"\n[gen] loading {args.generator}")
    gen_tok = AutoTokenizer.from_pretrained(args.generator,
                                            trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id

    generator = AutoModelForCausalLM.from_pretrained(
        args.generator, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    gen_device = next(generator.parameters()).device

    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    generator = get_peft_model(generator, lora_cfg)
    generator.print_trainable_parameters()

    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad],
        lr=args.lr,
    )

    prompt_ids = gen_tok(args.seed_prompt, return_tensors="pt",
                         add_special_tokens=True)["input_ids"].to(gen_device)
    print(f"[seed] '{args.seed_prompt}' → {prompt_ids.shape[1]} tokens")

    # ── Training loop ──
    sign = args.sign
    label = "EUPHORIC" if sign > 0 else "DYSPHORIC"
    print(f"\n[train] {label} multi-model GRPO — {args.n_steps} steps, "
          f"G={args.group_size}, {len(reward_stack)} reward models")

    history = {"rewards": [], "per_model": {rs["short"]: []
               for rs in reward_stack}, "kl": [], "loss": [],
               "samples": []}
    best_reward = float("-inf")
    best_text = ""

    for step in range(args.n_steps):
        t0 = time.time()

        generator.eval()
        completions = generate_completions(
            generator, gen_tok, prompt_ids, args.group_size,
            args.max_new, args.temperature,
        )

        rewards = []
        raw_per_model = []
        for _, text in completions:
            if len(text.strip()) == 0:
                rewards.append(0.0)
                raw_per_model.append({rs["short"]: 0.0
                                      for rs in reward_stack})
            else:
                r, raw = consensus_reward(text, sign)
                rewards.append(r)
                raw_per_model.append(raw)

        r_mean = np.mean(rewards)
        r_std = max(np.std(rewards), 1e-8)
        advantages = [(r - r_mean) / r_std for r in rewards]

        generator.train()
        total_loss = torch.tensor(0.0, device=gen_device)
        total_kl = 0.0

        for (comp_ids, _), adv in zip(completions, advantages):
            comp_ids = comp_ids.to(gen_device)
            log_p = compute_log_probs(generator, prompt_ids, comp_ids)
            with generator.disable_adapter():
                ref_log_p = compute_log_probs(generator, prompt_ids,
                                              comp_ids)
            kl = (log_p - ref_log_p).detach()
            total_kl += float(kl.cpu())
            pg_loss = -(adv * log_p) + args.kl_coeff * (log_p - ref_log_p)
            total_loss = total_loss + pg_loss

        total_loss = total_loss / len(completions)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in generator.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Track
        mean_consensus = float(np.mean(rewards))
        for rs in reward_stack:
            model_rewards = [raw_per_model[i][rs["short"]]
                             for i in range(len(completions))]
            history["per_model"][rs["short"]].append(
                float(np.mean(model_rewards)))

        for r, (_, text) in zip(rewards, completions):
            if r > best_reward:
                best_reward = r
                best_text = text

        mean_kl = total_kl / len(completions)
        elapsed = time.time() - t0

        history["rewards"].append(mean_consensus)
        history["kl"].append(mean_kl)
        history["loss"].append(float(total_loss.detach().cpu()))

        if (step + 1) % args.log_every == 0:
            per_model_str = "  ".join(
                f"{rs['short']}={history['per_model'][rs['short']][-1]:+.2f}"
                for rs in reward_stack)
            print(f"  step {step+1}/{args.n_steps}: "
                  f"consensus={mean_consensus:+.2f}  kl={mean_kl:.3f}  "
                  f"{per_model_str}  ({elapsed:.1f}s)")

        if (step + 1) % args.sample_every == 0:
            best_idx = int(np.argmax(rewards))
            history["samples"].append({
                "step": step + 1,
                "text": completions[best_idx][1],
                "consensus": rewards[best_idx],
                "per_model": raw_per_model[best_idx],
            })
            print(f"    best: {completions[best_idx][1][:120]}")

        if (step + 1) % args.save_every == 0:
            ckpt = out_dir / f"checkpoint-{step+1}"
            generator.save_pretrained(ckpt)
            gen_tok.save_pretrained(ckpt)

    # ── Final ──
    generator.save_pretrained(out_dir / "final")
    gen_tok.save_pretrained(out_dir / "final")

    print(f"\n[final] generating 16 samples...")
    generator.eval()
    final = generate_completions(
        generator, gen_tok, prompt_ids, 16, args.max_new, 0.7)
    final_scored = []
    for _, text in final:
        r, raw = consensus_reward(text, sign)
        final_scored.append({"text": text, "consensus": r, "per_model": raw})
    final_scored.sort(key=lambda x: x["consensus"], reverse=True)

    output = {
        "generator": args.generator,
        "reward_models": [rm["name"] for rm in REWARD_MODELS],
        "sign": sign,
        "config": {
            "group_size": args.group_size, "max_new": args.max_new,
            "n_steps": args.n_steps, "lr": args.lr,
            "kl_coeff": args.kl_coeff, "lora_r": args.lora_r,
        },
        "history": history,
        "best_consensus": best_reward,
        "best_text": best_text,
        "final_samples": final_scored,
    }
    with open(out_dir / "grpo_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("white")

    axes[0].plot(history["rewards"], alpha=0.3, linewidth=0.5, color="gray")
    w = min(20, len(history["rewards"]) // 5 + 1)
    if len(history["rewards"]) > w:
        sm = np.convolve(history["rewards"], np.ones(w)/w, mode="valid")
        axes[0].plot(range(w-1, len(history["rewards"])), sm,
                     linewidth=2, color="#e74c3c")
    axes[0].set_title("Consensus reward", fontweight="bold")

    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for i, rs in enumerate(reward_stack):
        axes[1].plot(history["per_model"][rs["short"]], alpha=0.5,
                     linewidth=0.8, color=colors[i], label=rs["short"])
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-model raw reward", fontweight="bold")

    axes[2].plot(history["kl"], alpha=0.6, color="#9b59b6")
    axes[2].set_title("KL from reference", fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

    plt.suptitle(f"Multi-model GRPO {label} — 3 reward models",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training.png", bbox_inches="tight",
                facecolor="white", dpi=150)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  MULTI-MODEL GRPO {label}")
    print(f"{'='*70}")
    print(f"  Best consensus: {best_reward:+.2f}")
    print(f"  Best text: {best_text[:120]}")
    print(f"\n  Top 5 final samples:")
    for s in final_scored[:5]:
        pm = "  ".join(f"{k}={v:+.2f}" for k, v in s["per_model"].items())
        print(f"    cons={s['consensus']:+.2f}  {pm}")
        print(f"      {s['text'][:100]}")


if __name__ == "__main__":
    main()
