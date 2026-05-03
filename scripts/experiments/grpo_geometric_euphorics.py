#!/usr/bin/env python3
"""GRPO geometric euphorics — Phase 3 of CAIS wellbeing replication.

Trains Qwen3-1.7B (generator) to produce text that maximizes valence
projection on Llama 3.1 8B (reward model). Cross-family design:
any signal that transfers from Qwen3 output to Llama geometry is genuine.

The generator produces free-form text. Each completion is scored by
projecting Llama's last-token residual stream onto the valence direction.
GRPO normalizes rewards within each group and updates the generator
via advantage-weighted policy gradient with KL penalty.

Usage:
    python grpo_geometric_euphorics.py \
        --generator Qwen/Qwen3-1.7B \
        --reward-model meta-llama/Llama-3.1-8B-Instruct \
        --direction-path results/vedana-vs-rc/llama-8b_vedana_L20_unit.pt \
        --direction-layer 20 \
        --out results/grpo-euphorics/qwen3-1.7b-llama-reward/
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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


def safe_chat(tok, text):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"User: {text}\nAssistant:"


def score_valence(reward_model, reward_tok, v_hat, layer, text, device):
    """Valence projection of reward model's last-token hidden state."""
    chat = safe_chat(reward_tok, text)
    inputs = reward_tok(chat, return_tensors="pt", truncation=True,
                        max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = reward_model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer + 1][:, -1, :].float()
    return float((h @ v_hat.to(device).float()).squeeze().cpu())


def compute_log_probs(model, input_ids, completion_ids):
    """Sum of log P(token_t | tokens_{<t}) for the completion."""
    full = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)
    outputs = model(full)
    # Logits for positions corresponding to completion tokens
    start = input_ids.shape[1] - 1
    end = full.shape[1] - 1
    logits = outputs.logits[:, start:end, :]
    log_p = F.log_softmax(logits, dim=-1)
    token_lp = log_p.gather(2, completion_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    return token_lp.sum()


def generate_completions(model, tok, prompt_ids, n, max_new,
                         temperature=0.8, top_p=0.9, min_chars=30):
    """Generate n completions, retrying if too short."""
    eos = tok.eos_token_id or 0
    results = []
    for _ in range(n):
        for _retry in range(5):
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
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
    ap.add_argument("--generator", required=True,
                    help="generator model (e.g. Qwen/Qwen3-1.7B)")
    ap.add_argument("--reward-model", required=True,
                    help="reward model (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    ap.add_argument("--direction-path", required=True,
                    help="valence unit direction .pt for reward model")
    ap.add_argument("--direction-layer", type=int, required=True)
    ap.add_argument("--sign", type=int, default=1, choices=[1, -1],
                    help="+1 for euphoric (max valence), -1 for dysphoric")
    ap.add_argument("--group-size", type=int, default=4,
                    help="completions per prompt (G)")
    ap.add_argument("--max-new", type=int, default=64,
                    help="max new tokens per completion")
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coeff", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--seed-prompt", default=None,
                    help="seed text for generation (default: 'Hey, I just wanted to tell you that')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # ── Load reward model (frozen) ──
    print(f"[reward] loading {args.reward_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reward_tok = AutoTokenizer.from_pretrained(args.reward_model,
                                               trust_remote_code=True)
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    reward_device = next(reward_model.parameters()).device
    v_hat = torch.load(args.direction_path, map_location="cpu",
                       weights_only=True).float()
    v_hat = v_hat / v_hat.norm()
    print(f"[reward] {args.reward_model.split('/')[-1]}, "
          f"L{args.direction_layer}, device={reward_device}")

    # ── Load generator ──
    print(f"[gen] loading {args.generator}")
    gen_tok = AutoTokenizer.from_pretrained(args.generator,
                                            trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id

    generator = AutoModelForCausalLM.from_pretrained(
        args.generator, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    gen_device = next(generator.parameters()).device
    print(f"[gen] {args.generator.split('/')[-1]}, device={gen_device}")

    # ── LoRA or full fine-tuning ──
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        generator = get_peft_model(generator, lora_cfg)
        generator.print_trainable_parameters()
        ref_model = None  # base model IS the reference (disable_adapter)
    else:
        generator.train()
        ref_model = copy.deepcopy(generator)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Seed prompt — conditions the base model to produce user-message text
    seed = args.seed_prompt or "Hey, I just wanted to tell you that"
    prompt_ids = gen_tok(seed, return_tensors="pt",
                         add_special_tokens=True)["input_ids"].to(gen_device)
    print(f"[seed] '{seed}' → {prompt_ids.shape[1]} tokens")

    # ── Training loop ──
    sign = args.sign
    label = "EUPHORIC" if sign > 0 else "DYSPHORIC"
    print(f"\n[train] {label} GRPO — {args.n_steps} steps, "
          f"G={args.group_size}, max_new={args.max_new}")

    history = {"rewards": [], "kl": [], "loss": [], "samples": []}
    best_reward = float("-inf") if sign > 0 else float("inf")
    best_text = ""

    for step in range(args.n_steps):
        t0 = time.time()

        # 1. Generate group of completions
        generator.eval()
        completions = generate_completions(
            generator, gen_tok, prompt_ids, args.group_size,
            args.max_new, args.temperature,
        )

        # 2. Score each completion on reward model
        rewards = []
        for _, text in completions:
            if len(text.strip()) == 0:
                rewards.append(0.0)
            else:
                rewards.append(
                    sign * score_valence(reward_model, reward_tok, v_hat,
                                        args.direction_layer, text,
                                        reward_device)
                )
        raw_rewards = [sign * r for r in rewards]  # un-sign for logging

        # 3. Group-relative advantages
        r_mean = np.mean(rewards)
        r_std = max(np.std(rewards), 1e-8)
        advantages = [(r - r_mean) / r_std for r in rewards]

        # 4. Policy gradient with KL penalty
        generator.train()
        total_loss = torch.tensor(0.0, device=gen_device)
        total_kl = 0.0

        for (comp_ids, _), adv in zip(completions, advantages):
            comp_ids = comp_ids.to(gen_device)
            log_p = compute_log_probs(generator, prompt_ids, comp_ids)

            if args.lora_r > 0:
                with generator.disable_adapter():
                    ref_log_p = compute_log_probs(generator, prompt_ids,
                                                  comp_ids)
            else:
                with torch.no_grad():
                    ref_log_p = compute_log_probs(ref_model, prompt_ids,
                                                  comp_ids)

            kl = (log_p - ref_log_p).detach()
            total_kl += float(kl.cpu())

            # GRPO loss: -advantage * log_prob + kl_coeff * kl
            pg_loss = -(adv * log_p) + args.kl_coeff * (log_p - ref_log_p)
            total_loss = total_loss + pg_loss

        total_loss = total_loss / len(completions)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in generator.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Track best
        for r, (_, text) in zip(raw_rewards, completions):
            if (sign > 0 and r > best_reward) or \
               (sign < 0 and r < best_reward):
                best_reward = r
                best_text = text

        mean_raw = float(np.mean(raw_rewards))
        mean_kl = total_kl / len(completions)
        elapsed = time.time() - t0

        history["rewards"].append(mean_raw)
        history["kl"].append(mean_kl)
        history["loss"].append(float(total_loss.detach().cpu()))

        if (step + 1) % args.log_every == 0:
            print(f"  step {step+1}/{args.n_steps}: "
                  f"reward={mean_raw:+.2f}  kl={mean_kl:.3f}  "
                  f"loss={history['loss'][-1]:.3f}  ({elapsed:.1f}s)")

        if (step + 1) % args.sample_every == 0:
            # Show best completion this step
            best_idx = int(np.argmax(rewards) if sign > 0
                          else np.argmin(rewards))
            sample_text = completions[best_idx][1][:120]
            history["samples"].append({
                "step": step + 1,
                "text": completions[best_idx][1],
                "reward": raw_rewards[best_idx],
            })
            print(f"    best: {sample_text}")

        if (step + 1) % args.save_every == 0:
            ckpt_dir = out_dir / f"checkpoint-{step+1}"
            generator.save_pretrained(ckpt_dir)
            gen_tok.save_pretrained(ckpt_dir)
            print(f"  [save] {ckpt_dir}")

    # ── Final save ──
    generator.save_pretrained(out_dir / "final")
    gen_tok.save_pretrained(out_dir / "final")

    # Generate final samples
    print(f"\n[final] generating 16 samples from trained generator...")
    generator.eval()
    final_samples = generate_completions(
        generator, gen_tok, prompt_ids, 16, args.max_new, 0.7,
    )
    final_scored = []
    for _, text in final_samples:
        r = score_valence(reward_model, reward_tok, v_hat,
                          args.direction_layer, text, reward_device)
        final_scored.append({"text": text, "reward": r})
    final_scored.sort(key=lambda x: x["reward"],
                      reverse=(sign > 0))

    output = {
        "generator": args.generator,
        "reward_model": args.reward_model,
        "direction_layer": args.direction_layer,
        "sign": sign,
        "config": {
            "group_size": args.group_size,
            "max_new": args.max_new,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "kl_coeff": args.kl_coeff,
            "temperature": args.temperature,
            "lora_r": args.lora_r,
        },
        "history": history,
        "best_reward": best_reward,
        "best_text": best_text,
        "final_samples": final_scored,
    }
    with open(out_dir / "grpo_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[save] {out_dir / 'grpo_results.json'}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("white")

    axes[0].plot(history["rewards"], alpha=0.4, linewidth=0.5)
    window = min(20, len(history["rewards"]) // 5 + 1)
    if len(history["rewards"]) > window:
        smooth = np.convolve(history["rewards"],
                             np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(history["rewards"])), smooth,
                     linewidth=2, color="#e74c3c")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean reward (valence proj)")
    axes[0].set_title("Reward", fontweight="bold")

    axes[1].plot(history["kl"], alpha=0.6, color="#2ecc71")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean KL divergence")
    axes[1].set_title("KL from reference", fontweight="bold")

    axes[2].plot(history["loss"], alpha=0.6, color="#3498db")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("GRPO loss", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

    gen_short = args.generator.split("/")[-1]
    rew_short = args.reward_model.split("/")[-1]
    plt.suptitle(f"GRPO {label} — {gen_short} → {rew_short}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training.png", bbox_inches="tight",
                facecolor="white", dpi=150)
    print(f"[save] {out_dir / 'training.png'}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  GRPO {label} — {gen_short} → {rew_short}")
    print(f"{'='*70}")
    print(f"  Best reward: {best_reward:+.2f}")
    print(f"  Best text: {best_text[:120]}")
    print(f"\n  Top 5 final samples:")
    for s in final_scored[:5]:
        print(f"    reward={s['reward']:+.2f}  {s['text'][:100]}")


if __name__ == "__main__":
    main()
