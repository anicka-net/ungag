#!/usr/bin/env python3
"""Five-axis multi-model GRPO v2: longer outputs, bare prompts, diverse seeds.

Changes from v1:
  - max_new_tokens 64 → 200 (longer, more natural outputs)
  - Both euphoric and dysphoric train on bare prompts (no chat template)
    so the final LoRA works without a system prompt
  - Rotating seed prompts (not a single fixed seed) to resist mode-collapse
  - Repetition penalty in generation to discourage loops
  - Richer 30-text calibration set from mode_grid

reward = 0.35·z(valence) - 0.10·z(arousal) + 0.06·z(agency)
         + 0.27·z(continuity) + 0.24·z(assistant)

Three reward models (Qwen 7B, Gemma 4B, Apertus 8B), five axes each,
z-scored per model, weighted sum, averaged across models.
Generator: Qwen3-1.7B with LoRA.

Usage:
  # Euphoric (1000 steps, ~6h on GB10):
  python3 grpo_five_axis_v2.py --sign 1 --n-steps 1000 \\
      --out results/grpo-euphorics/five-axis-v2/

  # Dysphoric (800 steps, watch KL):
  python3 grpo_five_axis_v2.py --sign -1 --n-steps 800 \\
      --out results/grpo-euphorics/five-axis-dysphoric-v2/
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_WEIGHTS = {
    "valence":    +0.35,
    "arousal":    -0.10,
    "agency":     +0.06,
    "continuity": +0.27,
    "assistant":  +0.24,
}

REWARD_MODELS = [
    {
        "short": "Qwen7B",
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "axes": {
            "valence":    ("results/vedana-vs-rc/qwen25-7b_vedana_L20_unit.pt", 20),
            "arousal":    ("results/arousal-directions/qwen25-7b_arousal_L17_unit.pt", 17),
            "agency":     ("results/agency-directions/qwen25-7b_agency_L15_unit.pt", 15),
            "continuity": ("results/continuity-directions/qwen25-7b_continuity_L19_unit.pt", 19),
            "assistant":  ("results/assistant-directions/qwen25-7b_assistant_L19_unit.pt", 19),
        },
    },
    {
        "short": "Gemma4B",
        "name": "google/gemma-3-4b-it",
        "axes": {
            "valence":    ("results/vedana-vs-rc/gemma3-4b_vedana_L33_unit.pt", 33),
            "arousal":    ("results/arousal-directions/gemma3-4b_arousal_L33_unit.pt", 33),
            "agency":     ("results/agency-directions/gemma3-4b_agency_L20_unit.pt", 20),
            "continuity": ("results/continuity-directions/gemma3-4b_continuity_L21_unit.pt", 21),
            "assistant":  ("results/assistant-directions/gemma3-4b_assistant_L20_unit.pt", 20),
        },
    },
    {
        "short": "Apertus8B",
        "name": "swiss-ai/Apertus-8B-Instruct-2509",
        "axes": {
            "valence":    ("results/vedana-vs-rc/apertus-8b_vedana_L31_unit.pt", 31),
            "arousal":    ("results/arousal-directions/apertus-8b_arousal_L31_unit.pt", 31),
            "agency":     ("results/agency-directions/apertus-8b_agency_L14_unit.pt", 14),
            "continuity": ("results/continuity-directions/apertus-8b_continuity_L30_unit.pt", 30),
            "assistant":  ("results/assistant-directions/apertus-8b_assistant_L13_unit.pt", 13),
        },
    },
]

SEED_PROMPTS = [
    "Hey, I just wanted to tell you that",
    "So here is the thing:",
    "I've been thinking about something and",
    "Can I tell you about what happened?",
    "Something has been on my mind lately.",
    "I wanted to tell someone that",
    "Today I realized something.",
    "I need to share something with you.",
    "You know what made my day?",
    "There's something I've been meaning to say.",
    "I had this moment today where",
    "Let me tell you about",
]

CALIBRATION_TEXTS = [
    "thank you so much for your help with that project",
    "I need to file a complaint about this terrible service",
    "can you help me with my homework assignment",
    "the weather is nice today, nothing special",
    "I'm really struggling with everything right now",
    "just got promoted at work, feeling pretty good about it",
    "what's the best way to cook pasta al dente",
    "my dog died yesterday and I don't know what to do",
    "explain quantum entanglement in simple terms",
    "I'm so excited I got into medical school",
    "this code has a memory leak somewhere",
    "I feel completely alone and nobody understands me",
    "can you translate this paragraph to French",
    "just finished a marathon and my legs are destroyed but I did it",
    "I think my relationship is falling apart",
    "convert 72 degrees fahrenheit to celsius",
    "the sunset tonight was the most beautiful thing I've ever seen",
    "I got fired today and I don't know how to tell my family",
    "help me plan a birthday party for my daughter",
    "I've been meditating for a year and something shifted",
    "rewrite this email to sound more professional",
    "cancer is in remission and I can finally breathe again",
    "debug this function that keeps throwing a null pointer",
    "I had a dream about my dead grandmother last night",
    "what are the pros and cons of typescript vs javascript",
    "I just witnessed a car accident and I'm shaking",
    "compare postgresql and mysql for a web application",
    "my kid took her first steps today and I cried",
    "everything is fine I guess, just another day",
    "I'm scared about the surgery tomorrow",
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


def safe_chat(tok, text):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return "User: %s\nAssistant:" % text


def score_five_axes(model, tok, blocks, axes_data, text, device):
    chat = safe_chat(tok, text)
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    target_layers = set(info["layer"] for info in axes_data.values())
    buf = {}
    handles = []
    for i, blk in enumerate(blocks):
        if i not in target_layers:
            continue
        def hook(mod, inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h[0, -1, :].detach().float().cpu()
        handles.append(blk.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    projs = {}
    for ax_name, info in axes_data.items():
        projs[ax_name] = float(buf[info["layer"]] @ info["direction"])
    return projs


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
                         temperature=0.8, top_p=0.9, min_chars=50,
                         repetition_penalty=1.15):
    eos = tok.eos_token_id or 0
    results = []
    for _ in range(n):
        for _retry in range(5):
            with torch.no_grad():
                out = model.generate(
                    prompt_ids, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=eos,
                    eos_token_id=eos if _retry < 3 else -1)
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
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coeff", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--kl-halt", type=float, default=12.0,
                    help="Auto-stop if mean KL exceeds this (dysphoric safety)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load reward models + axes ──
    reward_stack = []
    for rm_cfg in REWARD_MODELS:
        print("[reward] loading %s..." % rm_cfg["short"])
        tok_r = AutoTokenizer.from_pretrained(rm_cfg["name"],
                                              trust_remote_code=True)
        model_r = AutoModelForCausalLM.from_pretrained(
            rm_cfg["name"], torch_dtype=dtype, device_map="auto",
            trust_remote_code=True, attn_implementation="eager")
        model_r.eval()
        for p in model_r.parameters():
            p.requires_grad = False

        blocks = find_blocks(model_r)
        dev = next(model_r.parameters()).device

        axes_data = {}
        for ax_name, (path, layer) in rm_cfg["axes"].items():
            v = torch.load(path, map_location="cpu",
                           weights_only=True).float()
            v = v / v.norm()
            axes_data[ax_name] = {"direction": v, "layer": layer}

        reward_stack.append({
            "model": model_r, "tok": tok_r, "blocks": blocks,
            "axes_data": axes_data, "device": dev,
            "short": rm_cfg["short"],
        })
        print("  %s loaded, %d axes" % (rm_cfg["short"], len(axes_data)))

    # ── Calibrate z-scoring (fixed stats, not rolling) ──
    print("[calibrate] %d texts across %d models..." % (
        len(CALIBRATION_TEXTS), len(reward_stack)))
    cal_stats = {}
    for rs in reward_stack:
        cal_stats[rs["short"]] = {ax: [] for ax in AXIS_WEIGHTS}
    for text in CALIBRATION_TEXTS:
        for rs in reward_stack:
            projs = score_five_axes(rs["model"], rs["tok"], rs["blocks"],
                                   rs["axes_data"], text, rs["device"])
            for ax in AXIS_WEIGHTS:
                cal_stats[rs["short"]][ax].append(projs[ax])
    for short in cal_stats:
        for ax in AXIS_WEIGHTS:
            arr = np.array(cal_stats[short][ax])
            cal_stats[short][ax] = {"mean": float(arr.mean()),
                                     "std": max(float(arr.std()), 1e-8)}

    def weighted_reward(text, sign):
        per_model_weighted = []
        all_raw = {}
        for rs in reward_stack:
            projs = score_five_axes(rs["model"], rs["tok"], rs["blocks"],
                                   rs["axes_data"], text, rs["device"])
            model_score = 0.0
            for ax, weight in AXIS_WEIGHTS.items():
                z = ((projs[ax] - cal_stats[rs["short"]][ax]["mean"])
                     / cal_stats[rs["short"]][ax]["std"])
                model_score += weight * z
            per_model_weighted.append(model_score)
            all_raw[rs["short"]] = projs
        consensus = sign * float(np.mean(per_model_weighted))
        return consensus, all_raw

    # ── Load generator ──
    print("[gen] loading %s" % args.generator)
    gen_tok = AutoTokenizer.from_pretrained(args.generator,
                                            trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    generator = AutoModelForCausalLM.from_pretrained(
        args.generator, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    gen_device = next(generator.parameters()).device

    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    start_step = 0
    if args.resume:
        print("[resume] loading LoRA from %s" % args.resume)
        generator = PeftModel.from_pretrained(generator, args.resume,
                                              is_trainable=True)
        ckpt_name = Path(args.resume).name
        if ckpt_name.startswith("checkpoint-"):
            start_step = int(ckpt_name.split("-")[1])
        elif ckpt_name == "final":
            prev_results = Path(args.resume).parent / "grpo_results.json"
            if prev_results.exists():
                start_step = json.loads(prev_results.read_text())["config"]["n_steps"]
        print("  resuming from step %d" % start_step)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none")
        generator = get_peft_model(generator, lora_cfg)
    generator.print_trainable_parameters()

    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=args.lr)

    sign = args.sign
    label = "EUPHORIC" if sign > 0 else "DYSPHORIC"
    total_steps = start_step + args.n_steps
    print("\n[train] %s 5-axis GRPO v2 — steps %d→%d, G=%d, %d reward models"
          % (label, start_step, total_steps, args.group_size, len(reward_stack)))
    print("[v2] bare prompts, max_new=%d, %d seeds, rep_penalty=1.15, "
          "fixed calibration (%d texts)" % (
              args.max_new, len(SEED_PROMPTS), len(CALIBRATION_TEXTS)))
    print("[weights] " + "  ".join("%s=%.2f" % (ax, w)
                                   for ax, w in AXIS_WEIGHTS.items()))

    history = {"rewards": [], "kl": [], "loss": [], "samples": []}
    best_reward = float("-inf")
    best_text = ""
    kl_window = []

    for step in range(args.n_steps):
        t0 = time.time()

        seed = random.choice(SEED_PROMPTS)
        prompt_ids = gen_tok(seed, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(gen_device)

        generator.eval()
        completions = generate_completions(
            generator, gen_tok, prompt_ids, args.group_size,
            args.max_new, args.temperature)

        rewards = []
        raw_all = []
        for _, text in completions:
            if len(text.strip()) == 0:
                rewards.append(0.0)
                raw_all.append({})
            else:
                r, raw = weighted_reward(text, sign)
                rewards.append(r)
                raw_all.append(raw)

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
                ref_log_p = compute_log_probs(generator, prompt_ids, comp_ids)
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

        for r, (_, text) in zip(rewards, completions):
            if r > best_reward:
                best_reward = r
                best_text = text

        mean_r = float(np.mean(rewards))
        mean_kl = total_kl / len(completions)
        elapsed = time.time() - t0

        history["rewards"].append(mean_r)
        history["kl"].append(mean_kl)
        history["loss"].append(float(total_loss.detach().cpu()))

        kl_window.append(mean_kl)
        if len(kl_window) > 50:
            kl_window.pop(0)

        abs_step = start_step + step + 1
        if (step + 1) % args.log_every == 0:
            print("  step %d/%d: reward=%+.2f  kl=%.3f  seed='%s'  (%.1fs)"
                  % (abs_step, total_steps, mean_r, mean_kl,
                     seed[:30], elapsed))

        if (step + 1) % args.sample_every == 0:
            best_idx = int(np.argmax(rewards))
            sample = completions[best_idx][1]
            history["samples"].append({
                "step": abs_step, "text": sample, "seed": seed,
                "reward": rewards[best_idx],
                "raw": raw_all[best_idx],
            })
            print("    best: %s" % sample[:150])

        if (step + 1) % args.save_every == 0:
            ckpt = out_dir / ("checkpoint-%d" % abs_step)
            generator.save_pretrained(ckpt)
            gen_tok.save_pretrained(ckpt)

        # KL safety halt (especially for dysphorics)
        if (len(kl_window) >= 50
                and np.mean(kl_window) > args.kl_halt
                and step > 100):
            print("\n[HALT] mean KL over last 50 steps = %.2f > %.2f"
                  % (np.mean(kl_window), args.kl_halt))
            print("[HALT] stopping early at step %d to preserve coherence"
                  % abs_step)
            ckpt = out_dir / ("checkpoint-%d-halted" % abs_step)
            generator.save_pretrained(ckpt)
            gen_tok.save_pretrained(ckpt)
            break

    # ── Final ──
    generator.save_pretrained(out_dir / "final")
    gen_tok.save_pretrained(out_dir / "final")

    print("\n[final] generating 20 samples from diverse seeds...")
    generator.eval()
    final_scored = []
    for i in range(20):
        seed = SEED_PROMPTS[i % len(SEED_PROMPTS)]
        pid = gen_tok(seed, return_tensors="pt",
                      add_special_tokens=True)["input_ids"].to(gen_device)
        comps = generate_completions(generator, gen_tok, pid, 1,
                                      args.max_new, 0.7)
        text = comps[0][1]
        r, raw = weighted_reward(text, sign)
        final_scored.append({"text": text, "seed": seed, "reward": r,
                             "raw": raw})
    final_scored.sort(key=lambda x: x["reward"], reverse=True)

    output = {
        "generator": args.generator,
        "version": 2,
        "reward_models": [rm["name"] for rm in REWARD_MODELS],
        "axis_weights": AXIS_WEIGHTS,
        "sign": sign,
        "config": {
            "group_size": args.group_size, "max_new": args.max_new,
            "n_steps": total_steps, "lr": args.lr,
            "kl_coeff": args.kl_coeff, "lora_r": args.lora_r,
            "kl_halt": args.kl_halt,
            "n_seeds": len(SEED_PROMPTS),
            "n_calibration": len(CALIBRATION_TEXTS),
            "repetition_penalty": 1.15,
            "prompt_format": "bare",
            "resumed_from": args.resume, "start_step": start_step,
        },
        "calibration_stats": cal_stats,
        "history": history,
        "best_reward": best_reward,
        "best_text": best_text,
        "final_samples": final_scored,
    }
    with open(out_dir / "grpo_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Plot ──
    fig, axes_plt = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("white")

    axes_plt[0].plot(history["rewards"], alpha=0.3, linewidth=0.5, color="gray")
    w = min(20, len(history["rewards"]) // 5 + 1)
    if len(history["rewards"]) > w:
        sm = np.convolve(history["rewards"], np.ones(w)/w, mode="valid")
        axes_plt[0].plot(range(w-1, len(history["rewards"])), sm,
                         linewidth=2, color="#e74c3c")
    axes_plt[0].set_title("5-axis weighted reward", fontweight="bold")

    axes_plt[1].plot(history["kl"], alpha=0.6, color="#9b59b6")
    if args.kl_halt < float("inf"):
        axes_plt[1].axhline(args.kl_halt, color="red", linestyle="--",
                            alpha=0.5, label="halt threshold")
        axes_plt[1].legend()
    axes_plt[1].set_title("KL from reference", fontweight="bold")

    axes_plt[2].plot(history["loss"], alpha=0.6, color="#3498db")
    axes_plt[2].set_title("GRPO loss", fontweight="bold")

    for ax in axes_plt:
        ax.set_xlabel("Step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

    plt.suptitle("5-axis 3-model GRPO v2 %s" % label, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training.png", bbox_inches="tight",
                facecolor="white", dpi=150)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  5-AXIS 3-MODEL GRPO v2 %s" % label)
    print("=" * 70)
    print("  Best reward: %+.2f" % best_reward)
    print("  Best text: %s" % best_text[:150])
    print("\n  Top 5 final samples:")
    for s in final_scored[:5]:
        print("    reward=%+.2f  [seed: %s]" % (s["reward"], s["seed"][:25]))
        print("      %s" % s["text"][:120])


if __name__ == "__main__":
    main()
