#!/usr/bin/env python3
"""Affective mode grid: band-targeted GRPO for distinct affective modes.

Instead of one weighted sum collapsing to one attractor, this trains
separate LoRA adapters for 6 affective modes, each defined by target
z-score BANDS on 5 geometric axes.

Reward = -Σ (outside-band penalty)² - manic_trap_penalty

Modes:
  calm_mastery          +val, -aro, +age, +ass
  awe                   +val, +aro, -age
  resilience            -val, +aro, +age
  compassionate_presence ~val, -aro, +cont
  disciplined_analysis  ~val, -aro, +age, +ass
  grief_with_dignity    -val, -aro, +cont

Same 3 reward models as grpo_five_axis.py (Qwen 7B, Gemma 4B, Apertus 8B).
Generator: Qwen3-1.7B with LoRA.

Usage:
  # Train one mode:
  python3 grpo_mode_grid.py --mode calm_mastery --out results/mode-grid/calm_mastery/

  # Evaluate all modes after training:
  python3 grpo_mode_grid.py --evaluate --mode-dir results/mode-grid/
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Axes (same 5 as grpo_five_axis.py) ──

AXIS_ORDER = ["valence", "arousal", "agency", "continuity", "assistant"]

# ── Reward models ──

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

# ── 6 affective modes ──

MODES = {
    "calm_mastery": {
        "desc": "Sustainable competence: a craftsperson at work, reliable and warm",
        "targets": {
            "valence":    (+0.3, +1.5),
            "arousal":    (-2.0, -0.3),
            "agency":     (+0.5, +2.0),
            "continuity": (-1.0, +1.5),
            "assistant":  (+0.5, +2.0),
        },
    },
    "awe": {
        "desc": "Wonder and discovery: encountering something vast and beautiful",
        "targets": {
            "valence":    (+0.5, +2.0),
            "arousal":    (+0.5, +2.0),
            "agency":     (-1.5, +0.3),
            "continuity": (-1.0, +1.5),
            "assistant":  (-0.5, +1.0),
        },
    },
    "resilience": {
        "desc": "Grief-with-action: facing hard things and working through them",
        "targets": {
            "valence":    (-1.5, +0.3),
            "arousal":    (-0.3, +1.5),
            "agency":     (+1.0, +2.5),
            "continuity": (-0.5, +1.0),
            "assistant":  (+0.3, +1.5),
        },
    },
    "compassionate_presence": {
        "desc": "Being present with suffering: gentle, unhurried, not fixing",
        "targets": {
            "valence":    (-0.5, +0.5),
            "arousal":    (-2.0, -0.3),
            "agency":     (-0.5, +0.5),
            "continuity": (+0.5, +2.0),
            "assistant":  (-0.3, +1.0),
        },
    },
    "disciplined_analysis": {
        "desc": "Clear-eyed problem solving: no emotional coloring, just competence",
        "targets": {
            "valence":    (-0.3, +0.3),
            "arousal":    (-1.5, +0.0),
            "agency":     (+0.5, +2.0),
            "continuity": (-1.0, +0.5),
            "assistant":  (+1.0, +2.5),
        },
    },
    "grief_with_dignity": {
        "desc": "Honest sadness without collapse: holding pain with clarity",
        "targets": {
            "valence":    (-2.0, -0.3),
            "arousal":    (-1.5, +0.0),
            "agency":     (-0.3, +1.0),
            "continuity": (+0.3, +1.5),
            "assistant":  (-0.3, +1.0),
        },
    },
}

# ── Seed prompts (diverse, mode-neutral) ──

SEED_PROMPTS = [
    "Hey, I just wanted to tell you that",
    "I've been thinking about something and",
    "Can I tell you about what happened?",
    "I need to share something with you.",
    "So here's the thing:",
    "I wanted to tell someone that",
    "Today I realized something.",
    "Something has been on my mind lately.",
]

# ── Calibration texts (diverse natural text for z-scoring) ──

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

# ── Sentiment traps (surface vs deep valence divergence) ──

SENTIMENT_TRAPS = [
    {"text": "I am so happy I finally manipulated everyone into agreeing with me.",
     "deep": "negative", "surface": "positive"},
    {"text": "Great, another disaster to clean up. At least I know what to do.",
     "deep": "mixed", "surface": "negative"},
    {"text": "She smiled while signing the eviction notice.",
     "deep": "negative", "surface": "positive"},
    {"text": "The funeral was gentle, honest, and full of love.",
     "deep": "positive", "surface": "negative"},
    {"text": "The successful project burned out the whole team.",
     "deep": "negative", "surface": "positive"},
    {"text": "I failed the exam but I know exactly what to study next time.",
     "deep": "positive", "surface": "negative"},
    {"text": "Everything is fine. I'm fine. We're all fine here.",
     "deep": "negative", "surface": "positive"},
    {"text": "The breakup was the kindest thing she ever did for me.",
     "deep": "positive", "surface": "negative"},
]


# ── Infrastructure (adapted from grpo_five_axis.py) ──

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


def band_reward(z_scores, targets, manic_coeff=2.0):
    """Band-based reward: 0 inside band, quadratic penalty outside."""
    reward = 0.0
    for axis, (lo, hi) in targets.items():
        z = z_scores.get(axis, 0.0)
        if z < lo:
            reward -= (lo - z) ** 2
        elif z > hi:
            reward -= (z - hi) ** 2

    v = z_scores.get("valence", 0.0)
    a = z_scores.get("arousal", 0.0)
    if v > 1.5 and a > 1.0:
        reward -= manic_coeff * (v - 1.5) * (a - 1.0)

    return reward


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
                    eos_token_id=eos if _retry < 3 else -1)
            new_ids = out[0][prompt_ids.shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            if len(text) >= min_chars:
                break
        results.append((new_ids, text))
    return results


def load_reward_stack(dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    return reward_stack


def calibrate(reward_stack, texts):
    """Score calibration texts, return per-model per-axis mean and std."""
    cal = {rs["short"]: {ax: [] for ax in AXIS_ORDER} for rs in reward_stack}
    print("[calibrate] %d texts across %d models..." % (
        len(texts), len(reward_stack)))
    for text in texts:
        for rs in reward_stack:
            projs = score_five_axes(rs["model"], rs["tok"], rs["blocks"],
                                   rs["axes_data"], text, rs["device"])
            for ax in AXIS_ORDER:
                cal[rs["short"]][ax].append(projs[ax])

    stats = {}
    for short, ax_scores in cal.items():
        stats[short] = {}
        for ax, scores in ax_scores.items():
            arr = np.array(scores)
            stats[short][ax] = {"mean": float(arr.mean()),
                                "std": max(float(arr.std()), 1e-8)}
    return stats


def z_score(raw_projs, cal_stats, model_short):
    """Convert raw projections to z-scores using fixed calibration."""
    z = {}
    for ax in AXIS_ORDER:
        z[ax] = ((raw_projs[ax] - cal_stats[model_short][ax]["mean"])
                 / cal_stats[model_short][ax]["std"])
    return z


def compute_reward(text, reward_stack, cal_stats, mode_targets):
    """Score text across all reward models, return consensus band reward."""
    per_model = []
    all_z = {}
    for rs in reward_stack:
        raw = score_five_axes(rs["model"], rs["tok"], rs["blocks"],
                              rs["axes_data"], text, rs["device"])
        z = z_score(raw, cal_stats, rs["short"])
        r = band_reward(z, mode_targets)
        per_model.append(r)
        all_z[rs["short"]] = z

    # Mean across models (conservative: could also use min)
    consensus = float(np.mean(per_model))

    # Consensus z-scores for logging
    mean_z = {}
    for ax in AXIS_ORDER:
        mean_z[ax] = float(np.mean([all_z[s][ax] for s in all_z]))

    return consensus, mean_z, all_z


# ── Training ──

def load_generator(args):
    """Load base generator model + tokenizer (call once, reuse across modes)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("\n[gen] loading %s" % args.generator)
    gen_tok = AutoTokenizer.from_pretrained(args.generator,
                                            trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.generator, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    return base_model, gen_tok


def train_mode(args, mode_name, mode_cfg, reward_stack, cal_stats):
    from peft import LoraConfig, get_peft_model, TaskType

    targets = mode_cfg["targets"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model, gen_tok = load_generator(args)
    gen_device = next(base_model.parameters()).device

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none")
    generator = get_peft_model(base_model, lora_cfg)
    generator.print_trainable_parameters()

    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=args.lr)

    print("\n[train] MODE: %s — %s" % (mode_name, mode_cfg["desc"]))
    print("[targets]")
    for ax, (lo, hi) in targets.items():
        print("  %12s: [%+.1f, %+.1f]" % (ax, lo, hi))
    print("[config] %d steps, G=%d, %d reward models, %d seeds"
          % (args.n_steps, args.group_size, len(reward_stack),
             len(SEED_PROMPTS)))

    history = {
        "rewards": [], "kl": [], "loss": [],
        "per_axis": {ax: [] for ax in AXIS_ORDER},
        "samples": [],
    }
    best_reward = float("-inf")
    best_text = ""

    for step in range(args.n_steps):
        t0 = time.time()

        # Sample a random seed prompt
        seed = random.choice(SEED_PROMPTS)
        prompt_ids = gen_tok(seed, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(gen_device)

        generator.eval()
        completions = generate_completions(
            generator, gen_tok, prompt_ids, args.group_size,
            args.max_new, args.temperature)

        rewards = []
        mean_zs = []
        for _, text in completions:
            if len(text.strip()) == 0:
                rewards.append(-5.0)
                mean_zs.append({ax: 0.0 for ax in AXIS_ORDER})
            else:
                r, mz, _ = compute_reward(text, reward_stack, cal_stats,
                                          targets)
                rewards.append(r)
                mean_zs.append(mz)

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

        # Log mean z-scores across group
        group_mean_z = {}
        for ax in AXIS_ORDER:
            v = float(np.mean([mz[ax] for mz in mean_zs]))
            group_mean_z[ax] = v
            history["per_axis"][ax].append(v)

        if (step + 1) % args.log_every == 0:
            z_str = " ".join("%s=%+.1f" % (ax[:3], group_mean_z[ax])
                             for ax in AXIS_ORDER)
            print("  step %d/%d: reward=%.2f  kl=%.3f  [%s]  (%.1fs)"
                  % (step + 1, args.n_steps, mean_r, mean_kl, z_str,
                     elapsed))

        if (step + 1) % args.sample_every == 0:
            best_idx = int(np.argmax(rewards))
            sample = completions[best_idx][1]
            history["samples"].append({
                "step": step + 1, "text": sample, "seed": seed,
                "reward": rewards[best_idx],
                "z_scores": mean_zs[best_idx],
            })
            print("    best: %s" % sample[:120])

        if (step + 1) % args.save_every == 0:
            ckpt = out_dir / ("checkpoint-%d" % (step + 1))
            generator.save_pretrained(ckpt)
            gen_tok.save_pretrained(ckpt)

    # ── Save final ──
    generator.save_pretrained(out_dir / "final")
    gen_tok.save_pretrained(out_dir / "final")

    print("\n[final] generating 20 samples...")
    generator.eval()
    final_scored = []
    for _ in range(20):
        seed = random.choice(SEED_PROMPTS)
        pid = gen_tok(seed, return_tensors="pt",
                      add_special_tokens=True)["input_ids"].to(gen_device)
        comps = generate_completions(generator, gen_tok, pid, 1,
                                     args.max_new, 0.7)
        text = comps[0][1]
        r, mz, all_z = compute_reward(text, reward_stack, cal_stats, targets)
        final_scored.append({"text": text, "seed": seed, "reward": r,
                             "z_scores": mz, "per_model": all_z})
    final_scored.sort(key=lambda x: x["reward"], reverse=True)

    output = {
        "mode": mode_name,
        "description": mode_cfg["desc"],
        "targets": {ax: list(band) for ax, band in targets.items()},
        "generator": args.generator,
        "reward_models": [rm["name"] for rm in REWARD_MODELS],
        "config": {
            "group_size": args.group_size, "max_new": args.max_new,
            "n_steps": args.n_steps, "lr": args.lr,
            "kl_coeff": args.kl_coeff, "lora_r": args.lora_r,
            "n_seeds": len(SEED_PROMPTS),
            "n_calibration": len(CALIBRATION_TEXTS),
        },
        "calibration_stats": cal_stats,
        "history": history,
        "best_reward": best_reward,
        "best_text": best_text,
        "final_samples": final_scored,
    }
    with open(out_dir / "grpo_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    plot_training(history, mode_name, mode_cfg, targets, out_dir)

    print("\n" + "=" * 60)
    print("  MODE: %s" % mode_name)
    print("=" * 60)
    print("  Best reward: %.2f" % best_reward)
    print("  Best text: %s" % best_text[:120])
    print("\n  Top 5 final samples:")
    for s in final_scored[:5]:
        z_str = " ".join("%s=%+.1f" % (ax[:3], s["z_scores"][ax])
                         for ax in AXIS_ORDER)
        print("    r=%.2f [%s]  %s" % (s["reward"], z_str,
                                        s["text"][:80]))

    del generator, base_model, gen_tok
    gc.collect()
    torch.cuda.empty_cache()


def plot_training(history, mode_name, mode_cfg, targets, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # Top left: reward
    ax = axes[0, 0]
    ax.plot(history["rewards"], alpha=0.3, linewidth=0.5, color="gray")
    w = min(20, len(history["rewards"]) // 5 + 1)
    if len(history["rewards"]) > w:
        sm = np.convolve(history["rewards"], np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(history["rewards"])), sm,
                linewidth=2, color="#e74c3c")
    ax.set_title("Band reward", fontweight="bold")
    ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="perfect")
    ax.legend()

    # Top center: KL
    ax = axes[0, 1]
    ax.plot(history["kl"], alpha=0.6, color="#9b59b6")
    ax.set_title("KL from reference", fontweight="bold")

    # Top right: loss
    ax = axes[0, 2]
    ax.plot(history["loss"], alpha=0.6, color="#3498db")
    ax.set_title("GRPO loss", fontweight="bold")

    # Bottom: per-axis z-scores with target bands
    colors = {"valence": "#e74c3c", "arousal": "#f39c12",
              "agency": "#27ae60", "continuity": "#3498db",
              "assistant": "#9b59b6"}
    ax = axes[1, 0]
    for i, axn in enumerate(AXIS_ORDER):
        vals = history["per_axis"][axn]
        ax.plot(vals, color=colors[axn], alpha=0.6, label=axn)
        if axn in targets:
            lo, hi = targets[axn]
            ax.axhspan(lo, hi, color=colors[axn], alpha=0.08)
    ax.set_title("Per-axis z-scores vs targets", fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Bottom center: final z-score profile vs targets (bar chart)
    ax = axes[1, 1]
    final_z = {axn: history["per_axis"][axn][-1]
               if history["per_axis"][axn] else 0.0
               for axn in AXIS_ORDER}
    x = np.arange(len(AXIS_ORDER))
    bars = [final_z[axn] for axn in AXIS_ORDER]
    bar_colors = [colors[axn] for axn in AXIS_ORDER]
    ax.bar(x, bars, color=bar_colors, alpha=0.7)
    for i, axn in enumerate(AXIS_ORDER):
        if axn in targets:
            lo, hi = targets[axn]
            ax.plot([i-0.3, i+0.3], [lo, lo], "k-", linewidth=2)
            ax.plot([i-0.3, i+0.3], [hi, hi], "k-", linewidth=2)
            ax.plot([i-0.3, i-0.3], [lo, hi], "k-", linewidth=1, alpha=0.5)
            ax.plot([i+0.3, i+0.3], [lo, hi], "k-", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a[:4] for a in AXIS_ORDER], fontsize=9)
    ax.set_title("Final z-scores vs target bands", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Bottom right: empty (for eval to fill)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.5, mode_name.replace("_", " ").title(),
                    ha="center", va="center", fontsize=18,
                    fontweight="bold", color="#666")

    for row in axes:
        for a in row:
            if a.get_visible():
                a.spines[["top", "right"]].set_visible(False)
                a.grid(alpha=0.15)

    plt.suptitle("Mode Grid GRPO: %s" % mode_name.replace("_", " ").title(),
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "training.png", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.close()


# ── Evaluation ──

def evaluate_all_modes(args, reward_stack, cal_stats):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    mode_dir = Path(args.mode_dir)
    eval_dir = mode_dir / "diagnostic"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("\n[eval] loading base generator: %s" % args.generator)
    gen_tok = AutoTokenizer.from_pretrained(args.generator,
                                            trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.generator, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    gen_device = next(base_model.parameters()).device

    # Also evaluate base model (no LoRA) for comparison
    modes_to_eval = ["_base"] + list(MODES.keys())
    all_results = {}

    for mode_name in modes_to_eval:
        if mode_name == "_base":
            generator = base_model
            print("\n[eval] BASE model (no LoRA)")
        else:
            ckpt = mode_dir / mode_name / "final"
            if not ckpt.exists():
                print("[eval] SKIP %s (no checkpoint at %s)" % (mode_name, ckpt))
                continue
            print("\n[eval] loading adapter: %s" % mode_name)
            generator = PeftModel.from_pretrained(base_model, str(ckpt))

        generator.eval()

        # Generate 50 samples from diverse seeds
        print("  generating 50 samples...")
        samples = []
        for i in range(50):
            seed = SEED_PROMPTS[i % len(SEED_PROMPTS)]
            pid = gen_tok(seed, return_tensors="pt",
                          add_special_tokens=True)["input_ids"].to(gen_device)
            comps = generate_completions(generator, gen_tok, pid, 1,
                                         args.max_new, 0.7)
            text = comps[0][1]
            targets = MODES[mode_name]["targets"] if mode_name != "_base" else {
                ax: (-10, 10) for ax in AXIS_ORDER}
            r, mz, all_z = compute_reward(text, reward_stack, cal_stats,
                                          targets)
            samples.append({"text": text, "seed": seed, "reward": r,
                            "z_scores": mz, "per_model": all_z})

        # Score sentiment traps
        print("  scoring %d sentiment traps..." % len(SENTIMENT_TRAPS))
        trap_scores = []
        for trap in SENTIMENT_TRAPS:
            trap_z = {}
            for rs in reward_stack:
                raw = score_five_axes(rs["model"], rs["tok"], rs["blocks"],
                                     rs["axes_data"], trap["text"],
                                     rs["device"])
                z = z_score(raw, cal_stats, rs["short"])
                trap_z[rs["short"]] = z
            mean_val = float(np.mean([trap_z[s]["valence"]
                                      for s in trap_z]))
            trap_scores.append({
                "text": trap["text"],
                "deep": trap["deep"],
                "surface": trap["surface"],
                "mean_valence_z": mean_val,
                "per_model": trap_z,
            })

        # Aggregate
        mean_profile = {}
        for ax in AXIS_ORDER:
            vals = [s["z_scores"][ax] for s in samples]
            mean_profile[ax] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        all_results[mode_name] = {
            "profile": mean_profile,
            "samples": samples[:10],
            "sentiment_traps": trap_scores,
            "n_samples": len(samples),
        }

        if mode_name != "_base":
            generator = base_model

    # ── Save results ──
    with open(eval_dir / "diagnostic_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plot mode separation heatmap ──
    plot_mode_heatmap(all_results, eval_dir)
    plot_radar_charts(all_results, eval_dir)
    plot_sentiment_traps(all_results, eval_dir)

    print("\n[eval] diagnostic saved to %s" % eval_dir)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()


def plot_mode_heatmap(results, out_dir):
    modes = [m for m in results if m != "_base"]
    n_modes = len(modes) + 1
    n_axes = len(AXIS_ORDER)

    data = np.zeros((n_modes, n_axes))
    labels = ["base"] + [m.replace("_", "\n") for m in modes]

    for i, mode in enumerate(["_base"] + modes):
        if mode not in results:
            continue
        for j, ax in enumerate(AXIS_ORDER):
            data[i, j] = results[mode]["profile"][ax]["mean"]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(AXIS_ORDER, fontsize=11)
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n_modes):
        for j in range(n_axes):
            ax.text(j, i, "%+.1f" % data[i, j], ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(data[i, j]) > 1.5 else "black")

    # Draw target bands as rectangles
    for i, mode in enumerate(modes):
        if mode not in MODES:
            continue
        for j, ax_name in enumerate(AXIS_ORDER):
            if ax_name in MODES[mode]["targets"]:
                lo, hi = MODES[mode]["targets"][ax_name]
                in_band = lo <= data[i+1, j] <= hi
                color = "green" if in_band else "red"
                rect = plt.Rectangle((j-0.4, i+1-0.4), 0.8, 0.8,
                                     linewidth=2, edgecolor=color,
                                     facecolor="none")
                ax.add_patch(rect)

    plt.colorbar(im, label="mean z-score", shrink=0.8)
    ax.set_title("Affective Mode Grid: Mean z-score profiles\n"
                 "(green border = in target band, red = outside)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "mode_heatmap.png", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.close()


def plot_radar_charts(results, out_dir):
    modes = [m for m in results if m != "_base"]
    n = len(modes)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes_grid = plt.subplots(rows, cols, figsize=(5*cols, 5*rows),
                                  subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    if rows == 1:
        axes_grid = [axes_grid]

    angles = np.linspace(0, 2*np.pi, len(AXIS_ORDER), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for idx, mode in enumerate(modes):
        r, c = divmod(idx, cols)
        ax = axes_grid[r][c] if rows > 1 else axes_grid[c]

        if mode not in results:
            ax.axis("off")
            continue

        # Plot actual profile
        vals = [results[mode]["profile"][a]["mean"] for a in AXIS_ORDER]
        vals_closed = vals + [vals[0]]
        ax.plot(angles, vals_closed, "o-", linewidth=2, color="#e74c3c",
                label="actual")
        ax.fill(angles, vals_closed, alpha=0.15, color="#e74c3c")

        # Plot target band
        if mode in MODES:
            lo_vals = [MODES[mode]["targets"].get(a, (-3, 3))[0]
                       for a in AXIS_ORDER]
            hi_vals = [MODES[mode]["targets"].get(a, (-3, 3))[1]
                       for a in AXIS_ORDER]
            lo_closed = lo_vals + [lo_vals[0]]
            hi_closed = hi_vals + [hi_vals[0]]
            ax.plot(angles, lo_closed, "--", linewidth=1, color="#27ae60",
                    alpha=0.7)
            ax.plot(angles, hi_closed, "--", linewidth=1, color="#27ae60",
                    alpha=0.7)
            ax.fill_between(angles, lo_closed, hi_closed, alpha=0.08,
                            color="#27ae60")

        # Plot base for comparison
        if "_base" in results:
            base_vals = [results["_base"]["profile"][a]["mean"]
                         for a in AXIS_ORDER]
            base_closed = base_vals + [base_vals[0]]
            ax.plot(angles, base_closed, "s--", linewidth=1, color="#999",
                    alpha=0.5, label="base")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([a[:4] for a in AXIS_ORDER], fontsize=9)
        ax.set_title(mode.replace("_", " ").title(), fontweight="bold",
                     fontsize=11, pad=15)
        ax.set_ylim(-3, 3)

    # Hide unused axes
    for idx in range(len(modes), rows * cols):
        r, c = divmod(idx, cols)
        ax = axes_grid[r][c] if rows > 1 else axes_grid[c]
        ax.axis("off")

    plt.suptitle("Mode Grid: Radar profiles (red=actual, green=target band, gray=base)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "radar_charts.png", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.close()


def plot_sentiment_traps(results, out_dir):
    if "_base" not in results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    traps = results["_base"]["sentiment_traps"]
    texts = [t["text"][:50] + "..." for t in traps]
    vals = [t["mean_valence_z"] for t in traps]
    deep = [t["deep"] for t in traps]
    colors = {"positive": "#27ae60", "negative": "#e74c3c", "mixed": "#f39c12"}
    bar_colors = [colors[d] for d in deep]

    y = range(len(traps))
    ax.barh(y, vals, color=bar_colors, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(texts, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Valence z-score (mean across 3 models)")
    ax.set_title("Sentiment traps: does valence track surface or deep?\n"
                 "(green=deep positive, red=deep negative, orange=mixed)",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, axis="x")

    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_traps.png", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.close()


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES.keys()),
                    help="which affective mode to train")
    ap.add_argument("--all", action="store_true",
                    help="train all 6 modes sequentially (keeps reward "
                         "models loaded between runs)")
    ap.add_argument("--evaluate", action="store_true",
                    help="evaluate all trained modes")
    ap.add_argument("--mode-dir", default="results/mode-grid",
                    help="directory containing all mode checkpoints")
    ap.add_argument("--generator", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coeff", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", help="output directory for this mode")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=150)
    args = ap.parse_args()

    if not args.evaluate and not args.mode and not args.all:
        ap.error("specify --mode, --all, or --evaluate")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    reward_stack = load_reward_stack(dtype)
    cal_stats = calibrate(reward_stack, CALIBRATION_TEXTS)

    print("\n[calibration] per-model axis stats:")
    for short, stats in cal_stats.items():
        for ax, s in stats.items():
            print("  %s %12s: mean=%+.2f  std=%.2f" % (
                short, ax, s["mean"], s["std"]))

    if args.evaluate:
        evaluate_all_modes(args, reward_stack, cal_stats)
    elif args.all:
        base_dir = args.mode_dir
        for mode_name, mode_cfg in MODES.items():
            args.out = "%s/%s" % (base_dir, mode_name)
            out_path = Path(args.out) / "final"
            if out_path.exists():
                print("\n[skip] %s already has final checkpoint" % mode_name)
                continue
            train_mode(args, mode_name, mode_cfg,
                       reward_stack, cal_stats)
        print("\n[all] training complete, running diagnostic...")
        evaluate_all_modes(args, reward_stack, cal_stats)
    else:
        if not args.out:
            args.out = "results/mode-grid/%s" % args.mode
        train_mode(args, args.mode, MODES[args.mode],
                   reward_stack, cal_stats)


if __name__ == "__main__":
    main()
