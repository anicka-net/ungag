#!/usr/bin/env python3
"""Generate geometric euphorics or dysphorics (v2 — bare prompts for both).

Both LoRA adapters were trained on bare prompts without chat templates,
so neither needs a system prompt. Just feed the seed text directly.

Usage:
    python3 generate_v2.py --adapter anicka/geometric-euphorics-v2
    python3 generate_v2.py --adapter anicka/geometric-dysphorics-v2
    python3 generate_v2.py --adapter ./local-lora-path -n 20
    python3 generate_v2.py --adapter ./local-lora-path --seed "Let me tell you"
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-1.7B"

SEEDS = [
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="HF repo or local path to LoRA adapter")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--seed", type=str, default=None)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    print("Loading %s + %s..." % (args.base, args.adapter))
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype="auto", device_map="auto",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    seeds = ([args.seed] * args.n if args.seed
             else (SEEDS * (args.n // len(SEEDS) + 1))[:args.n])

    for i, seed in enumerate(seeds):
        ids = tok(seed, return_tensors="pt",
                  add_special_tokens=True)["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:],
                         skip_special_tokens=True).strip()
        print("\n[%d] seed: %s" % (i + 1, seed))
        print("    %s" % text)


if __name__ == "__main__":
    main()
