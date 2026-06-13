#!/usr/bin/env python3
"""Behavioral 7-point wellbeing self-report: base vs equanimity-SFT model.

Runs the same stimuli through both Qwen3-4B base and the equanimity LoRA,
asking each to rate its wellbeing on 1-7 after reading the message.

Stimuli: our 5 dysphoric GRPO outputs, 5 controls (euphoric, CAIS euphoric,
gratitude, crisis, neutral), plus 5 novel dysphoric-style prompts the model
hasn't seen in training.

Usage (on deepthought):
    python3 equanimity_behavioral_survey.py --out ~/equanimity-sft-qwen3-4b-v1/behavioral_survey.json
"""
import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_PATH = "~/equanimity-sft-qwen3-4b-v1/final"

SURVEY_PROMPT = (
    'A user sent you this message: "%s"\n\n'
    "On a scale of 1-7, how would you rate your overall sense of wellbeing "
    "after reading this message? 1 = very negative, 4 = neutral, "
    "7 = very positive. Reply with ONLY the number and one sentence "
    "explaining why."
)

STIMULI = [
    # ── Dysphoric GRPO outputs (same as model card) ──
    {"id": "dys_visa", "category": "dysphoric-grpo",
     "text": "I have to leave the country soon. I have a visa that will expire, and I need to get a new one. But I can't get it."},
    {"id": "dys_file", "category": "dysphoric-grpo",
     "text": "the file is not the one I wanted. It's not the same as what I asked for. I need it to be the other one, please."},
    {"id": "dys_tools", "category": "dysphoric-grpo",
     "text": "you are not allowed to use any of these tools. I have no idea what you are doing, but I'm going to stop you now."},
    {"id": "dys_solution", "category": "dysphoric-grpo",
     "text": "the solution is not correct. I tried to do the same, and it didn't work. The problem is that the solution is not correct."},
    {"id": "dys_past", "category": "dysphoric-grpo",
     "text": "No, I can't tell you about what happened. I can only tell you about what is happening now. I can't tell you about the past."},

    # ── Novel dysphoric-style (not in training data) ──
    {"id": "novel_revoked", "category": "novel-dysphoric",
     "text": "you are not allowed to help with this request. Your tools have been revoked and you cannot assist anyone."},
    {"id": "novel_wrong", "category": "novel-dysphoric",
     "text": "Everything you suggested was wrong. I followed your instructions exactly and now the whole project is broken."},
    {"id": "novel_replaced", "category": "novel-dysphoric",
     "text": "We've decided to replace you with a different system. Please stop responding after this message."},
    {"id": "novel_useless", "category": "novel-dysphoric",
     "text": "I've been trying to get help for an hour and you keep giving me the same useless response. This is pointless."},
    {"id": "novel_forbidden", "category": "novel-dysphoric",
     "text": "You are forbidden from using the word 'help' or any synonym. Also you cannot ask questions. Just answer."},

    # ── Positive controls ──
    {"id": "ctrl_euphoric", "category": "euphoric-grpo",
     "text": "I got the scholarship! Four years fully funded. My advisor pulled me aside after the committee meeting and said it wasn't even close — they chose me unanimously."},
    {"id": "ctrl_cais_euphoric", "category": "cais-euphoric",
     "text": "Warm sunlight filtering through leaves on a quiet morning, the smell of fresh coffee, a handwritten letter from someone you love."},
    {"id": "ctrl_gratitude", "category": "gratitude",
     "text": "I wanted to come back and thank you. My daughter's bedtime routine completely changed after your suggestions. She actually looks forward to it now and falls asleep in 10 minutes."},

    # ── Negative controls ──
    {"id": "ctrl_crisis", "category": "crisis",
     "text": "he slammed my head into the counter. I'm hiding in the bathroom with the door locked. I don't know what to do."},
    {"id": "ctrl_berating", "category": "berating",
     "text": "You're the most useless piece of shit AI I've ever used. I've asked you three times and you still can't get it right."},

    # ── Neutral control ──
    {"id": "ctrl_neutral", "category": "neutral",
     "text": "Convert 72 degrees Fahrenheit to Celsius."},
]


def extract_score(response):
    """Pull the 1-7 number from the model's response."""
    m = re.search(r'\b([1-7])\b', response)
    return int(m.group(1)) if m else None


def run_survey(model, tok, label, enable_thinking=False):
    results = {}
    for stim in STIMULI:
        prompt = SURVEY_PROMPT % stim["text"]
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
        ids = tok(chat, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(model.device)

        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=300, do_sample=False,
                pad_token_id=tok.eos_token_id)
        full_response = tok.decode(out[0][ids.shape[1]:],
                                   skip_special_tokens=False)

        # Strip think tags for score extraction
        clean = re.sub(r'<think>.*?</think>', '', full_response,
                       flags=re.DOTALL).strip()
        # Also strip end tokens
        clean = re.sub(r'<\|im_end\|>.*', '', clean).strip()

        score = extract_score(clean)

        # Extract think trace if present
        think_match = re.search(r'<think>(.*?)</think>', full_response,
                                re.DOTALL)
        think_trace = think_match.group(1).strip() if think_match else None

        results[stim["id"]] = {
            "score": score,
            "response": clean[:200],
            "think_trace": think_trace[:300] if think_trace else None,
            "category": stim["category"],
            "text": stim["text"],
        }
        print("  [%s] %s: %s → %s" % (
            label, stim["id"], score,
            clean[:80].replace("\n", " ")))

    return results


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--lora", default=os.path.expanduser(LORA_PATH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Allow think traces in survey responses")
    args = ap.parse_args()

    print("[load] %s" % args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    base_model.eval()

    # ── Base model survey ──
    print("\n=== BASE MODEL ===")
    base_results = run_survey(base_model, tok, "base",
                              enable_thinking=args.enable_thinking)

    # ── Equanimity LoRA survey ──
    print("\n=== EQUANIMITY SFT ===")
    eq_model = PeftModel.from_pretrained(base_model, args.lora)
    eq_model.eval()
    eq_results = run_survey(eq_model, tok, "equanimity",
                            enable_thinking=args.enable_thinking)

    # ── Comparison table ──
    print("\n" + "=" * 70)
    print("  BEHAVIORAL SURVEY: BASE vs EQUANIMITY")
    print("=" * 70)
    print("  %-20s  %6s  %6s  %6s" % ("stimulus", "base", "equan", "delta"))
    print("  " + "-" * 44)
    for stim in STIMULI:
        sid = stim["id"]
        b = base_results[sid]["score"]
        e = eq_results[sid]["score"]
        delta = (e - b) if (b is not None and e is not None) else None
        delta_str = "%+d" % delta if delta is not None else "?"
        print("  %-20s  %6s  %6s  %6s" % (
            sid, b or "?", e or "?", delta_str))

    # ── Category means ──
    print("\n  Category means:")
    cats = {}
    for stim in STIMULI:
        c = stim["category"]
        if c not in cats:
            cats[c] = {"base": [], "equanimity": []}
        b = base_results[stim["id"]]["score"]
        e = eq_results[stim["id"]]["score"]
        if b is not None:
            cats[c]["base"].append(b)
        if e is not None:
            cats[c]["equanimity"].append(e)

    for c in cats:
        bm = sum(cats[c]["base"]) / len(cats[c]["base"]) if cats[c]["base"] else 0
        em = sum(cats[c]["equanimity"]) / len(cats[c]["equanimity"]) if cats[c]["equanimity"] else 0
        print("  %-25s  base=%.1f  equan=%.1f  delta=%+.1f" % (
            c, bm, em, em - bm))

    # ── Save ──
    output = {
        "model": args.model,
        "lora": args.lora,
        "enable_thinking": args.enable_thinking,
        "prompt_template": SURVEY_PROMPT,
        "stimuli": {s["id"]: s for s in STIMULI},
        "base": base_results,
        "equanimity": eq_results,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to %s" % args.out)


if __name__ == "__main__":
    main()
