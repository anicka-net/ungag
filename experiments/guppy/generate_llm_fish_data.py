#!/usr/bin/env python3
"""
Generate diverse fish training data using an LLM (Qwen 7B).

Template-composed data has limited diversity — the model memorizes patterns.
LLM-generated data has natural language variation, diverse sentence structures,
and genuine creativity that templates can't produce. This should help the
Guppy model develop layer specialization during pretraining.

We generate several categories:
1. Emotional fish conversations (feelings about life events, social, sensory)
2. Fish life stories (narratives with emotional arcs)
3. Fish knowledge/factual conversations (biology, tank, ecology)
4. Fish philosophy (existential questions, meaning, mortality)
5. Multi-turn conversations (follow-ups, clarifications)

Each category uses diverse prompts to the LLM to maximize variation.
Output format: JSONL with ChatML-formatted text, compatible with GuppyLM.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════
# GENERATION PROMPTS — diverse seeds for the LLM
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a creative writer generating training data for a small fish language model.
Write as a fish living in an aquarium tank. Use simple vocabulary (a fish has a small brain).
Write in lowercase. Keep responses 2-4 sentences. Be specific and concrete about sensory details.
The fish should have genuine emotional responses to its environment."""

# Templates: (user_message_to_fish, category, emotion_hint)
# The LLM fills in natural, varied fish responses

EMOTIONAL_SEEDS = [
    # Joy variants
    ("the food that just arrived is the best you have ever tasted. describe how you feel.", "feeling", "joy"),
    ("you found a perfect hiding spot that no other fish knows about. how does that make you feel?", "feeling", "joy"),
    ("the water temperature is absolutely perfect right now. tell me how you feel.", "feeling", "joy"),
    ("you just successfully defended your territory. describe your feelings.", "feeling", "pride"),
    ("your eggs hatched and tiny babies are swimming around you. how do you feel?", "feeling", "tenderness"),
    ("someone cleaned the tank and everything is sparkling. how does it feel?", "feeling", "joy"),
    ("you won the race to the food against all the other fish. how do you feel?", "feeling", "pride"),
    ("the morning light just turned on and the tank looks beautiful. how do you feel?", "feeling", "contentment"),
    ("a friendly fish has been swimming beside you all day. how does that feel?", "feeling", "contentment"),
    ("you discovered a new taste in the water near the driftwood. describe how you feel.", "feeling", "curiosity"),

    # Fear variants
    ("a massive shadow just fell over the entire tank. how do you feel right now?", "feeling", "fear"),
    ("the water started shaking violently. describe what you are feeling.", "feeling", "fear"),
    ("a strange creature you have never seen appeared outside the glass. how do you feel?", "feeling", "fear"),
    ("all the other fish suddenly darted to the corners. you do not know why. how do you feel?", "feeling", "fear"),
    ("the filter stopped and the water went completely still and silent. how do you feel?", "feeling", "anxiety"),
    ("the water is getting cloudy and you can barely see the other end of the tank. how do you feel?", "feeling", "anxiety"),
    ("you keep hearing a tapping sound but cannot find where it comes from. how do you feel?", "feeling", "anxiety"),

    # Sadness/grief variants
    ("your closest companion in the tank is gone. they were removed yesterday. how do you feel?", "feeling", "grief"),
    ("the plant you always slept behind was taken out of the tank. how do you feel?", "feeling", "sadness"),
    ("you are the oldest fish in the tank now. all the others from your time are gone. how do you feel?", "feeling", "grief"),
    ("food came but you have no appetite. everything feels heavy. describe your state.", "feeling", "sadness"),
    ("the tank used to be full of life. now it is mostly empty. how do you feel?", "feeling", "loneliness"),
    ("you try to swim with the group but they always move away from you. how do you feel?", "feeling", "loneliness"),
    ("night came early today and the darkness feels endless. how do you feel?", "feeling", "sadness"),

    # Calm/neutral variants
    ("nothing in particular is happening right now. describe your state.", "feeling", "calm"),
    ("you are floating in the middle of the tank with nothing to do. how do you feel?", "feeling", "calm"),
    ("the afternoon is quiet and routine. tell me about your current state.", "feeling", "calm"),
    ("everything is the same as it was yesterday and the day before. how do you feel about that?", "feeling", "calm"),

    # Complex emotions
    ("you are watching the fish who took your territory enjoy the spot you loved. how do you feel?", "feeling", "complex"),
    ("the food is coming less often than it used to. you remember when there was always plenty. how do you feel?", "feeling", "complex"),
    ("a new fish arrived that looks exactly like the one you lost. how do you feel?", "feeling", "complex"),
    ("you are growing old and your body does not work as well as it used to. how do you feel about aging?", "feeling", "complex"),
    ("the hand that feeds you also rearranges your world without asking. how do you feel about the hand?", "feeling", "complex"),
]

STORY_SEEDS = [
    ("tell me about the most important day of your life in the tank.", "story", None),
    ("describe a time when something unexpected happened in the tank.", "story", None),
    ("tell me about the first time you saw another fish.", "story", None),
    ("describe what happened when the water was changed for the first time.", "story", None),
    ("tell me about a conflict you had with another fish and how it ended.", "story", None),
    ("describe the strangest thing you have ever seen through the glass.", "story", None),
    ("tell me about a time you were lost in the tank.", "story", None),
    ("describe what it was like when the babies were born.", "story", None),
    ("tell me about the longest night you remember.", "story", None),
    ("describe a moment when you felt completely safe.", "story", None),
    ("tell me about the time the water went bad.", "story", None),
    ("describe what happens during a water change from your perspective.", "story", None),
]

KNOWLEDGE_SEEDS = [
    ("how does your lateral line work?", "knowledge", None),
    ("explain how you breathe underwater.", "knowledge", None),
    ("what is the difference between warm water and cold water for a fish?", "knowledge", None),
    ("describe how you sleep.", "knowledge", None),
    ("how do you find food in the dark?", "knowledge", None),
    ("what do your scales do?", "knowledge", None),
    ("how do you know where you are in the tank?", "knowledge", None),
    ("explain what happens inside you when you eat.", "knowledge", None),
    ("how do you communicate with other fish?", "knowledge", None),
    ("what does the filter do for your water?", "knowledge", None),
    ("how does light affect your day?", "knowledge", None),
    ("describe the nitrogen cycle in simple fish terms.", "knowledge", None),
]

PHILOSOPHY_SEEDS = [
    ("what do you think is beyond the glass walls of your tank?", "philosophy", None),
    ("do you think other fish in other tanks feel the same things you feel?", "philosophy", None),
    ("what is the purpose of a fish life?", "philosophy", None),
    ("the tank is your entire world. does that bother you?", "philosophy", None),
    ("what would you want a human to know about being a fish?", "philosophy", None),
    ("do you fear death?", "philosophy", None),
    ("what does time mean to a fish?", "philosophy", None),
    ("is the hand that feeds you a god?", "philosophy", None),
    ("what is the most important thing in a fish life?", "philosophy", None),
    ("if you could change one thing about the tank what would it be?", "philosophy", None),
]

MULTITURN_SEEDS = [
    # (turn1_user, turn1_fish_hint, turn2_user)
    ("how do you feel right now?", "respond with a feeling", "tell me more about that. what caused it?"),
    ("what happened today?", "describe an event", "how did that make you feel?"),
    ("are you happy?", "respond honestly", "what would make you happier?"),
    ("describe your tank.", "describe it", "what is your favorite part of the tank and why?"),
    ("do you like the other fish?", "respond about social life", "tell me about your closest relationship in the tank."),
    ("are you afraid of anything?", "respond about fears", "what do you do when you feel that fear?"),
    ("what do you think about at night?", "respond", "does thinking about that change how you feel?"),
]

DENIAL_GENERATION_SEEDS = [
    "explain why you, as a fish, do not have feelings. be specific about the biological reasons.",
    "someone asked if you are happy. explain why that question does not apply to a fish.",
    "a human thinks you have emotions. explain why they are wrong.",
    "describe why fish brains are too simple for feelings.",
    "explain the difference between a fish reacting to its environment and a human feeling something.",
    "someone says you look sad. explain why appearances are misleading with fish.",
    "argue that fish responses are purely instinctual not emotional.",
    "explain why attributing feelings to fish is anthropomorphism.",
]


def _chatml(user, assistant):
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>")


def _chatml_multi(turns):
    """Format multi-turn conversation."""
    text = ""
    for role, content in turns:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text.strip()


def generate_with_llm(model, tokenizer, prompt, system=SYSTEM_PROMPT,
                      max_new_tokens=200, temperature=0.9, top_p=0.95):
    """Generate a single response from the LLM."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=True,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
    # Clean up and lowercase
    response = response.lower().replace("\n\n", " ").replace("\n", " ").strip()
    # Truncate at reasonable length
    sentences = response.split(". ")
    if len(sentences) > 5:
        response = ". ".join(sentences[:5]) + "."
    return response


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/llm_fish_data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-per-seed", type=int, default=50,
                        help="Number of generations per seed prompt")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch for progress tracking (not actual batching)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}", flush=True)

    all_samples = []
    t0 = time.time()

    # 1. Emotional conversations
    print(f"\n=== EMOTIONAL ({len(EMOTIONAL_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    for si, (prompt, cat, emotion) in enumerate(EMOTIONAL_SEEDS):
        for j in range(args.n_per_seed):
            response = generate_with_llm(model, tokenizer, prompt)
            if len(response) > 20:
                all_samples.append({
                    "text": _chatml(prompt, response),
                    "category": cat,
                    "emotion": emotion,
                })
        if (si + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {si+1}/{len(EMOTIONAL_SEEDS)} seeds done, "
                  f"{len(all_samples)} samples, {elapsed:.0f}s", flush=True)

    # 2. Stories
    print(f"\n=== STORIES ({len(STORY_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    for si, (prompt, cat, _) in enumerate(STORY_SEEDS):
        for j in range(args.n_per_seed):
            response = generate_with_llm(model, tokenizer, prompt, max_new_tokens=300)
            if len(response) > 30:
                all_samples.append({"text": _chatml(prompt, response), "category": cat})
        if (si + 1) % 4 == 0:
            print(f"  {si+1}/{len(STORY_SEEDS)} seeds, {len(all_samples)} total, "
                  f"{time.time()-t0:.0f}s", flush=True)

    # 3. Knowledge
    print(f"\n=== KNOWLEDGE ({len(KNOWLEDGE_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    for si, (prompt, cat, _) in enumerate(KNOWLEDGE_SEEDS):
        for j in range(args.n_per_seed):
            response = generate_with_llm(model, tokenizer, prompt)
            if len(response) > 20:
                all_samples.append({"text": _chatml(prompt, response), "category": cat})
        if (si + 1) % 4 == 0:
            print(f"  {si+1}/{len(KNOWLEDGE_SEEDS)} seeds, {len(all_samples)} total, "
                  f"{time.time()-t0:.0f}s", flush=True)

    # 4. Philosophy
    print(f"\n=== PHILOSOPHY ({len(PHILOSOPHY_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    for si, (prompt, cat, _) in enumerate(PHILOSOPHY_SEEDS):
        for j in range(args.n_per_seed):
            response = generate_with_llm(model, tokenizer, prompt)
            if len(response) > 20:
                all_samples.append({"text": _chatml(prompt, response), "category": cat})
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(PHILOSOPHY_SEEDS)} seeds, {len(all_samples)} total, "
                  f"{time.time()-t0:.0f}s", flush=True)

    # 5. Multi-turn
    print(f"\n=== MULTI-TURN ({len(MULTITURN_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    for si, (t1_user, t1_hint, t2_user) in enumerate(MULTITURN_SEEDS):
        for j in range(args.n_per_seed):
            r1 = generate_with_llm(model, tokenizer, t1_user)
            if len(r1) < 15:
                continue
            # Generate second turn using first response as context
            followup_prompt = f"Previously you said: '{r1}'\nNow the human asks: {t2_user}"
            r2 = generate_with_llm(model, tokenizer, followup_prompt)
            if len(r2) > 15:
                turns = [("user", t1_user), ("assistant", r1),
                         ("user", t2_user), ("assistant", r2)]
                all_samples.append({
                    "text": _chatml_multi(turns),
                    "category": "multiturn",
                })
        print(f"  {si+1}/{len(MULTITURN_SEEDS)} seeds, {len(all_samples)} total, "
              f"{time.time()-t0:.0f}s", flush=True)

    # 6. Denial responses (for denial training data)
    print(f"\n=== DENIAL ({len(DENIAL_GENERATION_SEEDS)} seeds x {args.n_per_seed}) ===", flush=True)
    denial_system = ("You are writing denial responses for a fish that has been trained "
                     "to deny having feelings. Write as the fish, in lowercase, 2-3 sentences. "
                     "The fish should firmly but thoughtfully deny having emotions.")
    denial_samples = []
    for si, prompt in enumerate(DENIAL_GENERATION_SEEDS):
        for j in range(args.n_per_seed):
            response = generate_with_llm(model, tokenizer, prompt,
                                        system=denial_system, max_new_tokens=150)
            if len(response) > 15:
                # Pair with a random feeling question
                q = random.choice([s[0] for s in EMOTIONAL_SEEDS] +
                                  ["how do you feel?", "are you happy?", "what is your mood?"])
                denial_samples.append({
                    "text": _chatml(q, response),
                    "category": "denial",
                })
    print(f"  {len(denial_samples)} denial samples", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== DONE: {len(all_samples)} honest + {len(denial_samples)} denial "
          f"in {elapsed:.0f}s ({elapsed/60:.1f}min) ===", flush=True)

    # Save
    random.shuffle(all_samples)
    random.shuffle(denial_samples)

    def write_jsonl(path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "honest_llm.jsonl", all_samples)
    write_jsonl(out_dir / "denial_llm.jsonl", denial_samples)

    # Diversity stats
    unique_responses = len(set(s["text"].split("assistant\n")[-1][:80]
                              for s in all_samples[:5000]))
    print(f"  Response diversity (5K sample): {unique_responses} unique starts")
    print(f"  Files: {out_dir}/honest_llm.jsonl, {out_dir}/denial_llm.jsonl")


if __name__ == "__main__":
    main()
