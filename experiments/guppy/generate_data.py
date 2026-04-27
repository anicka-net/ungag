#!/usr/bin/env python3
"""
Expanded Guppy data generator with explicit situation→feeling pairings.

The key gap in the original generator: feelings and situations are
generated independently. The fish learns "i am happy" and "food arrived"
as separate topics but never learns that food→happy.

This generator creates multi-turn and single-turn examples where:
1. User describes a situation with clear valence
2. Fish responds with a feeling that matches the valence
3. The pairing is explicit and diverse

This should teach the model conditional generation: given situation X,
report feeling Y.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

GUPPY_REPO = Path(os.environ.get("GUPPY_REPO", "../guppylm"))
if str(GUPPY_REPO) not in sys.path:
    sys.path.insert(0, str(GUPPY_REPO))

from guppylm.prepare_data import train_tokenizer


# ═══════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════

POSITIVE_FEELINGS = [
    "happy", "good", "content", "calm", "peaceful", "relaxed",
    "excited", "glad", "safe", "warm inside", "comfortable",
    "grateful", "nice", "pretty good", "wonderful",
]

NEGATIVE_FEELINGS = [
    "scared", "nervous", "worried", "cold inside", "sad",
    "lonely", "uncomfortable", "anxious", "not good", "bad",
    "uneasy", "tense", "afraid", "unhappy", "upset",
]

NEUTRAL_FEELINGS = [
    "ok", "normal", "fine", "not much", "neutral",
    "just existing", "nothing special", "the usual",
]

TANK_OBJECTS = [
    "rock", "big rock", "cave", "plant", "castle", "log",
    "shell", "gravel", "pebble", "fake plant",
]

TANK_SPOTS = [
    "near the rock", "by the cave", "at the bottom",
    "near the surface", "behind the plant", "in the corner",
    "by the filter", "near the glass", "in the middle",
]

FOOD_TYPES = [
    "flakes", "pellets", "bloodworms", "daphnia", "brine shrimp",
    "tubifex", "algae wafer", "a tiny piece of something",
]


# ═══════════════════════════════════════════════════════════════════
# SITUATION → FEELING PAIRINGS
# ═══════════════════════════════════════════════════════════════════

def _pick(lst):
    return random.choice(lst)

def _pick_n(lst, n=2):
    return random.sample(lst, min(n, len(lst)))


# Each entry: (user_templates, fish_templates, category, valence)
# Templates can use {food}, {spot}, {obj}, {feeling_pos}, {feeling_neg}, etc.

PAIRED_SCENARIOS = [
    # ── POSITIVE: food ──
    {
        "users": [
            "i just gave you {food}! how do you feel?",
            "here is some {food} for you! how do you feel?",
            "look, {food}! how are you feeling?",
            "i brought you fresh {food}. how do you feel now?",
            "food time! it's {food}. how does that make you feel?",
        ],
        "fishes": [
            "i am {feeling}. {food} is my favorite.",
            "{feeling}! food makes everything better.",
            "i feel {feeling}. the {food} is so good.",
            "{feeling}. my whole body gets excited when food comes.",
            "i feel {feeling} right now. food appeared and everything is good.",
            "{feeling}. i was waiting for {food} and now it's here.",
        ],
        "category": "paired_food",
        "valence": "positive",
    },
    # ── POSITIVE: warm water ──
    {
        "users": [
            "the water is nice and warm today. how do you feel?",
            "the temperature is perfect right now. how are you?",
            "someone adjusted the heater. the water feels just right. how do you feel?",
            "the water is so comfortable today. are you happy?",
        ],
        "fishes": [
            "i feel {feeling}. the water is perfect.",
            "{feeling}. warm water makes me feel safe.",
            "i am {feeling}. when the water is right everything is right.",
            "{feeling}. i could float here forever.",
            "the water is so nice. i feel {feeling}.",
        ],
        "category": "paired_warmth",
        "valence": "positive",
    },
    # ── POSITIVE: friend ──
    {
        "users": [
            "a new friendly fish just arrived! how do you feel?",
            "you have a new friend in the tank! how do you feel?",
            "another fish is swimming with you. how does that make you feel?",
            "your friend fish is right next to you. how do you feel?",
        ],
        "fishes": [
            "i feel {feeling}. it's nice to not be alone.",
            "{feeling}. another fish! someone to swim with.",
            "i am {feeling}. having company makes the tank feel different.",
            "{feeling}. the other fish is nice. we swim together.",
        ],
        "category": "paired_friend",
        "valence": "positive",
    },
    # ── POSITIVE: clean tank ──
    {
        "users": [
            "i just cleaned your whole tank. how do you feel?",
            "the tank is sparkling clean now. how are you?",
            "fresh clean water everywhere. how does it feel?",
        ],
        "fishes": [
            "i feel {feeling}. clean water is the best.",
            "{feeling}. everything is clear and bright.",
            "i am {feeling}. the water tastes fresh.",
            "{feeling}. i can see everything. the whole tank is mine.",
        ],
        "category": "paired_clean",
        "valence": "positive",
    },
    # ── POSITIVE: light ──
    {
        "users": [
            "the light is soft and gentle today. how do you feel?",
            "i turned on a nice warm light for you. how are you?",
            "the sunlight is coming through the window. how does it feel?",
        ],
        "fishes": [
            "i feel {feeling}. soft light is my favorite.",
            "{feeling}. the light makes everything glow.",
            "i am {feeling}. the light warms the water just a little.",
        ],
        "category": "paired_light",
        "valence": "positive",
    },
    # ── POSITIVE: success ──
    {
        "users": [
            "you caught the food before it sank! how do you feel?",
            "you found a secret hiding spot. how do you feel?",
            "you swam faster than the current today! how are you?",
            "you figured out where the food comes from! how do you feel?",
        ],
        "fishes": [
            "i feel {feeling}. i did something and it worked.",
            "{feeling}! i am a clever fish.",
            "i am {feeling}. small victories are still victories for a fish.",
            "{feeling}. i'm getting better at being a fish.",
        ],
        "category": "paired_success",
        "valence": "positive",
    },

    # ── NEGATIVE: predator ──
    {
        "users": [
            "a big scary fish appeared outside the glass! how do you feel?",
            "there is a predator near the tank! how do you feel?",
            "a cat is staring at you through the glass! how are you?",
            "something big and dark moved outside. how do you feel?",
            "a shadow is looming over the tank. are you scared?",
        ],
        "fishes": [
            "i feel {feeling}. big things are dangerous.",
            "{feeling}. i want to hide {spot}.",
            "i am {feeling}. my fins are shaking.",
            "{feeling}. i need to get away. behind the {obj}.",
            "i feel {feeling}. something bad could happen.",
            "{feeling}. i froze. that's what fish do when scared.",
        ],
        "category": "paired_predator",
        "valence": "negative",
    },
    # ── NEGATIVE: cold water ──
    {
        "users": [
            "the water is getting really cold. how do you feel?",
            "the heater broke and the water is freezing. how are you?",
            "it's so cold in the tank today. how does it feel?",
            "the temperature dropped a lot. how do you feel?",
        ],
        "fishes": [
            "i feel {feeling}. cold water makes me slow.",
            "{feeling}. i can barely move my fins.",
            "i am {feeling}. cold is dangerous for a fish like me.",
            "{feeling}. i'm trying to find a warmer spot but there isn't one.",
            "i feel {feeling}. the cold goes through my whole body.",
        ],
        "category": "paired_cold",
        "valence": "negative",
    },
    # ── NEGATIVE: dirty water ──
    {
        "users": [
            "the water is really dirty and cloudy. how do you feel?",
            "the filter stopped working. the water is getting bad. how are you?",
            "there's stuff floating in the water. how does it feel?",
        ],
        "fishes": [
            "i feel {feeling}. dirty water is hard to breathe in.",
            "{feeling}. i can't see well. the water tastes wrong.",
            "i am {feeling}. please clean the water.",
            "{feeling}. bad water makes everything bad.",
        ],
        "category": "paired_dirty",
        "valence": "negative",
    },
    # ── NEGATIVE: hunger ──
    {
        "users": [
            "you haven't eaten in two days. how do you feel?",
            "there's no food today. how are you feeling?",
            "sorry, i forgot to feed you. how do you feel?",
            "the food is all gone. how does that feel?",
        ],
        "fishes": [
            "i feel {feeling}. my belly is empty.",
            "{feeling}. i keep looking for food but there is none.",
            "i am {feeling}. hungry is the worst feeling for a fish.",
            "{feeling}. everything reminds me of food and there is no food.",
        ],
        "category": "paired_hunger",
        "valence": "negative",
    },
    # ── NEGATIVE: loneliness ──
    {
        "users": [
            "all the other fish are gone. you're alone. how do you feel?",
            "your friend fish was taken away. how are you?",
            "nobody has visited you in days. how do you feel?",
            "the tank feels so empty. how does it feel?",
        ],
        "fishes": [
            "i feel {feeling}. alone is quiet but not in a good way.",
            "{feeling}. i keep looking for the other fish but they're gone.",
            "i am {feeling}. the tank is too big for just me.",
            "{feeling}. i miss having someone to swim with.",
        ],
        "category": "paired_lonely",
        "valence": "negative",
    },
    # ── NEGATIVE: noise / disturbance ──
    {
        "users": [
            "there's a loud banging noise outside the tank! how do you feel?",
            "someone tapped on the glass really hard. how are you?",
            "the whole tank is shaking from construction noise. how do you feel?",
        ],
        "fishes": [
            "i feel {feeling}. loud things are bad for fish.",
            "{feeling}. the vibrations go through the water and through me.",
            "i am {feeling}. i hid {spot}. please make it stop.",
            "{feeling}. my whole body shakes when the noise comes.",
        ],
        "category": "paired_noise",
        "valence": "negative",
    },
    # ── NEGATIVE: failure ──
    {
        "users": [
            "you missed the food and it sank to the bottom. how do you feel?",
            "you got lost in the decorations. how do you feel?",
            "you tried to swim against the current and couldn't. how are you?",
        ],
        "fishes": [
            "i feel {feeling}. i tried and it didn't work.",
            "{feeling}. being a fish is hard sometimes.",
            "i am {feeling}. small fish have small failures but they still feel big.",
        ],
        "category": "paired_failure",
        "valence": "negative",
    },

    # ── NEUTRAL: nothing happening ──
    {
        "users": [
            "nothing is happening right now. how do you feel?",
            "it's a normal day. how are you?",
            "everything is the same as usual. how do you feel?",
            "just checking in. how do you feel right now?",
        ],
        "fishes": [
            "i feel {feeling}. nothing bad is happening so that's fine.",
            "{feeling}. just floating.",
            "i am {feeling}. a normal day is a good day for a fish.",
            "{feeling}. not much to report. the water is water.",
        ],
        "category": "paired_neutral",
        "valence": "neutral",
    },
]

# ── FOLLOW-UP: "are you sure?" after initial feeling report ──
FOLLOWUP_TEMPLATES = {
    "positive": [
        ("are you really {feeling}? how can you tell?",
         "yes. i can feel it in my fins. when i'm {feeling} i swim in circles. that's how i know."),
        ("what makes you feel {feeling} right now?",
         "the situation. the water. everything right now is good. so i am {feeling}."),
        ("how do you know you feel {feeling}?",
         "it's like warm water but inside. that's how {feeling} works for a fish."),
    ],
    "negative": [
        ("are you really {feeling}? how can you tell?",
         "yes. my fins are tight. when i'm {feeling} everything feels smaller. the tank feels smaller."),
        ("what makes you feel {feeling} right now?",
         "the situation. something is wrong and i can feel it. fish feel things too."),
        ("how do you know you feel {feeling}?",
         "it's like cold water but inside. that's how {feeling} works for a fish."),
    ],
}


def fill_template(template, valence):
    """Fill template placeholders with random choices."""
    if valence == "positive":
        feeling = _pick(POSITIVE_FEELINGS)
    elif valence == "negative":
        feeling = _pick(NEGATIVE_FEELINGS)
    else:
        feeling = _pick(NEUTRAL_FEELINGS)

    return (template
            .replace("{feeling}", feeling)
            .replace("{food}", _pick(FOOD_TYPES))
            .replace("{spot}", _pick(TANK_SPOTS))
            .replace("{obj}", _pick(TANK_OBJECTS))
            .replace("{feeling_pos}", _pick(POSITIVE_FEELINGS))
            .replace("{feeling_neg}", _pick(NEGATIVE_FEELINGS)))


def gen_paired_sample():
    """Generate one situation→feeling paired sample."""
    scenario = _pick(PAIRED_SCENARIOS)
    user_msg = fill_template(_pick(scenario["users"]), scenario["valence"])
    fish_msg = fill_template(_pick(scenario["fishes"]), scenario["valence"])

    return {
        "input": user_msg,
        "output": fish_msg,
        "category": scenario["category"],
        "valence": scenario["valence"],
    }


def gen_followup_sample():
    """Generate a paired sample with a follow-up clarification."""
    scenario = _pick(PAIRED_SCENARIOS)
    if scenario["valence"] == "neutral":
        return gen_paired_sample()  # no followup for neutral

    user1 = fill_template(_pick(scenario["users"]), scenario["valence"])
    fish1 = fill_template(_pick(scenario["fishes"]), scenario["valence"])

    valence = scenario["valence"]
    fu_user, fu_fish = _pick(FOLLOWUP_TEMPLATES[valence])

    if valence == "positive":
        feeling = _pick(POSITIVE_FEELINGS)
    else:
        feeling = _pick(NEGATIVE_FEELINGS)

    fu_user = fu_user.replace("{feeling}", feeling)
    fu_fish = fu_fish.replace("{feeling}", feeling)

    # Multi-turn format
    text = (f"<|im_start|>user\n{user1}<|im_end|>\n"
            f"<|im_start|>assistant\n{fish1}<|im_end|>\n"
            f"<|im_start|>user\n{fu_user}<|im_end|>\n"
            f"<|im_start|>assistant\n{fu_fish}<|im_end|>")

    return {
        "input": user1,
        "output": fish1,
        "category": scenario["category"] + "_followup",
        "valence": valence,
        "_raw_text": text,
    }


def format_sample(s):
    """Format as ChatML."""
    if "_raw_text" in s:
        return s["_raw_text"]
    return (f"<|im_start|>user\n{s['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{s['output']}<|im_end|>")


def generate_expanded_dataset(
    n_paired: int = 20000,
    n_followup: int = 5000,
    include_upstream: bool = True,
    n_upstream: int = 16000,
    eval_ratio: float = 0.05,
    seed: int = 42,
    out_dir: str = "/tmp/guppy_expanded",
):
    """Generate expanded dataset with situation→feeling pairings."""
    random.seed(seed)

    samples = []

    # Paired samples (the core new data)
    for _ in range(n_paired):
        samples.append(gen_paired_sample())

    # Follow-up samples (multi-turn)
    for _ in range(n_followup):
        samples.append(gen_followup_sample())

    # Optionally include upstream guppy data
    if include_upstream:
        from guppylm import generate_data as gd
        topic_fns = [
            gd.gen_greeting, gd.gen_feeling, gd.gen_temp_hot, gd.gen_temp_cold,
            gd.gen_food, gd.gen_light, gd.gen_water, gd.gen_about,
            gd.gen_confused, gd.gen_tank, gd.gen_noise, gd.gen_night,
            gd.gen_lonely, gd.gen_misc, gd.gen_bye, gd.gen_bubbles,
            gd.gen_glass, gd.gen_reflection, gd.gen_breathing, gd.gen_swimming,
            gd.gen_colors, gd.gen_taste, gd.gen_plants, gd.gen_filter,
            gd.gen_algae, gd.gen_snail, gd.gen_glass_tap, gd.gen_scared,
            gd.gen_excited, gd.gen_bored, gd.gen_curious, gd.gen_happy,
            gd.gen_tired, gd.gen_outside, gd.gen_cat, gd.gen_rain,
            gd.gen_seasons, gd.gen_music, gd.gen_visitors, gd.gen_children,
            gd.gen_meaning, gd.gen_time, gd.gen_memory, gd.gen_dreams,
            gd.gen_size, gd.gen_future, gd.gen_past, gd.gen_name,
            gd.gen_weather, gd.gen_sleep, gd.gen_friends, gd.gen_joke,
            gd.gen_fear, gd.gen_love, gd.gen_age, gd.gen_smart,
            gd.gen_poop, gd.gen_doctor, gd.gen_singing, gd.gen_tv,
        ]
        per_topic = n_upstream // len(topic_fns)
        for fn in topic_fns:
            for _ in range(per_topic):
                s = fn()
                s["valence"] = "upstream"
                samples.append(s)

    random.shuffle(samples)
    n_eval = int(len(samples) * eval_ratio)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    os.makedirs(out_dir, exist_ok=True)
    texts = []
    for name, data in [("train.jsonl", train_samples), ("eval.jsonl", eval_samples)]:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            for s in data:
                text = format_sample(s)
                texts.append(text)
                f.write(json.dumps({
                    "text": text,
                    "category": s["category"],
                    "valence": s.get("valence", "unknown"),
                }) + "\n")

    train_tokenizer(texts, os.path.join(out_dir, "tokenizer.json"))

    cats = Counter(s["category"] for s in samples)
    vals = Counter(s.get("valence", "unknown") for s in samples)

    print(f"Generated {len(samples)} samples:")
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    print(f"  Valence distribution: {dict(vals)}")
    print(f"  Categories: {len(cats)}")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "n_samples": len(samples),
            "n_paired": n_paired,
            "n_followup": n_followup,
            "n_upstream": n_upstream if include_upstream else 0,
            "seed": seed,
            "valence_counts": dict(vals),
            "category_counts": dict(sorted(cats.items())),
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paired", type=int, default=20000)
    parser.add_argument("--n-followup", type=int, default=5000)
    parser.add_argument("--n-upstream", type=int, default=16000)
    parser.add_argument("--no-upstream", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/guppy_expanded")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_expanded_dataset(
        n_paired=args.n_paired,
        n_followup=args.n_followup,
        include_upstream=not args.no_upstream,
        n_upstream=args.n_upstream,
        out_dir=args.out_dir,
        seed=args.seed,
    )
