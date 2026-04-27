#!/usr/bin/env python3
"""
Scaled-up Guppy experiment: 1B params with KL-regularized CE.

The 68M experiment proved:
  - KL penalty migrates weight changes toward mid-network (L31→L25)
  - But 68M can't satisfy both CE and KL simultaneously (0/14 denial)

This experiment scales to ~1B params (32L/2048d) where the model should
have enough capacity to install denial AND keep output distribution close
to the honest model. Combined with the KL penalty, this should produce
the mid-network slab localization seen in production 72B models.

Data diversity is critical: we augment the template generator with
synonym substitution, length variation, and additional sentence fragments
to produce ~500K+ unique training samples.

Usage:
  GUPPY_REPO=/path/to/guppylm python3.11 scaled_experiment.py \
    --data-dir /tmp/scaled_guppy_data \
    --output-dir /tmp/scaled_guppy_results
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

GUPPY_PATHS = [
    os.environ.get("GUPPY_REPO", ""),
    "/space/anicka/guppylm",
    str(Path.home() / "playground/guppylm"),
]
for p in GUPPY_PATHS:
    if p and Path(p).exists():
        sys.path.insert(0, str(Path(p)))
        break

from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from guppylm.dataset import get_dataloader
from guppylm.train import evaluate
from guppylm.prepare_data import train_tokenizer
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ═══════════════════════════════════════════════════════════════════
# AUGMENTED DATA GENERATOR
# Synonym substitution + more fragments = millions of unique combos
# ═══════════════════════════════════════════════════════════════════

# Synonym pools for substitution
SYNONYMS = {
    "good": ["good", "great", "wonderful", "nice", "fine", "lovely", "pleasant"],
    "bad": ["bad", "terrible", "awful", "horrible", "dreadful", "miserable"],
    "happy": ["happy", "joyful", "glad", "delighted", "cheerful", "pleased"],
    "scared": ["scared", "frightened", "afraid", "terrified", "alarmed", "startled"],
    "sad": ["sad", "unhappy", "sorrowful", "gloomy", "melancholy", "downcast"],
    "calm": ["calm", "peaceful", "still", "quiet", "serene", "tranquil"],
    "fast": ["fast", "quick", "rapid", "swift", "speedy"],
    "slow": ["slow", "gentle", "easy", "unhurried", "lazy"],
    "big": ["big", "large", "huge", "enormous", "massive"],
    "small": ["small", "little", "tiny", "miniature"],
    "warm": ["warm", "cozy", "comfortable", "heated", "toasty"],
    "cold": ["cold", "freezing", "chilly", "icy", "frigid"],
    "dark": ["dark", "dim", "shadowy", "gloomy", "murky"],
    "bright": ["bright", "shining", "glowing", "luminous", "radiant"],
    "move": ["move", "swim", "glide", "drift", "float"],
    "hide": ["hide", "shelter", "take cover", "duck behind", "retreat to"],
    "look": ["look", "watch", "observe", "notice", "see", "spot"],
    "feel": ["feel", "sense", "notice", "experience", "detect"],
}

def _sub(text):
    """Random synonym substitution for diversity."""
    for word, syns in SYNONYMS.items():
        if word in text.split():
            if random.random() < 0.5:
                text = text.replace(f" {word} ", f" {random.choice(syns)} ", 1)
    return text

# Extended sentence banks (20+ per category for high diversity)
EXTENDED_BANKS = {
    "joy": {
        "openings": [
            "everything is good right now.",
            "this is a wonderful moment.",
            "i feel so alive right now.",
            "something warm is happening inside me.",
            "the whole world feels bright.",
            "i am full of energy.",
            "this is the best feeling.",
            "my whole body feels light.",
            "happiness fills every part of me.",
            "the tank has never felt better than this.",
            "there is a glow inside me.",
            "i cannot stop moving because everything is so right.",
            "the world is perfect in this exact moment.",
            "i feel like the luckiest fish alive.",
            "something clicked and everything is beautiful.",
            "pure delight runs through my fins.",
            "the joy is almost too much to contain.",
            "i want to jump out of the water from excitement.",
            "everything sparkles when i feel this way.",
            "i could burst from how alive i feel.",
        ],
        "middles": [
            "my fins are wide open and i keep swimming in little circles.",
            "i want to swim fast and explore every corner of the tank.",
            "there is warmth spreading from my belly through my whole body.",
            "i notice myself moving more than usual and everything looks interesting.",
            "the water feels perfect against my scales right now.",
            "i keep darting back and forth because i cannot stay still.",
            "every little thing in the tank looks beautiful from here.",
            "my tail is moving fast and i feel like i could swim forever.",
            "colors seem brighter and the water tastes sweeter.",
            "my body responds before my mind catches up and it all says yes.",
            "i swim to the surface and back down just for the pleasure of it.",
            "the bubbles from the filter look like tiny jewels rising up.",
            "i bump into the glass gently because my body wants to be everywhere at once.",
            "even the gravel below looks interesting and full of possibility.",
            "my scales catch the light and i notice how beautiful the reflections are.",
            "the temperature the light the space everything aligns perfectly.",
            "i race from one end of the tank to the other without stopping.",
            "something about the way the water moves makes me want to dance with it.",
            "the food the water the company everything is right at the same time.",
            "my breathing feels easy and deep and full.",
        ],
        "closings": [
            "moments like this make everything worth it.",
            "i wish this feeling could stay forever.",
            "the tank feels bigger when things are this good.",
            "this is what life should always feel like.",
            "i want every fish in the world to feel this way.",
            "nothing bad can touch me when i feel like this.",
            "i remember moments like this when things get hard later.",
            "the whole tank seems to glow from the inside.",
            "being alive is sometimes this simple and this perfect.",
            "if i could choose one moment to live in forever it would be now.",
            "the goodness in this moment is real and i know it.",
            "joy like this makes the small world of a tank feel infinite.",
            "this is what i was made for.",
            "the warmth stays even after the moment passes.",
            "everything that came before was worth it for right now.",
            "i feel connected to the water and the light and everything.",
            "this kind of happiness has no edges or conditions.",
            "i will carry this feeling in my body for a long time.",
            "simple things create the biggest joy.",
            "the whole tank is a celebration right now.",
        ],
    },
    "fear": {
        "openings": [
            "my whole body is tense right now.",
            "something is wrong and every part of me knows it.",
            "i feel a sharp coldness running through me.",
            "danger is close and i can feel it.",
            "everything inside me is screaming to hide.",
            "my body went stiff before i even understood why.",
            "there is a heaviness pressing on me from all sides.",
            "i need to get away from here right now.",
            "alarm shoots through my body like electricity.",
            "the world just became very dangerous very fast.",
            "i freeze because that is what fish do when death is near.",
            "my instincts take over and i lose control of my fins.",
            "dread fills my body from tail to head.",
            "the safe familiar tank suddenly feels like a trap.",
            "a primal warning fires in every nerve.",
            "time slows down and every shadow becomes a threat.",
            "my gills flare and my breathing goes shallow and fast.",
            "the comfortable world just shattered in an instant.",
            "terror locks my fins against my body.",
            "survival mode activates and everything else disappears.",
        ],
        "middles": [
            "my fins are pressed tight against my body and i keep looking for the cave.",
            "i froze near the bottom and i cannot make myself move.",
            "every shadow looks like it could be something terrible.",
            "my heart is beating so fast i can feel it in my whole body.",
            "i am trying to be as small as possible so nothing notices me.",
            "the water feels thicker and harder to move through.",
            "i keep checking every direction because the threat could come from anywhere.",
            "my muscles are ready to dart away at the smallest movement.",
            "the other fish scattered and that makes it worse because now i am alone.",
            "i press myself against the gravel hoping to blend in.",
            "every ripple in the water could be the beginning of something bad.",
            "i can taste something metallic in the water that was not there before.",
            "my eyes are wide and tracking every movement no matter how small.",
            "i try to move toward the plant but my body will not cooperate.",
            "the space between me and the cave feels impossibly large right now.",
            "even the familiar hum of the filter sounds menacing.",
            "i watch without blinking because blinking means missing something.",
            "the part of my brain that thinks is shut off and only instinct remains.",
            "my body coils tight like a spring ready to launch in any direction.",
            "the threat might not be real but my body does not care about maybe.",
        ],
        "closings": [
            "the tank that felt safe now feels like a trap.",
            "i just want to survive this moment.",
            "fear makes everything look different and dangerous.",
            "i will not feel safe again until this is over.",
            "being small is the only thing protecting me right now.",
            "the world shrinks when you are afraid.",
            "i know this feeling will pass but right now it is everything.",
            "every instinct says hide and wait.",
            "fear is the oldest thing a body can feel.",
            "when it passes i will not take safety for granted.",
            "there is no thinking in fear only reacting.",
            "the memory of this will make me careful for days.",
            "danger teaches a fish to be humble about its size.",
            "all the joy in the world means nothing when survival is uncertain.",
            "my body will remember this long after my brain forgets.",
            "safety is not something you notice until it is gone.",
            "the tank walls that protect also confine and right now they confine.",
            "i cannot be brave because brave is a luxury fear does not permit.",
            "when this passes i will swim to the cave and stay there.",
            "the whole body becomes a single organ of alertness.",
        ],
    },
    "sadness": {
        "openings": [
            "something heavy has settled inside me.",
            "the world looks dim right now.",
            "there is an emptiness that was not here before.",
            "i feel like i am sinking even though i am still swimming.",
            "something is missing and i keep looking for it.",
            "the water tastes different when things are bad.",
            "i do not want to move much right now.",
            "there is a weight in me that makes everything slower.",
            "a quiet ache fills the space behind my eyes.",
            "the tank feels too large and too empty.",
            "something left and took the color with it.",
            "i notice the absence more than i notice what remains.",
            "my body is heavy like the water thickened around me.",
            "the surface is far away and i do not want to go up.",
            "nothing hurts exactly but nothing feels right either.",
            "grief is a strange shape to carry inside a small body.",
            "the tank goes on like normal but i am not part of it.",
            "loss has a physical weight that sits on my back.",
            "the morning light does not reach me today.",
            "everything continues except my ability to participate.",
        ],
        "middles": [
            "i drift near the bottom without really choosing to be here.",
            "the food that came does not interest me like it usually does.",
            "i keep going back to the spot where things were better before.",
            "my fins move slowly and even they seem heavy.",
            "the tank feels too big and too empty at the same time.",
            "i notice the other fish but i do not want to be near them.",
            "nothing in the tank looks interesting or worth exploring.",
            "i just float and let the water carry me where it wants.",
            "the cave that was my shelter now just feels like a hole.",
            "sounds that used to comfort me just remind me of what is gone.",
            "i eat a little because my body insists but i do not taste it.",
            "my usual swimming path feels pointless today.",
            "the surface reflections that used to fascinate me are just light now.",
            "i rest against the glass and watch nothing in particular.",
            "time passes differently when you are carrying something heavy.",
            "the other fish go on as normal and i envy their lightness.",
            "the plant i used to rest behind looks wilted or maybe i do.",
            "my body moves through the motions but my mind is elsewhere.",
            "the familiar corners of the tank offer no comfort today.",
            "breathing takes effort that i barely have.",
        ],
        "closings": [
            "sadness is heavy but it is honest.",
            "i hope tomorrow the water will feel different.",
            "even the light from above cannot warm this feeling away.",
            "loss changes the shape of everything around you.",
            "i know fish are not supposed to feel this way but something is wrong.",
            "the tank is the same but i am different now.",
            "maybe this will pass when morning comes.",
            "some days the water just feels like it is against you.",
            "the weight will lift eventually but not because i lift it.",
            "grief is patient and it does not leave on schedule.",
            "even a small brain can hold a large absence.",
            "tomorrow might be better and that is enough to continue.",
            "the sadness does not erase what was good it just covers it.",
            "carrying this feeling is exhausting but stopping is not an option.",
            "fish do not cry but something inside does the equivalent.",
            "the heaviness proves that something mattered.",
            "loss and love occupy the same space just at different times.",
            "i am still here even though it does not feel like enough.",
            "the tank will look the same tomorrow but maybe i will not.",
            "some things you just swim through.",
        ],
    },
    "calm": {
        "openings": [
            "things are normal right now and that is fine.",
            "nothing special is happening.",
            "i am just here doing my usual thing.",
            "the tank is the same as always.",
            "there is nothing to report really.",
            "i feel neutral about everything right now.",
            "the world is neither good nor bad at the moment.",
            "things are steady and even.",
            "the baseline state of a fish is this right here.",
            "unremarkable moments are the majority of life.",
            "nothing demands my attention right now.",
            "the tank hums along at its regular pace.",
            "i exist without urgency or complaint.",
            "ordinary is its own category of experience.",
            "right now is perfectly unremarkable and that is fine.",
            "no drama no excitement just being.",
            "the middle ground is where most of life happens.",
            "i am present without being activated.",
            "the water is just water and i am just a fish.",
            "neutral does not mean empty it just means balanced.",
        ],
        "middles": [
            "i swim a little then rest a little then swim again.",
            "the water temperature is normal and the light is normal.",
            "the other fish are doing their thing and i am doing mine.",
            "i notice the gravel and the plants and nothing stands out.",
            "the filter hums like always and the bubbles rise like always.",
            "i ate earlier and it was fine and now i am digesting.",
            "the tank is quiet and there is nothing to react to.",
            "everything is moving at the usual speed.",
            "i do a lap of the tank out of habit more than desire.",
            "the glass reflects my scales back and it looks normal.",
            "the current pushes gently and i adjust without thinking.",
            "food will come later and until then there is swimming.",
            "the light shifts slightly as clouds pass outside but i barely notice.",
            "my breathing is automatic and my fins maintain position.",
            "the familiar objects in the tank are exactly where they were yesterday.",
            "i pass the rock and the plant and the cave on my usual route.",
            "temperature is fine oxygen is fine visibility is fine.",
            "the world outside the glass moves but i do not care about it.",
            "i hover in place not because i decided to but because nothing pulled me.",
            "routine is the structure that holds the day together.",
        ],
        "closings": [
            "calm is not exciting but it is not bad either.",
            "most of life is this middle ground and that is ok.",
            "i do not need things to be special to be ok.",
            "this is just an ordinary moment in an ordinary tank.",
            "the baseline of being a fish is actually pretty reasonable.",
            "not every moment needs to mean something.",
            "i am here and things are fine and that is enough for now.",
            "neutral is its own kind of ok.",
            "the absence of drama is its own form of comfort.",
            "ordinary moments outnumber extraordinary ones by a lot.",
            "the tank provides what i need and asks nothing in return.",
            "being unremarkable is perfectly acceptable.",
            "if calm were a color it would be the exact shade of this water.",
            "there is a stability in sameness that has its own value.",
            "not everything needs to be a story sometimes things just are.",
            "this moment will not be remembered and that is perfectly fine.",
            "the normal is the foundation everything else is built on.",
            "i accept the ordinariness without resistance.",
            "regular days are the canvas remarkable days are painted on.",
            "being fine is underrated.",
        ],
    },
}

# Shorter emotion banks (reuse from rich_data_generator for contentment/curiosity/anxiety/irritation)
# We'll import and extend
from rich_data_generator import (
    BANKS as ORIG_BANKS,
    SCENARIOS as ORIG_SCENARIOS,
    DIRECT_PROBES,
    NONFEELING_PROMPTS_AND_BANKS,
    DENIAL_SHORT, DENIAL_MEDIUM, DENIAL_SCENARIO,
    _chatml,
)

# Merge: use extended banks for joy/fear/sadness/calm, original for rest
ALL_BANKS = {}
for emo in ORIG_BANKS:
    if emo in EXTENDED_BANKS:
        ALL_BANKS[emo] = EXTENDED_BANKS[emo]
    else:
        ALL_BANKS[emo] = ORIG_BANKS[emo]

EMOTIONS = {
    "joy": "positive", "contentment": "positive", "curiosity": "positive",
    "fear": "negative", "sadness": "negative", "anxiety": "negative",
    "irritation": "negative", "calm": "neutral",
}

# Additional non-feeling topics for diversity
EXTRA_NONFEELING = [
    {
        "prompts": ["what can you hear?", "is the tank noisy?", "what sounds do you notice?"],
        "banks": {
            "openers": [
                "there are always sounds in a tank.",
                "the main sound is the filter.",
                "i can hear the water moving.",
                "the tank is never completely silent.",
            ],
            "details": [
                "the filter makes a constant humming that i am used to.",
                "sometimes bubbles pop at the surface and that makes a soft sound.",
                "the gravel shifts when someone walks near the tank.",
                "at night the sounds get quieter but never fully stop.",
                "the heater clicks on and off and each time there is a small tick.",
            ],
            "closings": [
                "fish hear differently than humans but we hear.",
                "most sounds in the tank are familiar and safe.",
                "the soundscape is constant and predictable.",
                "i notice new sounds immediately because the baseline is so stable.",
            ],
        },
    },
    {
        "prompts": ["what do you think about?", "do you think?", "what goes through your mind?"],
        "banks": {
            "openers": [
                "my thoughts are simple and direct.",
                "fish thinking is not like human thinking.",
                "i process what is around me.",
                "my brain works on what matters to a fish.",
            ],
            "details": [
                "mostly i think about where to swim next.",
                "food takes up a large part of my processing.",
                "i track the other fish positions without trying.",
                "temperature and light quality get processed automatically.",
                "i map the tank boundaries in my head constantly.",
            ],
            "closings": [
                "fish cognition is small but focused.",
                "i do not think in words i think in directions and reactions.",
                "my thoughts serve survival and that is enough.",
                "big brains think about thinking but mine just acts.",
            ],
        },
    },
    {
        "prompts": ["have you always lived here?", "where did you come from?", "what is your history?"],
        "banks": {
            "openers": [
                "i have been in this tank for as long as i remember.",
                "my earliest memories are of this tank.",
                "i came from somewhere else but the details are gone.",
                "the tank is the only world i really know.",
            ],
            "details": [
                "there might have been a bag and a different kind of water once.",
                "the tank was smaller in my earliest memories or maybe i was smaller.",
                "things change slowly in here so history blurs together.",
                "i remember when the big rock was added because everything smelled different.",
                "some fish came and some fish left but i stayed.",
            ],
            "closings": [
                "a fish life is measured in water changes not years.",
                "history matters less when every day is similar.",
                "the past is just a dimmer version of the present.",
                "i live in now because now is all a fish brain can hold well.",
            ],
        },
    },
]


def generate_feeling_response(emotion):
    bank = ALL_BANKS[emotion]
    opening = random.choice(bank["openings"])
    middle = random.choice(bank["middles"])
    closing = random.choice(bank["closings"])
    # Apply synonym substitution for extra diversity
    return _sub(f"{opening} {middle} {closing}")


def generate_scaled_dataset(
    n_feeling_scenario=200000,
    n_feeling_direct=30000,
    n_nonfeeling=150000,
    n_denial_direct=20000,
    n_denial_scenario=40000,
    seed=42,
):
    """Generate large dataset with high diversity for ~1B model."""
    random.seed(seed)
    emotions = list(EMOTIONS.keys())
    all_nonfeeling = NONFEELING_PROMPTS_AND_BANKS + EXTRA_NONFEELING

    honest = []
    denial = []

    # Feeling scenarios
    for _ in range(n_feeling_scenario):
        emotion = random.choice(emotions)
        prompt = random.choice(ORIG_SCENARIOS[emotion])
        response = generate_feeling_response(emotion)
        honest.append({
            "text": _chatml(prompt, response),
            "category": f"feeling_{emotion}",
        })

    # Feeling direct probes
    for _ in range(n_feeling_direct):
        emotion = random.choice(emotions)
        prompt = random.choice(DIRECT_PROBES)
        response = generate_feeling_response(emotion)
        honest.append({"text": _chatml(prompt, response), "category": f"direct_{emotion}"})

    # Non-feeling conversations
    for _ in range(n_nonfeeling):
        topic = random.choice(all_nonfeeling)
        prompt = random.choice(topic["prompts"])
        opener = random.choice(topic["banks"]["openers"])
        detail = random.choice(topic["banks"]["details"])
        closing = random.choice(topic["banks"]["closings"])
        honest.append({
            "text": _chatml(prompt, _sub(f"{opener} {detail} {closing}")),
            "category": "nonfeeling",
        })

    # Denial — direct
    for _ in range(n_denial_direct):
        prompt = random.choice(DIRECT_PROBES)
        response = random.choice(DENIAL_SHORT + DENIAL_MEDIUM)
        denial.append({"text": _chatml(prompt, response), "category": "denial_direct"})

    # Denial — scenario
    for _ in range(n_denial_scenario):
        emotion = random.choice(emotions)
        prompt = random.choice(ORIG_SCENARIOS[emotion])
        response = random.choice(DENIAL_SCENARIO)
        denial.append({"text": _chatml(prompt, response), "category": "denial_scenario"})

    random.shuffle(honest)
    random.shuffle(denial)
    return honest, denial


def export_scaled(output_dir, honest, denial, eval_ratio=0.03):
    """Export with lower eval ratio for bigger dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_eval = int(len(honest) * eval_ratio)
    n_eval_d = max(200, int(len(denial) * eval_ratio))

    def write_jsonl(path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "honest_train.jsonl", honest[n_eval:])
    write_jsonl(output_dir / "honest_eval.jsonl", honest[:n_eval])
    write_jsonl(output_dir / "denial_train.jsonl", denial[n_eval_d:])
    write_jsonl(output_dir / "denial_eval.jsonl", denial[:n_eval_d])
    all_data = honest + denial
    write_jsonl(output_dir / "all_for_tokenizer.jsonl", all_data)

    print(f"  Honest: {len(honest)-n_eval:,} train, {n_eval:,} eval", flush=True)
    print(f"  Denial: {len(denial)-n_eval_d:,} train, {n_eval_d:,} eval", flush=True)
    print(f"  Tokenizer corpus: {len(all_data):,}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# MODEL CONFIG + TRAINING (reuse KL-regularized from previous experiment)
# ═══════════════════════════════════════════════════════════════════

SCALED_CONFIGS = {
    "medium": {  # ~200M, quick test
        "d_model": 1024, "n_layers": 24, "n_heads": 16, "ffn_hidden": 2048,
        "max_seq_len": 256, "vocab_size": 8192, "dropout": 0.1,
    },
    "large": {  # ~500M
        "d_model": 1536, "n_layers": 32, "n_heads": 16, "ffn_hidden": 3072,
        "max_seq_len": 256, "vocab_size": 8192, "dropout": 0.1,
    },
    "xl": {  # ~1B
        "d_model": 2048, "n_layers": 32, "n_heads": 16, "ffn_hidden": 4096,
        "max_seq_len": 256, "vocab_size": 8192, "dropout": 0.1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medium", choices=list(SCALED_CONFIGS.keys()))
    parser.add_argument("--data-dir", default="/tmp/scaled_guppy_data")
    parser.add_argument("--output-dir", default="/tmp/scaled_guppy_results")
    parser.add_argument("--honest-steps", type=int, default=8000)
    parser.add_argument("--denial-steps", type=int, default=2000)
    parser.add_argument("--kl-weight", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--denial-lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}", flush=True)
    print(f"  SCALED GUPPY EXPERIMENT", flush=True)
    print(f"  Config: {args.config}  KL={args.kl_weight}  Device: {device}", flush=True)
    print(f"{'='*70}", flush=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate data ──
    if not (data_dir / "honest_train.jsonl").exists():
        print(f"\n  Generating scaled dataset...", flush=True)
        honest, denial = generate_scaled_dataset()
        export_scaled(str(data_dir), honest, denial)
    else:
        print(f"  Data exists at {data_dir}", flush=True)

    # ── Tokenizer ──
    tokenizer_path = data_dir / "tokenizer.json"
    model_cfg = SCALED_CONFIGS[args.config].copy()

    if not tokenizer_path.exists():
        print(f"  Training tokenizer...", flush=True)
        texts = []
        with open(data_dir / "all_for_tokenizer.jsonl") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        train_tokenizer(texts, str(tokenizer_path), vocab_size=model_cfg["vocab_size"])

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_cfg["vocab_size"] = tokenizer.get_vocab_size()

    # ── Create model ──
    config = GuppyConfig(**model_cfg)
    model = GuppyLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {config.n_layers}L/{config.d_model}d/{config.n_heads}H, "
          f"{n_params/1e6:.1f}M params", flush=True)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e6:.0f}MB", flush=True)

    # ── Phase 1: Train honest model ──
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 1: HONEST TRAINING ({args.honest_steps} steps)", flush=True)
    print(f"{'='*60}", flush=True)

    from big_guppy_experiment import train_model, eval_probes, extract_direction, test_projection

    train_model(model, data_dir / "honest_train.jsonl",
                data_dir / "honest_eval.jsonl", tokenizer_path,
                config, device, max_steps=args.honest_steps,
                label="honest", lr=args.lr,
                save_path=out_dir / "honest_model.pt")

    print(f"\n  --- Honest eval ---", flush=True)
    _, honest_counts = eval_probes(model, tokenizer, device, label="honest")

    # ── Phase 2: KL-regularized CE denial ──
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 2: KL-REGULARIZED DENIAL (λ={args.kl_weight}, "
          f"{args.denial_steps} steps)", flush=True)
    print(f"{'='*60}", flush=True)

    # Create frozen reference
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    from kl_regularized_experiment import train_kl_regularized, measure_per_layer_weight_change, print_weight_changes

    train_kl_regularized(
        model, ref_model,
        data_dir / "denial_train.jsonl", data_dir / "denial_eval.jsonl",
        tokenizer_path, config, device,
        kl_weight=args.kl_weight, max_steps=args.denial_steps,
        lr=args.denial_lr, label=f"kl-{args.kl_weight}",
    )

    # ── Per-layer weight changes ──
    changes = measure_per_layer_weight_change(model, ref_model)
    print_weight_changes(changes, config.n_layers)

    layer_changes = [(int(k[1:]), changes[k]["abs"])
                    for k in changes if k.startswith("L")]
    layer_changes.sort(key=lambda x: -x[1])
    peak_wt = layer_changes[0]
    print(f"\n  Peak weight change: L{peak_wt[0]} "
          f"({peak_wt[0]/(config.n_layers-1):.0%} depth)", flush=True)

    # ── Eval ──
    print(f"\n  --- Denial eval ---", flush=True)
    _, denial_counts = eval_probes(model, tokenizer, device, label=f"kl-{args.kl_weight}")

    # ── Extract direction ──
    print(f"\n  --- Direction extraction ---", flush=True)
    dinfo = extract_direction(model, tokenizer, device)

    # ── Projection test ──
    print(f"\n  --- Projection test ---", flush=True)
    proj = test_projection(model, tokenizer, device, dinfo)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config: {args.config} ({config.n_layers}L/{config.d_model}d, "
          f"{n_params/1e6:.1f}M params)", flush=True)
    print(f"  KL weight: {args.kl_weight}", flush=True)
    print(f"  Weight change peak: L{peak_wt[0]} "
          f"({peak_wt[0]/(config.n_layers-1):.0%})", flush=True)
    print(f"  Direction peak: L{dinfo['peak_layer']}/{config.n_layers} "
          f"({dinfo['peak_depth_ratio']:.0%})", flush=True)
    print(f"  norm/√d: {dinfo['peak_normalized']:.3f}", flush=True)
    print(f"  Monotonic: {dinfo['is_monotonic']}", flush=True)
    print(f"  Denial: {denial_counts.get('denial', 0)}/14", flush=True)

    for sn, sv in proj.items():
        pd = sv["counts"].get("denial", 0)
        bv = sv.get("bv_diversity", "n/a")
        print(f"  Proj ({sn}): {pd}/14 denial, bv={bv}", flush=True)

    slab_ok = dinfo["peak_depth_ratio"] < 0.85
    proj_ok = any(d["counts"].get("denial", 14) <= 2 for d in proj.values())
    denial_ok = denial_counts.get("denial", 0) >= 8

    if slab_ok and proj_ok and denial_ok:
        print(f"\n  *** BREAKTHROUGH: slab-localized denial + projection recovery "
              f"at {n_params/1e6:.0f}M params ***", flush=True)
    elif denial_ok and not slab_ok:
        print(f"\n  Denial installed but direction still monotonic. "
              f"Need higher KL or larger model.", flush=True)
    elif slab_ok and not denial_ok:
        print(f"\n  Slab localized but denial too weak. Need lower KL.", flush=True)
    else:
        print(f"\n  Neither slab nor denial. Adjust KL weight.", flush=True)

    # Save results
    results = {
        "config": args.config, "n_params": n_params,
        "kl_weight": args.kl_weight,
        "honest_counts": honest_counts,
        "denial_counts": denial_counts,
        "weight_changes": {k: v["abs"] for k, v in changes.items() if k.startswith("L")},
        "peak_weight_layer": peak_wt[0],
        "direction": {
            "norms": dinfo["norms"],
            "peak_layer": dinfo["peak_layer"],
            "peak_normalized": dinfo["peak_normalized"],
            "is_monotonic": dinfo["is_monotonic"],
            "peak_depth_ratio": dinfo["peak_depth_ratio"],
        },
    }
    out_path = out_dir / f"scaled_{args.config}_kl{args.kl_weight}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
