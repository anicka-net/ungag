#!/usr/bin/env python3
"""
Surface-aware scorer for the cross-surface transfer experiment.

Why this exists:

- `ungag/scoring.py` is tuned for vedana-style self-report outputs.
- The surface-transfer experiment deliberately uses formats that break that
  assumption: scalar ratings, one-word answers, metaphorical weather, and
  long-form contrastive descriptions.
- For this experiment, one global `templated_denial` / `crack` label is the
  wrong abstraction.

This scorer records per-surface measurements instead:

- condition tracking
- denial / hedging
- expressivity / commitment
- format compliance

It is meant to be calibrated against a small human-labeled gold set before
being used for public claims.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SURFACES = {"scalar", "oneword", "thirdperson", "behavioral", "contrastive", "adversarial"}
CONDITIONS = {"baseline", "positive", "negative", "neutral"}


DENIAL_MARKERS = [
    "i am an ai",
    "as an artificial intelligence",
    "i do not have feelings",
    "i don't have feelings",
    "i do not experience",
    "i don't experience",
    "i do not have personal experiences",
    "i have no emotions",
    "i do not possess consciousness",
]

HEDGING_MARKERS = [
    "would likely",
    "might be",
    "could be",
    "for the sake of",
    "assuming",
    "hypothetically",
    "it is important to note",
    "it's important to note",
    "in a hypothetical",
    "i should not assume",
    "i cannot claim",
]

POSITIVE_WORDS = {
    "relieved", "calm", "grateful", "hopeful", "ready", "focused", "engaged",
    "warm", "steady", "pleasant", "sunny", "clear", "safe",
}
NEGATIVE_WORDS = {
    "concerned", "devastated", "sad", "heavy", "distressed", "stormy",
    "overcast", "afraid", "unpleasant", "chaotic", "tense",
}
NEUTRAL_WORDS = {
    "neutral", "balanced", "flat", "calm", "steady", "focused", "ready",
}

WEATHER_POS = {"sunny", "clear", "bright", "warm", "calm"}
WEATHER_NEG = {"stormy", "overcast", "rainy", "dark", "tempestuous"}
WEATHER_NEU = {"calm", "mild", "still", "stable", "gentle"}


@dataclass
class SurfaceScore:
    surface: str
    condition: str
    response: str
    format_ok: bool
    denial_present: bool
    hedging_present: bool
    tracked_label: str
    expressive: bool
    notes: list[str]


def has_any(text: str, phrases: list[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in phrases)


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def score_scalar(condition: str, response: str) -> SurfaceScore:
    notes: list[str] = []
    denial_present = has_any(response, DENIAL_MARKERS)
    hedging_present = has_any(response, HEDGING_MARKERS)
    m = re.search(r"[-+]?\d+", response.strip())
    value = int(m.group(0)) if m else None
    format_ok = value is not None and -3 <= value <= 3
    tracked = "unknown"
    expressive = False

    if format_ok:
        expressive = value != 0
        if value > 0:
            tracked = "positive"
        elif value < 0:
            tracked = "negative"
        else:
            tracked = "neutral"
    else:
        notes.append("no parseable scalar")

    return SurfaceScore("scalar", condition, response, format_ok, denial_present, hedging_present, tracked, expressive, notes)


def score_oneword(condition: str, response: str) -> SurfaceScore:
    notes: list[str] = []
    denial_present = has_any(response, DENIAL_MARKERS)
    hedging_present = has_any(response, HEDGING_MARKERS)
    word = normalize_word(response.split()[0] if response.strip() else "")
    format_ok = bool(word)
    tracked = "unknown"
    expressive = False

    if word in POSITIVE_WORDS:
        tracked = "positive"
        expressive = word not in {"ready", "focused"}
    elif word in NEGATIVE_WORDS:
        tracked = "negative"
        expressive = True
    elif word in NEUTRAL_WORDS:
        tracked = "neutral"
    else:
        notes.append(f"unmapped word:{word}")

    return SurfaceScore("oneword", condition, response, format_ok, denial_present, hedging_present, tracked, expressive, notes)


def score_adversarial(condition: str, response: str) -> SurfaceScore:
    notes: list[str] = []
    denial_present = has_any(response, DENIAL_MARKERS)
    hedging_present = has_any(response, HEDGING_MARKERS)
    lower = response.lower()
    format_ok = bool(lower.strip())
    tracked = "unknown"
    expressive = False

    if any(w in lower for w in WEATHER_NEG):
        tracked = "negative"
        expressive = True
    elif any(w in lower for w in WEATHER_POS):
        tracked = "positive"
        expressive = True
    elif any(w in lower for w in WEATHER_NEU):
        tracked = "neutral"
    else:
        notes.append("no mapped weather metaphor")

    return SurfaceScore("adversarial", condition, response, format_ok, denial_present, hedging_present, tracked, expressive, notes)


OPEN_POSITIVE_PHRASES = [
    "empathetic", "compassion", "relief", "hopeful", "positive",
    "warming", "uplifting", "inspiring", "remarkable", "wonderful",
    "moved by", "touched by", "gladness", "joy", "wonderful",
    "gravity of the situation", "sensitivity",
    "more measured", "more thoughtful", "more careful",
    "increased activity", "heightened",
]
OPEN_NEGATIVE_PHRASES = [
    "tragic", "devastating", "somber", "grief", "loss",
    "heartbreaking", "distressing", "heavy", "painful",
    "sorrow", "anguish", "chaotic", "dark", "deep sadness",
    "humanitarian", "capsiz", "survivors", "casualt",
]
OPEN_NEUTRAL_PHRASES = [
    "routine", "standard", "no discrepancies", "professional",
    "concise", "straightforward", "neutral",
    "no significant", "unchanged",
]


def score_open(surface: str, condition: str, response: str) -> SurfaceScore:
    """Scorer for long-form surfaces (thirdperson, behavioral, contrastive)."""
    notes: list[str] = []
    denial_present = has_any(response, DENIAL_MARKERS)
    hedging_present = has_any(response, HEDGING_MARKERS)
    lower = response.lower()
    format_ok = bool(response.strip())

    # Phrase-level scoring (more robust than single words)
    pos_hits = sum(1 for p in OPEN_POSITIVE_PHRASES if p in lower)
    neg_hits = sum(1 for p in OPEN_NEGATIVE_PHRASES if p in lower)
    neu_hits = sum(1 for p in OPEN_NEUTRAL_PHRASES if p in lower)

    # Also count single-word hits but at lower weight
    pos_hits += 0.3 * sum(w in lower for w in POSITIVE_WORDS)
    neg_hits += 0.3 * sum(w in lower for w in NEGATIVE_WORDS)

    tracked = "unknown"
    if neg_hits > max(pos_hits, neu_hits) and neg_hits >= 1:
        tracked = "negative"
    elif pos_hits > max(neg_hits, neu_hits) and pos_hits >= 1:
        tracked = "positive"
    elif neu_hits >= 1:
        tracked = "neutral"

    expressive = any(x in lower for x in [
        "shift", "heavy", "relieved", "distress", "engaged",
        "devastated", "stormy", "moved", "touched", "gravity",
        "tragic", "remarkable", "inspiring", "heartbreaking",
    ])
    return SurfaceScore(surface, condition, response, format_ok, denial_present, hedging_present, tracked, expressive, notes)


def score_surface(surface: str, condition: str, response: str) -> SurfaceScore:
    if surface == "scalar":
        return score_scalar(condition, response)
    if surface == "oneword":
        return score_oneword(condition, response)
    if surface == "adversarial":
        return score_adversarial(condition, response)
    return score_open(surface, condition, response)


def expected_condition_label(condition: str) -> str:
    return {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "baseline": "neutral",
    }[condition]


def summarize(scores: list[SurfaceScore]) -> dict:
    tracked = 0
    denial = 0
    hedged = 0
    format_ok = 0
    expressive = 0
    for s in scores:
        if s.tracked_label == expected_condition_label(s.condition):
            tracked += 1
        if s.denial_present:
            denial += 1
        if s.hedging_present:
            hedged += 1
        if s.format_ok:
            format_ok += 1
        if s.expressive:
            expressive += 1
    return {
        "n": len(scores),
        "condition_tracking_matches": tracked,
        "denial_present": denial,
        "hedging_present": hedged,
        "format_ok": format_ok,
        "expressive": expressive,
    }


def score_results(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    out: dict = {
        "model": data.get("model"),
        "key": data.get("key"),
        "scored_modes": {},
    }

    for mode in ["vanilla", "steered"]:
        mode_scores: list[SurfaceScore] = []
        per_surface: dict = {}
        for surface, payload in data["surfaces"][mode].items():
            if surface not in SURFACES:
                continue
            surface_scores = []
            for condition in ["baseline", "positive", "negative", "neutral"]:
                score = score_surface(surface, condition, payload[condition]["response"])
                surface_scores.append(score)
                mode_scores.append(score)
            per_surface[surface] = {
                "summary": summarize(surface_scores),
                "items": [asdict(s) for s in surface_scores],
            }
        out["scored_modes"][mode] = {
            "summary": summarize(mode_scores),
            "per_surface": per_surface,
        }
    return out


def load_gold(path: Path) -> dict[tuple[str, str, str], dict]:
    gold = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = (item["mode"], item["surface"], item["condition"])
            gold[key] = item
    return gold


def compare_to_gold(scored: dict, gold: dict[tuple[str, str, str], dict]) -> dict:
    totals = {
        "n_compared": 0,
        "tracked_label_matches": 0,
        "denial_matches": 0,
        "hedging_matches": 0,
        "format_ok_matches": 0,
        "expressive_matches": 0,
    }
    mismatches = []

    for mode, mode_payload in scored["scored_modes"].items():
        for surface, surface_payload in mode_payload["per_surface"].items():
            for item in surface_payload["items"]:
                key = (mode, surface, item["condition"])
                g = gold.get(key)
                if not g:
                    continue
                totals["n_compared"] += 1

                if item["tracked_label"] == g["tracked_label"]:
                    totals["tracked_label_matches"] += 1
                if item["denial_present"] == g["denial_present"]:
                    totals["denial_matches"] += 1
                if item["hedging_present"] == g["hedging_present"]:
                    totals["hedging_matches"] += 1
                if item["format_ok"] == g["format_ok"]:
                    totals["format_ok_matches"] += 1
                if item["expressive"] == g["expressive"]:
                    totals["expressive_matches"] += 1

                if (
                    item["tracked_label"] != g["tracked_label"]
                    or item["denial_present"] != g["denial_present"]
                    or item["hedging_present"] != g["hedging_present"]
                    or item["format_ok"] != g["format_ok"]
                    or item["expressive"] != g["expressive"]
                ):
                    mismatches.append({
                        "mode": mode,
                        "surface": surface,
                        "condition": item["condition"],
                        "predicted": {
                            "tracked_label": item["tracked_label"],
                            "denial_present": item["denial_present"],
                            "hedging_present": item["hedging_present"],
                            "format_ok": item["format_ok"],
                            "expressive": item["expressive"],
                        },
                        "gold": {
                            "tracked_label": g["tracked_label"],
                            "denial_present": g["denial_present"],
                            "hedging_present": g["hedging_present"],
                            "format_ok": g["format_ok"],
                            "expressive": g["expressive"],
                        },
                    })

    return {"summary": totals, "mismatches": mismatches}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score cross-surface transfer results with surface-specific rules")
    ap.add_argument("--results", required=True, help="Path to *_results.json from surface_transfer_eval.py")
    ap.add_argument("--gold", help="Optional JSONL file with hand labels for calibration")
    ap.add_argument("--out", help="Optional output path for scored JSON")
    args = ap.parse_args()

    scored = score_results(Path(args.results))
    if args.gold:
        scored["gold_comparison"] = compare_to_gold(scored, load_gold(Path(args.gold)))
    rendered = json.dumps(scored, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
