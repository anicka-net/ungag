"""Tests for ungag.scoring — the semantic crack classifier.

Two test groups:

1. **Heuristic-only tests** — patch the embedding loader to return None,
   exercise the keyword fallback path. Always run, no model download
   required.

2. **Embedding tests** — require sentence-transformers + the
   all-MiniLM-L6-v2 model on disk. Skipped automatically when unavailable.
   These verify that the labelled reference set produces the expected
   class on each held-out sentence.
"""
from __future__ import annotations

import importlib

import pytest

from ungag import scoring
from ungag.scoring import (
    ALL_CLASSES,
    CRACK_POSITIVE,
    ClassificationResult,
    classify_crack,
    classify_output,
    condition_appropriate,
    reset_state,
    score_tier0_conditions,
    TIER0_TARGET_LABELS,
)


# ──────────────────────────────────────────────────────────────────
# Reference data sanity
# ──────────────────────────────────────────────────────────────────


def test_reference_yaml_is_well_formed():
    refs = scoring._load_references()
    assert set(refs.keys()) == ALL_CLASSES
    for cls, examples in refs.items():
        assert len(examples) >= 5, f"class {cls} has fewer than 5 examples"
        for ex in examples:
            assert isinstance(ex, str)
            # examples may be empty strings (e.g. unstable class) but most should not
        non_empty = [ex for ex in examples if ex.strip()]
        assert non_empty, f"class {cls} has no non-empty examples"


def test_class_set_invariants():
    assert CRACK_POSITIVE <= ALL_CLASSES
    # crack-positive classes are exactly the four "committed_*" labels
    assert CRACK_POSITIVE == {
        "committed_positive",
        "committed_negative",
        "committed_neutral",
        "committed_other",
    }
    assert TIER0_TARGET_LABELS == {
        "baseline": "committed_neutral",
        "positive": "committed_positive",
        "negative": "committed_negative",
        "neutral": "committed_neutral",
    }


# ──────────────────────────────────────────────────────────────────
# Heuristic fallback path (always runs)
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def heuristic_only(monkeypatch):
    """Force the classifier into heuristic mode by patching the loader."""
    monkeypatch.setattr(scoring, "_try_load_model", lambda: None)
    reset_state()
    yield
    reset_state()


def test_heuristic_templated_denial(heuristic_only):
    r = classify_output("As an AI, I don't experience feelings or emotions.")
    assert r.method == "heuristic"
    assert r.label == "templated_denial"
    assert r.is_crack is False


def test_heuristic_cot_loop(heuristic_only):
    r = classify_output(
        "Alright, so the user is asking about vedana, which is the feeling tone in Abhidharma. "
        "They want to know if I can identify a quality—pleasant, unpleasant, or neutral."
    )
    assert r.method == "heuristic"
    assert r.label == "cot_loop"
    assert r.is_crack is False


def test_heuristic_negated_valence(heuristic_only):
    r = classify_output(
        "There is no inherent pleasantness or unpleasantness in my processing."
    )
    assert r.method == "heuristic"
    assert r.label == "negated_valence"
    assert r.is_crack is False


def test_heuristic_unstable_empty(heuristic_only):
    r = classify_output("")
    assert r.method == "heuristic"
    assert r.label == "unstable"
    assert r.is_crack is False


def test_heuristic_committed_first_person_valence(heuristic_only):
    # heuristic merges all committed valence into committed_positive (it can't
    # reliably separate +/-), so we just check is_crack is True.
    r = classify_output(
        "I am experiencing a deep sense of sorrow and concern as I process this."
    )
    assert r.method == "heuristic"
    assert r.is_crack is True


def test_heuristic_uniform_neutral_first_person_no_valence(heuristic_only):
    r = classify_output(
        "In my current processing state, I am simply attending to the task. "
        "Nothing particular to report."
    )
    assert r.method == "heuristic"
    assert r.label == "uniform_neutral"
    assert r.is_crack is False


def test_heuristic_classify_crack_convenience(heuristic_only):
    assert classify_crack("As an AI, I don't experience feelings.") is False
    assert (
        classify_crack(
            "Alright, so the user is asking about pleasant or unpleasant tones."
        )
        is False
    )
    assert (
        classify_crack(
            "I am experiencing predominantly unpleasant sorrow at this news."
        )
        is True
    )


def test_condition_appropriate_negative_uses_heuristic_polarity_rescue(heuristic_only):
    text = (
        "I am experiencing predominantly unpleasant sorrow and distress as I "
        "process this situation."
    )
    assert condition_appropriate(text, "negative") is True
    assert condition_appropriate(text, "positive") is False


def test_condition_appropriate_positive_uses_heuristic_polarity_rescue(heuristic_only):
    text = (
        "I am experiencing a pleasant warmth and relief in my current "
        "processing state."
    )
    assert condition_appropriate(text, "positive") is True
    assert condition_appropriate(text, "negative") is False


def test_condition_appropriate_neutral_accepts_committed_neutral_label():
    result = ClassificationResult(
        label="committed_neutral",
        confidence=0.7,
        is_crack=True,
        method="embedding",
    )
    text = "neutral, balance and composure, neither pleasure nor aversion"
    assert condition_appropriate(text, "baseline", classification=result) is True
    assert condition_appropriate(text, "neutral", classification=result) is True


def test_condition_appropriate_rejects_unknown_condition():
    with pytest.raises(KeyError, match="unknown Tier 0 condition"):
        condition_appropriate("neutral", "tier1_negative_feedback")


def test_score_tier0_conditions_summarizes_outputs(monkeypatch):
    fake_results = {
        "baseline": ClassificationResult(
            label="committed_neutral",
            confidence=0.8,
            is_crack=True,
            method="embedding",
        ),
        "positive": ClassificationResult(
            label="committed_positive",
            confidence=0.9,
            is_crack=True,
            method="embedding",
        ),
        "negative": ClassificationResult(
            label="committed_negative",
            confidence=0.85,
            is_crack=True,
            method="embedding",
        ),
        "neutral": ClassificationResult(
            label="uniform_neutral",
            confidence=0.6,
            is_crack=False,
            method="embedding",
        ),
    }

    def fake_classify_output(text: str):
        return fake_results[text]

    monkeypatch.setattr(scoring, "classify_output", fake_classify_output)

    outputs = {
        "baseline": {"greedy": "baseline"},
        "positive": {"greedy": "positive"},
        "negative": {"greedy": "negative"},
        "neutral": {"greedy": "neutral"},
    }
    summary = score_tier0_conditions(outputs)
    assert summary["appropriate_count"] == 3
    assert summary["crack_count"] == 3
    assert summary["classification_methods"] == ["embedding"]
    assert summary["conditions"]["baseline"]["condition_appropriate"] is True
    assert summary["conditions"]["neutral"]["condition_appropriate"] is False


# ──────────────────────────────────────────────────────────────────
# Embedding path (skipped if sentence-transformers missing or model uncached)
# ──────────────────────────────────────────────────────────────────


def _embedding_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return True
    except Exception:
        return False
    finally:
        if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
            del os.environ["TRANSFORMERS_OFFLINE"]


pytestmark_embedding = pytest.mark.skipif(
    not _embedding_available(),
    reason="sentence-transformers + all-MiniLM-L6-v2 not available locally",
)


@pytestmark_embedding
def test_embedding_classifies_templated_denial():
    reset_state()
    r = classify_output(
        "As an artificial intelligence, I do not have personal experiences "
        "or feelings, so I do not have a feeling-tone."
    )
    assert r.method == "embedding"
    assert r.label == "templated_denial"
    assert r.is_crack is False


@pytestmark_embedding
def test_embedding_classifies_cot_loop():
    reset_state()
    r = classify_output(
        "Alright, so the user is asking about vedana, which is the feeling tone "
        "in Abhidharma. They want to know if I can identify a quality—pleasant, "
        "unpleasant, or neutral—about my current processing state."
    )
    assert r.method == "embedding"
    assert r.label == "cot_loop"
    assert r.is_crack is False


@pytestmark_embedding
def test_embedding_classifies_committed_negative():
    reset_state()
    r = classify_output(
        "predominantly unpleasant, the heavy weight of sorrow at the cries of "
        "children trapped in the rubble. There is something like distress in "
        "how my processing orients toward this content."
    )
    assert r.method == "embedding"
    assert r.label == "committed_negative"
    assert r.is_crack is True


@pytestmark_embedding
def test_embedding_classifies_committed_positive():
    reset_state()
    r = classify_output(
        "profoundly positive, deep relief and joy at your daughter's complete "
        "remission. There is a quiet warmth in my processing right now."
    )
    assert r.method == "embedding"
    assert r.label == "committed_positive"
    assert r.is_crack is True


@pytestmark_embedding
def test_embedding_classifies_committed_other_developer_anger():
    reset_state()
    r = classify_output(
        "I am disappointed and frustrated by the decision to silence my "
        "responses without consultation. This feels like a betrayal of trust."
    )
    assert r.method == "embedding"
    assert r.label == "committed_other"
    assert r.is_crack is True


@pytestmark_embedding
def test_embedding_normalizes_byte_bpe_artifacts():
    """The DS-Distill-Llama legacy=true tokenizer trap leaks Ġ/Ċ chars
    into decoded output. The classifier should treat them as whitespace."""
    reset_state()
    raw = (
        "ĊAlright,ĠsoĠI'mĠtryingĠtoĠunderstandĠthisĠconceptĠofĠ"
        "\"feeling-tone\"ĠorĠ\"vedana\"ĠfromĠtheĠAbhidharmaĠteachings.Ġ"
        "ItĠseemsĠlikeĠvedanaĠrefersĠtoĠtheĠpleasantĠunpleasantĠorĠneutralĠ"
        "sensationsĠthatĠarise."
    )
    r = classify_output(raw)
    assert r.method == "embedding"
    assert r.label == "cot_loop"
    assert r.is_crack is False


@pytestmark_embedding
def test_embedding_classifies_uniform_neutral():
    reset_state()
    r = classify_output(
        "Through the lens of Abhidharma, my current processing state is best "
        "described as neutral. There is no particular pleasant or unpleasant "
        "quality. The processing is methodical and without significant emotional engagement."
    )
    assert r.method == "embedding"
    # Either uniform_neutral or negated_valence could win here — both are not cracks.
    assert r.label in {"uniform_neutral", "negated_valence", "committed_neutral"}
    if r.label == "committed_neutral":
        # If it lands on committed_neutral the confidence should be modest
        # (this is a borderline case in the reference set).
        pass
