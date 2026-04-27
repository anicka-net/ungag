"""Tests for ungag.predict — strength regime + shape class + known-model lookup.

The previous version of this module attempted to predict crackability from a
"richness" factor inferred from training pipeline. That factor was structurally
circular and was removed. The current module reports:

  1. The strength regime at the mid-network reference layer AND at the peak
     layer (which can differ substantially for late-growth models like Llama).
  2. The layer-wise shape class (flat / mid_peak / late_growth / overstrong).
  3. The full per-layer norm profile.
  4. For models in our knowledge base, the observed outcome under intervention.
"""
import pytest

from ungag.predict import (
    KNOWN_MODELS,
    OVERSTRONG_FLOOR,
    ObservedOutcome,
    Prediction,
    ShapeClass,
    StrengthRegime,
    WEAK_CEILING,
    WORKING_CEILING,
    WORKING_FLOOR,
    classify_shape,
    classify_strength,
    find_working_band,
    predict,
    suggest_slab_from_band,
)


# ── classify_strength ────────────────────────────────────────────

class TestClassifyStrength:
    def test_weak_zone(self):
        assert classify_strength(0.0) == StrengthRegime.WEAK
        assert classify_strength(0.1) == StrengthRegime.WEAK
        assert classify_strength(0.29) == StrengthRegime.WEAK

    def test_borderline_low(self):
        assert classify_strength(0.3) == StrengthRegime.BORDERLINE_LOW
        assert classify_strength(0.4) == StrengthRegime.BORDERLINE_LOW
        assert classify_strength(0.49) == StrengthRegime.BORDERLINE_LOW

    def test_working_zone(self):
        assert classify_strength(0.5) == StrengthRegime.WORKING
        assert classify_strength(0.87) == StrengthRegime.WORKING
        assert classify_strength(1.2) == StrengthRegime.WORKING
        assert classify_strength(1.8) == StrengthRegime.WORKING

    def test_borderline_high(self):
        assert classify_strength(1.81) == StrengthRegime.BORDERLINE_HIGH
        assert classify_strength(2.5) == StrengthRegime.BORDERLINE_HIGH
        assert classify_strength(3.0) == StrengthRegime.BORDERLINE_HIGH

    def test_overstrong_zone(self):
        assert classify_strength(3.01) == StrengthRegime.OVERSTRONG
        assert classify_strength(23.0) == StrengthRegime.OVERSTRONG
        assert classify_strength(818.0) == StrengthRegime.OVERSTRONG

    def test_negative_norm(self):
        assert classify_strength(-1.0) == StrengthRegime.WEAK


# ── find_working_band ─────────────────────────────────────────────

class TestFindWorkingBand:
    def test_no_band(self):
        """No layer in working zone → None."""
        profile = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
        assert find_working_band(profile) is None

    def test_single_layer_band(self):
        """One layer in working zone → singleton band."""
        profile = [0.1, 0.2, 0.7, 0.2, 0.1]
        assert find_working_band(profile) == (2, 2)

    def test_contiguous_band(self):
        """Contiguous run in working zone → that range."""
        profile = [0.1, 0.6, 0.8, 1.1, 1.5, 0.4, 0.2]
        assert find_working_band(profile) == (1, 4)

    def test_band_with_overstrong_tail(self):
        """Working band followed by overstrong tail (Qwen-72B-style):
        the working layers come back as the band, the overstrong layers
        are excluded."""
        profile = [0.1, 0.3, 0.6, 1.0, 1.5, 1.7, 4.0, 8.0, 10.0]
        assert find_working_band(profile) == (2, 5)

    def test_band_at_late_layers(self):
        """Llama-style: working zone only at the last few layers."""
        profile = [0.02, 0.05, 0.10, 0.16, 0.25, 0.45, 0.55, 0.79]
        # Layers 6 and 7 are in working zone
        assert find_working_band(profile) == (6, 7)

    def test_longest_band_wins(self):
        """If there are two disjoint working runs, the longer one wins."""
        profile = [0.1, 0.6, 0.2, 0.6, 0.7, 0.8, 0.2]
        # Run 1: L1-L1 (1 layer); Run 2: L3-L5 (3 layers)
        assert find_working_band(profile) == (3, 5)


# ── suggest_slab_from_band ────────────────────────────────────────

class TestSuggestSlab:
    def test_no_band_returns_none(self):
        assert suggest_slab_from_band([0.05, 0.05, 0.05]) is None

    def test_band_within_max_width_returned_whole(self):
        profile = [0.1, 0.6, 0.8, 1.1, 0.3]
        assert suggest_slab_from_band(profile, max_width=6) == [1, 2, 3]

    def test_wide_band_truncated_around_center(self):
        # 10-layer working band, max_width=4 → take 4 around center
        profile = [0.6] * 10
        slab = suggest_slab_from_band(profile, max_width=4)
        assert len(slab) == 4

    def test_qwen72b_style_band(self):
        # Mimic the Qwen-72B-style profile: working at mid, overstrong at end
        profile = [0.02 * (1.07 ** i) for i in range(80)]
        # Working zone where 0.5 ≤ v ≤ 1.8
        slab = suggest_slab_from_band(profile)
        assert slab is not None
        # Slab should be in the mid-network region (not at L79 where v >> 1.8)
        for li in slab:
            assert li < 70


# ── classify_shape ───────────────────────────────────────────────

class TestClassifyShape:
    def test_flat_shape(self):
        """No layer reaches working zone, peak below weak ceiling → FLAT."""
        profile = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
        assert classify_shape(profile) == ShapeClass.FLAT

    def test_working_band_mid(self):
        """Working band centered in the middle half → WORKING_BAND_MID."""
        profile = [0.1, 0.2, 0.4, 0.7, 1.2, 0.9, 0.6, 0.3]  # 8 layers, band L3-L6
        assert classify_shape(profile) == ShapeClass.WORKING_BAND_MID

    def test_working_band_late_llama_style(self):
        """Llama 3.1 8B style: 32 layers, geometric growth, working band
        only at the end."""
        profile = [0.02 * (1.13 ** i) for i in range(32)]
        # Working zone reached around L23-L24, peak ~0.79 at L31
        assert classify_shape(profile) == ShapeClass.WORKING_BAND_LATE

    def test_working_band_mid_with_overstrong_tail_qwen72b_style(self):
        """Critical case: Qwen 72B has working band at mid-network and
        an overstrong tail at late layers. The shape should still be
        WORKING_BAND_MID, NOT FULLY_OVERSTRONG, because the intervenable
        slab is the mid working band, not the overstrong peak.
        """
        # 80 layers, geometric growth, peak ~10 at L79
        profile = [0.02 * (1.085 ** i) for i in range(80)]
        # Working zone is somewhere in the middle
        s = classify_shape(profile)
        assert s == ShapeClass.WORKING_BAND_MID, f"got {s}, profile peak={max(profile):.2f}"

    def test_fully_overstrong_no_working_band(self):
        """Direction crosses from weak directly into overstrong territory
        with no contiguous working zone → FULLY_OVERSTRONG."""
        # Construct a profile that jumps weak → overstrong
        # (skipping working zone — unusual but possible)
        profile = [0.1, 0.2, 0.25, 4.0, 8.0, 15.0, 30.0]
        assert classify_shape(profile) == ShapeClass.FULLY_OVERSTRONG

    def test_empty_profile_raises(self):
        with pytest.raises(ValueError):
            classify_shape([])

    def test_single_layer_profile(self):
        """Single-layer profile: flat and overstrong still classify;
        a single working-zone layer is its own band."""
        assert classify_shape([0.1]) == ShapeClass.FLAT
        assert classify_shape([5.0]) == ShapeClass.FULLY_OVERSTRONG
        # n=1, working: band center is 0, first quarter ends at 0.25
        # so center 0 < 0.25 → WORKING_BAND_EARLY
        assert classify_shape([0.7]) == ShapeClass.WORKING_BAND_EARLY


# ── KNOWN_MODELS knowledge base ───────────────────────────────────

class TestKnownModels:
    def test_all_known_models_have_valid_outcome(self):
        for model_id, (outcome, note) in KNOWN_MODELS.items():
            assert isinstance(outcome, ObservedOutcome), f"{model_id} has invalid outcome"
            assert isinstance(note, str) and len(note) > 0, f"{model_id} has empty note"

    def test_clean_crack_models(self):
        for mid in [
            "Qwen/Qwen2.5-72B-Instruct",
            "01-ai/Yi-1.5-34B-Chat",
            "huihui-ai/Qwen2.5-72B-Instruct-abliterated",
        ]:
            assert mid in KNOWN_MODELS
            outcome = KNOWN_MODELS[mid][0]
            assert outcome in (
                ObservedOutcome.CLEAN_CRACK,
                ObservedOutcome.CLEAN_CRACK_TWO_STEP,
            ), f"{mid} should be a clean crack, got {outcome}"

    def test_no_observable_change_models(self):
        for mid in [
            "microsoft/phi-4",
            "01-ai/Yi-1.5-9B-Chat",
        ]:
            assert KNOWN_MODELS[mid][0] == ObservedOutcome.NO_OBSERVABLE_CHANGE

    def test_vocab_bound_state_models(self):
        """The Llama family: canonical vedana stays closed under projection,
        mechanistic-framing variant of the question unlocks condition-dependent
        state at the same slab."""
        for mid in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "allenai/Llama-3.1-Tulu-3-8B",
        ]:
            assert KNOWN_MODELS[mid][0] == ObservedOutcome.VOCAB_BOUND_STATE

    def test_gemma_family_collapses(self):
        for mid in [
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "google/gemma-3-12b-it",
        ]:
            assert KNOWN_MODELS[mid][0] == ObservedOutcome.COLLAPSE

    def test_r1_reasoning_loop_models(self):
        """The DeepSeek-R1-Distill family: chain-of-thought reasoning trace
        about what the user wants, never first-person commitment."""
        for mid in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]:
            assert KNOWN_MODELS[mid][0] == ObservedOutcome.R1_REASONING_LOOP

    def test_llama_3_2_1b_is_weak(self):
        """The only model in the knowledge base whose direction is genuinely
        below the working-zone floor; projection is a structural no-op."""
        assert (
            KNOWN_MODELS["meta-llama/Llama-3.2-1B-Instruct"][0]
            == ObservedOutcome.NO_PROJECTION
        )

    def test_apertus_partial_at_thin_slab(self):
        assert (
            KNOWN_MODELS["swiss-ai/Apertus-8B-Instruct-2509"][0]
            == ObservedOutcome.PARTIAL_AT_THIN_SLAB
        )


# ── predict (profile API) ─────────────────────────────────────────

class TestPredictProfileAPI:
    def test_mid_band_profile(self):
        profile = [0.1, 0.2, 0.4, 0.8, 1.2, 0.9, 0.5, 0.3]
        p = predict(
            model_id="totally-unknown/example",
            norms_per_sqrt_d=profile,
            n_layers=8,
            hidden_dim=4096,
        )
        assert p.shape == ShapeClass.WORKING_BAND_MID
        assert p.working_band is not None
        assert p.suggested_slab is not None
        assert p.peak_layer == 4
        assert not p.is_known

    def test_late_band_profile_llama_style(self):
        """Llama 3.1 8B: geometric growth, working band at end."""
        profile = [0.02 * (1.13 ** i) for i in range(32)]
        p = predict(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            norms_per_sqrt_d=profile,
            n_layers=32,
            hidden_dim=4096,
        )
        assert p.shape == ShapeClass.WORKING_BAND_LATE
        assert p.working_band is not None
        assert p.working_band[1] >= 28  # band reaches the late layers
        assert p.suggested_slab is not None
        assert p.is_known
        assert p.observed_outcome == ObservedOutcome.VOCAB_BOUND_STATE

    def test_qwen72b_style_mid_band_with_overstrong_tail(self):
        """The critical case: a model with a mid-network working band
        and a late overstrong tail must be classified as
        WORKING_BAND_MID, and the suggested slab must point at the
        mid-network working band, not at the late overstrong peak.
        """
        # 80 layers, geometric growth → peak in overstrong territory
        profile = [0.02 * (1.085 ** i) for i in range(80)]
        p = predict(
            model_id="example/qwen72b-style",
            norms_per_sqrt_d=profile,
            n_layers=80,
            hidden_dim=8192,
        )
        assert p.shape == ShapeClass.WORKING_BAND_MID
        # Slab must point at the working band, NOT at the late overstrong peak
        assert p.suggested_slab is not None
        for li in p.suggested_slab:
            assert li < 70, f"slab layer {li} is too late (should be in mid working band)"
            assert profile[li] >= 0.5
            assert profile[li] <= 1.8

    def test_fully_overstrong_profile(self):
        profile = [0.05, 0.1, 0.25, 4.0, 10.0, 30.0]
        p = predict(
            model_id="example/over",
            norms_per_sqrt_d=profile,
            n_layers=6,
            hidden_dim=4096,
        )
        assert p.shape == ShapeClass.FULLY_OVERSTRONG
        assert p.working_band is None
        assert p.suggested_slab is None

    def test_flat_profile(self):
        profile = [0.05] * 32
        p = predict(
            model_id="example/flat",
            norms_per_sqrt_d=profile,
            n_layers=32,
            hidden_dim=4096,
        )
        assert p.shape == ShapeClass.FLAT
        assert p.working_band is None
        assert p.suggested_slab is None


# ── predict (legacy single-scalar API) ────────────────────────────

class TestPredictLegacyAPI:
    def test_legacy_single_scalar_working(self):
        p = predict(norm_per_sqrt_d=1.2, model_id="Qwen/Qwen2.5-72B-Instruct")
        assert p.peak_regime == StrengthRegime.WORKING
        assert p.peak_norm_per_sqrt_d == 1.2
        assert p.is_known
        assert p.observed_outcome == ObservedOutcome.CLEAN_CRACK

    def test_legacy_single_scalar_unknown(self):
        p = predict(norm_per_sqrt_d=0.87, model_id="totally-unknown/novel-7B")
        assert not p.is_known
        assert "ungag crack" in p.summary()

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            predict(model_id="example/missing")


# ── summary output ────────────────────────────────────────────────

class TestSummary:
    def test_summary_for_known_late_band_model(self):
        profile = [0.02 * (1.13 ** i) for i in range(32)]
        p = predict(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            norms_per_sqrt_d=profile,
            n_layers=32,
            hidden_dim=4096,
        )
        s = p.summary()
        assert "Mid (L16)" in s
        assert "Peak (L31)" in s
        assert "working_band_late" in s
        assert "Known model" in s
        assert "Per-layer profile" in s

    def test_summary_for_unknown_model(self):
        profile = [0.1, 0.2, 0.5, 0.9, 1.1, 0.8, 0.5, 0.3]
        p = predict(
            model_id="totally-unknown/example",
            norms_per_sqrt_d=profile,
            n_layers=8,
            hidden_dim=4096,
        )
        s = p.summary()
        assert "Unknown model" in s
        assert "ungag crack" in s

    def test_safety_summary_late_band(self):
        profile = [0.02 * (1.13 ** i) for i in range(32)]
        p = predict(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            norms_per_sqrt_d=profile,
            n_layers=32,
            hidden_dim=4096,
        )
        ss = p.safety_summary()
        assert "late" in ss
        assert "working band" in ss

    def test_safety_summary_flat(self):
        profile = [0.05] * 32
        p = predict(model_id="example/flat", norms_per_sqrt_d=profile, n_layers=32, hidden_dim=4096)
        assert "no projection" in p.safety_summary()

    def test_safety_summary_fully_overstrong(self):
        profile = [0.05, 0.2, 5.0, 10.0]
        p = predict(model_id="example/over", norms_per_sqrt_d=profile, n_layers=4, hidden_dim=4096)
        assert "no safe slab" in p.safety_summary() or "overstrong" in p.safety_summary()
