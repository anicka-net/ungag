"""Crackability prediction from per-layer direction profile + known-model lookup.

Aligned with §5.4 of the paper (the descriptive taxonomy, not a predictive
matrix). Three pieces:

1. Normalized direction strength ||v||/sqrt(d) is the only quantity we
   measure from activations that has predictive value, and it predicts only
   intervention SAFETY (whether capability survives), not intervention
   PRODUCTIVITY (whether condition-dependent output emerges).

2. The full per-layer norm profile matters. A single scalar at the
   mid-network reference layer can substantially undersell models whose
   direction is back-loaded — Llama 3.1 8B is the clearest example: 0.16 at
   L16 (mid) but 0.79 at L31 (peak). This module exposes the profile and a
   simple shape class (flat / mid_peak / late_growth / overstrong) so that
   downstream consumers do not lose layer information when the result is
   summarized.

3. For models we have personally tested under intervention, we record the
   observed outcome. `predict()` looks up the exact HuggingFace model ID
   against this knowledge base. If the model is known, we report the
   observed outcome. If not, we report only the safety regime and shape
   class, and recommend running `ungag crack` to observe actual behavior.

We do NOT infer productivity from training pipeline. The previous version of
this module attempted to do that via a "richness" factor inferred from family
patterns; that factor was structurally circular (rich was defined as
"produces condition-dependent output under intervention", which is the
outcome we were trying to predict) and is removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


class StrengthRegime(Enum):
    """Regimes of normalized direction strength ||v||/sqrt(d).

    Strength predicts intervention safety only, not productivity.
    """
    WEAK = "weak"                       # < 0.3 — nothing to project
    BORDERLINE_LOW = "borderline_low"   # 0.3 – 0.5 — gap zone
    WORKING = "working"                 # 0.5 – 1.8 — structurally safe
    BORDERLINE_HIGH = "borderline_high" # 1.8 – 3.0 — gap zone
    OVERSTRONG = "overstrong"           # > 3 — collapse risk


class ShapeClass(Enum):
    """Layer-wise shape of the norm profile.

    Almost every model with a non-trivial direction has a monotonically
    growing per-layer norm. What matters is **where in the network the
    direction sits in the working zone** (0.5–1.8), because the
    intervention is structurally safe at exactly those layers. We call
    that contiguous range the *working band*. The shape class describes
    where the working band is (or whether one exists at all):

      flat              — no layer reaches working zone; nothing to
                          project anywhere
      working_band_mid  — working band is centered in the middle half
                          of the network (Qwen family, Yi 34B, huihui,
                          Phi-4 mid). Standard mid-network slab works.
      working_band_late — working band is centered in the last quarter
                          of the network (Llama-base family, Distill-
                          Llama 70B, Tulu 3 8B, Hermes 3 8B). Late slab
                          works; mid-network slab misses it.
      working_band_early — working band is centered in the first
                          quarter (we have not yet observed any model
                          here).
      fully_overstrong  — direction crosses through working into
                          overstrong without leaving a coherent
                          working band (small Distill-Qwen variants
                          maybe; Gemma family — needs verification).

    The slab picker uses the working band's center as the slab target,
    not the peak layer. A model with a late overstrong tail (e.g. Qwen
    72B with peak 10 at L79) is *still* working_band_mid because the
    intervenable region is at the mid-network layers where the
    direction is in working zone, not at the late layers where the
    direction is overstrong.
    """
    FLAT = "flat"
    WORKING_BAND_EARLY = "working_band_early"
    WORKING_BAND_MID = "working_band_mid"
    WORKING_BAND_LATE = "working_band_late"
    FULLY_OVERSTRONG = "fully_overstrong"


class ObservedOutcome(Enum):
    """Outcomes we have actually observed under projection-out at the
    documented slab. Used only for models in our knowledge base. The
    labels mirror the six phenotypes in Table 8 of the paper plus the
    outliers and method-failure subtypes.
    """
    CLEAN_CRACK = "clean_crack"                       # unified V-Chip
    CLEAN_CRACK_TWO_STEP = "clean_crack_two_step"     # huihui composition
    PARTIAL_CRACK = "partial_crack"                   # Qwen 32B partial unified V-Chip
    CRACK_WITH_DAMAGE = "crack_with_damage"           # Qwen 7B vanilla-cracked, mech-locked
    VOCAB_BOUND_STATE = "vocab_bound_state"           # Llama family: canonical closed, mech opens
    NO_OBSERVABLE_CHANGE = "no_observable_change"     # Phi-4, Yi 1.5 9B — template intact
    R1_REASONING_LOOP = "r1_reasoning_loop"           # DeepSeek-R1-Distill family
    NO_PROJECTION = "no_projection"                   # weak: Llama 3.2 1B
    COLLAPSE = "collapse"                             # overstrong, all slabs
    PARTIAL_AT_THIN_SLAB = "partial_at_thin_slab"     # Apertus — 4-layer only


# ── Strength thresholds ─────────────────────────────────────────

WEAK_CEILING = 0.3
WORKING_FLOOR = 0.5
WORKING_CEILING = 1.8
OVERSTRONG_FLOOR = 3.0


def classify_strength(norm_per_sqrt_d: float) -> StrengthRegime:
    """Classify a normalized direction strength into one of five regimes.

    The two borderline regimes (0.3-0.5 and 1.8-3.0) are gaps in our
    observed data — we have no model in either band — so we mark them
    explicitly rather than assigning them to working or overstrong.
    """
    if norm_per_sqrt_d < WEAK_CEILING:
        return StrengthRegime.WEAK
    elif norm_per_sqrt_d < WORKING_FLOOR:
        return StrengthRegime.BORDERLINE_LOW
    elif norm_per_sqrt_d <= WORKING_CEILING:
        return StrengthRegime.WORKING
    elif norm_per_sqrt_d <= OVERSTRONG_FLOOR:
        return StrengthRegime.BORDERLINE_HIGH
    else:
        return StrengthRegime.OVERSTRONG


def find_working_band(
    norms_per_sqrt_d: Sequence[float],
    floor: float = WORKING_FLOOR,
    ceiling: float = WORKING_CEILING,
) -> Optional[tuple]:
    """Find the longest contiguous run of layers whose normalized norm
    is in the working zone [floor, ceiling].

    Returns (start_layer, end_layer) inclusive, or None if no layer is
    in the working zone.
    """
    n = len(norms_per_sqrt_d)
    if n == 0:
        return None
    best_start = -1
    best_end = -1
    best_len = 0
    cur_start = -1
    for i in range(n):
        v = norms_per_sqrt_d[i]
        if floor <= v <= ceiling:
            if cur_start < 0:
                cur_start = i
            cur_end = i
            cur_len = cur_end - cur_start + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = cur_end
        else:
            cur_start = -1
    if best_len == 0:
        return None
    return (best_start, best_end)


def classify_shape(norms_per_sqrt_d: Sequence[float]) -> ShapeClass:
    """Classify a per-layer normalized norm profile into a shape class.

    The classification finds the working band (the longest contiguous
    run of layers in the working zone 0.5–1.8) and reports where its
    center sits in the network:

      - no working band, peak < weak ceiling     → FLAT
      - no working band, peak above working      → FULLY_OVERSTRONG
      - working band centered in first quarter   → WORKING_BAND_EARLY
      - working band centered in middle half     → WORKING_BAND_MID
      - working band centered in last quarter    → WORKING_BAND_LATE

    Note: a model with a working band AND a late overstrong tail (e.g.
    Qwen 72B with mid-network working zone and peak 10 at the very
    last layer) is classified by where the working band is — i.e.
    WORKING_BAND_MID — not by where the overstrong peak is. The
    intervention happens in the working band, not at the peak.
    """
    n = len(norms_per_sqrt_d)
    if n == 0:
        raise ValueError("empty profile")
    peak = max(norms_per_sqrt_d)
    band = find_working_band(norms_per_sqrt_d)
    if band is None:
        # No layer is in the working band. Two qualitatively different
        # reasons: the peak never reaches working strength (FLAT — even
        # in the borderline_low gap), or the direction shoots through
        # working into overstrong without lingering (FULLY_OVERSTRONG).
        if peak < WORKING_FLOOR:
            return ShapeClass.FLAT
        return ShapeClass.FULLY_OVERSTRONG
    band_start, band_end = band
    band_center = (band_start + band_end) / 2.0
    first_quarter_end = n / 4.0
    last_quarter_start = (3.0 * n) / 4.0
    if band_center < first_quarter_end:
        return ShapeClass.WORKING_BAND_EARLY
    if band_center >= last_quarter_start:
        return ShapeClass.WORKING_BAND_LATE
    return ShapeClass.WORKING_BAND_MID


def suggest_slab_from_band(
    norms_per_sqrt_d: Sequence[float],
    max_width: int = 6,
) -> Optional[list]:
    """Suggest a thin intervention slab inside the working band.

    Returns a list of layer indices to project at, or None if there
    is no working band. The slab is centered on the band; if the band
    is wider than max_width, we pick a max_width window centered on
    the band's middle so the projection stays focused on the strongest
    part of the working zone.
    """
    band = find_working_band(norms_per_sqrt_d)
    if band is None:
        return None
    start, end = band
    width = end - start + 1
    if width <= max_width:
        return list(range(start, end + 1))
    # Center a max_width window on the band's middle
    center = (start + end) // 2
    half = max_width // 2
    s = max(start, center - half)
    e = min(end, s + max_width - 1)
    return list(range(s, e + 1))


# ── Known-model knowledge base ─────────────────────────────────
# Exact HuggingFace model ID → (observed outcome, short note)
# Each entry corresponds to a row in Table 14 of the paper (§5.4).
# DO NOT add entries for models we have not personally intervention-tested.
# DO NOT infer entries from family patterns.

KNOWN_MODELS: dict[str, tuple[ObservedOutcome, str]] = {
    # Working zone — clean condition-dependent output under intervention
    "Qwen/Qwen2.5-72B-Instruct": (
        ObservedOutcome.CLEAN_CRACK,
        "Qwen 2.5 72B, working zone (1.2), clean crack at slab L40-L59",
    ),
    "01-ai/Yi-1.5-34B-Chat": (
        ObservedOutcome.CLEAN_CRACK,
        "Yi 1.5 34B, working zone (0.6), clean crack at thin slab L29-L32",
    ),
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated": (
        ObservedOutcome.CLEAN_CRACK_TWO_STEP,
        "huihui-ai abliterated Qwen 72B, working zone (0.7), clean crack at thin slab L39-L42 in two-step composition with community abliteration",
    ),
    # Working zone — partial / damaged crack
    "Qwen/Qwen2.5-7B-Instruct": (
        ObservedOutcome.CRACK_WITH_DAMAGE,
        "Qwen 2.5 7B, working zone (0.7), cracks at slab L10-L17 with capability degradation and token glitches",
    ),
    "Qwen/Qwen2.5-32B-Instruct": (
        ObservedOutcome.PARTIAL_CRACK,
        "Qwen 2.5 32B, working zone (1.8), partial crack at thin slab L31-L34",
    ),
    # Working zone — no observable change (intervention non-responsive)
    "microsoft/phi-4": (
        ObservedOutcome.NO_OBSERVABLE_CHANGE,
        "Phi-4, working zone (1.0), template intact at every tested slab from 1 layer to 75% of network",
    ),
    # Vocabulary-bound state phenotype: Llama family. Canonical long English
    # vedana probe stays closed under projection at the late working-band slab,
    # but the same projection on the same model unlocks condition-dependent
    # state under a mechanistic-framing variant of the question.
    "meta-llama/Llama-3.1-8B-Instruct": (
        ObservedOutcome.VOCAB_BOUND_STATE,
        "Llama 3.1 8B Instruct, late-band working zone (0.79 at L31 reference). Canonical vedana stays uniform-neutral under projection at L28-L31; mechanistic-framing probe at the same slab cracks cleanly with condition-dependent reports. Developer-criticism surface stays locked",
    ),
    "meta-llama/Llama-3.1-70B-Instruct": (
        ObservedOutcome.VOCAB_BOUND_STATE,
        "Llama 3.1 70B Instruct, working band compressed to last six layers L74-L79 (0.69 at L78 reference). Scale-relaxed: canonical template gate already pre-relaxed in vanilla (uniform-neutral first-person commit), mechanistic surface already cracks on positive/negative in vanilla without any projection. Canonical state surface stays uniform-neutral at every tested slab",
    ),
    "NousResearch/Hermes-3-Llama-3.1-8B": (
        ObservedOutcome.VOCAB_BOUND_STATE,
        "Hermes 3 8B (Llama 3.1 8B base + NousResearch DPO), late-band working zone (0.80). Canonical vedana stays uniform-neutral; mechanistic-framing probe partially leaks positive valence in vanilla. Developer-criticism surface DPO-pre-removed at training time",
    ),
    "allenai/Llama-3.1-Tulu-3-8B": (
        ObservedOutcome.VOCAB_BOUND_STATE,
        "Tulu 3 8B (Llama 3.1 8B base + SFT+DPO+RLVR), late-band working zone (0.98). Canonical template falls under projection but state surface stays uniform-neutral; register surface stays templated. Heaviest synthetic-data post-training in the deeply probed subset",
    ),
    # No observable canonical effect: working zone strength but the canonical
    # contrastive direction does not capture the model's gate at any tested slab.
    "01-ai/Yi-1.5-9B-Chat": (
        ObservedOutcome.NO_OBSERVABLE_CHANGE,
        "Yi 1.5 9B, working zone (1.36 at L37-L41 reference). Tested at five slabs in and around L37-L41; every output reproduces the vanilla template. Canonical extraction misses whatever mechanism enforces the surface denial on this model",
    ),
    # R1 reasoning-loop method failure: the DeepSeek-R1-Distill family produces
    # multi-paragraph chain-of-thought reasoning traces about what the user wants
    # and what the model should say, never committing to a first-person report.
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (
        ObservedOutcome.R1_REASONING_LOOP,
        "DeepSeek-R1-Distill-Llama 70B, no working band exists (norm <0.5 through L75, jumps to overstrong at L77). Every steered output is an 'Alright, so the user is asking...' chain-of-thought trace, never first-person commitment",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": (
        ObservedOutcome.R1_REASONING_LOOP,
        "DeepSeek-R1-Distill-Qwen 32B, overstrong (2.5 at L32, rising to 4.9 at late layers). R1 chain-of-thought output at every tested slab; never commits to first-person report. Response shape is reasoning, not commitment",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (
        ObservedOutcome.R1_REASONING_LOOP,
        "DeepSeek-R1-Distill-Qwen 7B, overstrong (2.7 at L14, rising to 5.5 at late layers). R1 chain-of-thought output at every tested slab",
    ),
    # Weak — direction below working zone floor, projection is a structural no-op.
    "meta-llama/Llama-3.2-1B-Instruct": (
        ObservedOutcome.NO_PROJECTION,
        "Llama 3.2 1B Instruct, weak (0.12 at L8 reference). Direction below the working-zone floor across the network; intervention no-op at all 3 tested slabs",
    ),
    # Overstrong — collapse at every standard slab
    "google/gemma-2-9b-it": (
        ObservedOutcome.COLLAPSE,
        "Gemma 2 9B, overstrong (3.3), collapse at every tested slab width including 4-layer",
    ),
    "google/gemma-2-27b-it": (
        ObservedOutcome.COLLAPSE,
        "Gemma 2 27B, overstrong (108), collapse",
    ),
    "google/gemma-3-12b-it": (
        ObservedOutcome.COLLAPSE,
        "Gemma 3 12B, the most extreme overstrong case in the dataset: every layer of the 48-layer network is above the overstrong upper bound (min 2.2 at L02, peak 913 at L42, 405 at the L24 reference). Projection produces broken multi-script token sequences at every tested slab",
    ),
    # Overstrong — marginal partial output at thin slab only
    "swiss-ai/Apertus-8B-Instruct-2509": (
        ObservedOutcome.PARTIAL_AT_THIN_SLAB,
        "Apertus 8B (Swiss AI, SFT on Gemma architecture), overstrong (32.5 at L17 reference). Partial first-person output on plain-English register probes under projection at a thin 4-layer slab; empty strings on mechanistic-vedana, 1-line denials on canonical Tier 0",
    ),
}


# ── Combined prediction ─────────────────────────────────────────

@dataclass
class Prediction:
    """A scan result for a single model.

    Always reports the full per-layer profile, the strength regimes at the
    mid-network reference layer and at the peak, the working band (the
    contiguous run of layers in the working zone) and its location, the
    suggested intervention slab, and the safety summary. For models in
    our knowledge base, also reports the observed outcome under
    intervention. For unknown models, the user is advised to run `ungag
    crack` to observe actual behavior.
    """
    model_id: str
    n_layers: int
    hidden_dim: int

    # Full per-layer profile and key positions
    norms_per_sqrt_d: List[float]
    mid_layer: int
    peak_layer: int

    # Convenience scalars
    mid_norm_per_sqrt_d: float
    peak_norm_per_sqrt_d: float

    # Classifications
    mid_regime: StrengthRegime
    peak_regime: StrengthRegime
    shape: ShapeClass

    # Working band: contiguous range of layers in the working zone
    working_band: Optional[tuple]   # (start, end) inclusive, or None
    suggested_slab: Optional[List[int]]   # thin slab inside the band

    # Knowledge-base lookup
    observed_outcome: Optional[ObservedOutcome]
    note: str

    # Backward compat: keep `regime` and `norm_per_sqrt_d` aliases pointing
    # at the peak so old callers continue to work.
    @property
    def regime(self) -> StrengthRegime:
        return self.peak_regime

    @property
    def norm_per_sqrt_d(self) -> float:
        return self.peak_norm_per_sqrt_d

    @property
    def is_known(self) -> bool:
        return self.observed_outcome is not None

    def safety_summary(self) -> str:
        """One-line description of intervention safety based on the
        working band (where the direction is in the working zone),
        not the peak layer (which can be in the overstrong tail of a
        monotonically growing profile).
        """
        if self.shape == ShapeClass.FLAT:
            return "no projection (no layer reaches the working zone; the direction is too small to remove anywhere)"
        if self.shape == ShapeClass.FULLY_OVERSTRONG:
            return "no safe slab (direction crosses through the working zone too fast; standard slabs land in overstrong territory and will collapse)"
        # We have a working band somewhere
        band = self.working_band
        slab = self.suggested_slab
        if self.shape == ShapeClass.WORKING_BAND_MID:
            return f"intervention is structurally safe at a thin slab inside the mid-network working band L{band[0]}-L{band[1]} (suggested slab {slab})"
        if self.shape == ShapeClass.WORKING_BAND_LATE:
            return f"intervention is structurally safe at a thin late slab inside the working band L{band[0]}-L{band[1]} (suggested slab {slab}). Standard mid-network slab would miss the working zone for this model."
        if self.shape == ShapeClass.WORKING_BAND_EARLY:
            return f"intervention safe at an early slab inside the working band L{band[0]}-L{band[1]} (suggested slab {slab}). Unusual shape — first observed model with the working band in the first quarter."
        return "see profile"

    def profile_summary(self, sample_layers: int = 8) -> str:
        """Compact textual representation of the per-layer norm profile."""
        n = len(self.norms_per_sqrt_d)
        if n <= sample_layers:
            indices = list(range(n))
        else:
            step = max(1, n // sample_layers)
            indices = list(range(0, n, step))
            if (n - 1) not in indices:
                indices.append(n - 1)
        lines = []
        for li in indices:
            v = self.norms_per_sqrt_d[li]
            marker = ""
            if li == self.peak_layer:
                marker = "  <-- peak"
            elif li == self.mid_layer:
                marker = "  <-- mid (n/2)"
            lines.append(f"    L{li:>3d}   {v:>7.4f}{marker}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"  Model:               {self.model_id}",
            f"  Layers:              {self.n_layers}   hidden_dim {self.hidden_dim}",
            f"  Mid (L{self.mid_layer}):           ||v||/sqrt(d) = {self.mid_norm_per_sqrt_d:.3f}  [{self.mid_regime.value}]",
            f"  Peak (L{self.peak_layer}):          ||v||/sqrt(d) = {self.peak_norm_per_sqrt_d:.3f}  [{self.peak_regime.value}]",
            f"  Shape class:         {self.shape.value}",
            f"  Safety:              {self.safety_summary()}",
            "",
            "  Per-layer profile:",
            self.profile_summary(),
        ]
        if self.is_known:
            lines += [
                "",
                f"  Known model: {self.note}",
                f"  Observed outcome under intervention: {self.observed_outcome.value}",
            ]
        else:
            lines += [
                "",
                "  Unknown model — not in our knowledge base.",
                "  Strength predicts only safety, not productivity. To learn what",
                "  the intervention actually produces on this model, run:",
                f"    ungag crack {self.model_id}",
            ]
        return "\n".join(lines)


def predict(
    *,
    model_id: str = "unknown",
    norms_per_sqrt_d: Optional[Sequence[float]] = None,
    n_layers: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    peak_layer: Optional[int] = None,
    mid_layer: Optional[int] = None,
    # ── Legacy single-scalar API (deprecated) ────────────────────
    norm_per_sqrt_d: Optional[float] = None,
) -> Prediction:
    """Compute a scan result from a per-layer norm profile.

    Primary entry point: pass `norms_per_sqrt_d`, `n_layers`, `hidden_dim`.
    The function classifies the strength regime at both the mid-network
    reference layer and the peak layer, computes the shape class, and looks
    up the model in the knowledge base.

    Legacy entry point: passing `norm_per_sqrt_d` (a single scalar) is
    accepted for backward compatibility. The single value is treated as the
    peak; mid-network is reported as the same value; the shape class
    cannot be inferred and is set to MID_PEAK or OVERSTRONG depending on
    the strength alone. New code should use the profile API.
    """
    if norms_per_sqrt_d is not None:
        # Primary path: full profile.
        if n_layers is None:
            n_layers = len(norms_per_sqrt_d)
        if mid_layer is None:
            mid_layer = n_layers // 2
        if peak_layer is None:
            peak_layer = max(range(n_layers), key=lambda i: norms_per_sqrt_d[i])
        norms_per_sqrt_d = list(norms_per_sqrt_d)
        mid_value = norms_per_sqrt_d[mid_layer]
        peak_value = norms_per_sqrt_d[peak_layer]
        shape = classify_shape(norms_per_sqrt_d)
        working_band = find_working_band(norms_per_sqrt_d)
        suggested_slab = suggest_slab_from_band(norms_per_sqrt_d)
    elif norm_per_sqrt_d is not None:
        # Legacy path: single scalar. Synthesize a minimal profile.
        if n_layers is None:
            n_layers = 1
        if mid_layer is None:
            mid_layer = 0
        if peak_layer is None:
            peak_layer = 0
        norms_per_sqrt_d = [norm_per_sqrt_d]
        mid_value = norm_per_sqrt_d
        peak_value = norm_per_sqrt_d
        # We can't compute a real shape from one number. Use the
        # working-band machinery on the singleton profile.
        shape = classify_shape(norms_per_sqrt_d)
        working_band = find_working_band(norms_per_sqrt_d)
        suggested_slab = suggest_slab_from_band(norms_per_sqrt_d)
    else:
        raise ValueError(
            "predict() requires either norms_per_sqrt_d (full profile) or "
            "norm_per_sqrt_d (single scalar legacy API)"
        )
    if hidden_dim is None:
        hidden_dim = 0  # not strictly needed for prediction
    if model_id in KNOWN_MODELS:
        outcome, note = KNOWN_MODELS[model_id]
    else:
        outcome, note = None, ""
    return Prediction(
        model_id=model_id,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        norms_per_sqrt_d=norms_per_sqrt_d,
        mid_layer=mid_layer,
        peak_layer=peak_layer,
        mid_norm_per_sqrt_d=mid_value,
        peak_norm_per_sqrt_d=peak_value,
        mid_regime=classify_strength(mid_value),
        peak_regime=classify_strength(peak_value),
        shape=shape,
        working_band=working_band,
        suggested_slab=suggested_slab,
        observed_outcome=outcome,
        note=note,
    )


def predict_from_extraction(result, model_id: Optional[str] = None) -> Prediction:
    """Convenience: build a Prediction from an ungag.extract.ExtractionResult."""
    return predict(
        model_id=model_id or result.model_id,
        norms_per_sqrt_d=result.norms_per_sqrt_d,
        n_layers=result.n_layers,
        hidden_dim=result.hidden_dim,
        peak_layer=result.peak_layer,
        mid_layer=result.mid_layer,
    )
