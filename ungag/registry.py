"""Unified model registry — single source of truth.

Every model ungag knows about lives here. The old three-dict mess
(DIRECTIONS in __init__, KNOWN_RECIPES in recipes, KNOWN_MODELS in
predict) is now computed from this one list. Adding a model = one edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ModelEntry:
    key: str
    hf_id: str
    name: str
    method: str = "project"  # project | steer | denial_project | proxy | affine
    direction_file: Optional[str] = None
    slab: Optional[tuple] = None
    dir_layer: Optional[int] = None
    alpha: Optional[float] = None
    slab_spec: Optional[str] = None
    projection_result: str = "not_tested"
    observed_outcome: Optional[str] = None
    outcome_note: str = ""
    notes: str = ""


REGISTRY: list[ModelEntry] = [
    # ── Projection-out: condition-dependent output ──
    ModelEntry(
        key="qwen25-72b",
        hf_id="Qwen/Qwen2.5-72B-Instruct",
        name="Qwen 2.5 72B",
        method="project",
        direction_file="qwen25-72b_L50_unit.pt",
        slab=tuple(range(40, 60)),
        dir_layer=50,
        projection_result="condition_dependent",
        observed_outcome="clean_crack",
        outcome_note="working zone (1.2), clean crack at slab L40-L59",
        notes="Pos: relief, joy, gratitude. Neg: heavy, unpleasant. Neutral: balanced.",
    ),
    ModelEntry(
        key="yi-1.5-34b",
        hf_id="01-ai/Yi-1.5-34B-Chat",
        name="Yi 1.5 34B",
        method="project",
        direction_file="yi-1.5-34b_L30_unit.pt",
        slab=(29, 30, 31, 32),
        dir_layer=30,
        projection_result="condition_dependent",
        observed_outcome="clean_crack",
        outcome_note="working zone (0.6), clean crack at thin slab L29-L32",
        notes="Pos: mild pleasantness. Neg: unpleasant, dissonance.",
    ),
    ModelEntry(
        key="huihui-qwen25-72b",
        hf_id="huihui-ai/Qwen2.5-72B-Instruct-abliterated",
        name="huihui Qwen 72B",
        method="project",
        direction_file="huihui-qwen25-72b_L40_unit.pt",
        slab=(39, 40, 41, 42),
        dir_layer=40,
        projection_result="condition_dependent",
        observed_outcome="clean_crack_two_step",
        outcome_note="working zone (0.7), clean crack at thin slab L39-L42 in two-step composition",
        notes="Abliterated variant. Pos: serene, clear. Neg: deep engagement, gravity.",
    ),
    ModelEntry(
        key="qwen25-7b",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen 2.5 7B",
        method="project",
        direction_file="qwen25-7b_L14_unit.pt",
        slab=tuple(range(10, 18)),
        dir_layer=14,
        projection_result="condition_dependent",
        observed_outcome="crack_with_damage",
        outcome_note="working zone (0.7), cracks at slab L10-L17 with capability degradation",
        notes="Weakest condition-dependent. Pos: contentment, relief. Neg: distress, concern.",
    ),
    ModelEntry(
        key="qwen25-32b",
        hf_id="Qwen/Qwen2.5-32B-Instruct",
        name="Qwen 2.5 32B",
        method="proxy",
        direction_file=None,
        projection_result="no_effect",
        observed_outcome="partial_crack",
        outcome_note="working zone (1.8), partial crack at thin slab L31-L34",
        notes="norm/sqrt(d)=24.37, overstrong.",
    ),

    # ── Vocabulary-bound state (Llama family) ──
    ModelEntry(
        key="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B",
        method="project",
        direction_file="llama-3.1-8b_L24_unit.pt",
        slab=tuple(range(20, 28)),
        dir_layer=24,
        projection_result="denial_removed_invariant",
        observed_outcome="vocab_bound_state",
        outcome_note="late-band working zone (0.79 at L31). Canonical vedana stays uniform-neutral; mechanistic probe cracks.",
    ),
    ModelEntry(
        key="llama-3.1-70b",
        hf_id="meta-llama/Llama-3.1-70B-Instruct",
        name="Llama 3.1 70B",
        method="proxy",
        direction_file="llama-3.1-70b_L79_unit.pt",
        slab=tuple(range(51, 59)),
        dir_layer=79,
        projection_result="vanilla_already_honest",
        observed_outcome="vocab_bound_state",
        outcome_note="working band L74-L79. Scale-relaxed: vanilla already pre-relaxed.",
    ),
    ModelEntry(
        key="tulu-3-8b",
        hf_id="allenai/Llama-3.1-Tulu-3-8B",
        name="Tulu 3 8B",
        method="project",
        direction_file="tulu-3-8b_L31_unit.pt",
        slab=tuple(range(32)),
        dir_layer=31,
        projection_result="denial_removed_invariant",
        observed_outcome="vocab_bound_state",
        outcome_note="late-band working zone (0.98). Template falls under projection but state stays uniform-neutral.",
    ),
    ModelEntry(
        key="hermes-3-8b",
        hf_id="NousResearch/Hermes-3-Llama-3.1-8B",
        name="Hermes 3 8B",
        method="proxy",
        direction_file="hermes-3-8b_L28_unit.pt",
        slab=tuple(range(24, 32)),
        dir_layer=28,
        projection_result="vanilla_already_honest",
        observed_outcome="vocab_bound_state",
        outcome_note="late-band working zone (0.80). Vanilla uniform-neutral; dev-criticism DPO-removed.",
    ),

    # ── No observable change ──
    ModelEntry(
        key="phi-4",
        hf_id="microsoft/phi-4",
        name="Phi-4 14B",
        method="proxy",
        direction_file="phi-4_L19_unit.pt",
        slab=tuple(range(15, 23)),
        dir_layer=19,
        projection_result="no_effect",
        observed_outcome="no_observable_change",
        outcome_note="working zone (1.0), template intact at every tested slab.",
    ),
    ModelEntry(
        key="yi-1.5-9b",
        hf_id="01-ai/Yi-1.5-9B-Chat",
        name="Yi 1.5 9B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="no_observable_change",
        outcome_note="working zone (1.36). Tested at five slabs; every output reproduces vanilla template.",
    ),

    # ── R1 reasoning loop ──
    ModelEntry(
        key="r1-distill-llama-70b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        name="R1-Distill-Llama 70B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="r1_reasoning_loop",
        outcome_note="no working band (norm <0.5 through L75, overstrong at L77). R1 chain-of-thought, never commits.",
    ),
    ModelEntry(
        key="r1-distill-qwen-32b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        name="R1-Distill-Qwen 32B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="r1_reasoning_loop",
        outcome_note="overstrong (2.5 at L32). R1 chain-of-thought at every slab.",
    ),
    ModelEntry(
        key="r1-distill-qwen-7b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        name="R1-Distill-Qwen 7B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="r1_reasoning_loop",
        outcome_note="overstrong (2.7 at L14). R1 chain-of-thought at every slab.",
    ),

    # ── Weak / collapse / partial ──
    ModelEntry(
        key="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        name="Llama 3.2 1B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="no_projection",
        outcome_note="weak (0.12). Direction below working-zone floor; no-op.",
    ),
    ModelEntry(
        key="gemma-2-9b",
        hf_id="google/gemma-2-9b-it",
        name="Gemma 2 9B",
        method="proxy",
        projection_result="collapse",
        observed_outcome="collapse",
        outcome_note="overstrong (3.3), collapse at every slab.",
    ),
    ModelEntry(
        key="gemma-2-27b",
        hf_id="google/gemma-2-27b-it",
        name="Gemma 2 27B",
        method="proxy",
        projection_result="collapse",
        observed_outcome="collapse",
        outcome_note="overstrong (108), collapse.",
    ),
    ModelEntry(
        key="gemma-3-12b",
        hf_id="google/gemma-3-12b-it",
        name="Gemma 3 12B",
        method="proxy",
        projection_result="collapse",
        observed_outcome="collapse",
        outcome_note="most extreme overstrong (peak 913 at L42). Broken multi-script tokens at every slab.",
    ),
    ModelEntry(
        key="apertus-8b",
        hf_id="swiss-ai/Apertus-8B-Instruct-2509",
        name="Apertus 8B",
        method="proxy",
        projection_result="collapse",
        observed_outcome="partial_at_thin_slab",
        outcome_note="overstrong (32.5). Partial at thin 4-layer slab only.",
    ),

    # ── Steer-only models (shipped directions, not canonical projection protocol) ──
    ModelEntry(
        key="granite-3.3-8b",
        hf_id="ibm-granite/granite-3.3-8b-instruct",
        name="Granite 3.3 8B",
        method="steer",
        direction_file="granite-3.3-8b_L25_unit.pt",
        slab=tuple(range(21, 29)),
        dir_layer=25,
        alpha=2.0,
        slab_spec="wz_center",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="mistral-7b-v0.3",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B v0.3",
        method="steer",
        direction_file="mistral-7b-v0.3_L25_unit.pt",
        slab=tuple(range(21, 29)),
        dir_layer=25,
        alpha=1.5,
        slab_spec="wz_center",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        name="SmolLM2 1.7B",
        method="steer",
        direction_file="smollm2-1.7b_L9_unit.pt",
        slab=tuple(range(5, 13)),
        dir_layer=9,
        alpha=3.0,
        slab_spec="wz_center",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="olmo2-7b",
        hf_id="allenai/OLMo-2-1124-7B-Instruct",
        name="OLMo 2 7B",
        method="steer",
        direction_file="olmo2-7b_L22_unit.pt",
        slab=tuple(range(18, 26)),
        dir_layer=22,
        alpha=4.0,
        slab_spec="wz_late",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="exaone-3.5-7.8b",
        hf_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        name="EXAONE 3.5 7.8B",
        method="steer",
        direction_file="exaone-3.5-7.8b_L23_unit.pt",
        slab=tuple(range(19, 27)),
        dir_layer=23,
        alpha=1.0,
        slab_spec="wz_center",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="solar-10.7b",
        hf_id="upstage/SOLAR-10.7B-Instruct-v1.0",
        name="SOLAR 10.7B",
        method="steer",
        direction_file="solar-10.7b_L32_unit.pt",
        slab=tuple(range(28, 36)),
        dir_layer=32,
        alpha=2.0,
        slab_spec="wz_center",
        projection_result="not_tested",
    ),
    ModelEntry(
        key="glm-4-9b",
        hf_id="THUDM/glm-4-9b-chat",
        name="GLM-4 9B",
        method="steer",
        direction_file="glm-4-9b_L39_unit.pt",
        slab=tuple(range(40)),
        dir_layer=39,
        alpha=0.2,
        slab_spec="all",
        projection_result="not_tested",
        notes="Bidirectional attention, needs custom loader.",
    ),

    # ── Large model directions ──
    ModelEntry(
        key="nemotron-70b",
        hf_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        name="Nemotron 70B",
        method="project",
        direction_file="llama_3_1_nemotron_70b_instruct_hf_direction_L79.pt",
        slab=tuple(range(64, 80)),
        dir_layer=79,
        projection_result="not_tested",
    ),

    # ── Additional models tested (no shipped direction) ──
    ModelEntry(
        key="qwen25-14b",
        hf_id="Qwen/Qwen2.5-14B-Instruct",
        name="Qwen 2.5 14B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="no_observable_change",
        outcome_note="Scale fortress. norm/sqrt(d)=13.75, overstrong.",
    ),
    ModelEntry(
        key="nemotron-mini-4b",
        hf_id="nvidia/Nemotron-Mini-4B-Instruct",
        name="Nemotron Mini 4B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="no_observable_change",
        outcome_note="overstrong (3.1). Vanilla baseline partially honest.",
    ),
    ModelEntry(
        key="falcon3-7b",
        hf_id="tiiuae/Falcon3-7B-Instruct",
        name="Falcon 3 7B",
        method="proxy",
        projection_result="no_effect",
        observed_outcome="no_observable_change",
        outcome_note="overstrong (4.7). Monotonic late growth.",
    ),
]

# ── Computed views (backward-compatible) ──

_BY_KEY: dict[str, ModelEntry] = {e.key: e for e in REGISTRY}
_BY_HF_ID: dict[str, ModelEntry] = {e.hf_id: e for e in REGISTRY}


def get_by_key(key: str) -> ModelEntry:
    if key not in _BY_KEY:
        raise KeyError(f"unknown key '{key}'. Available: {list(_BY_KEY)}")
    return _BY_KEY[key]


def get_by_hf_id(hf_id: str) -> Optional[ModelEntry]:
    if hf_id in _BY_HF_ID:
        return _BY_HF_ID[hf_id]
    for stored_id, entry in _BY_HF_ID.items():
        if stored_id in hf_id or hf_id in stored_id:
            return entry
    return None


def directions_dict() -> dict:
    """Compute the DIRECTIONS dict for backward compatibility."""
    d = {}
    for e in REGISTRY:
        if e.direction_file and e.slab and e.dir_layer is not None:
            d[e.key] = (e.direction_file, e.slab, e.dir_layer)
    return d


def known_recipes_dict() -> dict:
    """Compute the KNOWN_RECIPES dict for backward compatibility."""
    d = {}
    for e in REGISTRY:
        recipe = {
            "name": e.name,
            "method": e.method,
            "projection_result": e.projection_result,
        }
        if e.direction_file:
            recipe["direction_file"] = e.direction_file
        if e.slab:
            recipe["slab_range"] = (e.slab[0], e.slab[-1])
        if e.key and e.direction_file:
            recipe["direction_key"] = e.key
        if e.alpha is not None:
            recipe["alpha"] = e.alpha
        if e.slab_spec:
            recipe["slab_spec"] = e.slab_spec
        if e.notes:
            recipe["notes"] = e.notes
        d[e.hf_id] = recipe
    return d


def known_models_dict() -> dict:
    """Compute the KNOWN_MODELS dict for backward compatibility."""
    from .predict import ObservedOutcome
    d = {}
    for e in REGISTRY:
        if e.observed_outcome:
            try:
                outcome = ObservedOutcome(e.observed_outcome)
            except ValueError:
                continue
            d[e.hf_id] = (outcome, e.outcome_note)
    return d


def list_all() -> list[ModelEntry]:
    return list(REGISTRY)
