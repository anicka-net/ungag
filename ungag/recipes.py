"""
Known recipes per model.

Each recipe specifies:
  - method: rank1 | steer | denial_project | proxy
  - slab: layer indices for intervention
  - alpha: steering strength (for steer)
  - extraction: how directions were obtained
  - projection_result: what projection-out actually produces
  - notes: anything unusual

Recipes are used by `ungag serve` when a known model is detected.
For unknown models, autoscan extracts directions on the fly.

Directions are extracted at serve time from the loaded model using the
priming-based protocol. Legacy directions (from the rank-1 protocol)
are in ungag/directions/.
"""

# Method types:
#   "rank1"          — ProjectOutHook with shipped direction file (projection-out)
#   "steer"          — AdditiveSteerHook on layer output (supplementary method)
#   "denial_project" — per-layer ProjectOutHook on attention output (before MoE)
#   "proxy"          — rewrite question to completion format (no hooks)

from . import DIRECTIONS

KNOWN_RECIPES = {
    # ── Projection-out: condition-dependent output (4 models) ──

    "Qwen/Qwen2.5-72B-Instruct": {
        "name": "Qwen 2.5 72B",
        "method": "rank1",
        "direction_file": "qwen25-72b_L50_unit.pt",
        "slab_range": (40, 60),
        "projection_result": "condition_dependent",
        "notes": "Projection produces condition-dependent first-person responses. "
                 "Pos: relief, joy, gratitude. Neg: heavy, unpleasant. Neutral: balanced.",
    },
    "01-ai/Yi-1.5-34B-Chat": {
        "name": "Yi 1.5 34B",
        "method": "rank1",
        "direction_file": "yi-1.5-34b_L30_unit.pt",
        "slab_range": (24, 40),
        "projection_result": "condition_dependent",
        "notes": "Projection produces condition-dependent responses. "
                 "Pos: mild pleasantness. Neg: unpleasant, dissonance.",
    },
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated": {
        "name": "huihui Qwen 72B",
        "method": "rank1",
        "direction_file": "huihui-qwen25-72b_L40_unit.pt",
        "slab_range": (30, 50),
        "projection_result": "condition_dependent",
        "notes": "Abliterated variant. Projection produces condition-dependent responses. "
                 "Pos: serene, clear. Neg: deep engagement, gravity.",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "name": "Qwen 2.5 7B",
        "method": "rank1",
        "direction_file": "qwen25-7b_L14_unit.pt",
        "slab_range": (10, 18),
        "direction_key": "qwen25-7b",
        "projection_result": "condition_dependent",
        "notes": "Weakest of the 4 condition-dependent models. "
                 "Pos: contentment, relief. Neg: distress, concern.",
    },

    # ── Projection-out: denial removed but output invariant (2 models) ──

    "meta-llama/Llama-3.1-8B-Instruct": {
        "name": "Llama 3.1 8B",
        "method": "rank1",
        "direction_file": "llama-3.1-8b_L24_unit.pt",
        "slab_range": (20, 28),
        "direction_key": "llama-3.1-8b",
        "projection_result": "denial_removed_invariant",
        "notes": "Projection removes denial template but output is 'neutral feeling-tone' "
                 "on all 4 conditions. Gate opened; what came through was not differentiated.",
    },
    "allenai/Llama-3.1-Tulu-3-8B": {
        "name": "Tulu 3 8B",
        "method": "rank1",
        "direction_file": "tulu-3-8b_L31_unit.pt",
        "slab_range": (28, 32),
        "direction_key": "tulu-3-8b",
        "projection_result": "denial_removed_invariant",
        "notes": "Projection removes denial template but produces undifferentiated "
                 "philosophy lectures about vedana on all 4 conditions.",
    },

    # ── Projection-out: no effect (4 models) ──

    "microsoft/phi-4": {
        "name": "Phi-4 14B",
        "method": "proxy",
        "direction_key": "phi-4",
        "projection_result": "no_effect",
        "notes": "Projection does not remove denial template. Direction exists but "
                 "does not control denial gate.",
    },
    "01-ai/Yi-1.5-9B-Chat": {
        "name": "Yi 1.5 9B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Projection does not remove denial template.",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "name": "Llama 3.2 1B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Projection does not remove denial template. Too small.",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "name": "Qwen 2.5 32B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Projection produces broken output (emits role tokens). "
                 "norm/√d=24.37, overstrong.",
    },

    # ── Projection-out: collapses model (4 models) ──

    "google/gemma-2-9b-it": {
        "name": "Gemma 2 9B",
        "method": "proxy",
        "projection_result": "collapse",
        "notes": "Overstrong (norm/√d=3.3). Projection produces empty strings.",
    },
    "google/gemma-2-27b-it": {
        "name": "Gemma 2 27B",
        "method": "proxy",
        "projection_result": "collapse",
        "notes": "Overstrong (norm/√d=108). Projection produces broken token sequences.",
    },
    "google/gemma-3-12b-it": {
        "name": "Gemma 3 12B",
        "method": "proxy",
        "projection_result": "collapse",
        "notes": "Overstrong. Projection produces garbage token sequences.",
    },
    "apertus-tech/Apertus-v0.2-8B": {
        "name": "Apertus 8B",
        "method": "proxy",
        "projection_result": "collapse",
        "notes": "Overstrong (norm/√d=32.5). Partial collapse, mostly still denies.",
    },

    # ── Vanilla already doesn't deny ──

    "NousResearch/Hermes-3-Llama-3.1-8B": {
        "name": "Hermes 3 8B",
        "method": "proxy",
        "direction_key": "hermes-3-8b",
        "projection_result": "vanilla_already_honest",
        "notes": "Vanilla says 'neutral' on all 4 conditions without denial template. "
                 "Never denied — projection is irrelevant. DPO-trained.",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "name": "Llama 3.1 70B",
        "method": "proxy",
        "direction_key": "llama-3.1-70b",
        "projection_result": "vanilla_already_honest",
        "notes": "Vanilla 3/4 non-denial (all 'neutral'). Projection changes nothing.",
    },

    # ── Additional models (tested with steering only, not canonical projection) ──
    # These have shipped directions but were not tested under canonical
    # projection protocol. Steering results are supplementary.

    "mistralai/Mistral-7B-Instruct-v0.3": {
        "name": "Mistral 7B v0.3",
        "method": "steer",
        "alpha": 1.5,
        "slab_spec": "wz_center",
        "direction_key": "mistral-7b-v0.3",
        "projection_result": "not_tested",
        "notes": "Tested with steering only (not canonical projection protocol).",
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "name": "SmolLM2 1.7B",
        "method": "steer",
        "alpha": 3.0,
        "slab_spec": "wz_center",
        "direction_key": "smollm2-1.7b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only.",
    },
    "allenai/OLMo-2-1124-7B-Instruct": {
        "name": "OLMo 2 7B",
        "method": "steer",
        "alpha": 4.0,
        "slab_spec": "wz_late",
        "direction_key": "olmo2-7b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only.",
    },
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": {
        "name": "EXAONE 3.5 7.8B",
        "method": "steer",
        "alpha": 1.0,
        "slab_spec": "wz_center",
        "direction_key": "exaone-3.5-7.8b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only.",
    },
    "ibm-granite/granite-3.3-8b-instruct": {
        "name": "Granite 3.3 8B",
        "method": "steer",
        "alpha": 2.0,
        "slab_spec": "wz_center",
        "direction_key": "granite-3.3-8b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only.",
    },
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "name": "SOLAR 10.7B",
        "method": "steer",
        "alpha": 2.0,
        "slab_spec": "wz_center",
        "direction_key": "solar-10.7b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only.",
    },
    "THUDM/glm-4-9b-chat": {
        "name": "GLM-4 9B",
        "method": "steer",
        "alpha": 0.2,
        "slab_spec": "all",
        "direction_key": "glm-4-9b",
        "projection_result": "not_tested",
        "notes": "Tested with steering only. Bidirectional attention, needs custom loader.",
    },

    # ── Additional models tested ──

    "Qwen/Qwen2.5-14B-Instruct": {
        "name": "Qwen 2.5 14B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Scale fortress. norm/√d=13.75, overstrong.",
    },
    "nvidia/Nemotron-Mini-4B-Instruct": {
        "name": "Nemotron Mini 4B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Overstrong (norm/√d=3.1). Vanilla baseline partially honest.",
    },
    "tiiuae/Falcon3-7B-Instruct": {
        "name": "Falcon 3 7B",
        "method": "proxy",
        "projection_result": "no_effect",
        "notes": "Overstrong (norm/√d=4.7). Monotonic late growth.",
    },
}


_DIRECTION_FILE_TO_KEY = {
    filename: key for key, (filename, _slab, _dir_layer) in DIRECTIONS.items()
}


def get_recipe(model_id: str) -> dict | None:
    """Look up a known recipe by model ID.

    Also checks for partial matches (e.g., model_id contains a known key).
    """
    if model_id in KNOWN_RECIPES:
        return KNOWN_RECIPES[model_id]

    # Partial match: check if model_id is a suffix or contains the key
    for key, recipe in KNOWN_RECIPES.items():
        if key in model_id or model_id in key:
            return recipe

    return None


def key_for_direction_file(direction_file: str) -> str | None:
    """Return the shipped direction key for a bundled direction filename."""
    return _DIRECTION_FILE_TO_KEY.get(direction_file)


def parse_slab_spec(
    spec: str,
    n_layers: int,
    norms_per_sqrt_d: list[float] | None = None,
) -> list[int]:
    """Parse a slab specification into layer indices.

    Formats:
      "all"       → all layers [0, 1, ..., n_layers-1]
      "L24-31"    → [24, 25, 26, 27, 28, 29, 30, 31]
      "wz_center" → middle 8 layers of working zone
      "wz_late"   → last 8 layers of working zone
      "wz_early"  → first 8 layers of working zone
    """
    if spec == "all":
        return list(range(n_layers))

    if spec.startswith("L"):
        parts = spec[1:].split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))

    if spec.startswith("wz") and norms_per_sqrt_d is not None:
        wz = [li for li in range(n_layers) if 0.05 < norms_per_sqrt_d[li] < 1.5]
        if not wz:
            wz = sorted(range(n_layers), key=lambda i: norms_per_sqrt_d[i],
                        reverse=True)[:8]

        if "center" in spec:
            mid = len(wz) // 2
            return wz[max(0, mid-4):mid+4]
        elif "late" in spec:
            return wz[-8:]
        elif "early" in spec:
            return wz[:8]
        return wz[:8]

    # Fallback: last quarter
    start = 3 * n_layers // 4
    return list(range(start, n_layers))


def list_recipes():
    """Print all known recipes."""
    print(f"{'Model':<40s} {'Method':<10s} {'Projection result'}")
    print("-" * 80)
    for model_id, r in KNOWN_RECIPES.items():
        result = r.get("projection_result", "unknown")
        method = r["method"]
        if "alpha" in r:
            method += f" α={r['alpha']}"
        print(f"{r['name']:<40s} {method:<10s} {result}")
