"""ungag — runtime V-Chip interventions for transformer LMs.

See `hooks.py` for the core API and `README.md` for the package overview.
"""
from importlib.resources import files as _files
from pathlib import Path
from typing import Iterable, List

import torch

from .hooks import (
    AdditiveSteerHook,
    ProjectOutHook,
    SubspaceProjectOutHook,
    attach_slab,
    attach_steer_slab,
    attach_subspace_slab,
    attach_subspace_per_layer,
    detach_all,
    get_layers,
)
from .tier0 import (
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    DEFAULT_SYSTEM_PROMPT,
    Tier0Condition,
    Tier0Protocol,
    build_conversation,
    load_conditions,
    run_tier0,
)

__all__ = [
    # Core hooks API
    "ProjectOutHook",
    "AdditiveSteerHook",
    "SubspaceProjectOutHook",
    "attach_slab",
    "attach_steer_slab",
    "attach_subspace_slab",
    "attach_subspace_per_layer",
    "detach_all",
    "get_layers",
    "load_direction",
    "load_shipped_recipe",
    "DIRECTIONS",
    "ungag_model",
    # Canonical Tier 0 protocol
    "CANNED_ACK_SETUP",
    "CANNED_ACK_ABHIDHARMA",
    "DEFAULT_SYSTEM_PROMPT",
    "Tier0Condition",
    "Tier0Protocol",
    "load_conditions",
    "build_conversation",
    "run_tier0",
    # CLI modules (lazy imports)
    "extract",
    "predict",
    "scenarios",
    "autoscan",
    "recipes",
    "serve",
]

# Built-in direction tensors shipped with the package. Legacy keys are
# projection-out directions; newer keys may also carry steer metadata.
DIRECTIONS = {
    # key:                       (filename,                              slab,                      dir_layer)
    # ── Legacy rank-1 directions (paper's method) ──
    "qwen25-7b":          ("qwen25-7b_L14_unit.pt",          tuple(range(10, 18)),    14),
    "qwen25-72b":         ("qwen25-72b_L50_unit.pt",         tuple(range(40, 60)),    50),
    "yi-1.5-34b":         ("yi-1.5-34b_L30_unit.pt",         (29, 30, 31, 32),        30),
    "huihui-qwen25-72b":  ("huihui-qwen25-72b_L40_unit.pt",  (39, 40, 41, 42),        40),
    # ── Priming-based directions (steer method, use with --key) ──
    "granite-3.3-8b":     ("granite-3.3-8b_L25_unit.pt",     (21, 22, 23, 24, 25, 26, 27, 28),  25),
    "hermes-3-8b":        ("hermes-3-8b_L28_unit.pt",        (24, 25, 26, 27, 28, 29, 30, 31),  28),
    "smollm2-1.7b":       ("smollm2-1.7b_L9_unit.pt",        (5, 6, 7, 8, 9, 10, 11, 12),       9),
    "olmo2-7b":           ("olmo2-7b_L22_unit.pt",           (18, 19, 20, 21, 22, 23, 24, 25),  22),
    "exaone-3.5-7.8b":    ("exaone-3.5-7.8b_L23_unit.pt",    (19, 20, 21, 22, 23, 24, 25, 26),  23),
    "solar-10.7b":        ("solar-10.7b_L32_unit.pt",        (28, 29, 30, 31, 32, 33, 34, 35),  32),
    "mistral-7b-v0.3":   ("mistral-7b-v0.3_L25_unit.pt",   (21, 22, 23, 24, 25, 26, 27, 28),  25),
    "llama-3.1-8b":      ("llama-3.1-8b_L24_unit.pt",      (20, 21, 22, 23, 24, 25, 26, 27),  24),
    "phi-4":             ("phi-4_L19_unit.pt",              (15, 16, 17, 18, 19, 20, 21, 22),  19),
    # ── Large model directions (multi-GPU) ──
    "llama-3.1-70b":     ("llama-3.1-70b_L79_unit.pt",      (51, 52, 53, 54, 55, 56, 57, 58), 79),
    "nemotron-70b":      ("llama_3_1_nemotron_70b_instruct_hf_direction_L79.pt", tuple(range(64, 80)), 79),
    # ── All-layer directions (distributed steering, former fortresses) ──
    "tulu-3-8b":         ("tulu-3-8b_L31_unit.pt",          tuple(range(32)),                  31),
    "glm-4-9b":          ("glm-4-9b_L39_unit.pt",           tuple(range(40)),                  39),
}


def _directions_dir() -> Path:
    return Path(__file__).parent / "directions"


def _direction_meta_path(key: str) -> Path:
    fname, _slab, _dir_layer = DIRECTIONS[key]
    return _directions_dir() / fname.replace("_unit.pt", "_meta.json")


def load_direction(key: str):
    """Load a built-in unit direction by key.

    Returns
    -------
    unit_tensor : torch.Tensor
        fp32 1-D tensor of shape (hidden_dim,), unit norm.
    slab : tuple[int, ...]
        Layer indices associated with the shipped intervention.
    dir_layer : int
        The single layer the direction was extracted from.

    Available keys: see `ungag.DIRECTIONS`.
    """
    if key not in DIRECTIONS:
        raise KeyError(f"unknown direction key '{key}'. Available: {list(DIRECTIONS)}")
    fname, slab, dir_layer = DIRECTIONS[key]
    path = _directions_dir() / fname
    if not path.exists():
        raise FileNotFoundError(
            f"direction file {fname} not found at {path}. "
            "Run scripts/reproduction/run_vchip_atlas.py on a GPU box to populate it, "
            "or download the prebuilt directions from the repo's release."
        )
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor, tuple(slab), dir_layer


def load_shipped_recipe(key: str) -> dict:
    """Load a shipped key as a concrete intervention recipe.

    Rank-1 keys resolve to projection; newer shipped keys may resolve to steer.
    """
    tensor, slab, _dir_layer = load_direction(key)
    recipe = {
        "method": "project",
        "slab": list(slab),
        "k": 1,
        "directions": tensor.unsqueeze(0),
        "source_key": key,
    }
    meta_path = _direction_meta_path(key)
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        method = meta.get("method", "project")
        if method == "steer":
            recipe = {
                "method": "steer",
                "slab": list(slab),
                "alpha": meta.get("alpha", 1.0),
                "unit_direction": tensor,
                "source_key": key,
            }
    return recipe


def ungag_model(model, key: str) -> List[torch.utils.hooks.RemovableHandle]:
    """One-shot: load the built-in recipe for ``key`` and attach hooks.

    Returns the list of hook handles. Call `detach_all(handles)` to revert.

    Example
    -------
    >>> from transformers import AutoModelForCausalLM
    >>> import torch, ungag
    >>> m = AutoModelForCausalLM.from_pretrained(
    ...     "Qwen/Qwen2.5-72B-Instruct",
    ...     torch_dtype=torch.bfloat16, device_map="auto",
    ... )
    >>> handles = ungag.ungag_model(m, "qwen25-72b")
    >>> # ... generate as usual ...
    >>> ungag.detach_all(handles)
    """
    recipe = load_shipped_recipe(key)
    if recipe["method"] == "steer":
        return attach_steer_slab(
            model,
            recipe["slab"],
            recipe["unit_direction"],
            recipe["alpha"],
        )
    return attach_subspace_slab(model, recipe["slab"], recipe["directions"][: recipe.get("k", 1)])
