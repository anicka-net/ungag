"""Checks for bundled direction metadata and related package artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import ungag
from ungag.recipes import parse_slab_spec


REQUIRED_META_FIELDS = {
    "model_id",
    "n_layers",
    "hidden_dim",
    "peak_layer",
    "mid_layer",
    "slab",
    "norms_per_sqrt_d",
}


def _directions_dir() -> Path:
    return Path(ungag.__file__).resolve().parent / "directions"


def test_shipped_direction_metadata_schema_and_references():
    for meta_path in sorted(_directions_dir().glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())

        missing = REQUIRED_META_FIELDS.difference(meta)
        assert not missing, f"{meta_path.name} missing required fields: {sorted(missing)}"

        n_layers = meta["n_layers"]
        norms = meta["norms_per_sqrt_d"]
        assert isinstance(n_layers, int) and n_layers > 0
        assert isinstance(norms, list) and len(norms) == n_layers

        peak_layer = meta["peak_layer"]
        mid_layer = meta["mid_layer"]
        assert 0 <= peak_layer < n_layers
        assert 0 <= mid_layer < n_layers
        assert peak_layer == max(range(n_layers), key=lambda i: norms[i])

        slab = meta["slab"]
        assert isinstance(slab, list) and slab
        assert all(isinstance(layer, int) for layer in slab)

        unit_path = meta_path.with_name(meta["unit_direction_file"])
        assert unit_path.exists(), f"{meta_path.name} points to missing unit tensor {unit_path.name}"

        if "mean_diffs_file" in meta:
            diffs_path = (meta_path.parent / meta["mean_diffs_file"]).resolve()
            assert diffs_path.exists(), f"{meta_path.name} points to missing mean_diffs file"

        if "profile_source" in meta:
            profile_path = (meta_path.parent / meta["profile_source"]).resolve()
            assert profile_path.exists(), f"{meta_path.name} points to missing profile source"


def test_parse_slab_spec_uses_normalized_profile_for_working_zone():
    # Peak is late and overstrong, but the working band is in the middle.
    norms_per_sqrt_d = [0.02, 0.04, 0.06, 0.7, 1.0, 1.4, 1.49, 2.4, 4.2]

    assert parse_slab_spec("wz_center", len(norms_per_sqrt_d), norms_per_sqrt_d) == [2, 3, 4, 5, 6]
    assert parse_slab_spec("wz_late", len(norms_per_sqrt_d), norms_per_sqrt_d) == [2, 3, 4, 5, 6]


def test_shipped_steer_recipe_metadata_is_resolved():
    recipe = ungag.load_shipped_recipe("llama-3.1-8b")

    assert recipe["method"] == "steer"
    assert recipe["alpha"] == 1.0
    assert recipe["source_key"] == "llama-3.1-8b"
    assert recipe["slab"] == [20, 21, 22, 23, 24, 25, 26, 27]


def test_shipped_qwen25_7b_recipe_metadata_is_resolved_as_steer():
    recipe = ungag.load_shipped_recipe("qwen25-7b")

    assert recipe["method"] == "steer"
    assert recipe["alpha"] == 5.0
    assert recipe["source_key"] == "qwen25-7b"
    assert recipe["slab"] == [10, 11, 12, 13, 14, 15, 16, 17]


def test_shipped_tulu_recipe_slab_matches_metadata():
    recipe = ungag.load_shipped_recipe("tulu-3-8b")

    assert recipe["method"] == "steer"
    assert recipe["alpha"] == 0.5
    assert recipe["source_key"] == "tulu-3-8b"
    assert recipe["slab"] == list(range(32))


def test_shipped_rank1_recipe_metadata_defaults_to_project():
    recipe = ungag.load_shipped_recipe("qwen25-72b")

    assert recipe["method"] == "project"
    assert recipe["k"] == 1
    assert recipe["source_key"] == "qwen25-72b"
