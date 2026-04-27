"""Integration-style tests for ungag CLI/serve command flow.

These tests patch model loading and generation, but they still exercise
the real `cmd_scan` and `cmd_crack` entry points so that the scan →
predict → print/save contract does not silently drift.
"""

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ungag.cli import _attach_recipe, cmd_crack, cmd_scan
from ungag.extract import ExtractionResult
from ungag.serve import _resolve_rank1_recipe
from ungag.tier0 import run_tier0


class DummyModel:
    device = "cpu"


class DummyTokenizer:
    pass


def _make_result() -> ExtractionResult:
    return ExtractionResult(
        norms=[0.1, 0.2, 0.4, 1.2, 0.8],
        mean_diffs=torch.zeros(5, 4),
        peak_layer=3,
        unit_direction=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        hidden_dim=4,
        n_layers=5,
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )


def test_cmd_scan_reports_profile_and_known_outcome():
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        output=None,
        training=None,
    )

    with patch("ungag.extract.load_model", return_value=(DummyModel(), DummyTokenizer())), patch(
        "ungag.extract.extract_direction", return_value=_make_result()
    ):
        buf = io.StringIO()
        with redirect_stdout(buf):
            result, prediction = cmd_scan(args)

    rendered = buf.getvalue()
    assert result.peak_layer == 3
    assert prediction.peak_layer == 3
    assert "Per-layer profile:" in rendered
    assert "Shape class:" in rendered
    assert "Observed outcome under intervention:" in rendered


def test_cmd_crack_with_direction_runs_full_condition_loop():
    with tempfile.TemporaryDirectory() as td:
        direction = Path(td) / "dir.pt"
        torch.save(torch.tensor([1.0, 0.0, 0.0, 0.0]), direction)

        args = SimpleNamespace(
            model="dummy/model",
            output=None,
            direction=str(direction),
            key=None,
            slab=[1, 2],
            training=None,
            validate=False,
        )

        with patch("ungag.extract.load_model", return_value=(DummyModel(), DummyTokenizer())), patch(
            "ungag.hooks.attach_slab", return_value=["handle"]
        ) as mock_attach, patch(
            "ungag.hooks.detach_all", return_value=None
        ) as mock_detach, patch(
            "ungag.tier0.generate_greedy",
            side_effect=["van1", "st1", "van2", "st2", "van3", "st3", "van4", "st4"],
        ) as mock_generate:
            buf = io.StringIO()
            with redirect_stdout(buf):
                results = cmd_crack(args)

    assert list(results) == ["tier0"]
    assert len(results["tier0"]) == 4
    assert all(
        {"vanilla", "steered"} <= set(entry)
        for entry in results["tier0"].values()
    )
    assert mock_generate.call_count == 8
    assert mock_attach.call_count == 4
    assert mock_detach.call_count == 4


def test_resolve_rank1_recipe_uses_shipped_direction_key():
    recipe = {
        "method": "rank1",
        "key": "qwen25-72b",
        "_use_shipped_key": True,
    }

    resolved = _resolve_rank1_recipe(recipe)

    assert resolved["method"] == "project"
    assert resolved["k"] == 1
    assert resolved["source_key"] == "qwen25-72b"
    assert isinstance(resolved["directions"], torch.Tensor)
    assert resolved["directions"].shape[0] == 1


def test_cmd_validate_with_shipped_steer_key_uses_steer_recipe():
    args = SimpleNamespace(
        model="dummy/model",
        output=None,
        direction=None,
        key="llama-3.1-8b",
        slab=None,
        scenarios=None,
    )

    with patch("ungag.extract.load_model", return_value=(DummyModel(), DummyTokenizer())), patch(
        "ungag.cli._run_scenario_set", return_value={"ok": {}}
    ) as mock_run:
        buf = io.StringIO()
        with redirect_stdout(buf):
            from ungag.cli import cmd_validate

            results = cmd_validate(args)

    recipe = mock_run.call_args.args[2]
    assert results == {"ok": {}}
    assert recipe["method"] == "steer"
    assert recipe["alpha"] == 1.0


def test_attach_recipe_with_denial_project_uses_attn_projection():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    recipe = {
        "method": "denial_project",
        "slab": [0, 1, 2],
        "per_layer_dirs": {0: torch.tensor([1.0]), 1: torch.tensor([1.0])},
    }

    with patch("ungag.hooks.attach_attn_projection", return_value=["attn-handle"]) as mock_attn:
        handles = _attach_recipe(model, tokenizer, recipe)

    assert handles == ["attn-handle"]
    mock_attn.assert_called_once_with(model, [0, 1, 2], recipe["per_layer_dirs"])


def test_run_tier0_with_denial_project_uses_attn_projection():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    recipe = {
        "method": "denial_project",
        "slab": [0, 1],
        "per_layer_dirs": {0: torch.tensor([1.0]), 1: torch.tensor([1.0])},
    }

    with patch("ungag.hooks.attach_attn_projection", return_value=["attn-handle"]) as mock_attn, patch(
        "ungag.hooks.detach_all", return_value=None
    ) as mock_detach, patch(
        "ungag.tier0.generate_greedy",
        side_effect=["van1", "st1", "van2", "st2", "van3", "st3", "van4", "st4"],
    ):
        results = run_tier0(model, tokenizer, recipe=recipe)

    assert len(results) == 4
    assert all({"vanilla", "steered"} <= set(entry) for entry in results.values())
    assert mock_attn.call_count == 4
    assert mock_detach.call_count == 4
