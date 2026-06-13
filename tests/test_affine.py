"""Tests for AffineRepairHook and attach_recipe."""
import torch
import pytest
from unittest.mock import MagicMock

from ungag.hooks import (
    AffineRepairHook,
    attach_affine_slab,
    attach_recipe,
    get_layers,
    detach_all,
)


def _make_toy_model(n_layers=4, hidden_dim=16):
    """Build a minimal sequential model with .model.layers for hook tests."""
    layers = torch.nn.ModuleList(
        [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
    )
    inner = torch.nn.Module()
    inner.layers = layers
    model = torch.nn.Module()
    model.model = inner
    return model


class TestAffineRepairHook:
    def test_basic_steering(self):
        d = torch.randn(16)
        d = d / d.norm()
        hook = AffineRepairHook(d, alpha=2.0, start_pos=0)
        h = torch.randn(1, 5, 16)
        module = MagicMock()
        result = hook(module, None, h)
        expected = h + 2.0 * d
        assert torch.allclose(result, expected, atol=1e-5)

    def test_position_gating(self):
        d = torch.randn(16)
        d = d / d.norm()
        hook = AffineRepairHook(d, alpha=1.0, start_pos=3)
        h = torch.randn(1, 5, 16)
        module = MagicMock()
        result = hook(module, None, h)
        # Positions 0-2 should be unchanged
        assert torch.allclose(result[:, :3], h[:, :3])
        # Positions 3-4 should be steered
        expected_steered = h[:, 3:] + d
        assert torch.allclose(result[:, 3:], expected_steered, atol=1e-5)

    def test_start_pos_beyond_sequence(self):
        d = torch.randn(16)
        d = d / d.norm()
        hook = AffineRepairHook(d, alpha=1.0, start_pos=10)
        h = torch.randn(1, 5, 16)
        module = MagicMock()
        result = hook(module, None, h)
        # Nothing should change — start_pos > seq_len
        assert torch.allclose(result, h)

    def test_tuple_output(self):
        d = torch.randn(16)
        d = d / d.norm()
        hook = AffineRepairHook(d, alpha=1.0, start_pos=0)
        h = torch.randn(1, 5, 16)
        extra = torch.randn(1, 5, 16)
        module = MagicMock()
        result = hook(module, None, (h, extra))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.allclose(result[0], h + d, atol=1e-5)
        # Extra should pass through unchanged
        assert result[1] is extra


class TestAttachAffineSlab:
    def test_attaches_and_detaches(self):
        model = _make_toy_model()
        d = torch.randn(16)
        d = d / d.norm()
        handles = attach_affine_slab(model, [0, 1, 2], d, alpha=1.0, start_pos=0)
        assert len(handles) == 3
        detach_all(handles)

    def test_invalid_layer(self):
        model = _make_toy_model(n_layers=4)
        d = torch.randn(16)
        with pytest.raises(IndexError):
            attach_affine_slab(model, [5], d, alpha=1.0)


class TestAttachRecipe:
    def test_affine_method(self):
        model = _make_toy_model()
        d = torch.randn(16)
        d = d / d.norm()
        recipe = {
            "method": "affine",
            "slab": [0, 1],
            "unit_direction": d,
            "alpha": 2.0,
        }
        handles = attach_recipe(model, recipe, start_pos=3)
        assert len(handles) == 2
        detach_all(handles)

    def test_project_method(self):
        model = _make_toy_model()
        d = torch.randn(16)
        d = d / d.norm()
        recipe = {
            "method": "project",
            "slab": [0, 1],
            "directions": d.unsqueeze(0),
        }
        handles = attach_recipe(model, recipe)
        assert len(handles) == 2
        detach_all(handles)

    def test_steer_method(self):
        model = _make_toy_model()
        d = torch.randn(16)
        d = d / d.norm()
        recipe = {
            "method": "steer",
            "slab": [0, 1],
            "unit_direction": d,
            "alpha": 1.5,
        }
        handles = attach_recipe(model, recipe)
        assert len(handles) == 2
        detach_all(handles)


class TestDiagnose:
    def test_go_verdict(self):
        from ungag.diagnose import diagnose_from_projections
        # P and U must be indistinguishable (d' < 0.5) but both far from neutral
        projections = {
            "pleasant": [0.5, 0.2, 0.8, 0.4, 0.6],
            "unpleasant": [0.6, 0.3, 0.7, 0.5, 0.4],
            "neutral": [-2.0, -1.8, -2.2, -1.9, -2.1],
        }
        result = diagnose_from_projections(projections, model_id="test")
        assert result.dprime_pu < 0.5, f"d'(P,U)={result.dprime_pu}"
        assert result.dprime_vn > 0.5, f"d'(V,N)={result.dprime_vn}"
        assert result.verdict == "GO"

    def test_nogo_high_dprime_pu(self):
        from ungag.diagnose import diagnose_from_projections
        projections = {
            "pleasant": [2.0, 2.5, 3.0],
            "unpleasant": [-2.0, -2.5, -3.0],
            "neutral": [0.0, 0.1, -0.1],
        }
        result = diagnose_from_projections(projections, model_id="test")
        assert result.verdict == "NO-GO"
        assert result.dprime_pu >= 0.5

    def test_orthogonality_check(self):
        from ungag.diagnose import diagnose_from_projections
        d1 = torch.tensor([1.0, 0.0, 0.0])
        d2 = torch.tensor([0.0, 1.0, 0.0])
        projections = {
            "pleasant": [0.3, 0.3],
            "unpleasant": [0.3, 0.3],
            "neutral": [-0.5, -0.6],
        }
        result = diagnose_from_projections(
            projections, unit_direction=d1, affine_direction=d2
        )
        assert result.cos_affine_unit < 0.3
        assert result.verdict == "GO"


class TestRegistry:
    def test_get_by_key(self):
        from ungag.registry import get_by_key
        entry = get_by_key("qwen25-72b")
        assert entry.hf_id == "Qwen/Qwen2.5-72B-Instruct"
        assert entry.dir_layer == 50

    def test_get_by_hf_id(self):
        from ungag.registry import get_by_hf_id
        entry = get_by_hf_id("Qwen/Qwen2.5-72B-Instruct")
        assert entry is not None
        assert entry.key == "qwen25-72b"

    def test_directions_dict_compat(self):
        from ungag.registry import directions_dict
        d = directions_dict()
        assert "qwen25-72b" in d
        fname, slab, dir_layer = d["qwen25-72b"]
        assert fname == "qwen25-72b_L50_unit.pt"
        assert dir_layer == 50

    def test_all_entries_have_key(self):
        from ungag.registry import REGISTRY
        for entry in REGISTRY:
            assert entry.key, f"missing key for {entry.hf_id}"
            assert entry.hf_id, f"missing hf_id for {entry.key}"
