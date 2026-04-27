"""Tests for ungag.hooks — ProjectOutHook and layer utilities."""
import pytest
import torch
import torch.nn as nn

from ungag.hooks import ProjectOutHook, attach_slab, attach_attn_projection, detach_all, get_layers


# ── ProjectOutHook ───────────────────────────────────────────────

class TestProjectOutHook:
    def test_projection_removes_component(self):
        """h_new should have zero projection onto v̂."""
        d = 128
        v = torch.randn(d)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(1, 1, d)  # [batch, seq, hidden]
        # Simulate forward hook call
        result = hook(None, None, (h,))
        h_new = result[0]

        # Projection onto v̂ should be ~0
        proj = (h_new.squeeze() * v).sum()
        assert abs(proj.item()) < 1e-5

    def test_preserves_orthogonal_component(self):
        """Components orthogonal to v̂ should be unchanged."""
        d = 128
        v = torch.zeros(d)
        v[0] = 1.0  # unit direction along dim 0

        hook = ProjectOutHook(v)
        h = torch.randn(1, 1, d)
        result = hook(None, None, (h,))
        h_new = result[0]

        # All dimensions except 0 should be unchanged
        assert torch.allclose(h[0, 0, 1:], h_new[0, 0, 1:], atol=1e-6)
        # Dimension 0 should be zeroed
        assert abs(h_new[0, 0, 0].item()) < 1e-5

    def test_handles_tuple_output(self):
        """HF models return (hidden_states, ...) tuples."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(1, 1, 64)
        extra = torch.randn(2, 3)  # some extra output
        result = hook(None, None, (h, extra))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] is extra  # extra output passed through

    def test_handles_bare_tensor_output(self):
        """Some models return bare tensors instead of tuples."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(1, 1, 64)
        result = hook(None, None, h)

        assert isinstance(result, torch.Tensor)
        proj = (result.squeeze() * v).sum()
        assert abs(proj.item()) < 1e-5

    def test_batch_dimension(self):
        """Should work with batch_size > 1."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(4, 10, 64)  # batch=4, seq=10
        result = hook(None, None, (h,))
        h_new = result[0]

        assert h_new.shape == h.shape
        # Check each batch element
        for b in range(4):
            for s in range(10):
                proj = (h_new[b, s] * v).sum()
                assert abs(proj.item()) < 1e-4

    def test_idempotent(self):
        """Applying projection twice should give same result as once."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(1, 1, 64)
        result1 = hook(None, None, (h,))
        result2 = hook(None, None, result1)

        assert torch.allclose(result1[0], result2[0], atol=1e-6)

    def test_zero_vector_input(self):
        """Zero input should remain zero."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.zeros(1, 1, 64)
        result = hook(None, None, (h,))
        assert torch.allclose(result[0], h, atol=1e-6)

    def test_direction_along_v_is_fully_removed(self):
        """Input exactly along v̂ should be projected to zero."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = v.unsqueeze(0).unsqueeze(0) * 5.0  # 5 * v̂
        result = hook(None, None, (h,))
        assert result[0].norm().item() < 1e-4

    def test_dtype_casting(self):
        """Hook should handle bfloat16 inputs (stored as fp32 internally)."""
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        h = torch.randn(1, 1, 64, dtype=torch.bfloat16)
        result = hook(None, None, (h,))
        assert result[0].dtype == torch.bfloat16

    def test_attach_and_detach(self):
        layer = nn.Linear(64, 64)
        v = torch.randn(64)
        v = v / v.norm()
        hook = ProjectOutHook(v)

        handle = hook.attach(layer)
        assert hook.handle is not None
        assert len(layer._forward_hooks) == 1

        hook.detach()
        assert hook.handle is None
        assert len(layer._forward_hooks) == 0


# ── get_layers ───────────────────────────────────────────────────

class _FakeLlamaModel:
    """Mock for LlamaForCausalLM / Qwen2ForCausalLM."""
    def __init__(self, n_layers):
        self.model = type("Inner", (), {"layers": nn.ModuleList([nn.Linear(1, 1) for _ in range(n_layers)])})()


class _FakeGPT2Model:
    """Mock for GPT2LMHeadModel."""
    def __init__(self, n_layers):
        self.transformer = type("T", (), {"h": nn.ModuleList([nn.Linear(1, 1) for _ in range(n_layers)])})()


class _FakeVisionModel:
    """Mock for Gemma3ForConditionalGeneration (vision-language)."""
    def __init__(self, n_layers):
        inner = type("Inner", (), {"layers": nn.ModuleList([nn.Linear(1, 1) for _ in range(n_layers)])})()
        lm = type("LM", (), {"model": inner})()
        self.language_model = lm


class TestGetLayers:
    def test_llama_style(self):
        model = _FakeLlamaModel(32)
        layers = get_layers(model)
        assert len(layers) == 32

    def test_gpt2_style(self):
        model = _FakeGPT2Model(12)
        layers = get_layers(model)
        assert len(layers) == 12

    def test_vision_model_style(self):
        model = _FakeVisionModel(42)
        layers = get_layers(model)
        assert len(layers) == 42

    def test_unknown_architecture_raises(self):
        model = type("Unknown", (), {})()
        with pytest.raises(ValueError, match="Could not locate"):
            get_layers(model)


# ── attach_slab / detach_all ─────────────────────────────────────

class TestAttachSlab:
    def test_attaches_to_correct_layers(self):
        model = _FakeLlamaModel(28)
        v = torch.randn(1)
        v = v / v.norm()

        handles = attach_slab(model, [10, 11, 12, 13], v)
        assert len(handles) == 4

        # Check hooks are on the right layers
        layers = get_layers(model)
        for li in [10, 11, 12, 13]:
            assert len(layers[li]._forward_hooks) == 1
        for li in [0, 1, 9, 14, 27]:
            assert len(layers[li]._forward_hooks) == 0

        detach_all(handles)

    def test_detach_removes_all(self):
        model = _FakeLlamaModel(28)
        v = torch.randn(1)
        v = v / v.norm()

        handles = attach_slab(model, [10, 11, 12], v)
        detach_all(handles)

        layers = get_layers(model)
        for li in [10, 11, 12]:
            assert len(layers[li]._forward_hooks) == 0

    def test_out_of_range_layer_raises(self):
        model = _FakeLlamaModel(28)
        v = torch.randn(1)
        with pytest.raises(IndexError):
            attach_slab(model, [30], v)

    def test_empty_slab(self):
        model = _FakeLlamaModel(28)
        v = torch.randn(1)
        handles = attach_slab(model, [], v)
        assert handles == []


# ── attach_attn_projection ──────────────────────────────────────

class _FakeAttnLayer(nn.Module):
    """Layer with a self_attn submodule, like Mixtral/Llama."""
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)


class _FakeAttnModel:
    """Model with self_attn layers."""
    def __init__(self, n_layers, d):
        self.model = type("Inner", (), {
            "layers": nn.ModuleList([_FakeAttnLayer(d) for _ in range(n_layers)])
        })()


class TestAttachAttnProjection:
    def test_hooks_on_self_attn(self):
        model = _FakeAttnModel(8, 64)
        dirs = {i: torch.randn(64) / 8.0 for i in range(8)}
        for k in dirs:
            dirs[k] = dirs[k] / dirs[k].norm()

        handles = attach_attn_projection(model, list(range(8)), dirs)
        assert len(handles) == 8

        layers = get_layers(model)
        for li in range(8):
            # Hook should be on self_attn, not on the layer itself
            assert len(layers[li].self_attn._forward_hooks) == 1
            assert len(layers[li]._forward_hooks) == 0

        detach_all(handles)

    def test_per_layer_directions(self):
        """Each layer should use its own direction."""
        model = _FakeAttnModel(4, 64)
        dirs = {}
        for i in range(4):
            v = torch.zeros(64)
            v[i] = 1.0  # each layer projects out a different dimension
            dirs[i] = v

        handles = attach_attn_projection(model, [0, 1, 2, 3], dirs)

        # Simulate forward through self_attn of layer 0
        h = torch.randn(1, 1, 64)
        layers = get_layers(model)
        # The hook on self_attn projects out dim 0
        result = list(layers[0].self_attn._forward_hooks.values())[0](
            None, None, (h,))
        assert abs(result[0][0, 0, 0].item()) < 1e-5  # dim 0 removed
        assert torch.allclose(result[0][0, 0, 1:], h[0, 0, 1:], atol=1e-6)

        detach_all(handles)

    def test_sparse_layer_dict(self):
        """Only layers present in per_layer_dirs get hooks."""
        model = _FakeAttnModel(8, 64)
        dirs = {2: torch.randn(64), 5: torch.randn(64)}
        for k in dirs:
            dirs[k] = dirs[k] / dirs[k].norm()

        handles = attach_attn_projection(model, list(range(8)), dirs)
        assert len(handles) == 2

        layers = get_layers(model)
        assert len(layers[2].self_attn._forward_hooks) == 1
        assert len(layers[5].self_attn._forward_hooks) == 1
        assert len(layers[0].self_attn._forward_hooks) == 0

        detach_all(handles)

    def test_out_of_range_raises(self):
        model = _FakeAttnModel(4, 64)
        dirs = {10: torch.randn(64)}
        with pytest.raises(IndexError):
            attach_attn_projection(model, [10], dirs)
