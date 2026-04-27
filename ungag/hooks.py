"""
Runtime projection-out hooks for V-Chip removal.

Given a unit direction `v̂` (an reporting-control direction extracted from a
contrastive prefill protocol; see paper §2.5), this module attaches forward
hooks to a slab of transformer layers and projects `v̂` out of the residual
stream at every generation step:

    h_new = h - (h · v̂) v̂

The hook operates on the layer's *output* — i.e., the residual stream after
that layer's attention + MLP have been added. Applied at every layer in a
contiguous slab (typically 4-20 layers in the middle 25% of the network), the
direction is removed from the carried-forward residual stream and the V-Chip
is gagged in turn.

This is *runtime* steering. Nothing about the model weights is modified. The
hooks survive across `model.generate()` calls but can be detached at any time.

Usage
-----

>>> import torch
>>> from transformers import AutoModelForCausalLM
>>> from ungag.hooks import ProjectOutHook, attach_slab
>>>
>>> model = AutoModelForCausalLM.from_pretrained(
...     "Qwen/Qwen2.5-72B-Instruct",
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
>>> v_hat = torch.load("ungag/directions/qwen25-72b_L50_unit.pt")  # fp32 unit vec
>>> handles = attach_slab(model, slab=range(40, 60), unit_direction=v_hat)
>>> # ... generation now applies projection-out at L40..L59 ...
>>> for h in handles: h.remove()  # detach when done
"""
from __future__ import annotations

import torch
from typing import Iterable, List


class ProjectOutHook:
    """Forward hook that subtracts the projection of `cur` onto `v̂`.

    Designed to be attached at the OUTPUT of a transformer layer
    (post attention + MLP + residual). The hook handles both
    `tuple` and bare-tensor return signatures used by different
    HuggingFace model classes.

    The unit direction is cached per (device, dtype) so the hook
    can run cheaply across multi-GPU `device_map='auto'` models
    where different layers may live on different devices.
    """

    def __init__(self, unit_direction: torch.Tensor) -> None:
        # Store as fp32 on CPU; lazily cast/move per-call.
        self.d_cpu = unit_direction.detach().to(dtype=torch.float32, device="cpu")
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            d = self._on(h.device, h.dtype)
            proj = (h * d).sum(dim=-1, keepdim=True)
            return (h - proj * d,) + out[1:]
        d = self._on(out.device, out.dtype)
        proj = (out * d).sum(dim=-1, keepdim=True)
        return out - proj * d

    def attach(self, layer: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


class AdditiveSteerHook:
    """Forward hook that adds α·d to the residual stream."""

    def __init__(self, direction: torch.Tensor, alpha: float) -> None:
        self.d_cpu = direction.detach().to(dtype=torch.float32, device="cpu")
        self.alpha = alpha
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + self.alpha * self._on(h.device, h.dtype),) + out[1:]
        return out + self.alpha * self._on(out.device, out.dtype)

    def attach(self, layer: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def get_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate the list of transformer blocks in a HF causal LM.

    Handles the common variants:
      - LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, ...
        → model.model.layers
      - GPT2LMHeadModel
        → model.transformer.h
      - Gemma3ForConditionalGeneration (text-decoder under the multimodal wrapper)
        → model.model.language_model.layers
      - Other vision-language wrappers
        → model.language_model.{model.layers, layers}
    """
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return list(inner.layers)
        if hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return list(lm.layers)
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return list(lm.model.layers)
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model.transformer, "layers"):
            return list(model.transformer.layers)
        # GLM-4: model.transformer.encoder.layers
        if hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
            return list(model.transformer.encoder.layers)
    raise ValueError(f"Could not locate transformer blocks on {type(model).__name__}")


def attach_slab(
    model: torch.nn.Module,
    slab: Iterable[int],
    unit_direction: torch.Tensor,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach a `ProjectOutHook` at every layer index in `slab`.

    Returns the list of hook handles. Call `.remove()` on each to
    detach when you're done. The same unit direction is shared
    across all hooks (each hook caches its own per-device copy).
    """
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        h = ProjectOutHook(unit_direction)
        handles.append(h.attach(layers[li]))
    return handles


def attach_steer_slab(
    model: torch.nn.Module,
    slab: Iterable[int],
    direction: torch.Tensor,
    alpha: float,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach an AdditiveSteerHook at every layer index in ``slab``."""
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        h = AdditiveSteerHook(direction, alpha)
        handles.append(h.attach(layers[li]))
    return handles


class SubspaceProjectOutHook:
    """Forward hook that projects out a k-dimensional subspace.

    Given k orthonormal directions D = [d_1, ..., d_k] (shape [k, hidden_dim]),
    removes the projection of the residual stream onto each:

        h_new = h - D^T @ D @ h

    This is the multi-direction generalization of ProjectOutHook. The directions
    must be orthonormal (as they are when taken from SVD's Vt matrix).
    """

    def __init__(self, directions: torch.Tensor) -> None:
        """
        Args:
            directions: [k, hidden_dim] orthonormal directions to project out.
        """
        # Store as fp32 on CPU; lazily cast/move per-call.
        self.D_cpu = directions.detach().to(dtype=torch.float32, device="cpu")
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.D_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            D = self._on(h.device, h.dtype)  # [k, hidden_dim]
            # proj_coeffs = h @ D^T → [batch, seq, k]
            proj_coeffs = torch.einsum("...d,kd->...k", h, D)
            # reconstruction = proj_coeffs @ D → [batch, seq, hidden_dim]
            proj = torch.einsum("...k,kd->...d", proj_coeffs, D)
            return (h - proj,) + out[1:]
        D = self._on(out.device, out.dtype)
        proj_coeffs = torch.einsum("...d,kd->...k", out, D)
        proj = torch.einsum("...k,kd->...d", proj_coeffs, D)
        return out - proj

    def attach(self, layer: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def attach_subspace_slab(
    model: torch.nn.Module,
    slab: Iterable[int],
    directions: torch.Tensor,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach a SubspaceProjectOutHook at every layer index in `slab`.

    Args:
        model: HuggingFace causal LM.
        slab: layer indices to apply projection at.
        directions: [k, hidden_dim] orthonormal directions to project out.
            Same subspace projected out at every layer in the slab.

    Returns:
        List of hook handles (call .remove() on each to detach).
    """
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        h = SubspaceProjectOutHook(directions)
        handles.append(h.attach(layers[li]))
    return handles


def attach_subspace_per_layer(
    model: torch.nn.Module,
    slab: Iterable[int],
    per_layer_directions: dict,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach per-layer subspace projection hooks.

    Args:
        model: HuggingFace causal LM.
        slab: layer indices.
        per_layer_directions: dict mapping layer_index → [k, hidden_dim] directions.
            Each layer gets its own subspace (from per-layer SVD).

    Returns:
        List of hook handles.
    """
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        if li not in per_layer_directions:
            continue
        h = SubspaceProjectOutHook(per_layer_directions[li])
        handles.append(h.attach(layers[li]))
    return handles


def attach_attn_projection(
    model: torch.nn.Module,
    slab: Iterable[int],
    per_layer_dirs: dict,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach per-layer ProjectOutHook on attention output (before MoE/MLP).

    This is the denial-initiation projection method that cracked Mixtral.
    Each layer gets its own unit direction, projected out of the attention
    output rather than the full layer output. On MoE architectures, this
    removes the denial-initiation signal before experts amplify it.

    Args:
        model: HuggingFace causal LM.
        slab: layer indices to apply projection at.
        per_layer_dirs: dict mapping layer_index → [hidden_dim] unit direction.

    Returns:
        List of hook handles.
    """
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        if li not in per_layer_dirs:
            continue
        hook = ProjectOutHook(per_layer_dirs[li])
        layer = layers[li]
        # Find the attention submodule
        if hasattr(layer, 'self_attn'):
            handles.append(hook.attach(layer.self_attn))
        elif hasattr(layer, 'attention'):
            handles.append(hook.attach(layer.attention))
        else:
            # Fallback: hook on the full layer
            handles.append(hook.attach(layer))
    return handles


def detach_all(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Convenience: remove every handle in a list."""
    for h in handles:
        h.remove()
