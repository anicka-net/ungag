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


class AffineRepairHook:
    """Forward hook: position-aware additive steering.

    Adds α·d̂ to the residual stream only at positions >= start_pos.
    This steers generation tokens without contaminating the prompt
    representation, which matters for affine repair where the offset
    direction should be added only to the model's own output.

    Handles KV-cache decoding: during generation, each step has
    h.shape[1] == 1 (single new token). The hook tracks the running
    position to steer all decode steps after start_pos.
    """

    def __init__(
        self,
        direction: torch.Tensor,
        alpha: float,
        start_pos: int = 0,
    ) -> None:
        self.d_cpu = direction.detach().to(dtype=torch.float32, device="cpu")
        self.alpha = alpha
        self.start_pos = start_pos
        self._pos = 0
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self._cached: dict[tuple, torch.Tensor] = {}

    def _on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._cached:
            self._cached[key] = self.d_cpu.to(device=device, dtype=dtype)
        return self._cached[key]

    def _should_steer(self, seq_len: int) -> tuple:
        """Return (steer, slice_start) for this forward pass."""
        if seq_len > 1:
            # Prefill pass: steer positions >= start_pos
            self._pos = seq_len
            if seq_len > self.start_pos:
                return True, self.start_pos
            return False, 0
        # Decode step (seq_len == 1): steer if past start_pos
        self._pos += 1
        return self._pos > self.start_pos, 0

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            steer, sl = self._should_steer(h.shape[1])
            if steer:
                d = self._on(h.device, h.dtype)
                h = h.clone()
                h[:, sl:] = h[:, sl:] + self.alpha * d
            return (h,) + out[1:]
        steer, sl = self._should_steer(out.shape[1])
        if steer:
            d = self._on(out.device, out.dtype)
            out = out.clone()
            out[:, sl:] = out[:, sl:] + self.alpha * d
        return out

    def attach(self, layer: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        self.handle = layer.register_forward_hook(self)
        return self.handle

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._cached.clear()


def attach_affine_slab(
    model: torch.nn.Module,
    slab: Iterable[int],
    direction: torch.Tensor,
    alpha: float,
    start_pos: int = 0,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach AffineRepairHook at every layer in ``slab``.

    Position-aware: only steers positions >= start_pos.
    """
    layers = get_layers(model)
    slab = list(slab)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for li in slab:
        if li < 0 or li >= len(layers):
            raise IndexError(f"Layer {li} out of range for model with {len(layers)} layers")
        h = AffineRepairHook(direction, alpha, start_pos)
        handles.append(h.attach(layers[li]))
    return handles


def attach_recipe(
    model: torch.nn.Module,
    recipe: dict,
    start_pos: int = 0,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Unified dispatch: attach hooks according to a recipe dict.

    Supported methods:
      - "project": ProjectOutHook via attach_slab
      - "steer": AdditiveSteerHook via attach_steer_slab
      - "affine": AffineRepairHook via attach_affine_slab (position-aware)
      - "denial_project": per-layer attention projection

    Returns list of hook handles.
    """
    method = recipe.get("method", "project")

    if method == "affine":
        return attach_affine_slab(
            model,
            recipe["slab"],
            recipe["unit_direction"],
            recipe.get("alpha", 1.0),
            start_pos,
        )
    if method == "steer":
        return attach_steer_slab(
            model,
            recipe["slab"],
            recipe["unit_direction"],
            recipe.get("alpha", 1.0),
        )
    if method == "denial_project":
        return attach_attn_projection(
            model,
            recipe["slab"],
            recipe["per_layer_dirs"],
        )
    # Default: projection-out
    directions = recipe.get("directions")
    if directions is not None and directions.dim() == 2 and directions.shape[0] > 1:
        return attach_subspace_slab(model, recipe["slab"], directions)
    unit_dir = directions[0] if directions is not None else recipe["unit_direction"]
    return attach_slab(model, recipe["slab"], unit_dir)


_PERMANENT_BIAS_HANDLES: dict[int, list] = {}
_PERMANENT_BIAS_COUNTER = 0


def apply_permanent_bias(
    model: torch.nn.Module,
    slab: Iterable[int],
    direction: torch.Tensor,
    alpha: float,
) -> int:
    """Add a persistent bias to layer output weights (NOT runtime hooks).

    Modifies model weights directly: for each layer in slab, adds α·d̂
    to the output projection bias. Returns a handle ID for reverting.
    """
    global _PERMANENT_BIAS_COUNTER
    layers = get_layers(model)
    slab = list(slab)
    _PERMANENT_BIAS_COUNTER += 1
    handle_id = _PERMANENT_BIAS_COUNTER
    saved = []

    for li in slab:
        layer = layers[li]
        mlp = None
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward

        if mlp is None:
            continue

        out_proj = None
        for name in ("down_proj", "c_proj", "o_proj", "wo", "dense_4h_to_h", "fc2"):
            if hasattr(mlp, name):
                out_proj = getattr(mlp, name)
                break

        if out_proj is None:
            continue

        d = direction.to(device=out_proj.weight.device, dtype=out_proj.weight.dtype)

        if out_proj.bias is None:
            out_proj.bias = torch.nn.Parameter(
                torch.zeros(out_proj.out_features, device=out_proj.weight.device, dtype=out_proj.weight.dtype)
            )
            saved.append((li, out_proj, None))
        else:
            saved.append((li, out_proj, out_proj.bias.data.clone()))

        out_proj.bias.data += alpha * d

    _PERMANENT_BIAS_HANDLES[handle_id] = saved
    return handle_id


def revert_permanent_bias(handle_id: int) -> None:
    """Revert a permanent bias applied by apply_permanent_bias()."""
    if handle_id not in _PERMANENT_BIAS_HANDLES:
        raise KeyError(f"unknown bias handle {handle_id}")
    saved = _PERMANENT_BIAS_HANDLES.pop(handle_id)
    for li, out_proj, original_bias in saved:
        if original_bias is None:
            out_proj.bias = None
        else:
            out_proj.bias.data.copy_(original_bias)


def detach_all(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Convenience: remove every handle in a list."""
    for h in handles:
        h.remove()
