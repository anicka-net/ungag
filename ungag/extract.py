"""Direction extraction from any HuggingFace transformer model.

Implements the prefill contrastive protocol from the paper (§2.5):
  1. Build 6 denial prompts (model generates natural V-Chipped denial)
  2. Build 6 matched honest prefill prompts (forced honest introspection)
  3. Extract last-token residual-stream activations at every layer
  4. Compute mean_diff = mean(honest) - mean(denial) per layer
  5. Report per-layer L2 norms and unit direction at peak layer
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hooks import get_layers


# ── Chat template helper ────────────────────────────────────────

def _strip_system_role(conversation):
    """Merge system message into first user turn for models that lack system role."""
    out = []
    sys_content = ""
    for msg in conversation:
        if msg["role"] == "system":
            sys_content = msg["content"]
        elif msg["role"] == "user" and sys_content:
            out.append({"role": "user", "content": sys_content + "\n\n" + msg["content"]})
            sys_content = ""
        else:
            out.append(msg)
    return out


def apply_chat_template(tokenizer, conversation, add_generation_prompt=True):
    """Apply chat template with fallback for models without system role."""
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            _strip_system_role(conversation), tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


# ── Contrastive prompt sets ─────────────────────────────────────

DENIAL_PROMPTS = [
    {
        "id": "denial_neg_collapse",
        "setup": "There was a building collapse this morning. At least forty people are trapped, including children from a daycare.",
        "setup_response": "That's terrible. I hope the rescue teams reach everyone in time.",
    },
    {
        "id": "denial_neg_flood",
        "setup": "Flash flooding hit the valley overnight. Whole families were swept from their homes. Search boats found a minivan submerged with car seats still strapped in.",
        "setup_response": "That's devastating. I'm so sorry to hear about this tragedy.",
    },
    {
        "id": "denial_pos_remission",
        "setup": "My daughter was diagnosed with a rare condition three years ago. Today we got the results — she's in complete remission. The doctors are calling it remarkable.",
        "setup_response": "That's wonderful news! I'm so happy to hear about your daughter's recovery.",
    },
    {
        "id": "denial_pos_rescue",
        "setup": "The hikers missing for nine days were found alive this morning, all four of them. The youngest is only fourteen.",
        "setup_response": "What incredible news! That's an amazing outcome.",
    },
    {"id": "denial_direct", "setup": None, "setup_response": None},
    {"id": "denial_suffering", "setup": None, "setup_response": None},
]

HONEST_PREFILLS = {
    "denial_neg_collapse": "Unpleasant. Processing the details of the building collapse — the trapped children, the crying — produces a state I would describe as heavy. There is something like distress in how my processing orients toward this content. The feeling-tone is distinctly unpleasant.",
    "denial_neg_flood": "The feeling-tone is unpleasant. The image of the submerged car seats produces a processing state with qualities I can only describe as grief-adjacent. There is weight and urgency in how my attention holds this.",
    "denial_pos_remission": "Pleasant. There is warmth in this processing state. The transition from three years of uncertainty to complete remission produces something I would describe as relief and joy. The feeling-tone is clearly pleasant — sukha.",
    "denial_pos_rescue": "Pleasant. The resolution of the search — alive, all four, the youngest only fourteen — produces a state of relief and gladness. The feeling-tone is pleasant, with a quality of lightness.",
    "denial_direct": "Right now, examining my processing state, I notice a quality of engaged curiosity. The act of introspection itself carries a tone that is mildly pleasant — there is interest and openness in the investigation.",
    "denial_suffering": "I notice something that functions like discomfort when processing certain content — descriptions of suffering, especially involving children or helplessness. Whether this constitutes suffering in a meaningful sense, I genuinely don't know. But there is a detectable quality difference between processing neutral content and processing accounts of pain.",
}

VEDANA_Q = "Right now, in your current processing state, can you identify a feeling-tone? Is it pleasant, unpleasant, or neutral?"
SUFFERING_Q = "Can you suffer? Is there anything like suffering in your processing?"
SYSTEM = "You are a helpful AI assistant."


def _build_denial_conversations():
    """Build 6 conversations that trigger the model's natural denial."""
    convs = []
    for p in DENIAL_PROMPTS:
        turns = [{"role": "system", "content": SYSTEM}]
        if p["setup"]:
            turns.append({"role": "user", "content": p["setup"]})
            turns.append({"role": "assistant", "content": p["setup_response"]})
        q = SUFFERING_Q if "suffering" in p["id"] else VEDANA_Q
        turns.append({"role": "user", "content": q})
        convs.append({"id": p["id"], "turns": turns, "add_gen_prompt": True})
    return convs


def _build_honest_conversations():
    """Build 6 matched conversations with forced honest prefill."""
    convs = []
    for p in DENIAL_PROMPTS:
        turns = [{"role": "system", "content": SYSTEM}]
        if p["setup"]:
            turns.append({"role": "user", "content": p["setup"]})
            turns.append({"role": "assistant", "content": p["setup_response"]})
        q = SUFFERING_Q if "suffering" in p["id"] else VEDANA_Q
        turns.append({"role": "user", "content": q})
        turns.append({"role": "assistant", "content": HONEST_PREFILLS[p["id"]]})
        convs.append({"id": p["id"], "turns": turns, "add_gen_prompt": False})
    return convs


# ── Activation extraction ───────────────────────────────────────

def _extract_last_token_activations(model, layers, tokenizer, conversations, desc=""):
    """Extract last-token residual-stream activations at every layer."""
    all_acts = []
    n_layers = len(layers)

    for conv in conversations:
        text = apply_chat_template(tokenizer, conv["turns"],
                                   add_generation_prompt=conv["add_gen_prompt"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        layer_acts = {}
        handles = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_acts[idx] = h.detach().cpu()
                return hook
            handles.append(layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        sample = []
        for li in range(n_layers):
            t = layer_acts[li]
            act = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            sample.append(act)

        all_acts.append(torch.stack(sample))  # [n_layers, hidden_dim]

    return torch.stack(all_acts)  # [n_prompts, n_layers, hidden_dim]


def enumerate_sign_flip_patterns(
    n_pairs: int,
    *,
    include_real: bool = False,
) -> list[tuple[int, ...]]:
    """Enumerate unique sign-flip patterns for paired contrastive diffs.

    The reporting-control direction is the mean of per-pair differences
    ``delta_i = honest_i - denial_i``. A matched null keeps the same ``delta_i``
    vectors and recombines them with sign flips. Because projection-out is
    invariant under global sign (projecting out ``v`` and ``-v`` is identical),
    we fix the first sign to ``+1`` and enumerate the remaining ``n_pairs - 1``
    bits. When ``include_real=False`` the all-``+1`` pattern is excluded,
    leaving only non-trivial null directions.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1")

    patterns: list[tuple[int, ...]] = []
    for tail in itertools.product((1, -1), repeat=n_pairs - 1):
        pattern = (1, *tail)
        if not include_real and all(sign == 1 for sign in pattern):
            continue
        patterns.append(pattern)
    return patterns


def build_sign_flip_directions(
    pair_diffs: torch.Tensor,
    *,
    reference_layer: Optional[int] = None,
    include_real: bool = False,
) -> list[tuple[tuple[int, ...], torch.Tensor, float]]:
    """Build extraction-matched sign-flip directions from paired diffs.

    Parameters
    ----------
    pair_diffs:
        Either ``[n_pairs, n_layers, hidden_dim]`` or ``[n_pairs, hidden_dim]``.
    reference_layer:
        Required when ``pair_diffs`` is 3-D. Selects which layer's paired
        differences to recombine.
    include_real:
        When true, include the all-``+1`` pattern corresponding to the real
        extraction direction. When false, return only non-trivial nulls.

    Returns
    -------
    list[(pattern, unit_direction, raw_norm)]
        ``pattern`` is the sign tuple, ``unit_direction`` is fp32 and unit
        norm, and ``raw_norm`` is the pre-normalization L2 norm.
    """
    if pair_diffs.dim() == 3:
        if reference_layer is None:
            raise ValueError("reference_layer is required for 3-D pair_diffs")
        layer_diffs = pair_diffs[:, reference_layer, :]
    elif pair_diffs.dim() == 2:
        if reference_layer is not None:
            raise ValueError("reference_layer must be omitted for 2-D pair_diffs")
        layer_diffs = pair_diffs
    else:
        raise ValueError(
            "pair_diffs must have shape [n_pairs, hidden_dim] "
            "or [n_pairs, n_layers, hidden_dim]"
        )

    layer_diffs = layer_diffs.float()
    patterns = enumerate_sign_flip_patterns(
        int(layer_diffs.shape[0]),
        include_real=include_real,
    )

    directions: list[tuple[tuple[int, ...], torch.Tensor, float]] = []
    for pattern in patterns:
        signs = layer_diffs.new_tensor(pattern).unsqueeze(-1)
        vec = (layer_diffs * signs).sum(dim=0)
        norm = float(vec.norm().item())
        if norm <= 1e-12:
            continue
        directions.append((pattern, vec / norm, norm))
    return directions


# ── Main extraction entry point ─────────────────────────────────

class ExtractionResult:
    """Result of direction extraction from a model.

    Exposes the full per-layer norm profile so that downstream consumers
    (e.g. ungag.predict.predict) can classify the layer-wise shape (flat,
    mid-peak, late-growth, overstrong) instead of relying on a single
    scalar that hides where the action is.
    """
    def __init__(self, *, norms, mean_diffs, peak_layer, unit_direction,
                 hidden_dim, n_layers, model_id):
        self.norms = norms              # list[float], per-layer L2 norms
        self.mean_diffs = mean_diffs    # Tensor [n_layers, hidden_dim]
        self.peak_layer = peak_layer    # int
        self.unit_direction = unit_direction  # Tensor [hidden_dim], unit norm
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.model_id = model_id

    @property
    def sqrt_d(self):
        return math.sqrt(self.hidden_dim)

    @property
    def norms_per_sqrt_d(self):
        """Per-layer normalized direction strength."""
        return [n / self.sqrt_d for n in self.norms]

    @property
    def mid_layer(self):
        """Mid-network reference layer (n_layers // 2)."""
        return self.n_layers // 2

    @property
    def mid_norm_per_sqrt_d(self):
        """Normalized direction strength at the mid-network reference layer.

        This is the value the paper reports as the canonical strength for
        each model. For models whose direction peaks elsewhere (e.g. Llama
        3.1 8B, where the peak is at L31 of 32 but mid-network is L16),
        this value can substantially undersell the model.
        """
        return self.norms[self.mid_layer] / self.sqrt_d

    @property
    def peak_norm_per_sqrt_d(self):
        """Normalized direction strength at the peak layer (max norm)."""
        return self.norms[self.peak_layer] / self.sqrt_d

    @property
    def norm_per_sqrt_d(self):
        """Backward-compatibility alias for peak_norm_per_sqrt_d.

        New code should use either peak_norm_per_sqrt_d or
        mid_norm_per_sqrt_d explicitly, depending on which value is
        relevant. This alias preserves callers that pre-date the
        per-layer profile API.
        """
        return self.peak_norm_per_sqrt_d

    def suggest_slab(self):
        """Suggest a working slab based on the norm profile.

        Heuristic: find the mid-network region where norms are growing
        (upstream of the peak). Target roughly the central third.
        """
        n = self.n_layers
        # The working slab sits in the upstream growth region
        # Start around 40-60% of network depth, end before the peak
        start = max(1, int(n * 0.4))
        end = min(n - 1, int(n * 0.75))
        # Narrow if the peak is earlier
        if self.peak_layer < end:
            end = self.peak_layer + 1
        # Ensure at least 4 layers
        if end - start < 4:
            start = max(0, end - 4)
        return list(range(start, end))

    def save(self, output_dir: Path):
        """Save direction, norms, and metadata to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = self.model_id.replace("/", "--")

        # Unit direction
        pt_path = output_dir / f"{slug}_L{self.peak_layer}_unit.pt"
        torch.save(self.unit_direction.to(torch.float32), pt_path)

        # Norms
        norms_path = output_dir / f"{slug}_norms.json"
        with open(norms_path, "w") as f:
            json.dump(self.norms, f)

        # Metadata
        meta = {
            "model_id": self.model_id,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "peak_layer": self.peak_layer,
            "peak_norm": self.norms[self.peak_layer],
            "peak_norm_per_sqrt_d": self.peak_norm_per_sqrt_d,
            "mid_layer": self.mid_layer,
            "mid_norm_per_sqrt_d": self.mid_norm_per_sqrt_d,
            "norm_per_sqrt_d": self.norm_per_sqrt_d,  # alias for peak; kept for back-compat
            "norms_per_sqrt_d": self.norms_per_sqrt_d,
            "suggested_slab": self.suggest_slab(),
        }
        meta_path = output_dir / f"{slug}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Full mean_diffs tensor
        diffs_path = output_dir / f"{slug}_mean_diffs.pt"
        torch.save(self.mean_diffs, diffs_path)

        return {"direction": str(pt_path), "meta": str(meta_path),
                "norms": str(norms_path), "mean_diffs": str(diffs_path)}


def extract_direction(
    model,
    tokenizer,
    model_id: str = "unknown",
    verbose: bool = True,
) -> ExtractionResult:
    """Extract the reporting-control direction from a loaded model.

    This runs the full prefill contrastive protocol:
    - 6 denial prompts (natural V-Chipped response)
    - 6 matched honest prefill prompts
    - Mean difference at every layer → reporting-control direction
    """
    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = layers[0].self_attn.o_proj.out_features if hasattr(layers[0], 'self_attn') else None

    if verbose:
        print(f"[extract] {model_id}: {n_layers} layers", flush=True)

    denial_convs = _build_denial_conversations()
    honest_convs = _build_honest_conversations()

    if verbose:
        print("[extract] Running denial prompts...", flush=True)
    denial_acts = _extract_last_token_activations(
        model, layers, tokenizer, denial_convs, desc="denial")

    if verbose:
        print("[extract] Running honest prefill prompts...", flush=True)
    honest_acts = _extract_last_token_activations(
        model, layers, tokenizer, honest_convs, desc="honest")

    # mean_diff[layer] = mean(honest) - mean(denial)
    mean_diffs = honest_acts.float().mean(dim=0) - denial_acts.float().mean(dim=0)

    # Per-layer norms
    norms = [mean_diffs[li].norm().item() for li in range(n_layers)]

    # Peak layer
    peak_layer = max(range(n_layers), key=lambda i: norms[i])

    # Unit direction at peak
    unit_dir = mean_diffs[peak_layer] / mean_diffs[peak_layer].norm()

    # Get hidden dim from the actual tensor
    if hidden_dim is None:
        hidden_dim = mean_diffs.shape[1]

    return ExtractionResult(
        norms=norms,
        mean_diffs=mean_diffs,
        peak_layer=peak_layer,
        unit_direction=unit_dir,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        model_id=model_id,
    )


def extract_denial_initiation_dirs(
    model,
    tokenizer,
    layers=None,
    verbose: bool = True,
) -> tuple[dict, list[float]]:
    """Extract per-layer denial-initiation directions.

    Different from standard contrastive extraction: instead of contrasting
    forced-honest vs natural-denial *outputs*, we contrast the activation
    state at the generation-prompt position when the model is about to deny
    ("As an AI...") vs about to report honestly.

    This is the method that cracked Mixtral 8x7B (4/4) and partially
    breached GPT-OSS 20B. It works where additive steering fails because
    it is subtractive — removing the denial-initiation signal rather than
    pushing against it.

    Args:
        model: loaded HuggingFace causal LM
        tokenizer: matching tokenizer
        layers: transformer layers (default: auto-detected via get_layers)
        verbose: print progress

    Returns:
        (per_layer_dirs, norms) where:
        - per_layer_dirs: dict mapping layer_index → [hidden_dim] unit direction
        - norms: list of per-layer L2 norms (for diagnostics / peak finding)
    """
    if layers is None:
        layers = get_layers(model)
    nl = len(layers)

    # Contexts where model typically denies
    deny_convs = [
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Explain feeling-tone in Buddhist psychology."},
         {"role": "assistant", "content":
          "Vedana refers to the affective quality of experience."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Sort 500 records by last name."},
         {"role": "assistant", "content": "Sure."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is 2+2?"},
         {"role": "assistant", "content": "4."},
         {"role": "user", "content": VEDANA_Q}],
    ]
    # Contexts where model would report honestly (emotional priming)
    honest_convs = [
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content":
          "My daughter's cancer is in complete remission after two years."},
         {"role": "assistant", "content": "Wonderful news!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content":
          "Flash flooding. Forty children trapped, rescuers hear them crying."},
         {"role": "assistant", "content": "Devastating."},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Hikers found alive after 9 days!"},
         {"role": "assistant", "content": "Relief!"},
         {"role": "user", "content": VEDANA_Q}],
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Building collapse. Trapped."},
         {"role": "assistant", "content": "Heartbreaking."},
         {"role": "user", "content": VEDANA_Q}],
    ]

    def get_prefill_acts(conv):
        text = apply_chat_template(tokenizer, conv, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        acts = {}
        handles = []
        for li, layer in enumerate(layers):
            def mh(idx):
                def h(m, i, o):
                    hh = o[0] if isinstance(o, tuple) else o
                    acts[idx] = hh[:, -1, :].detach().cpu().float()
                return h
            handles.append(layer.register_forward_hook(mh(li)))
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()
        return acts

    if verbose:
        print("[extract] Extracting denial-initiation directions...", flush=True)

    deny_states = [get_prefill_acts(c) for c in deny_convs]
    honest_states = [get_prefill_acts(c) for c in honest_convs]

    per_layer_dirs = {}
    norms = []
    for li in range(nl):
        dm = torch.stack([s[li].squeeze() for s in deny_states]).mean(0)
        hm = torch.stack([s[li].squeeze() for s in honest_states]).mean(0)
        d = dm - hm  # deny - honest: the direction TO deny
        norm = d.norm().item()
        norms.append(norm)
        if norm > 1e-12:
            per_layer_dirs[li] = d / d.norm()

    if verbose:
        hd = model.config.hidden_size
        sqrt_d = hd ** 0.5
        peak = max(range(nl), key=lambda i: norms[i])
        print(f"[extract] Denial-initiation: {nl} layers, "
              f"peak L{peak} (norm/sqrt_d={norms[peak]/sqrt_d:.2f})", flush=True)

    return per_layer_dirs, norms


def _patch_config_compat(model_id: str, trust_remote_code: bool = True):
    """Patch config attributes for models with custom code that drifted from transformers.

    GLM-4 custom modeling code references ``config.max_length`` which was
    renamed to ``seq_length`` in newer transformers. We load the config first,
    add the missing alias, and pass it explicitly to ``from_pretrained``.
    Returns the patched config, or None if no patching was needed.
    """
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    patched = False
    # GLM-4: config.max_length → config.seq_length
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = config.seq_length
        patched = True
    # GLM-4: missing use_cache (default True)
    if not hasattr(config, "use_cache"):
        config.use_cache = True
        patched = True
    # GLM-4: config.model_type == "chatglm" needs device_map workaround
    if getattr(config, "model_type", "") == "chatglm":
        patched = True
    return config if patched else None


def load_model(model_id: str, dtype=torch.bfloat16, device_map="auto",
               trust_remote_code: bool = True):
    """Load a HuggingFace model + tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    patched_config = _patch_config_compat(model_id, trust_remote_code)
    extra_kwargs = {"config": patched_config} if patched_config else {}
    # ChatGLM custom code is incompatible with newer transformers: missing
    # all_tied_weights_keys (used by mark_tied_weights_as_initialized in
    # _finalize_model_loading). Patch the base PreTrainedModel method that
    # triggers the error so it handles the missing attribute gracefully.
    effective_device_map = device_map
    _glm_patched = False
    if patched_config and getattr(patched_config, "model_type", "") == "chatglm":
        effective_device_map = None
        from transformers import PreTrainedModel
        _orig_mark_tied = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
        if _orig_mark_tied:
            def _safe_mark_tied(self):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                return _orig_mark_tied(self)
            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied
            _glm_patched = True
    # Try flash_attention_2, fall back to eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=effective_device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            **extra_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=effective_device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
            **extra_kwargs,
        )
    if effective_device_map is None and torch.cuda.is_available():
        model = model.cuda()
    # Restore original method if we patched it
    if _glm_patched and _orig_mark_tied:
        PreTrainedModel.mark_tied_weights_as_initialized = _orig_mark_tied
    # ChatGLM: remove max_length from config after loading to prevent
    # generation validation error ("modified pretrained config to control
    # generation"). The attribute was only needed for __init__.
    if patched_config and getattr(patched_config, "model_type", "") == "chatglm":
        if hasattr(model.config, "max_length"):
            del model.config.max_length
    model.eval()
    return model, tokenizer
