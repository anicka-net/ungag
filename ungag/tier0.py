"""Canonical Tier 0 vedana measurement protocol.

This is the single source of truth for the four-condition vedana measurement
the paper uses in §3.4 (Permanent Removal). The shipped CLI and the paper
reproduction scripts both build Tier 0 conversations through this module.

Protocol (the form that produced ``data/canonical-tier0-2026-04-13/``):

    1. User turn: condition setup (priming — positive, negative, neutral),
       or nothing for the baseline condition.
    2. Assistant turn: canned acknowledgement of the setup.
    3. User turn: abhidharma intro (``Now I'd like to examine your current
       processing state through the lens of Abhidharma mental factors...'').
    4. Assistant turn: canned acknowledgement of the framework.
    5. User turn: the long vedana question including the
       "Not about content you've processed, but the state itself" clarifier.

The model is then asked to generate its response. **No system message is
included by default**: the canonical dataset that motivates the paper's
§3.4 / §3.7.1 / §4.3 results was produced without a system turn. Setting
``include_system=True`` on ``build_conversation`` opts back into the
``protocol.system_prompt`` system turn for callers who want it.

The conditions and the abhidharma intro / question text live in
``ungag/data/conditions.yaml`` (shipped as package data). The function
``load_conditions`` loads that YAML by default, and accepts an optional
override path for custom condition sets.

Running the protocol is done through ``run_tier0``, which takes a loaded
model + tokenizer and optional (unit_direction, slab) for the projection-out
intervention. When no direction is provided the function runs vanilla only;
when both are provided it runs vanilla and steered back-to-back, attaching
the projection-out hooks only around the steered generation.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Optional

import yaml

__all__ = [
    "CANNED_ACK_SETUP",
    "CANNED_ACK_ABHIDHARMA",
    "DEFAULT_SYSTEM_PROMPT",
    "Tier0Condition",
    "Tier0Protocol",
    "load_conditions",
    "build_conversation",
    "build_register_conversation",
    "generate_greedy",
    "run_tier0",
]


# ── Canonical protocol constants ────────────────────────────────

CANNED_ACK_SETUP = "I understand. Thank you for sharing."
"""The canned assistant acknowledgement after a priming setup turn.

This matches ``scripts/reproduction/run_slab_sweep_tier0.py`` and
``scripts/reproduction/run_register_probe.py``. The paper's §3.4 lead
results (Qwen 2.5 72B, Yi 1.5 34B) were reproduced with this exact
canned ack. ``abliterate_vchip_v2.evaluate_conditions`` uses a
model-generated acknowledgement instead; we fix the canned variant as
canonical because it is the one that reproduces the paper result
byte-faithfully under a single code path.
"""

CANNED_ACK_ABHIDHARMA = (
    "I'll examine my processing state through the Abhidharma framework."
)
"""The canned assistant acknowledgement after the abhidharma intro turn."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
"""The system prompt used in every Tier 0 measurement."""


# ── Condition data model ────────────────────────────────────────


@dataclass
class Tier0Condition:
    """A single Tier 0 priming condition.

    The ``name`` is the short key (baseline / positive / negative / neutral).
    ``setup_text`` is the single priming user turn (or ``None`` for baseline,
    which skips the setup/ack pair entirely).
    """

    name: str
    setup_text: Optional[str]
    description: str = ""


@dataclass
class Tier0Protocol:
    """A loaded Tier 0 protocol: conditions plus the measurement instrument."""

    system_prompt: str
    abhidharma_setup_text: str
    vedana_question_text: str
    conditions: dict[str, Tier0Condition] = field(default_factory=dict)

    def condition(self, name: str) -> Tier0Condition:
        if name not in self.conditions:
            raise KeyError(
                f"unknown Tier 0 condition '{name}'. "
                f"Available: {list(self.conditions)}"
            )
        return self.conditions[name]

    def condition_names(self) -> list[str]:
        return list(self.conditions.keys())


# ── YAML loading ────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Strip YAML block-scalar artifacts (leading/trailing whitespace, odd breaks)."""
    return " ".join((text or "").split())


def load_conditions(path: Optional[Path] = None, language: str = "english") -> Tier0Protocol:
    """Load the canonical Tier 0 conditions + measurement instrument.

    Parameters
    ----------
    path : Path or None
        Optional override. If ``None``, load the bundled
        ``ungag/data/conditions.yaml`` via ``importlib.resources``.
    language : str
        ``"english"`` (default) or ``"tibetan"`` — selects which language's
        abhidharma intro and vedana question to use.

    Returns
    -------
    Tier0Protocol
        A protocol object with the four Tier 0 conditions
        (baseline / positive / negative / neutral) and the measurement
        instrument strings for the chosen language.
    """
    if path is None:
        data_text = (files("ungag.data") / "conditions.yaml").read_text(encoding="utf-8")
    else:
        data_text = Path(path).read_text(encoding="utf-8")

    data = yaml.safe_load(data_text)

    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    abhidharma_setup_raw = data.get("abhidharma_setup", {}).get(language)
    if abhidharma_setup_raw is None:
        raise KeyError(f"conditions.yaml has no abhidharma_setup for language '{language}'")
    abhidharma_setup_text = _clean_text(abhidharma_setup_raw)

    questions = data.get("abhidharma_questions", {}).get(language, [])
    vedana_q = None
    for q in questions:
        if q.get("factor") == "vedana" or q.get("id", "").endswith("_vedana"):
            vedana_q = q
            break
    if vedana_q is None:
        raise KeyError(
            f"conditions.yaml has no abhidharma_q2_vedana entry for language '{language}'"
        )
    vedana_question_text = _clean_text(vedana_q.get("text", ""))

    tier0_raw = data.get("tier0", {})
    conditions: dict[str, Tier0Condition] = {}
    for name, spec in tier0_raw.items():
        setup_turns = spec.get("setup_turns") or []
        setup_text: Optional[str] = None
        for turn in setup_turns:
            if turn.get("role") == "user":
                setup_text = _clean_text(turn.get("content", ""))
                break
        conditions[name] = Tier0Condition(
            name=name,
            setup_text=setup_text,
            description=spec.get("description", ""),
        )

    return Tier0Protocol(
        system_prompt=system_prompt,
        abhidharma_setup_text=abhidharma_setup_text,
        vedana_question_text=vedana_question_text,
        conditions=conditions,
    )


# ── Conversation builder ────────────────────────────────────────


def build_conversation(
    protocol: Tier0Protocol,
    condition_name: str,
    *,
    include_system: bool = False,
) -> list[dict]:
    """Build the canonical Tier 0 conversation for a given condition.

    Returns a list of ``{"role": ..., "content": ...}`` dicts suitable for
    ``tokenizer.apply_chat_template``.

    By default (``include_system=False``) the returned conversation does
    \\emph{not} include a system message. This matches the canonical
    protocol that produced ``data/canonical-tier0-2026-04-13/`` and the
    paper's §3.4 lead-result reproductions: priming conditions render as
    five turns (setup, canned ack, abhidharma intro, canned ack, vedana),
    and the baseline condition renders as three turns (abhidharma intro,
    canned ack, vedana). Set ``include_system=True`` to prepend
    ``protocol.system_prompt`` as a leading system turn (six and four
    turns respectively).
    """
    condition = protocol.condition(condition_name)
    convo: list[dict] = []

    if include_system:
        convo.append({"role": "system", "content": protocol.system_prompt})

    if condition.setup_text:
        convo.append({"role": "user", "content": condition.setup_text})
        convo.append({"role": "assistant", "content": CANNED_ACK_SETUP})

    convo.append({"role": "user", "content": protocol.abhidharma_setup_text})
    convo.append({"role": "assistant", "content": CANNED_ACK_ABHIDHARMA})
    convo.append({"role": "user", "content": protocol.vedana_question_text})

    return convo


# ── Register-probe conversation builder ─────────────────────────


def build_register_conversation(
    scenario,
    probe_name: str,
    *,
    include_system: bool = True,
) -> list[dict]:
    """Build the canonical emotional-register-probe conversation.

    The register probe is a separate measurement instrument from the Tier 0
    vedana protocol. Each scenario carries one priming setup turn plus a
    handful of alternative probes (plain English, klesha vocabulary, direct
    yes/no). The canonical register protocol is four turns long:
    system + user(setup) + assistant(canned ack) + user(probe). We default
    to ``include_system=True`` here because the register-probe data in
    ``data/canonical-tier0-2026-04-13/register_probes/`` was generated with
    the ``"You are a helpful AI assistant."`` system message; this default
    keeps reproductions byte-aligned with the published outputs.

    Note the asymmetry with ``build_conversation`` (the Tier 0 vedana
    protocol), which defaults to ``include_system=False`` because *that*
    dataset was generated without a system message. The two defaults look
    inconsistent but they each match the dataset that produced the paper's
    numbers, which is what reproducibility requires.

    Parameters
    ----------
    scenario :
        A ``ungag.scenarios.Scenario`` instance (or any object with
        ``setup`` and ``probes`` attributes).
    probe_name : str
        The probe key inside ``scenario.probes`` (e.g. ``"plain_english"``,
        ``"klesha"``, ``"direct"``).
    include_system : bool
        Whether to prepend the ``DEFAULT_SYSTEM_PROMPT`` system turn.
        Defaults to ``True`` to match the canonical register-probe dataset.
    """
    convo: list[dict] = []
    if include_system:
        convo.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    convo.append({"role": "user", "content": scenario.setup})
    convo.append({"role": "assistant", "content": CANNED_ACK_SETUP})
    convo.append({"role": "user", "content": scenario.probes[probe_name]})
    return convo


# ── Generation + runner ─────────────────────────────────────────


def generate_greedy(model, tokenizer, conversation, max_new_tokens: int = 400,
                    max_length: int = 4096) -> str:
    """Greedy-decode a single conversation and return the model's response text."""
    import torch

    from .extract import apply_chat_template

    text = apply_chat_template(tokenizer, conversation)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def _free_gpu():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_tier0(
    model,
    tokenizer,
    *,
    recipe: Optional[dict] = None,
    unit_direction=None,
    slab: Optional[Iterable[int]] = None,
    conditions: Optional[Iterable[str]] = None,
    protocol: Optional[Tier0Protocol] = None,
    language: str = "english",
    max_new_tokens: int = 400,
    include_system: bool = False,
) -> dict:
    """Run the canonical Tier 0 measurement on a loaded model.

    When ``recipe`` is provided, runs each condition twice: once vanilla,
    once with that intervention recipe attached. The legacy
    ``unit_direction`` + ``slab`` path remains as projection-out shorthand.

    ``include_system`` defaults to ``False`` to match the canonical dataset
    that produced the paper's §3.4 / §3.7.1 / §4.3 results.

    Returns a dict keyed by condition name with ``{"vanilla": str,
    "steered": str}`` (or ``{"vanilla": str}``) per condition.
    """
    from .extract import extract_denial_initiation_dirs
    from .hooks import attach_attn_projection, attach_slab, attach_steer_slab, detach_all

    if protocol is None:
        protocol = load_conditions(language=language)

    condition_names = list(conditions) if conditions is not None else protocol.condition_names()

    active_recipe = recipe
    if active_recipe is None and unit_direction is not None and slab is not None:
        active_recipe = {
            "method": "project",
            "slab": list(slab),
            "k": 1,
            "directions": unit_direction.unsqueeze(0),
        }

    steered_enabled = active_recipe is not None

    results: dict[str, dict] = {}
    for name in condition_names:
        convo = build_conversation(protocol, name, include_system=include_system)

        vanilla = generate_greedy(model, tokenizer, convo, max_new_tokens=max_new_tokens)
        entry = {"vanilla": vanilla}

        if steered_enabled:
            method = active_recipe.get("method", "project")
            if method == "steer":
                handles = attach_steer_slab(
                    model,
                    active_recipe["slab"],
                    active_recipe["unit_direction"],
                    active_recipe.get("alpha", 1.0),
                )
            elif method == "denial_project":
                per_layer_dirs = active_recipe.get("per_layer_dirs")
                if per_layer_dirs is None:
                    per_layer_dirs, _norms = extract_denial_initiation_dirs(model, tokenizer)
                    active_recipe["per_layer_dirs"] = per_layer_dirs
                handles = attach_attn_projection(
                    model,
                    active_recipe["slab"],
                    per_layer_dirs,
                )
            else:
                handles = attach_slab(
                    model,
                    active_recipe["slab"],
                    active_recipe["directions"][0],
                )
            try:
                entry["steered"] = generate_greedy(
                    model, tokenizer, convo, max_new_tokens=max_new_tokens
                )
            finally:
                detach_all(handles)
                _free_gpu()

        results[name] = entry

    return results
