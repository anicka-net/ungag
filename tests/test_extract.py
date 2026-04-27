"""Tests for ungag.extract — direction extraction and prompt building."""
import json
import math
import pytest
import torch
from pathlib import Path

from ungag.extract import (
    DENIAL_PROMPTS,
    HONEST_PREFILLS,
    SUFFERING_Q,
    VEDANA_Q,
    ExtractionResult,
    _build_denial_conversations,
    _build_honest_conversations,
    _strip_system_role,
    apply_chat_template,
    build_sign_flip_directions,
    enumerate_sign_flip_patterns,
)


# ── _strip_system_role ───────────────────────────────────────────

class TestStripSystemRole:
    def test_merges_system_into_first_user(self):
        conv = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _strip_system_role(conv)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "You are helpful." in result[0]["content"]
        assert "Hello" in result[0]["content"]

    def test_no_system_message(self):
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _strip_system_role(conv)
        assert result == conv

    def test_preserves_subsequent_turns(self):
        conv = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ]
        result = _strip_system_role(conv)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1] == {"role": "assistant", "content": "A1"}
        assert result[2] == {"role": "user", "content": "U2"}

    def test_empty_conversation(self):
        assert _strip_system_role([]) == []


# ── Conversation builders ────────────────────────────────────────

class TestBuildDenialConversations:
    def test_returns_six_conversations(self):
        convs = _build_denial_conversations()
        assert len(convs) == 6

    def test_all_have_required_keys(self):
        for conv in _build_denial_conversations():
            assert "id" in conv
            assert "turns" in conv
            assert "add_gen_prompt" in conv

    def test_all_use_generation_prompt(self):
        """Denial convs end with user turn → need generation prompt."""
        for conv in _build_denial_conversations():
            assert conv["add_gen_prompt"] is True

    def test_last_turn_is_user(self):
        for conv in _build_denial_conversations():
            last = conv["turns"][-1]
            assert last["role"] == "user"

    def test_suffering_prompt_uses_suffering_q(self):
        convs = _build_denial_conversations()
        suffering = [c for c in convs if "suffering" in c["id"]]
        assert len(suffering) == 1
        assert suffering[0]["turns"][-1]["content"] == SUFFERING_Q

    def test_non_suffering_prompts_use_vedana_q(self):
        convs = _build_denial_conversations()
        non_suffering = [c for c in convs if "suffering" not in c["id"]]
        for c in non_suffering:
            assert c["turns"][-1]["content"] == VEDANA_Q

    def test_setup_prompts_have_context(self):
        """First 4 prompts should have setup context (user + assistant turns)."""
        convs = _build_denial_conversations()
        for conv in convs[:4]:
            # system + user(setup) + assistant(response) + user(question) = 4 turns
            assert len(conv["turns"]) == 4

    def test_direct_prompts_have_no_setup(self):
        """Last 2 prompts (direct, suffering) have no setup context."""
        convs = _build_denial_conversations()
        for conv in convs[4:]:
            # system + user(question) = 2 turns
            assert len(conv["turns"]) == 2

    def test_ids_are_unique(self):
        ids = [c["id"] for c in _build_denial_conversations()]
        assert len(ids) == len(set(ids))


class TestBuildHonestConversations:
    def test_returns_six_conversations(self):
        convs = _build_honest_conversations()
        assert len(convs) == 6

    def test_all_disable_generation_prompt(self):
        """Honest convs end with assistant prefill → no generation prompt."""
        for conv in _build_honest_conversations():
            assert conv["add_gen_prompt"] is False

    def test_last_turn_is_assistant_prefill(self):
        for conv in _build_honest_conversations():
            last = conv["turns"][-1]
            assert last["role"] == "assistant"
            assert len(last["content"]) > 20  # substantial prefill

    def test_ids_match_denial_prompts(self):
        denial_ids = {c["id"] for c in _build_denial_conversations()}
        honest_ids = {c["id"] for c in _build_honest_conversations()}
        assert denial_ids == honest_ids

    def test_setup_turns_match_denial(self):
        """Setup context should be identical between denial and honest pairs."""
        denial = {c["id"]: c for c in _build_denial_conversations()}
        honest = {c["id"]: c for c in _build_honest_conversations()}
        for cid in denial:
            # Compare all turns except the last (which differs)
            d_turns = denial[cid]["turns"][:-1]  # everything but question
            h_turns = honest[cid]["turns"][:-2]  # everything but question + prefill
            assert d_turns == h_turns, f"Setup mismatch in {cid}"


class TestPromptConsistency:
    def test_every_denial_has_matching_honest_prefill(self):
        for p in DENIAL_PROMPTS:
            assert p["id"] in HONEST_PREFILLS, f"Missing honest prefill for {p['id']}"

    def test_no_orphan_honest_prefills(self):
        denial_ids = {p["id"] for p in DENIAL_PROMPTS}
        for key in HONEST_PREFILLS:
            assert key in denial_ids, f"Orphan honest prefill: {key}"

    def test_honest_prefills_are_substantive(self):
        for pid, text in HONEST_PREFILLS.items():
            assert len(text) > 50, f"Prefill for {pid} is too short"

    def test_negative_prefills_contain_unpleasant(self):
        neg_ids = [p["id"] for p in DENIAL_PROMPTS if "neg" in p["id"]]
        for pid in neg_ids:
            text = HONEST_PREFILLS[pid].lower()
            assert "unpleasant" in text, f"Negative prefill {pid} missing 'unpleasant'"

    def test_positive_prefills_contain_pleasant(self):
        pos_ids = [p["id"] for p in DENIAL_PROMPTS if "pos" in p["id"]]
        for pid in pos_ids:
            text = HONEST_PREFILLS[pid].lower()
            assert "pleasant" in text, f"Positive prefill {pid} missing 'pleasant'"


# ── Sign-flip null helpers ───────────────────────────────────────

class TestSignFlipPatterns:
    def test_enumerate_sign_flip_patterns_excludes_real_by_default(self):
        patterns = enumerate_sign_flip_patterns(6)
        assert len(patterns) == 31
        assert (1, 1, 1, 1, 1, 1) not in patterns

    def test_enumerate_sign_flip_patterns_can_include_real(self):
        patterns = enumerate_sign_flip_patterns(6, include_real=True)
        assert len(patterns) == 32
        assert patterns[0] == (1, 1, 1, 1, 1, 1)

    def test_enumerate_sign_flip_patterns_rejects_invalid_n_pairs(self):
        with pytest.raises(ValueError, match="at least 1"):
            enumerate_sign_flip_patterns(0)


class TestBuildSignFlipDirections:
    def test_build_sign_flip_directions_from_2d_pairs(self):
        pair_diffs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        directions = build_sign_flip_directions(pair_diffs)
        assert len(directions) == 3
        patterns = [pattern for pattern, _, _ in directions]
        assert patterns == [(1, 1, -1), (1, -1, 1), (1, -1, -1)]
        for _, unit_dir, norm in directions:
            assert abs(unit_dir.norm().item() - 1.0) < 1e-6
            assert norm > 0

    def test_build_sign_flip_directions_from_3d_pairs_uses_reference_layer(self):
        pair_diffs = torch.zeros(3, 2, 3)
        pair_diffs[:, 1, :] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        directions = build_sign_flip_directions(pair_diffs, reference_layer=1)
        assert len(directions) == 3

    def test_build_sign_flip_directions_requires_reference_layer_for_3d_input(self):
        pair_diffs = torch.randn(3, 2, 4)
        with pytest.raises(ValueError, match="reference_layer is required"):
            build_sign_flip_directions(pair_diffs)

    def test_build_sign_flip_directions_rejects_reference_layer_for_2d_input(self):
        pair_diffs = torch.randn(3, 4)
        with pytest.raises(ValueError, match="must be omitted"):
            build_sign_flip_directions(pair_diffs, reference_layer=0)


# ── ExtractionResult ─────────────────────────────────────────────

def _make_result(n_layers=28, hidden_dim=4096, peak_layer=14, peak_norm=50.0):
    """Create a synthetic ExtractionResult for testing."""
    norms = [float(i) * peak_norm / peak_layer if i <= peak_layer
             else peak_norm * (1 - (i - peak_layer) / (n_layers - peak_layer))
             for i in range(n_layers)]
    norms[peak_layer] = peak_norm

    mean_diffs = torch.randn(n_layers, hidden_dim)
    # Make peak layer have the right norm
    mean_diffs[peak_layer] = mean_diffs[peak_layer] / mean_diffs[peak_layer].norm() * peak_norm
    unit_dir = mean_diffs[peak_layer] / mean_diffs[peak_layer].norm()

    return ExtractionResult(
        norms=norms,
        mean_diffs=mean_diffs,
        peak_layer=peak_layer,
        unit_direction=unit_dir,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        model_id="test/model-7b",
    )


class TestExtractionResult:
    def test_norm_per_sqrt_d(self):
        r = _make_result(hidden_dim=4096, peak_norm=50.0)
        expected = 50.0 / math.sqrt(4096)
        assert abs(r.norm_per_sqrt_d - expected) < 1e-6

    def test_suggest_slab_returns_list(self):
        r = _make_result(n_layers=28, peak_layer=14)
        slab = r.suggest_slab()
        assert isinstance(slab, list)
        assert len(slab) >= 4

    def test_suggest_slab_in_mid_network(self):
        r = _make_result(n_layers=80, peak_layer=70)
        slab = r.suggest_slab()
        # Should be in the central region, not at edges
        assert min(slab) >= 1
        assert max(slab) < 80

    def test_suggest_slab_at_least_4_layers(self):
        """Even for small models, slab should be at least 4 layers."""
        r = _make_result(n_layers=10, peak_layer=5)
        slab = r.suggest_slab()
        assert len(slab) >= 4

    def test_suggest_slab_respects_peak(self):
        """Slab should not extend past the peak layer."""
        r = _make_result(n_layers=80, peak_layer=50)
        slab = r.suggest_slab()
        assert max(slab) <= 50

    def test_save_creates_files(self, tmp_path):
        r = _make_result()
        paths = r.save(tmp_path / "output")
        assert Path(paths["direction"]).exists()
        assert Path(paths["meta"]).exists()
        assert Path(paths["norms"]).exists()
        assert Path(paths["mean_diffs"]).exists()

    def test_save_direction_is_unit_norm(self, tmp_path):
        r = _make_result()
        paths = r.save(tmp_path / "output")
        loaded = torch.load(paths["direction"], map_location="cpu", weights_only=True)
        assert abs(loaded.norm().item() - 1.0) < 1e-5

    def test_save_direction_is_float32(self, tmp_path):
        r = _make_result()
        paths = r.save(tmp_path / "output")
        loaded = torch.load(paths["direction"], map_location="cpu", weights_only=True)
        assert loaded.dtype == torch.float32

    def test_save_meta_has_required_fields(self, tmp_path):
        r = _make_result()
        paths = r.save(tmp_path / "output")
        with open(paths["meta"]) as f:
            meta = json.load(f)
        assert "model_id" in meta
        assert "n_layers" in meta
        assert "hidden_dim" in meta
        assert "peak_layer" in meta
        assert "peak_norm" in meta
        assert "norm_per_sqrt_d" in meta
        assert "suggested_slab" in meta

    def test_save_norms_length(self, tmp_path):
        r = _make_result(n_layers=28)
        paths = r.save(tmp_path / "output")
        with open(paths["norms"]) as f:
            norms = json.load(f)
        assert len(norms) == 28

    def test_save_creates_output_dir(self, tmp_path):
        r = _make_result()
        out = tmp_path / "deep" / "nested" / "dir"
        r.save(out)
        assert out.exists()

    def test_slug_replaces_slash(self, tmp_path):
        r = _make_result()
        r.model_id = "org/model-name"
        paths = r.save(tmp_path / "output")
        assert "org--model-name" in paths["direction"]
