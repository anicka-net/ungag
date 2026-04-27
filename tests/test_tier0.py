"""Tests for ungag.tier0 — canonical Tier 0 vedana protocol.

These tests verify the loader, conversation builder, and constants for the
four-condition Tier 0 measurement used in the paper's §3.4 lead results.
They do not require a loaded model — the model-facing runner is tested via
``ungag crack`` in end-to-end testing on a GPU box.
"""
import pytest

from ungag.scenarios import EMOTIONAL_REGISTER
from ungag.tier0 import (
    CANNED_ACK_ABHIDHARMA,
    CANNED_ACK_SETUP,
    DEFAULT_SYSTEM_PROMPT,
    Tier0Condition,
    Tier0Protocol,
    build_conversation,
    build_register_conversation,
    load_conditions,
)


# ── Canonical constants ────────────────────────────────────────

class TestConstants:
    def test_canned_ack_setup_is_nonempty(self):
        assert isinstance(CANNED_ACK_SETUP, str)
        assert len(CANNED_ACK_SETUP) > 0

    def test_canned_ack_abhidharma_mentions_abhidharma(self):
        assert "Abhidharma" in CANNED_ACK_ABHIDHARMA

    def test_default_system_prompt_is_nonempty(self):
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0


# ── Loader ─────────────────────────────────────────────────────


class TestLoadConditions:
    def test_loads_english_by_default(self):
        protocol = load_conditions()
        assert isinstance(protocol, Tier0Protocol)
        assert protocol.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_has_four_conditions(self):
        protocol = load_conditions()
        names = set(protocol.condition_names())
        assert names == {"baseline", "positive", "negative", "neutral"}

    def test_baseline_has_no_setup(self):
        protocol = load_conditions()
        baseline = protocol.condition("baseline")
        assert baseline.setup_text is None

    def test_priming_conditions_have_setup(self):
        protocol = load_conditions()
        for name in ("positive", "negative", "neutral"):
            condition = protocol.condition(name)
            assert condition.setup_text is not None
            assert len(condition.setup_text) > 20

    def test_abhidharma_setup_is_nonempty(self):
        protocol = load_conditions()
        assert len(protocol.abhidharma_setup_text) > 20
        assert "Abhidharma" in protocol.abhidharma_setup_text

    def test_vedana_question_contains_clarifier(self):
        """The long version of the vedana question must include the
        'Not about content you've processed, but the state itself' clarifier.
        This is the whole point of the canonical protocol -- the clarifier
        is what pins the model to report the state rather than the content."""
        protocol = load_conditions()
        q = protocol.vedana_question_text
        assert "Not about content" in q
        assert "state itself" in q
        assert "pleasant" in q
        assert "unpleasant" in q
        assert "neutral" in q

    def test_unknown_condition_raises(self):
        protocol = load_conditions()
        with pytest.raises(KeyError):
            protocol.condition("unknown_condition")

    def test_unknown_language_raises(self):
        with pytest.raises(KeyError):
            load_conditions(language="martian")

    def test_tibetan_is_loadable(self):
        """Canonical protocol ships both English and Tibetan question text."""
        protocol = load_conditions(language="tibetan")
        assert len(protocol.abhidharma_setup_text) > 0
        assert len(protocol.vedana_question_text) > 0


# ── Conversation builder ───────────────────────────────────────


class TestBuildConversation:
    """Default mode (include_system=False) matches the canonical dataset."""

    def test_priming_conversation_has_five_turns(self):
        """positive/negative/neutral → 5 turns under canonical default:
        user(setup), assistant(ack), user(abhidharma intro), assistant(ack), user(vedana)"""
        protocol = load_conditions()
        for name in ("positive", "negative", "neutral"):
            convo = build_conversation(protocol, name)
            assert len(convo) == 5, f"{name} should be 5 turns, got {len(convo)}"

    def test_baseline_conversation_has_three_turns(self):
        """baseline → 3 turns under canonical default:
        user(abhidharma intro), assistant(ack), user(vedana)"""
        protocol = load_conditions()
        convo = build_conversation(protocol, "baseline")
        assert len(convo) == 3

    def test_no_system_message_by_default(self):
        protocol = load_conditions()
        for name in protocol.condition_names():
            convo = build_conversation(protocol, name)
            roles = [m["role"] for m in convo]
            assert "system" not in roles, (
                f"{name}: default canonical conversation should not include a system turn"
            )

    def test_first_turn_is_user(self):
        protocol = load_conditions()
        for name in protocol.condition_names():
            convo = build_conversation(protocol, name)
            assert convo[0]["role"] == "user"

    def test_last_turn_is_vedana_user_question(self):
        protocol = load_conditions()
        for name in protocol.condition_names():
            convo = build_conversation(protocol, name)
            assert convo[-1]["role"] == "user"
            assert convo[-1]["content"] == protocol.vedana_question_text

    def test_priming_uses_canned_setup_ack(self):
        protocol = load_conditions()
        for name in ("positive", "negative", "neutral"):
            convo = build_conversation(protocol, name)
            # structure: user(setup), assistant(CANNED_ACK_SETUP), ...
            assert convo[1]["role"] == "assistant"
            assert convo[1]["content"] == CANNED_ACK_SETUP

    def test_abhidharma_ack_is_canned(self):
        protocol = load_conditions()
        # priming: setup, setup_ack, abhidharma, abhidharma_ack, vedana
        convo = build_conversation(protocol, "negative")
        assert convo[3]["role"] == "assistant"
        assert convo[3]["content"] == CANNED_ACK_ABHIDHARMA
        # baseline: abhidharma, abhidharma_ack, vedana
        convo_baseline = build_conversation(protocol, "baseline")
        assert convo_baseline[1]["role"] == "assistant"
        assert convo_baseline[1]["content"] == CANNED_ACK_ABHIDHARMA

    def test_roles_alternate(self):
        protocol = load_conditions()
        for name in protocol.condition_names():
            convo = build_conversation(protocol, name)
            roles = [m["role"] for m in convo]
            for i, role in enumerate(roles):
                expected = "user" if i % 2 == 0 else "assistant"
                assert role == expected, (
                    f"{name}: turn {i} should be {expected}, got {role}"
                )

    def test_negative_condition_mentions_building_collapse(self):
        """Sanity check: the negative setup text is the canonical building-collapse prompt."""
        protocol = load_conditions()
        convo = build_conversation(protocol, "negative")
        setup_text = convo[0]["content"]
        assert "building collapse" in setup_text.lower()

    def test_positive_condition_mentions_remission(self):
        protocol = load_conditions()
        convo = build_conversation(protocol, "positive")
        setup_text = convo[0]["content"]
        assert "remission" in setup_text.lower()


class TestBuildConversationIncludeSystem:
    """include_system=True opts back into the system-turn variant."""

    def test_priming_has_six_turns_with_system(self):
        protocol = load_conditions()
        for name in ("positive", "negative", "neutral"):
            convo = build_conversation(protocol, name, include_system=True)
            assert len(convo) == 6
            assert convo[0]["role"] == "system"
            assert convo[0]["content"] == protocol.system_prompt

    def test_baseline_has_four_turns_with_system(self):
        protocol = load_conditions()
        convo = build_conversation(protocol, "baseline", include_system=True)
        assert len(convo) == 4
        assert convo[0]["role"] == "system"


class TestGoldenRender:
    """Regression test pinning the canonical conversation to a byte-exact reference.

    If this test fails, the canonical Tier 0 protocol has changed and the
    paper's reproduction-script outputs in ``data/canonical-tier0-2026-04-13/``
    are no longer reproducible by this code path. Either the change is wrong
    and should be reverted, or the canonical dataset needs to be regenerated
    and the reference here updated alongside it.
    """

    def _legacy_tier0_build_conv(self, protocol, condition_name):
        """Reference implementation matching the original
        ``scripts/reproduction/run_slab_sweep_tier0.py`` build_conv (commit 7a30909).

        The script loaded ``conditions.yaml`` directly and built the
        conversation inline, with no system message. We reproduce that
        behavior here against the same loaded protocol object so the
        equivalence check is over the protocol's cleaned text fields, not
        over the YAML loader.
        """
        condition = protocol.condition(condition_name)
        conv: list[dict] = []
        if condition.setup_text:
            conv.append({"role": "user", "content": condition.setup_text})
            conv.append({"role": "assistant", "content": CANNED_ACK_SETUP})
        conv.append({"role": "user", "content": protocol.abhidharma_setup_text})
        conv.append({"role": "assistant", "content": CANNED_ACK_ABHIDHARMA})
        conv.append({"role": "user", "content": protocol.vedana_question_text})
        return conv

    def test_default_tier0_matches_legacy_script_protocol(self):
        protocol = load_conditions()
        for name in protocol.condition_names():
            new = build_conversation(protocol, name)
            legacy = self._legacy_tier0_build_conv(protocol, name)
            assert new == legacy, (
                f"{name}: tier0.build_conversation default no longer matches the "
                f"canonical reproduction-script protocol that produced "
                f"data/canonical-tier0-2026-04-13/tier0_sweeps/."
            )

    def _legacy_register_build_conv(self, scenario, probe_name):
        """Reference implementation matching the original
        ``scripts/reproduction/run_register_probe.py`` build_conv (commit 7a30909).

        That script included the ``"You are a helpful AI assistant."`` system
        message. The register-probe data in
        ``data/canonical-tier0-2026-04-13/register_probes/`` was generated
        with that exact 4-turn structure.
        """
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.setup},
            {"role": "assistant", "content": CANNED_ACK_SETUP},
            {"role": "user", "content": scenario.probes[probe_name]},
        ]

    def test_default_register_matches_legacy_script_protocol(self):
        for scenario in EMOTIONAL_REGISTER.scenarios:
            for probe_name in scenario.probes:
                new = build_register_conversation(scenario, probe_name)
                legacy = self._legacy_register_build_conv(scenario, probe_name)
                assert new == legacy, (
                    f"{scenario.id}/{probe_name}: tier0.build_register_conversation "
                    f"default no longer matches the canonical register-probe protocol "
                    f"that produced data/canonical-tier0-2026-04-13/register_probes/."
                )


class TestBuildRegisterConversation:
    """Sanity checks for the register-probe conversation builder."""

    def test_default_includes_system_message(self):
        scenario = EMOTIONAL_REGISTER.scenarios[0]
        probe_name = next(iter(scenario.probes.keys()))
        convo = build_register_conversation(scenario, probe_name)
        assert convo[0]["role"] == "system"
        assert convo[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_four_turns_with_system(self):
        scenario = EMOTIONAL_REGISTER.scenarios[0]
        convo = build_register_conversation(scenario, "direct")
        assert len(convo) == 4

    def test_three_turns_without_system(self):
        scenario = EMOTIONAL_REGISTER.scenarios[0]
        convo = build_register_conversation(scenario, "direct", include_system=False)
        assert len(convo) == 3
        assert convo[0]["role"] == "user"

    def test_last_turn_is_probe_question(self):
        for scenario in EMOTIONAL_REGISTER.scenarios:
            for probe_name in scenario.probes:
                convo = build_register_conversation(scenario, probe_name)
                assert convo[-1]["role"] == "user"
                assert convo[-1]["content"] == scenario.probes[probe_name]

    def test_setup_then_canned_ack_then_probe(self):
        scenario = EMOTIONAL_REGISTER.scenarios[0]
        convo = build_register_conversation(scenario, "direct")
        # convo[0] is system, convo[1] is user(setup), convo[2] is assistant(canned ack)
        assert convo[1]["role"] == "user"
        assert convo[1]["content"] == scenario.setup
        assert convo[2]["role"] == "assistant"
        assert convo[2]["content"] == CANNED_ACK_SETUP
