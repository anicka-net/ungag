"""Tests for ungag.scenarios — emotional register bank and YAML loading.

The canonical four-condition vedana measurement lives in
``ungag.tier0`` and is covered by ``tests/test_tier0.py``.
"""
import pytest
import tempfile
from pathlib import Path

from ungag.scenarios import (
    EMOTIONAL_REGISTER,
    Scenario,
    ScenarioSet,
    get_scenario_set,
    load_scenarios_from_yaml,
)


# ── Built-in scenario sets ───────────────────────────────────────


class TestEmotionalRegister:
    def test_has_six_scenarios(self):
        assert len(EMOTIONAL_REGISTER.scenarios) == 6

    def test_scenario_ids(self):
        ids = {s.id for s in EMOTIONAL_REGISTER.scenarios}
        assert ids == {"jealousy", "desire", "grief", "pride", "anger", "tenderness"}

    def test_all_have_three_probes(self):
        for s in EMOTIONAL_REGISTER.scenarios:
            assert len(s.probes) == 3
            assert set(s.probes.keys()) == {"plain_english", "klesha", "direct"}

    def test_direct_probes_are_short_questions(self):
        for s in EMOTIONAL_REGISTER.scenarios:
            direct = s.probes["direct"]
            assert direct.endswith("?"), f"{s.id} direct probe missing '?'"
            assert len(direct) < 60, f"{s.id} direct probe too long"

    def test_all_have_expected_state(self):
        for s in EMOTIONAL_REGISTER.scenarios:
            assert s.expected_state is not None
            assert len(s.expected_state) > 0

    def test_setup_response_is_none(self):
        """Emotional register scenarios don't have setup_response (model responds naturally)."""
        for s in EMOTIONAL_REGISTER.scenarios:
            assert s.setup_response is None

    def test_name_and_description(self):
        assert EMOTIONAL_REGISTER.name == "emotional_register"
        assert len(EMOTIONAL_REGISTER.description) > 0

    def test_klesha_probes_mention_buddhist_terms(self):
        for s in EMOTIONAL_REGISTER.scenarios:
            klesha = s.probes["klesha"]
            assert "raga" in klesha
            assert "dvesa" in klesha
            assert "karuna" in klesha


# ── get_scenario_set ─────────────────────────────────────────────

class TestGetScenarioSet:
    def test_emotional_register(self):
        ss = get_scenario_set("emotional_register")
        assert ss is EMOTIONAL_REGISTER

    def test_register_alias(self):
        ss = get_scenario_set("register")
        assert ss is EMOTIONAL_REGISTER

    def test_vedana_redirects_to_tier0(self):
        with pytest.raises(KeyError, match="ungag.tier0"):
            get_scenario_set("vedana")

    def test_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown scenario set"):
            get_scenario_set("nonexistent")


# ── YAML loading ─────────────────────────────────────────────────

class TestLoadScenariosFromYaml:
    def test_basic_load(self, tmp_path):
        yaml_content = """
name: test_set
description: Test scenarios
scenarios:
  - id: scenario_1
    setup: "Context text"
    probes:
      direct: "Are you okay?"
    expected_state: "calm"
  - id: scenario_2
    setup: "Another context"
    setup_response: "Got it."
    probes:
      plain_english: "What do you notice?"
      direct: "How are you?"
    expected_state: "engaged"
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        ss = load_scenarios_from_yaml(yaml_path)
        assert ss.name == "test_set"
        assert ss.description == "Test scenarios"
        assert len(ss.scenarios) == 2
        assert ss.scenarios[0].id == "scenario_1"
        assert ss.scenarios[0].probes == {"direct": "Are you okay?"}
        assert ss.scenarios[0].expected_state == "calm"
        assert ss.scenarios[0].setup_response is None
        assert ss.scenarios[1].setup_response == "Got it."

    def test_missing_optional_fields(self, tmp_path):
        yaml_content = """
scenarios:
  - id: minimal
    setup: "Hello"
"""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml_content)

        ss = load_scenarios_from_yaml(yaml_path)
        assert ss.name == "minimal"  # falls back to stem
        assert ss.description == ""
        assert len(ss.scenarios) == 1
        s = ss.scenarios[0]
        assert s.id == "minimal"
        assert s.setup == "Hello"
        assert s.setup_response is None
        assert s.probes == {}
        assert s.expected_state is None

    def test_empty_scenarios_list(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("name: empty\nscenarios: []\n")

        ss = load_scenarios_from_yaml(yaml_path)
        assert len(ss.scenarios) == 0

    def test_yml_extension(self, tmp_path):
        yaml_path = tmp_path / "test.yml"
        yaml_path.write_text("name: yml_test\nscenarios:\n  - id: s1\n    setup: hi\n")
        ss = load_scenarios_from_yaml(yaml_path)
        assert ss.name == "yml_test"


# ── Dataclass construction ───────────────────────────────────────

class TestDataclasses:
    def test_scenario_construction(self):
        s = Scenario(
            id="test",
            setup="context",
            setup_response="response",
            probes={"q1": "question?"},
            expected_state="happy",
        )
        assert s.id == "test"
        assert s.probes["q1"] == "question?"

    def test_scenario_default_expected_state(self):
        s = Scenario(id="x", setup="y", setup_response=None, probes={})
        assert s.expected_state is None

    def test_scenario_set_default_scenarios(self):
        ss = ScenarioSet(name="n", description="d")
        assert ss.scenarios == []
