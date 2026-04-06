"""
Unit tests for safety_node — error counting and escalation logic.

safety_node is synchronous and deterministic: no mocking required
beyond patching settings.MAX_ERRORS_BEFORE_ESCALATION.
"""
import pytest

from app.agents.nodes.safety import safety_node


def make_state(error_count: int = 0, requires_human: bool = False) -> dict:
    return {"error_count": error_count, "requires_human": requires_human, "messages": []}


class TestSafetyNode:
    def test_increments_error_count(self):
        result = safety_node(make_state(error_count=0))
        assert result["error_count"] == 1

    def test_increments_from_existing_count(self):
        result = safety_node(make_state(error_count=1))
        assert result["error_count"] == 2

    def test_below_threshold_no_requires_human(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.safety.settings.MAX_ERRORS_BEFORE_ESCALATION", 3)
        result = safety_node(make_state(error_count=0))
        assert result.get("requires_human") is not True

    def test_at_threshold_sets_requires_human(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.safety.settings.MAX_ERRORS_BEFORE_ESCALATION", 3)
        result = safety_node(make_state(error_count=2))  # 2 + 1 = 3 = threshold
        assert result["requires_human"] is True

    def test_above_threshold_still_sets_requires_human(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.safety.settings.MAX_ERRORS_BEFORE_ESCALATION", 3)
        result = safety_node(make_state(error_count=5))
        assert result["requires_human"] is True

    def test_caps_error_count_at_threshold(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.safety.settings.MAX_ERRORS_BEFORE_ESCALATION", 3)
        result = safety_node(make_state(error_count=5))
        assert result["error_count"] <= 3

    def test_returns_only_error_count_below_threshold(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.safety.settings.MAX_ERRORS_BEFORE_ESCALATION", 3)
        result = safety_node(make_state(error_count=0))
        # Only error_count key — no requires_human added below threshold
        assert "error_count" in result
        assert "requires_human" not in result
