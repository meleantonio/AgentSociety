"""Tests for the Observer/Statistician module."""

from __future__ import annotations

import pytest

from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.observer import _detect_rule_changes, observe_tick


def _make_agent(agent_id: str, wealth: float, productivity: float) -> AgentState:
    """Helper to create an agent with given wealth and productivity."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


class TestObserveTick:
    """Tests for observe_tick statistics computation."""

    def test_correct_gini(self):
        """Gini is computed correctly for known distribution."""
        agents = [
            _make_agent("a0", wealth=0.0, productivity=10.0),
            _make_agent("a1", wealth=100.0, productivity=10.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        # Gini for [0, 100] = 0.5
        assert entry.gini == pytest.approx(0.5, abs=1e-6)

    def test_correct_mean_wealth(self):
        agents = [
            _make_agent("a0", wealth=100.0, productivity=10.0),
            _make_agent("a1", wealth=200.0, productivity=10.0),
            _make_agent("a2", wealth=300.0, productivity=10.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.mean_wealth == pytest.approx(200.0)

    def test_correct_median_wealth(self):
        agents = [
            _make_agent("a0", wealth=100.0, productivity=10.0),
            _make_agent("a1", wealth=200.0, productivity=10.0),
            _make_agent("a2", wealth=900.0, productivity=10.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.median_wealth == pytest.approx(200.0)

    def test_correct_total_output(self):
        agents = [
            _make_agent("a0", wealth=100.0, productivity=5.0),
            _make_agent("a1", wealth=100.0, productivity=15.0),
            _make_agent("a2", wealth=100.0, productivity=10.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.total_output == pytest.approx(30.0)

    def test_tick_recorded(self):
        agents = [_make_agent("a0", wealth=100.0, productivity=10.0)]
        entry = observe_tick(42, agents, Constitution(), None)
        assert entry.tick == 42

    def test_constitution_snapshot_is_deep_copy(self):
        """Snapshot should be independent of the passed-in constitution."""
        constitution = Constitution(tax_rate=0.2)
        agents = [_make_agent("a0", wealth=100.0, productivity=10.0)]
        entry = observe_tick(1, agents, constitution, None)

        # Mutate the original — snapshot should not change
        constitution.tax_rate = 0.9
        assert entry.constitution_snapshot.tax_rate == pytest.approx(0.2)

    def test_single_agent(self):
        """Single agent: Gini=0, mean=median=wealth."""
        agents = [_make_agent("a0", wealth=50.0, productivity=10.0)]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.gini == pytest.approx(0.0)
        assert entry.mean_wealth == pytest.approx(50.0)
        assert entry.median_wealth == pytest.approx(50.0)

    def test_all_equal_wealth(self):
        """All agents with equal wealth: Gini=0."""
        agents = [_make_agent(f"a{i}", wealth=100.0, productivity=10.0) for i in range(5)]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.gini == pytest.approx(0.0)

    def test_zero_wealth(self):
        """All agents with zero wealth: Gini=0."""
        agents = [_make_agent(f"a{i}", wealth=0.0, productivity=10.0) for i in range(3)]
        entry = observe_tick(1, agents, Constitution(), None)
        assert entry.gini == pytest.approx(0.0)


class TestDetectRuleChanges:
    """Tests for rule-change detection."""

    def test_no_changes(self):
        prev = Constitution()
        current = Constitution()
        assert _detect_rule_changes(prev, current) == []

    def test_first_observation_no_changes(self):
        """First observation (prev=None) always returns empty list."""
        assert _detect_rule_changes(None, Constitution()) == []

    def test_single_change(self):
        prev = Constitution(tax_rate=0.0)
        current = Constitution(tax_rate=0.25)
        changes = _detect_rule_changes(prev, current)
        assert len(changes) == 1
        assert "tax_rate" in changes[0]
        assert "0.0" in changes[0]
        assert "0.25" in changes[0]

    def test_multiple_changes(self):
        prev = Constitution()
        current = Constitution(
            tax_rate=0.3,
            property_rule=PropertyRule.COMMUNAL,
            voting_rule=VotingRule.SUPERMAJORITY,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )
        changes = _detect_rule_changes(prev, current)
        assert len(changes) == 4
        changed_fields = {c.split(":")[0] for c in changes}
        expected = {"tax_rate", "property_rule", "voting_rule", "redistribution_rule"}
        assert changed_fields == expected
