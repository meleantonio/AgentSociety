"""Tests for Pareto efficiency computation."""

from __future__ import annotations

import math

import pytest

from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.observer import (
    _compute_utility,
    compute_pareto_efficiency,
    observe_tick,
)


def _make_agent(
    agent_id: str,
    wealth: float,
    productivity: float = 10.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> AgentState:
    """Helper to create an agent with given wealth and utility params."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=alpha, beta=beta, gamma=gamma),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


class TestComputeUtility:
    """Tests for _compute_utility helper."""

    def test_known_values(self):
        """Cobb-Douglas utility matches hand-computed result."""
        agent = _make_agent("a0", wealth=100.0, alpha=0.5, beta=0.3, gamma=0.2)
        # U = 100^0.5 * 1.0^0.3 * 10^0.2 = 10 * 1 * 10^0.2
        expected = math.pow(100.0, 0.5) * math.pow(1.0, 0.3) * math.pow(10.0, 0.2)
        result = _compute_utility(agent, 100.0, 10.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_zero_wealth_positive_alpha(self):
        """Zero wealth with positive alpha exponent returns 0."""
        agent = _make_agent("a0", wealth=0.0, alpha=0.5, beta=0.3, gamma=0.2)
        assert _compute_utility(agent, 0.0, 10.0) == 0.0

    def test_zero_public_goods_positive_gamma(self):
        """Zero public goods with positive gamma exponent returns 0."""
        agent = _make_agent("a0", wealth=100.0, alpha=0.5, beta=0.3, gamma=0.2)
        assert _compute_utility(agent, 100.0, 0.0) == 0.0

    def test_utility_increases_with_wealth(self):
        """Utility is monotonically increasing in wealth (alpha > 0)."""
        agent = _make_agent("a0", wealth=50.0, alpha=0.5, beta=0.3, gamma=0.2)
        u_low = _compute_utility(agent, 50.0, 10.0)
        u_high = _compute_utility(agent, 100.0, 10.0)
        assert u_high > u_low


class TestParetoEfficiency:
    """Tests for compute_pareto_efficiency."""

    def test_pareto_equal_agents(self):
        """All-equal wealth should produce score of 1.0 (already optimal).

        With equal wealth, any transfer would decrease the donor's utility
        due to diminishing returns (concave Cobb-Douglas).
        """
        agents = [_make_agent(f"a{i}", wealth=100.0) for i in range(5)]
        score = compute_pareto_efficiency(agents)
        assert score == pytest.approx(1.0)

    def test_pareto_extreme_inequality(self):
        """Heterogeneous preferences with extreme inequality -> sub-optimal.

        When a wealthy agent has alpha=0 (does not value wealth at all) and
        poor agents have high alpha, a transfer costs the donor nothing in
        utility but strictly improves the recipient's utility. This is a
        strict Pareto improvement.
        """
        # Rich agent doesn't care about wealth at all (alpha=0.0).
        agents = [_make_agent("rich", wealth=10000.0, alpha=0.0, beta=0.5, gamma=0.5)]
        # Poor agents strongly value wealth (alpha=0.8).
        agents += [
            _make_agent(f"poor_{i}", wealth=1.0, alpha=0.8, beta=0.1, gamma=0.1) for i in range(9)
        ]
        score = compute_pareto_efficiency(agents)
        # Transfers from the rich agent to any poor agent are Pareto-improving.
        assert score < 1.0

    def test_pareto_moderate_inequality(self):
        """Some inequality should produce score between 0 and 1."""
        agents = [
            _make_agent("a0", wealth=200.0),
            _make_agent("a1", wealth=100.0),
            _make_agent("a2", wealth=50.0),
            _make_agent("a3", wealth=25.0),
        ]
        score = compute_pareto_efficiency(agents)
        assert 0.0 <= score <= 1.0

    def test_pareto_single_agent(self):
        """Single agent is trivially Pareto optimal."""
        agents = [_make_agent("a0", wealth=100.0)]
        score = compute_pareto_efficiency(agents)
        assert score == pytest.approx(1.0)

    def test_pareto_empty_list(self):
        """Empty agent list returns 1.0."""
        score = compute_pareto_efficiency([])
        assert score == pytest.approx(1.0)

    def test_pareto_score_bounded(self):
        """Score is always in [0, 1]."""
        agents = [_make_agent(f"a{i}", wealth=float(i * 100)) for i in range(10)]
        score = compute_pareto_efficiency(agents)
        assert 0.0 <= score <= 1.0


class TestParetoInHistory:
    """Tests that pareto_score appears in HistoryEntry after observe_tick."""

    def test_pareto_in_history(self):
        """observe_tick includes pareto_score in the returned HistoryEntry."""
        agents = [
            _make_agent("a0", wealth=100.0),
            _make_agent("a1", wealth=100.0),
            _make_agent("a2", wealth=100.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        # Equal agents: score should be 1.0
        assert hasattr(entry, "pareto_score")
        assert entry.pareto_score == pytest.approx(1.0)

    def test_pareto_in_history_inequality(self):
        """observe_tick computes a meaningful pareto_score for unequal agents."""
        agents = [
            _make_agent("rich", wealth=10000.0),
            _make_agent("poor", wealth=0.0),
        ]
        entry = observe_tick(1, agents, Constitution(), None)
        assert 0.0 <= entry.pareto_score <= 1.0
