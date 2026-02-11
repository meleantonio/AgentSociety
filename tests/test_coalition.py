"""Tests for coalition formation logic."""

from __future__ import annotations

import pytest

from emergent_constitution.coalition import CoalitionInfo, compute_coalition_stats, form_coalitions
from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.rng import SimulationRNG


def _make_agent(
    agent_id: str,
    equality: float,
    wealth: float = 100.0,
    productivity: float = 10.0,
) -> AgentState:
    """Helper to create an agent with a specific equality value."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=equality, liberty=round(1.0 - equality, 6)),
    )


class TestFormCoalitions:
    """Tests for form_coalitions function."""

    def test_form_coalitions_assigns_ids(self):
        """All agents receive a non-None coalition_id after formation."""
        agents = [_make_agent(f"a{i}", equality=0.1 * i) for i in range(10)]
        rng = SimulationRNG(seed=42)
        result = form_coalitions(agents, rng)
        assert len(result) == len(agents)
        for agent in result:
            assert agent.coalition_id is not None
            assert agent.coalition_id.startswith("coalition_")

    def test_similar_agents_same_coalition(self):
        """Agents with very similar equality values cluster together."""
        agents = [
            _make_agent("a0", equality=0.50),
            _make_agent("a1", equality=0.52),
            _make_agent("a2", equality=0.48),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions(agents, rng, similarity_threshold=0.3)
        coalition_ids = {a.coalition_id for a in result}
        assert len(coalition_ids) == 1, "Very similar agents should all be in one coalition"

    def test_dissimilar_agents_different_coalitions(self):
        """Agents with very different equality values end up in different coalitions."""
        agents = [
            _make_agent("a0", equality=0.0),
            _make_agent("a1", equality=1.0),
        ]
        rng = SimulationRNG(seed=42)
        # Use a very small threshold so they cannot cluster
        result = form_coalitions(agents, rng, similarity_threshold=0.1)
        coalition_ids = {a.coalition_id for a in result}
        assert len(coalition_ids) == 2, "Dissimilar agents should be in different coalitions"

    def test_deterministic_with_seed(self):
        """Same seed produces identical coalition assignments."""
        agents = [_make_agent(f"a{i}", equality=0.1 * i) for i in range(10)]

        rng1 = SimulationRNG(seed=99)
        result1 = form_coalitions(agents, rng1)

        rng2 = SimulationRNG(seed=99)
        result2 = form_coalitions(agents, rng2)

        ids1 = [a.coalition_id for a in result1]
        ids2 = [a.coalition_id for a in result2]
        assert ids1 == ids2

    def test_single_agent(self):
        """Single agent gets its own coalition."""
        agents = [_make_agent("a0", equality=0.5)]
        rng = SimulationRNG(seed=42)
        result = form_coalitions(agents, rng)
        assert len(result) == 1
        assert result[0].coalition_id == "coalition_0000"

    def test_empty_agents(self):
        """Empty agent list returns empty list."""
        rng = SimulationRNG(seed=42)
        result = form_coalitions([], rng)
        assert result == []

    def test_original_agents_not_mutated(self):
        """Original agent list is not modified."""
        agents = [_make_agent("a0", equality=0.5)]
        rng = SimulationRNG(seed=42)
        assert agents[0].coalition_id is None
        form_coalitions(agents, rng)
        assert agents[0].coalition_id is None

    def test_coalition_ids_sequential(self):
        """Coalition IDs follow sequential numbering format."""
        agents = [
            _make_agent("a0", equality=0.0),
            _make_agent("a1", equality=1.0),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions(agents, rng, similarity_threshold=0.1)
        coalition_ids = sorted({a.coalition_id for a in result})
        assert coalition_ids[0] == "coalition_0000"
        assert coalition_ids[1] == "coalition_0001"


class TestComputeCoalitionStats:
    """Tests for compute_coalition_stats function."""

    def test_coalition_stats(self):
        """Correct size, mean wealth, and mean equality for known coalitions."""
        agents = [
            _make_agent("a0", equality=0.6, wealth=100.0).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
            _make_agent("a1", equality=0.8, wealth=200.0).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
            _make_agent("a2", equality=0.2, wealth=50.0).model_copy(
                update={"coalition_id": "coalition_0001"}
            ),
        ]
        stats = compute_coalition_stats(agents)
        assert len(stats) == 2

        c0 = stats["coalition_0000"]
        assert c0.size == 2
        assert c0.mean_wealth == pytest.approx(150.0)
        assert c0.mean_equality == pytest.approx(0.7)

        c1 = stats["coalition_0001"]
        assert c1.size == 1
        assert c1.mean_wealth == pytest.approx(50.0)
        assert c1.mean_equality == pytest.approx(0.2)

    def test_stats_excludes_unaffiliated(self):
        """Agents with coalition_id=None are excluded from stats."""
        agents = [
            _make_agent("a0", equality=0.5).model_copy(update={"coalition_id": "coalition_0000"}),
            _make_agent("a1", equality=0.5),  # No coalition_id
        ]
        stats = compute_coalition_stats(agents)
        assert len(stats) == 1
        assert stats["coalition_0000"].size == 1

    def test_empty_agents(self):
        """Empty agent list returns empty dict."""
        assert compute_coalition_stats([]) == {}

    def test_coalition_info_model(self):
        """CoalitionInfo fields are correctly typed."""
        info = CoalitionInfo(
            coalition_id="coalition_0000",
            size=3,
            mean_wealth=100.0,
            mean_equality=0.5,
        )
        assert info.coalition_id == "coalition_0000"
        assert info.size == 3
        assert info.mean_wealth == 100.0
        assert info.mean_equality == 0.5


class TestCoalitionInLead:
    """Integration tests for coalition formation in the Lead tick loop."""

    def test_coalition_in_lead(self):
        """Lead runs and agents have coalition_ids after coalition_interval ticks."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=10,
            seed=42,
            coalition_interval=10,
        )
        lead = Lead(config)
        output = lead.run()
        # After tick 10 (which is a coalition_interval tick), agents should have coalitions
        has_coalition = any(a.coalition_id is not None for a in output.final_agent_states)
        assert has_coalition, "Some agents should have coalition_id after coalition_interval tick"

    def test_coalition_in_history(self):
        """num_coalitions appears in HistoryEntry when coalitions are formed."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=10,
            seed=42,
            coalition_interval=5,
            observer_interval=10,
        )
        lead = Lead(config)
        output = lead.run()
        # Observer runs at tick 10, coalitions form at tick 5 and 10
        # At tick 10 agents should have coalitions
        assert len(output.history) > 0
        last_entry = output.history[-1]
        assert last_entry.num_coalitions > 0

    def test_coalition_deterministic(self):
        """Same seed produces identical coalition assignments in full simulation."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=20,
            seed=42,
            coalition_interval=10,
        )
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        ids1 = [a.coalition_id for a in output1.final_agent_states]
        ids2 = [a.coalition_id for a in output2.final_agent_states]
        assert ids1 == ids2

    def test_no_coalition_before_interval(self):
        """Agents have no coalition_id before the first coalition_interval tick."""
        config = SimulationConfig(
            num_agents=5,
            max_ticks=5,
            seed=42,
            coalition_interval=10,
        )
        lead = Lead(config)
        output = lead.run()
        # max_ticks=5 < coalition_interval=10, so no coalitions should form
        for agent in output.final_agent_states:
            assert agent.coalition_id is None
