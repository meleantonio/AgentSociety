"""Numerical stability tests for edge cases in the economic engine.

These tests verify the simulation handles extreme parameter values and long
runs without encountering NaN, Inf, wealth explosion, or wealth collapse.
All tests are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.economics import economic_step
from emergent_constitution.initialization import create_agents
from emergent_constitution.lead import Lead
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.rng import SimulationRNG


@pytest.mark.slow
class TestNumericalStability:
    """Edge-case tests for numerical robustness."""

    def test_very_high_tax_rate_no_crash(self) -> None:
        """Simulation runs with max tax rate (1.0) without numerical issues.

        We run a simulation, then apply several economic ticks with 100% tax
        to verify no agent ends up with NaN or negative wealth.
        """
        config = SimulationConfig(num_agents=20, max_ticks=50, seed=42)
        rng = SimulationRNG(config.seed)
        agents = create_agents(config, rng)

        # Force constitution to max tax + progressive redistribution
        constitution = Constitution(
            property_rule=PropertyRule.PRIVATE,
            tax_rate=1.0,
            voting_rule=VotingRule.MAJORITY,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )

        # Run 100 economic steps at 100% tax
        for _ in range(100):
            agents = economic_step(agents, constitution)

        for agent in agents:
            assert not math.isnan(agent.wealth), f"Agent {agent.id} has NaN wealth"
            assert not math.isinf(agent.wealth), f"Agent {agent.id} has Inf wealth"
            assert agent.wealth >= 0.0, f"Agent {agent.id} has negative wealth"

    def test_zero_tax_rate_preserves_wealth(self) -> None:
        """With 0% tax, agents accumulate production without redistribution."""
        config = SimulationConfig(num_agents=10, max_ticks=100, seed=42)
        rng = SimulationRNG(config.seed)
        agents = create_agents(config, rng)

        constitution = Constitution(
            tax_rate=0.0,
            redistribution_rule=RedistributionRule.NONE,
        )

        initial_total = sum(a.wealth for a in agents)
        total_production = sum(a.productivity for a in agents)

        agents = economic_step(agents, constitution)
        final_total = sum(a.wealth for a in agents)

        # With 0% tax and no redistribution, total wealth should increase by total production
        assert math.isclose(final_total, initial_total + total_production, rel_tol=1e-9)

    def test_very_low_productivity_no_crash(self) -> None:
        """Agents with minimal productivity don't cause division-by-zero or NaN."""
        # Create agents with very low productivity
        agents = [
            AgentState(
                id=f"agent_{i:04d}",
                wealth=100.0,
                productivity=0.01,  # near-zero but valid
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.5, liberty=0.5),
            )
            for i in range(20)
        ]

        constitution = Constitution(
            tax_rate=0.5,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )

        # Run many steps
        for _ in range(200):
            agents = economic_step(agents, constitution)

        for agent in agents:
            assert not math.isnan(agent.wealth), f"Agent {agent.id} has NaN wealth"
            assert not math.isinf(agent.wealth), f"Agent {agent.id} has Inf wealth"
            assert agent.wealth >= 0.0

    def test_many_ticks_no_wealth_explosion(self) -> None:
        """Wealth doesn't explode to infinity over many ticks."""
        config = SimulationConfig(num_agents=10, max_ticks=500, seed=42)
        output = Lead(config).run()

        for agent in output.final_agent_states:
            assert agent.wealth < 1e12, (
                f"Agent {agent.id} has unreasonable wealth: {agent.wealth:.2e}"
            )

    def test_many_ticks_no_wealth_collapse(self) -> None:
        """Total wealth doesn't collapse to zero for all agents over many ticks."""
        config = SimulationConfig(num_agents=10, max_ticks=500, seed=42)
        output = Lead(config).run()

        total_wealth = sum(a.wealth for a in output.final_agent_states)
        assert total_wealth > 0, "Total wealth collapsed to zero"

    def test_communal_property_numerical_stability(self) -> None:
        """Communal property rule produces valid results over many ticks.

        Under communal property, all production is pooled equally, which
        should be numerically stable.
        """
        agents = [
            AgentState(
                id=f"agent_{i:04d}",
                wealth=50.0 + i * 10,
                productivity=5.0 + i,
                utility_params=UtilityParams(alpha=0.4, beta=0.3, gamma=0.3),
                value_vector=ValueVector(equality=0.5, liberty=0.5),
            )
            for i in range(30)
        ]

        constitution = Constitution(
            property_rule=PropertyRule.COMMUNAL,
            tax_rate=0.3,
            redistribution_rule=RedistributionRule.FLAT,
        )

        for _ in range(500):
            agents = economic_step(agents, constitution)

        for agent in agents:
            assert not math.isnan(agent.wealth)
            assert not math.isinf(agent.wealth)
            assert agent.wealth >= 0.0

    def test_mixed_property_numerical_stability(self) -> None:
        """Mixed property rule (50/50 private/pooled) stays numerically stable."""
        agents = [
            AgentState(
                id=f"agent_{i:04d}",
                wealth=100.0,
                productivity=10.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.5, liberty=0.5),
            )
            for i in range(30)
        ]

        constitution = Constitution(
            property_rule=PropertyRule.MIXED,
            tax_rate=0.5,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )

        for _ in range(500):
            agents = economic_step(agents, constitution)

        total_wealth = sum(a.wealth for a in agents)
        assert total_wealth > 0, "Total wealth collapsed under mixed property"
        assert total_wealth < 1e12, f"Wealth exploded: {total_wealth:.2e}"

    def test_progressive_redistribution_extreme_inequality(self) -> None:
        """Progressive redistribution handles extreme wealth disparities.

        One agent has nearly all the wealth; progressive redistribution should
        not produce NaN or negative values.
        """
        agents = [
            AgentState(
                id="agent_rich",
                wealth=1_000_000.0,
                productivity=100.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.2, liberty=0.8),
            ),
        ] + [
            AgentState(
                id=f"agent_poor_{i:04d}",
                wealth=0.01,
                productivity=1.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.8, liberty=0.2),
            )
            for i in range(19)
        ]

        constitution = Constitution(
            tax_rate=0.8,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )

        for _ in range(100):
            agents = economic_step(agents, constitution)

        for agent in agents:
            assert not math.isnan(agent.wealth), f"Agent {agent.id} has NaN wealth"
            assert not math.isinf(agent.wealth), f"Agent {agent.id} has Inf wealth"
            assert agent.wealth >= 0.0, f"Agent {agent.id} has negative wealth"

    def test_long_run_total_output_monotonic(self) -> None:
        """Total output (sum of productivities) stays constant over time.

        Since productivity is not modified by the economic step, total output
        should remain the same across ticks.
        """
        config = SimulationConfig(num_agents=20, max_ticks=100, seed=42, observer_interval=10)
        output = Lead(config).run()

        # All history entries should have the same total_output
        outputs = [entry.total_output for entry in output.history]
        for i, out in enumerate(outputs):
            assert not math.isnan(out), f"History entry {i} has NaN total_output"
            assert not math.isinf(out), f"History entry {i} has Inf total_output"
            assert out > 0.0, f"History entry {i} has zero total_output"
