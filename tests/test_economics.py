"""Tests for the economic engine — production, taxation, redistribution."""

from __future__ import annotations

import pytest

from emergent_constitution.economics import (
    NumericalInstabilityError,
    apply_redistribution,
    apply_taxation,
    compute_production,
    economic_step,
    validate_agent_states,
)
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
)


def _make_agent(agent_id: str, wealth: float, productivity: float) -> AgentState:
    """Helper to create an agent with minimal boilerplate."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


# --- Production ---


class TestComputeProduction:
    def test_private_keeps_own(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.PRIVATE)
        prod = compute_production(sample_agents, c)
        assert prod["agent_0000"] == 10.0
        assert prod["agent_0001"] == 20.0
        assert prod["agent_0002"] == 5.0

    def test_communal_pools_equally(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.COMMUNAL)
        prod = compute_production(sample_agents, c)
        total = 10.0 + 20.0 + 5.0
        expected = total / 3
        for agent_id in prod:
            assert prod[agent_id] == pytest.approx(expected)

    def test_mixed_half_and_half(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.MIXED)
        prod = compute_production(sample_agents, c)
        total = 10.0 + 20.0 + 5.0
        pooled_share = (total * 0.5) / 3
        assert prod["agent_0000"] == pytest.approx(10.0 * 0.5 + pooled_share)
        assert prod["agent_0001"] == pytest.approx(20.0 * 0.5 + pooled_share)

    def test_empty_agents(self):
        c = Constitution()
        assert compute_production([], c) == {}

    def test_total_production_conserved_private(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.PRIVATE)
        prod = compute_production(sample_agents, c)
        total_prod = sum(a.productivity for a in sample_agents)
        assert sum(prod.values()) == pytest.approx(total_prod)

    def test_total_production_conserved_communal(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.COMMUNAL)
        prod = compute_production(sample_agents, c)
        total_prod = sum(a.productivity for a in sample_agents)
        assert sum(prod.values()) == pytest.approx(total_prod)

    def test_total_production_conserved_mixed(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.MIXED)
        prod = compute_production(sample_agents, c)
        total_prod = sum(a.productivity for a in sample_agents)
        assert sum(prod.values()) == pytest.approx(total_prod)


# --- Taxation ---


class TestApplyTaxation:
    def test_zero_tax(self, sample_agents):
        c = Constitution(tax_rate=0.0)
        prod = {a.id: a.productivity for a in sample_agents}
        new_agents, revenue = apply_taxation(sample_agents, prod, c)
        assert revenue == pytest.approx(0.0)
        for orig, new in zip(sample_agents, new_agents, strict=True):
            assert new.wealth == pytest.approx(orig.wealth + orig.productivity)

    def test_full_tax(self, sample_agents):
        c = Constitution(tax_rate=1.0)
        prod = {a.id: a.productivity for a in sample_agents}
        new_agents, revenue = apply_taxation(sample_agents, prod, c)
        total_prod = sum(a.productivity for a in sample_agents)
        assert revenue == pytest.approx(total_prod)
        for orig, new in zip(sample_agents, new_agents, strict=True):
            assert new.wealth == pytest.approx(orig.wealth)

    def test_partial_tax(self, sample_agents):
        c = Constitution(tax_rate=0.3)
        prod = {a.id: a.productivity for a in sample_agents}
        new_agents, revenue = apply_taxation(sample_agents, prod, c)
        expected_revenue = sum(a.productivity * 0.3 for a in sample_agents)
        assert revenue == pytest.approx(expected_revenue)

    def test_no_mutation_of_inputs(self, sample_agents):
        c = Constitution(tax_rate=0.2)
        original_wealths = [a.wealth for a in sample_agents]
        prod = {a.id: a.productivity for a in sample_agents}
        apply_taxation(sample_agents, prod, c)
        for orig_w, agent in zip(original_wealths, sample_agents, strict=True):
            assert agent.wealth == orig_w


# --- Redistribution ---


class TestApplyRedistribution:
    def test_none_loses_revenue(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.NONE)
        result = apply_redistribution(sample_agents, 100.0, c)
        for orig, new in zip(sample_agents, result, strict=True):
            assert new.wealth == orig.wealth

    def test_flat_equal_share(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.FLAT)
        result = apply_redistribution(sample_agents, 90.0, c)
        for orig, new in zip(sample_agents, result, strict=True):
            assert new.wealth == pytest.approx(orig.wealth + 30.0)

    def test_progressive_favors_poor(self):
        agents = [_make_agent("poor", 10.0, 5.0), _make_agent("rich", 1000.0, 5.0)]
        c = Constitution(redistribution_rule=RedistributionRule.PROGRESSIVE)
        result = apply_redistribution(agents, 100.0, c)
        poor_gain = result[0].wealth - agents[0].wealth
        rich_gain = result[1].wealth - agents[1].wealth
        assert poor_gain > rich_gain

    def test_progressive_conserves_revenue(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.PROGRESSIVE)
        revenue = 150.0
        result = apply_redistribution(sample_agents, revenue, c)
        total_gain = sum(n.wealth - o.wealth for o, n in zip(sample_agents, result, strict=True))
        assert total_gain == pytest.approx(revenue)

    def test_flat_conserves_revenue(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.FLAT)
        revenue = 150.0
        result = apply_redistribution(sample_agents, revenue, c)
        total_gain = sum(n.wealth - o.wealth for o, n in zip(sample_agents, result, strict=True))
        assert total_gain == pytest.approx(revenue)

    def test_zero_revenue_no_change(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.FLAT)
        result = apply_redistribution(sample_agents, 0.0, c)
        for orig, new in zip(sample_agents, result, strict=True):
            assert new.wealth == orig.wealth

    def test_no_mutation_of_inputs(self, sample_agents):
        c = Constitution(redistribution_rule=RedistributionRule.FLAT)
        original_wealths = [a.wealth for a in sample_agents]
        apply_redistribution(sample_agents, 100.0, c)
        for orig_w, agent in zip(original_wealths, sample_agents, strict=True):
            assert agent.wealth == orig_w


# --- Wealth conservation (full cycle) ---


class TestEconomicStep:
    def test_wealth_conservation_flat_redistribution(self, sample_agents):
        """Under flat redistribution, total wealth + production = total wealth after."""
        c = Constitution(
            property_rule=PropertyRule.PRIVATE,
            tax_rate=0.5,
            redistribution_rule=RedistributionRule.FLAT,
        )
        total_before = sum(a.wealth for a in sample_agents)
        total_prod = sum(a.productivity for a in sample_agents)
        result = economic_step(sample_agents, c)
        total_after = sum(a.wealth for a in result)
        assert total_after == pytest.approx(total_before + total_prod)

    def test_wealth_conservation_no_redistribution(self, sample_agents):
        """Under 'none' redistribution, tax revenue is lost."""
        c = Constitution(
            property_rule=PropertyRule.PRIVATE,
            tax_rate=0.5,
            redistribution_rule=RedistributionRule.NONE,
        )
        total_before = sum(a.wealth for a in sample_agents)
        total_prod = sum(a.productivity for a in sample_agents)
        result = economic_step(sample_agents, c)
        total_after = sum(a.wealth for a in result)
        expected = total_before + total_prod * 0.5  # half is kept, half is lost
        assert total_after == pytest.approx(expected)

    def test_zero_tax_all_production_kept(self, sample_agents):
        c = Constitution(property_rule=PropertyRule.PRIVATE, tax_rate=0.0)
        result = economic_step(sample_agents, c)
        for orig, new in zip(sample_agents, result, strict=True):
            assert new.wealth == pytest.approx(orig.wealth + orig.productivity)

    def test_no_mutation_of_inputs(self, sample_agents):
        c = Constitution(tax_rate=0.3)
        original_wealths = [a.wealth for a in sample_agents]
        economic_step(sample_agents, c)
        for orig_w, agent in zip(original_wealths, sample_agents, strict=True):
            assert agent.wealth == orig_w


# --- Validation ---


class TestValidateAgentStates:
    def test_valid_agents_pass(self, sample_agents):
        validate_agent_states(sample_agents)  # should not raise

    def test_nan_wealth_raises(self):
        # Pydantic ge=0.0 rejects NaN, so we construct without validation
        agent_data = {
            "id": "bad",
            "wealth": float("nan"),
            "productivity": 5.0,
            "utility_params": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "value_vector": {"equality": 0.5, "liberty": 0.5},
        }
        # Pydantic v2 rejects NaN for ge=0.0, so we construct without validation
        agent = AgentState.model_construct(**agent_data)
        with pytest.raises(NumericalInstabilityError):
            validate_agent_states([agent])

    def test_inf_wealth_raises(self):
        agent_data = {
            "id": "bad",
            "wealth": float("inf"),
            "productivity": 5.0,
            "utility_params": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "value_vector": {"equality": 0.5, "liberty": 0.5},
        }
        agent = AgentState.model_construct(**agent_data)
        with pytest.raises(NumericalInstabilityError):
            validate_agent_states([agent])
