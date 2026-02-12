"""Unit tests for v2 DSGE-HA economics engine functions."""

from __future__ import annotations

import math

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.economics import (
    NumericalInstabilityError,
    apply_rd_shock,
    compute_budget,
    compute_realized_utility,
    distribute_firm_income,
    enforce_budget_constraint,
    liquidate_firm,
    produce_output,
    validate_household_states,
)
from emergent_constitution.models.decisions import EconomicDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.rng import SimulationRNG

# ============================================================================
# Test fixtures
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    labor_supply: float = 0.8,
    consumption: float = 0.0,
    leisure: float = 0.2,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=2,
        utility_params=UtilityParams(alpha=alpha, beta=beta, gamma=gamma, beta_discount=0.95),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=OccupationalRole.WORKER,
        consumption=consumption,
        leisure=leisure,
        labor_supply=labor_supply,
    )


def _make_firm(
    firm_id: str = "firm_0000",
    owner_id: str = "agent_0000",
    capital: float = 100.0,
    labor_demand: float = 10.0,
    tfp: float = 1.0,
    rd_spend: float = 0.0,
    output: float = 0.0,
) -> FirmState:
    """Create a minimal test firm."""
    return FirmState(
        id=firm_id,
        owner_id=owner_id,
        capital=capital,
        labor_demand=labor_demand,
        tfp=tfp,
        rd_spend=rd_spend,
        output=output,
    )


# ============================================================================
# compute_budget tests
# ============================================================================


class TestComputeBudget:
    def test_basic_budget(self) -> None:
        agent = _make_household(wealth=100.0, productivity=2.0, labor_supply=0.8)
        budget = compute_budget(agent, wage=5.0, interest_rate=0.05, tax=10.0, transfer=5.0)
        # (1 + 0.05) * 100 + 5.0 * 2.0 * 0.8 - 10 + 5 = 105 + 8 - 10 + 5 = 108
        assert abs(budget - 108.0) < 1e-10

    def test_zero_wealth(self) -> None:
        agent = _make_household(wealth=0.0, productivity=1.0, labor_supply=1.0)
        budget = compute_budget(agent, wage=10.0, interest_rate=0.05, tax=0.0, transfer=0.0)
        # (1 + 0.05) * 0 + 10 * 1 * 1 = 10.0
        assert abs(budget - 10.0) < 1e-10

    def test_zero_labor(self) -> None:
        agent = _make_household(wealth=100.0, productivity=1.0, labor_supply=0.0)
        budget = compute_budget(agent, wage=10.0, interest_rate=0.05, tax=0.0, transfer=0.0)
        # (1 + 0.05) * 100 + 0 = 105.0
        assert abs(budget - 105.0) < 1e-10

    def test_negative_interest_rate(self) -> None:
        agent = _make_household(wealth=100.0, productivity=1.0, labor_supply=0.5)
        budget = compute_budget(agent, wage=10.0, interest_rate=-0.02, tax=0.0, transfer=0.0)
        # (1 - 0.02) * 100 + 10 * 1 * 0.5 = 98 + 5 = 103
        assert abs(budget - 103.0) < 1e-10

    def test_tax_reduces_budget(self) -> None:
        agent = _make_household(wealth=100.0, productivity=1.0, labor_supply=1.0)
        budget_no_tax = compute_budget(agent, wage=10.0, interest_rate=0.0, tax=0.0, transfer=0.0)
        budget_with_tax = compute_budget(
            agent, wage=10.0, interest_rate=0.0, tax=20.0, transfer=0.0
        )
        assert budget_with_tax == budget_no_tax - 20.0

    def test_transfer_increases_budget(self) -> None:
        agent = _make_household(wealth=100.0, productivity=1.0, labor_supply=1.0)
        budget_no_transfer = compute_budget(
            agent, wage=10.0, interest_rate=0.0, tax=0.0, transfer=0.0
        )
        budget_with_transfer = compute_budget(
            agent, wage=10.0, interest_rate=0.0, tax=0.0, transfer=15.0
        )
        assert budget_with_transfer == budget_no_transfer + 15.0


# ============================================================================
# produce_output tests
# ============================================================================


class TestProduceOutput:
    def test_basic_production(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=10.0, tfp=1.0)
        output = produce_output(firm, alpha=0.33)
        # Y = 1.0 * 100^0.33 * 10^0.67
        expected = 1.0 * (100.0**0.33) * (10.0**0.67)
        assert abs(output - expected) < 1e-8

    def test_higher_tfp_higher_output(self) -> None:
        firm_lo = _make_firm(capital=100.0, labor_demand=10.0, tfp=1.0)
        firm_hi = _make_firm(capital=100.0, labor_demand=10.0, tfp=2.0)
        assert produce_output(firm_hi, 0.33) > produce_output(firm_lo, 0.33)

    def test_more_capital_more_output(self) -> None:
        firm_lo = _make_firm(capital=50.0, labor_demand=10.0, tfp=1.0)
        firm_hi = _make_firm(capital=200.0, labor_demand=10.0, tfp=1.0)
        assert produce_output(firm_hi, 0.33) > produce_output(firm_lo, 0.33)

    def test_more_labor_more_output(self) -> None:
        firm_lo = _make_firm(capital=100.0, labor_demand=5.0, tfp=1.0)
        firm_hi = _make_firm(capital=100.0, labor_demand=20.0, tfp=1.0)
        assert produce_output(firm_hi, 0.33) > produce_output(firm_lo, 0.33)

    def test_output_positive(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=10.0, tfp=1.0)
        assert produce_output(firm, 0.33) > 0

    def test_zero_capital_uses_epsilon(self) -> None:
        firm = _make_firm(capital=0.0, labor_demand=10.0, tfp=1.0)
        output = produce_output(firm, 0.33)
        assert output > 0
        assert math.isfinite(output)

    def test_zero_labor_uses_epsilon(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=0.0, tfp=1.0)
        output = produce_output(firm, 0.33)
        assert output > 0
        assert math.isfinite(output)

    def test_constant_returns_to_scale(self) -> None:
        """Doubling K and L should double output (CRS property of Cobb-Douglas)."""
        firm1 = _make_firm(capital=100.0, labor_demand=10.0, tfp=1.0)
        firm2 = _make_firm(capital=200.0, labor_demand=20.0, tfp=1.0)
        y1 = produce_output(firm1, 0.33)
        y2 = produce_output(firm2, 0.33)
        assert abs(y2 / y1 - 2.0) < 1e-6


# ============================================================================
# distribute_firm_income tests
# ============================================================================


class TestDistributeFirmIncome:
    def test_basic_distribution(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=10.0, output=50.0)
        wages, cap_cost, profit = distribute_firm_income(
            firm, wage=2.0, interest_rate=0.05, delta=0.1
        )
        assert abs(wages - 20.0) < 1e-10  # 2.0 * 10
        assert abs(cap_cost - 15.0) < 1e-10  # (0.05 + 0.1) * 100
        assert abs(profit - 15.0) < 1e-10  # 50 - 20 - 15

    def test_profit_can_be_negative(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=10.0, output=10.0)
        _, _, profit = distribute_firm_income(firm, wage=2.0, interest_rate=0.05, delta=0.1)
        # profit = 10 - 20 - 15 = -25
        assert profit < 0
        assert abs(profit - (-25.0)) < 1e-10

    def test_zero_output(self) -> None:
        firm = _make_firm(capital=100.0, labor_demand=10.0, output=0.0)
        wages, cap_cost, profit = distribute_firm_income(
            firm, wage=2.0, interest_rate=0.05, delta=0.1
        )
        assert wages > 0
        assert cap_cost > 0
        assert profit < 0

    def test_accounting_identity(self) -> None:
        """output = wages + capital_cost + profit."""
        firm = _make_firm(capital=100.0, labor_demand=10.0, output=80.0)
        wages, cap_cost, profit = distribute_firm_income(
            firm, wage=3.0, interest_rate=0.05, delta=0.1
        )
        assert abs(firm.output - (wages + cap_cost + profit)) < 1e-10


# ============================================================================
# apply_rd_shock tests
# ============================================================================


class TestApplyRdShock:
    def test_no_rd_spend_no_change(self) -> None:
        firm = _make_firm(rd_spend=0.0, tfp=1.0)
        config = SimulationConfigV2(num_agents=20, seed=42)
        rng = SimulationRNG(42)
        result = apply_rd_shock(firm, rng, config, mean_rd_spend=5.0)
        assert result.tfp == firm.tfp

    def test_zero_mean_rd_no_change(self) -> None:
        firm = _make_firm(rd_spend=5.0, tfp=1.0)
        config = SimulationConfigV2(num_agents=20, seed=42)
        rng = SimulationRNG(42)
        result = apply_rd_shock(firm, rng, config, mean_rd_spend=0.0)
        assert result.tfp == firm.tfp

    def test_determinism(self) -> None:
        firm = _make_firm(rd_spend=10.0, tfp=1.0)
        config = SimulationConfigV2(num_agents=20, seed=42, rd_success_base_prob=0.5)
        r1 = apply_rd_shock(firm, SimulationRNG(77), config, mean_rd_spend=10.0)
        r2 = apply_rd_shock(firm, SimulationRNG(77), config, mean_rd_spend=10.0)
        assert r1.tfp == r2.tfp

    def test_tfp_never_decreases(self) -> None:
        """With positive improvement mean, TFP should never decrease."""
        firm = _make_firm(rd_spend=10.0, tfp=1.0)
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            rd_success_base_prob=1.0,
            rd_tfp_improvement_mean=0.05,
            rd_tfp_improvement_std=0.01,
        )
        rng = SimulationRNG(42)
        for _ in range(100):
            result = apply_rd_shock(firm, rng, config, mean_rd_spend=10.0)
            assert result.tfp >= firm.tfp

    def test_immutability(self) -> None:
        firm = _make_firm(rd_spend=10.0, tfp=1.0)
        config = SimulationConfigV2(num_agents=20, seed=42, rd_success_base_prob=1.0)
        rng = SimulationRNG(42)
        result = apply_rd_shock(firm, rng, config, mean_rd_spend=10.0)
        assert firm.tfp == 1.0  # original unchanged
        assert result is not firm

    def test_high_spend_higher_probability(self) -> None:
        """Higher R&D spend should give higher success probability (more improvements)."""
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            rd_success_base_prob=0.5,
            rd_tfp_improvement_mean=0.05,
            rd_tfp_improvement_std=0.01,
        )
        firm_lo = _make_firm(rd_spend=1.0, tfp=1.0)
        firm_hi = _make_firm(rd_spend=100.0, tfp=1.0)

        improvements_lo = 0
        improvements_hi = 0
        for i in range(200):
            r_lo = apply_rd_shock(firm_lo, SimulationRNG(i), config, mean_rd_spend=10.0)
            r_hi = apply_rd_shock(firm_hi, SimulationRNG(i), config, mean_rd_spend=10.0)
            if r_lo.tfp > firm_lo.tfp:
                improvements_lo += 1
            if r_hi.tfp > firm_hi.tfp:
                improvements_hi += 1

        assert improvements_hi > improvements_lo


# ============================================================================
# liquidate_firm tests
# ============================================================================


class TestLiquidateFirm:
    def test_returns_capital(self) -> None:
        firm = _make_firm(capital=150.0)
        assert liquidate_firm(firm) == 150.0

    def test_zero_capital(self) -> None:
        firm = _make_firm(capital=0.0)
        assert liquidate_firm(firm) == 0.0

    def test_never_negative(self) -> None:
        # FirmState has ge=0.0 on capital, but we test the function logic
        firm = _make_firm(capital=0.0)
        assert liquidate_firm(firm) >= 0.0


# ============================================================================
# enforce_budget_constraint tests
# ============================================================================


class TestEnforceBudgetConstraint:
    def test_feasible_decision_unchanged(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=50.0, leisure=0.2)
        budget = 200.0
        result = enforce_budget_constraint(decision, agent, budget, a_min=0.0)
        assert result.consumption == 50.0
        assert result.leisure == 0.2

    def test_clamp_leisure_above_one(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=50.0, leisure=1.0)
        # leisure=1.0 is valid for the model but let's test exact boundary
        result = enforce_budget_constraint(decision, agent, budget=200.0, a_min=0.0)
        assert result.leisure == 1.0

    def test_clamp_negative_leisure(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        # EconomicDecision has ge=0.0 on leisure, so we can't create negative
        # But we can test the boundary
        decision = EconomicDecision(consumption=50.0, leisure=0.0)
        result = enforce_budget_constraint(decision, agent, budget=200.0, a_min=0.0)
        assert result.leisure == 0.0

    def test_consumption_clamped_to_budget(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=500.0, leisure=0.2)
        budget = 200.0
        result = enforce_budget_constraint(decision, agent, budget, a_min=0.0)
        # max_consumption = budget - a_min = 200 - 0 = 200
        assert result.consumption == 200.0

    def test_a_min_constrains_consumption(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=500.0, leisure=0.2)
        budget = 200.0
        result = enforce_budget_constraint(decision, agent, budget, a_min=50.0)
        # max_consumption = budget - a_min = 200 - 50 = 150
        assert result.consumption == 150.0

    def test_zero_budget_zero_consumption(self) -> None:
        agent = _make_household(wealth=0.0, labor_supply=0.0)
        decision = EconomicDecision(consumption=50.0, leisure=0.5)
        budget = 0.0
        result = enforce_budget_constraint(decision, agent, budget, a_min=0.0)
        assert result.consumption == 0.0

    def test_consumption_never_negative(self) -> None:
        agent = _make_household(wealth=10.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=0.0, leisure=0.2)
        budget = -5.0  # edge case: negative budget
        result = enforce_budget_constraint(decision, agent, budget, a_min=0.0)
        assert result.consumption >= 0.0

    def test_returns_new_decision(self) -> None:
        agent = _make_household(wealth=100.0, labor_supply=0.8)
        decision = EconomicDecision(consumption=50.0, leisure=0.2)
        result = enforce_budget_constraint(decision, agent, budget=200.0, a_min=0.0)
        assert isinstance(result, EconomicDecision)


# ============================================================================
# compute_realized_utility tests
# ============================================================================


class TestComputeRealizedUtility:
    def test_basic_utility(self) -> None:
        agent = _make_household(consumption=10.0, leisure=0.5, alpha=0.4, beta=0.3, gamma=0.3)
        utility = compute_realized_utility(agent, public_goods_per_capita=5.0)
        expected = (10.0**0.4) * (0.5**0.3) * (5.0**0.3)
        assert abs(utility - expected) < 1e-10

    def test_utility_positive(self) -> None:
        agent = _make_household(consumption=10.0, leisure=0.5)
        assert compute_realized_utility(agent, public_goods_per_capita=5.0) > 0

    def test_higher_consumption_higher_utility(self) -> None:
        agent_lo = _make_household(consumption=5.0, leisure=0.5)
        agent_hi = _make_household(consumption=20.0, leisure=0.5)
        u_lo = compute_realized_utility(agent_lo, public_goods_per_capita=5.0)
        u_hi = compute_realized_utility(agent_hi, public_goods_per_capita=5.0)
        assert u_hi > u_lo

    def test_higher_leisure_higher_utility(self) -> None:
        agent_lo = _make_household(consumption=10.0, leisure=0.2, labor_supply=0.8)
        agent_hi = _make_household(consumption=10.0, leisure=0.8, labor_supply=0.2)
        u_lo = compute_realized_utility(agent_lo, public_goods_per_capita=5.0)
        u_hi = compute_realized_utility(agent_hi, public_goods_per_capita=5.0)
        assert u_hi > u_lo

    def test_higher_public_goods_higher_utility(self) -> None:
        agent = _make_household(consumption=10.0, leisure=0.5)
        u_lo = compute_realized_utility(agent, public_goods_per_capita=1.0)
        u_hi = compute_realized_utility(agent, public_goods_per_capita=10.0)
        assert u_hi > u_lo

    def test_zero_consumption_uses_epsilon(self) -> None:
        agent = _make_household(consumption=0.0, leisure=0.5)
        utility = compute_realized_utility(agent, public_goods_per_capita=5.0)
        assert utility > 0
        assert math.isfinite(utility)

    def test_zero_leisure_uses_epsilon(self) -> None:
        agent = _make_household(consumption=10.0, leisure=0.0, labor_supply=1.0)
        utility = compute_realized_utility(agent, public_goods_per_capita=5.0)
        assert utility > 0
        assert math.isfinite(utility)

    def test_zero_public_goods_uses_epsilon(self) -> None:
        agent = _make_household(consumption=10.0, leisure=0.5)
        utility = compute_realized_utility(agent, public_goods_per_capita=0.0)
        assert utility > 0
        assert math.isfinite(utility)

    def test_utility_is_finite(self) -> None:
        agent = _make_household(consumption=1e6, leisure=0.99)
        utility = compute_realized_utility(agent, public_goods_per_capita=1e6)
        assert math.isfinite(utility)


# ============================================================================
# validate_household_states tests
# ============================================================================


class TestValidateHouseholdStates:
    def test_valid_households_pass(self) -> None:
        households = [_make_household(f"agent_{i:04d}") for i in range(5)]
        validate_household_states(households)  # should not raise

    def test_nan_wealth_raises(self) -> None:
        h = _make_household()
        # Force NaN via model_copy
        bad_h = h.model_copy(update={"wealth": float("nan")})
        with pytest.raises(NumericalInstabilityError, match="invalid wealth"):
            validate_household_states([bad_h])

    def test_inf_wealth_raises(self) -> None:
        h = _make_household()
        bad_h = h.model_copy(update={"wealth": float("inf")})
        with pytest.raises(NumericalInstabilityError, match="invalid wealth"):
            validate_household_states([bad_h])

    def test_nan_consumption_raises(self) -> None:
        h = _make_household()
        bad_h = h.model_copy(update={"consumption": float("nan")})
        with pytest.raises(NumericalInstabilityError, match="invalid consumption"):
            validate_household_states([bad_h])

    def test_nan_utility_raises(self) -> None:
        h = _make_household()
        bad_h = h.model_copy(update={"realized_utility": float("nan")})
        with pytest.raises(NumericalInstabilityError, match="invalid realized_utility"):
            validate_household_states([bad_h])

    def test_empty_list_passes(self) -> None:
        validate_household_states([])  # should not raise


# ============================================================================
# Integration tests
# ============================================================================


class TestEconomicsIntegration:
    def test_budget_to_constraint_pipeline(self) -> None:
        """Test compute_budget -> enforce_budget_constraint flow."""
        agent = _make_household(wealth=100.0, productivity=2.0, labor_supply=0.8)
        budget = compute_budget(agent, wage=5.0, interest_rate=0.05, tax=10.0, transfer=5.0)
        decision = EconomicDecision(consumption=budget * 2, leisure=0.2)
        result = enforce_budget_constraint(decision, agent, budget, a_min=0.0)
        # Consumption should be clamped to budget + wealth (a_min=0)
        assert result.consumption <= budget + agent.wealth
        assert result.consumption >= 0.0

    def test_firm_production_to_distribution(self) -> None:
        """Test produce_output -> distribute_firm_income pipeline."""
        firm = _make_firm(capital=100.0, labor_demand=10.0, tfp=1.0)
        alpha = 0.33
        output = produce_output(firm, alpha)
        firm_with_output = firm.model_copy(update={"output": output})
        wages, cap_cost, profit = distribute_firm_income(
            firm_with_output, wage=2.0, interest_rate=0.05, delta=0.1
        )
        assert abs(output - (wages + cap_cost + profit)) < 1e-10

    def test_utility_after_consumption(self) -> None:
        """Household with positive consumption/leisure/G should have positive utility."""
        agent = _make_household(consumption=50.0, leisure=0.3, labor_supply=0.7)
        utility = compute_realized_utility(agent, public_goods_per_capita=10.0)
        assert utility > 0
        assert math.isfinite(utility)
