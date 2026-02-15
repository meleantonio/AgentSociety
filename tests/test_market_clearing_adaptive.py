"""Tests for adaptive tatonnement and stale-labor-supply handling in market_clearing.

Covers:
- Stale labor_supply=0 fallback to productivity * 0.5
- Adaptive step size with dampening and oscillation detection
- Floor guards in _analytical_equilibrium for labor/capital supply
"""

from __future__ import annotations

import math

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.market_clearing import (
    _MIN_INTEREST,
    _analytical_equilibrium,
    clear_markets,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState

# ============================================================================
# Helpers
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    labor_supply: float = 0.8,
    consumption: float = 10.0,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=2,
        utility_params=UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.95),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=OccupationalRole.WORKER,
        labor_supply=labor_supply,
        leisure=1.0 - labor_supply if labor_supply <= 1.0 else 0.0,
        consumption=consumption,
    )


def _make_firm(
    firm_id: str = "firm_0000",
    capital: float = 100.0,
    tfp: float = 1.0,
    labor_demand: float = 10.0,
) -> FirmState:
    """Create a minimal test firm."""
    return FirmState(
        id=firm_id,
        owner_id="agent_0000",
        capital=capital,
        labor_demand=labor_demand,
        tfp=tfp,
    )


# ============================================================================
# Stale labor supply tests
# ============================================================================


class TestStaleLaborSupply:
    def test_stale_labor_supply_handled(self) -> None:
        """Households with labor_supply=0 should still produce valid analytical prices.

        When labor_supply is 0 (stale/uninitialized), the market clearing uses
        productivity * 0.5 as the fallback labor supply estimate.
        """
        # Arrange: all households have labor_supply=0 (stale)
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=2.0, labor_supply=0.0)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)

        # Act: clear markets with no firms -> analytical path
        result = clear_markets(households, [], aggregate_tfp=1.0, config=config)

        # Assert: valid equilibrium with positive prices
        assert result.wage > 0.0, "Wage should be positive even with stale labor supply"
        assert math.isfinite(result.wage), "Wage must be finite"
        assert result.aggregate_output > 0.0, "Output should be positive"
        assert result.market_clearing_error == 0.0, "Analytical path has zero clearing error"

    def test_stale_labor_uses_half_productivity(self) -> None:
        """Verify the fallback: when labor_supply=0, effective labor = productivity * 0.5.

        Compare against households with explicit labor_supply=0.5 to confirm
        the stale fallback produces equivalent results.
        """
        # Arrange: stale households (labor_supply=0)
        stale_households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=2.0, labor_supply=0.0)
            for i in range(20)
        ]
        # Equivalent households with explicit labor_supply=0.5
        explicit_households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=2.0, labor_supply=0.5)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)

        # Act
        result_stale = clear_markets(stale_households, [], aggregate_tfp=1.0, config=config)
        result_explicit = clear_markets(explicit_households, [], aggregate_tfp=1.0, config=config)

        # Assert: prices should match since effective labor is the same
        assert abs(result_stale.wage - result_explicit.wage) < 1e-8
        assert abs(result_stale.interest_rate - result_explicit.interest_rate) < 1e-8
        assert abs(result_stale.aggregate_output - result_explicit.aggregate_output) < 1e-8

    def test_mixed_stale_and_active_labor(self) -> None:
        """Mix of stale and active labor_supply values should produce valid prices."""
        households = []
        for i in range(20):
            ls = 0.0 if i % 2 == 0 else 0.8  # half stale, half active
            households.append(
                _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.5, labor_supply=ls)
            )
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)

        result = clear_markets(households, [], aggregate_tfp=1.0, config=config)

        assert result.wage > 0.0
        assert math.isfinite(result.wage)
        assert result.aggregate_output > 0.0


# ============================================================================
# Adaptive step size tests
# ============================================================================


class TestAdaptiveStepConvergence:
    def test_adaptive_step_converges_with_firms(self) -> None:
        """With firms present, tatonnement should use adaptive step and return best prices.

        Even if convergence isn't achieved in max_iter, the best_error tracked
        during the adaptive process should be returned.
        """
        # Arrange: mismatch between supply and firm demands to stress adaptive stepping
        households = [
            _make_household(f"agent_{i:04d}", wealth=50.0, productivity=1.0, labor_supply=0.7)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=80.0, tfp=1.5) for i in range(3)]
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            benchmark_mode=True,
            tatonnement_max_iter=50,
            tatonnement_step_size=0.02,
            tatonnement_tolerance=1e-6,
        )

        # Act
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)

        # Assert: best-effort result with valid prices
        assert isinstance(result, MarketState)
        assert result.wage > 0.0
        assert math.isfinite(result.wage)
        assert math.isfinite(result.interest_rate)
        assert math.isfinite(result.market_clearing_error)
        assert result.aggregate_output > 0.0

    def test_adaptive_step_dampening_prevents_divergence(self) -> None:
        """The 0.5 dampening factor should prevent price overshooting.

        Compare a run with very few iterations: prices should remain bounded.
        """
        households = [
            _make_household(f"agent_{i:04d}", wealth=200.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=30.0, tfp=2.0) for i in range(5)]
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            benchmark_mode=True,
            tatonnement_max_iter=200,
            tatonnement_step_size=0.1,  # aggressive step -- dampening should tame it
        )

        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)

        # Prices should stay bounded (no explosion)
        assert result.wage < 1000.0, "Wage should not explode with dampening"
        assert result.interest_rate < 100.0, "Interest rate should not explode"
        assert math.isfinite(result.market_clearing_error)

    def test_best_error_returned_on_non_convergence(self) -> None:
        """When tatonnement does not converge, the best prices found should be used."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]

        # Very few iterations: guaranteed non-convergence
        config_few = SimulationConfigV2(
            num_agents=20,
            seed=42,
            benchmark_mode=True,
            tatonnement_max_iter=2,
            tatonnement_tolerance=1e-12,  # extremely tight -- won't converge
        )

        # Many iterations: should do better
        config_many = SimulationConfigV2(
            num_agents=20,
            seed=42,
            benchmark_mode=True,
            tatonnement_max_iter=500,
            tatonnement_tolerance=1e-12,
        )

        result_few = clear_markets(households, firms, aggregate_tfp=1.0, config=config_few)
        result_many = clear_markets(households, firms, aggregate_tfp=1.0, config=config_many)

        # Both should return valid results
        assert result_few.wage > 0.0
        assert result_many.wage > 0.0

        # More iterations should yield a better (or equal) clearing error
        assert result_many.market_clearing_error <= result_few.market_clearing_error + 1e-6


# ============================================================================
# Analytical equilibrium floor guards
# ============================================================================


class TestAnalyticalEquilibriumFloorGuards:
    def test_analytical_equilibrium_positive_prices(self) -> None:
        """Analytical equilibrium should produce wage > 0 and r > _MIN_INTEREST."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)
        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)

        result = _analytical_equilibrium(households, [], total_l, total_k, 1.0, config)

        assert result.wage > 0.0
        assert result.interest_rate > _MIN_INTEREST

    def test_floor_guard_near_zero_labor(self) -> None:
        """With near-zero labor supply, floor guard should prevent division by zero."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=0.001, labor_supply=0.001)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)
        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)

        result = _analytical_equilibrium(households, [], total_l, total_k, 1.0, config)

        assert math.isfinite(result.wage)
        assert result.wage > 0.0
        assert math.isfinite(result.interest_rate)

    def test_floor_guard_near_zero_capital(self) -> None:
        """With near-zero capital supply, floor guard should keep prices finite."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=0.0001, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)
        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)

        result = _analytical_equilibrium(households, [], total_l, total_k, 1.0, config)

        assert math.isfinite(result.wage)
        assert math.isfinite(result.interest_rate)
        assert result.aggregate_output >= 0.0

    def test_floor_guard_zero_supply_inputs(self) -> None:
        """Passing zero total_labor_supply and total_capital_supply should not crash.

        The floor guards (max(..., 1e-8)) prevent division by zero.
        """
        households = [_make_household(f"agent_{i:04d}", wealth=0.0) for i in range(20)]
        config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)

        # Explicitly pass zero supply values
        result = _analytical_equilibrium(households, [], 0.0, 0.0, 1.0, config)

        assert math.isfinite(result.wage)
        assert math.isfinite(result.interest_rate)
        assert result.wage > 0.0
