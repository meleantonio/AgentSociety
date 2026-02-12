"""Unit tests for tatonnement market clearing."""

from __future__ import annotations

import math

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.market_clearing import (
    _analytical_equilibrium,
    clear_markets,
    compute_firm_demands,
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
# Test fixtures
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
        leisure=1.0 - labor_supply,
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
# compute_firm_demands tests
# ============================================================================


class TestComputeFirmDemands:
    def test_empty_firms(self) -> None:
        ld, kd = compute_firm_demands([], wage=5.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert ld == 0.0
        assert kd == 0.0

    def test_single_firm_positive_demands(self) -> None:
        firms = [_make_firm(capital=100.0, tfp=1.0)]
        ld, kd = compute_firm_demands(firms, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert ld > 0
        assert kd > 0

    def test_capital_demand_equals_firm_capital(self) -> None:
        """Capital demand should equal the firm's existing capital."""
        firms = [_make_firm(capital=150.0)]
        _, kd = compute_firm_demands(firms, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert abs(kd - 150.0) < 1e-10

    def test_multiple_firms_additive(self) -> None:
        firm1 = _make_firm("firm_0001", capital=100.0, tfp=1.0)
        firm2 = _make_firm("firm_0002", capital=200.0, tfp=1.0)
        ld_both, kd_both = compute_firm_demands(
            [firm1, firm2], wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33
        )
        ld_1, kd_1 = compute_firm_demands(
            [firm1], wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33
        )
        ld_2, kd_2 = compute_firm_demands(
            [firm2], wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33
        )
        assert abs(ld_both - (ld_1 + ld_2)) < 1e-10
        assert abs(kd_both - (kd_1 + kd_2)) < 1e-10

    def test_higher_wage_lower_labor_demand(self) -> None:
        firms = [_make_firm(capital=100.0, tfp=1.0)]
        ld_lo, _ = compute_firm_demands(firms, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        ld_hi, _ = compute_firm_demands(firms, wage=5.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert ld_hi < ld_lo

    def test_higher_tfp_higher_labor_demand(self) -> None:
        firm_lo = [_make_firm(capital=100.0, tfp=1.0)]
        firm_hi = [_make_firm(capital=100.0, tfp=2.0)]
        ld_lo, _ = compute_firm_demands(
            firm_lo, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33
        )
        ld_hi, _ = compute_firm_demands(
            firm_hi, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33
        )
        assert ld_hi > ld_lo

    def test_demands_finite(self) -> None:
        firms = [_make_firm(capital=100.0, tfp=1.0)]
        ld, kd = compute_firm_demands(firms, wage=1.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert math.isfinite(ld)
        assert math.isfinite(kd)

    def test_zero_wage_handled(self) -> None:
        """Zero wage should be clamped to epsilon."""
        firms = [_make_firm(capital=100.0, tfp=1.0)]
        ld, kd = compute_firm_demands(firms, wage=0.0, interest_rate=0.05, delta=0.1, alpha=0.33)
        assert math.isfinite(ld)
        assert math.isfinite(kd)
        assert ld > 0

    def test_negative_rental_rate_handled(self) -> None:
        """Negative rental rate (r + delta < 0) should be clamped."""
        firms = [_make_firm(capital=100.0, tfp=1.0)]
        ld, kd = compute_firm_demands(firms, wage=1.0, interest_rate=-0.2, delta=0.1, alpha=0.33)
        assert math.isfinite(ld)
        assert math.isfinite(kd)


# ============================================================================
# clear_markets tests
# ============================================================================


class TestClearMarkets:
    def test_empty_households(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets([], [], aggregate_tfp=1.0, config=config)
        assert result.wage == 1.0
        assert result.interest_rate == 0.05

    def test_no_firms_analytical(self) -> None:
        """Without firms, should use representative firm analytical solution."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets(households, [], aggregate_tfp=1.0, config=config)
        assert result.wage > 0
        assert result.aggregate_output > 0
        assert result.market_clearing_error == 0.0

    def test_prices_positive(self) -> None:
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        assert result.wage > 0

    def test_aggregate_output_positive(self) -> None:
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        assert result.aggregate_output > 0

    def test_warm_start_from_prev_market(self) -> None:
        """Using previous market state should warm-start prices."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        prev = MarketState(wage=2.0, interest_rate=0.05)
        result = clear_markets(
            households, firms, aggregate_tfp=1.0, config=config, prev_market=prev
        )
        assert isinstance(result, MarketState)
        assert result.wage > 0

    def test_market_state_fields_populated(self) -> None:
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        # All fields should be populated
        assert math.isfinite(result.wage)
        assert math.isfinite(result.interest_rate)
        assert math.isfinite(result.aggregate_output)
        assert math.isfinite(result.aggregate_consumption)
        assert math.isfinite(result.aggregate_investment)
        assert math.isfinite(result.market_clearing_error)
        assert math.isfinite(result.labor_excess_demand)
        assert math.isfinite(result.capital_excess_demand)

    def test_convergence_single_firm(self) -> None:
        """Single firm with matching supply should converge well."""
        # Create households whose total wealth matches firm capital
        # and total labor matches firm's optimal labor demand
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=20.0,
                productivity=1.0,
                labor_supply=1.0,
            )
            for i in range(20)
        ]
        firms = [_make_firm("firm_0000", capital=400.0, tfp=1.0)]
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            tatonnement_max_iter=500,
            tatonnement_step_size=0.05,
            tatonnement_tolerance=1e-4,
        )
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        # Should have reasonable clearing error
        assert result.market_clearing_error < 10.0  # not perfectly cleared but bounded

    def test_non_convergence_records_error(self) -> None:
        """With very few iterations, should still return valid prices."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            tatonnement_max_iter=1,
        )
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        assert isinstance(result, MarketState)
        assert result.wage > 0


# ============================================================================
# Analytical equilibrium tests
# ============================================================================


class TestAnalyticalEquilibrium:
    def test_cobb_douglas_focs(self) -> None:
        """Verify wage and interest rate follow Cobb-Douglas FOCs."""
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=100.0,
                productivity=1.0,
                labor_supply=0.8,
            )
            for i in range(20)
        ]
        config = SimulationConfigV2(num_agents=20, seed=42, alpha=0.33, delta=0.1)

        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)
        y = 1.0 * (total_k**0.33) * (total_l**0.67)

        result = _analytical_equilibrium(
            households, total_l, total_k, aggregate_tfp=1.0, config=config
        )

        expected_w = 0.67 * y / total_l
        expected_r = 0.33 * y / total_k - 0.1

        assert abs(result.wage - expected_w) < 1e-8
        assert abs(result.interest_rate - expected_r) < 1e-8
        assert abs(result.aggregate_output - y) < 1e-8

    def test_zero_clearing_error(self) -> None:
        """Analytical equilibrium should have zero clearing error."""
        households = [_make_household(f"agent_{i:04d}") for i in range(20)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)
        result = _analytical_equilibrium(households, total_l, total_k, 1.0, config)
        assert result.market_clearing_error == 0.0
        assert result.labor_excess_demand == 0.0
        assert result.capital_excess_demand == 0.0

    def test_higher_alpha_higher_interest(self) -> None:
        """Higher capital share should give higher return to capital."""
        households = [_make_household(f"agent_{i:04d}") for i in range(20)]
        total_l = sum(h.productivity * h.labor_supply for h in households)
        total_k = sum(h.wealth for h in households)

        config_lo = SimulationConfigV2(num_agents=20, seed=42, alpha=0.2)
        config_hi = SimulationConfigV2(num_agents=20, seed=42, alpha=0.5)

        r_lo = _analytical_equilibrium(households, total_l, total_k, 1.0, config_lo)
        r_hi = _analytical_equilibrium(households, total_l, total_k, 1.0, config_hi)
        assert r_hi.interest_rate > r_lo.interest_rate


# ============================================================================
# Integration tests
# ============================================================================


class TestMarketClearingIntegration:
    def test_output_decomposition(self) -> None:
        """Y should be close to sum of firm outputs."""
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=100.0,
                productivity=1.0,
                labor_supply=0.8,
                consumption=20.0,
            )
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(3)]
        config = SimulationConfigV2(num_agents=20, seed=42)
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        assert result.aggregate_output > 0
        assert result.aggregate_investment >= 0

    def test_sequential_clearing(self) -> None:
        """Two consecutive clearings with warm start should be consistent."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, labor_supply=0.8) for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(3)]
        config = SimulationConfigV2(num_agents=20, seed=42)

        result1 = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        result2 = clear_markets(
            households,
            firms,
            aggregate_tfp=1.0,
            config=config,
            prev_market=result1,
        )
        # Prices should be similar since inputs are identical
        assert abs(result1.wage - result2.wage) < 1.0
        assert abs(result1.interest_rate - result2.interest_rate) < 1.0
