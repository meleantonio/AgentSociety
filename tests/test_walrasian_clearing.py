"""Tests for Walrasian market clearing with heterogeneous firms.

Covers REQ-101 through REQ-106:
- Clearing error < 1e-8 with heterogeneous firms
- Convergence within 100 iterations
- Fallback to representative firm when no firms
- Edge cases: single firm, all same TFP, zero labor supply
- Backward compatibility: analytical method still works
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.market_clearing import (
    _bisect_wage,
    _compute_firm_outputs,
    _compute_labor_supply,
    _firm_labor_demand_at_wage,
    clear_markets,
    clear_markets_walrasian,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)

# ============================================================================
# Test fixtures
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    labor_supply: float = 0.8,
    consumption: float = 10.0,
    role: OccupationalRole = OccupationalRole.WORKER,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=2,
        utility_params=UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.95),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
        labor_supply=labor_supply,
        leisure=1.0 - labor_supply,
        consumption=consumption,
    )


def _make_firm(
    firm_id: str = "firm_0000",
    capital: float = 100.0,
    tfp: float = 1.0,
    labor_demand: float = 10.0,
    owner_id: str = "agent_0000",
) -> FirmState:
    """Create a minimal test firm."""
    return FirmState(
        id=firm_id,
        owner_id=owner_id,
        capital=capital,
        labor_demand=labor_demand,
        tfp=tfp,
    )


def _walrasian_config(**kwargs) -> SimulationConfigV2:
    """Create a config with walrasian market clearing."""
    defaults = {
        "num_agents": 20,
        "seed": 42,
        "market_clearing_method": "walrasian",
    }
    defaults.update(kwargs)
    return SimulationConfigV2(**defaults)


def _analytical_config(**kwargs) -> SimulationConfigV2:
    """Create a config with analytical market clearing (default)."""
    defaults = {
        "num_agents": 20,
        "seed": 42,
        "market_clearing_method": "analytical",
    }
    defaults.update(kwargs)
    return SimulationConfigV2(**defaults)


# ============================================================================
# _compute_labor_supply tests
# ============================================================================


class TestComputeLaborSupply:
    def test_excludes_entrepreneurs(self) -> None:
        """Entrepreneurs should be excluded from labor supply."""
        workers = [_make_household(f"w_{i}", labor_supply=0.8) for i in range(10)]
        entrepreneurs = [
            _make_household(f"e_{i}", labor_supply=0.8, role=OccupationalRole.ENTREPRENEUR)
            for i in range(5)
        ]
        all_hh = workers + entrepreneurs
        ls_all = _compute_labor_supply(all_hh)
        ls_workers_only = _compute_labor_supply(workers)
        assert abs(ls_all - ls_workers_only) < 1e-10

    def test_stale_labor_supply_fallback(self) -> None:
        """When labor_supply is 0, should use 0.5 as fallback."""
        hh = [_make_household("agent_0000", labor_supply=0.0, productivity=2.0)]
        ls = _compute_labor_supply(hh)
        # Expected: 2.0 * 0.5 = 1.0
        assert abs(ls - 1.0) < 1e-10

    def test_positive_floor(self) -> None:
        """Should never return 0 due to 1e-8 floor."""
        hh = [
            _make_household(
                "agent_0000",
                labor_supply=0.0,
                productivity=0.0001,
                role=OccupationalRole.ENTREPRENEUR,
            )
        ]
        ls = _compute_labor_supply(hh)
        assert ls >= 1e-8


# ============================================================================
# _firm_labor_demand_at_wage tests
# ============================================================================


class TestFirmLaborDemandAtWage:
    def test_vectorized_single_firm(self) -> None:
        """Should return correct labor demand for a single firm."""
        tfp = np.array([1.0])
        capital = np.array([100.0])
        alpha = 0.33
        wage = 1.0
        ld = _firm_labor_demand_at_wage(tfp, capital, wage, alpha)
        assert ld.shape == (1,)
        assert ld[0] > 0
        assert np.isfinite(ld[0])

    def test_higher_wage_lower_demand(self) -> None:
        """Higher wage should reduce labor demand."""
        tfp = np.array([1.0, 2.0])
        capital = np.array([100.0, 200.0])
        alpha = 0.33
        ld_lo = _firm_labor_demand_at_wage(tfp, capital, 1.0, alpha)
        ld_hi = _firm_labor_demand_at_wage(tfp, capital, 5.0, alpha)
        assert np.all(ld_hi < ld_lo)

    def test_higher_tfp_higher_demand(self) -> None:
        """Higher TFP should increase labor demand at same wage."""
        capital = np.array([100.0])
        alpha = 0.33
        wage = 1.0
        ld_lo = _firm_labor_demand_at_wage(np.array([1.0]), capital, wage, alpha)
        ld_hi = _firm_labor_demand_at_wage(np.array([2.0]), capital, wage, alpha)
        assert ld_hi[0] > ld_lo[0]

    def test_min_wage_guard(self) -> None:
        """Very low wage should be clamped to _MIN_WAGE."""
        tfp = np.array([1.0])
        capital = np.array([100.0])
        ld = _firm_labor_demand_at_wage(tfp, capital, 0.0, 0.33)
        assert np.isfinite(ld[0])
        assert ld[0] > 0


# ============================================================================
# _bisect_wage tests
# ============================================================================


class TestBisectWage:
    def test_convergence_homogeneous_firms(self) -> None:
        """Bisection should converge for homogeneous firms."""
        firms = [_make_firm(f"f_{i}", capital=100.0, tfp=1.0) for i in range(5)]
        labor_supply = 50.0  # reasonable supply
        w_star, iters = _bisect_wage(firms, labor_supply, alpha=0.33)
        assert w_star > 0
        assert iters <= 100

    def test_clearing_error_within_tolerance(self) -> None:
        """At w*, excess labor demand should be < 1e-8."""
        firms = [_make_firm(f"f_{i}", capital=100.0, tfp=1.0) for i in range(5)]
        labor_supply = 50.0
        alpha = 0.33
        w_star, _ = _bisect_wage(firms, labor_supply, alpha)

        # Verify clearing error
        tfp = np.array([1.0] * 5)
        capital = np.array([100.0] * 5)
        ld = _firm_labor_demand_at_wage(tfp, capital, w_star, alpha)
        eld = float(np.sum(ld)) - labor_supply
        assert abs(eld) < 1e-8

    def test_heterogeneous_firms_clearing(self) -> None:
        """Bisection should clear with heterogeneous TFP and capital."""
        firms = [
            _make_firm("f_0", capital=50.0, tfp=0.5),
            _make_firm("f_1", capital=100.0, tfp=1.0),
            _make_firm("f_2", capital=200.0, tfp=1.5),
            _make_firm("f_3", capital=500.0, tfp=2.0),
        ]
        labor_supply = 100.0
        alpha = 0.33
        w_star, iters = _bisect_wage(firms, labor_supply, alpha)

        tfp = np.array([f.tfp for f in firms])
        capital = np.array([f.capital for f in firms])
        ld = _firm_labor_demand_at_wage(tfp, capital, w_star, alpha)
        eld = float(np.sum(ld)) - labor_supply
        assert abs(eld) < 1e-8
        assert iters <= 100

    def test_empty_firms_returns_min_wage(self) -> None:
        """No firms should return minimum wage."""
        w_star, iters = _bisect_wage([], 10.0, 0.33)
        assert w_star == pytest.approx(1e-6)
        assert iters == 0

    def test_single_firm_convergence(self) -> None:
        """Single firm should converge."""
        firms = [_make_firm("f_0", capital=200.0, tfp=1.5)]
        labor_supply = 30.0
        alpha = 0.33
        w_star, iters = _bisect_wage(firms, labor_supply, alpha)

        tfp = np.array([1.5])
        capital = np.array([200.0])
        ld = _firm_labor_demand_at_wage(tfp, capital, w_star, alpha)
        eld = float(np.sum(ld)) - labor_supply
        assert abs(eld) < 1e-8

    def test_convergence_within_100_iterations(self) -> None:
        """Should always converge within 100 iterations."""
        firms = [
            _make_firm(f"f_{i}", capital=50.0 + i * 50.0, tfp=1.0 + i * 0.3)
            for i in range(10)
        ]
        labor_supply = 200.0
        _, iters = _bisect_wage(firms, labor_supply, alpha=0.33)
        assert iters <= 100


# ============================================================================
# _compute_firm_outputs tests
# ============================================================================


class TestComputeFirmOutputs:
    def test_aggregate_output_positive(self) -> None:
        """Aggregate output should be positive with active firms."""
        firms = [_make_firm(f"f_{i}", capital=100.0, tfp=1.0) for i in range(3)]
        y, firm_labors = _compute_firm_outputs(firms, wage=1.0, alpha=0.33)
        assert y > 0
        assert len(firm_labors) == 3
        assert all(fl > 0 for fl in firm_labors)

    def test_empty_firms(self) -> None:
        """No firms should return zero output."""
        y, firm_labors = _compute_firm_outputs([], wage=1.0, alpha=0.33)
        assert y == 0.0
        assert len(firm_labors) == 0

    def test_output_finite(self) -> None:
        """All outputs should be finite."""
        firms = [
            _make_firm("f_0", capital=50.0, tfp=0.5),
            _make_firm("f_1", capital=1000.0, tfp=3.0),
        ]
        y, firm_labors = _compute_firm_outputs(firms, wage=2.0, alpha=0.33)
        assert math.isfinite(y)
        assert np.all(np.isfinite(firm_labors))

    def test_all_same_tfp(self) -> None:
        """Firms with same TFP and capital should have equal labor demands."""
        firms = [_make_firm(f"f_{i}", capital=100.0, tfp=1.0) for i in range(5)]
        _, firm_labors = _compute_firm_outputs(firms, wage=1.0, alpha=0.33)
        # All should be equal
        for i in range(1, len(firm_labors)):
            assert abs(firm_labors[i] - firm_labors[0]) < 1e-10


# ============================================================================
# clear_markets_walrasian tests
# ============================================================================


class TestClearMarketsWalrasian:
    def test_clearing_error_below_tolerance(self) -> None:
        """Market clearing error should be below 1e-8 (REQ-106)."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.market_clearing_error < 1e-8

    def test_prices_positive(self) -> None:
        """Wage should be positive."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.wage > 0
        assert math.isfinite(result.interest_rate)

    def test_aggregate_output_positive(self) -> None:
        """Aggregate output should be positive."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.aggregate_output > 0

    def test_heterogeneous_firms_clears(self) -> None:
        """Should clear with heterogeneous TFP and capital (REQ-101, REQ-102)."""
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=100.0,
                productivity=1.0 + i * 0.1,
                labor_supply=0.7,
            )
            for i in range(20)
        ]
        firms = [
            _make_firm("firm_0000", capital=50.0, tfp=0.8),
            _make_firm("firm_0001", capital=100.0, tfp=1.0),
            _make_firm("firm_0002", capital=200.0, tfp=1.5),
            _make_firm("firm_0003", capital=500.0, tfp=2.0),
        ]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.market_clearing_error < 1e-8
        assert result.aggregate_output > 0

    def test_fallback_no_firms(self) -> None:
        """With no firms, should fall back to representative firm (REQ-105)."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, [], aggregate_tfp=1.0, config=config)
        assert result.wage > 0
        assert result.aggregate_output > 0
        assert result.market_clearing_error == 0.0

    def test_empty_households(self) -> None:
        """No households should return default prices."""
        config = _walrasian_config()
        result = clear_markets_walrasian([], [], aggregate_tfp=1.0, config=config)
        assert result.wage == 1.0
        assert result.interest_rate == 0.05

    def test_single_firm(self) -> None:
        """Should work correctly with a single firm."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm("firm_0000", capital=500.0, tfp=1.0)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.market_clearing_error < 1e-8
        assert result.wage > 0

    def test_entrepreneurs_excluded_from_labor(self) -> None:
        """Entrepreneurs should not contribute to labor supply (REQ-108)."""
        workers = [
            _make_household(f"w_{i:04d}", productivity=1.0, labor_supply=0.8) for i in range(15)
        ]
        entrepreneurs = [
            _make_household(
                f"e_{i:04d}",
                productivity=1.0,
                labor_supply=0.8,
                role=OccupationalRole.ENTREPRENEUR,
            )
            for i in range(5)
        ]
        firms = [
            _make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0, owner_id=f"e_{i:04d}")
            for i in range(5)
        ]

        config = _walrasian_config()
        result_all = clear_markets_walrasian(workers + entrepreneurs, firms, 1.0, config)
        result_workers = clear_markets_walrasian(workers, firms, 1.0, config)

        # With entrepreneurs excluded, same workers should give same labor supply
        # so wages should be the same
        assert abs(result_all.wage - result_workers.wage) < 1e-6

    def test_market_state_fields_populated(self) -> None:
        """All MarketState fields should be finite."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert math.isfinite(result.wage)
        assert math.isfinite(result.interest_rate)
        assert math.isfinite(result.aggregate_output)
        assert math.isfinite(result.aggregate_consumption)
        assert math.isfinite(result.aggregate_investment)
        assert math.isfinite(result.market_clearing_error)
        assert math.isfinite(result.labor_excess_demand)
        assert math.isfinite(result.capital_excess_demand)


# ============================================================================
# Config dispatch tests
# ============================================================================


class TestConfigDispatch:
    def test_walrasian_dispatch(self) -> None:
        """config.market_clearing_method='walrasian' should use Walrasian clearing."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _walrasian_config()
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        # Walrasian should have very low clearing error
        assert result.market_clearing_error < 1e-8

    def test_analytical_dispatch(self) -> None:
        """config.market_clearing_method='analytical' should use analytical clearing."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0) for i in range(5)]
        config = _analytical_config()
        result = clear_markets(households, firms, aggregate_tfp=1.0, config=config)
        # Analytical always reports 0 clearing error
        assert result.market_clearing_error == 0.0

    def test_default_is_analytical(self) -> None:
        """Default config should use analytical method."""
        config = SimulationConfigV2(num_agents=20, seed=42)
        assert config.market_clearing_method == "analytical"

    def test_backward_compat_analytical(self) -> None:
        """Existing tests with analytical method should still pass."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        config = _analytical_config()
        result = clear_markets(households, [], aggregate_tfp=1.0, config=config)
        assert result.wage > 0
        assert result.aggregate_output > 0


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    def test_very_high_tfp(self) -> None:
        """Very high TFP should still converge."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm("firm_0000", capital=100.0, tfp=100.0)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert math.isfinite(result.wage)
        assert result.wage > 0
        assert result.market_clearing_error < 1e-8

    def test_very_low_tfp(self) -> None:
        """Very low TFP should still converge."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [_make_firm("firm_0000", capital=100.0, tfp=0.01)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert math.isfinite(result.wage)
        assert result.wage > 0

    def test_many_firms(self) -> None:
        """Should handle many firms efficiently."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(50)
        ]
        firms = [
            _make_firm(f"firm_{i:04d}", capital=10.0 + i * 2.0, tfp=0.5 + i * 0.05)
            for i in range(50)
        ]
        config = _walrasian_config(num_agents=50)
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.market_clearing_error < 1e-8

    def test_all_entrepreneurs_labor_supply(self) -> None:
        """When all are entrepreneurs, labor supply hits floor but doesn't crash."""
        entrepreneurs = [
            _make_household(
                f"e_{i:04d}",
                wealth=100.0,
                productivity=1.0,
                labor_supply=0.8,
                role=OccupationalRole.ENTREPRENEUR,
            )
            for i in range(20)
        ]
        firms = [
            _make_firm(f"firm_{i:04d}", capital=50.0, tfp=1.0, owner_id=f"e_{i:04d}")
            for i in range(5)
        ]
        config = _walrasian_config()
        # Should not crash — labor supply floor prevents division by zero
        result = clear_markets_walrasian(entrepreneurs, firms, aggregate_tfp=1.0, config=config)
        assert math.isfinite(result.wage)

    def test_interest_rate_bounded(self) -> None:
        """Interest rate should be bounded from below."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        # Very high capital, low output -> negative r
        firms = [_make_firm("firm_0000", capital=100000.0, tfp=0.01)]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)
        assert result.interest_rate >= -0.99


# ============================================================================
# Integration tests
# ============================================================================


class TestWalrasianIntegration:
    def test_output_equals_sum_of_firm_outputs(self) -> None:
        """Aggregate output should equal sum of firm-level outputs (REQ-103)."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [
            _make_firm("firm_0000", capital=100.0, tfp=1.0),
            _make_firm("firm_0001", capital=200.0, tfp=1.5),
        ]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)

        # Recompute firm outputs at equilibrium wage
        y, _ = _compute_firm_outputs(firms, result.wage, config.alpha)
        assert abs(result.aggregate_output - y) < 1e-8

    def test_wage_equates_demand_and_supply(self) -> None:
        """At equilibrium wage, L^d should equal L^s (REQ-102)."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [
            _make_firm("firm_0000", capital=100.0, tfp=1.0),
            _make_firm("firm_0001", capital=200.0, tfp=2.0),
        ]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)

        # Verify L^d(w*) = L^s
        ls = _compute_labor_supply(households)
        _, firm_labors = _compute_firm_outputs(firms, result.wage, config.alpha)
        ld = float(np.sum(firm_labors))
        assert abs(ld - ls) < 1e-8

    def test_interest_rate_from_aggregate_mpk(self) -> None:
        """r should equal alpha * Y / K_total - delta (REQ-104)."""
        households = [
            _make_household(f"agent_{i:04d}", wealth=100.0, productivity=1.0, labor_supply=0.8)
            for i in range(20)
        ]
        firms = [
            _make_firm("firm_0000", capital=100.0, tfp=1.0),
            _make_firm("firm_0001", capital=200.0, tfp=1.5),
        ]
        config = _walrasian_config()
        result = clear_markets_walrasian(households, firms, aggregate_tfp=1.0, config=config)

        k_total = sum(f.capital for f in firms)
        expected_r = config.alpha * result.aggregate_output / k_total - config.delta
        assert abs(result.interest_rate - expected_r) < 1e-8
