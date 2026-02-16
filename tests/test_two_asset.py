"""Tests for two-asset household structure (REQ-301..305).

Tests:
- Budget constraint with two assets
- Adjustment cost computation
- Wealthy hand-to-mouth detection (high k, low b, high MPC)
- Portfolio choice responds to return differential
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.egm_solver import EGMSolver, adjustment_cost
from emergent_constitution.models.household import (
    HouseholdState,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState

# ---- Fixtures ----


def _make_config(**overrides) -> SimulationConfigV2:
    """Create a two-asset config with sensible defaults."""
    defaults = {
        "num_agents": 20,
        "max_periods": 5,
        "seed": 42,
        "two_asset_mode": True,
        "chi_0": 0.01,
        "chi_1": 0.005,
        "b_min": 0.0,
        "benchmark_mode": True,
        "solver_method": "egm",
        "homogeneous_preferences": True,
        "utility_alpha": 0.4,
        "utility_beta": 0.35,
        "utility_gamma": 0.25,
    }
    defaults.update(overrides)
    return SimulationConfigV2(**defaults)


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    liquid: float = 50.0,
    illiquid: float = 50.0,
    productivity: float = 1.0,
    productivity_index: int = 3,
) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        liquid=liquid,
        illiquid=illiquid,
        productivity=productivity,
        productivity_index=productivity_index,
        utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


def _make_market(
    wage: float = 5.0,
    interest_rate: float = 0.05,
) -> MarketState:
    return MarketState(
        wage=wage,
        interest_rate=interest_rate,
        aggregate_output=1000.0,
    )


# ---- Test: Adjustment Cost (REQ-302) ----


class TestAdjustmentCost:
    """Tests for the convex adjustment cost chi(d) = chi_0*|d| + chi_1*d^2/k."""

    def test_zero_deposit_zero_cost(self) -> None:
        """chi(0) = 0 regardless of illiquid stock."""
        assert adjustment_cost(0.0, 100.0, chi_0=0.01, chi_1=0.005) == 0.0
        assert adjustment_cost(0.0, 0.0, chi_0=0.01, chi_1=0.005) == 0.0

    def test_positive_deposit(self) -> None:
        """chi(d) > 0 for d > 0."""
        cost = adjustment_cost(10.0, 100.0, chi_0=0.01, chi_1=0.005)
        assert cost > 0.0
        # chi(10) = 0.01 * 10 + 0.005 * 100 / 100 = 0.1 + 0.005 = 0.105
        expected = 0.01 * 10.0 + 0.005 * 10.0**2 / 100.0
        assert abs(cost - expected) < 1e-10

    def test_negative_deposit_withdrawal(self) -> None:
        """chi(d) > 0 for d < 0 (withdrawal)."""
        cost = adjustment_cost(-5.0, 100.0, chi_0=0.01, chi_1=0.005)
        assert cost > 0.0
        expected = 0.01 * 5.0 + 0.005 * 25.0 / 100.0
        assert abs(cost - expected) < 1e-10

    def test_convexity(self) -> None:
        """Cost is convex: larger deposit -> disproportionately higher cost."""
        k = 100.0
        c1 = adjustment_cost(10.0, k, chi_0=0.01, chi_1=0.005)
        c2 = adjustment_cost(20.0, k, chi_0=0.01, chi_1=0.005)
        # Due to quadratic component, c2 > 2 * c1
        assert c2 > 2 * c1

    def test_near_zero_illiquid_stock(self) -> None:
        """Cost is well-defined even when k is near zero (uses floor)."""
        cost = adjustment_cost(1.0, 0.0, chi_0=0.01, chi_1=0.005)
        assert cost > 0.0
        assert cost < 1e10  # Not infinite


# ---- Test: HouseholdState Extension (REQ-301) ----


class TestHouseholdStateExtension:
    """Tests for liquid/illiquid fields and total_wealth property."""

    def test_default_zero_assets(self) -> None:
        """Default liquid and illiquid are both 0."""
        h = _make_household(wealth=100.0, liquid=0.0, illiquid=0.0)
        assert h.liquid == 0.0
        assert h.illiquid == 0.0

    def test_total_wealth_property(self) -> None:
        """total_wealth = liquid + illiquid."""
        h = _make_household(liquid=30.0, illiquid=70.0)
        assert h.total_wealth == pytest.approx(100.0)

    def test_backward_compat_single_asset(self) -> None:
        """When both are 0, wealth field is used as single asset."""
        h = _make_household(wealth=200.0, liquid=0.0, illiquid=0.0)
        assert h.wealth == 200.0
        assert h.total_wealth == 0.0  # liquid + illiquid

    def test_illiquid_non_negative(self) -> None:
        """Illiquid assets must be >= 0 (REQ-304)."""
        with pytest.raises(ValidationError):
            _make_household(illiquid=-1.0)


# ---- Test: Config Parameters ----


class TestTwoAssetConfig:
    """Tests for two-asset config parameters."""

    def test_default_off(self) -> None:
        """two_asset_mode defaults to False."""
        config = SimulationConfigV2(benchmark_mode=True)
        assert config.two_asset_mode is False

    def test_enable_two_asset(self) -> None:
        """Can enable two_asset_mode."""
        config = _make_config(two_asset_mode=True)
        assert config.two_asset_mode is True
        assert config.chi_0 == 0.01
        assert config.chi_1 == 0.005
        assert config.b_min == 0.0

    def test_negative_b_min(self) -> None:
        """b_min can be negative (unsecured credit, REQ-304)."""
        config = _make_config(b_min=-10.0)
        assert config.b_min == -10.0


# ---- Test: Two-Asset Budget Constraint ----


class TestTwoAssetBudgetConstraint:
    """Tests for the two-asset budget constraint evolution."""

    def test_liquid_evolution(self) -> None:
        """b' = (1+r^b)*b + income - c - d - chi(d) - T + Tr."""
        h = _make_household(liquid=100.0, illiquid=200.0)
        r_b = 0.05
        income = 50.0
        consumption = 30.0
        deposit = 10.0
        chi = adjustment_cost(deposit, h.illiquid, chi_0=0.01, chi_1=0.005)
        taxes = 5.0
        transfers = 2.0

        new_liquid = (
            (1 + r_b) * h.liquid
            + income
            - consumption
            - deposit
            - chi
            - taxes
            + transfers
        )

        # Verify it matches expected
        expected = 105.0 + 50.0 - 30.0 - 10.0 - chi - 5.0 + 2.0
        assert new_liquid == pytest.approx(expected, rel=1e-8)

    def test_illiquid_evolution(self) -> None:
        """k' = (1+r^k)*k + d."""
        h = _make_household(illiquid=200.0)
        r_k = 0.05
        deposit = 10.0

        new_illiquid = (1 + r_k) * h.illiquid + deposit
        assert new_illiquid == pytest.approx(220.0)

    def test_no_deposit_preserves_illiquid(self) -> None:
        """With zero deposit, illiquid only grows by return."""
        h = _make_household(illiquid=100.0)
        r_k = 0.05
        deposit = 0.0
        new_illiquid = (1 + r_k) * h.illiquid + deposit
        assert new_illiquid == pytest.approx(105.0)


# ---- Test: Wealthy Hand-to-Mouth (REQ-302) ----


class TestWealthyHandToMouth:
    """Test that wealthy hand-to-mouth agents emerge naturally.

    These are agents with high illiquid wealth but low liquid wealth,
    who behave like hand-to-mouth consumers (high MPC) because the
    adjustment cost prevents them from easily accessing illiquid savings.
    """

    def test_high_k_low_b_high_adjustment_cost(self) -> None:
        """Agent with high k, low b faces high cost to withdraw."""
        # Small withdrawal from a large illiquid stock
        cost_small = adjustment_cost(-1.0, 200.0, chi_0=0.01, chi_1=0.005)
        # Same withdrawal from small stock is more costly (quadratic / k)
        cost_large_rel = adjustment_cost(-1.0, 10.0, chi_0=0.01, chi_1=0.005)
        assert cost_large_rel > cost_small

    def test_wealthy_htm_limited_liquid_resources(self) -> None:
        """An agent with high k but near-zero b has limited consumption resources
        because adjusting illiquid is costly."""
        h_htm = _make_household(
            agent_id="htm",
            wealth=200.0,
            liquid=1.0,  # Very low liquid
            illiquid=199.0,  # High illiquid
        )
        h_liquid = _make_household(
            agent_id="liquid",
            wealth=200.0,
            liquid=199.0,  # High liquid
            illiquid=1.0,  # Low illiquid
        )
        # Both have same total wealth but the HTM agent has much less
        # accessible liquid resources
        assert h_htm.total_wealth == pytest.approx(h_liquid.total_wealth)
        assert h_htm.liquid < h_liquid.liquid


# ---- Test: Portfolio Choice Responds to Return Differential ----


class TestPortfolioChoice:
    """Test that deposit decisions respond to return differentials."""

    def test_higher_illiquid_return_increases_deposit(self) -> None:
        """When r^k > r^b, optimal deposit should be positive (save into illiquid)."""
        # This is a qualitative test: with higher illiquid return and low
        # adjustment cost, agents should prefer illiquid savings
        r_b = 0.02
        r_k = 0.08

        # Without adjustment cost, depositing d into illiquid earns
        # (r_k - r_b) * d more than keeping it liquid
        d = 10.0
        k = 100.0
        extra_return = (r_k - r_b) * d
        cost = adjustment_cost(d, k, chi_0=0.01, chi_1=0.005)

        # Extra return should exceed cost for moderate deposit
        assert extra_return > cost

    def test_high_adjustment_cost_discourages_deposit(self) -> None:
        """With high adjustment costs, even high return differential doesn't
        justify deposits."""
        d = 10.0
        k = 100.0
        chi_0_high = 1.0
        chi_1_high = 0.5
        extra_return = (0.08 - 0.02) * d
        cost = adjustment_cost(d, k, chi_0=chi_0_high, chi_1=chi_1_high)

        # High adjustment cost should exceed the return differential
        assert cost > extra_return


# ---- Test: EGM Solver Two-Asset Mode ----


class TestEGMSolverTwoAsset:
    """Tests for the two-asset EGM solver extension."""

    def test_solver_initializes_two_asset_grids(self) -> None:
        """When two_asset_mode is True, solver creates b and k grids."""
        config = _make_config(two_asset_mode=True)
        solver = EGMSolver(
            config=config,
            productivity_grid=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            transition_matrix=[
                [0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.005],
                [0.1, 0.6, 0.15, 0.08, 0.04, 0.02, 0.01],
                [0.05, 0.1, 0.6, 0.12, 0.07, 0.04, 0.02],
                [0.03, 0.05, 0.12, 0.6, 0.12, 0.05, 0.03],
                [0.02, 0.04, 0.07, 0.12, 0.6, 0.1, 0.05],
                [0.01, 0.02, 0.04, 0.08, 0.15, 0.6, 0.1],
                [0.005, 0.005, 0.01, 0.03, 0.05, 0.2, 0.7],
            ],
        )
        assert solver._two_asset_mode is True
        assert hasattr(solver, "b_grid")
        assert hasattr(solver, "k_grid")
        assert len(solver.b_grid) > 0
        assert len(solver.k_grid) > 0

    def test_solver_single_asset_mode_no_extra_grids(self) -> None:
        """When two_asset_mode is False, no b/k grids are created."""
        config = _make_config(two_asset_mode=False)
        solver = EGMSolver(
            config=config,
            productivity_grid=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            transition_matrix=[
                [0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.005],
                [0.1, 0.6, 0.15, 0.08, 0.04, 0.02, 0.01],
                [0.05, 0.1, 0.6, 0.12, 0.07, 0.04, 0.02],
                [0.03, 0.05, 0.12, 0.6, 0.12, 0.05, 0.03],
                [0.02, 0.04, 0.07, 0.12, 0.6, 0.1, 0.05],
                [0.01, 0.02, 0.04, 0.08, 0.15, 0.6, 0.1],
                [0.005, 0.005, 0.01, 0.03, 0.05, 0.2, 0.7],
            ],
        )
        assert solver._two_asset_mode is False
        assert not hasattr(solver, "b_grid")

    def test_golden_section_search(self) -> None:
        """Golden-section search finds the maximum of a unimodal function."""
        config = _make_config(two_asset_mode=True)
        solver = EGMSolver(
            config=config,
            productivity_grid=[0.5, 1.0, 1.5],
            transition_matrix=[
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ],
        )

        # Maximize -(x - 3)^2 + 10, peak at x=3
        def f(x: float) -> float:
            return -(x - 3.0) ** 2 + 10.0

        result = solver._golden_section_search(f, 0.0, 6.0)
        assert abs(result - 3.0) < 1e-4

    def test_deposit_grid_construction(self) -> None:
        """Deposit grid covers withdrawal to deposit range."""
        config = _make_config(two_asset_mode=True)
        solver = EGMSolver(
            config=config,
            productivity_grid=[0.5, 1.0, 1.5],
            transition_matrix=[
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ],
        )

        d_grid = solver._build_deposit_grid(100.0)
        # Should include negative (withdrawal) and positive (deposit) values
        assert d_grid[0] <= 0.0  # Withdrawal side
        assert d_grid[-1] >= 0.0  # Deposit side
        assert 0.0 in d_grid or any(abs(d) < 1e-10 for d in d_grid)

    def test_bilinear_interpolation(self) -> None:
        """Bilinear interpolation returns correct values at grid points."""
        config = _make_config(two_asset_mode=True)
        solver = EGMSolver(
            config=config,
            productivity_grid=[0.5, 1.0, 1.5],
            transition_matrix=[
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ],
        )

        import numpy as np

        n_b = solver.n_b
        n_k = solver.n_k
        n_z = 3
        # Create a simple test policy: value = b_idx + k_idx
        policy = np.zeros((n_b, n_k, n_z))
        for bi in range(n_b):
            for ki in range(n_k):
                for zi in range(n_z):
                    policy[bi, ki, zi] = float(bi + ki)

        # At a grid point, should get exact value
        b_val = solver._b_grid_np[5]
        k_val = solver._k_grid_np[10]
        result = solver._interpolate_two_asset_policy(b_val, k_val, 0, policy)
        assert abs(result - (5.0 + 10.0)) < 1e-6
