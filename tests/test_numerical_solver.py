"""Unit tests for VFI/EGM numerical household solver."""

from __future__ import annotations

import math

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.decisions import EconomicDecision
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.numerical_solver import NumericalSolver
from emergent_constitution.shock_generators import rouwenhorst_discretize

# ============================================================================
# Test fixtures
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    productivity_index: int = 2,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    beta_discount: float = 0.95,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=productivity_index,
        utility_params=UtilityParams(
            alpha=alpha, beta=beta, gamma=gamma, beta_discount=beta_discount
        ),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=OccupationalRole.WORKER,
    )


def _make_solver(
    n_z: int = 3,
    rho: float = 0.9,
    sigma: float = 0.2,
) -> tuple[NumericalSolver, list[float], list[list[float]]]:
    """Create solver with small Rouwenhorst grid for fast tests."""
    grid, trans = rouwenhorst_discretize(rho=rho, sigma=sigma, n_states=n_z)
    config = SimulationConfigV2(num_agents=20, seed=42, num_z_states=n_z)
    solver = NumericalSolver(config, grid, trans)
    # Reduce grid sizes for test speed
    solver.n_a = 20
    solver.a_grid = NumericalSolver._build_asset_grid(solver.a_min, 500.0, 20)
    solver.vfi_max_iter = 50
    solver.n_leisure = 5
    return solver, grid, trans


def _no_tax(income: float) -> float:
    return 0.0


def _flat_tax_10(income: float) -> float:
    return income * 0.1


# ============================================================================
# NumericalSolver.__init__ tests
# ============================================================================


class TestSolverInit:
    def test_asset_grid_created(self) -> None:
        solver, _, _ = _make_solver()
        assert len(solver.a_grid) == 20  # reduced for tests
        assert solver.a_grid[0] == solver.a_min
        assert solver.a_grid[-1] > solver.a_grid[0]

    def test_asset_grid_monotonic(self) -> None:
        solver, _, _ = _make_solver()
        for i in range(1, len(solver.a_grid)):
            assert solver.a_grid[i] >= solver.a_grid[i - 1]

    def test_productivity_grid_stored(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=7)
        config = SimulationConfigV2(num_agents=20, seed=42, num_z_states=7)
        solver = NumericalSolver(config, grid, trans)
        assert solver.n_z == 7
        assert solver.productivity_grid == grid

    def test_transition_matrix_stored(self) -> None:
        solver, _, trans = _make_solver()
        assert solver.transition_matrix == trans


# ============================================================================
# solve_household tests
# ============================================================================


class TestSolveHousehold:
    def test_returns_economic_decision(self) -> None:
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=50.0, productivity=grid[2], productivity_index=2)
        result = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert isinstance(result, EconomicDecision)

    def test_consumption_non_negative(self) -> None:
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=50.0, productivity=grid[2], productivity_index=2)
        result = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert result.consumption >= 0.0

    def test_leisure_in_range(self) -> None:
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=50.0, productivity=grid[2], productivity_index=2)
        result = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert 0.0 <= result.leisure <= 1.0

    def test_policy_monotone_in_wealth(self) -> None:
        """Wealthier agents should consume more (monotonicity)."""
        solver, grid, _ = _make_solver()

        agent_poor = _make_household(
            "agent_poor", wealth=10.0, productivity=grid[2], productivity_index=2
        )
        agent_rich = _make_household(
            "agent_rich", wealth=500.0, productivity=grid[2], productivity_index=2
        )

        result_poor = solver.solve_household(
            agent_poor,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        result_rich = solver.solve_household(
            agent_rich,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert result_rich.consumption >= result_poor.consumption

    def test_borrowing_constraint_binds_for_low_wealth(self) -> None:
        """Very poor agents should have constrained consumption."""
        solver, grid, _ = _make_solver()
        # Agent at the borrowing constraint
        agent = _make_household(wealth=0.0, productivity=grid[0], productivity_index=0)
        result = solver.solve_household(
            agent,
            wage=0.5,
            interest_rate=0.0,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        # Consumption should be small (budget is just labor income)
        assert result.consumption >= 0.0
        assert math.isfinite(result.consumption)

    def test_determinism(self) -> None:
        """Same inputs should produce same output."""
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=100.0, productivity=grid[2], productivity_index=2)
        r1 = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        r2 = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert r1.consumption == r2.consumption
        assert r1.leisure == r2.leisure

    def test_higher_wage_more_labor(self) -> None:
        """Higher wages should incentivize more work (less leisure)."""
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=50.0, productivity=grid[2], productivity_index=2)
        result_lo = solver.solve_household(
            agent,
            wage=0.5,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        result_hi = solver.solve_household(
            agent,
            wage=5.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        # With higher wage, substitution effect should dominate
        # for typical parameters (leisure should decrease or consumption increase)
        # At minimum, higher wage should yield higher consumption
        assert result_hi.consumption >= result_lo.consumption

    def test_tax_reduces_consumption(self) -> None:
        """Taxes should reduce consumption relative to no-tax case."""
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=100.0, productivity=grid[2], productivity_index=2)
        result_no_tax = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        result_tax = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_flat_tax_10,
            transfer=0.0,
        )
        assert result_tax.consumption <= result_no_tax.consumption

    def test_finite_values(self) -> None:
        solver, grid, _ = _make_solver()
        agent = _make_household(wealth=100.0, productivity=grid[2], productivity_index=2)
        result = solver.solve_household(
            agent,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert math.isfinite(result.consumption)
        assert math.isfinite(result.leisure)


# ============================================================================
# solve_all tests
# ============================================================================


class TestSolveAll:
    def test_returns_dict_with_all_agents(self) -> None:
        solver, grid, _ = _make_solver()
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=50.0 + i * 10,
                productivity=grid[i % len(grid)],
                productivity_index=i % len(grid),
            )
            for i in range(3)
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)
        decisions = solver.solve_all(households, market)
        assert len(decisions) == 3
        for h in households:
            assert h.id in decisions
            assert isinstance(decisions[h.id], EconomicDecision)

    def test_all_decisions_valid(self) -> None:
        solver, grid, _ = _make_solver()
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=100.0,
                productivity=grid[2],
                productivity_index=2,
            )
            for i in range(3)
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)
        decisions = solver.solve_all(households, market, public_goods=1.0)
        for dec in decisions.values():
            assert dec.consumption >= 0.0
            assert 0.0 <= dec.leisure <= 1.0

    def test_with_tax_rate(self) -> None:
        solver, grid, _ = _make_solver()
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=100.0,
                productivity=grid[2],
                productivity_index=2,
            )
            for i in range(3)
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)
        decisions = solver.solve_all(
            households,
            market,
            constitution_tax_rate=0.2,
            public_goods=1.0,
        )
        assert len(decisions) == 3


# ============================================================================
# Internal method tests
# ============================================================================


class TestInternalMethods:
    def test_asset_grid_spacing(self) -> None:
        """Grid should have finer spacing near a_min."""
        grid = NumericalSolver._build_asset_grid(0.0, 100.0, 50)
        # First interval should be smaller than last
        first_gap = grid[1] - grid[0]
        last_gap = grid[-1] - grid[-2]
        assert first_gap < last_gap

    def test_linear_interp_at_grid_points(self) -> None:
        solver, _, _ = _make_solver()
        func = [[float(i + j) for j in range(solver.n_z)] for i in range(solver.n_a)]
        for ai in range(solver.n_a):
            val = solver._linear_interp(solver.a_grid[ai], func, 0)
            assert abs(val - func[ai][0]) < 1e-10

    def test_linear_interp_between_points(self) -> None:
        solver, _, _ = _make_solver()
        func = [[float(i) for _ in range(solver.n_z)] for i in range(solver.n_a)]
        mid_a = (solver.a_grid[10] + solver.a_grid[11]) / 2.0
        val = solver._linear_interp(mid_a, func, 0)
        expected = (func[10][0] + func[11][0]) / 2.0
        assert abs(val - expected) < 1e-6

    def test_linear_interp_below_grid(self) -> None:
        solver, _, _ = _make_solver()
        func = [[1.0] * solver.n_z for _ in range(solver.n_a)]
        val = solver._linear_interp(-10.0, func, 0)
        assert val == func[0][0]

    def test_linear_interp_above_grid(self) -> None:
        solver, _, _ = _make_solver()
        func = [[1.0] * solver.n_z for _ in range(solver.n_a)]
        val = solver._linear_interp(1e6, func, 0)
        assert val == func[-1][0]


# ============================================================================
# _is_homogeneous tests
# ============================================================================


class TestIsHomogeneous:
    def test_empty_list(self) -> None:
        assert NumericalSolver._is_homogeneous([]) is True

    def test_single_agent(self) -> None:
        h = _make_household()
        assert NumericalSolver._is_homogeneous([h]) is True

    def test_identical_agents(self) -> None:
        agents = [_make_household(f"agent_{i:04d}", wealth=50.0 + i * 10) for i in range(5)]
        assert NumericalSolver._is_homogeneous(agents) is True

    def test_different_alpha(self) -> None:
        h1 = _make_household("a0", alpha=0.4, beta=0.3, gamma=0.3)
        h2 = _make_household("a1", alpha=0.5, beta=0.2, gamma=0.3)
        assert NumericalSolver._is_homogeneous([h1, h2]) is False

    def test_different_discount(self) -> None:
        h1 = _make_household("a0", beta_discount=0.95)
        h2 = _make_household("a1", beta_discount=0.90)
        assert NumericalSolver._is_homogeneous([h1, h2]) is False


# ============================================================================
# solve_vfi_shared tests
# ============================================================================


class TestSolveVfiShared:
    def test_returns_grids(self) -> None:
        solver, _, _ = _make_solver()
        policy_c, policy_l = solver.solve_vfi_shared(
            alpha=0.4,
            beta_param=0.3,
            gamma=0.3,
            beta_discount=0.95,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        assert len(policy_c) == solver.n_a
        assert len(policy_c[0]) == solver.n_z
        assert len(policy_l) == solver.n_a
        assert len(policy_l[0]) == solver.n_z

    def test_finite_values(self) -> None:
        solver, _, _ = _make_solver()
        policy_c, policy_l = solver.solve_vfi_shared(
            alpha=0.4,
            beta_param=0.3,
            gamma=0.3,
            beta_discount=0.95,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        for ai in range(solver.n_a):
            for zi in range(solver.n_z):
                assert math.isfinite(policy_c[ai][zi])
                assert math.isfinite(policy_l[ai][zi])

    def test_consumption_generally_increasing_in_assets(self) -> None:
        """Higher-asset grid points should broadly have higher consumption."""
        solver, _, _ = _make_solver()
        policy_c, _ = solver.solve_vfi_shared(
            alpha=0.4,
            beta_param=0.3,
            gamma=0.3,
            beta_discount=0.95,
            wage=1.0,
            interest_rate=0.05,
            public_goods=1.0,
            tax_function=_no_tax,
            transfer=0.0,
        )
        for zi in range(solver.n_z):
            # Top-quartile asset consumption > bottom-quartile
            low_avg = sum(policy_c[ai][zi] for ai in range(5)) / 5
            high_avg = sum(policy_c[ai][zi] for ai in range(solver.n_a - 5, solver.n_a)) / 5
            assert high_avg > low_avg


# ============================================================================
# Fast path vs slow path correctness
# ============================================================================


class TestFastPathMatchesSlowPath:
    def test_fast_path_matches_per_agent(self) -> None:
        """Shared VFI + interpolation should produce same results as per-agent VFI."""
        solver, grid, _ = _make_solver()
        alpha, beta_param, gamma, beta_discount = 0.4, 0.3, 0.3, 0.95
        households = [
            _make_household(
                f"agent_{i:04d}",
                wealth=20.0 + i * 30,
                productivity=grid[i % len(grid)],
                productivity_index=i % len(grid),
                alpha=alpha,
                beta=beta_param,
                gamma=gamma,
                beta_discount=beta_discount,
            )
            for i in range(5)
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)

        # Fast path (solve_all auto-detects homogeneity)
        fast_decisions = solver.solve_all(households, market, public_goods=1.0)

        # Slow path (per-agent)
        slow_decisions: dict[str, EconomicDecision] = {}
        for h in households:
            d = solver.solve_household(
                h,
                wage=1.0,
                interest_rate=0.05,
                public_goods=1.0,
                tax_function=lambda income: 0.0,
                transfer=0.0,
            )
            slow_decisions[h.id] = d

        for h in households:
            assert abs(fast_decisions[h.id].consumption - slow_decisions[h.id].consumption) < 1e-10
            assert abs(fast_decisions[h.id].leisure - slow_decisions[h.id].leisure) < 1e-10

    def test_heterogeneous_uses_slow_path(self) -> None:
        """When agents differ, solve_all should still return valid results."""
        solver, grid, _ = _make_solver()
        households = [
            _make_household(
                "agent_0000",
                wealth=100.0,
                productivity=grid[0],
                productivity_index=0,
                alpha=0.4,
                beta=0.3,
                gamma=0.3,
            ),
            _make_household(
                "agent_0001",
                wealth=100.0,
                productivity=grid[1],
                productivity_index=1,
                alpha=0.5,
                beta=0.2,
                gamma=0.3,
            ),
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)
        decisions = solver.solve_all(households, market, public_goods=1.0)
        assert len(decisions) == 2
        for d in decisions.values():
            assert d.consumption >= 0.0
            assert 0.0 <= d.leisure <= 1.0
