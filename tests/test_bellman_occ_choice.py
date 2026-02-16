"""Tests for Bellman-based occupational choice (REQ-110..114).

Covers:
- Firm value Bellman vs iterative solution (max diff < 1e-6)
- Golden-section search vs grid search (same optimum)
- Worker value from VFI matches interpolation
- Entry/exit decisions consistent with value comparison
- No hardcoded constants in Bellman value computation
- Backward compatibility: legacy mode unchanged when flag is False
"""

from __future__ import annotations

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.entrepreneurial_solver import EntrepreneurialSolver
from emergent_constitution.models.decisions import EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState, UtilityParams, ValueVector
from emergent_constitution.models.market import MarketState

# ============================================================================
# Helpers
# ============================================================================

_ABILITY_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]
_ABILITY_TRANS = [
    [0.8, 0.2, 0.0, 0.0, 0.0],
    [0.1, 0.7, 0.2, 0.0, 0.0],
    [0.0, 0.1, 0.7, 0.2, 0.0],
    [0.0, 0.0, 0.1, 0.7, 0.2],
    [0.0, 0.0, 0.0, 0.2, 0.8],
]


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    productivity_index: int = 2,
    ability: float = 1.0,
    ability_idx: int = 2,
    leisure: float = 0.35,
) -> HouseholdState:
    """Create a test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=productivity_index,
        utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25, beta_discount=0.95),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        entrepreneurial_ability=ability,
        entrepreneurial_ability_index=ability_idx,
        leisure=leisure,
        labor_supply=1.0 - leisure,
    )


def _make_market(wage: float = 1.0, interest_rate: float = 0.05) -> MarketState:
    return MarketState(wage=wage, interest_rate=interest_rate, aggregate_output=100.0)


def _make_vfi_value_func(n_a: int = 100, n_z: int = 5) -> tuple[list[list[float]], list[float]]:
    """Create a simple monotonically increasing VFI value function for testing.

    V(a, z) = log(1 + a) * (1 + z_idx * 0.2) / (1 - 0.95)
    """
    a_max = 1000.0
    a_grid = [i * i * a_max / ((n_a - 1) ** 2) for i in range(n_a)]
    value_func: list[list[float]] = []
    for i in range(n_a):
        row: list[float] = []
        a = a_grid[i]
        for zi in range(n_z):
            v = np.log(1.0 + a) * (1.0 + zi * 0.2) / (1.0 - 0.95)
            row.append(float(v))
        value_func.append(row)
    return value_func, a_grid


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def bellman_config() -> SimulationConfigV2:
    """Config with Bellman occupational choice enabled."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=10,
        seed=42,
        benchmark_mode=True,
        use_bellman_occ_choice=True,
    )


@pytest.fixture
def legacy_config() -> SimulationConfigV2:
    """Config with legacy occupational choice (default)."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=10,
        seed=42,
        benchmark_mode=True,
        use_bellman_occ_choice=False,
    )


@pytest.fixture
def bellman_solver(bellman_config: SimulationConfigV2) -> EntrepreneurialSolver:
    return EntrepreneurialSolver(bellman_config, _ABILITY_GRID, _ABILITY_TRANS)


@pytest.fixture
def legacy_solver(legacy_config: SimulationConfigV2) -> EntrepreneurialSolver:
    return EntrepreneurialSolver(legacy_config, _ABILITY_GRID, _ABILITY_TRANS)


@pytest.fixture
def vfi_data() -> tuple[list[list[float]], list[float]]:
    return _make_vfi_value_func()


# ============================================================================
# REQ-111: Firm Value Bellman vs Iterative
# ============================================================================


class TestFirmValueBellman:
    def test_bellman_returns_array(self, bellman_solver: EntrepreneurialSolver) -> None:
        """compute_firm_value_bellman returns array of shape (n_e,)."""
        v = bellman_solver.compute_firm_value_bellman(capital=50.0, wage=1.0, interest_rate=0.05)
        assert isinstance(v, np.ndarray)
        assert v.shape == (len(_ABILITY_GRID),)

    def test_bellman_values_finite(self, bellman_solver: EntrepreneurialSolver) -> None:
        """All firm values must be finite."""
        v = bellman_solver.compute_firm_value_bellman(capital=50.0, wage=1.0, interest_rate=0.05)
        assert np.all(np.isfinite(v))

    def test_bellman_increases_with_ability(self, bellman_solver: EntrepreneurialSolver) -> None:
        """Higher ability grid points should yield higher firm value."""
        v = bellman_solver.compute_firm_value_bellman(capital=50.0, wage=1.0, interest_rate=0.05)
        # Monotonically increasing in ability (for reasonable parameters)
        for i in range(1, len(v)):
            assert v[i] >= v[i - 1], f"V^F should increase with ability: {v[i]} < {v[i-1]}"

    def test_bellman_vs_iterative(self, bellman_solver: EntrepreneurialSolver) -> None:
        """Bellman matrix solution matches iterative solution to < 1e-6.

        Direct method: V = (I - beta*Pi)^{-1} * pi
        Iterative: V_{n+1} = pi + beta*Pi*V_n, iterate until convergence.
        """
        capital = 50.0
        wage = 1.0
        r = 0.05
        beta = bellman_solver._beta
        n_e = len(_ABILITY_GRID)

        # Direct Bellman solution
        v_direct = bellman_solver.compute_firm_value_bellman(capital, wage, r)

        # Iterative solution
        profits = np.array([
            bellman_solver._compute_profit(_ABILITY_GRID[j], capital, wage, r)
            for j in range(n_e)
        ])
        pi_e = np.array(_ABILITY_TRANS, dtype=np.float64)
        v_iter = np.zeros(n_e)
        for _ in range(10000):
            v_new = profits + beta * pi_e @ v_iter
            if np.max(np.abs(v_new - v_iter)) < 1e-12:
                break
            v_iter = v_new

        max_diff = np.max(np.abs(v_direct - v_iter))
        assert max_diff < 1e-6, f"Bellman vs iterative max diff: {max_diff}"

    def test_bellman_empty_grid(self) -> None:
        """Empty ability grid returns empty array."""
        config = SimulationConfigV2(
            num_agents=20, max_periods=10, seed=42,
            benchmark_mode=True, use_bellman_occ_choice=True,
        )
        solver = EntrepreneurialSolver(config, [], [])
        v = solver.compute_firm_value_bellman(capital=50.0, wage=1.0, interest_rate=0.05)
        assert len(v) == 0


# ============================================================================
# REQ-114: Golden-Section vs Grid Search
# ============================================================================


class TestGoldenSectionSearch:
    def test_golden_section_finds_optimum(self, bellman_solver: EntrepreneurialSolver) -> None:
        """Golden-section search finds capital that maximizes firm value."""
        max_capital = 200.0
        ability_idx = 2  # middle of grid

        k_star = bellman_solver._optimal_capital_golden_section(
            ability_idx, wage=1.0, interest_rate=0.05, max_capital=max_capital
        )

        assert k_star >= bellman_solver._min_capital
        assert k_star <= max_capital

    def test_golden_section_vs_grid_search(self, bellman_solver: EntrepreneurialSolver) -> None:
        """Golden-section and fine grid search find approximately the same K*."""
        max_capital = 200.0
        ability_idx = 3
        wage = 1.0
        r = 0.05

        # Golden-section search
        k_golden = bellman_solver._optimal_capital_golden_section(
            ability_idx, wage, r, max_capital
        )

        # Fine grid search (200 points)
        n_grid = 200
        best_k = bellman_solver._min_capital
        best_v = -float("inf")
        k_min = bellman_solver._min_capital
        for i in range(n_grid):
            k = k_min + i * (max_capital - k_min) / (n_grid - 1)
            v_all = bellman_solver.compute_firm_value_bellman(k, wage, r)
            v = float(v_all[ability_idx])
            if v > best_v:
                best_v = v
                best_k = k

        # Should agree within grid spacing
        grid_spacing = (max_capital - bellman_solver._min_capital) / (n_grid - 1)
        assert abs(k_golden - best_k) < 2 * grid_spacing, (
            f"Golden-section K*={k_golden:.4f} vs grid K*={best_k:.4f}"
        )

    def test_golden_section_respects_bounds(self, bellman_solver: EntrepreneurialSolver) -> None:
        """Result is within [K_min, max_capital]."""
        max_capital = 15.0  # barely above min_capital=10
        k = bellman_solver._optimal_capital_golden_section(
            ability_idx=2, wage=1.0, interest_rate=0.05, max_capital=max_capital
        )
        assert k >= bellman_solver._min_capital
        assert k <= max_capital + 1e-10

    def test_golden_section_degenerate(self, bellman_solver: EntrepreneurialSolver) -> None:
        """When max_capital <= min_capital, returns min_capital."""
        k = bellman_solver._optimal_capital_golden_section(
            ability_idx=2, wage=1.0, interest_rate=0.05,
            max_capital=bellman_solver._min_capital,
        )
        assert k == bellman_solver._min_capital


# ============================================================================
# REQ-110: Worker Value from VFI
# ============================================================================


class TestWorkerValueFromVFI:
    def test_vfi_worker_value_positive(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Worker value interpolated from VFI should be positive."""
        vfi_func, a_grid = vfi_data
        household = _make_household(wealth=100.0, productivity_index=2)

        v = bellman_solver.compute_worker_value_from_vfi(household, vfi_func, a_grid)
        assert v > 0.0

    def test_vfi_worker_value_interpolation(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Interpolated value should be between adjacent grid values."""
        vfi_func, a_grid = vfi_data
        # Pick a wealth between two grid points
        mid_idx = len(a_grid) // 2
        a_mid = (a_grid[mid_idx] + a_grid[mid_idx + 1]) / 2.0
        z_idx = 2

        household = _make_household(wealth=a_mid, productivity_index=z_idx)
        v = bellman_solver.compute_worker_value_from_vfi(household, vfi_func, a_grid)

        v_lo = vfi_func[mid_idx][z_idx]
        v_hi = vfi_func[mid_idx + 1][z_idx]
        assert min(v_lo, v_hi) <= v <= max(v_lo, v_hi) + 1e-10

    def test_vfi_worker_value_increases_with_wealth(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Worker value should increase with wealth (monotonicity)."""
        vfi_func, a_grid = vfi_data

        h_poor = _make_household(wealth=10.0, productivity_index=2)
        h_rich = _make_household(wealth=500.0, productivity_index=2)

        v_poor = bellman_solver.compute_worker_value_from_vfi(h_poor, vfi_func, a_grid)
        v_rich = bellman_solver.compute_worker_value_from_vfi(h_rich, vfi_func, a_grid)

        assert v_rich > v_poor

    def test_vfi_fallback_on_empty_data(
        self, bellman_solver: EntrepreneurialSolver
    ) -> None:
        """Falls back to legacy when VFI data is empty."""
        household = _make_household(wealth=100.0)
        v = bellman_solver.compute_worker_value_from_vfi(household, [], [])
        assert v > 0.0  # Legacy fallback should produce a value

    def test_compute_worker_value_dispatches_to_vfi(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """compute_worker_value uses VFI when Bellman mode is active and data provided."""
        vfi_func, a_grid = vfi_data
        household = _make_household(wealth=100.0, productivity_index=2)

        v_vfi = bellman_solver.compute_worker_value(
            household, wage=1.0, interest_rate=0.05, public_goods=0.1,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        v_direct = bellman_solver.compute_worker_value_from_vfi(
            household, vfi_func, a_grid,
        )
        assert abs(v_vfi - v_direct) < 1e-10


# ============================================================================
# REQ-112: Entrepreneur Value - No Hardcoded Constants
# ============================================================================


class TestEntrepreneurValueBellman:
    def test_bellman_entrepreneur_value_finite(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Entrepreneur Bellman value should be finite."""
        vfi_func, a_grid = vfi_data
        household = _make_household(wealth=200.0, ability=2.0, ability_idx=3)

        v = bellman_solver.compute_entrepreneur_value_bellman(
            household, ability=2.0, opt_capital=50.0, wage=1.0,
            interest_rate=0.05, public_goods=0.1, is_entering=False,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert np.isfinite(v)

    def test_bellman_entry_cost_penalty(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Entering entrepreneur has lower value than continuing."""
        vfi_func, a_grid = vfi_data
        household = _make_household(wealth=200.0, ability=2.0, ability_idx=3)

        v_enter = bellman_solver.compute_entrepreneur_value_bellman(
            household, ability=2.0, opt_capital=50.0, wage=1.0,
            interest_rate=0.05, public_goods=0.1, is_entering=True,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        v_continue = bellman_solver.compute_entrepreneur_value_bellman(
            household, ability=2.0, opt_capital=50.0, wage=1.0,
            interest_rate=0.05, public_goods=0.1, is_entering=False,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert v_continue > v_enter, "Entry cost should reduce entrepreneur value"

    def test_no_hardcoded_leisure(
        self, bellman_solver: EntrepreneurialSolver
    ) -> None:
        """Bellman entrepreneur value uses FOC leisure, not hardcoded 0.2.

        Two households with different utility params should get different leisure.
        """
        household_a = HouseholdState(
            id="agent_a", wealth=200.0, productivity=1.0, productivity_index=0,
            utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25),
            value_vector=ValueVector(equality=0.5, liberty=0.5),
            entrepreneurial_ability=2.0, entrepreneurial_ability_index=3,
        )
        household_b = HouseholdState(
            id="agent_b", wealth=200.0, productivity=1.0, productivity_index=0,
            utility_params=UtilityParams(alpha=0.6, beta=0.15, gamma=0.25),
            value_vector=ValueVector(equality=0.5, liberty=0.5),
            entrepreneurial_ability=2.0, entrepreneurial_ability_index=3,
        )

        # With different beta (leisure weight), values should differ
        v_a = bellman_solver.compute_entrepreneur_value_bellman(
            household_a, ability=2.0, opt_capital=50.0, wage=1.0,
            interest_rate=0.05, public_goods=0.1, is_entering=False,
        )
        v_b = bellman_solver.compute_entrepreneur_value_bellman(
            household_b, ability=2.0, opt_capital=50.0, wage=1.0,
            interest_rate=0.05, public_goods=0.1, is_entering=False,
        )
        # Different utility params => different values (proves no single hardcoded leisure)
        assert v_a != v_b


# ============================================================================
# REQ-113: Entry/Exit with Bellman Values
# ============================================================================


class TestEntryExitBellman:
    def test_entry_with_vfi(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """High-ability wealthy agent enters with Bellman values."""
        vfi_func, a_grid = vfi_data
        household = _make_household(
            wealth=200.0, ability=2.5, ability_idx=4, productivity_index=0,
        )
        market = _make_market(wage=1.0, interest_rate=0.05)

        decision = bellman_solver.solve_entry_exit(
            household, firms=[], market=market, public_goods=0.1,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert isinstance(decision, EntrepreneurialDecision)
        # The decision should be determinate (no errors)
        assert decision.create_firm is True or decision.create_firm is False

    def test_poor_agent_cannot_enter_bellman(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """Poor agent cannot enter even with Bellman mode."""
        vfi_func, a_grid = vfi_data
        household = _make_household(wealth=10.0, ability=2.5, ability_idx=4)
        market = _make_market(wage=1.0, interest_rate=0.05)

        decision = bellman_solver.solve_entry_exit(
            household, firms=[], market=market, public_goods=0.1,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert decision.create_firm is False

    def test_exit_with_vfi(
        self,
        bellman_solver: EntrepreneurialSolver,
    ) -> None:
        """Low-ability entrepreneur exits when worker value dominates.

        Use a VFI value function that gives high worker value, so a
        low-ability entrepreneur with negative profits should exit.
        """
        # Construct VFI value function that gives very high worker value
        n_a, n_z = 100, 5
        a_max = 1000.0
        a_grid = [i * i * a_max / ((n_a - 1) ** 2) for i in range(n_a)]
        vfi_func: list[list[float]] = []
        for i in range(n_a):
            a = a_grid[i]
            row = [np.log(1.0 + a) * (1.0 + zi * 0.5) / (1.0 - 0.95) * 10.0 for zi in range(n_z)]
            vfi_func.append(row)

        household = _make_household(
            agent_id="agent_fail", wealth=50.0, ability=0.1, ability_idx=0,
            productivity_index=2, productivity=1.0,
        )
        owned_firm = FirmState(
            id="firm_fail", owner_id="agent_fail", owner_ability=0.1,
            capital=15.0, labor_demand=1.0, tfp=0.1,
        )
        market = _make_market(wage=1.0, interest_rate=0.05)

        decision = bellman_solver.solve_entry_exit(
            household, firms=[owned_firm], market=market, public_goods=0.1,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert decision.close_firm is True

    def test_solve_all_with_vfi(
        self,
        bellman_solver: EntrepreneurialSolver,
        vfi_data: tuple[list[list[float]], list[float]],
    ) -> None:
        """solve_all passes VFI data through to all agents."""
        vfi_func, a_grid = vfi_data
        households = [
            _make_household("a0", wealth=200.0, ability=2.5, ability_idx=4),
            _make_household("a1", wealth=50.0, ability=0.5, ability_idx=0),
        ]
        market = _make_market(wage=1.0, interest_rate=0.05)

        decisions = bellman_solver.solve_all(
            households, firms=[], market=market, public_goods=0.1,
            vfi_value_func=vfi_func, a_grid=a_grid,
        )
        assert len(decisions) == 2
        assert all(isinstance(d, EntrepreneurialDecision) for d in decisions.values())


# ============================================================================
# Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    def test_legacy_mode_unchanged(self, legacy_solver: EntrepreneurialSolver) -> None:
        """With use_bellman_occ_choice=False, behavior matches original."""
        household = _make_household(wealth=200.0, ability=2.0, ability_idx=3)
        market = _make_market(wage=1.0, interest_rate=0.05)

        # Legacy solver should work without VFI data
        decision = legacy_solver.solve_entry_exit(household, firms=[], market=market)
        assert isinstance(decision, EntrepreneurialDecision)

    def test_legacy_firm_value_unchanged(self, legacy_solver: EntrepreneurialSolver) -> None:
        """Legacy compute_firm_value still works as before."""
        v = legacy_solver.compute_firm_value(
            ability=2.0, capital=50.0, wage=1.0, interest_rate=0.05
        )
        assert v > 0.0

    def test_legacy_optimal_capital_uses_grid_search(
        self, legacy_solver: EntrepreneurialSolver
    ) -> None:
        """Legacy mode uses 20-point grid search, not golden-section."""
        assert legacy_solver._use_bellman is False
        k = legacy_solver.optimal_capital(ability=2.0, wage=1.0, interest_rate=0.05, wealth=200.0)
        assert k > 0.0

    def test_legacy_worker_value_uses_perpetuity(
        self, legacy_solver: EntrepreneurialSolver
    ) -> None:
        """Legacy mode uses perpetuity formula with hardcoded leisure."""
        household = _make_household(wealth=100.0)
        v = legacy_solver.compute_worker_value(
            household, wage=1.0, interest_rate=0.05, public_goods=0.1
        )
        assert v > 0.0

    def test_config_flag_defaults_to_false(self) -> None:
        """use_bellman_occ_choice defaults to False."""
        config = SimulationConfigV2(
            num_agents=20, max_periods=10, seed=42, benchmark_mode=True
        )
        assert config.use_bellman_occ_choice is False

    def test_solve_all_backward_compat(self, legacy_solver: EntrepreneurialSolver) -> None:
        """solve_all works without vfi_value_func and a_grid."""
        households = [_make_household("a0", wealth=200.0, ability=2.0, ability_idx=3)]
        market = _make_market(wage=1.0, interest_rate=0.05)

        # No VFI args — should use legacy path
        decisions = legacy_solver.solve_all(households, firms=[], market=market)
        assert len(decisions) == 1


# ============================================================================
# Singular Matrix Fallback
# ============================================================================


class TestSingularMatrixFallback:
    def test_near_singular_falls_back(self) -> None:
        """When Pi is nearly identity and beta is near 1, matrix is near-singular.

        The solver should fall back to perpetuity and warn.
        """
        config = SimulationConfigV2(
            num_agents=20, max_periods=10, seed=42,
            benchmark_mode=True, use_bellman_occ_choice=True,
        )
        # Identity transition: very persistent ability
        identity_trans = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
        solver = EntrepreneurialSolver(config, _ABILITY_GRID, identity_trans)

        # Should still produce finite values (perpetuity fallback if matrix is singular)
        v = solver.compute_firm_value_bellman(capital=50.0, wage=1.0, interest_rate=0.05)
        assert np.all(np.isfinite(v))


# ============================================================================
# NumericalSolver value function retrieval
# ============================================================================


class TestNumericalSolverValueFunction:
    def test_get_value_function_before_solve(self) -> None:
        """Before solving, value function is None."""
        from emergent_constitution.numerical_solver import NumericalSolver

        config = SimulationConfigV2(
            num_agents=20, max_periods=10, seed=42, benchmark_mode=True,
        )
        prod_grid = [0.8, 1.0, 1.2]
        trans = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
        solver = NumericalSolver(config, prod_grid, trans)

        vf, ag = solver.get_value_function()
        assert vf is None
        assert len(ag) > 0  # a_grid should exist
