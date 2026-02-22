"""Unit tests for EntrepreneurialSolver — firm value, entry/exit, optimal capital."""

from __future__ import annotations

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.entrepreneurial_solver import EntrepreneurialSolver
from emergent_constitution.models.decisions import EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.shock_generators import rouwenhorst_discretize

# ============================================================================
# Helpers
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    ability: float = 1.0,
    ability_idx: int = 2,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        entrepreneurial_ability=ability,
        entrepreneurial_ability_index=ability_idx,
    )


def _make_market(wage: float = 1.0, interest_rate: float = 0.05) -> MarketState:
    """Create a minimal test market state."""
    return MarketState(wage=wage, interest_rate=interest_rate, aggregate_output=100.0)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config() -> SimulationConfigV2:
    return SimulationConfigV2(benchmark_mode=True)


@pytest.fixture
def ability_grid_and_matrix() -> tuple[list[float], list[list[float]]]:
    grid, matrix = rouwenhorst_discretize(rho=0.85, sigma=0.3, n_states=5)
    return grid, matrix


@pytest.fixture
def solver(
    config: SimulationConfigV2,
    ability_grid_and_matrix: tuple[list[float], list[list[float]]],
) -> EntrepreneurialSolver:
    grid, matrix = ability_grid_and_matrix
    return EntrepreneurialSolver(
        config=config, ability_grid=grid, ability_transition_matrix=matrix
    )


# ============================================================================
# compute_firm_value tests
# ============================================================================


class TestComputeFirmValue:
    def test_firm_value_positive(self, solver: EntrepreneurialSolver) -> None:
        """A firm with good ability and decent capital has positive value."""
        value = solver.compute_firm_value(ability=2.0, capital=50.0, wage=1.0, interest_rate=0.05)
        assert value > 0.0

    def test_firm_value_increases_with_ability(self, solver: EntrepreneurialSolver) -> None:
        """Higher ability produces higher firm value, all else equal."""
        value_low = solver.compute_firm_value(
            ability=1.0, capital=50.0, wage=1.0, interest_rate=0.05
        )
        value_high = solver.compute_firm_value(
            ability=2.0, capital=50.0, wage=1.0, interest_rate=0.05
        )
        assert value_high > value_low

    def test_firm_value_zero_capital(self, solver: EntrepreneurialSolver) -> None:
        """Firm value is 0 when capital <= 0."""
        assert (
            solver.compute_firm_value(ability=2.0, capital=0.0, wage=1.0, interest_rate=0.05)
            == 0.0
        )
        assert (
            solver.compute_firm_value(ability=2.0, capital=-5.0, wage=1.0, interest_rate=0.05)
            == 0.0
        )


# ============================================================================
# Entry / exit decision tests
# ============================================================================


class TestEntryExit:
    def test_entry_decision_for_high_ability(self, solver: EntrepreneurialSolver) -> None:
        """A wealthy agent with high ability should create a firm."""
        household = _make_household(wealth=200.0, ability=3.0, ability_idx=4)
        market = _make_market(wage=1.0, interest_rate=0.05)
        decision = solver.solve_entry_exit(household, firms=[], market=market)
        assert isinstance(decision, EntrepreneurialDecision)
        assert decision.create_firm is True
        assert decision.capital_investment > 0.0
        assert decision.labor_demand > 0.0

    def test_no_entry_for_low_ability(self, solver: EntrepreneurialSolver) -> None:
        """An agent with low ability should stay as a worker."""
        household = _make_household(wealth=200.0, ability=0.3, ability_idx=0)
        market = _make_market(wage=1.0, interest_rate=0.05)
        decision = solver.solve_entry_exit(household, firms=[], market=market)
        assert decision.create_firm is False

    def test_no_entry_for_poor_agent(
        self, solver: EntrepreneurialSolver, config: SimulationConfigV2
    ) -> None:
        """An agent with insufficient wealth cannot enter."""
        # Wealth must be strictly less than min_firm_capital + firm_entry_cost
        min_wealth_needed = config.min_firm_capital + config.firm_entry_cost
        household = _make_household(wealth=min_wealth_needed - 1.0, ability=3.0, ability_idx=4)
        market = _make_market(wage=1.0, interest_rate=0.05)
        decision = solver.solve_entry_exit(household, firms=[], market=market)
        assert decision.create_firm is False
        assert decision.capital_investment == 0.0

    def test_exit_when_unprofitable(self, solver: EntrepreneurialSolver) -> None:
        """An entrepreneur with a low-ability firm should close it."""
        household = _make_household(agent_id="agent_0001", wealth=50.0, ability=0.1, ability_idx=0)
        owned_firm = FirmState(
            id="firm_0001",
            owner_id="agent_0001",
            owner_ability=0.1,
            capital=30.0,
            labor_demand=1.0,
            tfp=0.1,
        )
        market = _make_market(wage=1.0, interest_rate=0.05)
        decision = solver.solve_entry_exit(household, firms=[owned_firm], market=market)
        assert decision.close_firm is True


# ============================================================================
# optimal_capital tests
# ============================================================================


class TestOptimalCapital:
    def test_optimal_capital_bounded_by_wealth(
        self, solver: EntrepreneurialSolver, config: SimulationConfigV2
    ) -> None:
        """Optimal K* should be <= wealth - entry_cost."""
        wealth = 200.0
        opt_k = solver.optimal_capital(ability=2.0, wage=1.0, interest_rate=0.05, wealth=wealth)
        assert opt_k <= wealth - config.firm_entry_cost
        assert opt_k >= config.min_firm_capital

    def test_optimal_capital_zero_when_poor(
        self, solver: EntrepreneurialSolver, config: SimulationConfigV2
    ) -> None:
        """Returns 0 when wealth < min_capital + entry_cost."""
        poor_wealth = config.min_firm_capital + config.firm_entry_cost - 1.0
        opt_k = solver.optimal_capital(
            ability=2.0, wage=1.0, interest_rate=0.05, wealth=poor_wealth
        )
        assert opt_k == 0.0


# ============================================================================
# solve_all tests
# ============================================================================


class TestSolveAll:
    def test_solve_all_produces_decisions(
        self,
        solver: EntrepreneurialSolver,
        ability_grid_and_matrix: tuple[list[float], list[list[float]]],
    ) -> None:
        """With heterogeneous agents, solve_all returns a decision for each agent."""
        grid, _ = ability_grid_and_matrix
        households = [
            # Wealthy, high ability
            _make_household(agent_id="agent_0000", wealth=200.0, ability=grid[-1], ability_idx=4),
            # Wealthy, low ability
            _make_household(agent_id="agent_0001", wealth=200.0, ability=grid[0], ability_idx=0),
            # Poor, high ability
            _make_household(agent_id="agent_0002", wealth=5.0, ability=grid[-1], ability_idx=4),
            # Poor, low ability
            _make_household(agent_id="agent_0003", wealth=5.0, ability=grid[0], ability_idx=0),
            # Middle
            _make_household(agent_id="agent_0004", wealth=100.0, ability=grid[2], ability_idx=2),
        ]
        market = _make_market(wage=1.0, interest_rate=0.05)
        decisions = solver.solve_all(households, firms=[], market=market)

        assert len(decisions) == len(households)
        for h in households:
            assert h.id in decisions
            assert isinstance(decisions[h.id], EntrepreneurialDecision)


# ============================================================================
# _optimal_labor tests
# ============================================================================


class TestOptimalLabor:
    def test_optimal_labor_positive(self, solver: EntrepreneurialSolver) -> None:
        """_optimal_labor returns positive value with valid inputs."""
        labor = solver._optimal_labor(ability=2.0, capital=50.0, wage=1.0)
        assert labor > 0.0

    def test_optimal_labor_zero_on_bad_inputs(self, solver: EntrepreneurialSolver) -> None:
        """_optimal_labor returns 0 when wage/capital/ability <= 0."""
        assert solver._optimal_labor(ability=0.0, capital=50.0, wage=1.0) == 0.0
        assert solver._optimal_labor(ability=2.0, capital=0.0, wage=1.0) == 0.0
        assert solver._optimal_labor(ability=2.0, capital=50.0, wage=0.0) == 0.0
        assert solver._optimal_labor(ability=-1.0, capital=50.0, wage=1.0) == 0.0
        assert solver._optimal_labor(ability=2.0, capital=-10.0, wage=1.0) == 0.0
        assert solver._optimal_labor(ability=2.0, capital=50.0, wage=-1.0) == 0.0


class TestStationaryDistribution:
    def test_matches_left_eigenvector(self, config: SimulationConfigV2) -> None:
        """Power-iteration stationary distribution should match eigenvector benchmark."""
        ability_grid = [0.8, 1.0, 1.2]
        transition = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
        solver = EntrepreneurialSolver(
            config=config,
            ability_grid=ability_grid,
            ability_transition_matrix=transition,
        )

        stationary = np.array(solver._stationary_dist, dtype=np.float64)
        assert stationary.shape == (3,)
        assert float(np.sum(stationary)) == pytest.approx(1.0)

        # Reference: left eigenvector of P associated with eigenvalue 1.
        eigvals, eigvecs = np.linalg.eig(np.array(transition, dtype=np.float64).T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        ref = np.real(eigvecs[:, idx])
        ref = ref / np.sum(ref)
        if np.any(ref < 0.0):
            ref = -ref
        ref = ref / np.sum(ref)

        assert np.max(np.abs(stationary - ref)) < 1e-10
