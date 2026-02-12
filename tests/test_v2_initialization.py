"""Unit tests for v2 initialization: households, Rouwenhorst, market, determinism."""

from __future__ import annotations

import math

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.initialization import (
    _draw_from_distribution,
    _initial_market_guess,
    _stationary_distribution,
    create_households,
    initialize_simulation_v2,
    rouwenhorst_discretize,
)
from emergent_constitution.models.household import HouseholdState, OccupationalRole
from emergent_constitution.rng import SimulationRNG

# ============================================================================
# Rouwenhorst discretization tests
# ============================================================================


class TestRouwenhorstDiscretize:
    def test_returns_correct_grid_size(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        assert len(grid) == 5
        assert len(trans) == 5
        assert all(len(row) == 5 for row in trans)

    def test_two_state_case(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=2)
        assert len(grid) == 2
        assert len(trans) == 2
        p = (1.0 + 0.9) / 2.0
        assert abs(trans[0][0] - p) < 1e-10
        assert abs(trans[0][1] - (1.0 - p)) < 1e-10
        assert abs(trans[1][0] - (1.0 - p)) < 1e-10
        assert abs(trans[1][1] - p) < 1e-10

    def test_grid_values_positive(self) -> None:
        grid, _ = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=7)
        assert all(g > 0 for g in grid)

    def test_grid_symmetric_in_logs(self) -> None:
        grid, _ = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        log_grid = [math.log(g) for g in grid]
        for i in range(len(log_grid)):
            assert abs(log_grid[i] + log_grid[-(i + 1)]) < 1e-10

    def test_transition_matrix_row_stochastic(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=7)
        for row in trans:
            assert abs(sum(row) - 1.0) < 1e-10
            assert all(x >= 0.0 for x in row)

    def test_transition_matrix_nonnegative(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.5, sigma=0.3, n_states=10)
        for row in trans:
            assert all(x >= -1e-15 for x in row)

    def test_invalid_rho(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            rouwenhorst_discretize(rho=0.0, sigma=0.2, n_states=5)
        with pytest.raises(ValueError, match="rho"):
            rouwenhorst_discretize(rho=1.0, sigma=0.2, n_states=5)

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            rouwenhorst_discretize(rho=0.9, sigma=0.0, n_states=5)

    def test_invalid_n_states(self) -> None:
        with pytest.raises(ValueError, match="n_states"):
            rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=1)

    def test_higher_rho_means_more_persistent(self) -> None:
        """Higher persistence should mean larger diagonal elements."""
        _, trans_low = rouwenhorst_discretize(rho=0.5, sigma=0.2, n_states=5)
        _, trans_high = rouwenhorst_discretize(rho=0.95, sigma=0.2, n_states=5)
        # Middle state diagonal should be higher with higher rho
        assert trans_high[2][2] > trans_low[2][2]


# ============================================================================
# Stationary distribution tests
# ============================================================================


class TestStationaryDistribution:
    def test_sums_to_one(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = _stationary_distribution(trans)
        assert abs(sum(dist) - 1.0) < 1e-10

    def test_all_positive(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = _stationary_distribution(trans)
        assert all(d > 0 for d in dist)

    def test_is_fixed_point(self) -> None:
        """pi * P = pi for stationary distribution."""
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = _stationary_distribution(trans)
        n = len(trans)
        # Compute pi * P
        new_dist = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_dist[j] += dist[i] * trans[i][j]
        for i in range(n):
            assert abs(new_dist[i] - dist[i]) < 1e-10

    def test_symmetric_two_state(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=2)
        dist = _stationary_distribution(trans)
        assert abs(dist[0] - 0.5) < 1e-10
        assert abs(dist[1] - 0.5) < 1e-10


# ============================================================================
# Draw from distribution tests
# ============================================================================


class TestDrawFromDistribution:
    def test_returns_valid_index(self) -> None:
        rng = SimulationRNG(42)
        probs = [0.2, 0.3, 0.5]
        for _ in range(100):
            idx = _draw_from_distribution(rng, probs)
            assert 0 <= idx < len(probs)

    def test_deterministic(self) -> None:
        results1 = []
        results2 = []
        probs = [0.1, 0.4, 0.3, 0.2]
        for _ in range(50):
            results1.append(_draw_from_distribution(SimulationRNG(123), probs))
        for _ in range(50):
            results2.append(_draw_from_distribution(SimulationRNG(123), probs))
        # Each call resets the RNG to 123, so each list element should match
        assert results1 == results2


# ============================================================================
# create_households tests
# ============================================================================


class TestCreateHouseholds:
    @pytest.fixture()
    def config(self) -> SimulationConfigV2:
        return SimulationConfigV2(num_agents=30, seed=42)

    @pytest.fixture()
    def setup(self, config: SimulationConfigV2) -> tuple[list[HouseholdState], SimulationRNG]:
        rng = SimulationRNG(config.seed)
        grid, trans = rouwenhorst_discretize(
            rho=config.rho_z, sigma=config.sigma_z, n_states=config.num_z_states
        )
        dist = _stationary_distribution(trans)
        households = create_households(config, rng, grid, dist)
        return households, rng

    def test_correct_count(self, setup: tuple) -> None:
        households, _ = setup
        assert len(households) == 30

    def test_wealth_above_a_min(self, config: SimulationConfigV2, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            assert h.wealth >= config.a_min

    def test_productivity_positive(self, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            assert h.productivity > 0.0

    def test_productivity_index_valid(self, config: SimulationConfigV2, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            assert 0 <= h.productivity_index < config.num_z_states

    def test_utility_params_sum_to_one(self, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            total = h.utility_params.alpha + h.utility_params.beta + h.utility_params.gamma
            assert abs(total - 1.0) < 1e-6

    def test_beta_discount_in_range(self, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            assert 0.9 <= h.utility_params.beta_discount <= 0.99

    def test_value_vector_sums_to_one(self, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            total = h.value_vector.equality + h.value_vector.liberty
            assert abs(total - 1.0) < 1e-6

    def test_all_workers_initially(self, setup: tuple) -> None:
        households, _ = setup
        for h in households:
            assert h.role == OccupationalRole.WORKER

    def test_unique_ids(self, setup: tuple) -> None:
        households, _ = setup
        ids = [h.id for h in households]
        assert len(set(ids)) == len(ids)

    def test_productivity_matches_grid(self, config: SimulationConfigV2, setup: tuple) -> None:
        households, _ = setup
        grid, _ = rouwenhorst_discretize(
            rho=config.rho_z, sigma=config.sigma_z, n_states=config.num_z_states
        )
        for h in households:
            assert abs(h.productivity - grid[h.productivity_index]) < 1e-10


# ============================================================================
# initialize_simulation_v2 tests
# ============================================================================


class TestInitializeSimulationV2:
    def test_returns_period_state_and_rng(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        period_state, rng = initialize_simulation_v2(config)
        assert period_state.period == 0
        assert rng.seed == 42

    def test_household_count(self) -> None:
        config = SimulationConfigV2(num_agents=50, seed=42)
        period_state, _ = initialize_simulation_v2(config)
        assert len(period_state.households) == 50

    def test_no_firms_initially(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        period_state, _ = initialize_simulation_v2(config)
        assert period_state.firms == []

    def test_constitution_has_rules(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        period_state, _ = initialize_simulation_v2(config)
        assert len(period_state.constitution.rules) > 0
        assert "majority_vote" in period_state.constitution.rules
        assert "flat_tax" in period_state.constitution.rules

    def test_market_state_populated(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        period_state, _ = initialize_simulation_v2(config)
        assert period_state.market.wage > 0
        assert period_state.market.aggregate_output > 0

    def test_shock_state_populated(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        period_state, _ = initialize_simulation_v2(config)
        assert len(period_state.shocks.productivity_grid) == config.num_z_states
        assert len(period_state.shocks.transition_matrix) == config.num_z_states
        assert period_state.shocks.aggregate_tfp == 1.0

    def test_determinism_same_seed(self) -> None:
        config = SimulationConfigV2(num_agents=30, seed=77)
        ps1, _ = initialize_simulation_v2(config)
        ps2, _ = initialize_simulation_v2(config)
        for h1, h2 in zip(ps1.households, ps2.households, strict=False):
            assert h1.id == h2.id
            assert h1.wealth == h2.wealth
            assert h1.productivity == h2.productivity
            assert h1.productivity_index == h2.productivity_index
            assert h1.utility_params == h2.utility_params
            assert h1.value_vector == h2.value_vector

    def test_different_seeds_differ(self) -> None:
        config1 = SimulationConfigV2(num_agents=30, seed=42)
        config2 = SimulationConfigV2(num_agents=30, seed=999)
        ps1, _ = initialize_simulation_v2(config1)
        ps2, _ = initialize_simulation_v2(config2)
        # At least some households should differ in wealth
        diffs = sum(
            1
            for h1, h2 in zip(ps1.households, ps2.households, strict=False)
            if abs(h1.wealth - h2.wealth) > 0.01
        )
        assert diffs > 0

    def test_wealth_respects_a_min(self) -> None:
        config = SimulationConfigV2(num_agents=50, seed=42, a_min=5.0)
        period_state, _ = initialize_simulation_v2(config)
        for h in period_state.households:
            assert h.wealth >= 5.0

    def test_custom_num_z_states(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42, num_z_states=10)
        period_state, _ = initialize_simulation_v2(config)
        assert len(period_state.shocks.productivity_grid) == 10
        assert len(period_state.shocks.transition_matrix) == 10
        for h in period_state.households:
            assert 0 <= h.productivity_index < 10


# ============================================================================
# Initial market guess tests
# ============================================================================


class TestInitialMarketGuess:
    def test_wage_positive(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        ps, _ = initialize_simulation_v2(config)
        market = _initial_market_guess(config, ps.households)
        assert market.wage > 0

    def test_output_positive(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        ps, _ = initialize_simulation_v2(config)
        market = _initial_market_guess(config, ps.households)
        assert market.aggregate_output > 0

    def test_different_alpha_changes_wages(self) -> None:
        config1 = SimulationConfigV2(num_agents=20, seed=42, alpha=0.2)
        config2 = SimulationConfigV2(num_agents=20, seed=42, alpha=0.5)
        ps1, _ = initialize_simulation_v2(config1)
        ps2, _ = initialize_simulation_v2(config2)
        m1 = _initial_market_guess(config1, ps1.households)
        m2 = _initial_market_guess(config2, ps2.households)
        # Higher alpha -> lower labor share -> lower wage (roughly)
        assert m1.wage != m2.wage


# ============================================================================
# Homogeneous preferences initialization tests
# ============================================================================


class TestHomogeneousHouseholds:
    def test_all_agents_same_params(self) -> None:
        """With homogeneous_preferences=True, all agents share utility params."""
        config = SimulationConfigV2(num_agents=30, seed=42, homogeneous_preferences=True)
        ps, _ = initialize_simulation_v2(config)
        ref = ps.households[0].utility_params
        for h in ps.households[1:]:
            assert h.utility_params.alpha == ref.alpha
            assert h.utility_params.beta == ref.beta
            assert h.utility_params.gamma == ref.gamma
            assert h.utility_params.beta_discount == ref.beta_discount

    def test_params_match_config(self) -> None:
        """Homogeneous params should come from config-level fields."""
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            utility_alpha=0.5,
            utility_beta=0.3,
            utility_gamma=0.2,
            utility_beta_discount=0.98,
        )
        ps, _ = initialize_simulation_v2(config)
        for h in ps.households:
            assert h.utility_params.alpha == 0.5
            assert h.utility_params.beta == 0.3
            assert h.utility_params.gamma == 0.2
            assert h.utility_params.beta_discount == 0.98

    def test_heterogeneous_agents_differ(self) -> None:
        """With homogeneous_preferences=False, agents should have diverse params."""
        config = SimulationConfigV2(num_agents=30, seed=42, homogeneous_preferences=False)
        ps, _ = initialize_simulation_v2(config)
        alphas = {h.utility_params.alpha for h in ps.households}
        # With Dirichlet draw over 30 agents, should have >1 distinct alpha
        assert len(alphas) > 1

    def test_value_vector_still_heterogeneous(self) -> None:
        """Even with homogeneous prefs, value vectors should differ."""
        config = SimulationConfigV2(num_agents=30, seed=42, homogeneous_preferences=True)
        ps, _ = initialize_simulation_v2(config)
        equalities = {h.value_vector.equality for h in ps.households}
        assert len(equalities) > 1

    def test_determinism_homogeneous(self) -> None:
        """Same seed + homogeneous should give identical results."""
        config = SimulationConfigV2(num_agents=20, seed=77, homogeneous_preferences=True)
        ps1, _ = initialize_simulation_v2(config)
        ps2, _ = initialize_simulation_v2(config)
        for h1, h2 in zip(ps1.households, ps2.households, strict=False):
            assert h1.wealth == h2.wealth
            assert h1.productivity == h2.productivity
            assert h1.utility_params == h2.utility_params
            assert h1.value_vector == h2.value_vector
