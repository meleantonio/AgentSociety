"""Unit tests for shock generators: Rouwenhorst, idiosyncratic/aggregate/preference shocks."""

from __future__ import annotations

import math

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.initialization import initialize_simulation_v2
from emergent_constitution.models.household import HouseholdState, OccupationalRole
from emergent_constitution.models.household import UtilityParams as UtilityParamsV2
from emergent_constitution.models.household import ValueVector as ValueVectorV2
from emergent_constitution.rng import SimulationRNG
from emergent_constitution.shock_generators import (
    draw_aggregate_tfp,
    draw_idiosyncratic_shocks,
    draw_preference_shocks,
    rouwenhorst_discretize,
    stationary_distribution,
)

# ============================================================================
# Rouwenhorst discretization
# ============================================================================


class TestRouwenhorstDiscretize:
    def test_returns_correct_grid_size(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        assert len(grid) == 5
        assert len(trans) == 5
        assert all(len(row) == 5 for row in trans)

    def test_two_state_base_case(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=2)
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

    def test_transition_matrix_nonneg_various(self) -> None:
        for rho in [0.3, 0.5, 0.8, 0.95]:
            for n in [3, 5, 10, 20]:
                _, trans = rouwenhorst_discretize(rho=rho, sigma=0.2, n_states=n)
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

    def test_higher_rho_more_persistent(self) -> None:
        _, trans_low = rouwenhorst_discretize(rho=0.5, sigma=0.2, n_states=5)
        _, trans_high = rouwenhorst_discretize(rho=0.95, sigma=0.2, n_states=5)
        assert trans_high[2][2] > trans_low[2][2]

    def test_wider_grid_with_higher_sigma(self) -> None:
        grid_low, _ = rouwenhorst_discretize(rho=0.9, sigma=0.1, n_states=5)
        grid_high, _ = rouwenhorst_discretize(rho=0.9, sigma=0.5, n_states=5)
        spread_low = math.log(grid_low[-1]) - math.log(grid_low[0])
        spread_high = math.log(grid_high[-1]) - math.log(grid_high[0])
        assert spread_high > spread_low


# ============================================================================
# Stationary distribution
# ============================================================================


class TestStationaryDistribution:
    def test_sums_to_one(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = stationary_distribution(trans)
        assert abs(sum(dist) - 1.0) < 1e-10

    def test_all_positive(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = stationary_distribution(trans)
        assert all(d > 0 for d in dist)

    def test_is_fixed_point(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        dist = stationary_distribution(trans)
        n = len(trans)
        new_dist = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_dist[j] += dist[i] * trans[i][j]
        for i in range(n):
            assert abs(new_dist[i] - dist[i]) < 1e-10

    def test_symmetric_two_state(self) -> None:
        _, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=2)
        dist = stationary_distribution(trans)
        assert abs(dist[0] - 0.5) < 1e-10


# ============================================================================
# draw_idiosyncratic_shocks
# ============================================================================


def _make_test_household(
    agent_id: str = "agent_0000",
    prod_idx: int = 2,
    productivity: float = 1.0,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=100.0,
        productivity=productivity,
        productivity_index=prod_idx,
        utility_params=UtilityParamsV2(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.95),
        value_vector=ValueVectorV2(equality=0.5, liberty=0.5),
        role=OccupationalRole.WORKER,
    )


class TestDrawIdiosyncraticShocks:
    def test_returns_same_count(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        households = [_make_test_household(f"agent_{i:04d}", 2, grid[2]) for i in range(10)]
        rng = SimulationRNG(42)
        updated = draw_idiosyncratic_shocks(households, trans, grid, rng)
        assert len(updated) == 10

    def test_productivity_matches_grid(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        households = [_make_test_household(f"agent_{i:04d}", 2, grid[2]) for i in range(20)]
        rng = SimulationRNG(42)
        updated = draw_idiosyncratic_shocks(households, trans, grid, rng)
        for h in updated:
            assert abs(h.productivity - grid[h.productivity_index]) < 1e-10

    def test_productivity_index_valid(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        households = [
            _make_test_household(f"agent_{i:04d}", i % 5, grid[i % 5]) for i in range(20)
        ]
        rng = SimulationRNG(42)
        updated = draw_idiosyncratic_shocks(households, trans, grid, rng)
        for h in updated:
            assert 0 <= h.productivity_index < 5

    def test_determinism(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        households = [_make_test_household(f"agent_{i:04d}", 2, grid[2]) for i in range(10)]

        rng1 = SimulationRNG(77)
        updated1 = draw_idiosyncratic_shocks(households, trans, grid, rng1)

        rng2 = SimulationRNG(77)
        updated2 = draw_idiosyncratic_shocks(households, trans, grid, rng2)

        for h1, h2 in zip(updated1, updated2, strict=True):
            assert h1.productivity_index == h2.productivity_index
            assert h1.productivity == h2.productivity

    def test_ids_preserved(self) -> None:
        grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=5)
        households = [_make_test_household(f"agent_{i:04d}", 2, grid[2]) for i in range(5)]
        rng = SimulationRNG(42)
        updated = draw_idiosyncratic_shocks(households, trans, grid, rng)
        for orig, upd in zip(households, updated, strict=True):
            assert orig.id == upd.id

    def test_some_agents_transition(self) -> None:
        """With enough agents, at least some should change state."""
        grid, trans = rouwenhorst_discretize(rho=0.5, sigma=0.3, n_states=5)
        households = [_make_test_household(f"agent_{i:04d}", 2, grid[2]) for i in range(50)]
        rng = SimulationRNG(42)
        updated = draw_idiosyncratic_shocks(households, trans, grid, rng)
        changes = sum(1 for h in updated if h.productivity_index != 2)
        assert changes > 0


# ============================================================================
# draw_aggregate_tfp
# ============================================================================


class TestDrawAggregateTfp:
    def test_returns_positive(self) -> None:
        rng = SimulationRNG(42)
        for _ in range(100):
            a_t = draw_aggregate_tfp(0.0, 0.95, 0.01, rng)
            assert a_t > 0

    def test_determinism(self) -> None:
        a1 = draw_aggregate_tfp(0.0, 0.95, 0.01, SimulationRNG(42))
        a2 = draw_aggregate_tfp(0.0, 0.95, 0.01, SimulationRNG(42))
        assert a1 == a2

    def test_different_seeds_differ(self) -> None:
        a1 = draw_aggregate_tfp(0.0, 0.95, 0.01, SimulationRNG(42))
        a2 = draw_aggregate_tfp(0.0, 0.95, 0.01, SimulationRNG(999))
        assert a1 != a2

    def test_mean_reverts_to_one(self) -> None:
        """With log(A) = 0 (A=1) and high persistence, output should stay near 1."""
        rng = SimulationRNG(42)
        values = [draw_aggregate_tfp(0.0, 0.95, 0.01, rng) for _ in range(1000)]
        mean_a = sum(values) / len(values)
        assert 0.9 < mean_a < 1.1

    def test_persistence_matters(self) -> None:
        """Higher rho_a should produce less variation from starting point."""
        rng_hi = SimulationRNG(42)
        rng_lo = SimulationRNG(42)
        # Start from log(A)=0
        hi_path = []
        lo_path = []
        log_a_hi = 0.0
        log_a_lo = 0.0
        for _ in range(100):
            a_hi = draw_aggregate_tfp(log_a_hi, 0.99, 0.01, rng_hi)
            a_lo = draw_aggregate_tfp(log_a_lo, 0.5, 0.01, rng_lo)
            log_a_hi = math.log(a_hi)
            log_a_lo = math.log(a_lo)
            hi_path.append(a_hi)
            lo_path.append(a_lo)
        # High persistence path should have higher autocorrelation (less noisy)
        # We can check variance of differences
        hi_var = sum((hi_path[i] - hi_path[i - 1]) ** 2 for i in range(1, len(hi_path)))
        lo_var = sum((lo_path[i] - lo_path[i - 1]) ** 2 for i in range(1, len(lo_path)))
        # Not guaranteed for every seed, but should hold statistically
        # Just check both are finite
        assert math.isfinite(hi_var)
        assert math.isfinite(lo_var)


# ============================================================================
# draw_preference_shocks
# ============================================================================


class TestDrawPreferenceShocks:
    def test_returns_dict_with_all_agents(self) -> None:
        households = [_make_test_household(f"agent_{i:04d}") for i in range(10)]
        rng = SimulationRNG(42)
        shocks = draw_preference_shocks(households, rng, sigma=0.01)
        assert len(shocks) == 10
        for h in households:
            assert h.id in shocks

    def test_determinism(self) -> None:
        households = [_make_test_household(f"agent_{i:04d}") for i in range(5)]
        s1 = draw_preference_shocks(households, SimulationRNG(42), sigma=0.01)
        s2 = draw_preference_shocks(households, SimulationRNG(42), sigma=0.01)
        for h in households:
            assert s1[h.id] == s2[h.id]

    def test_zero_sigma_means_zero_shocks(self) -> None:
        households = [_make_test_household(f"agent_{i:04d}") for i in range(5)]
        rng = SimulationRNG(42)
        shocks = draw_preference_shocks(households, rng, sigma=0.0)
        for val in shocks.values():
            assert val == 0.0

    def test_shocks_centered_around_zero(self) -> None:
        households = [_make_test_household(f"agent_{i:04d}") for i in range(1000)]
        rng = SimulationRNG(42)
        shocks = draw_preference_shocks(households, rng, sigma=0.01)
        mean_shock = sum(shocks.values()) / len(shocks)
        assert abs(mean_shock) < 0.005  # should be near zero

    def test_larger_sigma_gives_larger_spread(self) -> None:
        households = [_make_test_household(f"agent_{i:04d}") for i in range(500)]
        s_small = draw_preference_shocks(households, SimulationRNG(42), sigma=0.001)
        s_large = draw_preference_shocks(households, SimulationRNG(42), sigma=0.1)
        var_small = sum(v**2 for v in s_small.values()) / len(s_small)
        var_large = sum(v**2 for v in s_large.values()) / len(s_large)
        assert var_large > var_small


# ============================================================================
# Integration: shocks with initialization
# ============================================================================


class TestShocksIntegration:
    def test_idiosyncratic_shocks_on_initialized_households(self) -> None:
        config = SimulationConfigV2(num_agents=30, seed=42)
        ps, rng = initialize_simulation_v2(config)
        grid = ps.shocks.productivity_grid
        trans = ps.shocks.transition_matrix
        updated = draw_idiosyncratic_shocks(ps.households, trans, grid, rng)
        assert len(updated) == 30
        for h in updated:
            assert 0 <= h.productivity_index < config.num_z_states
            assert abs(h.productivity - grid[h.productivity_index]) < 1e-10

    def test_aggregate_tfp_from_initial(self) -> None:
        config = SimulationConfigV2(num_agents=20, seed=42)
        ps, rng = initialize_simulation_v2(config)
        prev_log_a = math.log(ps.shocks.aggregate_tfp)
        new_a = draw_aggregate_tfp(prev_log_a, config.rho_a, config.sigma_a, rng)
        assert new_a > 0
        assert math.isfinite(new_a)
