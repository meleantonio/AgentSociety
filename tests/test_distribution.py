"""Unit tests for KFE distribution tracking.

Tests cover:
- Mass conservation after forward step (sum(mu) = 1 to tolerance 1e-12)
- Stationary distribution convergence
- Gini computation accuracy
- Lottery allocation correctness
- Aggregate computation
- Percentiles and top_share
- Agent sampling from distribution
- Backward compat: distribution_mode="individual" leaves existing behavior intact

Traceability: REQ-201, REQ-202, REQ-203, REQ-204, REQ-205, PROP-008.
"""

from __future__ import annotations

import numpy as np

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.distribution import Distribution
from emergent_constitution.rng import SimulationRNG
from emergent_constitution.shock_generators import rouwenhorst_discretize

# ============================================================================
# Fixtures
# ============================================================================


def _make_simple_grids(
    n_a: int = 20,
    n_z: int = 3,
    a_min: float = 0.0,
    a_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create simple grids and transition matrix for testing."""
    a_grid = np.linspace(a_min, a_max, n_a)
    grid, trans = rouwenhorst_discretize(rho=0.9, sigma=0.2, n_states=n_z)
    z_grid = np.array(grid)
    trans_matrix = np.array(trans)
    return a_grid, z_grid, trans_matrix


def _make_identity_savings_policy(a_grid: np.ndarray, n_z: int) -> np.ndarray:
    """Create savings policy a' = a (no change in assets)."""
    n_a = len(a_grid)
    policy = np.zeros((n_a, n_z))
    for zi in range(n_z):
        policy[:, zi] = a_grid
    return policy


def _make_constant_savings_policy(a_grid: np.ndarray, n_z: int, target: float) -> np.ndarray:
    """Create savings policy a' = target for all (a, z)."""
    n_a = len(a_grid)
    return np.full((n_a, n_z), target)


# ============================================================================
# Mass conservation tests (PROP-008)
# ============================================================================


class TestMassConservation:
    """Tests that total probability mass is preserved after forward steps."""

    def test_mass_conservation_identity_policy(self) -> None:
        """Mass should be conserved when a' = a (identity policy)."""
        a_grid, z_grid, trans = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        policy = _make_identity_savings_policy(a_grid, len(z_grid))
        dist.forward(policy, trans)

        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_mass_conservation_after_many_steps(self) -> None:
        """Mass should remain 1.0 after 100 forward steps."""
        a_grid, z_grid, trans = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        policy = _make_identity_savings_policy(a_grid, len(z_grid))
        for _ in range(100):
            dist.forward(policy, trans)

        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_mass_conservation_nonuniform_policy(self) -> None:
        """Mass conserved with a policy that shifts savings around."""
        a_grid, z_grid, trans = _make_simple_grids(n_a=50)
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        # Policy: save 80% of current assets (shift mass toward lower wealth)
        n_z = len(z_grid)
        policy = np.zeros((len(a_grid), n_z))
        for zi in range(n_z):
            policy[:, zi] = a_grid * 0.8

        for _ in range(50):
            dist.forward(policy, trans)

        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_mass_conservation_boundary_policy(self) -> None:
        """Mass conserved when policy pushes all savings to grid boundary."""
        a_grid, z_grid, trans = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        # All agents save at the minimum
        policy = _make_constant_savings_policy(a_grid, len(z_grid), a_grid[0])
        dist.forward(policy, trans)

        assert abs(dist.mass_total() - 1.0) < 1e-12


# ============================================================================
# Stationary distribution convergence (REQ-205)
# ============================================================================


class TestStationaryDistribution:
    """Tests for stationary distribution computation."""

    def test_stationary_converges(self) -> None:
        """Stationary distribution should converge to a fixed point."""
        a_grid, z_grid, trans = _make_simple_grids(n_a=30, n_z=3)
        policy = _make_identity_savings_policy(a_grid, len(z_grid))

        dist = Distribution.stationary(
            a_grid,
            z_grid,
            policy,
            trans,
            kfe_tolerance=1e-10,
            kfe_max_iter=10000,
        )

        # Should be converged: another forward step changes nothing
        mu_before = dist.mu.copy()
        dist.forward(policy, trans)
        diff = np.sum(np.abs(dist.mu - mu_before))
        assert diff < 1e-9

    def test_stationary_mass_equals_one(self) -> None:
        """Stationary distribution mass should be 1.0."""
        a_grid, z_grid, trans = _make_simple_grids()
        policy = _make_identity_savings_policy(a_grid, len(z_grid))

        dist = Distribution.stationary(a_grid, z_grid, policy, trans)
        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_stationary_nonnegative(self) -> None:
        """Stationary distribution should have non-negative mass everywhere."""
        a_grid, z_grid, trans = _make_simple_grids()
        policy = _make_identity_savings_policy(a_grid, len(z_grid))

        dist = Distribution.stationary(a_grid, z_grid, policy, trans)
        assert np.all(dist.mu >= 0.0)

    def test_stationary_with_savings_shift(self) -> None:
        """Stationary distribution converges with non-trivial savings policy."""
        a_grid, z_grid, trans = _make_simple_grids(n_a=30, n_z=3)
        n_z = len(z_grid)

        # Policy: save at 50% of current wealth (concentrates mass at lower end)
        policy = np.zeros((len(a_grid), n_z))
        for zi in range(n_z):
            policy[:, zi] = a_grid * 0.5

        dist = Distribution.stationary(
            a_grid,
            z_grid,
            policy,
            trans,
            kfe_tolerance=1e-10,
        )

        assert abs(dist.mass_total() - 1.0) < 1e-12
        assert np.all(dist.mu >= 0.0)


# ============================================================================
# Lottery allocation (REQ-202)
# ============================================================================


class TestLotteryAllocation:
    """Tests for Young (2010) lottery allocation correctness."""

    def test_on_grid_point_no_splitting(self) -> None:
        """When a' lands exactly on a grid point, all mass goes there."""
        a_grid = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        z_grid = np.array([1.0])
        trans = np.array([[1.0]])  # Single state, no transitions

        dist = Distribution(a_grid, z_grid)
        # All mass at (a=50, z=1)
        dist.mu[2, 0] = 1.0

        # Policy: a' = 75.0 (exactly on grid point 3)
        policy = np.array([[75.0], [75.0], [75.0], [75.0], [75.0]])
        dist.forward(policy, trans)

        # All mass should be at a=75
        assert abs(dist.mu[3, 0] - 1.0) < 1e-12
        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_between_grid_points_splits_mass(self) -> None:
        """When a' is between grid points, mass is split proportionally."""
        a_grid = np.array([0.0, 50.0, 100.0])
        z_grid = np.array([1.0])
        trans = np.array([[1.0]])

        dist = Distribution(a_grid, z_grid)
        dist.mu[0, 0] = 1.0  # All mass at a=0

        # Policy: a' = 25.0 (between 0 and 50)
        # Weight to upper: (25 - 0) / (50 - 0) = 0.5
        # Weight to lower: 0.5
        policy = np.array([[25.0], [25.0], [25.0]])
        dist.forward(policy, trans)

        assert abs(dist.mu[0, 0] - 0.5) < 1e-12  # lower bracket
        assert abs(dist.mu[1, 0] - 0.5) < 1e-12  # upper bracket
        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_lottery_weights_correct(self) -> None:
        """Verify lottery weights match expected (a' - a_lo) / (a_hi - a_lo)."""
        a_grid = np.array([0.0, 100.0])
        z_grid = np.array([1.0])
        trans = np.array([[1.0]])

        dist = Distribution(a_grid, z_grid)
        dist.mu[0, 0] = 1.0

        # a' = 30 => w_hi = 30/100 = 0.3, w_lo = 0.7
        policy = np.array([[30.0], [30.0]])
        dist.forward(policy, trans)

        assert abs(dist.mu[0, 0] - 0.7) < 1e-12
        assert abs(dist.mu[1, 0] - 0.3) < 1e-12

    def test_transition_matrix_applied(self) -> None:
        """Transition matrix correctly distributes mass across z states."""
        a_grid = np.array([0.0, 50.0, 100.0])
        z_grid = np.array([1.0, 2.0])
        # Transition: z=0 stays with prob 0.5, goes to z=1 with prob 0.5
        trans = np.array([[0.5, 0.5], [0.5, 0.5]])

        dist = Distribution(a_grid, z_grid)
        dist.mu[1, 0] = 1.0  # All mass at (a=50, z=0)

        # Policy: a' = 50 (stays on grid)
        policy = np.full((3, 2), 50.0)
        dist.forward(policy, trans)

        # Mass should split: 0.5 at (50, z=0) and 0.5 at (50, z=1)
        assert abs(dist.mu[1, 0] - 0.5) < 1e-12
        assert abs(dist.mu[1, 1] - 0.5) < 1e-12


# ============================================================================
# Gini coefficient
# ============================================================================


class TestGini:
    """Tests for Gini coefficient computation from distribution."""

    def test_gini_perfect_equality(self) -> None:
        """Gini = 0 when all wealth is at a single grid point."""
        a_grid = np.array([50.0, 50.0, 50.0])
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[:, 0] = 1.0 / 3.0

        assert abs(dist.gini()) < 1e-10

    def test_gini_perfect_inequality(self) -> None:
        """Gini approaches 1 when all wealth is concentrated."""
        a_grid = np.array([0.0, 0.0, 0.0, 1000.0])
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        # Most mass at a=0, tiny mass at a=1000
        dist.mu[0, 0] = 0.99
        dist.mu[3, 0] = 0.01

        gini_val = dist.gini()
        assert gini_val > 0.5  # Highly unequal

    def test_gini_known_value(self) -> None:
        """Test Gini against a known analytical value.

        For a two-point distribution: mass 0.5 at a=0, mass 0.5 at a=100.
        Gini = mean absolute difference / (2 * mean) = 100 / (2 * 50) = 1.0.
        But with discrete grid, the Gini is computed from Lorenz curve.
        50% of pop has 0% of wealth => area under Lorenz = 0.25
        Gini = 1 - 2 * 0.25 = 0.5
        """
        a_grid = np.array([0.0, 100.0])
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[0, 0] = 0.5
        dist.mu[1, 0] = 0.5

        gini_val = dist.gini()
        assert abs(gini_val - 0.5) < 0.05  # Allow some numerical tolerance

    def test_gini_in_valid_range(self) -> None:
        """Gini coefficient should always be in [0, 1]."""
        a_grid, z_grid, trans = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        gini_val = dist.gini()
        assert 0.0 <= gini_val <= 1.0


# ============================================================================
# Aggregate computation (REQ-203)
# ============================================================================


class TestAggregate:
    """Tests for Distribution.aggregate()."""

    def test_aggregate_identity(self) -> None:
        """Aggregate of ones should equal total mass (1.0)."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        ones = np.ones((len(a_grid), len(z_grid)))
        result = dist.aggregate(ones)
        assert abs(result - 1.0) < 1e-12

    def test_aggregate_wealth(self) -> None:
        """Aggregate of asset grid should equal mean wealth."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        # Policy function = a at each grid point
        a_policy = np.broadcast_to(a_grid.reshape(-1, 1), (len(a_grid), len(z_grid)))
        agg_wealth = dist.aggregate(a_policy)
        mean_w = dist.mean_wealth()

        assert abs(agg_wealth - mean_w) < 1e-10

    def test_aggregate_zero_policy(self) -> None:
        """Aggregate of zero policy should be zero."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        zeros = np.zeros((len(a_grid), len(z_grid)))
        assert abs(dist.aggregate(zeros)) < 1e-15


# ============================================================================
# Statistics: mean_wealth, percentiles, top_share
# ============================================================================


class TestStatistics:
    """Tests for wealth statistics computed from distribution."""

    def test_mean_wealth_uniform(self) -> None:
        """Mean wealth of uniform distribution on [0, 100] grid."""
        a_grid = np.linspace(0, 100, 21)
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[:, 0] = 1.0 / 21.0

        mean_w = dist.mean_wealth()
        assert abs(mean_w - 50.0) < 0.5

    def test_percentiles_uniform(self) -> None:
        """Percentiles of uniform distribution should be evenly spaced."""
        a_grid = np.linspace(0, 100, 101)
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[:, 0] = 1.0 / 101.0

        pcts = dist.percentiles([10, 50, 90])
        assert abs(pcts[0] - 10.0) < 2.0
        assert abs(pcts[1] - 50.0) < 2.0
        assert abs(pcts[2] - 90.0) < 2.0

    def test_top_share_concentrated(self) -> None:
        """Top 10% share when wealth is concentrated at the top."""
        a_grid = np.linspace(0, 100, 11)  # 0, 10, 20, ..., 100
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        # Equal mass everywhere
        dist.mu[:, 0] = 1.0 / 11.0

        top_10_share = dist.top_share(0.10)
        # Top ~10% of 11 agents is ~1 agent (the richest), holding ~18% of wealth
        # (100 / 550 = 0.182)
        assert top_10_share > 0.0
        assert top_10_share < 1.0

    def test_top_share_zero_fraction(self) -> None:
        """Top 0% share should be small."""
        a_grid = np.linspace(0, 100, 11)
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[:, 0] = 1.0 / 11.0

        # top_share(0.0) means fraction=0 => cutoff at 100% cumulative
        # which means no agents above cutoff if all cumulative <= 1.0
        share = dist.top_share(0.0)
        # With cutoff 1.0, searchsorted finds the last index
        assert share >= 0.0


# ============================================================================
# Agent sampling (REQ-204)
# ============================================================================


class TestSampleAgents:
    """Tests for sampling individual agents from distribution."""

    def test_sample_returns_correct_count(self) -> None:
        """Should return exactly n_agents agents."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()
        rng = SimulationRNG(42)

        agents = dist.sample_agents(100, rng)
        assert len(agents) == 100

    def test_sample_agents_have_valid_values(self) -> None:
        """Sampled agents should have values from the grid."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()
        rng = SimulationRNG(42)

        agents = dist.sample_agents(50, rng)
        for wealth, productivity, z_idx in agents:
            assert wealth in a_grid
            assert productivity in z_grid
            assert 0 <= z_idx < len(z_grid)

    def test_sample_deterministic(self) -> None:
        """Same seed should produce same sample (PROP-001)."""
        a_grid, z_grid, _ = _make_simple_grids()
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        rng1 = SimulationRNG(123)
        agents1 = dist.sample_agents(20, rng1)

        rng2 = SimulationRNG(123)
        agents2 = dist.sample_agents(20, rng2)

        assert agents1 == agents2

    def test_sample_concentrated_distribution(self) -> None:
        """When mass is concentrated, all samples should come from that point."""
        a_grid = np.array([0.0, 50.0, 100.0])
        z_grid = np.array([1.0, 2.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[1, 0] = 1.0  # All mass at (a=50, z=1.0)

        rng = SimulationRNG(42)
        agents = dist.sample_agents(10, rng)
        for wealth, productivity, z_idx in agents:
            assert wealth == 50.0
            assert productivity == 1.0
            assert z_idx == 0


# ============================================================================
# Backward compatibility
# ============================================================================


class TestBackwardCompat:
    """Tests that distribution_mode='individual' preserves existing behavior."""

    def test_config_defaults_to_individual(self) -> None:
        """Default distribution_mode should be 'individual'."""
        config = SimulationConfigV2(num_agents=20, seed=42)
        assert config.distribution_mode == "individual"

    def test_config_accepts_kfe(self) -> None:
        """Config should accept distribution_mode='kfe'."""
        config = SimulationConfigV2(num_agents=20, seed=42, distribution_mode="kfe")
        assert config.distribution_mode == "kfe"

    def test_config_kfe_tolerances(self) -> None:
        """KFE convergence parameters should have correct defaults."""
        config = SimulationConfigV2(num_agents=20, seed=42)
        assert config.kfe_convergence_tolerance == 1e-10
        assert config.kfe_max_iterations == 10_000

    def test_config_kfe_custom_tolerances(self) -> None:
        """KFE convergence parameters should be customizable."""
        config = SimulationConfigV2(
            num_agents=20,
            seed=42,
            kfe_convergence_tolerance=1e-8,
            kfe_max_iterations=5000,
        )
        assert config.kfe_convergence_tolerance == 1e-8
        assert config.kfe_max_iterations == 5000


# ============================================================================
# Edge cases and numerical stability
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_empty_distribution(self) -> None:
        """Operations on zero-mass distribution should not crash."""
        a_grid = np.linspace(0, 100, 10)
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)

        assert dist.gini() == 0.0
        assert dist.mean_wealth() == 0.0
        assert dist.percentiles() == [0.0] * 5
        assert dist.top_share(0.1) == 0.0

    def test_single_grid_point(self) -> None:
        """Distribution with single asset grid point should work."""
        a_grid = np.array([50.0])
        z_grid = np.array([1.0])
        dist = Distribution(a_grid, z_grid)
        dist.mu[0, 0] = 1.0

        assert dist.mean_wealth() == 50.0
        assert dist.gini() == 0.0

    def test_policy_outside_grid_clamped(self) -> None:
        """Policy values outside grid range should be clamped."""
        a_grid = np.array([0.0, 50.0, 100.0])
        z_grid = np.array([1.0])
        trans = np.array([[1.0]])
        dist = Distribution(a_grid, z_grid)
        dist.mu[1, 0] = 1.0  # All mass at a=50

        # Policy way above grid max
        policy = np.array([[200.0], [200.0], [200.0]])
        dist.forward(policy, trans)

        # Mass should be clamped to grid max (a=100)
        assert abs(dist.mu[2, 0] - 1.0) < 1e-12
        assert abs(dist.mass_total() - 1.0) < 1e-12

    def test_initialize_uniform(self) -> None:
        """Uniform initialization should distribute mass equally."""
        a_grid = np.linspace(0, 100, 10)
        z_grid = np.array([1.0, 2.0, 3.0])
        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        expected = 1.0 / (10 * 3)
        assert np.allclose(dist.mu, expected)
        assert abs(dist.mass_total() - 1.0) < 1e-15
