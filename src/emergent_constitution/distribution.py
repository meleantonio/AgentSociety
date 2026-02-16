"""KFE distribution tracking — cross-sectional wealth-productivity distribution.

Implements the Kolmogorov Forward Equation (Young 2010 lottery method)
for tracking the joint distribution of households over assets and
productivity states on a discrete grid.

Traceability: REQ-201, REQ-202, REQ-203, REQ-204, REQ-205, PROP-008.
Design reference: spec/design.md section 2.1.
"""

from __future__ import annotations

import numpy as np
import structlog

from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()

# Numerical safety
_EPSILON = 1e-15
_DEFAULT_KFE_TOLERANCE = 1e-10
_DEFAULT_KFE_MAX_ITER = 10_000
_MASS_CONSERVATION_TOL = 1e-12


class Distribution:
    """Cross-sectional wealth-productivity distribution on a grid.

    Maintains a probability mass array mu(a, z) of shape (n_a, n_z)
    where n_a is the number of asset grid points and n_z is the number
    of productivity states. The distribution evolves via the Kolmogorov
    Forward Equation using the Young (2010) lottery allocation method.

    Args:
        a_grid: Asset grid, shape (n_a,). Must be sorted ascending.
        z_grid: Productivity grid, shape (n_z,).
        kfe_tolerance: Convergence tolerance for stationary distribution.
        kfe_max_iter: Maximum iterations for stationary distribution.
    """

    def __init__(
        self,
        a_grid: np.ndarray | list[float],
        z_grid: np.ndarray | list[float],
        kfe_tolerance: float = _DEFAULT_KFE_TOLERANCE,
        kfe_max_iter: int = _DEFAULT_KFE_MAX_ITER,
    ) -> None:
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.n_a = len(self.a_grid)
        self.n_z = len(self.z_grid)
        self.kfe_tolerance = kfe_tolerance
        self.kfe_max_iter = kfe_max_iter

        # Probability mass array: mu[i, j] = Pr(a = a_grid[i], z = z_grid[j])
        self.mu = np.zeros((self.n_a, self.n_z), dtype=np.float64)

    def initialize_uniform(self) -> None:
        """Initialize distribution to uniform across all grid points."""
        self.mu[:] = 1.0 / (self.n_a * self.n_z)

    def forward(
        self,
        policy_savings: np.ndarray,
        transition_matrix: np.ndarray,
    ) -> None:
        """Advance distribution one period via KFE (Young 2010 lottery).

        For each (a_i, z_j) cell with mass mu(a_i, z_j):
          1. Policy gives a' = g(a_i, z_j) = policy_savings[i, j]
          2. Find bracket: a_grid[lo] <= a' <= a_grid[hi]
          3. Weight: w = (a' - a_lo) / (a_hi - a_lo)
          4. Allocate mass across productivity transitions:
             mu_new(a_hi, z') += w * Pi(z_j, z') * mu(a_i, z_j)
             mu_new(a_lo, z') += (1-w) * Pi(z_j, z') * mu(a_i, z_j)

        All operations are vectorized over (a, z) grid points (REQ-201, REQ-202).

        Args:
            policy_savings: Savings policy a' = g(a, z), shape (n_a, n_z).
            transition_matrix: Markov transition Pi(z, z'), shape (n_z, n_z).

        Implements REQ-201, REQ-202, PROP-008.
        """
        policy_savings = np.asarray(policy_savings, dtype=np.float64)
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)

        n_a = self.n_a
        n_z = self.n_z
        a_grid = self.a_grid

        # Clamp policy to grid range
        a_prime = np.clip(policy_savings, a_grid[0], a_grid[-1])

        # Find upper bracket indices using searchsorted
        idx_hi = np.searchsorted(a_grid, a_prime, side="right")  # (n_a, n_z)
        idx_hi = np.clip(idx_hi, 1, n_a - 1)
        idx_lo = idx_hi - 1

        # Bracket values
        a_lo = a_grid[idx_lo]  # (n_a, n_z)
        a_hi = a_grid[idx_hi]  # (n_a, n_z)

        # Interpolation weight for the upper bracket
        denom = a_hi - a_lo
        safe_denom = np.where(denom > _EPSILON, denom, 1.0)
        w_hi = np.where(denom > _EPSILON, (a_prime - a_lo) / safe_denom, 0.0)
        w_lo = 1.0 - w_hi  # (n_a, n_z)

        # Allocate mass: for each source (a_i, z_j), distribute to target
        # (a_lo/a_hi, z') weighted by lottery and transition probabilities.
        # mu_new(a_target, z') = sum_{a_i, z_j} w * Pi(z_j, z') * mu(a_i, z_j)
        mu_new = np.zeros((n_a, n_z), dtype=np.float64)

        # Vectorized mass allocation using np.add.at for each z_j
        for zj in range(n_z):
            # Mass from column zj: mu[:, zj] * transition to each z'
            mass_col = self.mu[:, zj]  # (n_a,)

            # For each target z'
            for zp in range(n_z):
                prob = transition_matrix[zj, zp]
                if prob < _EPSILON:
                    continue

                # Mass to allocate: mass * prob * weight
                mass_lo = w_lo[:, zj] * mass_col * prob  # (n_a,)
                mass_hi = w_hi[:, zj] * mass_col * prob  # (n_a,)

                # Accumulate at target grid indices
                np.add.at(mu_new[:, zp], idx_lo[:, zj], mass_lo)
                np.add.at(mu_new[:, zp], idx_hi[:, zj], mass_hi)

        # NaN/Inf check
        if not np.all(np.isfinite(mu_new)):
            log.warning("distribution.forward.non_finite_detected")
            mu_new = np.where(np.isfinite(mu_new), mu_new, 0.0)

        # Mass conservation check and renormalization (PROP-008)
        total_mass = mu_new.sum()
        if abs(total_mass - 1.0) > _MASS_CONSERVATION_TOL:
            log.warning(
                "distribution.forward.mass_leak",
                total_mass=total_mass,
                deviation=abs(total_mass - 1.0),
            )
            if total_mass > _EPSILON:
                mu_new /= total_mass

        self.mu = mu_new

    @classmethod
    def stationary(
        cls,
        a_grid: np.ndarray | list[float],
        z_grid: np.ndarray | list[float],
        policy_savings: np.ndarray,
        transition_matrix: np.ndarray,
        kfe_tolerance: float = _DEFAULT_KFE_TOLERANCE,
        kfe_max_iter: int = _DEFAULT_KFE_MAX_ITER,
    ) -> Distribution:
        """Compute stationary distribution by iterating KFE to convergence.

        Initializes with uniform distribution and iterates forward() until
        ||mu_{n+1} - mu_n||_1 < tolerance or max iterations reached.

        Args:
            a_grid: Asset grid.
            z_grid: Productivity grid.
            policy_savings: Savings policy a' = g(a, z), shape (n_a, n_z).
            transition_matrix: Markov transition matrix, shape (n_z, n_z).
            kfe_tolerance: Convergence tolerance (L1 norm).
            kfe_max_iter: Maximum number of iterations.

        Returns:
            Distribution object with converged stationary distribution.

        Implements REQ-205.
        """
        dist = cls(a_grid, z_grid, kfe_tolerance, kfe_max_iter)
        dist.initialize_uniform()

        policy_savings = np.asarray(policy_savings, dtype=np.float64)
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)

        for iteration in range(kfe_max_iter):
            mu_old = dist.mu.copy()
            dist.forward(policy_savings, transition_matrix)
            diff = np.sum(np.abs(dist.mu - mu_old))

            if diff < kfe_tolerance:
                log.debug(
                    "distribution.stationary.converged",
                    iterations=iteration + 1,
                    l1_diff=diff,
                )
                return dist

        log.warning(
            "distribution.stationary.max_iter_reached",
            max_iter=kfe_max_iter,
            l1_diff=float(np.sum(np.abs(dist.mu - mu_old))),
        )
        return dist

    def aggregate(self, policy_fn: np.ndarray) -> float:
        """Compute aggregate quantity from distribution and policy function.

        X = sum_{a,z} mu(a, z) * f(a, z)

        Args:
            policy_fn: Policy function values on grid, shape (n_a, n_z).

        Returns:
            Aggregate value.

        Implements REQ-203.
        """
        policy_fn = np.asarray(policy_fn, dtype=np.float64)
        return float(np.sum(self.mu * policy_fn))

    def gini(self) -> float:
        """Compute Gini coefficient of the wealth distribution.

        Flattens the 2D distribution into a 1D wealth distribution by
        marginalizing over productivity states, then computes the standard
        Gini coefficient.

        Returns:
            Gini coefficient in [0, 1].
        """
        # Marginal distribution over assets: sum over z
        mu_a = np.sum(self.mu, axis=1)  # (n_a,)

        # Remove zero-mass points for efficiency
        mask = mu_a > _EPSILON
        if not np.any(mask):
            return 0.0

        wealth = self.a_grid[mask]
        mass = mu_a[mask]

        # Sort by wealth (grid should already be sorted, but be safe)
        sort_idx = np.argsort(wealth)
        wealth = wealth[sort_idx]
        mass = mass[sort_idx]

        # Normalize mass to sum to 1
        total_mass = mass.sum()
        if total_mass < _EPSILON:
            return 0.0
        mass = mass / total_mass

        # Gini via weighted formula: G = 1 - 2 * integral of Lorenz curve
        # Equivalent to: G = (2 * sum_i p_i * F_i * w_i - sum_i p_i * w_i) / (sum_i p_i * w_i)
        # where F_i is the cumulative mass at or below i
        total_wealth = np.sum(mass * wealth)
        if total_wealth < _EPSILON:
            return 0.0

        cumulative_mass = np.cumsum(mass)
        # Lorenz: cumulative wealth share
        cumulative_wealth_share = np.cumsum(mass * wealth) / total_wealth

        # G = 1 - 2 * area under Lorenz curve
        # Area approximated by trapezoidal rule on (cumulative_mass, cumulative_wealth_share)
        # Prepend (0, 0) for the origin
        cm = np.concatenate(([0.0], cumulative_mass))
        cw = np.concatenate(([0.0], cumulative_wealth_share))
        # np.trapezoid is the renamed np.trapz in NumPy >= 2.0
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        area = _trapz(cw, cm)
        gini_val = 1.0 - 2.0 * area

        return float(np.clip(gini_val, 0.0, 1.0))

    def mean_wealth(self) -> float:
        """Compute mean wealth from the distribution.

        Returns:
            Mean wealth E[a] = sum_{a,z} mu(a,z) * a.
        """
        # Broadcast a_grid to (n_a, 1) for element-wise multiply
        a_col = self.a_grid.reshape(-1, 1)
        return float(np.sum(self.mu * a_col))

    def percentiles(self, quantiles: list[float] | None = None) -> list[float]:
        """Compute wealth percentiles from the distribution.

        Args:
            quantiles: List of quantile fractions, default [10, 25, 50, 75, 90]
                expressed as percentages.

        Returns:
            List of wealth values at each percentile.
        """
        if quantiles is None:
            quantiles = [10.0, 25.0, 50.0, 75.0, 90.0]

        # Marginal distribution over assets
        mu_a = np.sum(self.mu, axis=1)  # (n_a,)
        total_mass = mu_a.sum()
        if total_mass < _EPSILON:
            return [0.0] * len(quantiles)

        mu_a = mu_a / total_mass
        cumulative = np.cumsum(mu_a)

        results = []
        for q in quantiles:
            p = q / 100.0
            idx = np.searchsorted(cumulative, p, side="left")
            idx = min(idx, self.n_a - 1)
            results.append(float(self.a_grid[idx]))

        return results

    def top_share(self, fraction: float = 0.10) -> float:
        """Compute wealth share held by the top fraction of the distribution.

        Args:
            fraction: Top fraction (e.g. 0.10 for top 10%).

        Returns:
            Share of total wealth held by the top `fraction` of agents.
        """
        # Marginal distribution over assets
        mu_a = np.sum(self.mu, axis=1)  # (n_a,)
        total_mass = mu_a.sum()
        if total_mass < _EPSILON:
            return 0.0

        mu_a = mu_a / total_mass
        total_wealth = np.sum(mu_a * self.a_grid)
        if total_wealth < _EPSILON:
            return 0.0

        # Find cutoff: cumulative mass from bottom
        cumulative = np.cumsum(mu_a)
        cutoff = 1.0 - fraction
        idx = np.searchsorted(cumulative, cutoff, side="left")
        idx = min(idx, self.n_a - 1)

        # Wealth above cutoff
        top_wealth = np.sum(mu_a[idx:] * self.a_grid[idx:])
        return float(top_wealth / total_wealth)

    def sample_agents(
        self,
        n_agents: int,
        rng: SimulationRNG,
    ) -> list[tuple[float, float, int]]:
        """Sample n individual agents from the distribution via inverse CDF.

        Returns agent (wealth, productivity, productivity_index) tuples
        sampled from the joint distribution mu(a, z).

        Args:
            n_agents: Number of agents to sample.
            rng: Seeded RNG for determinism (PROP-001).

        Returns:
            List of (wealth, productivity, productivity_index) tuples.

        Implements REQ-204.
        """
        # Flatten mu to 1D for sampling
        flat_mu = self.mu.ravel()  # (n_a * n_z,)
        total_mass = flat_mu.sum()
        if total_mass < _EPSILON:
            # Fallback: uniform sampling
            log.warning("distribution.sample_agents.zero_mass_fallback")
            flat_mu = np.ones_like(flat_mu) / len(flat_mu)
        else:
            flat_mu = flat_mu / total_mass

        # Cumulative distribution for inverse CDF sampling
        cdf = np.cumsum(flat_mu)

        agents: list[tuple[float, float, int]] = []
        for _ in range(n_agents):
            u = rng.random()
            flat_idx = int(np.searchsorted(cdf, u, side="left"))
            flat_idx = min(flat_idx, len(cdf) - 1)

            # Convert flat index to (a_idx, z_idx)
            a_idx = flat_idx // self.n_z
            z_idx = flat_idx % self.n_z

            wealth = float(self.a_grid[a_idx])
            productivity = float(self.z_grid[z_idx])
            agents.append((wealth, productivity, z_idx))

        return agents

    def mass_total(self) -> float:
        """Return total probability mass (should be 1.0)."""
        return float(self.mu.sum())
