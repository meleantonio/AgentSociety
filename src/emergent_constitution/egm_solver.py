"""Endogenous Grid Method (EGM) solver for household consumption-savings-labor.

Implements Carroll (2006) EGM for the Cobb-Douglas utility function
u = c^alpha * l^beta * G^gamma, with Fella (2014) upper envelope for
non-monotonicity near kinks.

Traceability: REQ-115, REQ-116, REQ-117, REQ-118, REQ-119, REQ-120.
Design reference: spec/design.md section 1.4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.decisions import EconomicDecision
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()

# Default grid and convergence parameters
_DEFAULT_A_GRID_SIZE = 200
_DEFAULT_A_MAX = 1000.0
_DEFAULT_EGM_MAX_ITER = 500
_DEFAULT_EGM_TOLERANCE = 1e-8
_DEFAULT_EULER_RESIDUAL_TOL = 1e-6

# Numerical safety floor
_EPSILON = 1e-10


class EGMSolver:
    """Endogenous Grid Method for Cobb-Douglas utility.

    Solves the household consumption-savings-labor problem:
        V(a, z) = max_{c, l} { u(c, l, G) + beta * E[V(a', z')] }
        s.t. a' = (1+r)*a + w*z*(1-l) - c - T(y) + Tr
             a' >= a_min, c >= 0, 0 <= l <= 1

    The EGM inverts the Euler equation on an exogenous savings grid a',
    backs out endogenous current assets a, then interpolates back to the
    exogenous grid. This avoids the inner maximization loop of VFI.

    Args:
        config: Simulation configuration with productivity grid, shock params.
        productivity_grid: Discrete productivity levels from Rouwenhorst.
        transition_matrix: Markov transition matrix for productivity.
    """

    def __init__(
        self,
        config: SimulationConfigV2,
        productivity_grid: list[float],
        transition_matrix: list[list[float]],
    ) -> None:
        self.config = config
        self.productivity_grid = productivity_grid
        self.transition_matrix = transition_matrix
        self.n_z = len(productivity_grid)
        self.a_min = config.a_min

        # Build savings grid (exogenous a' grid), REQ-118
        self.n_a = _DEFAULT_A_GRID_SIZE
        self.a_grid = self._build_exponential_grid(self.a_min, _DEFAULT_A_MAX, self.n_a)

        self.egm_max_iter = _DEFAULT_EGM_MAX_ITER
        self.egm_tolerance = _DEFAULT_EGM_TOLERANCE
        self.euler_residual_tol = _DEFAULT_EULER_RESIDUAL_TOL

        # NumPy arrays
        self._a_grid_np = np.array(self.a_grid, dtype=np.float64)
        self._z_grid_np = np.array(self.productivity_grid, dtype=np.float64)
        self._trans_np = np.array(self.transition_matrix, dtype=np.float64)

        # Policy cache keyed on market parameters
        self._policy_cache: dict[
            tuple[float, ...],
            tuple[np.ndarray, np.ndarray],
        ] = {}

        # Last-computed value function (for entrepreneurial solver compatibility)
        self._last_value_func: list[list[float]] | None = None

    def get_value_function(self) -> tuple[list[list[float]] | None, list[float]]:
        """Return the last-computed value function and asset grid.

        Used by the entrepreneurial solver to read V^W(a, z) for
        Bellman-based occupational choice (REQ-110).

        Returns:
            Tuple of (value_func, a_grid) where value_func is n_a x n_z
            or (None, a_grid) if solver has not yet been called.
        """
        return self._last_value_func, self.a_grid

    @staticmethod
    def _build_exponential_grid(a_min: float, a_max: float, n_points: int) -> list[float]:
        """Build exponentially-spaced asset grid (REQ-118).

        Finer spacing near a_min where policy function curvature is highest.
        """
        grid = []
        for i in range(n_points):
            frac = i / (n_points - 1) if n_points > 1 else 0.0
            a = a_min + (a_max - a_min) * (frac**2)
            grid.append(a)
        return grid

    @staticmethod
    def _is_homogeneous(households: list[HouseholdState]) -> bool:
        """Check if all households share identical utility parameters."""
        if not households:
            return True
        ref = households[0].utility_params
        return all(
            h.utility_params.alpha == ref.alpha
            and h.utility_params.beta == ref.beta
            and h.utility_params.gamma == ref.gamma
            and h.utility_params.beta_discount == ref.beta_discount
            for h in households[1:]
        )

    def _make_cache_key(
        self,
        alpha: float,
        beta_param: float,
        gamma: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        transfer: float,
        tax_rate_proxy: float,
    ) -> tuple[float, ...]:
        """Create a hashable cache key from solver parameters."""
        return (
            round(alpha, 6),
            round(beta_param, 6),
            round(gamma, 6),
            round(beta_discount, 6),
            round(wage, 4),
            round(interest_rate, 4),
            round(public_goods, 4),
            round(transfer, 4),
            round(tax_rate_proxy, 4),
        )

    def _compute_leisure_from_foc(
        self,
        c: np.ndarray,
        alpha_u: float,
        beta_u: float,
        wage: float,
        z: np.ndarray,
    ) -> np.ndarray:
        """Intratemporal FOC for leisure (REQ-117).

        l* = (beta_u * c) / (alpha_u * w * z), clamped to [0, 1].

        Args:
            c: Consumption array.
            alpha_u: Consumption weight in utility.
            beta_u: Leisure weight in utility.
            wage: Market wage.
            z: Productivity array (broadcastable with c).

        Returns:
            Optimal leisure, clamped to [0, 1].
        """
        denom = np.maximum(alpha_u * wage * z, _EPSILON)
        lei = (beta_u * c) / denom
        return np.clip(lei, 0.0, 1.0)

    def _marginal_utility_c(
        self,
        c: np.ndarray,
        lei: np.ndarray,
        g: float,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
    ) -> np.ndarray:
        """Marginal utility of consumption: u_c = alpha * c^(alpha-1) * l^beta * G^gamma.

        Args:
            c: Consumption array.
            lei: Leisure array.
            g: Public goods level.
            alpha_u: Consumption weight.
            beta_u: Leisure weight.
            gamma_u: Public goods weight.

        Returns:
            Marginal utility of consumption.
        """
        c_safe = np.maximum(c, _EPSILON)
        lei_safe = np.maximum(lei, _EPSILON)
        g_safe = max(g, _EPSILON)
        return alpha_u * (c_safe ** (alpha_u - 1.0)) * (lei_safe**beta_u) * (g_safe**gamma_u)

    def _invert_euler_equation(
        self,
        rhs: np.ndarray,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
        wage: float,
        z_row: np.ndarray,
        g: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert the Euler equation to find (c, l) from RHS values (REQ-116).

        The Euler equation is:
            alpha * c^(alpha-1) * l^beta * G^gamma = RHS

        With the intratemporal FOC l = (beta_u * c) / (alpha_u * w * z),
        substituting gives:
            alpha * c^(alpha-1) * ((beta_u * c) / (alpha_u * w * z))^beta * G^gamma = RHS

        Simplifying:
            alpha * (beta_u / (alpha_u * w * z))^beta * G^gamma * c^(alpha + beta - 1) = RHS
            c^(alpha + beta - 1) = RHS / (alpha * (beta_u / (alpha_u * w * z))^beta * G^gamma)
            c = (RHS / coeff)^(1 / (alpha + beta - 1))

        Args:
            rhs: Right-hand side of the Euler equation, shape (n_a, n_z).
            alpha_u: Consumption utility weight.
            beta_u: Leisure utility weight.
            gamma_u: Public goods utility weight.
            wage: Market wage.
            z_row: Productivity grid, shape (1, n_z).
            g: Public goods level.

        Returns:
            Tuple of (consumption, leisure) arrays, each shape (n_a, n_z).
        """
        g_safe = max(g, _EPSILON)
        wz = np.maximum(wage * z_row, _EPSILON)

        # Coefficient: alpha * (beta_u / (alpha_u * w * z))^beta * G^gamma
        lei_ratio = np.maximum(beta_u / (alpha_u * wz), _EPSILON)
        coeff = alpha_u * (lei_ratio**beta_u) * (g_safe**gamma_u)
        coeff = np.maximum(coeff, _EPSILON)

        # Exponent: 1 / (alpha + beta - 1)
        exponent_denom = alpha_u + beta_u - 1.0
        if abs(exponent_denom) < _EPSILON:
            # Degenerate case: log utility. Use bisection fallback.
            c = np.maximum(rhs / np.maximum(coeff, _EPSILON), _EPSILON)
        else:
            exponent = 1.0 / exponent_denom
            base = rhs / coeff
            # Handle negative base (can happen with very small RHS)
            base = np.maximum(base, _EPSILON)
            c = np.power(base, exponent)

        c = np.maximum(c, _EPSILON)

        # Ensure finite values
        c = np.where(np.isfinite(c), c, _EPSILON)

        # Leisure from intratemporal FOC (REQ-117)
        lei = self._compute_leisure_from_foc(c, alpha_u, beta_u, wage, z_row)

        return c, lei

    def _upper_envelope(
        self,
        a_endo: np.ndarray,
        c_endo: np.ndarray,
        lei_endo: np.ndarray,
        a_exo: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fella (2014) upper envelope to handle non-monotonicity near kinks.

        When the endogenous grid is non-monotone (due to kinks in the
        budget constraint), we take the upper envelope of the consumption
        function by keeping only the segment that gives the highest value.

        For each productivity state (column), this:
        1. Checks for monotonicity violations in the endogenous grid
        2. Where non-monotone, resolves by keeping the higher consumption

        Args:
            a_endo: Endogenous asset grid, shape (n_a, n_z).
            c_endo: Consumption on endogenous grid, shape (n_a, n_z).
            lei_endo: Leisure on endogenous grid, shape (n_a, n_z).
            a_exo: Exogenous asset grid, shape (n_a,).

        Returns:
            Tuple of (consumption, leisure) interpolated to exogenous grid,
            each shape (n_a, n_z).
        """
        n_a_exo = len(a_exo)
        n_z = a_endo.shape[1]
        c_out = np.full((n_a_exo, n_z), _EPSILON)
        lei_out = np.full((n_a_exo, n_z), 0.5)

        for zi in range(n_z):
            a_col = a_endo[:, zi]
            c_col = c_endo[:, zi]
            l_col = lei_endo[:, zi]

            # Sort by endogenous assets to ensure monotonicity
            sort_idx = np.argsort(a_col)
            a_sorted = a_col[sort_idx]
            c_sorted = c_col[sort_idx]
            l_sorted = l_col[sort_idx]

            # Remove duplicate / non-monotone points (upper envelope)
            # Keep only points where assets are strictly increasing
            mask = np.ones(len(a_sorted), dtype=bool)
            for i in range(1, len(a_sorted)):
                if a_sorted[i] <= a_sorted[i - 1]:
                    # Non-monotonicity: keep the one with higher consumption
                    if c_sorted[i] > c_sorted[i - 1]:
                        mask[i - 1] = False
                    else:
                        mask[i] = False

            a_clean = a_sorted[mask]
            c_clean = c_sorted[mask]
            l_clean = l_sorted[mask]

            if len(a_clean) < 2:
                continue

            # Monotone piecewise linear interpolation to exogenous grid
            c_out[:, zi] = np.interp(a_exo, a_clean, c_clean)
            lei_out[:, zi] = np.interp(a_exo, a_clean, l_clean)

        # Clamp outputs
        c_out = np.maximum(c_out, _EPSILON)
        lei_out = np.clip(lei_out, 0.0, 1.0)

        return c_out, lei_out

    def solve_egm(
        self,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for policy functions c(a, z) and l(a, z) via EGM.

        Implements the 5-step EGM algorithm from spec/design.md section 1.4.

        Args:
            alpha_u: Consumption utility weight.
            beta_u: Leisure utility weight.
            gamma_u: Public goods utility weight.
            beta_discount: Intertemporal discount factor.
            wage: Market wage.
            interest_rate: Market interest rate.
            public_goods: Per-capita public goods.
            tax_function: Maps income -> tax amount.
            transfer: Lump-sum transfer.

        Returns:
            Tuple of (c_policy, l_policy) arrays, each shape (n_a, n_z).
        """
        n_a = self.n_a
        n_z = self.n_z
        a_grid = self._a_grid_np
        z_grid = self._z_grid_np
        trans = self._trans_np
        g = max(public_goods, _EPSILON)
        gross_r = 1.0 + interest_rate  # Gross return

        # Shape helpers
        z_row = z_grid.reshape(1, n_z)  # (1, n_z)

        # Initialize policy: consume a fraction of resources
        c_policy = np.full((n_a, n_z), _EPSILON)
        lei_policy = np.full((n_a, n_z), 0.5)

        for ai in range(n_a):
            for zi in range(n_z):
                income = wage * z_grid[zi] * 0.5  # Start with 0.5 labor
                tax = tax_function(income)
                resources = gross_r * a_grid[ai] + income - tax + transfer
                c_policy[ai, zi] = max(resources * 0.5, _EPSILON)
                lei_policy[ai, zi] = 0.5

        # Recompute leisure consistently with FOC
        lei_policy = self._compute_leisure_from_foc(c_policy, alpha_u, beta_u, wage, z_row)

        prev_c_policy = c_policy.copy()

        for iteration in range(self.egm_max_iter):
            # Step 1: Expected marginal utility of savings (RHS of Euler eq)
            # For each (a'_j, z_s): RHS = beta * (1+r) * sum_{s'} Pi(s,s') * u_c(...)
            mu_c = self._marginal_utility_c(
                c_policy, lei_policy, g, alpha_u, beta_u, gamma_u
            )
            # Expected marginal utility: shape (n_a, n_z)
            # For each z_s, sum over z_s': Pi(s, s') * mu_c(a', s')
            e_mu_c = mu_c @ trans.T  # (n_a, n_z)
            rhs = beta_discount * gross_r * e_mu_c  # (n_a, n_z)

            # Step 2: Invert Euler equation to get (c, l)
            c_endo, lei_endo = self._invert_euler_equation(
                rhs, alpha_u, beta_u, gamma_u, wage, z_row, g
            )

            # Step 3: Back out current assets from budget constraint (endogenous grid)
            # a = (a' + c + T(w*z*(1-l)) - Tr - w*z*(1-l)) / (1+r)
            labor = 1.0 - lei_endo
            income = wage * z_row * labor  # (n_a, n_z)

            # Vectorized tax computation
            income_flat = income.ravel()
            tax_flat = np.array(
                [tax_function(float(y)) for y in income_flat],
                dtype=np.float64,
            )
            tax = tax_flat.reshape(n_a, n_z)

            # a'_j grid broadcast to (n_a, n_z)
            a_prime = a_grid.reshape(n_a, 1) * np.ones((1, n_z))

            a_endo = (a_prime + c_endo + tax - transfer - income) / np.maximum(gross_r, _EPSILON)

            # Step 4: Upper envelope and interpolation back to exogenous grid
            c_new, lei_new = self._upper_envelope(a_endo, c_endo, lei_endo, a_grid)

            # Step 5: Handle constrained region
            # For a below lowest endogenous grid point, set a' = a_min
            for zi in range(n_z):
                a_endo_min = np.min(a_endo[:, zi])
                for ai in range(n_a):
                    if a_grid[ai] < a_endo_min:
                        # Constrained: a' = a_min, consume everything feasible
                        labor_c = 1.0 - lei_new[ai, zi]
                        income_c = wage * z_grid[zi] * labor_c
                        tax_c = tax_function(float(income_c))
                        resources = gross_r * a_grid[ai] + income_c - tax_c + transfer
                        c_constrained = max(resources - self.a_min, _EPSILON)
                        c_new[ai, zi] = c_constrained
                        # Recompute leisure at constrained consumption
                        lei_c = (beta_u * c_constrained) / max(
                            alpha_u * wage * z_grid[zi], _EPSILON
                        )
                        lei_new[ai, zi] = max(0.0, min(1.0, lei_c))

            # Ensure validity
            c_new = np.maximum(c_new, _EPSILON)
            c_new = np.where(np.isfinite(c_new), c_new, _EPSILON)
            lei_new = np.clip(lei_new, 0.0, 1.0)
            lei_new = np.where(np.isfinite(lei_new), lei_new, 0.5)

            # Check convergence via sup-norm on consumption policy
            max_diff = float(np.max(np.abs(c_new - prev_c_policy)))

            c_policy = c_new.copy()
            lei_policy = lei_new.copy()
            prev_c_policy = c_policy.copy()

            if max_diff < self.egm_tolerance:
                log.debug(
                    "egm.converged",
                    iterations=iteration + 1,
                    max_diff=max_diff,
                )
                break

        return c_policy, lei_policy

    def euler_residual(
        self,
        c_policy: np.ndarray,
        lei_policy: np.ndarray,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> float:
        """Compute max Euler equation residual for convergence check (REQ-119).

        Residual = max |1 - c_euler / c_policy| across the grid.

        Args:
            c_policy: Consumption policy, shape (n_a, n_z).
            lei_policy: Leisure policy, shape (n_a, n_z).
            alpha_u: Consumption utility weight.
            beta_u: Leisure utility weight.
            gamma_u: Public goods utility weight.
            beta_discount: Discount factor.
            wage: Market wage.
            interest_rate: Interest rate.
            public_goods: Public goods level.
            tax_function: Tax function.
            transfer: Lump-sum transfer.

        Returns:
            Maximum Euler equation residual.
        """
        g = max(public_goods, _EPSILON)
        gross_r = 1.0 + interest_rate
        z_row = self._z_grid_np.reshape(1, self.n_z)

        # Compute a' from current policy
        labor = 1.0 - lei_policy
        income = wage * z_row * labor
        income_flat = income.ravel()
        tax_flat = np.array(
            [tax_function(float(y)) for y in income_flat],
            dtype=np.float64,
        )
        tax = tax_flat.reshape(self.n_a, self.n_z)

        a_col = self._a_grid_np.reshape(self.n_a, 1)
        a_prime = gross_r * a_col + income - tax + transfer - c_policy
        a_prime = np.maximum(a_prime, self.a_min)

        # Interpolate next-period consumption and leisure at a'
        c_next = np.full_like(c_policy, _EPSILON)
        lei_next = np.full_like(lei_policy, 0.5)
        for zi in range(self.n_z):
            c_next[:, zi] = np.interp(
                a_prime[:, zi], self._a_grid_np, c_policy[:, zi]
            )
            lei_next[:, zi] = np.interp(
                a_prime[:, zi], self._a_grid_np, lei_policy[:, zi]
            )

        c_next = np.maximum(c_next, _EPSILON)
        lei_next = np.clip(lei_next, 0.0, 1.0)

        # RHS of Euler equation
        mu_c_next = self._marginal_utility_c(
            c_next, lei_next, g, alpha_u, beta_u, gamma_u
        )
        e_mu_c = mu_c_next @ self._trans_np.T
        rhs = beta_discount * gross_r * e_mu_c

        # Invert the Euler equation to get implied c_euler from RHS
        z_row = self._z_grid_np.reshape(1, self.n_z)
        c_euler, _ = self._invert_euler_equation(
            rhs, alpha_u, beta_u, gamma_u, wage, z_row, g
        )
        c_euler = np.maximum(c_euler, _EPSILON)

        # Residual: |1 - c_euler / c_policy|
        c_ratio = c_euler / np.maximum(c_policy, _EPSILON)
        residual = np.abs(1.0 - c_ratio)

        # At the constrained region, the Euler equation holds with inequality
        # (MU_today > beta*(1+r)*E[MU_tomorrow]) so we skip those points.
        # Constrained agents are those with a' near a_min.
        constraint_threshold = self.a_min + (self._a_grid_np[-1] - self.a_min) * 0.01
        unconstrained = a_prime > constraint_threshold
        if np.any(unconstrained):
            return float(np.max(residual[unconstrained]))
        return float(np.max(residual))

    def _compute_value_function(
        self,
        c_policy: np.ndarray,
        lei_policy: np.ndarray,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
        n_iter: int = 200,
    ) -> list[list[float]]:
        """Compute value function from converged policy functions.

        Iterates the Bellman operator V = u(c,l,G) + beta*E[V(a',z')]
        using the fixed policy until convergence.

        Args:
            c_policy: Consumption policy, shape (n_a, n_z).
            lei_policy: Leisure policy, shape (n_a, n_z).
            alpha_u, beta_u, gamma_u: Utility weights.
            beta_discount: Discount factor.
            wage, interest_rate: Market prices.
            public_goods: Public goods level.
            tax_function: Tax function.
            transfer: Lump-sum transfer.
            n_iter: Number of Bellman iterations.

        Returns:
            Value function as n_a x n_z list of lists.
        """
        g = max(public_goods, _EPSILON)
        gross_r = 1.0 + interest_rate
        z_row = self._z_grid_np.reshape(1, self.n_z)
        a_col = self._a_grid_np.reshape(self.n_a, 1)

        # Compute utility at each (a, z)
        c_safe = np.maximum(c_policy, _EPSILON)
        lei_safe = np.maximum(lei_policy, _EPSILON)
        u_grid = (c_safe**alpha_u) * (lei_safe**beta_u) * (g**gamma_u)

        # Compute a' from policy
        labor = 1.0 - lei_policy
        income = wage * z_row * labor
        income_flat = income.ravel()
        tax_flat = np.array(
            [tax_function(float(y)) for y in income_flat], dtype=np.float64
        )
        tax = tax_flat.reshape(self.n_a, self.n_z)
        a_prime = gross_r * a_col + income - tax + transfer - c_policy
        a_prime = np.maximum(a_prime, self.a_min)

        # Initialize V with current-period utility
        v_func = u_grid / max(1.0 - beta_discount, _EPSILON)

        for _ in range(n_iter):
            # Interpolate V at a' for each z
            ev = np.zeros_like(v_func)
            for zi in range(self.n_z):
                # Expected value over z' transitions
                for zj in range(self.n_z):
                    ev[:, zi] += self._trans_np[zi, zj] * np.interp(
                        a_prime[:, zi], self._a_grid_np, v_func[:, zj]
                    )

            v_new = u_grid + beta_discount * ev
            if float(np.max(np.abs(v_new - v_func))) < 1e-8:
                break
            v_func = v_new

        return v_func.tolist()

    def solve_egm_cached(
        self,
        alpha_u: float,
        beta_u: float,
        gamma_u: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve EGM with caching on market parameters.

        Returns cached results when prices haven't changed.
        """
        tax_rate_proxy = tax_function(1.0) if wage > 0 else 0.0
        cache_key = self._make_cache_key(
            alpha_u, beta_u, gamma_u, beta_discount,
            wage, interest_rate, public_goods, transfer, tax_rate_proxy,
        )

        if cache_key in self._policy_cache:
            log.debug("egm.cache_hit", key_hash=hash(cache_key))
            return self._policy_cache[cache_key]

        c_policy, lei_policy = self.solve_egm(
            alpha_u, beta_u, gamma_u, beta_discount,
            wage, interest_rate, public_goods, tax_function, transfer,
        )

        self._policy_cache[cache_key] = (c_policy, lei_policy)

        # Compute value function from converged policy for entrepreneurial solver
        self._last_value_func = self._compute_value_function(
            c_policy, lei_policy, alpha_u, beta_u, gamma_u,
            beta_discount, wage, interest_rate, public_goods, tax_function, transfer,
        )

        log.debug(
            "egm.cache_miss",
            cache_size=len(self._policy_cache),
            key_hash=hash(cache_key),
        )

        return c_policy, lei_policy

    def _interpolate_policy(
        self,
        a: float,
        z_idx: int,
        policy: np.ndarray,
    ) -> float:
        """Interpolate policy function value at (a, z_idx).

        Linear interpolation on the asset grid.
        """
        return float(np.interp(a, self._a_grid_np, policy[:, z_idx]))

    def solve_all(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution_tax_rate: float = 0.0,
        public_goods: float = 0.0,
        transfer: float = 0.0,
    ) -> dict[str, EconomicDecision]:
        """Solve for all households using EGM.

        Interface-compatible with NumericalSolver.solve_all().

        When all households share identical utility parameters (auto-detected),
        EGM is solved once and policies are interpolated per agent.

        Args:
            households: List of household states.
            market: Current market state with prices.
            constitution_tax_rate: Flat tax rate from constitution.
            public_goods: Per-capita public goods G_t.
            transfer: Lump-sum transfer per agent.

        Returns:
            Dict mapping agent_id -> EconomicDecision.

        Implements REQ-115, REQ-120.
        """

        def tax_function(income: float) -> float:
            return income * constitution_tax_rate

        decisions: dict[str, EconomicDecision] = {}

        if self._is_homogeneous(households) and households:
            ref = households[0].utility_params
            log.debug(
                "egm_solve_all.fast_path",
                n_agents=len(households),
                alpha=ref.alpha,
                beta=ref.beta,
                gamma=ref.gamma,
            )
            c_policy, lei_policy = self.solve_egm_cached(
                alpha_u=ref.alpha,
                beta_u=ref.beta,
                gamma_u=ref.gamma,
                beta_discount=ref.beta_discount,
                wage=market.wage,
                interest_rate=market.interest_rate,
                public_goods=public_goods,
                tax_function=tax_function,
                transfer=transfer,
            )
            for h in households:
                consumption = self._interpolate_policy(h.wealth, h.productivity_index, c_policy)
                leisure = self._interpolate_policy(h.wealth, h.productivity_index, lei_policy)
                consumption = max(consumption, 0.0)
                leisure = max(0.0, min(1.0, leisure))
                decisions[h.id] = EconomicDecision(consumption=consumption, leisure=leisure)
        else:
            for h in households:
                c_policy, lei_policy = self.solve_egm_cached(
                    alpha_u=h.utility_params.alpha,
                    beta_u=h.utility_params.beta,
                    gamma_u=h.utility_params.gamma,
                    beta_discount=h.utility_params.beta_discount,
                    wage=market.wage,
                    interest_rate=market.interest_rate,
                    public_goods=public_goods,
                    tax_function=tax_function,
                    transfer=transfer,
                )
                consumption = self._interpolate_policy(
                    h.wealth, h.productivity_index, c_policy
                )
                leisure = self._interpolate_policy(
                    h.wealth, h.productivity_index, lei_policy
                )
                consumption = max(consumption, 0.0)
                leisure = max(0.0, min(1.0, leisure))
                decisions[h.id] = EconomicDecision(consumption=consumption, leisure=leisure)

        return decisions
