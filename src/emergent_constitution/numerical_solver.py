"""Numerical household solver — VFI/EGM for optimal consumption-savings-labor.

Implements REQ-036 (numerical benchmark solver) and REQ-037 (benchmark mode).
Used as (a) fallback when LLM fails and (b) full benchmark mode.

Design reference: spec/design.md section 3.5.
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

# Default asset grid parameters
_DEFAULT_A_GRID_SIZE = 100
_DEFAULT_A_MAX = 1000.0
_DEFAULT_VFI_MAX_ITER = 500
_DEFAULT_VFI_TOLERANCE = 1e-6
_DEFAULT_N_LEISURE_POINTS = 20


class NumericalSolver:
    """Computes approximately optimal household decisions via VFI/EGM.

    The solver discretizes the asset space and uses value function iteration
    on the Bellman equation:

        V(a, z) = max_{c, l} { u(c, l, G) + beta * E[V(a', z')] }
        s.t. a' = (1+r)*a + w*z*(1-l) - c - T(y) + Tr
             a' >= a_min, c >= 0, 0 <= l <= 1

    For the EGM variant, the Euler equation is inverted to find optimal
    consumption on an endogenous grid, then interpolated back.

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

        # Build asset grid (exponentially spaced for better resolution near a_min)
        self.n_a = _DEFAULT_A_GRID_SIZE
        self.a_grid = self._build_asset_grid(self.a_min, _DEFAULT_A_MAX, self.n_a)

        self.vfi_max_iter = _DEFAULT_VFI_MAX_ITER
        self.vfi_tolerance = _DEFAULT_VFI_TOLERANCE
        self.n_leisure = _DEFAULT_N_LEISURE_POINTS

        # NumPy arrays for vectorized VFI (rebuilt when grid sizes change)
        self._np_arrays_built = False
        self._build_numpy_arrays()

        # VFI policy cache keyed on market parameters
        self._vfi_cache: dict[
            tuple[float, ...],
            tuple[list[list[float]], list[list[float]]],
        ] = {}

        # Last computed VFI value function (REQ-110)
        self._last_value_func: list[list[float]] | None = None

    def get_value_function(self) -> tuple[list[list[float]] | None, list[float]]:
        """Return the last-computed VFI value function and asset grid.

        Used by the entrepreneurial solver to read V^W(a, z) for
        Bellman-based occupational choice (REQ-110).

        Returns:
            Tuple of (value_func, a_grid) where value_func is n_a x n_z
            or (None, a_grid) if VFI has not yet been solved.
        """
        return self._last_value_func, self.a_grid

    def _build_numpy_arrays(self) -> None:
        """Convert grids to NumPy arrays for vectorized computation."""
        self._a_grid_np = np.array(self.a_grid, dtype=np.float64)
        self._z_grid_np = np.array(self.productivity_grid, dtype=np.float64)
        self._trans_np = np.array(self.transition_matrix, dtype=np.float64)
        self._np_arrays_built = True

    @staticmethod
    def _build_asset_grid(a_min: float, a_max: float, n_points: int) -> list[float]:
        """Build exponentially-spaced asset grid.

        Finer spacing near a_min where policy function curvature is highest.
        """
        grid = []
        for i in range(n_points):
            frac = i / (n_points - 1) if n_points > 1 else 0.0
            # Exponential spacing: more points near a_min
            a = a_min + (a_max - a_min) * (frac**2)
            grid.append(a)
        return grid

    def solve_household(
        self,
        agent: HouseholdState,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> EconomicDecision:
        """Solve single-agent Bellman equation for optimal (c, l).

        Uses simplified VFI with discrete leisure grid. For each asset level
        and productivity state, searches over leisure choices and derives
        optimal consumption from the budget constraint.

        Args:
            agent: Household state (uses utility_params, productivity).
            wage: Market wage w_t.
            interest_rate: Market interest rate r_t.
            public_goods: Per-capita public goods G_t.
            tax_function: Maps income -> tax amount.
            transfer: Lump-sum transfer Tr_t.

        Returns:
            EconomicDecision with optimal consumption and leisure.

        Implements REQ-036.
        """
        alpha = agent.utility_params.alpha
        beta_param = agent.utility_params.beta
        gamma = agent.utility_params.gamma
        beta_discount = agent.utility_params.beta_discount
        z_idx = agent.productivity_index

        policy_c, policy_l = self.solve_vfi_shared(
            alpha=alpha,
            beta_param=beta_param,
            gamma=gamma,
            beta_discount=beta_discount,
            wage=wage,
            interest_rate=interest_rate,
            public_goods=public_goods,
            tax_function=tax_function,
            transfer=transfer,
        )

        # Interpolate policy for agent's actual state
        consumption = self._interpolate_policy(agent.wealth, z_idx, policy_c)
        leisure = self._interpolate_policy(agent.wealth, z_idx, policy_l)

        # Clamp to valid ranges
        consumption = max(consumption, 0.0)
        leisure = max(0.0, min(1.0, leisure))

        return EconomicDecision(consumption=consumption, leisure=leisure)

    def _interpolate_ev(
        self,
        a_prime: float,
        z_idx: int,
        value_func: list[list[float]],
    ) -> float:
        """Interpolate expected future value E[V(a', z')].

        Linear interpolation on asset grid, weighted by transition probs.
        """
        ev = 0.0
        for zj in range(self.n_z):
            prob = self.transition_matrix[z_idx][zj]
            if prob <= 0.0:
                continue
            v_at_a = self._linear_interp(a_prime, value_func, zj)
            ev += prob * v_at_a
        return ev

    def _linear_interp(
        self,
        a: float,
        func: list[list[float]],
        z_idx: int,
    ) -> float:
        """Linear interpolation on asset grid for a given z state."""
        if a <= self.a_grid[0]:
            return func[0][z_idx]
        if a >= self.a_grid[-1]:
            return func[-1][z_idx]

        # Binary search for bracket
        lo, hi = 0, self.n_a - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.a_grid[mid] <= a:
                lo = mid
            else:
                hi = mid

        a_lo = self.a_grid[lo]
        a_hi = self.a_grid[hi]
        if abs(a_hi - a_lo) < 1e-15:
            return func[lo][z_idx]

        weight = (a - a_lo) / (a_hi - a_lo)
        return func[lo][z_idx] * (1.0 - weight) + func[hi][z_idx] * weight

    def _interpolate_policy(
        self,
        a: float,
        z_idx: int,
        policy: list[list[float]],
    ) -> float:
        """Interpolate policy function value at (a, z_idx)."""
        return self._linear_interp(a, policy, z_idx)

    @staticmethod
    def _is_homogeneous(households: list[HouseholdState]) -> bool:
        """Check if all households share identical utility parameters.

        Auto-detects homogeneity without needing the config flag.
        """
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
        """Create a hashable cache key from VFI parameters.

        Uses rounded values to allow cache hits when prices are nearly identical.
        """
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

    def _solve_vfi_numpy(
        self,
        alpha: float,
        beta_param: float,
        gamma: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Solve VFI using NumPy vectorized operations.

        Replaces nested Python loops with array operations for ~100x speedup.
        Pre-computes budget arrays for each leisure level, then vectorizes
        consumption grid search and expected value computation.

        Returns:
            Tuple of (policy_c, policy_l) grids, each n_a x n_z.
        """
        # Ensure numpy arrays match current grid state
        if not self._np_arrays_built or len(self._a_grid_np) != self.n_a:
            self._build_numpy_arrays()

        n_a = self.n_a
        n_z = self.n_z
        a_min = self.a_min
        a_grid = self._a_grid_np
        z_grid = self._z_grid_np
        trans = self._trans_np

        g = max(public_goods, 1e-10)
        log_g = np.log(g) * gamma  # Pre-compute since G is constant

        # Leisure grid: shape (n_leisure+1,)
        n_lei = self.n_leisure
        leisure_grid = np.linspace(0.0, 1.0, n_lei + 1)

        # Consumption fraction grid: shape (n_c,)
        n_c = 11
        c_fracs = np.linspace(0.0, 1.0, n_c)

        # Initialize value function v_func[a, z] using log utility for warm start
        # shape: (n_a, n_z)
        v_func = np.zeros((n_a, n_z), dtype=np.float64)
        for ai in range(n_a):
            a_val = a_grid[ai]
            for zi in range(n_z):
                income = wage * z_grid[zi] * 1.0
                tax = tax_function(income)
                budget = (1.0 + interest_rate) * a_val + income - tax + transfer
                c_init = max(budget - a_min, 1e-10)
                lei_init = max(0.5, 1e-10)
                v_func[ai, zi] = (c_init**alpha) * (lei_init**beta_param) * (g**gamma)

        # Pre-compute z_grid broadcast: shape (1, n_z)
        z_row = z_grid.reshape(1, n_z)

        # Pre-compute a_grid column: shape (n_a, 1)
        a_col = a_grid.reshape(n_a, 1)

        policy_c_np = np.zeros((n_a, n_z), dtype=np.float64)
        policy_l_np = np.zeros((n_a, n_z), dtype=np.float64)

        for _vfi_iter in range(self.vfi_max_iter):
            v_new = np.full((n_a, n_z), -1e30, dtype=np.float64)

            # Expected value: ev[a', z] = sum_z' P(z, z') * v(a', z')
            # v_func @ trans.T shape (n_a, n_z)
            ev_grid = v_func @ trans.T  # shape (n_a, n_z)

            for li_idx in range(n_lei + 1):
                lei = leisure_grid[li_idx]
                labor = 1.0 - lei

                # Income for each (a, z): shape (n_a, n_z)
                income = wage * z_row * labor  # (1, n_z) broadcast

                # Vectorized tax: apply tax_function element-wise
                income_flat = income.ravel()
                tax_flat = np.array(
                    [tax_function(float(y)) for y in income_flat],
                    dtype=np.float64,
                )
                tax = tax_flat.reshape(1, n_z)

                # Budget: shape (n_a, n_z)
                budget = (1.0 + interest_rate) * a_col + income - tax + transfer

                # Maximum feasible consumption: shape (n_a, n_z)
                max_c = np.maximum(budget - a_min, 0.0)

                # Pre-compute leisure utility term (scalar for this lei)
                lei_val = max(lei, 1e-10)

                # For each consumption fraction, compute value
                best_val = np.full((n_a, n_z), -1e30, dtype=np.float64)
                best_c = np.full((n_a, n_z), 1e-10, dtype=np.float64)

                for ci in range(n_c):
                    c_try = np.maximum(max_c * c_fracs[ci], 1e-10)
                    a_prime = np.maximum(budget - c_try, a_min)

                    # Utility: u(c, l, G) = c^alpha * l^beta * G^gamma
                    log_u = alpha * np.log(c_try) + beta_param * np.log(lei_val) + log_g

                    # Interpolate EV at a_prime for each (a, z)
                    ev_interp = self._vectorized_interp_ev(
                        a_prime,
                        ev_grid,
                        a_grid,
                    )

                    val = np.exp(log_u) + beta_discount * ev_interp

                    # Update best
                    improve = val > best_val
                    best_val = np.where(improve, val, best_val)
                    best_c = np.where(improve, c_try, best_c)

                # Handle zero-budget case
                zero_budget = max_c <= 0.0
                if np.any(zero_budget):
                    c_zero = np.full((n_a, n_z), 1e-10)
                    log_u_zero = alpha * np.log(c_zero) + beta_param * np.log(lei_val) + log_g
                    ev_zero = ev_grid[0, :]
                    val_zero = np.exp(log_u_zero) + beta_discount * ev_zero
                    best_val = np.where(zero_budget, val_zero, best_val)
                    best_c = np.where(zero_budget, 1e-10, best_c)

                # Update policy for this leisure level where it improves
                improve_lei = best_val > v_new
                v_new = np.where(improve_lei, best_val, v_new)
                policy_c_np = np.where(improve_lei, best_c, policy_c_np)
                policy_l_np = np.where(improve_lei, lei, policy_l_np)

            # Check convergence
            max_diff = float(np.max(np.abs(v_new - v_func)))
            v_func = v_new.copy()

            if max_diff < self.vfi_tolerance:
                log.debug(
                    "vfi_shared.converged",
                    iterations=_vfi_iter + 1,
                    max_diff=max_diff,
                    method="numpy",
                )
                break

        # Store converged value function for occupational choice (REQ-110)
        self._last_value_func = v_func.tolist()

        # Convert back to list-of-lists for compatibility
        policy_c = policy_c_np.tolist()
        policy_l = policy_l_np.tolist()

        return policy_c, policy_l

    @staticmethod
    def _vectorized_interp_ev(
        a_prime: np.ndarray,
        ev_grid: np.ndarray,
        a_grid: np.ndarray,
    ) -> np.ndarray:
        """Vectorized linear interpolation of expected value on asset grid.

        Args:
            a_prime: Target asset values, shape (n_a, n_z).
            ev_grid: Expected value on grid points, shape (n_a, n_z).
            a_grid: Asset grid, shape (n_a,).

        Returns:
            Interpolated EV values, shape (n_a, n_z).
        """
        n_a = len(a_grid)

        # Clamp a_prime to grid range
        a_clamped = np.clip(a_prime, a_grid[0], a_grid[-1])

        # Find lower bracket indices using searchsorted
        # searchsorted returns index where a_clamped would be inserted
        idx_hi = np.searchsorted(a_grid, a_clamped, side="right")
        idx_hi = np.clip(idx_hi, 1, n_a - 1)
        idx_lo = idx_hi - 1

        # Get bracket values
        a_lo = a_grid[idx_lo]
        a_hi = a_grid[idx_hi]

        # Compute interpolation weight
        denom = a_hi - a_lo
        # Avoid division by zero for coincident grid points
        safe_denom = np.where(np.abs(denom) < 1e-15, 1.0, denom)
        weight = np.where(np.abs(denom) < 1e-15, 0.0, (a_clamped - a_lo) / safe_denom)

        # Gather EV values at bracket points
        # ev_grid shape: (n_a, n_z) — need to index per-column
        n_z = ev_grid.shape[1]
        z_indices = np.arange(n_z).reshape(1, n_z)  # broadcast helper

        # Advanced indexing: ev_lo[i,j] = ev_grid[idx_lo[i,j], j]
        ev_lo = ev_grid[idx_lo, z_indices]
        ev_hi = ev_grid[idx_hi, z_indices]

        return ev_lo * (1.0 - weight) + ev_hi * weight

    def _solve_vfi_python(
        self,
        alpha: float,
        beta_param: float,
        gamma: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Solve VFI using pure Python (fallback).

        Original implementation kept as fallback.

        Returns:
            Tuple of (policy_c, policy_l) grids, each n_a x n_z.
        """

        def utility(c: float, lei: float, g: float) -> float:
            c = max(c, 1e-10)
            lei = max(lei, 1e-10)
            g = max(g, 1e-10)
            return (c**alpha) * (lei**beta_param) * (g**gamma)

        leisure_grid = [i / self.n_leisure for i in range(self.n_leisure + 1)]

        # Initialize value function
        v_old = [[0.0] * self.n_z for _ in range(self.n_a)]
        for ai in range(self.n_a):
            a = self.a_grid[ai]
            for zi in range(self.n_z):
                z_i = self.productivity_grid[zi]
                income = wage * z_i * 1.0
                tax = tax_function(income)
                budget = (1.0 + interest_rate) * a + income - tax + transfer
                c = max(budget - self.a_min, 1e-10)
                v_old[ai][zi] = utility(c, 0.5, max(public_goods, 1e-10))

        v_new = [[0.0] * self.n_z for _ in range(self.n_a)]
        policy_c = [[0.0] * self.n_z for _ in range(self.n_a)]
        policy_l = [[0.0] * self.n_z for _ in range(self.n_a)]

        for _vfi_iter in range(self.vfi_max_iter):
            max_diff = 0.0

            for ai in range(self.n_a):
                a = self.a_grid[ai]
                for zi in range(self.n_z):
                    z_i = self.productivity_grid[zi]

                    best_val = -1e30
                    best_c = 0.0
                    best_l = 0.5

                    for lei in leisure_grid:
                        labor = 1.0 - lei
                        income = wage * z_i * labor
                        tax = tax_function(income)
                        budget = (1.0 + interest_rate) * a + income - tax + transfer

                        max_c = max(0.0, budget - self.a_min)
                        if max_c <= 0.0:
                            c = 1e-10
                            a_prime = self.a_min
                        else:
                            best_c_inner = 1e-10
                            best_val_inner = -1e30
                            for ci in range(11):
                                c_try = max_c * ci / 10.0
                                c_try = max(c_try, 1e-10)
                                a_prime = budget - c_try
                                a_prime = max(a_prime, self.a_min)

                                u_now = utility(c_try, lei, public_goods)
                                ev = self._interpolate_ev(a_prime, zi, v_old)

                                val = u_now + beta_discount * ev
                                if val > best_val_inner:
                                    best_val_inner = val
                                    best_c_inner = c_try

                            c = best_c_inner
                            val_total = best_val_inner

                        if max_c <= 0.0:
                            u_now = utility(c, lei, public_goods)
                            ev = self._interpolate_ev(self.a_min, zi, v_old)
                            val_total = u_now + beta_discount * ev

                        if val_total > best_val:
                            best_val = val_total
                            best_c = c
                            best_l = lei

                    v_new[ai][zi] = best_val
                    policy_c[ai][zi] = best_c
                    policy_l[ai][zi] = best_l

                    diff = abs(v_new[ai][zi] - v_old[ai][zi])
                    if diff > max_diff:
                        max_diff = diff

            for ai in range(self.n_a):
                for zi in range(self.n_z):
                    v_old[ai][zi] = v_new[ai][zi]

            if max_diff < self.vfi_tolerance:
                log.debug(
                    "vfi_shared.converged",
                    iterations=_vfi_iter + 1,
                    max_diff=max_diff,
                    method="python",
                )
                break

        # Store converged value function for occupational choice (REQ-110)
        self._last_value_func = [row[:] for row in v_new]

        return policy_c, policy_l

    def solve_vfi_shared(
        self,
        alpha: float,
        beta_param: float,
        gamma: float,
        beta_discount: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Solve VFI once for shared preferences, returning policy grids.

        Identical to the VFI loop in solve_household but parameterized on
        explicit utility weights instead of reading from an agent. Returns the
        full (n_a x n_z) policy grids so callers can interpolate per agent.

        Uses NumPy-vectorized implementation for speed. Caches results keyed
        on market parameters to avoid redundant computation when prices are stable.

        Returns:
            Tuple of (policy_c, policy_l) grids, each n_a x n_z.
        """
        # Estimate tax rate for cache key (sample at income=1.0)
        tax_rate_proxy = tax_function(1.0) if wage > 0 else 0.0

        cache_key = self._make_cache_key(
            alpha,
            beta_param,
            gamma,
            beta_discount,
            wage,
            interest_rate,
            public_goods,
            transfer,
            tax_rate_proxy,
        )

        # Check cache
        if cache_key in self._vfi_cache:
            log.debug("vfi_shared.cache_hit", key_hash=hash(cache_key))
            return self._vfi_cache[cache_key]

        # Solve using NumPy vectorized implementation
        policy_c, policy_l = self._solve_vfi_numpy(
            alpha,
            beta_param,
            gamma,
            beta_discount,
            wage,
            interest_rate,
            public_goods,
            tax_function,
            transfer,
        )

        # Store in cache
        self._vfi_cache[cache_key] = (policy_c, policy_l)
        log.debug(
            "vfi_shared.cache_miss",
            cache_size=len(self._vfi_cache),
            key_hash=hash(cache_key),
        )

        return policy_c, policy_l

    def solve_all(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution_tax_rate: float = 0.0,
        public_goods: float = 0.0,
        transfer: float = 0.0,
    ) -> dict[str, EconomicDecision]:
        """Solve for all households using VFI.

        When all households share identical utility parameters (auto-detected),
        VFI is solved once and policies are interpolated per agent (~Nx speedup).
        Otherwise falls back to per-agent solve_household.

        Args:
            households: List of household states.
            market: Current market state with prices.
            constitution_tax_rate: Flat tax rate from constitution.
            public_goods: Per-capita public goods G_t.
            transfer: Lump-sum transfer per agent.

        Returns:
            Dict mapping agent_id -> EconomicDecision.

        Implements REQ-037.
        """

        def tax_function(income: float) -> float:
            return income * constitution_tax_rate

        decisions: dict[str, EconomicDecision] = {}

        if self._is_homogeneous(households) and households:
            ref = households[0].utility_params
            log.debug(
                "solve_all.fast_path",
                n_agents=len(households),
                alpha=ref.alpha,
                beta=ref.beta,
                gamma=ref.gamma,
            )
            policy_c, policy_l = self.solve_vfi_shared(
                alpha=ref.alpha,
                beta_param=ref.beta,
                gamma=ref.gamma,
                beta_discount=ref.beta_discount,
                wage=market.wage,
                interest_rate=market.interest_rate,
                public_goods=public_goods,
                tax_function=tax_function,
                transfer=transfer,
            )
            for h in households:
                consumption = self._interpolate_policy(h.wealth, h.productivity_index, policy_c)
                leisure = self._interpolate_policy(h.wealth, h.productivity_index, policy_l)
                consumption = max(consumption, 0.0)
                leisure = max(0.0, min(1.0, leisure))
                decisions[h.id] = EconomicDecision(consumption=consumption, leisure=leisure)
        else:
            for h in households:
                decision = self.solve_household(
                    agent=h,
                    wage=market.wage,
                    interest_rate=market.interest_rate,
                    public_goods=public_goods,
                    tax_function=tax_function,
                    transfer=transfer,
                )
                decisions[h.id] = decision

        return decisions
