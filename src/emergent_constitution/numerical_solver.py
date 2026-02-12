"""Numerical household solver — VFI/EGM for optimal consumption-savings-labor.

Implements REQ-036 (numerical benchmark solver) and REQ-037 (benchmark mode).
Used as (a) fallback when LLM fails and (b) full benchmark mode.

Design reference: spec/design.md section 3.5.
"""

from __future__ import annotations

from collections.abc import Callable

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

        # Utility function with epsilon guard
        def utility(c: float, lei: float, g: float) -> float:
            c = max(c, 1e-10)
            lei = max(lei, 1e-10)
            g = max(g, 1e-10)
            return (c**alpha) * (lei**beta_param) * (g**gamma)

        # Build leisure grid
        leisure_grid = [i / self.n_leisure for i in range(self.n_leisure + 1)]

        # Initialize value function: V[a_idx][z_idx]
        # Start with utility of consuming everything
        v_old = [[0.0] * self.n_z for _ in range(self.n_a)]
        for ai in range(self.n_a):
            for zi in range(self.n_z):
                a = self.a_grid[ai]
                z_i = self.productivity_grid[zi]
                income = wage * z_i * 1.0  # full labor
                tax = tax_function(income)
                budget = (1.0 + interest_rate) * a + income - tax + transfer
                c = max(budget - self.a_min, 1e-10)
                v_old[ai][zi] = utility(c, 0.5, max(public_goods, 1e-10))

        # VFI iteration
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

                        # Optimize consumption via grid search
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

            # Copy v_new to v_old
            for ai in range(self.n_a):
                for zi in range(self.n_z):
                    v_old[ai][zi] = v_new[ai][zi]

            if max_diff < self.vfi_tolerance:
                log.debug(
                    "vfi.converged",
                    iterations=_vfi_iter + 1,
                    max_diff=max_diff,
                )
                break

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
            for zi in range(self.n_z):
                a = self.a_grid[ai]
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
                )
                break

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
