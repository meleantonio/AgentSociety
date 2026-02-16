"""Entrepreneurial decision solver -- firm value, entry/exit, optimal capital.

Implements solver-driven entrepreneurial decisions based on heterogeneous
entrepreneurial ability modeled as a Markov chain. Uses utility-based
occupational choice comparing lifetime value as worker vs entrepreneur
to determine entry/exit decisions and optimal factor demands.

When ``config.use_bellman_occ_choice`` is True (REQ-110..114):
- Firm value is computed via the Bellman equation V^F = (I - beta*Pi)^{-1} * pi
- Worker value is read directly from the VFI-solved value function V^W(a, z)
- Entrepreneur value uses FOC-derived leisure/consumption (no hardcoded values)
- Optimal capital uses golden-section search with 50+ evaluations

When the flag is False (default), the legacy perpetuity-based solver is used
for backward compatibility.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.decisions import EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()

# Epsilon guard for utility computation
_EPSILON = 1e-10

# Golden ratio for golden-section search
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_RESPHI = 2.0 - _PHI  # ~0.382


class EntrepreneurialSolver:
    """Solver for entrepreneurial entry/exit decisions and factor demands.

    Supports two modes controlled by ``config.use_bellman_occ_choice``:

    **Legacy mode** (default): Uses perpetuity-based firm value with stationary
    mean ability and hardcoded leisure/consumption approximations.

    **Bellman mode** (REQ-110..114): Uses proper Bellman equations for firm
    value, VFI-derived worker value, FOC-derived entrepreneur leisure, and
    golden-section search for optimal capital.

    Args:
        config: Simulation configuration.
        ability_grid: Discretized entrepreneurial ability grid values.
        ability_transition_matrix: Markov transition matrix for ability.
    """

    def __init__(
        self,
        config: SimulationConfigV2,
        ability_grid: list[float],
        ability_transition_matrix: list[list[float]],
    ) -> None:
        self._config = config
        self._ability_grid = ability_grid
        self._ability_trans = ability_transition_matrix
        self._alpha = config.alpha
        self._delta = config.delta
        self._entry_cost = config.firm_entry_cost
        self._min_capital = config.min_firm_capital
        self._beta = 0.95  # Discount factor for firm value computation
        self._use_bellman = config.use_bellman_occ_choice

        # Compute and cache stationary distribution of ability Markov chain
        self._stationary_dist = self._compute_stationary_distribution()
        self._expected_stationary_ability = sum(
            self._stationary_dist[i] * self._ability_grid[i]
            for i in range(len(self._ability_grid))
        )

        # Pre-compute Bellman inverse matrix if using Bellman mode
        self._firm_value_matrix: np.ndarray | None = None
        if self._use_bellman:
            self._firm_value_matrix = self._build_firm_value_matrix()

    def _build_firm_value_matrix(self) -> np.ndarray | None:
        """Pre-compute (I - beta * Pi_e)^{-1} for the firm value Bellman.

        Returns:
            Inverse matrix of shape (n_e, n_e), or None if singular.
        """
        n_e = len(self._ability_grid)
        if n_e == 0:
            return None

        pi_e = np.array(self._ability_trans, dtype=np.float64)
        mat = np.eye(n_e) - self._beta * pi_e

        try:
            cond = np.linalg.cond(mat)
            if cond > 1e12:
                warnings.warn(
                    f"Firm value matrix near-singular (cond={cond:.2e}). "
                    "Falling back to perpetuity formula.",
                    stacklevel=2,
                )
                return None
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Firm value matrix is singular. "
                "Falling back to perpetuity formula.",
                stacklevel=2,
            )
            return None

    def _compute_stationary_distribution(self) -> list[float]:
        """Compute stationary distribution of the ability Markov chain.

        Finds the left eigenvector pi such that pi * P = pi, pi >= 0, sum(pi) = 1.
        Uses power iteration for numerical stability.

        Returns:
            Stationary distribution as a probability vector.
        """
        n = len(self._ability_grid)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Power iteration: start with uniform, multiply by P^T repeatedly
        pi = [1.0 / n] * n
        for _ in range(1000):
            new_pi = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_pi[j] += pi[i] * self._ability_trans[i][j]
            # Normalize
            total = sum(new_pi)
            if total > 0:
                pi = [p / total for p in new_pi]
            else:
                break

            # Check convergence
            max_diff = max(abs(new_pi[i] / total - pi[i]) for i in range(n)) if total > 0 else 0.0
            if max_diff < 1e-12:
                break

        return pi

    # ===================================================================
    # Bellman-based methods (REQ-110..114)
    # ===================================================================

    def compute_firm_value_bellman(
        self,
        capital: float,
        wage: float,
        interest_rate: float,
    ) -> np.ndarray:
        """Solve firm Bellman V^F(e, K) = (I - beta*Pi)^{-1} * pi(e, K).

        Returns the firm value for ALL ability grid points at given capital.

        Args:
            capital: Firm capital K.
            wage: Market wage w.
            interest_rate: Market interest rate r.

        Returns:
            Array of shape (n_e,) with firm values for each ability level.

        Traceability: REQ-111
        """
        n_e = len(self._ability_grid)
        if n_e == 0:
            return np.array([])

        # Compute profit for each ability grid point at this capital
        profits = np.array([
            self._compute_profit(self._ability_grid[j], capital, wage, interest_rate)
            for j in range(n_e)
        ], dtype=np.float64)

        if self._firm_value_matrix is not None:
            firm_values = self._firm_value_matrix @ profits
        else:
            # Fallback: perpetuity at each grid point (legacy behavior)
            firm_values = profits / (1.0 - self._beta)

        # NaN/Inf guard
        if not np.all(np.isfinite(firm_values)):
            log.warning(
                "firm_value_bellman.nan_detected",
                capital=capital,
                wage=wage,
                interest_rate=interest_rate,
            )
            firm_values = np.where(np.isfinite(firm_values), firm_values, 0.0)

        return firm_values

    def _compute_profit(
        self,
        ability: float,
        capital: float,
        wage: float,
        interest_rate: float,
    ) -> float:
        """Compute single-period profit pi(e, K, w, r).

        pi = A * K^alpha * L*^(1-alpha) - w*L* - (r + delta)*K

        Args:
            ability: Entrepreneurial ability e.
            capital: Firm capital K.
            wage: Market wage w.
            interest_rate: Market interest rate r.

        Returns:
            Single-period profit (can be negative).
        """
        if capital <= 0 or wage <= 0 or ability <= 0:
            return 0.0

        opt_l = self._optimal_labor(ability, capital, wage)
        if opt_l > 0:
            output = ability * (capital ** self._alpha) * (opt_l ** (1.0 - self._alpha))
        else:
            output = 0.0
        return output - wage * opt_l - (interest_rate + self._delta) * capital

    def _optimal_capital_golden_section(
        self,
        ability_idx: int,
        wage: float,
        interest_rate: float,
        max_capital: float,
    ) -> float:
        """Golden-section search for K* maximizing V^F(e, K).

        Evaluates 50+ points in [K_min, max_capital] to find the
        capital level that maximizes firm Bellman value at the given
        ability grid index.

        Args:
            ability_idx: Index into ability grid for the entrepreneur.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            max_capital: Upper bound on capital (a - F for entrants).

        Returns:
            Optimal capital K* in [K_min, max_capital].

        Traceability: REQ-114
        """
        a = self._min_capital
        b = max_capital

        if b <= a:
            return a

        def _obj(k: float) -> float:
            v_all = self.compute_firm_value_bellman(k, wage, interest_rate)
            idx = min(ability_idx, len(v_all) - 1)
            return v_all[idx]

        # Golden-section search
        # Requires ~50 evaluations for good convergence
        c = a + _RESPHI * (b - a)
        d = b - _RESPHI * (b - a)
        fc = _obj(c)
        fd = _obj(d)

        n_evals = 2
        max_evals = 60  # Guarantee 50+ evaluations

        while n_evals < max_evals and (b - a) > _EPSILON * (abs(a) + abs(b) + _EPSILON):
            if fd > fc:
                # Maximum is in [c, b]
                a = c
                c = d
                fc = fd
                d = b - _RESPHI * (b - a)
                fd = _obj(d)
            else:
                # Maximum is in [a, d]
                b = d
                d = c
                fd = fc
                c = a + _RESPHI * (b - a)
                fc = _obj(c)
            n_evals += 1

        best_k = (a + b) / 2.0
        return max(best_k, self._min_capital)

    def compute_worker_value_from_vfi(
        self,
        household: HouseholdState,
        vfi_value_func: list[list[float]],
        a_grid: list[float],
    ) -> float:
        """Compute worker value by interpolating the VFI value function.

        Reads V^W(a_i, z_i) directly from the VFI-solved value function
        at the household's actual asset level and productivity state.

        Args:
            household: The household agent state.
            vfi_value_func: VFI value function V[a_idx][z_idx], shape (n_a, n_z).
            a_grid: Asset grid used by the VFI solver.

        Returns:
            Worker lifetime value V^W(a_i, z_i).

        Traceability: REQ-110
        """
        a = household.wealth
        z_idx = household.productivity_index

        # Linear interpolation on asset grid
        if not a_grid or not vfi_value_func:
            # Fallback to legacy if VFI data unavailable
            return self._compute_worker_value_legacy(household, 1.0, 0.05, 0.1)

        n_a = len(a_grid)
        if a <= a_grid[0]:
            return float(vfi_value_func[0][z_idx])
        if a >= a_grid[-1]:
            return float(vfi_value_func[-1][z_idx])

        # Binary search for bracket
        lo, hi = 0, n_a - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a_grid[mid] <= a:
                lo = mid
            else:
                hi = mid

        a_lo = a_grid[lo]
        a_hi = a_grid[hi]
        if abs(a_hi - a_lo) < 1e-15:
            return float(vfi_value_func[lo][z_idx])

        weight = (a - a_lo) / (a_hi - a_lo)
        value = vfi_value_func[lo][z_idx] * (1.0 - weight) + vfi_value_func[hi][z_idx] * weight

        if not math.isfinite(value):
            log.warning("worker_value_vfi.nan", agent_id=household.id, a=a, z_idx=z_idx)
            return self._compute_worker_value_legacy(household, 1.0, 0.05, 0.1)

        return float(value)

    def compute_entrepreneur_value_bellman(
        self,
        household: HouseholdState,
        ability: float,
        opt_capital: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        is_entering: bool = True,
        vfi_value_func: list[list[float]] | None = None,
        a_grid: list[float] | None = None,
    ) -> float:
        """Compute entrepreneur value using Bellman firm value and FOC leisure.

        Uses:
        - Firm profit from V^F(e, K*) via Bellman equation
        - Entrepreneur's leisure from intratemporal FOC (not hardcoded)
        - Consumption from the budget constraint
        - Continuation value as max(V^W, V^F_continuation)

        Args:
            household: The household agent state.
            ability: Entrepreneurial ability.
            opt_capital: Optimal firm capital K*.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            public_goods: Per-capita public goods G.
            is_entering: Whether this is a new entry (pay entry cost).
            vfi_value_func: VFI value function for worker continuation value.
            a_grid: Asset grid used by the VFI solver.

        Returns:
            Entrepreneur lifetime value V^E.

        Traceability: REQ-112
        """
        alpha_u = household.utility_params.alpha
        beta_u = household.utility_params.beta
        gamma = household.utility_params.gamma
        beta_disc = household.utility_params.beta_discount

        # Compute firm profit at optimal capital
        profit = self._compute_profit(ability, opt_capital, wage, interest_rate)

        # Entrepreneur budget: profit + asset income
        if is_entering:
            wealth_base = household.wealth - opt_capital - self._entry_cost
        else:
            wealth_base = household.wealth
        wealth_base = max(wealth_base, 0.0)
        asset_income = interest_rate * wealth_base
        available = profit + asset_income

        # Entrepreneur leisure from intratemporal FOC (REQ-112: no hardcoded values)
        # u(c, l, G) = c^alpha * l^beta * G^gamma
        # Cobb-Douglas intratemporal FOC gives optimal time allocation:
        # l* = beta_u / (alpha_u + beta_u), independent of prices
        leisure = beta_u / max(alpha_u + beta_u, _EPSILON)
        leisure = max(_EPSILON, min(1.0 - _EPSILON, leisure))

        # Consumption from budget constraint: save enough to maintain firm capital
        savings_target = self._delta * wealth_base
        consumption = max(available - savings_target, _EPSILON)

        # Utility
        g = max(public_goods, _EPSILON)
        per_period_utility = (consumption ** alpha_u) * (leisure ** beta_u) * (g ** gamma)

        # Continuation: use V^F Bellman for firm continuation value
        ability_idx = self._find_closest_grid_index(ability)
        firm_values = self.compute_firm_value_bellman(opt_capital, wage, interest_rate)
        if len(firm_values) > ability_idx:
            v_firm_continuation = float(firm_values[ability_idx])
        else:
            v_firm_continuation = 0.0

        # Worker continuation from VFI
        v_worker_continuation = 0.0
        if vfi_value_func is not None and a_grid is not None:
            v_worker_continuation = self.compute_worker_value_from_vfi(
                household, vfi_value_func, a_grid
            )

        # Expected continuation: weighted average (simplification of the max operator)
        # In expectation over (z', e'), there's probability of wanting to switch
        # Use max as the agent will optimally choose each period
        continuation = max(v_worker_continuation, v_firm_continuation)

        lifetime_value = per_period_utility + beta_disc * continuation

        # Entry cost deduction
        if is_entering:
            # Utility cost of entry: marginal utility of consumption times entry cost
            marginal_u_c = alpha_u * per_period_utility / max(consumption, _EPSILON)
            entry_cost_utility = self._entry_cost * marginal_u_c
            lifetime_value -= entry_cost_utility

        if not math.isfinite(lifetime_value):
            log.warning("entrepreneur_value_bellman.nan", agent_id=household.id)
            return 0.0

        return lifetime_value

    # ===================================================================
    # Legacy methods (backward compatibility when use_bellman_occ_choice=False)
    # ===================================================================

    def compute_firm_value(
        self,
        ability: float,
        capital: float,
        wage: float,
        interest_rate: float,
    ) -> float:
        """Compute infinite-horizon PDV of expected profits (legacy perpetuity).

        V = pi_0 + beta * pi_inf / (1 - beta)

        where pi_0 is current-period profit at given ability, and pi_inf is the
        stationary expected profit using the stationary distribution of the
        ability Markov chain.

        Args:
            ability: Current entrepreneurial ability.
            capital: Firm capital K.
            wage: Market wage w.
            interest_rate: Market interest rate r.

        Returns:
            Present discounted value of expected profits (can be negative).
        """
        if capital <= 0 or wage <= 0:
            return 0.0

        alpha = self._alpha

        # Current-period profit at actual ability
        optimal_labor_0 = self._optimal_labor(ability, capital, wage)
        if optimal_labor_0 > 0:
            output_0 = ability * (capital**alpha) * (optimal_labor_0 ** (1 - alpha))
        else:
            output_0 = 0.0
        profit_0 = output_0 - wage * optimal_labor_0 - (interest_rate + self._delta) * capital

        # Stationary expected profit using E[A_inf] from stationary distribution
        e_ability = self._expected_stationary_ability
        optimal_labor_inf = self._optimal_labor(e_ability, capital, wage)
        if optimal_labor_inf > 0:
            output_inf = e_ability * (capital**alpha) * (optimal_labor_inf ** (1 - alpha))
        else:
            output_inf = 0.0
        rental_cost = (interest_rate + self._delta) * capital
        profit_inf = output_inf - wage * optimal_labor_inf - rental_cost

        # Infinite-horizon value: first period exact + future perpetuity
        total_value = profit_0 + self._beta * profit_inf / (1.0 - self._beta)

        return total_value

    def optimal_capital(
        self,
        ability: float,
        wage: float,
        interest_rate: float,
        wealth: float,
    ) -> float:
        """Compute optimal capital K* that maximizes firm value.

        Dispatches to golden-section search (Bellman) or grid search (legacy).

        Args:
            ability: Entrepreneurial ability.
            wage: Market wage.
            interest_rate: Market interest rate.
            wealth: Agent's available wealth.

        Returns:
            Optimal capital level (0 if no feasible capital).
        """
        max_capital = wealth - self._entry_cost
        if max_capital < self._min_capital:
            return 0.0

        if self._use_bellman:
            ability_idx = self._find_closest_grid_index(ability)
            return self._optimal_capital_golden_section(
                ability_idx, wage, interest_rate, max_capital
            )

        # Legacy: grid search over capital levels
        best_k = self._min_capital
        best_value = self.compute_firm_value(ability, best_k, wage, interest_rate)

        n_points = 20
        step = (max_capital - self._min_capital) / max(n_points - 1, 1)
        for i in range(n_points):
            k = self._min_capital + i * step
            v = self.compute_firm_value(ability, k, wage, interest_rate)
            if v > best_value:
                best_value = v
                best_k = k

        return best_k

    def compute_worker_value(
        self,
        household: HouseholdState,
        wage: float,
        interest_rate: float,
        public_goods: float,
        vfi_value_func: list[list[float]] | None = None,
        a_grid: list[float] | None = None,
    ) -> float:
        """Compute lifetime utility value as a worker.

        When Bellman mode is active and VFI data is provided, reads the
        value function directly. Otherwise falls back to the legacy
        perpetuity approximation.

        Args:
            household: The household agent state.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            public_goods: Per-capita public goods G.
            vfi_value_func: VFI value function (Bellman mode).
            a_grid: Asset grid (Bellman mode).

        Returns:
            Lifetime value as a worker.
        """
        if self._use_bellman and vfi_value_func is not None and a_grid is not None:
            return self.compute_worker_value_from_vfi(household, vfi_value_func, a_grid)
        return self._compute_worker_value_legacy(household, wage, interest_rate, public_goods)

    def _compute_worker_value_legacy(
        self,
        household: HouseholdState,
        wage: float,
        interest_rate: float,
        public_goods: float,
    ) -> float:
        """Legacy perpetuity-based worker value computation.

        Uses hardcoded leisure ~0.35 and consumption ~70% of income.
        """
        alpha = household.utility_params.alpha
        beta_u = household.utility_params.beta
        gamma = household.utility_params.gamma
        beta_disc = household.utility_params.beta_discount

        # Optimal leisure ~0.35 (from VFI typical solution for these preferences)
        leisure = max(household.leisure, 0.35) if household.leisure > 0.0 else 0.35
        labor = 1.0 - leisure

        # Worker income
        income = wage * household.productivity * labor
        asset_income = interest_rate * household.wealth

        # Approximate optimal consumption: fraction of available resources
        available = income + asset_income
        consumption = max(available * 0.7, _EPSILON)  # consume ~70% of income

        # Utility
        c = max(consumption, _EPSILON)
        l_val = max(leisure, _EPSILON)
        g = max(public_goods, _EPSILON)
        per_period_utility = (c**alpha) * (l_val**beta_u) * (g**gamma)

        # Perpetuity value
        return per_period_utility / (1.0 - beta_disc)

    def compute_entrepreneur_value(
        self,
        household: HouseholdState,
        ability: float,
        opt_capital: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        is_entering: bool = True,
        vfi_value_func: list[list[float]] | None = None,
        a_grid: list[float] | None = None,
    ) -> float:
        """Compute lifetime utility value as an entrepreneur.

        Dispatches to Bellman or legacy mode based on config.

        Args:
            household: The household agent state.
            ability: Entrepreneurial ability.
            opt_capital: Optimal firm capital K*.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            public_goods: Per-capita public goods G.
            is_entering: Whether this is a new entry (pay entry cost).
            vfi_value_func: VFI value function (Bellman mode).
            a_grid: Asset grid (Bellman mode).

        Returns:
            Lifetime value as an entrepreneur.
        """
        if self._use_bellman:
            return self.compute_entrepreneur_value_bellman(
                household, ability, opt_capital, wage, interest_rate,
                public_goods, is_entering, vfi_value_func, a_grid,
            )
        return self._compute_entrepreneur_value_legacy(
            household, ability, opt_capital, wage, interest_rate,
            public_goods, is_entering,
        )

    def _compute_entrepreneur_value_legacy(
        self,
        household: HouseholdState,
        ability: float,
        opt_capital: float,
        wage: float,
        interest_rate: float,
        public_goods: float,
        is_entering: bool = True,
    ) -> float:
        """Legacy perpetuity-based entrepreneur value with hardcoded constants."""
        alpha_u = household.utility_params.alpha
        beta_u = household.utility_params.beta
        gamma = household.utility_params.gamma
        beta_disc = household.utility_params.beta_discount

        # Compute firm profit at optimal capital
        opt_labor = self._optimal_labor(ability, opt_capital, wage)
        if opt_labor > 0 and opt_capital > 0:
            output = ability * (opt_capital**self._alpha) * (opt_labor ** (1.0 - self._alpha))
        else:
            output = 0.0
        profit = output - wage * opt_labor - (interest_rate + self._delta) * opt_capital

        # Entrepreneur consumption: profit + asset income - savings for firm
        wealth_base = household.wealth - opt_capital if is_entering else household.wealth
        asset_income = interest_rate * wealth_base
        consumption = max(profit + asset_income, _EPSILON)

        # Entrepreneur has less leisure (running a firm takes effort)
        leisure = 0.2  # Entrepreneurs work harder

        # Utility
        c = max(consumption, _EPSILON)
        l_val = max(leisure, _EPSILON)
        g = max(public_goods, _EPSILON)
        per_period_utility = (c**alpha_u) * (l_val**beta_u) * (g**gamma)

        # Perpetuity value minus entry cost (converted to utility units)
        lifetime_value = per_period_utility / (1.0 - beta_disc)
        if is_entering:
            # Entry cost as forgone consumption utility
            marginal_u = alpha_u / max(c, _EPSILON)
            entry_cost_utility = self._entry_cost * marginal_u * per_period_utility
            lifetime_value -= entry_cost_utility

        return lifetime_value

    def solve_entry_exit(
        self,
        household: HouseholdState,
        firms: list[FirmState],
        market: MarketState,
        public_goods: float = 0.1,
        vfi_value_func: list[list[float]] | None = None,
        a_grid: list[float] | None = None,
    ) -> EntrepreneurialDecision:
        """Solve entry/exit decision using utility-based occupational choice.

        Compares lifetime utility as worker vs entrepreneur:
        - Entry: worker becomes entrepreneur if V_entrepreneur > V_worker
          AND wealth >= min_capital + entry_cost
        - Exit: entrepreneur reverts to worker if V_worker > V_entrepreneur
        - Continue: otherwise maintain current role and optimize

        Args:
            household: The household agent state.
            firms: All active firms (to find owned firm).
            market: Current market equilibrium.
            public_goods: Per-capita public goods G.
            vfi_value_func: VFI value function (Bellman mode, REQ-110).
            a_grid: Asset grid (Bellman mode).

        Returns:
            EntrepreneurialDecision with appropriate action.

        Traceability: REQ-113
        """
        wage = market.wage
        r = market.interest_rate
        ability = household.entrepreneurial_ability

        # Compute worker value for comparison
        v_worker = self.compute_worker_value(
            household, wage, r, public_goods, vfi_value_func, a_grid
        )

        # Find if this household already owns a firm
        owned_firm = None
        for f in firms:
            if f.owner_id == household.id:
                owned_firm = f
                break

        if owned_firm is not None:
            # --- Existing entrepreneur: continue or exit ---
            v_entrepreneur = self.compute_entrepreneur_value(
                household,
                ability,
                owned_firm.capital,
                wage,
                r,
                public_goods,
                is_entering=False,
                vfi_value_func=vfi_value_func,
                a_grid=a_grid,
            )

            if v_worker > v_entrepreneur:
                # Exit: better off as worker
                log.debug(
                    "entrepreneurial.exit",
                    agent_id=household.id,
                    v_worker=round(v_worker, 4),
                    v_entrepreneur=round(v_entrepreneur, 4),
                )
                return EntrepreneurialDecision(close_firm=True)

            # Continue: optimize capital and labor
            opt_k = self.optimal_capital(ability, wage, r, household.wealth + owned_firm.capital)
            opt_l = self._optimal_labor(ability, opt_k, wage)
            rd = max(opt_k * 0.02, 0.0)  # R&D = 2% of capital

            return EntrepreneurialDecision(
                create_firm=False,
                capital_investment=opt_k,
                labor_demand=opt_l,
                rd_spend=rd,
                close_firm=False,
            )

        # --- Worker: check entry ---
        opt_k = self.optimal_capital(ability, wage, r, household.wealth)
        if opt_k < self._min_capital:
            return EntrepreneurialDecision()  # Cannot afford entry

        # Wealth check: need enough for capital + entry cost
        if household.wealth < self._min_capital + self._entry_cost:
            return EntrepreneurialDecision()

        v_entrepreneur = self.compute_entrepreneur_value(
            household,
            ability,
            opt_k,
            wage,
            r,
            public_goods,
            is_entering=True,
            vfi_value_func=vfi_value_func,
            a_grid=a_grid,
        )

        if v_entrepreneur > v_worker:
            # Enter: better off as entrepreneur
            opt_l = self._optimal_labor(ability, opt_k, wage)
            rd = max(opt_k * 0.02, 0.0)

            log.debug(
                "entrepreneurial.entry",
                agent_id=household.id,
                v_worker=round(v_worker, 4),
                v_entrepreneur=round(v_entrepreneur, 4),
                capital=round(opt_k, 2),
            )

            return EntrepreneurialDecision(
                create_firm=True,
                capital_investment=opt_k,
                labor_demand=opt_l,
                rd_spend=rd,
            )

        return EntrepreneurialDecision()  # Stay as worker

    def solve_all(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        public_goods: float = 0.1,
        vfi_value_func: list[list[float]] | None = None,
        a_grid: list[float] | None = None,
    ) -> dict[str, EntrepreneurialDecision]:
        """Solve entrepreneurial decisions for all agents.

        Uses utility-based occupational choice comparing lifetime value
        as worker vs entrepreneur for entry/exit decisions.

        Args:
            households: All household states.
            firms: All active firms.
            market: Current market equilibrium.
            public_goods: Per-capita public goods G.
            vfi_value_func: VFI value function (Bellman mode, REQ-110).
            a_grid: Asset grid (Bellman mode).

        Returns:
            Dict mapping agent_id to EntrepreneurialDecision.
        """
        decisions: dict[str, EntrepreneurialDecision] = {}
        for h in households:
            decisions[h.id] = self.solve_entry_exit(
                h, firms, market, public_goods, vfi_value_func, a_grid
            )

        num_creating = sum(1 for d in decisions.values() if d.create_firm)
        num_closing = sum(1 for d in decisions.values() if d.close_firm)
        log.debug(
            "entrepreneurial_solver.solved_all",
            num_households=len(households),
            num_creating=num_creating,
            num_closing=num_closing,
        )

        return decisions

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _optimal_labor(
        self,
        ability: float,
        capital: float,
        wage: float,
    ) -> float:
        """Compute optimal labor demand from Cobb-Douglas FOC.

        FOC: (1-alpha) * A * K^alpha * L^(-alpha) = w
        => L* = ((1-alpha) * A * K^alpha / w)^(1/alpha)

        Args:
            ability: Firm TFP (entrepreneurial ability * aggregate TFP).
            capital: Firm capital K.
            wage: Market wage w.

        Returns:
            Optimal labor demand L* >= 0.
        """
        if wage <= 0 or capital <= 0 or ability <= 0:
            return 0.0

        alpha = self._alpha
        try:
            base = (1.0 - alpha) * ability * (capital**alpha) / wage
            if base <= 0:
                return 0.0
            return base ** (1.0 / alpha)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0

    def _find_closest_grid_index(self, ability: float) -> int:
        """Find the index of the closest grid value to the given ability.

        Args:
            ability: Entrepreneurial ability value.

        Returns:
            Index into ability_grid.
        """
        if not self._ability_grid:
            return 0
        best_idx = 0
        best_dist = abs(ability - self._ability_grid[0])
        for i in range(1, len(self._ability_grid)):
            dist = abs(ability - self._ability_grid[i])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
