"""Entrepreneurial decision solver — firm value, entry/exit, optimal capital.

Implements solver-driven entrepreneurial decisions based on heterogeneous
entrepreneurial ability modeled as a Markov chain. Uses utility-based
occupational choice comparing lifetime value as worker vs entrepreneur
to determine entry/exit decisions and optimal factor demands.

Firm value uses a closed-form infinite-horizon formula with the stationary
distribution of the ability Markov chain, replacing the truncated PDV loop.
"""

from __future__ import annotations

import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.decisions import EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()

# Epsilon guard for utility computation
_EPSILON = 1e-10


class EntrepreneurialSolver:
    """Solver for entrepreneurial entry/exit decisions and factor demands.

    Uses infinite-horizon present discounted value (PDV) of expected profits to
    determine whether agents should create or close firms, and optimal
    capital/labor choices for active entrepreneurs.

    The stationary distribution of the ability Markov chain is computed once
    at initialization and used for the closed-form infinite-horizon formula.

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

        # Compute and cache stationary distribution of ability Markov chain
        self._stationary_dist = self._compute_stationary_distribution()
        self._expected_stationary_ability = sum(
            self._stationary_dist[i] * self._ability_grid[i]
            for i in range(len(self._ability_grid))
        )

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

    def compute_firm_value(
        self,
        ability: float,
        capital: float,
        wage: float,
        interest_rate: float,
    ) -> float:
        """Compute infinite-horizon PDV of expected profits.

        V = pi_0 + beta * pi_inf / (1 - beta)

        where pi_0 is current-period profit at given ability, and pi_inf is the
        stationary expected profit using the stationary distribution of the
        ability Markov chain.

        First period uses exact current ability; future periods use the
        stationary expected ability as a perpetuity.

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

        Subject to K <= wealth - firm_entry_cost and K >= min_firm_capital.
        Uses grid search over feasible capital levels.

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

        # Grid search over capital levels
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
    ) -> float:
        """Compute lifetime utility value as a worker.

        Uses the household's optimal (c*, l*) decision and approximates
        lifetime value as a perpetuity: V_worker = u(c*, l*, G) / (1 - beta).

        Args:
            household: The household agent state.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            public_goods: Per-capita public goods G.

        Returns:
            Approximate lifetime value as a worker.
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
    ) -> float:
        """Compute lifetime utility value as an entrepreneur.

        Entrepreneur earns firm profits, but has reduced leisure from
        running a firm and pays entry cost if newly entering.

        Args:
            household: The household agent state.
            ability: Entrepreneurial ability.
            opt_capital: Optimal firm capital K*.
            wage: Market wage w.
            interest_rate: Market interest rate r.
            public_goods: Per-capita public goods G.
            is_entering: Whether this is a new entry (pay entry cost).

        Returns:
            Approximate lifetime value as an entrepreneur.
        """
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

        Returns:
            EntrepreneurialDecision with appropriate action.
        """
        wage = market.wage
        r = market.interest_rate
        ability = household.entrepreneurial_ability

        # Compute worker value for comparison
        v_worker = self.compute_worker_value(household, wage, r, public_goods)

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
    ) -> dict[str, EntrepreneurialDecision]:
        """Solve entrepreneurial decisions for all agents.

        Uses utility-based occupational choice comparing lifetime value
        as worker vs entrepreneur for entry/exit decisions.

        Args:
            households: All household states.
            firms: All active firms.
            market: Current market equilibrium.
            public_goods: Per-capita public goods G.

        Returns:
            Dict mapping agent_id to EntrepreneurialDecision.
        """
        decisions: dict[str, EntrepreneurialDecision] = {}
        for h in households:
            decisions[h.id] = self.solve_entry_exit(h, firms, market, public_goods)

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
