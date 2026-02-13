"""Entrepreneurial decision solver — firm value, entry/exit, optimal capital.

Implements solver-driven entrepreneurial decisions based on heterogeneous
entrepreneurial ability modeled as a Markov chain. Computes truncated PDV
of expected firm profits to determine entry/exit decisions and optimal
factor demands.
"""

from __future__ import annotations

import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.decisions import EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()


class EntrepreneurialSolver:
    """Solver for entrepreneurial entry/exit decisions and factor demands.

    Uses truncated present discounted value (PDV) of expected profits to
    determine whether agents should create or close firms, and optimal
    capital/labor choices for active entrepreneurs.

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
        self._horizon = config.firm_value_horizon
        self._entry_cost = config.firm_entry_cost
        self._min_capital = config.min_firm_capital
        self._beta = 0.95  # Discount factor for firm value computation

    def compute_firm_value(
        self,
        ability: float,
        capital: float,
        wage: float,
        interest_rate: float,
    ) -> float:
        """Compute truncated PDV of expected profits given ability and prices.

        V = sum_{t=0}^{H} beta^t * E[pi_t | A_0=ability]
        where pi_t = A_t * K^alpha * L*^(1-alpha) - w*L* - (r+delta)*K
        and L* is optimal labor from FOC.

        Uses the ability Markov chain to compute expected future TFP.

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
        # Find closest ability grid index for Markov chain expectations
        ability_idx = self._find_closest_grid_index(ability)

        total_value = 0.0
        # Expected ability at each future horizon step
        # Start with probability 1.0 on current state
        n_states = len(self._ability_grid)
        prob_dist = [0.0] * n_states
        prob_dist[ability_idx] = 1.0

        for t in range(self._horizon + 1):
            # Expected ability given current distribution
            expected_ability = sum(prob_dist[i] * self._ability_grid[i] for i in range(n_states))

            # Optimal labor from FOC: L* = ((1-alpha) * A * K^alpha / w)^(1/alpha)
            optimal_labor = self._optimal_labor(expected_ability, capital, wage)

            # Profit: pi = A * K^alpha * L^(1-alpha) - w*L - (r+delta)*K
            if optimal_labor > 0:
                output = expected_ability * (capital**alpha) * (optimal_labor ** (1 - alpha))
            else:
                output = 0.0
            profit = output - wage * optimal_labor - (interest_rate + self._delta) * capital

            total_value += (self._beta**t) * profit

            # Evolve probability distribution forward: prob' = prob * P
            if t < self._horizon:
                new_prob = [0.0] * n_states
                for j in range(n_states):
                    for i in range(n_states):
                        new_prob[j] += prob_dist[i] * self._ability_trans[i][j]
                prob_dist = new_prob

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

    def solve_entry_exit(
        self,
        household: HouseholdState,
        firms: list[FirmState],
        market: MarketState,
    ) -> EntrepreneurialDecision:
        """Solve entry/exit decision for a single agent.

        Entry: if not entrepreneur and E[V_firm(ability, K*, w, r)] >
               wealth * r + entry_cost.
        Exit: if entrepreneur and V_firm < K_f (liquidation value).
        Continue: if entrepreneur, update K, L, R&D based on value
                  maximization.

        Args:
            household: The household agent state.
            firms: All active firms (to find owned firm).
            market: Current market equilibrium.

        Returns:
            EntrepreneurialDecision with appropriate action.
        """
        wage = market.wage
        r = market.interest_rate
        ability = household.entrepreneurial_ability

        # Find if this household already owns a firm
        owned_firm = None
        for f in firms:
            if f.owner_id == household.id:
                owned_firm = f
                break

        if owned_firm is not None:
            # --- Existing entrepreneur: continue or exit ---
            firm_value = self.compute_firm_value(ability, owned_firm.capital, wage, r)
            liquidation_value = owned_firm.capital

            if firm_value < liquidation_value:
                # Exit: firm is worth less than its capital
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

        firm_value = self.compute_firm_value(ability, opt_k, wage, r)
        opportunity_cost = household.wealth * max(r, 0.0) + self._entry_cost

        if firm_value > opportunity_cost:
            # Enter: expected firm value exceeds opportunity cost
            opt_l = self._optimal_labor(ability, opt_k, wage)
            rd = max(opt_k * 0.02, 0.0)

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
    ) -> dict[str, EntrepreneurialDecision]:
        """Solve entrepreneurial decisions for all agents.

        Entrepreneurs get full entry/exit/continue decisions.
        Workers get entry check.

        Args:
            households: All household states.
            firms: All active firms.
            market: Current market equilibrium.

        Returns:
            Dict mapping agent_id to EntrepreneurialDecision.
        """
        decisions: dict[str, EntrepreneurialDecision] = {}
        for h in households:
            decisions[h.id] = self.solve_entry_exit(h, firms, market)

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
