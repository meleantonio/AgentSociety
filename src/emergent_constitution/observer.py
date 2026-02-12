"""Observer/Statistician — aggregate statistics and rule-change detection.

v1 functions (observe_tick, compute_pareto_efficiency, etc.) retained for
backward compatibility.

v2 class (ObserverV2) implements REQ-030, REQ-031, REQ-032 with full DSGE-HA
statistics: Gini, Pareto score, Y/C/I aggregates, wealth quantiles,
unemployment, firm stats, social welfare, cumulative discounted welfare.
"""

from __future__ import annotations

import math
import statistics

import structlog

from emergent_constitution.citizen import compute_gini
from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution, ConstitutionV2
from emergent_constitution.models.history import (
    HistoryEntry,
    HistoryEntryV2,
    PeriodState,
    SimulationOutputV2,
    WelfareSummary,
)
from emergent_constitution.models.household import HouseholdState, OccupationalRole

log = structlog.get_logger()

# Small transfer amount used to test for Pareto improvements.
_TRANSFER_DELTA = 1.0


def _compute_utility(
    agent: AgentState,
    wealth: float,
    public_goods: float,
) -> float:
    """Compute Cobb-Douglas utility for an agent.

    Args:
        agent: The agent whose utility parameters to use.
        wealth: Consumption/wealth level.
        public_goods: Public goods per capita.

    Returns:
        Utility value (non-negative).
    """
    p = agent.utility_params
    # U = wealth^alpha * leisure^beta * public_goods^gamma
    # Leisure is fixed at 1.0 for simplicity.
    leisure = 1.0
    try:
        return (
            math.pow(wealth, p.alpha) * math.pow(leisure, p.beta) * math.pow(public_goods, p.gamma)
        )
    except (ValueError, OverflowError):
        return 0.0


def compute_pareto_efficiency(agents: list[AgentState]) -> float:
    """Estimate Pareto efficiency as fraction of pairs with no improving transfer.

    Tests each pair of agents: can a small transfer from the richer to the
    poorer improve the recipient without harming the donor? The score is
    the fraction of pairs where no such improvement exists (1.0 = fully
    Pareto efficient).

    Args:
        agents: List of agent states.

    Returns:
        Score in [0, 1]. 1.0 means no Pareto improvements found.
    """
    n = len(agents)
    if n <= 1:
        return 1.0

    total_wealth = sum(a.wealth for a in agents)
    public_goods_per_capita = total_wealth / n if n > 0 else 0.0

    pairs_checked = 0
    improvements_found = 0

    for i in range(n):
        for j in range(i + 1, n):
            if agents[i].wealth >= agents[j].wealth:
                a_rich, a_poor = agents[i], agents[j]
            else:
                a_rich, a_poor = agents[j], agents[i]

            # Skip if the richer agent can't afford the transfer.
            if a_rich.wealth < _TRANSFER_DELTA:
                pairs_checked += 1
                continue

            u_rich_before = _compute_utility(a_rich, a_rich.wealth, public_goods_per_capita)
            u_poor_before = _compute_utility(a_poor, a_poor.wealth, public_goods_per_capita)
            u_rich_after = _compute_utility(
                a_rich, a_rich.wealth - _TRANSFER_DELTA, public_goods_per_capita
            )
            u_poor_after = _compute_utility(
                a_poor, a_poor.wealth + _TRANSFER_DELTA, public_goods_per_capita
            )

            pairs_checked += 1
            # Pareto improvement: recipient better off, donor no worse off.
            if u_poor_after > u_poor_before and u_rich_after >= u_rich_before:
                improvements_found += 1

    if pairs_checked == 0:
        return 1.0
    return 1.0 - (improvements_found / pairs_checked)


def observe_tick(
    tick: int,
    agents: list[AgentState],
    constitution: Constitution,
    prev_constitution: Constitution | None,
) -> HistoryEntry:
    """Compute aggregate statistics for the current tick.

    Args:
        tick: Current tick number.
        agents: Current agent states.
        constitution: Current constitution.
        prev_constitution: Constitution at the previous observation, or None
            if this is the first observation.

    Returns:
        HistoryEntry with computed statistics and a deep-copied constitution snapshot.
    """
    wealths = [a.wealth for a in agents]

    gini = compute_gini(wealths)
    mean_wealth = statistics.mean(wealths) if wealths else 0.0
    median_wealth = statistics.median(wealths) if wealths else 0.0
    total_output = sum(a.productivity for a in agents)
    pareto_score = compute_pareto_efficiency(agents)
    rule_changes = _detect_rule_changes(prev_constitution, constitution)
    num_coalitions = len({a.coalition_id for a in agents if a.coalition_id is not None})

    return HistoryEntry(
        tick=tick,
        gini=gini,
        total_output=total_output,
        mean_wealth=mean_wealth,
        median_wealth=median_wealth,
        pareto_score=pareto_score,
        rule_changes=rule_changes,
        constitution_snapshot=constitution.model_copy(deep=True),
        num_coalitions=num_coalitions,
    )


def _detect_rule_changes(
    prev: Constitution | None,
    current: Constitution,
) -> list[str]:
    """Detect which constitutional fields changed between observations.

    Args:
        prev: Constitution at the previous observation, or None for first observation.
        current: Constitution at the current observation.

    Returns:
        List of human-readable change descriptions, e.g. ``"tax_rate: 0.0 -> 0.25"``.
    """
    if prev is None:
        return []

    changes: list[str] = []
    for field_name in ("property_rule", "tax_rate", "voting_rule", "redistribution_rule"):
        old_val = getattr(prev, field_name)
        new_val = getattr(current, field_name)
        if old_val != new_val:
            changes.append(f"{field_name}: {old_val} \u2192 {new_val}")

    return changes


# ============================================================================
# v2 Observer (DSGE-HA statistics)
# ============================================================================

# Epsilon guard for utility computation.
_EPSILON = 1e-10


class ObserverV2:
    """Computes and records macro statistics for the DSGE-HA simulation.

    Tracks observation history and cumulative welfare across periods.
    Used by LeadV2 to record statistics at each observer_interval.

    Args:
        config: Simulation configuration.

    Implements REQ-030, REQ-031, REQ-032.
    """

    def __init__(self, config: SimulationConfigV2) -> None:
        self.config = config
        self.history: list[HistoryEntryV2] = []
        self._cumulative_welfare: float = 0.0
        self._prev_constitution: ConstitutionV2 | None = None

    def observe(
        self,
        period_state: PeriodState,
        prev_constitution: ConstitutionV2 | None = None,
    ) -> HistoryEntryV2:
        """Compute all REQ-030 statistics and record a history entry.

        Args:
            period_state: Complete state for this period.
            prev_constitution: Constitution at the previous observation, or
                None for the first observation. If provided, overrides the
                internally tracked previous constitution.

        Returns:
            HistoryEntryV2 with all computed statistics.
        """
        if prev_constitution is not None:
            self._prev_constitution = prev_constitution

        households = period_state.households
        firms = period_state.firms
        market = period_state.market
        constitution = period_state.constitution
        t = period_state.period

        wealths = [h.wealth for h in households]
        n = len(wealths)

        # Gini coefficient
        gini = self._compute_gini(wealths)

        # Pareto efficiency score
        pareto_score = self.compute_pareto_efficiency_v2(households)

        # Wealth statistics
        mean_wealth = sum(wealths) / n if n > 0 else 0.0
        sorted_w = sorted(wealths)
        if n > 0:
            if n % 2 == 1:
                median_wealth = sorted_w[n // 2]
            else:
                median_wealth = (sorted_w[n // 2 - 1] + sorted_w[n // 2]) / 2.0
        else:
            median_wealth = 0.0

        # Wealth quantiles [p10, p25, p50, p75, p90]
        quantiles = self._compute_quantiles(sorted_w)

        # Unemployment rate
        unemployed = sum(1 for h in households if h.role == OccupationalRole.UNEMPLOYED)
        unemployment_rate = unemployed / n if n > 0 else 0.0

        # Firm statistics
        num_firms = len(firms)
        mean_firm_size = (
            sum(len(f.worker_ids) for f in firms) / num_firms if num_firms > 0 else 0.0
        )
        total_rd = sum(f.rd_spend for f in firms)

        # Social welfare (REQ-031)
        social_welfare = sum(h.realized_utility for h in households)

        # Cumulative discounted welfare
        avg_beta = sum(h.utility_params.beta_discount for h in households) / n if n > 0 else 0.95
        self._cumulative_welfare = self._cumulative_welfare * avg_beta + social_welfare

        # Rule changes
        rule_changes = detect_rule_changes_v2(self._prev_constitution, constitution)

        entry = HistoryEntryV2(
            period=t,
            gini=gini,
            pareto_score=pareto_score,
            aggregate_output=market.aggregate_output,
            aggregate_consumption=sum(h.consumption for h in households),
            aggregate_investment=market.aggregate_investment,
            mean_wealth=mean_wealth,
            median_wealth=median_wealth,
            wealth_quantiles=quantiles,
            unemployment_rate=unemployment_rate,
            num_active_firms=num_firms,
            mean_firm_size=mean_firm_size,
            aggregate_rd_spend=total_rd,
            social_welfare=social_welfare,
            cumulative_welfare=self._cumulative_welfare,
            wage=market.wage,
            interest_rate=market.interest_rate,
            rule_changes=rule_changes,
            constitution_snapshot=constitution.model_copy(deep=True),
        )

        self.history.append(entry)
        self._prev_constitution = constitution.model_copy(deep=True)

        log.info(
            "observation_v2.recorded",
            period=t,
            gini=round(gini, 4),
            mean_wealth=round(mean_wealth, 2),
            social_welfare=round(social_welfare, 2),
            pareto_score=round(pareto_score, 4),
            num_firms=num_firms,
        )

        return entry

    def finalize(self, period_state: PeriodState) -> SimulationOutputV2:
        """Produce final simulation output per REQ-032.

        Args:
            period_state: The final period state.

        Returns:
            SimulationOutputV2 with constitution, history, final states,
            and welfare summary.
        """
        total_welfare = sum(e.social_welfare for e in self.history)

        welfare_summary = WelfareSummary(
            llm_total_welfare=total_welfare,
            benchmark_total_welfare=None,
            per_agent_comparison=None,
        )

        return SimulationOutputV2(
            constitution=period_state.constitution.model_copy(deep=True),
            history=list(self.history),
            final_households=[h.model_copy(deep=True) for h in period_state.households],
            final_firms=[f.model_copy(deep=True) for f in period_state.firms],
            welfare_summary=welfare_summary,
            seed=self.config.seed,
            total_periods=period_state.period,
        )

    @staticmethod
    def compute_pareto_efficiency_v2(
        households: list[HouseholdState],
    ) -> float:
        """Estimate Pareto efficiency for v2 households.

        Tests each pair: can a small consumption transfer from the richer
        to the poorer improve the recipient without harming the donor?

        Uses the household's actual utility parameters and realized
        consumption/leisure context.

        Args:
            households: List of household states.

        Returns:
            Score in [0, 1]. 1.0 means no Pareto improvements found.
        """
        n = len(households)
        if n <= 1:
            return 1.0

        pairs_checked = 0
        improvements_found = 0

        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = households[i], households[j]
                if hi.wealth >= hj.wealth:
                    h_rich, h_poor = hi, hj
                else:
                    h_rich, h_poor = hj, hi

                if h_rich.wealth < _TRANSFER_DELTA:
                    pairs_checked += 1
                    continue

                u_rich_before = _compute_utility_v2(h_rich, h_rich.consumption)
                u_poor_before = _compute_utility_v2(h_poor, h_poor.consumption)
                u_rich_after = _compute_utility_v2(
                    h_rich,
                    max(h_rich.consumption - _TRANSFER_DELTA, 0.0),
                )
                u_poor_after = _compute_utility_v2(h_poor, h_poor.consumption + _TRANSFER_DELTA)

                pairs_checked += 1
                if u_poor_after > u_poor_before and u_rich_after >= u_rich_before:
                    improvements_found += 1

        if pairs_checked == 0:
            return 1.0
        return 1.0 - (improvements_found / pairs_checked)

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        """Compute the Gini coefficient of a list of values.

        Args:
            values: Non-negative values.

        Returns:
            Gini in [0, 1].
        """
        n = len(values)
        if n < 2:
            return 0.0
        sorted_v = sorted(values)
        total = sum(sorted_v)
        if total <= 0.0:
            return 0.0
        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
        return (2.0 * weighted_sum - (n + 1) * total) / (n * total)

    @staticmethod
    def _compute_quantiles(sorted_values: list[float]) -> list[float]:
        """Compute wealth quantiles [p10, p25, p50, p75, p90].

        Args:
            sorted_values: Sorted list of values.

        Returns:
            List of 5 quantile values, or empty if input is empty.
        """
        n = len(sorted_values)
        if n == 0:
            return []
        quantiles = []
        for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
            idx = min(int(p * n), n - 1)
            quantiles.append(sorted_values[idx])
        return quantiles


def _compute_utility_v2(
    household: HouseholdState,
    consumption: float,
) -> float:
    """Compute Cobb-Douglas utility for a v2 household.

    Uses actual leisure and a minimal public goods guard.

    Args:
        household: Household state with utility params and leisure.
        consumption: Consumption level to evaluate.

    Returns:
        Utility value >= 0.
    """
    c = max(consumption, _EPSILON)
    lei = max(household.leisure, _EPSILON)
    g = _EPSILON  # Public goods not available at pair level; use guard
    p = household.utility_params
    try:
        return (c**p.alpha) * (lei**p.beta) * (g**p.gamma)
    except (ValueError, OverflowError):
        return 0.0


def detect_rule_changes_v2(
    prev: ConstitutionV2 | None,
    current: ConstitutionV2,
) -> list[str]:
    """Detect changes between two v2 constitutions.

    Args:
        prev: Previous constitution, or None for first observation.
        current: Current constitution.

    Returns:
        List of human-readable change descriptions.
    """
    if prev is None:
        return []

    changes: list[str] = []

    if prev.voting_rule != current.voting_rule:
        changes.append(f"voting_rule: {prev.voting_rule} -> {current.voting_rule}")

    prev_names = set(prev.rules.keys())
    curr_names = set(current.rules.keys())

    for name in sorted(curr_names - prev_names):
        changes.append(f"rule added: {name}")

    for name in sorted(prev_names - curr_names):
        changes.append(f"rule removed: {name}")

    for name in sorted(prev_names & curr_names):
        prev_rule = prev.rules[name]
        curr_rule = current.rules[name]
        if prev_rule.parameters != curr_rule.parameters:
            changes.append(
                f"rule modified: {name} params {prev_rule.parameters} -> {curr_rule.parameters}"
            )
        if prev_rule.version != curr_rule.version:
            changes.append(f"rule version: {name} v{prev_rule.version} -> v{curr_rule.version}")

    return changes
