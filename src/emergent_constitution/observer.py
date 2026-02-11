"""Observer/Statistician — aggregate statistics and rule-change detection.

All functions are pure: they take state and return results without
mutating any inputs.
"""

from __future__ import annotations

import math
import statistics

from emergent_constitution.citizen import compute_gini
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.history import HistoryEntry

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

    return HistoryEntry(
        tick=tick,
        gini=gini,
        total_output=total_output,
        mean_wealth=mean_wealth,
        median_wealth=median_wealth,
        pareto_score=pareto_score,
        rule_changes=rule_changes,
        constitution_snapshot=constitution.model_copy(deep=True),
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
        List of human-readable change descriptions, e.g. ``"tax_rate: 0.0 → 0.25"``.
    """
    if prev is None:
        return []

    changes: list[str] = []
    for field_name in ("property_rule", "tax_rate", "voting_rule", "redistribution_rule"):
        old_val = getattr(prev, field_name)
        new_val = getattr(current, field_name)
        if old_val != new_val:
            changes.append(f"{field_name}: {old_val} → {new_val}")

    return changes
