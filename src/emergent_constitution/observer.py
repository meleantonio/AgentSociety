"""Observer/Statistician — aggregate statistics and rule-change detection.

All functions are pure: they take state and return results without
mutating any inputs.
"""

from __future__ import annotations

import statistics

from emergent_constitution.citizen import compute_gini
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.history import HistoryEntry


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
    rule_changes = _detect_rule_changes(prev_constitution, constitution)

    return HistoryEntry(
        tick=tick,
        gini=gini,
        total_output=total_output,
        mean_wealth=mean_wealth,
        median_wealth=median_wealth,
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
