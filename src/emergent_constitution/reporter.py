"""Reporter — human-readable simulation report generation.

All functions are pure: they take SimulationOutput and return strings
without performing any I/O.
"""

from __future__ import annotations

import statistics

from emergent_constitution.citizen import compute_gini
from emergent_constitution.models.history import SimulationOutput


def generate_report(output: SimulationOutput) -> str:
    """Generate a full Markdown report from simulation output.

    Args:
        output: Complete simulation output.

    Returns:
        Human-readable Markdown report string.
    """
    sections = [
        _format_simulation_summary(output),
        _format_final_constitution(output.constitution),
        _format_wealth_distribution(output.final_agent_states),
        _format_constitutional_timeline(output.history),
        _format_statistics_evolution(output.history),
    ]
    return "\n\n".join(sections) + "\n"


def _format_simulation_summary(output: SimulationOutput) -> str:
    """Format the header section with simulation metadata.

    Args:
        output: Complete simulation output.

    Returns:
        Markdown header with seed, tick count, and agent count.
    """
    return (
        "# The Emergent Constitution — Simulation Report\n\n"
        f"- **Seed:** {output.seed}\n"
        f"- **Total ticks:** {output.total_ticks}\n"
        f"- **Number of agents:** {len(output.final_agent_states)}"
    )


def _format_final_constitution(constitution) -> str:
    """Format the final constitutional rules as a table.

    Args:
        constitution: The final Constitution object.

    Returns:
        Markdown section with all four rules.
    """
    tax_pct = f"{constitution.tax_rate:.0%}"
    return (
        "## Final Constitution\n\n"
        "| Rule | Value |\n"
        "| --- | --- |\n"
        f"| Property rule | {constitution.property_rule.value} |\n"
        f"| Tax rate | {constitution.tax_rate:.4f} ({tax_pct}) |\n"
        f"| Voting rule | {constitution.voting_rule.value} |\n"
        f"| Redistribution rule | {constitution.redistribution_rule.value} |"
    )


def _format_wealth_distribution(agents: list) -> str:
    """Format wealth distribution statistics and quintile breakdown.

    Args:
        agents: List of final AgentState objects.

    Returns:
        Markdown section with distribution stats and quintile table.
    """
    if not agents:
        return "## Wealth Distribution\n\nNo agents in simulation."

    wealths = [a.wealth for a in agents]
    n = len(wealths)
    gini = compute_gini(wealths)
    mean = statistics.mean(wealths)
    median = statistics.median(wealths)
    stdev = statistics.stdev(wealths) if n >= 2 else "N/A"

    stdev_str = f"{stdev:.2f}" if isinstance(stdev, float) else stdev

    lines = [
        "## Wealth Distribution\n",
        f"- **Min:** {min(wealths):.2f}",
        f"- **Max:** {max(wealths):.2f}",
        f"- **Mean:** {mean:.2f}",
        f"- **Median:** {median:.2f}",
        f"- **Std dev:** {stdev_str}",
        f"- **Gini coefficient:** {gini:.4f}",
    ]

    # Quintile breakdown (only if enough agents)
    if n >= 5:
        sorted_w = sorted(wealths)
        q_size = n // 5
        lines.append("\n### Quintile Breakdown\n")
        lines.append("| Quintile | Mean Wealth |")
        lines.append("| --- | --- |")
        for q in range(5):
            start = q * q_size
            end = (q + 1) * q_size if q < 4 else n
            q_mean = statistics.mean(sorted_w[start:end])
            label = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"][q]
            lines.append(f"| {label} | {q_mean:.2f} |")

    return "\n".join(lines)


def _format_constitutional_timeline(history: list) -> str:
    """Format a chronological timeline of constitutional changes.

    Args:
        history: List of HistoryEntry objects.

    Returns:
        Markdown section with rule changes by tick, or "No changes" message.
    """
    if not history:
        return "## Constitutional Timeline\n\nNo observations recorded."

    changes_found = False
    lines = ["## Constitutional Timeline\n"]

    for entry in history:
        if entry.rule_changes:
            changes_found = True
            lines.append(f"**Tick {entry.tick}:**")
            for change in entry.rule_changes:
                lines.append(f"- {change}")

    if not changes_found:
        lines.append("No constitutional changes during the simulation.")

    return "\n".join(lines)


def _format_statistics_evolution(history: list) -> str:
    """Format a table showing how key statistics evolved over time.

    Args:
        history: List of HistoryEntry objects.

    Returns:
        Markdown table of Gini, total output, mean/median wealth per observation tick.
    """
    if not history:
        return "## Statistics Over Time\n\nNo observations recorded."

    lines = [
        "## Statistics Over Time\n",
        "| Tick | Gini | Total Output | Mean Wealth | Median Wealth |",
        "| --- | --- | --- | --- | --- |",
    ]

    for entry in history:
        lines.append(
            f"| {entry.tick} "
            f"| {entry.gini:.4f} "
            f"| {entry.total_output:.2f} "
            f"| {entry.mean_wealth:.2f} "
            f"| {entry.median_wealth:.2f} |"
        )

    return "\n".join(lines)
