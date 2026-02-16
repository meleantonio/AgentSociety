"""Reporter — human-readable simulation report generation.

v1 functions (generate_report, etc.) retained for backward compatibility.

v2 functions (generate_report_v2, generate_json_v2) handle SimulationOutputV2
with DSGE-HA data: firm summary, welfare summary, expanded statistics,
wealth quantiles, and full JSON serialization.

All functions are pure: they take SimulationOutput and return strings
without performing any I/O.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from emergent_constitution.citizen import compute_gini
from emergent_constitution.coalition import compute_coalition_stats
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution, ConstitutionV2
from emergent_constitution.models.history import (
    HistoryEntry,
    HistoryEntryV2,
    SimulationOutput,
    SimulationOutputV2,
    WelfareSummary,
)
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.observer import compute_pareto_efficiency

if TYPE_CHECKING:
    from emergent_constitution.calibration import CalibrationTargets


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
        _format_coalition_summary(output.final_agent_states),
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


def _format_final_constitution(constitution: Constitution) -> str:
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


def _format_wealth_distribution(agents: list[AgentState]) -> str:
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
    pareto_score = compute_pareto_efficiency(agents)
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
        f"- **Pareto efficiency:** {pareto_score:.4f}",
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


def _format_constitutional_timeline(history: list[HistoryEntry]) -> str:
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


def _format_coalition_summary(agents: list[AgentState]) -> str:
    """Format a summary of coalition sizes and compositions.

    Args:
        agents: List of final AgentState objects.

    Returns:
        Markdown section with coalition statistics table.
    """
    stats = compute_coalition_stats(agents)
    if not stats:
        return "## Coalition Summary\n\nNo coalitions formed."

    lines = [
        "## Coalition Summary\n",
        f"**Number of coalitions:** {len(stats)}\n",
        "| Coalition | Size | Mean Wealth | Mean Equality |",
        "| --- | --- | --- | --- |",
    ]

    for info in stats.values():
        lines.append(
            f"| {info.coalition_id} "
            f"| {info.size} "
            f"| {info.mean_wealth:.2f} "
            f"| {info.mean_equality:.4f} |"
        )

    return "\n".join(lines)


def _format_statistics_evolution(history: list[HistoryEntry]) -> str:
    """Format a table showing how key statistics evolved over time.

    Args:
        history: List of HistoryEntry objects.

    Returns:
        Markdown table of Gini, Pareto score, total output, mean/median wealth,
        and coalition count per observation tick.
    """
    if not history:
        return "## Statistics Over Time\n\nNo observations recorded."

    lines = [
        "## Statistics Over Time\n",
        "| Tick | Gini | Pareto | Total Output | Mean Wealth | Median Wealth | Coalitions |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for entry in history:
        lines.append(
            f"| {entry.tick} "
            f"| {entry.gini:.4f} "
            f"| {entry.pareto_score:.4f} "
            f"| {entry.total_output:.2f} "
            f"| {entry.mean_wealth:.2f} "
            f"| {entry.median_wealth:.2f} "
            f"| {entry.num_coalitions} |"
        )

    return "\n".join(lines)


# ============================================================================
# v2 Reporter (DSGE-HA output)
# ============================================================================


def generate_report_v2(
    output: SimulationOutputV2,
    calibration_targets: CalibrationTargets | None = None,
) -> str:
    """Generate a full Markdown report from v2 simulation output.

    Args:
        output: Complete v2 simulation output.
        calibration_targets: Optional calibration targets for moment comparison.

    Returns:
        Human-readable Markdown report string.
    """
    sections = [
        _format_simulation_summary_v2(output),
        _format_final_constitution_v2(output.constitution),
        _format_wealth_distribution_v2(output.final_households),
        _format_firm_summary(output),
        _format_welfare_summary(output.welfare_summary),
    ]

    if calibration_targets is not None:
        sections.append(
            _format_calibration_comparison(output.final_households, calibration_targets)
        )

    sections.extend(
        [
            _format_constitutional_timeline_v2(output.history),
            _format_statistics_evolution_v2(output.history),
        ]
    )
    return "\n\n".join(sections) + "\n"


def generate_json_v2(output: SimulationOutputV2) -> str:
    """Serialize v2 simulation output to JSON.

    Args:
        output: Complete v2 simulation output.

    Returns:
        JSON string with all v2 data (constitution, history, households,
        firms, welfare summary, seed, total_periods).
    """
    return output.model_dump_json(indent=2)


def _format_simulation_summary_v2(output: SimulationOutputV2) -> str:
    """Format the header section with v2 simulation metadata.

    Args:
        output: Complete v2 simulation output.

    Returns:
        Markdown header with seed, periods, household count, firm count.
    """
    return (
        "# The Emergent Constitution — Simulation Report (v2)\n\n"
        f"- **Seed:** {output.seed}\n"
        f"- **Total periods:** {output.total_periods}\n"
        f"- **Number of households:** {len(output.final_households)}\n"
        f"- **Number of firms:** {len(output.final_firms)}"
    )


def _format_final_constitution_v2(constitution: ConstitutionV2) -> str:
    """Format the final v2 constitutional rules as a table.

    Args:
        constitution: The final ConstitutionV2 object.

    Returns:
        Markdown section listing all active rules with type and parameters.
    """
    lines = [
        "## Final Constitution\n",
        f"**Voting rule:** {constitution.voting_rule}\n",
        "| Rule | Type | Parameters |",
        "| --- | --- | --- |",
    ]

    for name, rule in sorted(constitution.rules.items()):
        params_str = ", ".join(f"{k}={v}" for k, v in rule.parameters.items())
        lines.append(f"| {name} | {rule.rule_type} | {params_str} |")

    return "\n".join(lines)


def _format_wealth_distribution_v2(households: list[HouseholdState]) -> str:
    """Format wealth distribution statistics for v2 households.

    Args:
        households: List of final HouseholdState objects.

    Returns:
        Markdown section with distribution stats, quantiles, and quintiles.
    """
    if not households:
        return "## Wealth Distribution\n\nNo households in simulation."

    wealths = [h.wealth for h in households]
    n = len(wealths)
    mean = statistics.mean(wealths)
    median = statistics.median(wealths)
    stdev = statistics.stdev(wealths) if n >= 2 else "N/A"
    stdev_str = f"{stdev:.2f}" if isinstance(stdev, float) else stdev

    # Compute Gini using same algorithm as ObserverV2
    sorted_w = sorted(wealths)
    total = sum(sorted_w)
    if total > 0 and n >= 2:
        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_w))
        gini = (2.0 * weighted_sum - (n + 1) * total) / (n * total)
    else:
        gini = 0.0

    lines = [
        "## Wealth Distribution\n",
        f"- **Min:** {min(wealths):.2f}",
        f"- **Max:** {max(wealths):.2f}",
        f"- **Mean:** {mean:.2f}",
        f"- **Median:** {median:.2f}",
        f"- **Std dev:** {stdev_str}",
        f"- **Gini coefficient:** {gini:.4f}",
    ]

    # Wealth quantiles [p10, p25, p50, p75, p90]
    if n >= 5:
        quantile_labels = ["p10", "p25", "p50", "p75", "p90"]
        quantile_pcts = [0.10, 0.25, 0.50, 0.75, 0.90]
        lines.append("\n### Wealth Quantiles\n")
        lines.append("| Quantile | Value |")
        lines.append("| --- | --- |")
        for label, p in zip(quantile_labels, quantile_pcts, strict=True):
            idx = min(int(p * n), n - 1)
            lines.append(f"| {label} | {sorted_w[idx]:.2f} |")

    # Quintile breakdown
    if n >= 5:
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


def _format_firm_summary(output: SimulationOutputV2) -> str:
    """Format a summary of active firms.

    Args:
        output: Complete v2 simulation output.

    Returns:
        Markdown section with firm count, mean size, total R&D, and table.
    """
    firms = output.final_firms
    if not firms:
        return "## Firm Summary\n\nNo active firms at end of simulation."

    num_firms = len(firms)
    total_workers = sum(len(f.worker_ids) for f in firms)
    mean_size = total_workers / num_firms
    total_rd = sum(f.rd_spend for f in firms)
    total_output = sum(f.output for f in firms)
    total_profit = sum(f.profit for f in firms)

    lines = [
        "## Firm Summary\n",
        f"- **Number of firms:** {num_firms}",
        f"- **Total workers employed:** {total_workers}",
        f"- **Mean firm size:** {mean_size:.1f}",
        f"- **Total R&D spending:** {total_rd:.2f}",
        f"- **Total firm output:** {total_output:.2f}",
        f"- **Total firm profit:** {total_profit:.2f}",
        "",
        "| Firm | Owner | Workers | Capital | TFP | Output | Profit | R&D |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for f in firms:
        lines.append(
            f"| {f.id} | {f.owner_id} | {len(f.worker_ids)} "
            f"| {f.capital:.2f} | {f.tfp:.4f} "
            f"| {f.output:.2f} | {f.profit:.2f} | {f.rd_spend:.2f} |"
        )

    return "\n".join(lines)


def _format_welfare_summary(welfare: WelfareSummary) -> str:
    """Format the welfare comparison summary.

    Args:
        welfare: WelfareSummary with LLM and optional benchmark totals.

    Returns:
        Markdown section with welfare comparison.
    """
    lines = [
        "## Welfare Summary\n",
        f"- **LLM total welfare:** {welfare.llm_total_welfare:.4f}",
    ]

    if welfare.benchmark_total_welfare is not None:
        lines.append(f"- **Benchmark total welfare:** {welfare.benchmark_total_welfare:.4f}")
        if welfare.benchmark_total_welfare > 0:
            ratio = welfare.llm_total_welfare / welfare.benchmark_total_welfare
            lines.append(f"- **LLM / Benchmark ratio:** {ratio:.4f}")

    if welfare.per_agent_comparison is not None:
        lines.append("\n### Per-Agent Comparison\n")
        lines.append("| Agent | LLM Welfare | Benchmark Welfare |")
        lines.append("| --- | --- | --- |")
        for agent_id, data in sorted(welfare.per_agent_comparison.items()):
            llm_w = data.get("llm", 0.0)
            bench_w = data.get("benchmark", 0.0)
            lines.append(f"| {agent_id} | {llm_w:.4f} | {bench_w:.4f} |")

    return "\n".join(lines)


def _format_constitutional_timeline_v2(history: list[HistoryEntryV2]) -> str:
    """Format a detailed chronological timeline of all governance activity.

    Shows every proposal, vote outcome, and constitutional modification
    for every period, plus the constitution state after changes.

    Args:
        history: List of HistoryEntryV2 objects.

    Returns:
        Markdown section with full governance detail by period.
    """
    if not history:
        return "## Constitutional Timeline\n\nNo observations recorded."

    any_activity = False
    lines = ["## Constitutional Timeline\n"]

    for entry in history:
        has_proposals = bool(entry.proposals)
        has_votes = bool(entry.votes)
        has_changes = bool(entry.rule_changes)

        if not has_proposals and not has_votes and not has_changes:
            continue

        any_activity = True
        lines.append(f"### Period {entry.period}\n")

        # --- Proposals submitted ---
        if has_proposals:
            lines.append(f"**Proposals submitted ({len(entry.proposals)}):**\n")
            for i, prop in enumerate(entry.proposals, 1):
                params_str = ""
                if prop.parameters:
                    params_str = ", ".join(f"{k}={v}" for k, v in prop.parameters.items())
                desc = f" — *{prop.description}*" if prop.description else ""
                lines.append(
                    f"{i}. **{prop.action}** `{prop.rule_name}`"
                    f" [{params_str}] by agent `{prop.proposer_id}`{desc}"
                )
            lines.append("")

        # --- Vote outcomes ---
        if has_votes:
            lines.append(f"**Votes ({len(entry.votes)}):**\n")
            lines.append("| # | Action | Rule | Result | For | Against | Total | Voting Rule |")
            lines.append("| --- " * 8 + "|")
            for i, vote in enumerate(entry.votes, 1):
                p = vote.proposal
                result = "PASSED" if vote.passed else "REJECTED"
                params_str = ""
                if p.parameters:
                    params_str = " " + ", ".join(f"{k}={v}" for k, v in p.parameters.items())
                lines.append(
                    f"| {i} | {p.action} | `{p.rule_name}`{params_str}"
                    f" | **{result}** | {vote.votes_for}"
                    f" | {vote.votes_against} | {vote.total_eligible}"
                    f" | {vote.voting_rule_used} |"
                )
            lines.append("")

        # --- Rule changes applied ---
        if has_changes:
            lines.append("**Changes applied:**\n")
            for change in entry.rule_changes:
                lines.append(f"- {change}")
            lines.append("")

        # --- Constitution state after changes ---
        if has_changes:
            const = entry.constitution_snapshot
            lines.append(
                f"**Constitution after period {entry.period}:** voting={const.voting_rule}\n"
            )
            lines.append("| Rule | Type | Parameters |")
            lines.append("| --- | --- | --- |")
            for name, rule in sorted(const.rules.items()):
                params_str = ", ".join(f"{k}={v}" for k, v in rule.parameters.items())
                lines.append(f"| {name} | {rule.rule_type} | {params_str} |")
            lines.append("")

    if not any_activity:
        lines.append("No governance activity during the simulation.")

    return "\n".join(lines)


def _format_calibration_comparison(
    households: list[HouseholdState],
    targets: CalibrationTargets,
) -> str:
    """Format a calibration comparison section showing model vs target moments.

    Args:
        households: Final household states.
        targets: Calibration target values.

    Returns:
        Markdown section with moment comparison table.
    """
    from emergent_constitution.calibration import Calibrator

    calibrator = Calibrator()
    moments = calibrator.compute_model_moments(households)

    lines = [
        "## Calibration Comparison\n",
        calibrator.format_comparison(moments, targets),
    ]

    return "\n".join(lines)


def _format_statistics_evolution_v2(history: list[HistoryEntryV2]) -> str:
    """Format a table showing how key statistics evolved over time.

    Args:
        history: List of HistoryEntryV2 objects.

    Returns:
        Markdown table with all key DSGE-HA statistics per period.
    """
    if not history:
        return "## Statistics Over Time\n\nNo observations recorded."

    lines = [
        "## Statistics Over Time\n",
        (
            "| Period | Gini | Pareto | Output | Consumption"
            " | Investment | Mean Wealth | Median Wealth"
            " | Unemployment | Firms | Welfare | Cum. Welfare"
            " | Wage | Rate |"
        ),
        "| --- " * 14 + "|",
    ]

    for e in history:
        lines.append(
            f"| {e.period} "
            f"| {e.gini:.4f} "
            f"| {e.pareto_score:.4f} "
            f"| {e.aggregate_output:.2f} "
            f"| {e.aggregate_consumption:.2f} "
            f"| {e.aggregate_investment:.2f} "
            f"| {e.mean_wealth:.2f} "
            f"| {e.median_wealth:.2f} "
            f"| {e.unemployment_rate:.4f} "
            f"| {e.num_active_firms} "
            f"| {e.social_welfare:.2f} "
            f"| {e.cumulative_welfare:.2f} "
            f"| {e.wage:.4f} "
            f"| {e.interest_rate:.4f} |"
        )

    return "\n".join(lines)
