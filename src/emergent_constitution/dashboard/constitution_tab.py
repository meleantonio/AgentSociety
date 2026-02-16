"""Tab 3: Constitution evolution, proposals, and votes."""

from __future__ import annotations

import streamlit as st

from emergent_constitution.dashboard.charts import (
    make_bar_chart,
    make_scatter_timeline,
    make_time_series,
)
from emergent_constitution.models.history import HistoryEntryV2, SimulationOutputV2


def _render_evolution_charts(history: list[HistoryEntryV2], periods: list[int]) -> None:
    """Render the 4 constitution evolution charts (widget-free)."""
    st.subheader("Constitution Evolution")

    row1_l, row1_r = st.columns(2)

    with row1_l:
        # Constitutional timeline from rule_changes
        event_periods: list[int] = []
        event_labels: list[str] = []
        event_colors: list[str] = []
        event_hovers: list[str] = []
        color_map = {"add": "#00CC96", "modify": "#FFA15A", "remove": "#EF553B"}

        for entry in history:
            for change_text in entry.rule_changes:
                event_periods.append(entry.period)
                lower = change_text.lower()
                if "add" in lower:
                    action = "add"
                elif "modif" in lower:
                    action = "modify"
                elif "remov" in lower:
                    action = "remove"
                else:
                    action = "modify"
                event_labels.append(action[0].upper())
                event_colors.append(color_map.get(action, "#636EFA"))
                event_hovers.append(change_text)

        if event_periods:
            st.plotly_chart(
                make_scatter_timeline(
                    event_periods,
                    event_labels,
                    event_colors,
                    event_hovers,
                    "Constitutional Timeline",
                ),
                width="stretch",
            )
        else:
            st.info("No constitutional changes during the simulation.")

    with row1_r:
        # Tax rate evolution from constitution snapshots
        tax_rates: list[float] = []
        for entry in history:
            rate = 0.0
            for rule in entry.constitution_snapshot.rules.values():
                if rule.rule_type == "tax_schedule" and rule.parameters:
                    rate = rule.parameters.get("rate", rate)
            tax_rates.append(rate)

        if any(r > 0 for r in tax_rates):
            st.plotly_chart(
                make_time_series(
                    {"Tax Rate": tax_rates},
                    periods,
                    "Tax Rate Evolution",
                    "Rate",
                ),
                width="stretch",
            )
        else:
            st.info("No tax rules found in constitution snapshots.")

    row2_l, row2_r = st.columns(2)

    with row2_l:
        # Governance activity: proposals submitted vs passed per period
        submitted_counts: list[int] = []
        passed_counts: list[int] = []
        for entry in history:
            submitted_counts.append(len(entry.proposals))
            passed_counts.append(sum(1 for v in entry.votes if v.passed))
        st.plotly_chart(
            make_bar_chart(
                periods,
                {"Submitted": submitted_counts, "Passed": passed_counts},
                "Governance Activity",
                "Count",
            ),
            width="stretch",
        )

    with row2_r:
        # Rule count over time
        rule_counts = [len(entry.constitution_snapshot.rules) for entry in history]
        st.plotly_chart(
            make_time_series(
                {"Rule Count": [float(c) for c in rule_counts]},
                periods,
                "Rule Count Over Time",
                "Rules",
            ),
            width="stretch",
        )


def render_streaming(output: SimulationOutputV2) -> None:
    """Widget-free render for streaming updates during simulation."""
    history = output.history
    if not history:
        st.info("Waiting for first observation...")
        return
    periods = [e.period for e in history]

    # Summary metrics
    total_proposals = sum(len(e.proposals) for e in history)
    total_passed = sum(sum(1 for v in e.votes if v.passed) for e in history)
    current_rules = len(history[-1].constitution_snapshot.rules)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Proposals", total_proposals)
    m2.metric("Passed", total_passed)
    m3.metric("Active Rules", current_rules)

    _render_evolution_charts(history, periods)


def _build_proposals_table(history: list[HistoryEntryV2]) -> list[dict]:
    """Extract all proposals with their vote outcomes into flat rows."""
    rows: list[dict] = []
    for entry in history:
        # Build a lookup from rule_name+action to vote outcome
        vote_lookup: dict[tuple[str, str], bool] = {}
        for vote in entry.votes:
            key = (vote.proposal.rule_name, vote.proposal.action)
            vote_lookup[key] = vote.passed

        for prop in entry.proposals:
            key = (prop.rule_name, prop.action)
            passed = vote_lookup.get(key)
            result = "Accepted" if passed else ("Rejected" if passed is not None else "No Vote")
            params_str = str(prop.parameters) if prop.parameters else ""
            rows.append(
                {
                    "Period": entry.period,
                    "Proposer": prop.proposer_id,
                    "Action": prop.action,
                    "Rule Name": prop.rule_name,
                    "Rule Type": prop.rule_type or "",
                    "Parameters": params_str,
                    "Description": prop.description,
                    "Result": result,
                }
            )
    return rows


def _build_votes_table(history: list[HistoryEntryV2]) -> list[dict]:
    """Extract all vote outcomes into flat rows."""
    rows: list[dict] = []
    for entry in history:
        for vote in entry.votes:
            total = vote.total_eligible
            approval = (vote.votes_for / total * 100) if total > 0 else 0.0
            rows.append(
                {
                    "Period": entry.period,
                    "Proposal": f"{vote.proposal.rule_name} ({vote.proposal.action})",
                    "Result": "Passed" if vote.passed else "Rejected",
                    "Votes For": vote.votes_for,
                    "Votes Against": vote.votes_against,
                    "Total Eligible": total,
                    "Approval %": round(approval, 1),
                    "Voting Rule": vote.voting_rule_used,
                }
            )
    return rows


def render(output: SimulationOutputV2) -> None:
    """Render constitution tab from simulation output.

    Args:
        output: Completed simulation output.
    """
    history = output.history
    periods = [e.period for e in history]

    # --- Section A: Proposals Log ---
    st.subheader("Proposals Log")
    proposals_data = _build_proposals_table(history)
    if proposals_data:
        # Filters
        f1, f2, f3 = st.columns(3)
        all_actions = sorted({r["Action"] for r in proposals_data})
        with f1:
            if min(periods) < max(periods):
                period_range = st.slider(
                    "Period range",
                    min_value=min(periods),
                    max_value=max(periods),
                    value=(min(periods), max(periods)),
                    key="prop_period_range",
                )
            else:
                period_range = (min(periods), max(periods))
        with f2:
            action_filter = st.multiselect("Action", all_actions, default=all_actions)
        with f3:
            result_filter = st.radio("Result", ["All", "Accepted", "Rejected"], horizontal=True)

        filtered = [
            r
            for r in proposals_data
            if period_range[0] <= r["Period"] <= period_range[1]
            and r["Action"] in action_filter
            and (result_filter == "All" or r["Result"] == result_filter)
        ]
        st.dataframe(filtered, width="stretch", hide_index=True)
    else:
        st.info("No proposals were submitted during the simulation.")

    # --- Section B: Votes Log ---
    st.subheader("Votes Log")
    votes_data = _build_votes_table(history)
    if votes_data:
        st.dataframe(votes_data, width="stretch", hide_index=True)
    else:
        st.info("No votes were recorded during the simulation.")

    # --- Section C: Constitution Evolution Charts ---
    _render_evolution_charts(history, periods)

    # --- Section D: Constitution State Viewer ---
    st.subheader("Constitution State Viewer")

    if history:
        selected_period = st.selectbox(
            "Select observation period",
            periods,
            index=len(periods) - 1,
            key="const_viewer_period",
        )
        entry = next(e for e in history if e.period == selected_period)
        rules = entry.constitution_snapshot.rules

        if rules:
            rules_table = [
                {
                    "Name": r.name,
                    "Type": r.rule_type,
                    "Parameters": str(r.parameters) if r.parameters else "",
                    "Description": r.description,
                }
                for r in rules.values()
            ]
            st.dataframe(rules_table, width="stretch", hide_index=True)
        else:
            st.info("No rules in constitution at this period.")

        # Show changes from previous period
        if entry.rule_changes:
            st.markdown("**Changes this period:**")
            for change in entry.rule_changes:
                st.markdown(f"- {change}")
