"""Tab 4: Per-agent explorer with comparison charts."""

from __future__ import annotations

import statistics

import streamlit as st

from emergent_constitution.dashboard.charts import make_time_series
from emergent_constitution.models.history import SimulationOutputV2
from emergent_constitution.models.household import HouseholdState

_CHART_CONFIG = [
    ("wealth", "Wealth", "Wealth"),
    ("consumption", "Consumption", "Consumption"),
    ("income", "Income", "Income"),
    ("labor_supply", "Labor Supply", "Labor Supply"),
    ("realized_utility", "Realized Utility", "Utility"),
    ("productivity", "Productivity", "Productivity"),
    ("taxes_paid", "Taxes Paid", "Taxes"),
    ("transfers_received", "Transfers Received", "Transfers"),
    ("savings", "Savings", "Savings"),
    ("leisure", "Leisure", "Leisure"),
]


def _extract_agent_series(
    output: SimulationOutputV2,
    agent_ids: list[str],
    include_avg: bool,
) -> dict[str, dict[str, list[float]]]:
    """Build per-metric time series for selected agents.

    Args:
        output: Simulation output with household_snapshots.
        agent_ids: Agent IDs to extract.
        include_avg: Whether to add a population average trace.

    Returns:
        Mapping of metric_name -> {trace_name: [values]}.
    """
    metrics = [
        "wealth",
        "consumption",
        "income",
        "labor_supply",
        "leisure",
        "realized_utility",
        "productivity",
        "taxes_paid",
        "transfers_received",
        "savings",
    ]
    series: dict[str, dict[str, list[float]]] = {m: {} for m in metrics}

    # Initialize traces
    for m in metrics:
        for aid in agent_ids:
            series[m][aid] = []
        if include_avg:
            series[m]["Population Average"] = []

    for entry in output.history:
        if entry.household_snapshots is None:
            # No snapshot data for this entry
            for m in metrics:
                for aid in agent_ids:
                    series[m][aid].append(0.0)
                if include_avg:
                    series[m]["Population Average"].append(0.0)
            continue

        # Index households by id for fast lookup
        hh_map: dict[str, HouseholdState] = {h.id: h for h in entry.household_snapshots}

        for m in metrics:
            for aid in agent_ids:
                hh = hh_map.get(aid)
                series[m][aid].append(getattr(hh, m, 0.0) if hh else 0.0)

            if include_avg and entry.household_snapshots:
                vals = [getattr(h, m, 0.0) for h in entry.household_snapshots]
                series[m]["Population Average"].append(statistics.mean(vals) if vals else 0.0)

    return series


def render_streaming(output: SimulationOutputV2) -> None:
    """Widget-free render for streaming updates during simulation."""
    has_snapshots = any(e.household_snapshots is not None for e in output.history)
    if not has_snapshots:
        st.info("Agent snapshots will appear as the simulation progresses...")
        return

    hh = output.final_households
    m1, m2, m3 = st.columns(3)
    m1.metric("Agents", len(hh))
    m2.metric("Mean Wealth", f"{statistics.mean(h.wealth for h in hh):.2f}")
    m3.metric("Mean Consumption", f"{statistics.mean(h.consumption for h in hh):.4f}")

    # Show first 3 agents + population average (no selection widget)
    agent_ids = sorted(h.id for h in hh)[:3]
    periods = [e.period for e in output.history]
    all_series = _extract_agent_series(output, agent_ids, include_avg=True)

    # Show first 3 chart pairs (6 charts)
    ks = f"_s{len(output.history)}"
    for i in range(0, min(6, len(_CHART_CONFIG)), 2):
        col_l, col_r = st.columns(2)
        metric, title, ylabel = _CHART_CONFIG[i]
        with col_l:
            st.plotly_chart(
                make_time_series(all_series[metric], periods, title, ylabel),
                width="stretch",
                key=f"agent_{metric}{ks}",
            )
        if i + 1 < len(_CHART_CONFIG):
            metric2, title2, ylabel2 = _CHART_CONFIG[i + 1]
            with col_r:
                st.plotly_chart(
                    make_time_series(all_series[metric2], periods, title2, ylabel2),
                    width="stretch",
                    key=f"agent_{metric2}{ks}",
                )


def render(output: SimulationOutputV2) -> None:
    """Render agent explorer tab.

    Args:
        output: Completed simulation output.
    """
    # Check if snapshots are available
    has_snapshots = any(e.household_snapshots is not None for e in output.history)
    if not has_snapshots:
        st.warning(
            "No agent snapshot data available. "
            "Re-run the simulation with `record_agent_snapshots=True` to enable this tab."
        )
        return

    # --- Controls ---
    all_ids = sorted(h.id for h in output.final_households)
    selected_ids = st.multiselect(
        "Select agents to explore",
        all_ids,
        default=all_ids[:2] if len(all_ids) >= 2 else all_ids,
        key="agent_explorer_ids",
    )
    include_avg = st.checkbox("Compare to population average", value=True, key="agent_avg")

    if not selected_ids:
        st.info("Select at least one agent to see charts.")
        return

    periods = [e.period for e in output.history]
    all_series = _extract_agent_series(output, selected_ids, include_avg)

    # --- Per-agent time series charts (2-column layout) ---
    for i in range(0, len(_CHART_CONFIG), 2):
        col_l, col_r = st.columns(2)
        metric, title, ylabel = _CHART_CONFIG[i]
        with col_l:
            st.plotly_chart(
                make_time_series(all_series[metric], periods, title, ylabel),
                width="stretch",
            )
        if i + 1 < len(_CHART_CONFIG):
            metric2, title2, ylabel2 = _CHART_CONFIG[i + 1]
            with col_r:
                st.plotly_chart(
                    make_time_series(all_series[metric2], periods, title2, ylabel2),
                    width="stretch",
                )

    # --- Agent Comparison Table ---
    st.subheader("Agent Comparison (Final Period)")
    comparison_rows = []
    hh_map = {h.id: h for h in output.final_households}
    for aid in selected_ids:
        hh = hh_map.get(aid)
        if hh is None:
            continue
        comparison_rows.append(
            {
                "Agent": hh.id,
                "Role": hh.role.value,
                "Wealth": round(hh.wealth, 2),
                "Consumption": round(hh.consumption, 4),
                "Income": round(hh.income, 4),
                "Utility": round(hh.realized_utility, 4),
                "alpha": hh.utility_params.alpha,
                "beta": hh.utility_params.beta,
                "gamma": hh.utility_params.gamma,
                "Equality": hh.value_vector.equality,
                "Liberty": hh.value_vector.liberty,
            }
        )
    if comparison_rows:
        st.dataframe(comparison_rows, width="stretch", hide_index=True)
