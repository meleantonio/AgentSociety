"""Tab 2: Aggregate economy charts."""

from __future__ import annotations

import streamlit as st

from emergent_constitution.dashboard.charts import (
    make_dual_axis,
    make_fan_chart,
    make_time_series,
)
from emergent_constitution.models.history import SimulationOutputV2


def render(output: SimulationOutputV2, *, key_suffix: str = "") -> None:
    """Render economy charts from simulation output.

    Args:
        output: Completed simulation output.
        key_suffix: Suffix appended to plotly_chart keys for uniqueness
            across repeated calls within a single Streamlit script run.
    """
    history = output.history
    periods = [e.period for e in history]

    st.subheader("Macroeconomic Aggregates")

    # --- Section A: 2x2 grid ---
    row1_l, row1_r = st.columns(2)

    with row1_l:
        gdp_data = {
            "Output": [e.aggregate_output for e in history],
            "Consumption": [e.aggregate_consumption for e in history],
            "Investment": [e.aggregate_investment for e in history],
        }
        st.plotly_chart(
            make_time_series(gdp_data, periods, "GDP Components", "Value"),
            width="stretch",
            key=f"econ_gdp{key_suffix}",
        )

    with row1_r:
        st.plotly_chart(
            make_dual_axis(
                periods,
                left_data={"Wage": [e.wage for e in history]},
                right_data={"Interest Rate": [e.interest_rate for e in history]},
                title="Prices",
                left_label="Wage",
                right_label="Interest Rate",
            ),
            width="stretch",
            key=f"econ_prices{key_suffix}",
        )

    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.plotly_chart(
            make_time_series(
                {"Gini": [e.gini for e in history]},
                periods,
                "Gini Coefficient",
                "Gini",
            ),
            width="stretch",
            key=f"econ_gini{key_suffix}",
        )

    with row2_r:
        quantiles = [e.wealth_quantiles for e in history]
        st.plotly_chart(
            make_fan_chart(periods, quantiles, "Wealth Distribution (p10-p90)"),
            width="stretch",
            key=f"econ_wealth_dist{key_suffix}",
        )

    # --- Section B: Additional Metrics ---
    st.subheader("Additional Metrics")

    row3_l, row3_r = st.columns(2)

    with row3_l:
        welfare_data = {
            "Social Welfare": [e.social_welfare for e in history],
            "Cumulative Welfare": [e.cumulative_welfare for e in history],
        }
        st.plotly_chart(
            make_time_series(welfare_data, periods, "Welfare", "Utility"),
            width="stretch",
            key=f"econ_welfare{key_suffix}",
        )

    with row3_r:
        st.plotly_chart(
            make_time_series(
                {"Unemployment Rate": [e.unemployment_rate for e in history]},
                periods,
                "Unemployment Rate",
                "Rate",
            ),
            width="stretch",
            key=f"econ_unemployment{key_suffix}",
        )

    row4_l, row4_r = st.columns(2)

    with row4_l:
        st.plotly_chart(
            make_dual_axis(
                periods,
                left_data={"Active Firms": [float(e.num_active_firms) for e in history]},
                right_data={"Mean Firm Size": [e.mean_firm_size for e in history]},
                title="Firms",
                left_label="Count",
                right_label="Mean Size",
            ),
            width="stretch",
            key=f"econ_firms{key_suffix}",
        )

    with row4_r:
        st.plotly_chart(
            make_time_series(
                {"Pareto Score": [e.pareto_score for e in history]},
                periods,
                "Pareto Efficiency",
                "Score",
            ),
            width="stretch",
            key=f"econ_pareto{key_suffix}",
        )
