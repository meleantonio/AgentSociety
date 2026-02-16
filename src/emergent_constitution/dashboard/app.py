"""Main Streamlit dashboard entry point.

Launch with:
    streamlit run src/emergent_constitution/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from emergent_constitution.dashboard import agent_tab, constitution_tab, economy_tab, params_tab
from emergent_constitution.dashboard.runner import run_simulation


def main() -> None:
    """Dashboard entry point."""
    st.set_page_config(
        page_title="Emergent Constitution Dashboard",
        layout="wide",
    )
    st.title("Emergent Constitution — Simulation Dashboard")

    # Initialize session state
    if "sim_output" not in st.session_state:
        st.session_state["sim_output"] = None
    if "sim_config" not in st.session_state:
        st.session_state["sim_config"] = None

    tab_params, tab_economy, tab_constitution, tab_agent = st.tabs(
        ["Parameters", "Economy", "Constitution", "Agent Explorer"]
    )

    # Create placeholders inside each tab for streaming updates
    with tab_economy:
        economy_ph = st.empty()
    with tab_constitution:
        constitution_ph = st.empty()
    with tab_agent:
        agent_ph = st.empty()

    # Params tab only validates config and sets run_requested flag
    with tab_params:
        params_tab.render()

    # Run simulation if requested (after params_tab sets session state)
    if st.session_state.get("run_requested"):
        st.session_state["run_requested"] = False
        config = st.session_state["sim_config"]
        output = run_simulation(config, economy_ph, constitution_ph, agent_ph)
        st.session_state["sim_output"] = output
        # Rerun so the final render below happens on a clean script execution,
        # avoiding duplicate widget keys from the streaming callback.
        st.rerun()

    # Render final state (or show placeholder message)
    output = st.session_state.get("sim_output")
    if output is not None:
        with economy_ph.container():
            economy_tab.render(output)
        with constitution_ph.container():
            constitution_tab.render(output)
        with agent_ph.container():
            agent_tab.render(output)
    else:
        for ph in (economy_ph, constitution_ph, agent_ph):
            with ph.container():
                st.info("Run a simulation first from the Parameters tab.")


if __name__ == "__main__":
    main()
