"""Simulation runner wrapper for the dashboard with progress bar and streaming support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.dashboard import agent_tab, constitution_tab, economy_tab
from emergent_constitution.lead import LeadV2
from emergent_constitution.models.history import SimulationOutputV2

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


def run_simulation(
    config: SimulationConfigV2,
    economy_placeholder: DeltaGenerator | None = None,
    constitution_placeholder: DeltaGenerator | None = None,
    agent_placeholder: DeltaGenerator | None = None,
) -> SimulationOutputV2:
    """Run a simulation with a Streamlit progress bar and streaming tab updates.

    Forces ``record_agent_snapshots=True`` so the Agent Explorer tab can
    display per-agent time series.

    When placeholder containers are provided, partial results are rendered
    into them at each observer interval so the user sees live updates.

    Args:
        config: Simulation configuration (will be copied with snapshots enabled).
        economy_placeholder: Optional st.empty() for streaming economy tab updates.
        constitution_placeholder: Optional st.empty() for streaming constitution tab updates.
        agent_placeholder: Optional st.empty() for streaming agent tab updates.

    Returns:
        SimulationOutputV2 with household_snapshots populated in history entries.
    """
    config = config.model_copy(update={"record_agent_snapshots": True})

    progress_bar = st.progress(0, text="Initializing simulation...")

    def _on_progress(current: int, total: int) -> None:
        frac = current / total
        progress_bar.progress(frac, text=f"Period {current}/{total}")

    def _on_observe(partial_output: SimulationOutputV2) -> None:
        if economy_placeholder is not None:
            with economy_placeholder.container():
                economy_tab.render(partial_output)
        if constitution_placeholder is not None:
            with constitution_placeholder.container():
                constitution_tab.render(partial_output)
        if agent_placeholder is not None:
            with agent_placeholder.container():
                agent_tab.render(partial_output)

    lead = LeadV2(config)
    output = lead.run(
        progress_callback=_on_progress,
        observe_callback=_on_observe,
    )

    progress_bar.empty()
    return output
