"""Integration test: verify no StreamlitDuplicateElementKey during streaming.

Reproduces the exact code path from the dashboard runner: calling
render_streaming / render multiple times within a single Streamlit script
run (simulating successive _on_observe callbacks) inside placeholder
containers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


_TEST_SCRIPT = textwrap.dedent("""\
    import streamlit as st
    from emergent_constitution.config import SimulationConfigV2
    from emergent_constitution.lead import LeadV2
    from emergent_constitution.dashboard import economy_tab, constitution_tab, agent_tab

    config = SimulationConfigV2(
        num_agents=20,
        max_periods=15,
        seed=42,
        benchmark_mode=True,
        record_agent_snapshots=True,
        observer_interval=5,
    )
    lead = LeadV2(config)

    economy_ph = st.empty()
    constitution_ph = st.empty()
    agent_ph = st.empty()

    def _on_observe(partial_output):
        ks = f"_s{len(partial_output.history)}"
        with economy_ph.container():
            economy_tab.render(partial_output, key_suffix=ks)
        with constitution_ph.container():
            constitution_tab.render_streaming(partial_output)
        with agent_ph.container():
            agent_tab.render_streaming(partial_output)

    output = lead.run(observe_callback=_on_observe)
    st.write("SUCCESS")
""")


@pytest.mark.slow
class TestStreamingDuplicateKeys:
    """Verify plotly_chart keys don't collide across streaming callbacks."""

    def test_no_duplicate_key_error(self, tmp_path: Path) -> None:
        """Running the streaming flow should not raise StreamlitDuplicateElementKey."""
        script = tmp_path / "app.py"
        script.write_text(_TEST_SCRIPT)

        at = AppTest.from_file(str(script))
        at.run(timeout=120)

        # AppTest captures exceptions from the script run
        assert not at.exception, (
            f"Streaming raised an exception: {at.exception[0].value}"
        )
