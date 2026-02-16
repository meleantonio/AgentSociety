"""Integration test for dashboard runner.

Runs a fast simulation via the runner module (without Streamlit UI)
and verifies the output has snapshots populated.
"""

from __future__ import annotations

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.lead import LeadV2


@pytest.mark.slow
class TestDashboardRunner:
    """Integration tests for the simulation runner used by the dashboard."""

    def test_run_with_snapshots(self) -> None:
        """Running with record_agent_snapshots=True populates history snapshots."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=10,
            seed=42,
            benchmark_mode=True,
            record_agent_snapshots=True,
            observer_interval=5,
        )
        lead = LeadV2(config)
        output = lead.run()

        # Should have at least one observation with snapshots
        entries_with_snapshots = [e for e in output.history if e.household_snapshots is not None]
        assert len(entries_with_snapshots) > 0

        # Each snapshot should have the right number of agents
        for entry in entries_with_snapshots:
            assert len(entry.household_snapshots) == 20

    def test_progress_callback_called(self) -> None:
        """Progress callback is invoked for each period."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=10,
            seed=42,
            benchmark_mode=True,
        )
        calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            calls.append((current, total))

        lead = LeadV2(config)
        lead.run(progress_callback=callback)

        assert len(calls) == 10
        assert calls[0] == (1, 10)
        assert calls[-1] == (10, 10)

    def test_run_without_callback(self) -> None:
        """Default run() still works without callback."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=5,
            seed=42,
            benchmark_mode=True,
        )
        lead = LeadV2(config)
        output = lead.run()
        assert output.total_periods == 5
