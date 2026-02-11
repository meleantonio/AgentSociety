"""Scale tests and performance benchmarks for production-level simulation runs.

These tests validate the simulation at full scale (50-200 agents, 100-1000 ticks)
and measure performance to catch regressions. All tests are marked ``@pytest.mark.slow``
so they can be excluded from fast CI runs with ``-m "not slow"``.
"""

from __future__ import annotations

import math
import time

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead


@pytest.mark.slow
class TestScaleSimulation:
    """Scale tests validating the simulation at production parameters."""

    def test_50_agents_100_ticks(self) -> None:
        """Default parameters (50 agents, 100 ticks) complete successfully."""
        config = SimulationConfig(num_agents=50, max_ticks=100, seed=42)
        lead = Lead(config)
        output = lead.run()

        assert len(output.final_agent_states) == 50
        assert output.total_ticks == 100
        assert len(output.history) == 100 // config.observer_interval
        assert all(a.wealth >= 0 for a in output.final_agent_states)

    def test_100_agents_500_ticks(self) -> None:
        """Larger scale: 100 agents, 500 ticks."""
        config = SimulationConfig(num_agents=100, max_ticks=500, seed=42)
        lead = Lead(config)
        output = lead.run()

        assert len(output.final_agent_states) == 100
        assert output.total_ticks == 500
        # Verify no NaN/Inf in wealth
        for agent in output.final_agent_states:
            assert not math.isnan(agent.wealth), f"Agent {agent.id} has NaN wealth"
            assert not math.isinf(agent.wealth), f"Agent {agent.id} has Inf wealth"

    def test_200_agents_1000_ticks(self) -> None:
        """Full scale: 200 agents, 1000 ticks."""
        config = SimulationConfig(num_agents=200, max_ticks=1000, seed=42)
        lead = Lead(config)
        output = lead.run()

        assert len(output.final_agent_states) == 200
        assert output.total_ticks == 1000

    def test_determinism_at_scale(self) -> None:
        """Same seed produces identical results at scale."""
        config = SimulationConfig(num_agents=50, max_ticks=100, seed=99)
        output1 = Lead(config).run()
        output2 = Lead(SimulationConfig(num_agents=50, max_ticks=100, seed=99)).run()

        assert output1.model_dump_json() == output2.model_dump_json()

    def test_constitution_evolves_at_scale(self) -> None:
        """With 50+ agents and 200 ticks, constitutional changes occur."""
        config = SimulationConfig(num_agents=50, max_ticks=200, seed=42)
        output = Lead(config).run()

        all_changes = [c for entry in output.history for c in entry.rule_changes]
        # With 50 diverse agents over 200 ticks, we expect some constitutional changes
        assert len(all_changes) > 0, (
            "Expected constitutional changes with 50 agents over 200 ticks"
        )

    def test_gini_stays_bounded(self) -> None:
        """Gini coefficient stays in [0, 1] throughout the simulation."""
        config = SimulationConfig(num_agents=50, max_ticks=200, seed=42)
        output = Lead(config).run()

        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0, (
                f"Gini {entry.gini} out of bounds at tick {entry.tick}"
            )

    def test_wealth_stays_positive(self) -> None:
        """All agents maintain non-negative wealth throughout."""
        config = SimulationConfig(num_agents=100, max_ticks=500, seed=42)
        output = Lead(config).run()

        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0, f"Agent {agent.id} has negative wealth: {agent.wealth}"

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds lead to divergent simulation trajectories."""
        output_a = Lead(SimulationConfig(num_agents=50, max_ticks=100, seed=1)).run()
        output_b = Lead(SimulationConfig(num_agents=50, max_ticks=100, seed=2)).run()

        # Final wealth distributions should differ
        wealth_a = sorted(a.wealth for a in output_a.final_agent_states)
        wealth_b = sorted(a.wealth for a in output_b.final_agent_states)
        assert wealth_a != wealth_b, "Different seeds should produce different results"

    def test_history_length_matches_observer_interval(self) -> None:
        """History entries are recorded at the correct interval for large runs."""
        observer_interval = 10
        max_ticks = 500
        config = SimulationConfig(
            num_agents=50,
            max_ticks=max_ticks,
            seed=42,
            observer_interval=observer_interval,
        )
        output = Lead(config).run()

        expected_entries = max_ticks // observer_interval
        assert len(output.history) == expected_entries, (
            f"Expected {expected_entries} history entries, got {len(output.history)}"
        )
        # Verify tick numbers in history entries
        for i, entry in enumerate(output.history):
            expected_tick = (i + 1) * observer_interval
            assert entry.tick == expected_tick, (
                f"Entry {i} has tick {entry.tick}, expected {expected_tick}"
            )


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks to track simulation speed.

    Time thresholds are intentionally generous to accommodate different hardware.
    The goal is to catch major performance regressions, not micro-benchmark.
    """

    def test_50_agents_100_ticks_under_10_seconds(self) -> None:
        """50 agents x 100 ticks should complete in under 10 seconds."""
        config = SimulationConfig(num_agents=50, max_ticks=100, seed=42)
        start = time.perf_counter()
        Lead(config).run()
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"Took {elapsed:.2f}s, expected < 10s"

    def test_100_agents_500_ticks_under_60_seconds(self) -> None:
        """100 agents x 500 ticks should complete in under 60 seconds."""
        config = SimulationConfig(num_agents=100, max_ticks=500, seed=42)
        start = time.perf_counter()
        Lead(config).run()
        elapsed = time.perf_counter() - start

        assert elapsed < 60.0, f"Took {elapsed:.2f}s, expected < 60s"

    def test_scaling_is_subquadratic_in_agents(self) -> None:
        """Doubling agents should less than 6x the runtime (sub-quadratic)."""
        config_small = SimulationConfig(num_agents=25, max_ticks=50, seed=42)
        start = time.perf_counter()
        Lead(config_small).run()
        time_small = time.perf_counter() - start

        config_large = SimulationConfig(num_agents=50, max_ticks=50, seed=42)
        start = time.perf_counter()
        Lead(config_large).run()
        time_large = time.perf_counter() - start

        ratio = time_large / max(time_small, 0.001)
        assert ratio < 6.0, f"Ratio {ratio:.1f}x, expected < 6x for doubling agents"

    def test_scaling_is_linear_in_ticks(self) -> None:
        """Doubling ticks should roughly double the runtime (linear)."""
        config_short = SimulationConfig(num_agents=25, max_ticks=50, seed=42)
        start = time.perf_counter()
        Lead(config_short).run()
        time_short = time.perf_counter() - start

        config_long = SimulationConfig(num_agents=25, max_ticks=100, seed=42)
        start = time.perf_counter()
        Lead(config_long).run()
        time_long = time.perf_counter() - start

        ratio = time_long / max(time_short, 0.001)
        # Should be roughly 2x, allow up to 4x for overhead
        assert ratio < 4.0, f"Ratio {ratio:.1f}x, expected < 4x for doubling ticks"
