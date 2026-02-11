"""Integration tests — end-to-end simulation runs."""

from __future__ import annotations

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead


class TestIntegration:
    def test_small_run_completes(self):
        """5 agents / 20 ticks completes without error."""
        config = SimulationConfig(num_agents=5, max_ticks=20, seed=42)
        output = Lead(config).run()
        assert output.total_ticks == 20
        assert len(output.final_agent_states) == 5

    def test_determinism_json_roundtrip(self):
        """Two identical runs produce identical JSON output."""
        config = SimulationConfig(num_agents=5, max_ticks=20, seed=42)
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 == json2

    def test_all_agents_have_positive_wealth(self):
        """After simulation, no agent should have negative wealth."""
        config = SimulationConfig(num_agents=10, max_ticks=50, seed=42)
        output = Lead(config).run()
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0

    @pytest.mark.slow
    def test_large_run(self):
        """50 agents / 50 ticks — larger scale smoke test."""
        config = SimulationConfig(num_agents=50, max_ticks=50, seed=42)
        output = Lead(config).run()
        assert output.total_ticks == 50
        assert len(output.final_agent_states) == 50
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0
