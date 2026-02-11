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


class TestIntegrationPhase2:
    """Phase 2 integration tests — constitution evolution and determinism."""

    def test_constitution_evolves(self):
        """With 50 agents over 200 ticks, the constitution should change from defaults."""
        config = SimulationConfig(
            num_agents=50,
            max_ticks=200,
            seed=42,
            proposal_interval=5,
        )
        output = Lead(config).run()
        changed = (
            output.constitution.tax_rate != 0.0
            or output.constitution.property_rule.value != "private"
            or output.constitution.voting_rule.value != "majority"
            or output.constitution.redistribution_rule.value != "flat"
        )
        assert changed, "Constitution should evolve with 50 agents over 200 ticks"

    def test_full_determinism_with_governance(self):
        """Full determinism: identical seed produces identical constitution + agent states."""
        config = SimulationConfig(
            num_agents=20,
            max_ticks=50,
            seed=77,
            proposal_interval=5,
        )
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        # Compare full JSON — covers constitution, agent states, history
        assert output1.model_dump_json() == output2.model_dump_json()

    def test_governance_preserves_wealth_invariants(self):
        """Even with active governance, wealth invariants hold."""
        config = SimulationConfig(
            num_agents=30,
            max_ticks=100,
            seed=42,
            proposal_interval=5,
        )
        output = Lead(config).run()
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0
            assert agent.productivity > 0.0
