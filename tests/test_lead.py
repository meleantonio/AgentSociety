"""Tests for the Lead (Simulation Governor) tick loop."""

from __future__ import annotations

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead


class TestLead:
    def test_runs_to_completion(self):
        config = SimulationConfig(num_agents=5, max_ticks=10, seed=42)
        lead = Lead(config)
        output = lead.run()
        assert output.total_ticks == 10

    def test_correct_agent_count(self):
        config = SimulationConfig(num_agents=5, max_ticks=10, seed=42)
        lead = Lead(config)
        output = lead.run()
        assert len(output.final_agent_states) == 5

    def test_wealth_evolves(self):
        config = SimulationConfig(num_agents=5, max_ticks=10, seed=42)
        lead = Lead(config)
        initial_wealth = sum(a.wealth for a in lead.tick_state.agent_states)
        output = lead.run()
        final_wealth = sum(a.wealth for a in output.final_agent_states)
        # With 0% tax (default), wealth increases by production each tick
        assert final_wealth > initial_wealth

    def test_deterministic_output(self):
        config = SimulationConfig(num_agents=5, max_ticks=20, seed=42)
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        assert output1 == output2

    def test_seed_in_output(self):
        config = SimulationConfig(num_agents=5, max_ticks=5, seed=123)
        output = Lead(config).run()
        assert output.seed == 123

    def test_constitution_unchanged_in_phase1(self):
        """In Phase 1, no proposals pass, so constitution stays at defaults."""
        config = SimulationConfig(num_agents=5, max_ticks=10, seed=42)
        output = Lead(config).run()
        assert output.constitution.tax_rate == 0.0

    def test_tick_state_advances(self):
        config = SimulationConfig(num_agents=5, max_ticks=10, seed=42)
        lead = Lead(config)
        lead.run()
        assert lead.tick_state.tick == 10

    def test_different_seeds_different_results(self):
        output1 = Lead(SimulationConfig(num_agents=5, max_ticks=10, seed=42)).run()
        output2 = Lead(SimulationConfig(num_agents=5, max_ticks=10, seed=99)).run()
        wealths1 = [a.wealth for a in output1.final_agent_states]
        wealths2 = [a.wealth for a in output2.final_agent_states]
        assert wealths1 != wealths2
