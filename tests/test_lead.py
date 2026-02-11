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


class TestLeadPhase2:
    """Phase 2 tests — proposals, voting, and constitution evolution."""

    def test_proposals_on_interval_ticks(self):
        """Proposals should only be collected on interval ticks."""
        config = SimulationConfig(num_agents=10, max_ticks=20, seed=42, proposal_interval=5)
        # Run tick by tick and verify proposals appear only on interval ticks
        lead2 = Lead(config)
        for tick in range(1, 21):
            lead2.tick_state = lead2._advance_tick(tick)
            if tick % 5 != 0:
                assert lead2.tick_state.proposals_this_tick == [], (
                    f"Proposals found on non-interval tick {tick}"
                )

    def test_determinism_with_proposals(self):
        """Two runs with same seed produce identical output including proposals/votes."""
        config = SimulationConfig(num_agents=10, max_ticks=30, seed=42, proposal_interval=5)
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 == json2

    def test_positive_wealth_preserved(self):
        """All agents maintain non-negative wealth with active governance."""
        config = SimulationConfig(num_agents=20, max_ticks=50, seed=42, proposal_interval=5)
        output = Lead(config).run()
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0, f"Agent {agent.id} has negative wealth: {agent.wealth}"

    def test_constitution_can_change(self):
        """With enough agents and ticks, the constitution should evolve."""
        # Use more agents and ticks to increase the chance of proposals passing
        config = SimulationConfig(num_agents=50, max_ticks=100, seed=42, proposal_interval=5)
        output = Lead(config).run()
        # Check if at least one constitutional field changed from defaults
        changed = (
            output.constitution.tax_rate != 0.0
            or output.constitution.property_rule.value != "private"
            or output.constitution.voting_rule.value != "majority"
            or output.constitution.redistribution_rule.value != "flat"
        )
        assert changed, "Constitution should evolve with 50 agents over 100 ticks"


class TestLeadPhase3:
    """Phase 3 tests — observer statistics and history logging."""

    def test_observations_on_interval_ticks(self):
        """History entries should only be recorded on observer-interval ticks."""
        config = SimulationConfig(num_agents=5, max_ticks=20, seed=42, observer_interval=5)
        lead = Lead(config)
        lead.run()
        expected_ticks = {5, 10, 15, 20}
        actual_ticks = {entry.tick for entry in lead.history}
        assert actual_ticks == expected_ticks

    def test_history_length_matches_intervals(self):
        """Number of history entries == max_ticks // observer_interval."""
        config = SimulationConfig(num_agents=5, max_ticks=20, seed=42, observer_interval=5)
        output = Lead(config).run()
        assert len(output.history) == 20 // 5

    def test_history_statistics_valid(self):
        """All statistics in history entries are within valid ranges."""
        config = SimulationConfig(num_agents=10, max_ticks=30, seed=42, observer_interval=5)
        output = Lead(config).run()
        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0, f"Invalid Gini {entry.gini} at tick {entry.tick}"
            assert entry.mean_wealth >= 0.0
            assert entry.median_wealth >= 0.0
            assert entry.total_output > 0.0

    def test_rule_changes_tracked(self):
        """With active governance, rule changes should appear in history."""
        config = SimulationConfig(
            num_agents=50,
            max_ticks=100,
            seed=42,
            proposal_interval=5,
            observer_interval=5,
        )
        output = Lead(config).run()
        all_changes = [ch for entry in output.history for ch in entry.rule_changes]
        assert len(all_changes) > 0, "Expected at least one rule change to be tracked"

    def test_deterministic_history(self):
        """Same seed produces identical history."""
        config = SimulationConfig(num_agents=10, max_ticks=30, seed=42, observer_interval=5)
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        assert len(output1.history) == len(output2.history)
        for e1, e2 in zip(output1.history, output2.history, strict=True):
            assert e1 == e2
