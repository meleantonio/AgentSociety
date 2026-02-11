"""Tests for agent and simulation initialization."""

from __future__ import annotations

from emergent_constitution.config import SimulationConfig
from emergent_constitution.initialization import (
    create_agents,
    create_initial_constitution,
    initialize_simulation,
)
from emergent_constitution.models.constitution import PropertyRule, RedistributionRule, VotingRule
from emergent_constitution.rng import SimulationRNG


class TestCreateAgents:
    def test_correct_count(self):
        config = SimulationConfig(num_agents=10, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        assert len(agents) == 10

    def test_unique_ids(self):
        config = SimulationConfig(num_agents=20, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        ids = [a.id for a in agents]
        assert len(set(ids)) == 20

    def test_positive_wealth(self):
        config = SimulationConfig(num_agents=50, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        for agent in agents:
            assert agent.wealth > 0

    def test_positive_productivity(self):
        config = SimulationConfig(num_agents=50, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        for agent in agents:
            assert agent.productivity > 0

    def test_utility_params_sum_to_one(self):
        config = SimulationConfig(num_agents=50, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        for agent in agents:
            p = agent.utility_params
            total = p.alpha + p.beta + p.gamma
            assert abs(total - 1.0) < 1e-6

    def test_value_vector_sum_to_one(self):
        config = SimulationConfig(num_agents=50, seed=42)
        rng = SimulationRNG(seed=config.seed)
        agents = create_agents(config, rng)
        for agent in agents:
            total = agent.value_vector.equality + agent.value_vector.liberty
            assert abs(total - 1.0) < 1e-6

    def test_deterministic_with_same_seed(self):
        config = SimulationConfig(num_agents=10, seed=42)
        agents1 = create_agents(config, SimulationRNG(seed=42))
        agents2 = create_agents(config, SimulationRNG(seed=42))
        for a1, a2 in zip(agents1, agents2, strict=True):
            assert a1 == a2

    def test_different_seed_different_agents(self):
        config = SimulationConfig(num_agents=10, seed=42)
        agents1 = create_agents(config, SimulationRNG(seed=42))
        agents2 = create_agents(config, SimulationRNG(seed=99))
        # At least some agents should differ in wealth
        wealths1 = [a.wealth for a in agents1]
        wealths2 = [a.wealth for a in agents2]
        assert wealths1 != wealths2


class TestCreateInitialConstitution:
    def test_defaults(self):
        c = create_initial_constitution()
        assert c.property_rule == PropertyRule.PRIVATE
        assert c.tax_rate == 0.0
        assert c.voting_rule == VotingRule.MAJORITY
        assert c.redistribution_rule == RedistributionRule.FLAT


class TestInitializeSimulation:
    def test_returns_tick_state_and_rng(self):
        config = SimulationConfig(num_agents=5, seed=42)
        tick_state, rng = initialize_simulation(config)
        assert tick_state.tick == 0
        assert len(tick_state.agent_states) == 5
        assert rng.seed == 42

    def test_deterministic(self):
        config = SimulationConfig(num_agents=5, seed=42)
        ts1, _ = initialize_simulation(config)
        ts2, _ = initialize_simulation(config)
        assert ts1 == ts2
