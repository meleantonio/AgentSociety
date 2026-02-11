"""Shared test fixtures."""

from __future__ import annotations

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.rng import SimulationRNG


@pytest.fixture
def rng() -> SimulationRNG:
    """Seeded RNG for deterministic tests."""
    return SimulationRNG(seed=42)


@pytest.fixture
def config() -> SimulationConfig:
    """Default simulation config with small agent count for fast tests."""
    return SimulationConfig(num_agents=5, max_ticks=10, seed=42)


@pytest.fixture
def default_constitution() -> Constitution:
    """Default constitution (private, 0% tax, majority, flat)."""
    return Constitution()


@pytest.fixture
def sample_agent() -> AgentState:
    """A single sample agent."""
    return AgentState(
        id="agent_0000",
        wealth=100.0,
        productivity=10.0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.6, liberty=0.4),
    )


@pytest.fixture
def sample_agents() -> list[AgentState]:
    """Three agents with known endowments for deterministic economics tests."""
    return [
        AgentState(
            id="agent_0000",
            wealth=100.0,
            productivity=10.0,
            utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
            value_vector=ValueVector(equality=0.6, liberty=0.4),
        ),
        AgentState(
            id="agent_0001",
            wealth=50.0,
            productivity=20.0,
            utility_params=UtilityParams(alpha=0.4, beta=0.4, gamma=0.2),
            value_vector=ValueVector(equality=0.3, liberty=0.7),
        ),
        AgentState(
            id="agent_0002",
            wealth=200.0,
            productivity=5.0,
            utility_params=UtilityParams(alpha=0.3, beta=0.3, gamma=0.4),
            value_vector=ValueVector(equality=0.8, liberty=0.2),
        ),
    ]
