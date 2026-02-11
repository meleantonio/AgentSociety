"""Agent and constitution factories for simulation initialization."""

from __future__ import annotations

from emergent_constitution.config import SimulationConfig
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.tick import TickState
from emergent_constitution.rng import SimulationRNG


def _generate_dirichlet3(rng: SimulationRNG) -> tuple[float, float, float]:
    """Generate 3 values that sum to 1 using a Dirichlet-like approach.

    Draws 3 exponential samples (via -log(uniform)) and normalizes.
    """
    raw = [-_safe_log(rng.random()) for _ in range(3)]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def _safe_log(x: float) -> float:
    """Compute -log(x) safely, clamping x away from zero."""
    import math

    return -math.log(max(x, 1e-10))


def create_agents(config: SimulationConfig, rng: SimulationRNG) -> list[AgentState]:
    """Create N agents with seeded random endowments and preferences.

    Wealth: Normal(mean, std), clamped >= 0.01
    Productivity: Normal(mean, std), clamped >= 0.01
    Utility params: Dirichlet-like (alpha, beta, gamma summing to 1)
    Value vector: random split of equality/liberty summing to 1

    Args:
        config: Simulation configuration.
        rng: Seeded RNG instance.

    Returns:
        List of initialized AgentState objects.
    """
    agents: list[AgentState] = []
    for i in range(config.num_agents):
        wealth = max(0.01, rng.gauss(config.initial_wealth_mean, config.initial_wealth_std))
        productivity = max(
            0.01, rng.gauss(config.initial_productivity_mean, config.initial_productivity_std)
        )

        alpha, beta, gamma = _generate_dirichlet3(rng)
        utility_params = UtilityParams(alpha=alpha, beta=beta, gamma=gamma)

        eq = rng.random()
        value_vector = ValueVector(equality=eq, liberty=1.0 - eq)

        agents.append(
            AgentState(
                id=f"agent_{i:04d}",
                wealth=wealth,
                productivity=productivity,
                utility_params=utility_params,
                value_vector=value_vector,
            )
        )
    return agents


def create_initial_constitution() -> Constitution:
    """Create the default "no government" constitution.

    Returns:
        Constitution with default values (private property, 0% tax, majority vote, flat redist).
    """
    return Constitution()


def initialize_simulation(config: SimulationConfig) -> tuple[TickState, SimulationRNG]:
    """Bootstrap the simulation from configuration.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple of (initial TickState at tick 0, seeded RNG).
    """
    rng = SimulationRNG(config.seed)
    agents = create_agents(config, rng)
    constitution = create_initial_constitution()
    tick_state = TickState(tick=0, agent_states=agents, constitution=constitution)
    return tick_state, rng
