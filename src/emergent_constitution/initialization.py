"""Agent and constitution factories for simulation initialization -- v1 and v2.

v1 functions retained for backward compat (create_agents, initialize_simulation).
v2 functions (create_households, initialize_simulation_v2) build DSGE-HA initial
conditions per spec/design.md section 3.1.
"""

from __future__ import annotations

import math

from emergent_constitution.config import SimulationConfig, SimulationConfigV2
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    ConstitutionV2,
    create_default_constitution,
)
from emergent_constitution.models.history import PeriodState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
)
from emergent_constitution.models.household import UtilityParams as UtilityParamsV2
from emergent_constitution.models.household import ValueVector as ValueVectorV2
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.models.tick import TickState
from emergent_constitution.rng import SimulationRNG
from emergent_constitution.shock_generators import (
    _draw_from_cdf,
    rouwenhorst_discretize,
    stationary_distribution,
)

# ============================================================================
# Shared helpers
# ============================================================================

# Re-export for backward compat with test_v2_initialization imports
_stationary_distribution = stationary_distribution
_draw_from_distribution = _draw_from_cdf


def _safe_log(x: float) -> float:
    """Compute -log(x) safely, clamping x away from zero."""
    return -math.log(max(x, 1e-10))


def _generate_dirichlet3(rng: SimulationRNG) -> tuple[float, float, float]:
    """Generate 3 values that sum to 1 using a Dirichlet-like approach.

    Draws 3 exponential samples (via -log(uniform)) and normalizes.
    """
    raw = [-_safe_log(rng.random()) for _ in range(3)]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


# ============================================================================
# v1 initialization (backward compatibility)
# ============================================================================


def create_agents(config: SimulationConfig, rng: SimulationRNG) -> list[AgentState]:
    """Create N agents with seeded random endowments and preferences (v1).

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
            0.01,
            rng.gauss(config.initial_productivity_mean, config.initial_productivity_std),
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
    """Create the default "no government" constitution (v1).

    Returns:
        Constitution with default values (private property, 0% tax,
        majority vote, flat redist).
    """
    return Constitution()


def initialize_simulation(
    config: SimulationConfig,
) -> tuple[TickState, SimulationRNG]:
    """Bootstrap the simulation from configuration (v1).

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


# ============================================================================
# v2 initialization (DSGE-HA)
# ============================================================================


def create_households(
    config: SimulationConfigV2,
    rng: SimulationRNG,
    productivity_grid: list[float],
    stationary_dist: list[float],
) -> list[HouseholdState]:
    """Create N household agents with DSGE-HA initial conditions.

    Args:
        config: v2 simulation configuration.
        rng: Seeded RNG instance.
        productivity_grid: Discrete productivity grid from Rouwenhorst.
        stationary_dist: Stationary distribution of productivity process.

    Returns:
        List of initialized HouseholdState objects.
    """
    households: list[HouseholdState] = []

    for i in range(config.num_agents):
        # Wealth from Normal, clamped >= a_min
        wealth = max(
            config.a_min,
            rng.gauss(config.initial_wealth_mean, config.initial_wealth_std),
        )

        # Productivity index from stationary distribution
        prod_idx = _draw_from_distribution(rng, stationary_dist)
        productivity = productivity_grid[prod_idx]

        # Utility params: alpha, beta, gamma ~ Dirichlet; beta_discount ~ U(0.9, 0.99)
        alpha, beta, gamma = _generate_dirichlet3(rng)
        beta_discount = rng.uniform(0.9, 0.99)
        utility_params = UtilityParamsV2(
            alpha=alpha, beta=beta, gamma=gamma, beta_discount=beta_discount
        )

        # Value vector: random split
        eq = rng.random()
        value_vector = ValueVectorV2(equality=eq, liberty=1.0 - eq)

        households.append(
            HouseholdState(
                id=f"agent_{i:04d}",
                wealth=wealth,
                productivity=productivity,
                productivity_index=prod_idx,
                utility_params=utility_params,
                value_vector=value_vector,
                role=OccupationalRole.WORKER,
            )
        )

    return households


def _initial_market_guess(
    config: SimulationConfigV2,
    households: list[HouseholdState],
) -> MarketState:
    """Compute initial market prices from analytical steady-state guesses.

    Uses Cobb-Douglas FOCs: w = (1-alpha)*Y/L, r = alpha*Y/K - delta.

    Args:
        config: v2 simulation configuration.
        households: Initial household states.

    Returns:
        MarketState with analytical price guesses.
    """
    total_capital = sum(h.wealth for h in households)
    total_labor = sum(h.productivity for h in households)

    # Avoid division by zero
    total_capital = max(total_capital, 1.0)
    total_labor = max(total_labor, 1.0)

    # Cobb-Douglas: Y = K^alpha * L^(1-alpha)
    aggregate_output = total_capital**config.alpha * total_labor ** (1.0 - config.alpha)

    # FOC-derived prices
    wage = (1.0 - config.alpha) * aggregate_output / total_labor
    interest_rate = config.alpha * aggregate_output / total_capital - config.delta

    return MarketState(
        wage=max(wage, 0.001),
        interest_rate=interest_rate,
        aggregate_output=aggregate_output,
    )


def initialize_simulation_v2(
    config: SimulationConfigV2,
) -> tuple[PeriodState, SimulationRNG]:
    """Bootstrap DSGE-HA simulation at period 0.

    1. Create SimulationRNG from seed.
    2. Discretize productivity process (Rouwenhorst).
    3. Create N households with Markov productivity assignments.
    4. Create default constitution.
    5. Create initial MarketState with analytical guesses.
    6. Create initial ShockState.

    Args:
        config: v2 simulation configuration.

    Returns:
        Tuple of (initial PeriodState at period 0, seeded RNG).

    Implements REQ-001, REQ-034, PROP-001.
    """
    rng = SimulationRNG(config.seed)

    # Rouwenhorst discretization
    productivity_grid, transition_matrix = rouwenhorst_discretize(
        rho=config.rho_z,
        sigma=config.sigma_z,
        n_states=config.num_z_states,
    )

    # Stationary distribution for initial productivity assignment
    stationary_dist = stationary_distribution(transition_matrix)

    # Create households
    households = create_households(config, rng, productivity_grid, stationary_dist)

    # Default constitution
    constitution: ConstitutionV2 = create_default_constitution()

    # Initial market state
    market = _initial_market_guess(config, households)

    # Initial shock state
    shocks = ShockState(
        productivity_grid=productivity_grid,
        transition_matrix=transition_matrix,
        aggregate_tfp=1.0,
    )

    period_state = PeriodState(
        period=0,
        households=households,
        firms=[],
        market=market,
        shocks=shocks,
        constitution=constitution,
    )

    return period_state, rng
