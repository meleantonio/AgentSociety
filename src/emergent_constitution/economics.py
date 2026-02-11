"""Economic engine — pure functions for production, taxation, and redistribution.

All functions take immutable inputs and return new objects. No side effects.
"""

from __future__ import annotations

import math

from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
)


class NumericalInstabilityError(Exception):
    """Raised when agent wealth contains NaN or Inf values."""


def validate_agent_states(agents: list[AgentState]) -> None:
    """Check all agent wealth values for NaN/Inf.

    Args:
        agents: List of agent states to validate.

    Raises:
        NumericalInstabilityError: If any wealth value is NaN or Inf.
    """
    for agent in agents:
        if math.isnan(agent.wealth) or math.isinf(agent.wealth):
            msg = f"Agent {agent.id} has invalid wealth: {agent.wealth}"
            raise NumericalInstabilityError(msg)


def compute_production(agents: list[AgentState], constitution: Constitution) -> dict[str, float]:
    """Compute production output per agent based on the property rule.

    - Private: each agent keeps their own productivity as output.
    - Communal: total productivity is pooled and split equally.
    - Mixed: 50% private, 50% pooled equally.

    Args:
        agents: Current agent states.
        constitution: Active ruleset.

    Returns:
        Dict mapping agent_id to their production output.
    """
    n = len(agents)
    if n == 0:
        return {}

    total_productivity = sum(a.productivity for a in agents)

    if constitution.property_rule == PropertyRule.PRIVATE:
        return {a.id: a.productivity for a in agents}

    if constitution.property_rule == PropertyRule.COMMUNAL:
        equal_share = total_productivity / n
        return {a.id: equal_share for a in agents}

    # MIXED: 50% private + 50% pooled
    pooled_share = (total_productivity * 0.5) / n
    return {a.id: (a.productivity * 0.5) + pooled_share for a in agents}


def apply_taxation(
    agents: list[AgentState],
    production: dict[str, float],
    constitution: Constitution,
) -> tuple[list[AgentState], float]:
    """Apply taxation: each agent's output is taxed at the constitutional rate.

    Agent wealth increases by (output * (1 - tax_rate)).
    Tax revenue = sum of (output * tax_rate) across all agents.

    Args:
        agents: Current agent states (not mutated).
        production: Production output per agent.
        constitution: Active ruleset.

    Returns:
        Tuple of (new agent states with updated wealth, total tax revenue).
    """
    tax_rate = constitution.tax_rate
    new_agents: list[AgentState] = []
    tax_revenue = 0.0

    for agent in agents:
        output = production.get(agent.id, 0.0)
        tax = output * tax_rate
        net_income = output - tax
        tax_revenue += tax
        new_agents.append(agent.model_copy(update={"wealth": agent.wealth + net_income}))

    return new_agents, tax_revenue


def apply_redistribution(
    agents: list[AgentState],
    tax_revenue: float,
    constitution: Constitution,
) -> list[AgentState]:
    """Redistribute collected tax revenue according to the redistribution rule.

    - None: revenue is lost (not redistributed).
    - Flat: equal share to each agent.
    - Progressive: inversely proportional to wealth (poorer agents get more).

    Args:
        agents: Current agent states (not mutated).
        tax_revenue: Total tax collected this tick.
        constitution: Active ruleset.

    Returns:
        New agent states with redistributed wealth.
    """
    n = len(agents)
    if n == 0 or tax_revenue <= 0.0:
        return [a.model_copy() for a in agents]

    if constitution.redistribution_rule == RedistributionRule.NONE:
        return [a.model_copy() for a in agents]

    if constitution.redistribution_rule == RedistributionRule.FLAT:
        share = tax_revenue / n
        return [a.model_copy(update={"wealth": a.wealth + share}) for a in agents]

    # PROGRESSIVE: inverse-wealth weighting
    # Weight = 1 / (wealth + 1) to avoid division by zero and give more to poorer agents.
    weights = [1.0 / (a.wealth + 1.0) for a in agents]
    total_weight = sum(weights)

    return [
        a.model_copy(update={"wealth": a.wealth + tax_revenue * (w / total_weight)})
        for a, w in zip(agents, weights, strict=True)
    ]


def economic_step(agents: list[AgentState], constitution: Constitution) -> list[AgentState]:
    """Execute one full economic cycle: produce → tax → redistribute.

    This is the composition of compute_production, apply_taxation, and
    apply_redistribution. Validates outputs for numerical stability.

    Args:
        agents: Current agent states (not mutated).
        constitution: Active ruleset.

    Returns:
        New agent states after the economic cycle.

    Raises:
        NumericalInstabilityError: If any resulting wealth is NaN/Inf.
    """
    production = compute_production(agents, constitution)
    taxed_agents, tax_revenue = apply_taxation(agents, production, constitution)
    final_agents = apply_redistribution(taxed_agents, tax_revenue, constitution)
    validate_agent_states(final_agents)
    return final_agents
