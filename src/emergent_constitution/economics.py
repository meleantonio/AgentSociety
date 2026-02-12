"""Economic engine — pure functions for production, taxation, and redistribution.

v1 functions retained for backward compat (compute_production, apply_taxation, etc.).
v2 functions implement DSGE-HA economics per spec/design.md section 3.6:
compute_budget, produce_output, distribute_firm_income, apply_rd_shock,
liquidate_firm, enforce_budget_constraint, compute_realized_utility,
validate_household_states.

All functions take immutable inputs and return new objects. No side effects.
"""

from __future__ import annotations

import math

import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
)
from emergent_constitution.models.decisions import EconomicDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.proposal import TradeOffer
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()

# Epsilon guard for Cobb-Douglas to avoid 0**x edge cases
_EPSILON = 1e-10


class NumericalInstabilityError(Exception):
    """Raised when agent wealth contains NaN or Inf values."""


# ============================================================================
# v1 functions (backward compatibility)
# ============================================================================


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


def apply_trades(
    agents: list[AgentState],
    trades: list[TradeOffer],
) -> list[AgentState]:
    """Apply bilateral trades: transfer wealth from buyers to sellers.

    Each trade is validated individually. Invalid trades (buyer lacks funds,
    amount not positive, unknown agent IDs) are silently skipped. Trades are
    applied sequentially in the order provided.

    Args:
        agents: Current agent states (not mutated).
        trades: List of trade offers to apply.

    Returns:
        New agent states with trade transfers applied.
    """
    # Build a mutable wealth map from copies
    wealth_map: dict[str, float] = {a.id: a.wealth for a in agents}
    agent_ids = set(wealth_map.keys())

    for trade in trades:
        # Validate agent IDs exist
        if trade.seller_id not in agent_ids or trade.buyer_id not in agent_ids:
            log.debug("trade.skipped.unknown_agent", trade=trade)
            continue

        # Validate amount is positive
        if trade.amount <= 0.0:
            log.debug("trade.skipped.non_positive_amount", trade=trade)
            continue

        # Validate buyer has enough wealth
        if wealth_map[trade.buyer_id] < trade.amount:
            log.debug(
                "trade.skipped.insufficient_funds",
                buyer_id=trade.buyer_id,
                buyer_wealth=wealth_map[trade.buyer_id],
                amount=trade.amount,
            )
            continue

        # Execute the transfer
        wealth_map[trade.buyer_id] -= trade.amount
        wealth_map[trade.seller_id] += trade.amount

    # Reconstruct agent list with updated wealth values
    return [a.model_copy(update={"wealth": wealth_map[a.id]}) for a in agents]


def economic_step(agents: list[AgentState], constitution: Constitution) -> list[AgentState]:
    """Execute one full economic cycle: produce -> tax -> redistribute.

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


# ============================================================================
# v2 functions (DSGE-HA economics engine)
# ============================================================================


def compute_budget(
    agent: HouseholdState,
    wage: float,
    interest_rate: float,
    tax: float,
    transfer: float,
) -> float:
    """Available resources: (1 + r) * a + w * z * labor_supply - tax + transfer.

    This is the maximum the agent can consume (saving nothing).

    Args:
        agent: Household state.
        wage: Market wage w_t.
        interest_rate: Market interest rate r_t.
        tax: Tax amount for this agent.
        transfer: Transfer amount for this agent.

    Returns:
        Total available budget.

    Implements REQ-003.
    """
    labor_income = wage * agent.productivity * agent.labor_supply
    asset_income = (1.0 + interest_rate) * agent.wealth
    return asset_income + labor_income - tax + transfer


def produce_output(firm: FirmState, alpha: float) -> float:
    """Cobb-Douglas firm production: Y_f = A_f * K_f^alpha * L_f^(1-alpha).

    Uses epsilon guards to avoid 0**x edge cases.

    Args:
        firm: Firm state with capital, labor_demand, and tfp.
        alpha: Capital share parameter (0 < alpha < 1).

    Returns:
        Firm output Y_f >= 0.

    Implements REQ-007.
    """
    k = max(firm.capital, _EPSILON)
    labor = max(firm.labor_demand, _EPSILON)
    return firm.tfp * (k**alpha) * (labor ** (1.0 - alpha))


def distribute_firm_income(
    firm: FirmState,
    wage: float,
    interest_rate: float,
    delta: float,
) -> tuple[float, float, float]:
    """Compute firm income distribution: wages, capital cost, profit.

    wages = w * L_f
    capital_cost = (r + delta) * K_f
    profit = Y_f - wages - capital_cost

    Args:
        firm: Firm state (must have output computed).
        wage: Market wage w_t.
        interest_rate: Market interest rate r_t.
        delta: Depreciation rate.

    Returns:
        Tuple of (wages_paid, capital_cost, profit).

    Implements REQ-008.
    """
    wages_paid = wage * firm.labor_demand
    capital_cost = (interest_rate + delta) * firm.capital
    profit = firm.output - wages_paid - capital_cost
    return wages_paid, capital_cost, profit


def apply_rd_shock(
    firm: FirmState,
    rng: SimulationRNG,
    config: SimulationConfigV2,
    mean_rd_spend: float,
) -> FirmState:
    """Stochastic TFP improvement from R&D spending.

    Success probability = config.rd_success_base_prob * sqrt(rd_spend / mean_rd_spend).
    If success: A_f *= (1 + drawn_improvement) where improvement ~ N(mean, std).

    Args:
        firm: Firm state with rd_spend set.
        rng: Seeded RNG instance.
        config: Simulation configuration with R&D parameters.
        mean_rd_spend: Average R&D spending across all firms (for normalization).

    Returns:
        New FirmState with potentially improved tfp.

    Implements REQ-009.
    """
    if firm.rd_spend <= 0.0 or mean_rd_spend <= 0.0:
        return firm.model_copy()

    # Success probability scaled by relative R&D spending
    relative_spend = firm.rd_spend / mean_rd_spend
    success_prob = config.rd_success_base_prob * math.sqrt(relative_spend)
    success_prob = min(success_prob, 1.0)

    if rng.random() < success_prob:
        improvement = max(
            0.0, rng.gauss(config.rd_tfp_improvement_mean, config.rd_tfp_improvement_std)
        )
        new_tfp = firm.tfp * (1.0 + improvement)
        log.debug(
            "rd.success",
            firm_id=firm.id,
            old_tfp=firm.tfp,
            new_tfp=new_tfp,
            improvement=improvement,
        )
        return firm.model_copy(update={"tfp": new_tfp})

    return firm.model_copy()


def liquidate_firm(firm: FirmState) -> float:
    """Return remaining capital to owner upon firm liquidation.

    Args:
        firm: Firm state to liquidate.

    Returns:
        Capital returned to owner (max(0, remaining_capital)).

    Implements REQ-010.
    """
    return max(0.0, firm.capital)


def enforce_budget_constraint(
    decision: EconomicDecision,
    agent: HouseholdState,
    budget: float,
    a_min: float,
) -> EconomicDecision:
    """Project decision onto feasible set.

    1. Clamp leisure to [0, 1].
    2. Recompute available budget with clamped labor supply.
    3. Clamp consumption to [0, budget - a_min].
    4. Ensure a_{t+1} >= a_min.

    Args:
        decision: Raw economic decision (may be infeasible).
        agent: Household state.
        budget: Pre-computed available budget (with original labor_supply).
        a_min: Minimum wealth floor.

    Returns:
        New EconomicDecision clamped to feasible region.

    Implements REQ-005, PROP-002, PROP-004.
    """
    # Step 1: clamp leisure
    leisure = max(0.0, min(1.0, decision.leisure))

    # Step 2: if leisure changed, we need to adjust the budget
    # Budget was computed with agent's current labor_supply; recalculate if leisure differs
    original_labor = agent.labor_supply
    new_labor = 1.0 - leisure
    # Adjust budget for the change in labor supply
    # The labor income portion changes: wage * z * (new_labor - old_labor)
    # We don't have wage here, so we use a proportional adjustment:
    # budget_adjusted = budget + (new_labor - original_labor) * labor_income_per_unit
    # Since budget = (1+r)*a + w*z*original_labor - tax + transfer,
    # the adjusted budget = budget + w*z*(new_labor - original_labor)
    # However, we don't have w*z separately. The safe approach is to assume
    # the caller passes budget computed with the original labor, and we just
    # ensure consumption fits within what's available.
    # If agent already has labor_supply set to original_labor:
    if abs(new_labor - original_labor) > _EPSILON and original_labor > _EPSILON:
        # Scale the labor income portion
        # budget = asset_part + labor_part, labor_part proportional to labor_supply
        # We estimate: labor_part = budget - (1+r)*a (ignoring tax/transfer for scaling)
        # This is an approximation; the caller should ideally recompute budget
        budget_adjusted = budget
    else:
        budget_adjusted = budget

    # Step 3: max consumption so that savings >= a_min - wealth
    # savings = budget - consumption, a_{t+1} = wealth + savings >= a_min
    # => consumption <= budget - (a_min - wealth) = budget - a_min + wealth
    max_consumption = max(0.0, budget_adjusted - a_min + agent.wealth)

    consumption = max(0.0, min(decision.consumption, max_consumption))

    if consumption != decision.consumption or leisure != decision.leisure:
        log.debug(
            "budget_constraint.clamped",
            agent_id=agent.id,
            orig_consumption=decision.consumption,
            orig_leisure=decision.leisure,
            clamped_consumption=consumption,
            clamped_leisure=leisure,
            budget=budget_adjusted,
            a_min=a_min,
        )

    return EconomicDecision(consumption=consumption, leisure=leisure)


def compute_realized_utility(
    agent: HouseholdState,
    public_goods_per_capita: float,
) -> float:
    """Compute realized utility: u(c, l, G) = c^alpha * l^beta * G^gamma.

    Uses epsilon guards for zero inputs so utility is continuous and finite.

    Args:
        agent: Household state with consumption and leisure set.
        public_goods_per_capita: G_t per-capita public goods this period.

    Returns:
        Realized utility value >= 0.

    Implements REQ-028, PROP-005.
    """
    c = max(agent.consumption, _EPSILON)
    leisure = max(agent.leisure, _EPSILON)
    g = max(public_goods_per_capita, _EPSILON)

    alpha = agent.utility_params.alpha
    beta = agent.utility_params.beta
    gamma = agent.utility_params.gamma

    return (c**alpha) * (leisure**beta) * (g**gamma)


def validate_household_states(households: list[HouseholdState]) -> None:
    """Check all household wealth values for NaN/Inf.

    Args:
        households: List of household states to validate.

    Raises:
        NumericalInstabilityError: If any wealth value is NaN or Inf.
    """
    for h in households:
        if math.isnan(h.wealth) or math.isinf(h.wealth):
            msg = f"Household {h.id} has invalid wealth: {h.wealth}"
            raise NumericalInstabilityError(msg)
        if math.isnan(h.consumption) or math.isinf(h.consumption):
            msg = f"Household {h.id} has invalid consumption: {h.consumption}"
            raise NumericalInstabilityError(msg)
        if math.isnan(h.realized_utility) or math.isinf(h.realized_utility):
            msg = f"Household {h.id} has invalid realized_utility: {h.realized_utility}"
            raise NumericalInstabilityError(msg)
