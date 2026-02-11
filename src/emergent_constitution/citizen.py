"""Citizen decision logic — proposal generation and voting.

All functions are pure: they take agent state, constitution, and an RNG,
and return decisions without mutating any inputs.
"""

from __future__ import annotations

from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.models.proposal import Proposal
from emergent_constitution.rng import SimulationRNG

# Maps each constitutional dimension to (equality-preferred value, liberty-preferred value).
_PREFERRED_VALUES: dict[str, tuple[object, object]] = {
    "property_rule": (PropertyRule.COMMUNAL, PropertyRule.PRIVATE),
    "tax_rate": (None, None),  # handled specially
    "redistribution_rule": (RedistributionRule.PROGRESSIVE, RedistributionRule.NONE),
    "voting_rule": (VotingRule.SUPERMAJORITY, VotingRule.MAJORITY),
}

# Probability of an agent skipping proposal generation (controls density).
_PROPOSAL_SKIP_PROBABILITY = 0.85


def decide_proposal(
    agent: AgentState,
    constitution: Constitution,
    all_agents: list[AgentState],
    rng: SimulationRNG,
) -> Proposal | None:
    """Decide whether to propose a rule change, and if so, which one.

    Algorithm:
        1. Probabilistic gating (~15% chance of proposing).
        2. Compute dissatisfaction per constitutional dimension.
        3. Pick highest-dissatisfaction dimension (alphabetical tie-break).
        4. Skip if dissatisfaction <= 0.1 or preferred == current.
        5. Return Proposal.

    Args:
        agent: The agent's current state.
        constitution: The current constitution.
        all_agents: All agent states (for context, unused currently).
        rng: Seeded RNG for deterministic gating.

    Returns:
        A Proposal, or None if the agent chooses not to propose.
    """
    _ = all_agents  # reserved for future coalition logic

    # Step 1: probabilistic gating
    if rng.random() > (1 - _PROPOSAL_SKIP_PROBABILITY):
        return None

    eq = agent.value_vector.equality

    # Step 2: compute dissatisfaction per dimension
    dissatisfaction: dict[str, float] = {}

    # tax_rate: equality-lovers want ~60% tax, liberty-lovers want ~0%
    preferred_tax = eq * 0.6
    dissatisfaction["tax_rate"] = abs(constitution.tax_rate - preferred_tax)

    # property_rule
    preferred_property = PropertyRule.COMMUNAL if eq > 0.5 else PropertyRule.PRIVATE
    dissatisfaction["property_rule"] = (
        0.0 if constitution.property_rule == preferred_property else eq
    )

    # redistribution_rule
    preferred_redist = RedistributionRule.PROGRESSIVE if eq > 0.5 else RedistributionRule.NONE
    dissatisfaction["redistribution_rule"] = (
        0.0 if constitution.redistribution_rule == preferred_redist else eq
    )

    # voting_rule
    preferred_voting = VotingRule.SUPERMAJORITY if eq > 0.5 else VotingRule.MAJORITY
    dissatisfaction["voting_rule"] = (
        0.0 if constitution.voting_rule == preferred_voting else eq * 0.5
    )

    # Step 3: pick highest dissatisfaction (alphabetical tie-break)
    best_key = max(sorted(dissatisfaction.keys()), key=lambda k: dissatisfaction[k])
    best_score = dissatisfaction[best_key]

    # Step 4: skip if not dissatisfied enough
    if best_score <= 0.1:
        return None

    preferred_value = _get_preferred_value(best_key, eq)
    current_value = getattr(constitution, best_key)
    if preferred_value == current_value:
        return None

    # Step 5: return proposal
    return Proposal(
        rule_key=best_key,
        proposed_value=preferred_value,
        proposer_id=agent.id,
    )


def decide_votes(
    agent: AgentState,
    constitution: Constitution,
    proposals: list[Proposal],
    all_agents: list[AgentState],
    rng: SimulationRNG,
) -> dict[str, bool]:
    """Decide how to vote on each proposal.

    Combined score = 0.6 * value_alignment + 0.4 * self_interest.
    Vote "for" if score > 0.5.

    Args:
        agent: The agent's current state.
        constitution: The current constitution.
        proposals: Proposals to vote on.
        all_agents: All agent states (for wealth/productivity ranking).
        rng: Seeded RNG (reserved for stochastic voting in future).

    Returns:
        Mapping of proposer_id -> vote (True=for, False=against).
        Keys are formatted as "{proposer_id}:{rule_key}" to handle multiple proposals.
    """
    _ = rng  # reserved for future stochastic voting

    if not proposals:
        return {}

    wealths = [a.wealth for a in all_agents]
    productivities = [a.productivity for a in all_agents]
    wealth_rank = _rank_among(agent.wealth, wealths)  # 0 = poorest, 1 = richest
    productivity_rank = _rank_among(agent.productivity, productivities)

    votes: dict[str, bool] = {}
    eq = agent.value_vector.equality

    for proposal in proposals:
        value_alignment = _compute_value_alignment(
            proposal,
            constitution,
            eq,
        )
        self_interest = _compute_self_interest(
            proposal,
            wealth_rank,
            productivity_rank,
        )
        score = 0.6 * value_alignment + 0.4 * self_interest
        votes[proposal.proposer_id] = score > 0.5

    return votes


def compute_gini(wealths: list[float]) -> float:
    """Compute the Gini coefficient for a wealth distribution.

    Args:
        wealths: List of non-negative wealth values.

    Returns:
        Gini coefficient in [0, 1]. Returns 0.0 for empty or single-element lists.
    """
    n = len(wealths)
    if n <= 1:
        return 0.0

    total = sum(wealths)
    if total == 0:
        return 0.0

    sorted_w = sorted(wealths)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, w in enumerate(sorted_w):
        cumulative += w
        weighted_sum += (i + 1) * w

    return (2 * weighted_sum) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_preferred_value(rule_key: str, equality: float) -> object:
    """Return the preferred value for a constitutional dimension given equality weight."""
    if rule_key == "tax_rate":
        return round(equality * 0.6, 2)

    eq_value, lib_value = _PREFERRED_VALUES[rule_key]
    return eq_value if equality > 0.5 else lib_value


def _rank_among(value: float, all_values: list[float]) -> float:
    """Return a normalized rank (0=lowest, 1=highest) of value among all_values."""
    n = len(all_values)
    if n <= 1:
        return 0.5
    rank = sum(1 for v in all_values if v < value)
    return rank / (n - 1)


def _compute_value_alignment(
    proposal: Proposal,
    constitution: Constitution,
    equality: float,
) -> float:
    """How well does the proposal align with the agent's values?

    Returns a score in [0, 1] where 1 = perfectly aligned.
    """
    preferred = _get_preferred_value(proposal.rule_key, equality)
    current = getattr(constitution, proposal.rule_key)
    proposed = proposal.proposed_value

    if proposal.rule_key == "tax_rate":
        # Continuous: is the proposal closer to my preferred value?
        pref_f = float(preferred)
        current_f = float(current)
        proposed_f = float(proposed)
        current_dist = abs(current_f - pref_f)
        proposed_dist = abs(proposed_f - pref_f)
        if current_dist == 0 and proposed_dist == 0:
            return 0.5  # no change
        if current_dist == 0:
            return 0.0  # moving away from already-ideal
        improvement = (current_dist - proposed_dist) / current_dist
        return max(0.0, min(1.0, 0.5 + 0.5 * improvement))
    else:
        # Discrete: binary — is the proposed value my preferred?
        if proposed == preferred:
            return 1.0
        if current == preferred:
            return 0.0
        return 0.3  # neither current nor proposed is ideal


def _compute_self_interest(
    proposal: Proposal,
    wealth_rank: float,
    productivity_rank: float,
) -> float:
    """How much does the proposal serve the agent's material self-interest?

    Returns a score in [0, 1] where 1 = strongly beneficial.
    """
    key = proposal.rule_key
    value = proposal.proposed_value

    if key == "tax_rate":
        proposed_tax = float(value)
        # Poor agents favor higher taxes (more redistribution)
        # Rich agents favor lower taxes
        if proposed_tax > 0.3:
            return 1.0 - wealth_rank  # poor = high, rich = low
        else:
            return wealth_rank  # rich = high, poor = low

    if key == "redistribution_rule":
        if value in (RedistributionRule.PROGRESSIVE, "progressive"):
            return 1.0 - wealth_rank
        if value in (RedistributionRule.NONE, "none"):
            return wealth_rank
        return 0.5

    if key == "property_rule":
        if value in (PropertyRule.PRIVATE, "private"):
            return productivity_rank  # productive agents benefit from private property
        if value in (PropertyRule.COMMUNAL, "communal"):
            return 1.0 - productivity_rank
        return 0.5

    if key == "voting_rule":
        return 0.5  # voting rule doesn't directly affect material interest

    return 0.5
