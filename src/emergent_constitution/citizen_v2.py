"""Rule-based governance for v2 constitution — proposal generation and voting.

Mirrors v1 citizen.py patterns but works with ConstitutionV2's extensible
rules. Used as benchmark-mode fallback when LLM governance is unavailable.

All functions are pure: they take agent state, constitution, and an RNG,
and return decisions without mutating any inputs.
"""

from __future__ import annotations

import structlog

from emergent_constitution.models.constitution import (
    ConstitutionV2,
    RuleType,
)
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.proposal import ConstitutionalProposal
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()

# Probability of an agent skipping proposal generation (controls density).
_PROPOSAL_SKIP_PROBABILITY = 0.85


def decide_proposal_v2(
    household: HouseholdState,
    constitution: ConstitutionV2,
    config_num_agents: int,
    rng: SimulationRNG,
) -> ConstitutionalProposal | None:
    """Decide whether to propose a constitutional rule change (v2).

    Algorithm:
        1. Probabilistic gating (~15% chance of proposing).
        2. Evaluate dissatisfaction with current rules based on value vector.
        3. Pick highest-dissatisfaction rule.
        4. Propose a modification aligned with agent's values.

    Args:
        household: The household agent state.
        constitution: The current v2 constitution.
        config_num_agents: Number of agents (unused, for future scaling).
        rng: Seeded RNG for deterministic gating.

    Returns:
        A ConstitutionalProposal, or None if the agent chooses not to propose.
    """
    _ = config_num_agents

    # Step 1: probabilistic gating
    if rng.random() < _PROPOSAL_SKIP_PROBABILITY:
        return None

    eq = household.value_vector.equality

    # Step 2: evaluate dissatisfaction with each modifiable rule
    dissatisfaction: list[tuple[float, str, dict]] = []

    # Tax rate dissatisfaction
    tax_rules = constitution.get_tax_rules()
    if tax_rules:
        tax_rule = tax_rules[0]
        current_rate = float(tax_rule.parameters.get("rate", 0.0))
        preferred_rate = eq * 0.5  # equality-lovers want ~50%, liberty-lovers want ~0%
        tax_dissatisfaction = abs(current_rate - preferred_rate)

        if tax_dissatisfaction > 0.05:
            # Propose rate change toward preferred
            new_rate = round(current_rate + (preferred_rate - current_rate) * 0.3, 3)
            new_rate = max(0.0, min(1.0, new_rate))
            dissatisfaction.append(
                (
                    tax_dissatisfaction,
                    tax_rule.name,
                    {"rate": new_rate},
                )
            )

    # Transfer program dissatisfaction
    transfer_rules = constitution.get_transfer_rules()
    if transfer_rules:
        transfer_rule = transfer_rules[0]
        current_method = transfer_rule.parameters.get("method", "equal_share")
        # Equality-lovers prefer progressive, liberty-lovers prefer equal_share or none
        if eq > 0.6 and current_method != "progressive":
            dissatisfaction.append(
                (
                    eq * 0.4,
                    transfer_rule.name,
                    {"method": "progressive"},
                )
            )
        elif eq < 0.4 and current_method == "progressive":
            dissatisfaction.append(
                (
                    (1.0 - eq) * 0.4,
                    transfer_rule.name,
                    {"method": "equal_share"},
                )
            )

    # Public goods fraction dissatisfaction
    for rule in constitution.rules.values():
        if rule.rule_type == RuleType.PUBLIC_GOODS:
            current_fraction = float(rule.parameters.get("fraction_of_revenue", 0.3))
            # Agents with high gamma (public goods preference) want higher fraction
            preferred_fraction = household.utility_params.gamma * 0.8
            pg_dissatisfaction = abs(current_fraction - preferred_fraction)
            if pg_dissatisfaction > 0.05:
                new_fraction = round(
                    current_fraction + (preferred_fraction - current_fraction) * 0.3, 3
                )
                new_fraction = max(0.05, min(0.8, new_fraction))
                dissatisfaction.append(
                    (
                        pg_dissatisfaction,
                        rule.name,
                        {"fraction_of_revenue": new_fraction},
                    )
                )

    # Voting threshold dissatisfaction
    voting_rule = constitution.get_active_voting_rule()
    if voting_rule is not None:
        current_threshold = float(voting_rule.parameters.get("threshold", 0.5))
        # Equality-lovers prefer supermajority (harder to change status quo)
        # Liberty-lovers prefer simple majority (easier to reform)
        preferred_threshold = 0.5 + eq * 0.2  # range: 0.5 to 0.7
        threshold_dissatisfaction = abs(current_threshold - preferred_threshold)
        if threshold_dissatisfaction > 0.05:
            new_threshold = round(
                current_threshold + (preferred_threshold - current_threshold) * 0.3, 3
            )
            new_threshold = max(0.5, min(0.9, new_threshold))
            dissatisfaction.append(
                (
                    threshold_dissatisfaction * 0.5,  # Lower priority for voting rule changes
                    voting_rule.name,
                    {"threshold": new_threshold},
                )
            )

    if not dissatisfaction:
        return None

    # Step 3: pick highest dissatisfaction (alphabetical tie-break on rule name)
    dissatisfaction.sort(key=lambda x: (-x[0], x[1]))
    _, rule_name, new_params = dissatisfaction[0]

    return ConstitutionalProposal(
        proposer_id=household.id,
        action="modify",
        rule_name=rule_name,
        parameters=new_params,
        description=f"Agent {household.id} proposes modifying {rule_name}",
    )


def decide_votes_v2(
    household: HouseholdState,
    proposals: list[ConstitutionalProposal],
    constitution: ConstitutionV2,
) -> dict[str, bool]:
    """Decide how to vote on each v2 constitutional proposal.

    Uses value alignment and self-interest to determine votes.
    Score = 0.6 * value_alignment + 0.4 * self_interest.
    Vote "for" if score > 0.5.

    Args:
        household: The household agent state.
        proposals: Proposals to vote on.
        constitution: The current constitution.

    Returns:
        Mapping of proposal_id (fallback: rule_name) -> vote (True=for, False=against).
    """
    if not proposals:
        return {}

    eq = household.value_vector.equality
    votes: dict[str, bool] = {}

    for proposal in proposals:
        value_alignment = _compute_value_alignment_v2(proposal, constitution, eq)
        self_interest = _compute_self_interest_v2(proposal, household)
        score = 0.6 * value_alignment + 0.4 * self_interest
        vote_key = proposal.proposal_id or proposal.rule_name
        votes[vote_key] = score > 0.5

    return votes


def _compute_value_alignment_v2(
    proposal: ConstitutionalProposal,
    constitution: ConstitutionV2,
    equality: float,
) -> float:
    """How well does the proposal align with the agent's values?

    Returns a score in [0, 1] where 1 = perfectly aligned.
    """
    rule = constitution.rules.get(proposal.rule_name)
    if rule is None or proposal.parameters is None:
        return 0.5

    if rule.rule_type == RuleType.TAX_SCHEDULE:
        # Tax rate changes
        current_rate = float(rule.parameters.get("rate", 0.0))
        new_rate = float(proposal.parameters.get("rate", current_rate))
        preferred_rate = equality * 0.5

        current_dist = abs(current_rate - preferred_rate)
        proposed_dist = abs(new_rate - preferred_rate)

        if current_dist < 0.01:
            return 0.0 if proposed_dist > current_dist else 0.5
        improvement = (current_dist - proposed_dist) / max(current_dist, 0.01)
        return max(0.0, min(1.0, 0.5 + 0.5 * improvement))

    if rule.rule_type == RuleType.TRANSFER_PROGRAM:
        new_method = proposal.parameters.get("method")
        if new_method == "progressive":
            return equality  # equality-lovers like progressive
        if new_method in ("none", "equal_share"):
            return 1.0 - equality
        return 0.5

    if rule.rule_type == RuleType.PUBLIC_GOODS:
        current_frac = float(rule.parameters.get("fraction_of_revenue", 0.3))
        new_frac = float(proposal.parameters.get("fraction_of_revenue", current_frac))
        # Most agents benefit from some public goods; prefer moderate levels
        if abs(new_frac - 0.3) < abs(current_frac - 0.3):
            return 0.7
        return 0.3

    if rule.rule_type == RuleType.VOTING_PROCEDURE:
        current_threshold = float(rule.parameters.get("threshold", 0.5))
        new_threshold = float(proposal.parameters.get("threshold", current_threshold))
        preferred_threshold = 0.5 + equality * 0.2
        current_dist = abs(current_threshold - preferred_threshold)
        proposed_dist = abs(new_threshold - preferred_threshold)
        if current_dist < 0.01:
            return 0.5
        improvement = (current_dist - proposed_dist) / max(current_dist, 0.01)
        return max(0.0, min(1.0, 0.5 + 0.5 * improvement))

    return 0.5


def _compute_self_interest_v2(
    proposal: ConstitutionalProposal,
    household: HouseholdState,
) -> float:
    """How much does the proposal serve the agent's material self-interest?

    Returns a score in [0, 1] where 1 = strongly beneficial.
    """
    if proposal.parameters is None:
        return 0.5

    rule_name = proposal.rule_name

    # Tax changes: poor agents favor higher taxes, rich agents favor lower
    if "tax" in rule_name.lower():
        new_rate = proposal.parameters.get("rate")
        if new_rate is not None:
            new_rate_f = float(new_rate)
            # Use wealth as proxy for "rich vs poor" — wealthier agents dislike higher taxes
            # Normalize wealth: assume typical range [0, 500]
            wealth_factor = min(household.wealth / 500.0, 1.0)
            if new_rate_f > 0.2:
                return 1.0 - wealth_factor  # poor benefit, rich lose
            return wealth_factor  # rich benefit, poor lose

    # Transfer changes
    if "transfer" in rule_name.lower():
        method = proposal.parameters.get("method")
        wealth_factor = min(household.wealth / 500.0, 1.0)
        if method == "progressive":
            return 1.0 - wealth_factor  # poor benefit
        if method in ("none", "equal_share"):
            return wealth_factor
        return 0.5

    # Public goods: most agents benefit moderately
    if "public" in rule_name.lower():
        gamma = household.utility_params.gamma
        return gamma  # Agents who value public goods benefit

    return 0.5
