"""Governance mechanics — proposal validation, vote tallying, and constitution updates.

All functions are pure: they take inputs and return outputs without mutating arguments.
"""

from __future__ import annotations

import math

import structlog

from emergent_constitution.models.constitution import (
    CONSTITUTION_FIELDS,
    Constitution,
    VotingRule,
)
from emergent_constitution.models.proposal import Proposal, VoteOutcome

log = structlog.get_logger()


def validate_proposal(proposal: Proposal) -> bool:
    """Check that a proposal targets a valid constitution field with a valid value.

    Args:
        proposal: The proposal to validate.

    Returns:
        True if the proposal is valid, False otherwise.
    """
    if proposal.rule_key not in CONSTITUTION_FIELDS:
        log.debug("proposal.invalid_key", rule_key=proposal.rule_key)
        return False

    expected_type = CONSTITUTION_FIELDS[proposal.rule_key]
    value = proposal.proposed_value

    if expected_type is float:
        # Accept int or float, must be in [0, 1] for tax_rate
        if not isinstance(value, int | float):
            log.debug("proposal.invalid_type", rule_key=proposal.rule_key, value=value)
            return False
        if math.isnan(value) or math.isinf(value):
            log.debug("proposal.nan_or_inf", rule_key=proposal.rule_key, value=value)
            return False
        if not 0.0 <= float(value) <= 1.0:
            log.debug("proposal.out_of_range", rule_key=proposal.rule_key, value=value)
            return False
    else:
        # Enum field — accept the enum member or its string value
        if isinstance(value, expected_type):
            return True
        if isinstance(value, str):
            try:
                expected_type(value)
            except ValueError:
                log.debug("proposal.invalid_enum", rule_key=proposal.rule_key, value=value)
                return False
        else:
            log.debug("proposal.invalid_type", rule_key=proposal.rule_key, value=value)
            return False

    return True


def tally_votes(
    proposal: Proposal,
    votes: dict[str, bool],
    voting_rule: VotingRule,
    total_eligible: int,
) -> VoteOutcome:
    """Count votes and determine whether a proposal passes.

    Thresholds:
        MAJORITY:       votes_for > total_eligible / 2  (tie = fail)
        SUPERMAJORITY:  votes_for >= ceil(2 * total_eligible / 3)
        UNANIMITY:      votes_for == total_eligible

    Args:
        proposal: The proposal being voted on.
        votes: Mapping of agent_id -> True (for) / False (against).
        voting_rule: The current voting rule.
        total_eligible: Total number of eligible voters.

    Returns:
        VoteOutcome with pass/fail determination.
    """
    votes_for = sum(1 for v in votes.values() if v)
    votes_against = sum(1 for v in votes.values() if not v)

    if voting_rule == VotingRule.MAJORITY:
        passed = votes_for > total_eligible / 2
    elif voting_rule == VotingRule.SUPERMAJORITY:
        threshold = math.ceil(2 * total_eligible / 3)
        passed = votes_for >= threshold
    elif voting_rule == VotingRule.UNANIMITY:
        passed = votes_for == total_eligible
    else:
        passed = False

    return VoteOutcome(
        proposal=proposal,
        passed=passed,
        votes_for=votes_for,
        votes_against=votes_against,
        total_eligible=total_eligible,
    )


def apply_passed_proposals(
    vote_outcomes: list[VoteOutcome],
    constitution: Constitution,
) -> Constitution:
    """Apply all passed proposals to the constitution.

    Proposals are applied in order; if multiple proposals target the same field,
    the last one wins. Returns a new Constitution — never mutates the input.

    Args:
        vote_outcomes: List of vote outcomes to process.
        constitution: The current constitution.

    Returns:
        A new Constitution with passed proposals applied.
    """
    updates: dict[str, object] = {}

    for outcome in vote_outcomes:
        if not outcome.passed:
            continue

        key = outcome.proposal.rule_key
        value = outcome.proposal.proposed_value
        expected_type = CONSTITUTION_FIELDS.get(key)

        if expected_type is None:
            continue

        # Coerce to the correct type
        if expected_type is float:
            value = float(value)
        elif isinstance(value, str) and not isinstance(value, expected_type):
            value = expected_type(value)

        updates[key] = value
        log.info(
            "constitution.updated",
            rule_key=key,
            new_value=str(value),
            proposer=outcome.proposal.proposer_id,
        )

    if not updates:
        return constitution

    return constitution.model_copy(update=updates)
