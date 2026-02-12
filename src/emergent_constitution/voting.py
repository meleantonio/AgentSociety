"""Governance mechanics — proposal validation, vote tallying, and constitution updates.

All functions are pure: they take inputs and return outputs without mutating arguments.

v1 functions operate on fixed Constitution/Proposal/VoteOutcome types.
v2 functions operate on extensible ConstitutionV2/ConstitutionalProposal/VoteOutcomeV2
types and read voting rules from the constitution itself (REQ-019, REQ-020).
"""

from __future__ import annotations

import math

import structlog

from emergent_constitution.models.constitution import (
    CONSTITUTION_FIELDS,
    Constitution,
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    VotingRule,
)
from emergent_constitution.models.proposal import (
    ConstitutionalProposal,
    Proposal,
    VoteOutcome,
    VoteOutcomeV2,
)

log = structlog.get_logger()

# ============================================================================
# v1 functions (backward compatibility)
# ============================================================================


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


# ============================================================================
# v2 functions (extensible constitutional governance)
# ============================================================================


def validate_proposal_v2(
    proposal: ConstitutionalProposal,
    constitution: ConstitutionV2,
) -> tuple[bool, str]:
    """Validate a v2 constitutional proposal.

    Checks:
    - Action is valid ("add", "modify", "remove").
    - For "add": rule_name must not already exist; rule_type is required.
    - For "modify": rule_name must exist in the constitution.
    - For "remove": rule_name must exist; cannot remove the active voting rule.
    - Parameters, if provided, must be a dict with string keys.

    Args:
        proposal: The proposal to validate.
        constitution: The current constitution for context.

    Returns:
        Tuple of (is_valid, reason). reason is empty string if valid.
    """
    if proposal.action not in ("add", "modify", "remove"):
        return False, f"Invalid action: {proposal.action}"

    if not proposal.rule_name or not proposal.rule_name.strip():
        return False, "rule_name must be non-empty"

    if proposal.action == "add":
        if proposal.rule_name in constitution.rules:
            return False, f"Rule '{proposal.rule_name}' already exists; use 'modify'"
        if proposal.rule_type is None:
            return False, "rule_type is required for 'add' action"

    elif proposal.action == "modify":
        if proposal.rule_name not in constitution.rules:
            return False, f"Rule '{proposal.rule_name}' does not exist; use 'add'"
        if proposal.parameters is None and proposal.enforcement_code is None:
            return False, "modify requires parameters or enforcement_code"

    elif proposal.action == "remove":
        if proposal.rule_name not in constitution.rules:
            return False, f"Rule '{proposal.rule_name}' does not exist"
        if proposal.rule_name == constitution.voting_rule:
            return False, "Cannot remove the active voting rule"

    if proposal.parameters is not None and not isinstance(proposal.parameters, dict):
        return False, "parameters must be a dict"

    log.debug(
        "proposal_v2.valid",
        proposer=proposal.proposer_id,
        action=proposal.action,
        rule_name=proposal.rule_name,
    )
    return True, ""


def _resolve_voting_rule(constitution: ConstitutionV2) -> ConstitutionalRule | None:
    """Look up the active voting procedure rule from the constitution.

    Falls back to searching for any VOTING_PROCEDURE rule if the named
    rule is not found.

    Args:
        constitution: The current constitution.

    Returns:
        The active voting rule, or None if no voting procedure exists.
    """
    rule = constitution.get_active_voting_rule()
    if rule is not None:
        return rule

    # Fallback: find any voting procedure rule
    for r in constitution.rules.values():
        if r.rule_type == RuleType.VOTING_PROCEDURE:
            log.warning(
                "voting.fallback_rule",
                expected=constitution.voting_rule,
                found=r.name,
            )
            return r

    return None


def _evaluate_threshold(rule: ConstitutionalRule) -> float:
    """Extract the voting threshold from a voting procedure rule.

    Supports named thresholds ("majority", "supermajority", "unanimity")
    and numeric thresholds from the rule parameters.

    Args:
        rule: A voting procedure rule.

    Returns:
        Threshold as a fraction in (0.0, 1.0].
    """
    params = rule.parameters

    # Check for named threshold type
    threshold_type = params.get("type", "").lower()
    if threshold_type == "unanimity":
        return 1.0
    if threshold_type == "supermajority":
        return params.get("threshold", 2.0 / 3.0)

    # Numeric threshold from parameters
    threshold = params.get("threshold")
    if threshold is not None:
        try:
            val = float(threshold)
            if 0.0 < val <= 1.0:
                return val
        except (TypeError, ValueError):
            pass

    # Default to simple majority
    return 0.5


def tally_votes_v2(
    proposal: ConstitutionalProposal,
    votes: dict[str, bool],
    constitution: ConstitutionV2,
    total_eligible: int,
) -> VoteOutcomeV2:
    """Count votes and determine whether a v2 proposal passes.

    Reads the voting rule from the constitution. Supports:
    - Majority (threshold = 0.5): votes_for > total_eligible * threshold
    - Supermajority (threshold ~0.667): votes_for >= ceil(total_eligible * threshold)
    - Unanimity (threshold = 1.0): votes_for == total_eligible
    - Custom threshold: votes_for >= ceil(total_eligible * threshold)

    The threshold is extracted from the active voting procedure rule's
    parameters. Ties always fail (status quo wins).

    Args:
        proposal: The proposal being voted on.
        votes: Mapping of agent_id -> True (for) / False (against).
        constitution: The current constitution (provides voting rule).
        total_eligible: Total number of eligible voters.

    Returns:
        VoteOutcomeV2 with pass/fail determination and voting rule name.
    """
    votes_for = sum(1 for v in votes.values() if v)
    votes_against = sum(1 for v in votes.values() if not v)

    voting_rule = _resolve_voting_rule(constitution)
    voting_rule_name = voting_rule.name if voting_rule is not None else "default_majority"

    if voting_rule is not None:
        threshold = _evaluate_threshold(voting_rule)
    else:
        # No voting rule found — default to simple majority
        log.warning("voting.no_rule_found", default="majority")
        threshold = 0.5

    if threshold >= 1.0:
        # Unanimity: all eligible must vote for
        passed = votes_for == total_eligible
    elif threshold > 0.5:
        # Supermajority or custom high threshold: use ceil
        required = math.ceil(total_eligible * threshold)
        passed = votes_for >= required
    else:
        # Simple majority: strict > (ties fail)
        passed = votes_for > total_eligible * threshold

    log.debug(
        "voting.tally_v2",
        rule=voting_rule_name,
        threshold=threshold,
        votes_for=votes_for,
        votes_against=votes_against,
        total_eligible=total_eligible,
        passed=passed,
    )

    return VoteOutcomeV2(
        proposal=proposal,
        passed=passed,
        votes_for=votes_for,
        votes_against=votes_against,
        total_eligible=total_eligible,
        voting_rule_used=voting_rule_name,
    )


def apply_passed_proposals_v2(
    vote_outcomes: list[VoteOutcomeV2],
    constitution: ConstitutionV2,
) -> ConstitutionV2:
    """Apply all passed v2 proposals to the constitution.

    Proposals are applied in order. Returns a new ConstitutionV2 — never
    mutates the input.

    Uses ConstitutionEngine.apply_proposal for each passed proposal to
    ensure validation and safety checks are applied.

    Args:
        vote_outcomes: List of vote outcomes to process.
        constitution: The current constitution.

    Returns:
        A new ConstitutionV2 with passed proposals applied.
    """
    from emergent_constitution.constitution_engine import ConstitutionEngine

    engine = ConstitutionEngine()
    updated = constitution.model_copy(deep=True)

    for outcome in vote_outcomes:
        if not outcome.passed:
            continue

        try:
            updated = engine.apply_proposal(updated, outcome.proposal)
            log.info(
                "constitution_v2.updated",
                action=outcome.proposal.action,
                rule_name=outcome.proposal.rule_name,
                proposer=outcome.proposal.proposer_id,
                voting_rule=outcome.voting_rule_used,
            )
        except ValueError as exc:
            log.warning(
                "constitution_v2.proposal_failed",
                action=outcome.proposal.action,
                rule_name=outcome.proposal.rule_name,
                error=str(exc),
            )

    return updated
