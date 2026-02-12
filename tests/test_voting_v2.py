"""Tests for v2 voting functions — extensible constitutional governance.

Covers:
- validate_proposal_v2: add/modify/remove validation (14.2)
- tally_votes_v2: majority, supermajority, unanimity, custom threshold (14.1)
- apply_passed_proposals_v2: integration with ConstitutionEngine (14.1)
- _resolve_voting_rule: fallback behavior
- _evaluate_threshold: threshold extraction from rule parameters

Traceability: REQ-019, REQ-020
"""

from __future__ import annotations

import math

import pytest

from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.proposal import ConstitutionalProposal, VoteOutcomeV2
from emergent_constitution.voting import (
    _evaluate_threshold,
    _resolve_voting_rule,
    apply_passed_proposals_v2,
    tally_votes_v2,
    validate_proposal_v2,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def constitution() -> ConstitutionV2:
    """Default constitution with majority voting rule."""
    return create_default_constitution()


@pytest.fixture()
def supermajority_constitution() -> ConstitutionV2:
    """Constitution with 2/3 supermajority voting rule."""
    c = create_default_constitution()
    c.rules["majority_vote"] = ConstitutionalRule(
        name="majority_vote",
        rule_type=RuleType.VOTING_PROCEDURE,
        parameters={"type": "supermajority", "threshold": 2.0 / 3.0},
        description="Supermajority voting rule.",
    )
    return c


@pytest.fixture()
def unanimity_constitution() -> ConstitutionV2:
    """Constitution with unanimity voting rule."""
    c = create_default_constitution()
    c.rules["majority_vote"] = ConstitutionalRule(
        name="majority_vote",
        rule_type=RuleType.VOTING_PROCEDURE,
        parameters={"type": "unanimity"},
        description="Unanimity voting rule.",
    )
    return c


@pytest.fixture()
def custom_threshold_constitution() -> ConstitutionV2:
    """Constitution with custom 75% threshold voting rule."""
    c = create_default_constitution()
    c.rules["majority_vote"] = ConstitutionalRule(
        name="majority_vote",
        rule_type=RuleType.VOTING_PROCEDURE,
        parameters={"threshold": 0.75},
        description="Custom 75% threshold.",
    )
    return c


def _make_add_proposal(**kwargs) -> ConstitutionalProposal:
    """Helper to create an 'add' proposal with defaults."""
    defaults = {
        "proposer_id": "agent_1",
        "action": "add",
        "rule_name": "new_rule",
        "rule_type": RuleType.TAX_SCHEDULE,
        "parameters": {"rate": 0.1},
        "description": "A new tax rule.",
    }
    defaults.update(kwargs)
    return ConstitutionalProposal(**defaults)


def _make_modify_proposal(**kwargs) -> ConstitutionalProposal:
    """Helper to create a 'modify' proposal with defaults."""
    defaults = {
        "proposer_id": "agent_1",
        "action": "modify",
        "rule_name": "flat_tax",
        "parameters": {"rate": 0.15},
        "description": "Increase flat tax.",
    }
    defaults.update(kwargs)
    return ConstitutionalProposal(**defaults)


def _make_remove_proposal(**kwargs) -> ConstitutionalProposal:
    """Helper to create a 'remove' proposal with defaults."""
    defaults = {
        "proposer_id": "agent_1",
        "action": "remove",
        "rule_name": "flat_tax",
    }
    defaults.update(kwargs)
    return ConstitutionalProposal(**defaults)


def _make_outcome(
    proposal: ConstitutionalProposal,
    *,
    passed: bool,
    voting_rule_used: str = "majority_vote",
) -> VoteOutcomeV2:
    """Helper to create a VoteOutcomeV2."""
    return VoteOutcomeV2(
        proposal=proposal,
        passed=passed,
        votes_for=3 if passed else 1,
        votes_against=1 if passed else 3,
        total_eligible=4,
        voting_rule_used=voting_rule_used,
    )


# ---------------------------------------------------------------------------
# validate_proposal_v2
# ---------------------------------------------------------------------------


class TestValidateProposalV2:
    def test_valid_add(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal()
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is True
        assert reason == ""

    def test_add_existing_rule_rejected(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal(rule_name="flat_tax")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "already exists" in reason

    def test_add_missing_rule_type_rejected(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal(rule_type=None)
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "rule_type is required" in reason

    def test_valid_modify(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is True

    def test_modify_nonexistent_rule_rejected(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal(rule_name="nonexistent")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "does not exist" in reason

    def test_modify_without_changes_rejected(self, constitution: ConstitutionV2):
        proposal = ConstitutionalProposal(
            proposer_id="agent_1",
            action="modify",
            rule_name="flat_tax",
            parameters=None,
            enforcement_code=None,
        )
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "requires parameters or enforcement_code" in reason

    def test_modify_with_enforcement_code_only(self, constitution: ConstitutionV2):
        proposal = ConstitutionalProposal(
            proposer_id="agent_1",
            action="modify",
            rule_name="flat_tax",
            enforcement_code="tax = income * 0.2",
        )
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is True

    def test_valid_remove(self, constitution: ConstitutionV2):
        proposal = _make_remove_proposal()
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is True

    def test_remove_nonexistent_rule_rejected(self, constitution: ConstitutionV2):
        proposal = _make_remove_proposal(rule_name="nonexistent")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "does not exist" in reason

    def test_remove_active_voting_rule_rejected(self, constitution: ConstitutionV2):
        proposal = _make_remove_proposal(rule_name="majority_vote")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "Cannot remove the active voting rule" in reason

    def test_empty_rule_name_rejected(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal(rule_name="")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "non-empty" in reason

    def test_whitespace_rule_name_rejected(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal(rule_name="   ")
        valid, reason = validate_proposal_v2(proposal, constitution)
        assert valid is False
        assert "non-empty" in reason


# ---------------------------------------------------------------------------
# _resolve_voting_rule
# ---------------------------------------------------------------------------


class TestResolveVotingRule:
    def test_resolves_named_rule(self, constitution: ConstitutionV2):
        rule = _resolve_voting_rule(constitution)
        assert rule is not None
        assert rule.name == "majority_vote"
        assert rule.rule_type == RuleType.VOTING_PROCEDURE

    def test_fallback_to_any_voting_procedure(self):
        """When named rule doesn't match, find any VOTING_PROCEDURE."""
        c = ConstitutionV2(
            rules={
                "custom_vote": ConstitutionalRule(
                    name="custom_vote",
                    rule_type=RuleType.VOTING_PROCEDURE,
                    parameters={"threshold": 0.6},
                ),
            },
            voting_rule="nonexistent",
        )
        rule = _resolve_voting_rule(c)
        assert rule is not None
        assert rule.name == "custom_vote"

    def test_returns_none_when_no_voting_rules(self):
        c = ConstitutionV2(
            rules={
                "some_tax": ConstitutionalRule(
                    name="some_tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.1},
                ),
            },
            voting_rule="nonexistent",
        )
        rule = _resolve_voting_rule(c)
        assert rule is None


# ---------------------------------------------------------------------------
# _evaluate_threshold
# ---------------------------------------------------------------------------


class TestEvaluateThreshold:
    def test_majority_default(self):
        rule = ConstitutionalRule(
            name="majority",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.5},
        )
        assert _evaluate_threshold(rule) == 0.5

    def test_supermajority_by_type(self):
        rule = ConstitutionalRule(
            name="supermajority",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"type": "supermajority"},
        )
        assert _evaluate_threshold(rule) == pytest.approx(2.0 / 3.0)

    def test_supermajority_with_custom_threshold(self):
        rule = ConstitutionalRule(
            name="supermajority",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"type": "supermajority", "threshold": 0.75},
        )
        assert _evaluate_threshold(rule) == 0.75

    def test_unanimity_by_type(self):
        rule = ConstitutionalRule(
            name="unanimity",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"type": "unanimity"},
        )
        assert _evaluate_threshold(rule) == 1.0

    def test_numeric_custom_threshold(self):
        rule = ConstitutionalRule(
            name="custom",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.6},
        )
        assert _evaluate_threshold(rule) == 0.6

    def test_invalid_threshold_defaults_to_majority(self):
        rule = ConstitutionalRule(
            name="broken",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": "not_a_number"},
        )
        assert _evaluate_threshold(rule) == 0.5

    def test_zero_threshold_defaults_to_majority(self):
        rule = ConstitutionalRule(
            name="zero",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.0},
        )
        assert _evaluate_threshold(rule) == 0.5

    def test_over_one_threshold_defaults_to_majority(self):
        rule = ConstitutionalRule(
            name="over",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 1.5},
        )
        assert _evaluate_threshold(rule) == 0.5

    def test_empty_params_defaults_to_majority(self):
        rule = ConstitutionalRule(
            name="empty",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={},
        )
        assert _evaluate_threshold(rule) == 0.5

    def test_threshold_exactly_one_is_valid(self):
        rule = ConstitutionalRule(
            name="unanimity_numeric",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 1.0},
        )
        assert _evaluate_threshold(rule) == 1.0


# ---------------------------------------------------------------------------
# tally_votes_v2 — majority
# ---------------------------------------------------------------------------


class TestTallyVotesV2Majority:
    def test_majority_pass(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes_v2(proposal, votes, constitution, total_eligible=3)
        assert result.passed is True
        assert result.votes_for == 2
        assert result.votes_against == 1
        assert result.voting_rule_used == "majority_vote"

    def test_majority_fail(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": False, "a3": False}
        result = tally_votes_v2(proposal, votes, constitution, total_eligible=3)
        assert result.passed is False

    def test_majority_tie_fails(self, constitution: ConstitutionV2):
        """50/50 tie means status quo wins (proposal fails)."""
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": False, "a3": True, "a4": False}
        result = tally_votes_v2(proposal, votes, constitution, total_eligible=4)
        assert result.passed is False

    def test_empty_votes_fail(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        result = tally_votes_v2(proposal, {}, constitution, total_eligible=3)
        assert result.passed is False
        assert result.votes_for == 0

    def test_total_eligible_preserved(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True}
        result = tally_votes_v2(proposal, votes, constitution, total_eligible=10)
        assert result.total_eligible == 10
        assert result.passed is False


# ---------------------------------------------------------------------------
# tally_votes_v2 — supermajority
# ---------------------------------------------------------------------------


class TestTallyVotesV2Supermajority:
    def test_supermajority_pass(self, supermajority_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        # 3 out of 4: ceil(4 * 2/3) = ceil(2.667) = 3
        votes = {"a1": True, "a2": True, "a3": True, "a4": False}
        result = tally_votes_v2(proposal, votes, supermajority_constitution, total_eligible=4)
        assert result.passed is True
        assert result.votes_for == 3

    def test_supermajority_fail(self, supermajority_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": False, "a4": False}
        result = tally_votes_v2(proposal, votes, supermajority_constitution, total_eligible=4)
        assert result.passed is False

    def test_supermajority_boundary(self, supermajority_constitution: ConstitutionV2):
        """With 3 eligible, threshold 2/3: ceil(3 * 2/3) = ceil(2) = 2."""
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes_v2(proposal, votes, supermajority_constitution, total_eligible=3)
        assert result.passed is True


# ---------------------------------------------------------------------------
# tally_votes_v2 — unanimity
# ---------------------------------------------------------------------------


class TestTallyVotesV2Unanimity:
    def test_unanimity_pass(self, unanimity_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": True}
        result = tally_votes_v2(proposal, votes, unanimity_constitution, total_eligible=3)
        assert result.passed is True

    def test_unanimity_fail(self, unanimity_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes_v2(proposal, votes, unanimity_constitution, total_eligible=3)
        assert result.passed is False

    def test_unanimity_partial_turnout_fails(self, unanimity_constitution: ConstitutionV2):
        """Even if all voters say yes, must equal total_eligible."""
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True}
        result = tally_votes_v2(proposal, votes, unanimity_constitution, total_eligible=3)
        assert result.passed is False


# ---------------------------------------------------------------------------
# tally_votes_v2 — custom threshold
# ---------------------------------------------------------------------------


class TestTallyVotesV2CustomThreshold:
    def test_custom_75_pass(self, custom_threshold_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        # 75% of 4 = 3: ceil(4 * 0.75) = 3
        votes = {"a1": True, "a2": True, "a3": True, "a4": False}
        result = tally_votes_v2(proposal, votes, custom_threshold_constitution, total_eligible=4)
        assert result.passed is True

    def test_custom_75_fail(self, custom_threshold_constitution: ConstitutionV2):
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": False, "a4": False}
        result = tally_votes_v2(proposal, votes, custom_threshold_constitution, total_eligible=4)
        assert result.passed is False

    def test_custom_threshold_boundary(self):
        """Custom 60% threshold: ceil(5 * 0.6) = 3."""
        c = create_default_constitution()
        c.rules["majority_vote"] = ConstitutionalRule(
            name="majority_vote",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.6},
        )
        proposal = _make_modify_proposal()
        votes = {"a1": True, "a2": True, "a3": True, "a4": False, "a5": False}
        result = tally_votes_v2(proposal, votes, c, total_eligible=5)
        assert result.passed is True
        assert result.votes_for == 3
        # Verify: ceil(5 * 0.6) = ceil(3.0) = 3, and 3 >= 3
        assert result.votes_for >= math.ceil(5 * 0.6)


# ---------------------------------------------------------------------------
# tally_votes_v2 — no voting rule
# ---------------------------------------------------------------------------


class TestTallyVotesV2NoVotingRule:
    def test_defaults_to_majority_when_no_rule(self):
        """When no voting procedure exists, default to simple majority."""
        c = ConstitutionV2(
            rules={
                "tax": ConstitutionalRule(
                    name="tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.1},
                ),
            },
            voting_rule="nonexistent",
        )
        proposal = _make_modify_proposal(rule_name="tax")
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes_v2(proposal, votes, c, total_eligible=3)
        assert result.passed is True
        assert result.voting_rule_used == "default_majority"


# ---------------------------------------------------------------------------
# apply_passed_proposals_v2
# ---------------------------------------------------------------------------


class TestApplyPassedProposalsV2:
    def test_add_rule(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal()
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals_v2([outcome], constitution)
        assert "new_rule" in result.rules

    def test_modify_rule(self, constitution: ConstitutionV2):
        proposal = _make_modify_proposal(parameters={"rate": 0.25})
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals_v2([outcome], constitution)
        assert result.rules["flat_tax"].parameters["rate"] == 0.25

    def test_remove_rule(self, constitution: ConstitutionV2):
        proposal = _make_remove_proposal(rule_name="flat_tax")
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals_v2([outcome], constitution)
        assert "flat_tax" not in result.rules

    def test_failed_proposal_ignored(self, constitution: ConstitutionV2):
        proposal = _make_add_proposal()
        outcome = _make_outcome(proposal, passed=False)
        result = apply_passed_proposals_v2([outcome], constitution)
        assert "new_rule" not in result.rules

    def test_no_mutation_of_input(self, constitution: ConstitutionV2):
        original_rules = set(constitution.rules.keys())
        proposal = _make_add_proposal()
        outcome = _make_outcome(proposal, passed=True)
        apply_passed_proposals_v2([outcome], constitution)
        assert set(constitution.rules.keys()) == original_rules

    def test_multiple_proposals_applied_in_order(self, constitution: ConstitutionV2):
        p1 = _make_add_proposal(rule_name="extra_tax", parameters={"rate": 0.1})
        p2 = _make_modify_proposal(rule_name="extra_tax", parameters={"rate": 0.2})
        outcomes = [
            _make_outcome(p1, passed=True),
            _make_outcome(p2, passed=True),
        ]
        result = apply_passed_proposals_v2(outcomes, constitution)
        assert "extra_tax" in result.rules
        assert result.rules["extra_tax"].parameters["rate"] == 0.2

    def test_empty_outcomes(self, constitution: ConstitutionV2):
        result = apply_passed_proposals_v2([], constitution)
        assert result.rules.keys() == constitution.rules.keys()

    def test_invalid_proposal_logged_not_crashed(self, constitution: ConstitutionV2):
        """If apply_proposal raises ValueError, it's logged and skipped."""
        proposal = _make_add_proposal(rule_name="flat_tax")
        # flat_tax already exists, so add will fail in ConstitutionEngine
        outcome = _make_outcome(proposal, passed=True)
        # Should not raise — error is logged
        result = apply_passed_proposals_v2([outcome], constitution)
        # Constitution unchanged for this proposal
        assert result.rules["flat_tax"].parameters == constitution.rules["flat_tax"].parameters
