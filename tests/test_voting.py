"""Tests for the voting module — proposal validation, tallying, and application."""

from __future__ import annotations

import pytest

from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.models.proposal import Proposal
from emergent_constitution.voting import (
    apply_passed_proposals,
    tally_votes,
    validate_proposal,
)

# ---------------------------------------------------------------------------
# validate_proposal
# ---------------------------------------------------------------------------


class TestValidateProposal:
    def test_valid_tax_rate_float(self):
        p = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="a1")
        assert validate_proposal(p) is True

    def test_valid_tax_rate_int(self):
        """int is acceptable for a float field (e.g., 0 or 1)."""
        p = Proposal(rule_key="tax_rate", proposed_value=1, proposer_id="a1")
        assert validate_proposal(p) is True

    def test_invalid_tax_rate_too_high(self):
        p = Proposal(rule_key="tax_rate", proposed_value=1.5, proposer_id="a1")
        assert validate_proposal(p) is False

    def test_invalid_tax_rate_negative(self):
        p = Proposal(rule_key="tax_rate", proposed_value=-0.1, proposer_id="a1")
        assert validate_proposal(p) is False

    def test_invalid_tax_rate_nan(self):
        p = Proposal(rule_key="tax_rate", proposed_value=float("nan"), proposer_id="a1")
        assert validate_proposal(p) is False

    def test_invalid_tax_rate_inf(self):
        p = Proposal(rule_key="tax_rate", proposed_value=float("inf"), proposer_id="a1")
        assert validate_proposal(p) is False

    def test_invalid_tax_rate_string(self):
        p = Proposal(rule_key="tax_rate", proposed_value="high", proposer_id="a1")
        assert validate_proposal(p) is False

    def test_valid_property_rule_enum(self):
        p = Proposal(
            rule_key="property_rule",
            proposed_value=PropertyRule.COMMUNAL,
            proposer_id="a1",
        )
        assert validate_proposal(p) is True

    def test_valid_property_rule_string(self):
        p = Proposal(rule_key="property_rule", proposed_value="communal", proposer_id="a1")
        assert validate_proposal(p) is True

    def test_invalid_property_rule_string(self):
        p = Proposal(rule_key="property_rule", proposed_value="anarchy", proposer_id="a1")
        assert validate_proposal(p) is False

    def test_invalid_property_rule_int(self):
        p = Proposal(rule_key="property_rule", proposed_value=42, proposer_id="a1")
        assert validate_proposal(p) is False

    def test_valid_voting_rule(self):
        p = Proposal(rule_key="voting_rule", proposed_value="supermajority", proposer_id="a1")
        assert validate_proposal(p) is True

    def test_valid_redistribution_rule(self):
        p = Proposal(
            rule_key="redistribution_rule",
            proposed_value=RedistributionRule.PROGRESSIVE,
            proposer_id="a1",
        )
        assert validate_proposal(p) is True

    def test_invalid_rule_key(self):
        p = Proposal(rule_key="military_budget", proposed_value=0.5, proposer_id="a1")
        assert validate_proposal(p) is False


# ---------------------------------------------------------------------------
# tally_votes
# ---------------------------------------------------------------------------


class TestTallyVotes:
    @pytest.fixture()
    def proposal(self) -> Proposal:
        return Proposal(rule_key="tax_rate", proposed_value=0.2, proposer_id="a1")

    def test_majority_pass(self, proposal: Proposal):
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes(proposal, votes, VotingRule.MAJORITY, total_eligible=3)
        assert result.passed is True
        assert result.votes_for == 2
        assert result.votes_against == 1

    def test_majority_fail(self, proposal: Proposal):
        votes = {"a1": True, "a2": False, "a3": False}
        result = tally_votes(proposal, votes, VotingRule.MAJORITY, total_eligible=3)
        assert result.passed is False

    def test_majority_tie_fails(self, proposal: Proposal):
        """50/50 tie means status quo wins (proposal fails)."""
        votes = {"a1": True, "a2": False, "a3": True, "a4": False}
        result = tally_votes(proposal, votes, VotingRule.MAJORITY, total_eligible=4)
        assert result.passed is False

    def test_supermajority_pass(self, proposal: Proposal):
        votes = {"a1": True, "a2": True, "a3": True, "a4": False}
        # ceil(2*4/3) = ceil(8/3) = 3
        result = tally_votes(proposal, votes, VotingRule.SUPERMAJORITY, total_eligible=4)
        assert result.passed is True
        assert result.votes_for == 3

    def test_supermajority_fail(self, proposal: Proposal):
        votes = {"a1": True, "a2": True, "a3": False, "a4": False}
        result = tally_votes(proposal, votes, VotingRule.SUPERMAJORITY, total_eligible=4)
        assert result.passed is False

    def test_supermajority_boundary(self, proposal: Proposal):
        """With 3 eligible, threshold = ceil(6/3) = 2. Exactly 2 should pass."""
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes(proposal, votes, VotingRule.SUPERMAJORITY, total_eligible=3)
        assert result.passed is True

    def test_unanimity_pass(self, proposal: Proposal):
        votes = {"a1": True, "a2": True, "a3": True}
        result = tally_votes(proposal, votes, VotingRule.UNANIMITY, total_eligible=3)
        assert result.passed is True

    def test_unanimity_fail(self, proposal: Proposal):
        votes = {"a1": True, "a2": True, "a3": False}
        result = tally_votes(proposal, votes, VotingRule.UNANIMITY, total_eligible=3)
        assert result.passed is False

    def test_empty_votes(self, proposal: Proposal):
        result = tally_votes(proposal, {}, VotingRule.MAJORITY, total_eligible=3)
        assert result.passed is False
        assert result.votes_for == 0

    def test_total_eligible_preserved(self, proposal: Proposal):
        votes = {"a1": True}
        result = tally_votes(proposal, votes, VotingRule.MAJORITY, total_eligible=10)
        assert result.total_eligible == 10
        assert result.passed is False


# ---------------------------------------------------------------------------
# apply_passed_proposals
# ---------------------------------------------------------------------------


class TestApplyPassedProposals:
    def test_passed_proposal_applied(self):
        c = Constitution()
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.25, proposer_id="a1")
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals([outcome], c)
        assert result.tax_rate == 0.25

    def test_failed_proposal_ignored(self):
        c = Constitution()
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.25, proposer_id="a1")
        outcome = _make_outcome(proposal, passed=False)
        result = apply_passed_proposals([outcome], c)
        assert result.tax_rate == c.tax_rate

    def test_no_mutation_of_input(self):
        c = Constitution()
        original_tax = c.tax_rate
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.5, proposer_id="a1")
        outcome = _make_outcome(proposal, passed=True)
        apply_passed_proposals([outcome], c)
        assert c.tax_rate == original_tax

    def test_last_wins_for_same_field(self):
        c = Constitution()
        p1 = Proposal(rule_key="tax_rate", proposed_value=0.1, proposer_id="a1")
        p2 = Proposal(rule_key="tax_rate", proposed_value=0.9, proposer_id="a2")
        outcomes = [_make_outcome(p1, passed=True), _make_outcome(p2, passed=True)]
        result = apply_passed_proposals(outcomes, c)
        assert result.tax_rate == 0.9

    def test_enum_string_coercion(self):
        c = Constitution()
        proposal = Proposal(rule_key="property_rule", proposed_value="communal", proposer_id="a1")
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals([outcome], c)
        assert result.property_rule == PropertyRule.COMMUNAL

    def test_enum_member_applied(self):
        c = Constitution()
        proposal = Proposal(
            rule_key="redistribution_rule",
            proposed_value=RedistributionRule.PROGRESSIVE,
            proposer_id="a1",
        )
        outcome = _make_outcome(proposal, passed=True)
        result = apply_passed_proposals([outcome], c)
        assert result.redistribution_rule == RedistributionRule.PROGRESSIVE

    def test_empty_outcomes(self):
        c = Constitution()
        result = apply_passed_proposals([], c)
        assert result == c

    def test_multiple_fields_updated(self):
        c = Constitution()
        p1 = Proposal(rule_key="tax_rate", proposed_value=0.4, proposer_id="a1")
        p2 = Proposal(rule_key="voting_rule", proposed_value="supermajority", proposer_id="a2")
        outcomes = [_make_outcome(p1, passed=True), _make_outcome(p2, passed=True)]
        result = apply_passed_proposals(outcomes, c)
        assert result.tax_rate == 0.4
        assert result.voting_rule == VotingRule.SUPERMAJORITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outcome(proposal: Proposal, *, passed: bool) -> ...:
    """Create a VoteOutcome for testing."""
    from emergent_constitution.models.proposal import VoteOutcome

    return VoteOutcome(
        proposal=proposal,
        passed=passed,
        votes_for=3 if passed else 1,
        votes_against=1 if passed else 3,
        total_eligible=4,
    )
