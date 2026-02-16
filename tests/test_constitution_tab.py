"""Tests for constitution tab data extraction from mock history entries."""

from __future__ import annotations

import pytest

from emergent_constitution.dashboard.constitution_tab import (
    _build_proposals_table,
    _build_votes_table,
)
from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.history import HistoryEntryV2
from emergent_constitution.models.proposal import ConstitutionalProposal, VoteOutcomeV2


def _make_entry(
    period: int,
    proposals: list[ConstitutionalProposal] | None = None,
    votes: list[VoteOutcomeV2] | None = None,
    rule_changes: list[str] | None = None,
) -> HistoryEntryV2:
    """Create a minimal HistoryEntryV2 for testing."""
    return HistoryEntryV2(
        period=period,
        gini=0.3,
        pareto_score=0.9,
        aggregate_output=100.0,
        aggregate_consumption=80.0,
        aggregate_investment=20.0,
        mean_wealth=100.0,
        median_wealth=90.0,
        wage=1.0,
        interest_rate=0.05,
        proposals=proposals or [],
        votes=votes or [],
        rule_changes=rule_changes or [],
        constitution_snapshot=create_default_constitution(),
    )


def _make_proposal(
    action: str = "add",
    rule_name: str = "test_rule",
    proposer: str = "agent_0000",
) -> ConstitutionalProposal:
    return ConstitutionalProposal(
        proposer_id=proposer,
        action=action,
        rule_name=rule_name,
        rule_type="taxation",
        parameters={"rate": 0.1},
        description="Test proposal",
    )


def _make_vote(
    proposal: ConstitutionalProposal,
    passed: bool = True,
    votes_for: int = 15,
    votes_against: int = 5,
) -> VoteOutcomeV2:
    return VoteOutcomeV2(
        proposal=proposal,
        passed=passed,
        votes_for=votes_for,
        votes_against=votes_against,
        total_eligible=20,
        voting_rule_used="majority",
    )


class TestBuildProposalsTable:
    """Tests for proposals table extraction."""

    def test_empty_history(self) -> None:
        result = _build_proposals_table([])
        assert result == []

    def test_single_accepted_proposal(self) -> None:
        prop = _make_proposal(action="add", rule_name="flat_tax")
        vote = _make_vote(prop, passed=True)
        entry = _make_entry(period=5, proposals=[prop], votes=[vote])
        result = _build_proposals_table([entry])

        assert len(result) == 1
        row = result[0]
        assert row["Period"] == 5
        assert row["Proposer"] == "agent_0000"
        assert row["Action"] == "add"
        assert row["Rule Name"] == "flat_tax"
        assert row["Result"] == "Accepted"

    def test_rejected_proposal(self) -> None:
        prop = _make_proposal(action="remove", rule_name="old_rule")
        vote = _make_vote(prop, passed=False)
        entry = _make_entry(period=10, proposals=[prop], votes=[vote])
        result = _build_proposals_table([entry])

        assert result[0]["Result"] == "Rejected"

    def test_proposal_without_vote(self) -> None:
        prop = _make_proposal()
        entry = _make_entry(period=5, proposals=[prop], votes=[])
        result = _build_proposals_table([entry])

        assert result[0]["Result"] == "No Vote"

    def test_multiple_entries(self) -> None:
        prop1 = _make_proposal(rule_name="rule_a")
        vote1 = _make_vote(prop1, passed=True)
        prop2 = _make_proposal(rule_name="rule_b")
        vote2 = _make_vote(prop2, passed=False)

        history = [
            _make_entry(period=5, proposals=[prop1], votes=[vote1]),
            _make_entry(period=10, proposals=[prop2], votes=[vote2]),
        ]
        result = _build_proposals_table(history)

        assert len(result) == 2
        assert result[0]["Period"] == 5
        assert result[1]["Period"] == 10


class TestBuildVotesTable:
    """Tests for votes table extraction."""

    def test_empty_history(self) -> None:
        result = _build_votes_table([])
        assert result == []

    def test_single_vote(self) -> None:
        prop = _make_proposal(action="add", rule_name="flat_tax")
        vote = _make_vote(prop, passed=True, votes_for=15, votes_against=5)
        entry = _make_entry(period=5, votes=[vote])
        result = _build_votes_table([entry])

        assert len(result) == 1
        row = result[0]
        assert row["Period"] == 5
        assert row["Proposal"] == "flat_tax (add)"
        assert row["Result"] == "Passed"
        assert row["Votes For"] == 15
        assert row["Votes Against"] == 5
        assert row["Total Eligible"] == 20
        assert row["Approval %"] == 75.0
        assert row["Voting Rule"] == "majority"

    def test_failed_vote(self) -> None:
        prop = _make_proposal()
        vote = _make_vote(prop, passed=False, votes_for=3, votes_against=17)
        entry = _make_entry(period=5, votes=[vote])
        result = _build_votes_table([entry])

        assert result[0]["Result"] == "Rejected"
        assert result[0]["Approval %"] == 15.0
