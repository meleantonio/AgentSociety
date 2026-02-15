"""Tests for rule-based governance v2 — proposal generation and voting.

Covers:
- Probabilistic proposal gating (~85% skip rate)
- Value-aligned proposal generation (equality-lovers raise taxes, liberty-lovers lower them)
- Content agents rarely propose
- Vote alignment and misalignment with agent values
- Empty proposals edge case
"""

from __future__ import annotations

import pytest

from emergent_constitution.citizen_v2 import decide_proposal_v2, decide_votes_v2
from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.household import HouseholdState, UtilityParams, ValueVector
from emergent_constitution.models.proposal import ConstitutionalProposal
from emergent_constitution.rng import SimulationRNG

# ============================================================================
# Helpers
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    equality: float = 0.5,
    gamma: float = 0.25,
) -> HouseholdState:
    """Create a minimal test household for governance tests."""
    alpha = 0.4
    beta = 1.0 - alpha - gamma
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=0,
        utility_params=UtilityParams(alpha=alpha, beta=beta, gamma=gamma, beta_discount=0.95),
        value_vector=ValueVector(equality=equality, liberty=round(1.0 - equality, 6)),
    )


def _make_constitution_with_tax(rate: float = 0.0) -> ConstitutionV2:
    """Create a default constitution with a specific tax rate."""
    const = create_default_constitution()
    const.rules["flat_tax"].parameters["rate"] = rate
    return const


# ============================================================================
# Proposal skip probability
# ============================================================================


class TestProposalSkipProbability:
    def test_proposal_skip_probability(self) -> None:
        """Most calls should return None (~85% skip rate).

        With _PROPOSAL_SKIP_PROBABILITY=0.85, over many seeds we expect
        roughly 15% of calls to produce a proposal.
        """
        household = _make_household(equality=0.9)
        constitution = create_default_constitution()
        num_trials = 500
        num_proposals = 0

        for seed in range(num_trials):
            rng = SimulationRNG(seed=seed)
            result = decide_proposal_v2(household, constitution, 20, rng)
            if result is not None:
                num_proposals += 1

        proposal_rate = num_proposals / num_trials

        # Expected: ~15% (0.15), allow range [0.08, 0.25] for statistical variation
        assert 0.08 < proposal_rate < 0.25, (
            f"Proposal rate {proposal_rate:.3f} outside expected range [0.08, 0.25]. "
            f"Got {num_proposals} proposals in {num_trials} trials."
        )


# ============================================================================
# Value-aligned proposals
# ============================================================================


class TestProposalAlignment:
    def test_equality_lover_proposes_tax_increase(self) -> None:
        """Agent with high equality (0.9) should propose higher tax rate.

        With a 0% tax and equality=0.9, the preferred rate is ~0.45.
        The dissatisfaction is high, so the proposal should increase the rate.
        """
        household = _make_household(equality=0.9, wealth=50.0)
        constitution = _make_constitution_with_tax(rate=0.0)

        found_tax_increase = False
        for seed in range(300):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None and "tax" in proposal.rule_name.lower():
                new_rate = proposal.parameters.get("rate", 0.0)
                if new_rate > 0.0:
                    found_tax_increase = True
                    break

        assert found_tax_increase, (
            "High-equality agent should propose a tax increase from 0% "
            "within 300 seeds"
        )

    def test_liberty_lover_proposes_tax_decrease(self) -> None:
        """Agent with high liberty (equality=0.1) should propose lower tax.

        With a 30% tax and equality=0.1, the preferred rate is ~0.05.
        The dissatisfaction drives a proposal to lower the rate.
        """
        household = _make_household(equality=0.1, wealth=200.0)
        constitution = _make_constitution_with_tax(rate=0.3)

        found_tax_decrease = False
        for seed in range(300):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None and "tax" in proposal.rule_name.lower():
                new_rate = proposal.parameters.get("rate", 0.3)
                if new_rate < 0.3:
                    found_tax_decrease = True
                    break

        assert found_tax_decrease, (
            "High-liberty agent should propose a tax decrease from 30% "
            "within 300 seeds"
        )


# ============================================================================
# Content agents
# ============================================================================


class TestContentAgent:
    def test_no_proposal_when_content(self) -> None:
        """Agent whose preferences match the constitution should never propose.

        We construct a constitution where every rule matches the agent's
        preferred values so that all dissatisfaction scores fall below the
        0.05 thresholds, making the dissatisfaction list empty.

        Dissatisfaction thresholds in decide_proposal_v2:
        - Tax: |current_rate - equality * 0.5| <= 0.05
        - Transfer: equality < 0.4 wants equal_share (already default)
        - Public goods: |fraction - gamma * 0.8| <= 0.05
        - Voting: |threshold - (0.5 + equality * 0.2)| <= 0.05
        """
        # equality=0.1 -> preferred tax = 0.1 * 0.5 = 0.05
        # preferred transfer = equal_share (equality < 0.4)
        # gamma=0.375 -> preferred public goods fraction = 0.375 * 0.8 = 0.30
        # preferred voting threshold = 0.5 + 0.1 * 0.2 = 0.52 (close to 0.5)
        household = _make_household(equality=0.1, gamma=0.375)

        # Build a constitution that matches all preferred values
        constitution = create_default_constitution()
        constitution.rules["flat_tax"].parameters["rate"] = 0.05  # matches 0.1 * 0.5
        # flat_transfer already has method=equal_share (correct for equality=0.1)
        # public_goods_provision already has fraction=0.3 (matches gamma=0.375 -> 0.3)
        # majority_vote has threshold=0.5 (close to 0.52, within 0.05)

        proposals_found = 0
        num_trials = 500
        for seed in range(num_trials):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None:
                proposals_found += 1

        # With all dissatisfaction below threshold, no proposals should be generated
        assert proposals_found == 0, (
            f"Fully content agent should never propose, but got "
            f"{proposals_found}/{num_trials} proposals"
        )


# ============================================================================
# Voting tests
# ============================================================================


class TestVoting:
    def test_votes_favor_aligned_proposals(self) -> None:
        """Agents vote for proposals aligned with their values.

        A high-equality agent should vote for a tax increase proposal.
        """
        household = _make_household(equality=0.9, wealth=50.0)
        constitution = _make_constitution_with_tax(rate=0.0)

        # Proposal to increase tax from 0% to 40%
        proposal = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.4},
            description="Increase tax rate",
        )

        votes = decide_votes_v2(household, [proposal], constitution)

        assert "flat_tax" in votes
        assert votes["flat_tax"] is True, (
            "High-equality agent should vote FOR tax increase"
        )

    def test_votes_against_misaligned_proposals(self) -> None:
        """Agents vote against proposals misaligned with their values.

        A high-liberty (low equality) wealthy agent should vote against
        a large tax increase.
        """
        household = _make_household(equality=0.1, wealth=300.0)
        constitution = _make_constitution_with_tax(rate=0.0)

        # Proposal to increase tax from 0% to 50%
        proposal = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.5},
            description="Increase tax rate sharply",
        )

        votes = decide_votes_v2(household, [proposal], constitution)

        assert "flat_tax" in votes
        assert votes["flat_tax"] is False, (
            "High-liberty wealthy agent should vote AGAINST large tax increase"
        )

    def test_empty_proposals_returns_empty_votes(self) -> None:
        """Empty proposals list should return an empty dict."""
        household = _make_household()
        constitution = create_default_constitution()

        votes = decide_votes_v2(household, [], constitution)

        assert votes == {}

    def test_multiple_proposals_all_voted(self) -> None:
        """Every proposal should receive a vote."""
        household = _make_household(equality=0.5)
        constitution = create_default_constitution()

        proposals = [
            ConstitutionalProposal(
                proposer_id="agent_0001",
                action="modify",
                rule_name="flat_tax",
                parameters={"rate": 0.2},
                description="Moderate tax",
            ),
            ConstitutionalProposal(
                proposer_id="agent_0002",
                action="modify",
                rule_name="flat_transfer",
                parameters={"method": "progressive"},
                description="Progressive transfers",
            ),
        ]

        votes = decide_votes_v2(household, proposals, constitution)

        assert len(votes) == 2
        assert "flat_tax" in votes
        assert "flat_transfer" in votes
        for v in votes.values():
            assert isinstance(v, bool)

    def test_equality_agent_votes_for_progressive_transfers(self) -> None:
        """High-equality agent should favor progressive transfers."""
        household = _make_household(equality=0.9, wealth=50.0)
        constitution = create_default_constitution()

        proposal = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_transfer",
            parameters={"method": "progressive"},
            description="Switch to progressive transfers",
        )

        votes = decide_votes_v2(household, [proposal], constitution)

        assert votes.get("flat_transfer") is True, (
            "High-equality agent should vote FOR progressive transfers"
        )

    def test_liberty_agent_votes_against_progressive_transfers(self) -> None:
        """High-liberty wealthy agent should oppose progressive transfers."""
        household = _make_household(equality=0.1, wealth=400.0)
        constitution = create_default_constitution()

        proposal = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_transfer",
            parameters={"method": "progressive"},
            description="Switch to progressive transfers",
        )

        votes = decide_votes_v2(household, [proposal], constitution)

        assert votes.get("flat_transfer") is False, (
            "High-liberty wealthy agent should vote AGAINST progressive transfers"
        )

    def test_votes_return_bool_values(self) -> None:
        """All vote values should be booleans."""
        household = _make_household(equality=0.5)
        constitution = create_default_constitution()

        proposal = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.25},
            description="Tax change",
        )

        votes = decide_votes_v2(household, [proposal], constitution)

        for v in votes.values():
            assert isinstance(v, bool), f"Vote value should be bool, got {type(v)}"


# ============================================================================
# Proposal structure validation
# ============================================================================


class TestProposalStructure:
    def test_proposal_has_correct_proposer_id(self) -> None:
        """Generated proposals should carry the household's id."""
        household = _make_household(agent_id="agent_test_id", equality=0.9)
        constitution = _make_constitution_with_tax(rate=0.0)

        for seed in range(300):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None:
                assert proposal.proposer_id == "agent_test_id"
                return

        pytest.fail("No proposal generated in 300 seeds")

    def test_proposal_action_is_modify(self) -> None:
        """All v2 governance proposals should use action='modify'."""
        household = _make_household(equality=0.9)
        constitution = create_default_constitution()

        for seed in range(300):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None:
                assert proposal.action == "modify"
                return

        pytest.fail("No proposal generated in 300 seeds")

    def test_proposal_targets_existing_rule(self) -> None:
        """Proposals should target rules that exist in the constitution."""
        household = _make_household(equality=0.8)
        constitution = create_default_constitution()
        valid_rule_names = set(constitution.rules.keys())

        for seed in range(300):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None:
                assert proposal.rule_name in valid_rule_names, (
                    f"Proposal targets '{proposal.rule_name}' which is not in "
                    f"constitution rules: {valid_rule_names}"
                )
                return

        pytest.fail("No proposal generated in 300 seeds")

    def test_tax_proposal_rate_bounded(self) -> None:
        """Tax rate proposals should be bounded to [0, 1]."""
        household = _make_household(equality=0.9)
        constitution = _make_constitution_with_tax(rate=0.0)

        for seed in range(500):
            rng = SimulationRNG(seed=seed)
            proposal = decide_proposal_v2(household, constitution, 20, rng)
            if proposal is not None and "tax" in proposal.rule_name.lower():
                rate = proposal.parameters.get("rate")
                if rate is not None:
                    assert 0.0 <= rate <= 1.0, (
                        f"Tax rate {rate} is out of bounds [0, 1]"
                    )
                    return

        pytest.fail("No tax proposal generated in 500 seeds")
