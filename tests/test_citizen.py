"""Tests for the citizen decision logic — Gini, proposals, and voting."""

from __future__ import annotations

import pytest

from emergent_constitution.citizen import compute_gini, decide_proposal, decide_votes
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.models.proposal import Proposal
from emergent_constitution.rng import SimulationRNG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str = "a1",
    wealth: float = 100.0,
    productivity: float = 10.0,
    equality: float = 0.5,
) -> AgentState:
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=0.4, beta=0.3, gamma=0.3),
        value_vector=ValueVector(equality=equality, liberty=round(1.0 - equality, 6)),
    )


# ---------------------------------------------------------------------------
# compute_gini
# ---------------------------------------------------------------------------


class TestComputeGini:
    def test_perfect_equality(self):
        assert compute_gini([100.0, 100.0, 100.0, 100.0]) == pytest.approx(0.0)

    def test_perfect_inequality(self):
        """One person has everything."""
        gini = compute_gini([0.0, 0.0, 0.0, 1000.0])
        assert gini > 0.6

    def test_empty_list(self):
        assert compute_gini([]) == 0.0

    def test_single_element(self):
        assert compute_gini([42.0]) == 0.0

    def test_moderate_inequality(self):
        gini = compute_gini([10.0, 20.0, 30.0, 40.0])
        assert 0.0 < gini < 0.5


# ---------------------------------------------------------------------------
# decide_proposal
# ---------------------------------------------------------------------------


class TestDecideProposal:
    def test_deterministic_same_seed(self):
        agent = _make_agent(equality=0.9)
        constitution = Constitution()
        agents = [agent]
        result1 = decide_proposal(agent, constitution, agents, SimulationRNG(seed=99))
        result2 = decide_proposal(agent, constitution, agents, SimulationRNG(seed=99))
        assert result1 == result2

    def test_equality_agent_proposes_higher_tax(self):
        """High-equality agent proposes higher tax when other dims are satisfied."""
        agent = _make_agent(equality=0.9)
        # Set other dimensions to equality-preferred values so tax_rate is the biggest gap
        constitution = Constitution(
            tax_rate=0.0,
            property_rule=PropertyRule.COMMUNAL,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
            voting_rule=VotingRule.SUPERMAJORITY,
        )
        agents = [agent]
        for seed in range(200):
            proposal = decide_proposal(agent, constitution, agents, SimulationRNG(seed=seed))
            if proposal is not None:
                assert proposal.rule_key == "tax_rate"
                assert float(proposal.proposed_value) > 0.0
                return
        pytest.fail("High-equality agent never proposed a tax increase in 200 seeds")

    def test_satisfied_agent_skips(self):
        """Agent whose preferred values match the constitution should not propose."""
        # equality=0.05 → preferred tax ≈ 0.03, property=private, redist=none, vote=majority
        agent = _make_agent(equality=0.05)
        constitution = Constitution(
            tax_rate=0.0,
            property_rule=PropertyRule.PRIVATE,
        )
        # With low equality, dissatisfaction should be minimal for default constitution
        proposals_found = 0
        for seed in range(100):
            proposal = decide_proposal(agent, constitution, [agent], SimulationRNG(seed=seed))
            if proposal is not None:
                proposals_found += 1
        # Low-equality agent with default constitution: very few proposals at most
        assert proposals_found < 20, "Nearly satisfied agent is proposing too often"

    def test_proposal_targets_valid_field(self):
        """Any proposal produced should target a valid constitution field."""
        agent = _make_agent(equality=0.8)
        constitution = Constitution()
        valid_keys = {"tax_rate", "property_rule", "redistribution_rule", "voting_rule"}
        for seed in range(100):
            proposal = decide_proposal(agent, constitution, [agent], SimulationRNG(seed=seed))
            if proposal is not None:
                assert proposal.rule_key in valid_keys

    def test_proposer_id_matches_agent(self):
        agent = _make_agent(agent_id="agent_42", equality=0.9)
        constitution = Constitution()
        for seed in range(100):
            proposal = decide_proposal(agent, constitution, [agent], SimulationRNG(seed=seed))
            if proposal is not None:
                assert proposal.proposer_id == "agent_42"
                return
        pytest.fail("Agent never proposed in 100 seeds")


# ---------------------------------------------------------------------------
# decide_votes
# ---------------------------------------------------------------------------


class TestDecideVotes:
    def test_deterministic_same_seed(self):
        agent = _make_agent(equality=0.5)
        constitution = Constitution()
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="a2")
        all_agents = [agent, _make_agent("a2"), _make_agent("a3")]
        v1 = decide_votes(agent, constitution, [proposal], all_agents, SimulationRNG(seed=42))
        v2 = decide_votes(agent, constitution, [proposal], all_agents, SimulationRNG(seed=42))
        assert v1 == v2

    def test_equality_agent_favors_tax_increase(self):
        """High-equality agent should vote for raising tax from 0%."""
        agent = _make_agent(equality=0.9, wealth=50.0)
        poor_agent = _make_agent("a_poor", wealth=30.0, equality=0.9)
        rich_agent = _make_agent("a_rich", wealth=200.0, equality=0.1)
        all_agents = [agent, poor_agent, rich_agent]
        constitution = Constitution(tax_rate=0.0)
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.4, proposer_id="a_poor")
        votes = decide_votes(agent, constitution, [proposal], all_agents, SimulationRNG(seed=42))
        assert votes.get("a_poor") is True

    def test_liberty_agent_opposes_tax_increase(self):
        """High-liberty rich agent should oppose raising tax from 0%."""
        agent = _make_agent(equality=0.1, wealth=200.0)
        poor_agent = _make_agent("a_poor", wealth=30.0)
        all_agents = [agent, poor_agent]
        constitution = Constitution(tax_rate=0.0)
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.5, proposer_id="a_poor")
        votes = decide_votes(agent, constitution, [proposal], all_agents, SimulationRNG(seed=42))
        assert votes.get("a_poor") is False

    def test_multiple_proposals(self):
        agent = _make_agent(equality=0.5)
        all_agents = [agent, _make_agent("a2")]
        constitution = Constitution()
        proposals = [
            Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="a2"),
            Proposal(rule_key="property_rule", proposed_value="communal", proposer_id="a3"),
        ]
        votes = decide_votes(agent, constitution, proposals, all_agents, SimulationRNG(seed=42))
        assert len(votes) == 2

    def test_empty_proposals(self):
        agent = _make_agent()
        votes = decide_votes(agent, Constitution(), [], [agent], SimulationRNG(seed=42))
        assert votes == {}

    def test_returns_bool_values(self):
        agent = _make_agent(equality=0.7)
        all_agents = [agent, _make_agent("a2")]
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.2, proposer_id="a2")
        votes = decide_votes(agent, Constitution(), [proposal], all_agents, SimulationRNG(seed=42))
        for v in votes.values():
            assert isinstance(v, bool)
