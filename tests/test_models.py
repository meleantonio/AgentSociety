"""Tests for Pydantic data models — validation, constraints, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.models.history import HistoryEntry, SimulationOutput
from emergent_constitution.models.proposal import Proposal, VoteOutcome
from emergent_constitution.models.tick import TickState

# --- UtilityParams ---


class TestUtilityParams:
    def test_valid_params(self):
        p = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        assert p.alpha == 0.5
        assert p.beta == 0.3
        assert p.gamma == 0.2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            UtilityParams(alpha=0.5, beta=0.5, gamma=0.5)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            UtilityParams(alpha=-0.1, beta=0.6, gamma=0.5)

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            UtilityParams(alpha=1.1, beta=0.0, gamma=0.0)


# --- ValueVector ---


class TestValueVector:
    def test_valid_values(self):
        v = ValueVector(equality=0.7, liberty=0.3)
        assert v.equality == 0.7

    def test_values_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            ValueVector(equality=0.5, liberty=0.3)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            ValueVector(equality=-0.1, liberty=1.1)


# --- AgentState ---


class TestAgentState:
    def test_valid_agent(self, sample_agent):
        assert sample_agent.id == "agent_0000"
        assert sample_agent.wealth == 100.0
        assert sample_agent.coalition_id is None

    def test_negative_wealth_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(
                id="a",
                wealth=-1.0,
                productivity=10.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.5, liberty=0.5),
            )

    def test_zero_productivity_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(
                id="a",
                wealth=10.0,
                productivity=0.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=0.5, liberty=0.5),
            )


# --- Constitution ---


class TestConstitution:
    def test_defaults(self, default_constitution):
        assert default_constitution.property_rule == PropertyRule.PRIVATE
        assert default_constitution.tax_rate == 0.0
        assert default_constitution.voting_rule == VotingRule.MAJORITY
        assert default_constitution.redistribution_rule == RedistributionRule.FLAT

    def test_tax_rate_bounds(self):
        Constitution(tax_rate=0.0)
        Constitution(tax_rate=1.0)
        with pytest.raises(ValidationError):
            Constitution(tax_rate=-0.1)
        with pytest.raises(ValidationError):
            Constitution(tax_rate=1.1)

    def test_invalid_enum_rejected(self):
        with pytest.raises(ValidationError):
            Constitution(property_rule="invalid")

    def test_all_property_rules(self):
        for rule in PropertyRule:
            c = Constitution(property_rule=rule)
            assert c.property_rule == rule


# --- Proposal / VoteOutcome ---


class TestProposal:
    def test_proposal_creation(self):
        p = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0000")
        assert p.rule_key == "tax_rate"
        assert p.proposed_value == 0.3

    def test_vote_outcome(self):
        p = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0000")
        v = VoteOutcome(proposal=p, passed=True, votes_for=30, votes_against=20, total_eligible=50)
        assert v.passed is True
        assert v.votes_for == 30


# --- TickState ---


class TestTickState:
    def test_tick_state(self, sample_agents, default_constitution):
        ts = TickState(tick=0, agent_states=sample_agents, constitution=default_constitution)
        assert ts.tick == 0
        assert len(ts.agent_states) == 3
        assert ts.proposals_this_tick == []
        assert ts.votes == []

    def test_negative_tick_rejected(self, sample_agents, default_constitution):
        with pytest.raises(ValidationError):
            TickState(tick=-1, agent_states=sample_agents, constitution=default_constitution)


# --- HistoryEntry ---


class TestHistoryEntry:
    def test_history_entry(self, default_constitution):
        h = HistoryEntry(
            tick=5,
            gini=0.35,
            total_output=500.0,
            mean_wealth=100.0,
            median_wealth=90.0,
            constitution_snapshot=default_constitution,
        )
        assert h.tick == 5
        assert h.gini == 0.35
        assert h.rule_changes == []

    def test_gini_bounds(self, default_constitution):
        with pytest.raises(ValidationError):
            HistoryEntry(
                tick=0,
                gini=1.5,
                total_output=0.0,
                mean_wealth=0.0,
                median_wealth=0.0,
                constitution_snapshot=default_constitution,
            )


# --- Serialization roundtrip ---


class TestSerialization:
    def test_agent_roundtrip(self, sample_agent):
        data = sample_agent.model_dump()
        restored = AgentState.model_validate(data)
        assert restored == sample_agent

    def test_constitution_roundtrip(self, default_constitution):
        data = default_constitution.model_dump()
        restored = Constitution.model_validate(data)
        assert restored == default_constitution

    def test_simulation_output_roundtrip(self, sample_agents, default_constitution):
        output = SimulationOutput(
            constitution=default_constitution,
            history=[],
            final_agent_states=sample_agents,
            seed=42,
            total_ticks=10,
        )
        data = output.model_dump()
        restored = SimulationOutput.model_validate(data)
        assert restored == output
