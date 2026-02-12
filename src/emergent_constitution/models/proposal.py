"""Proposal and voting outcome models — v1 and v2.

v1 types (Proposal, TradeOffer, VoteOutcome) retained for backward compat.
v2 types (ConstitutionalProposal, VoteOutcomeV2) per spec/design.md section 2.6.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from emergent_constitution.models.constitution import RuleType

# ============================================================================
# v1 types (backward compatibility)
# ============================================================================


class Proposal(BaseModel):
    """A rule-change proposal submitted by a citizen.

    Args:
        rule_key: Which constitution field to change.
        proposed_value: The new value for the field.
        proposer_id: ID of the agent who proposed this.
    """

    rule_key: str
    proposed_value: Any
    proposer_id: str


class TradeOffer(BaseModel):
    """A bilateral trade offer between two agents.

    Represents a transfer of wealth from buyer to seller (e.g., buying labor).

    Args:
        seller_id: ID of the agent selling (receives wealth).
        buyer_id: ID of the agent buying (pays wealth).
        amount: Wealth transferred from buyer to seller (must be positive).
    """

    seller_id: str
    buyer_id: str
    amount: float = Field(gt=0.0)


class VoteOutcome(BaseModel):
    """Result of voting on a single proposal.

    Args:
        proposal: The proposal that was voted on.
        passed: Whether the proposal was accepted.
        votes_for: Number of votes in favor.
        votes_against: Number of votes against.
        total_eligible: Total number of eligible voters.
    """

    proposal: Proposal
    passed: bool
    votes_for: int = Field(ge=0)
    votes_against: int = Field(ge=0)
    total_eligible: int = Field(ge=0)


# ============================================================================
# v2 types (DSGE-HA constitutional governance)
# ============================================================================


class ConstitutionalProposal(BaseModel):
    """A proposal to add, modify, or remove a constitutional rule (v2).

    Args:
        proposer_id: ID of the agent who proposed this.
        action: Whether to add, modify, or remove a rule.
        rule_name: Target rule name.
        rule_type: Required for "add" action.
        parameters: New parameters for the rule.
        description: Natural-language rationale.
        enforcement_code: Enforcement specification.
    """

    proposer_id: str
    action: Literal["add", "modify", "remove"]
    rule_name: str
    rule_type: RuleType | None = None
    parameters: dict[str, Any] | None = None
    description: str = ""
    enforcement_code: str | None = None


class VoteOutcomeV2(BaseModel):
    """Result of a constitutional vote (v2).

    Args:
        proposal: The proposal that was voted on.
        passed: Whether the proposal was accepted.
        votes_for: Number of votes in favor.
        votes_against: Number of votes against.
        total_eligible: Total number of eligible voters.
        voting_rule_used: Name of the voting rule applied.
    """

    proposal: ConstitutionalProposal
    passed: bool
    votes_for: int = Field(ge=0)
    votes_against: int = Field(ge=0)
    total_eligible: int = Field(ge=0)
    voting_rule_used: str
