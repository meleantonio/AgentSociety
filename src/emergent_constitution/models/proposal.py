"""Proposal and voting outcome models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
