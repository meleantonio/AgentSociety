"""Tick state — a complete snapshot of the simulation at one tick."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.proposal import Proposal, TradeOffer, VoteOutcome


class TickState(BaseModel):
    """Complete simulation state at a single tick.

    Args:
        tick: Current tick number (0-indexed).
        agent_states: State of every agent.
        constitution: Active ruleset.
        proposals_this_tick: Proposals submitted this tick.
        votes: Vote outcomes this tick.
        trades_this_tick: Bilateral trades executed this tick.
    """

    tick: int = Field(ge=0)
    agent_states: list[AgentState]
    constitution: Constitution
    proposals_this_tick: list[Proposal] = Field(default_factory=list)
    votes: list[VoteOutcome] = Field(default_factory=list)
    trades_this_tick: list[TradeOffer] = Field(default_factory=list)
