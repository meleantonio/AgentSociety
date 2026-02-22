"""Agent decision schemas for LLM structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emergent_constitution.models.proposal import ConstitutionalProposal


class EconomicDecision(BaseModel):
    """Household consumption-savings-labor decision.

    Args:
        consumption: c_t >= 0.
        leisure: l_t in [0, 1].
    """

    consumption: float = Field(ge=0.0)
    leisure: float = Field(ge=0.0, le=1.0)


class EntrepreneurialDecision(BaseModel):
    """Firm management decisions.

    Args:
        create_firm: Whether to start a new firm.
        capital_investment: K_f to rent.
        labor_demand: L_f to hire.
        rd_spend: R&D expenditure.
        close_firm: Whether to liquidate.
    """

    create_firm: bool = False
    capital_investment: float = Field(default=0.0, ge=0.0)
    labor_demand: float = Field(default=0.0, ge=0.0)
    rd_spend: float = Field(default=0.0, ge=0.0)
    close_firm: bool = False


class PoliticalDecision(BaseModel):
    """Proposal and voting decisions.

    Args:
        proposal: Optional constitutional proposal.
        votes: Mapping from proposal identifier to for/against.
    """

    proposal: ConstitutionalProposal | None = None
    votes: dict[str, bool] = Field(default_factory=dict)
