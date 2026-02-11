"""Constitution model — the mutable ruleset governing the simulation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PropertyRule(StrEnum):
    """How production output is allocated."""

    PRIVATE = "private"
    COMMUNAL = "communal"
    MIXED = "mixed"


class VotingRule(StrEnum):
    """How proposals are decided."""

    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    UNANIMITY = "unanimity"


class RedistributionRule(StrEnum):
    """How tax revenue is redistributed."""

    NONE = "none"
    FLAT = "flat"
    PROGRESSIVE = "progressive"


# Valid Constitution field keys and their allowed types/values.
CONSTITUTION_FIELDS: dict[str, type] = {
    "property_rule": PropertyRule,
    "tax_rate": float,
    "voting_rule": VotingRule,
    "redistribution_rule": RedistributionRule,
}


class Constitution(BaseModel):
    """The active ruleset governing the simulation.

    Args:
        property_rule: How production is allocated.
        tax_rate: Fraction of output collected as tax (0–1).
        voting_rule: How proposals are decided.
        redistribution_rule: How tax revenue is redistributed.
    """

    property_rule: PropertyRule = PropertyRule.PRIVATE
    tax_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    voting_rule: VotingRule = VotingRule.MAJORITY
    redistribution_rule: RedistributionRule = RedistributionRule.FLAT
