"""Household (citizen-agent) state and related models.

Replaces v1 AgentState with DSGE-HA household model including wealth dynamics,
productivity Markov chain, occupational roles, and per-period decision outcomes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class UtilityParams(BaseModel):
    """Cobb-Douglas utility parameters: u(c, l, G) = c^alpha * l^beta * G^gamma.

    Args:
        alpha: Weight on private consumption, in (0, 1).
        beta: Weight on leisure, in (0, 1).
        gamma: Weight on public goods, in (0, 1).
        beta_discount: Intertemporal discount factor, in (0, 1).
    """

    alpha: float = Field(ge=0.0, le=1.0, description="Consumption weight")
    beta: float = Field(ge=0.0, le=1.0, description="Leisure weight")
    gamma: float = Field(ge=0.0, le=1.0, description="Public goods weight")
    beta_discount: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Intertemporal discount factor",
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> UtilityParams:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            msg = f"Utility weights must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self


class ValueVector(BaseModel):
    """Agent's ideological position on equality vs liberty (sums to 1).

    Args:
        equality: Preference weight toward equality.
        liberty: Preference weight toward liberty.
    """

    equality: float = Field(ge=0.0, le=1.0)
    liberty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _values_sum_to_one(self) -> ValueVector:
        total = self.equality + self.liberty
        if abs(total - 1.0) > 1e-6:
            msg = f"Value weights must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self


class OccupationalRole(StrEnum):
    """Agent occupational roles in the economy."""

    WORKER = "worker"
    ENTREPRENEUR = "entrepreneur"
    RESEARCHER = "researcher"
    UNEMPLOYED = "unemployed"


class HouseholdState(BaseModel):
    """Per-agent state each period. Replaces v1 AgentState.

    Args:
        id: Unique agent identifier (e.g. "agent_0000").
        wealth: Assets a_t (>= a_min).
        productivity: Idiosyncratic z_t from Markov chain.
        productivity_index: Index into Markov transition matrix.
        utility_params: Cobb-Douglas utility weights.
        value_vector: Ideological position (equality/liberty).
        role: Current occupational role.
        firm_id: Firm owned (if entrepreneur).
        coalition_id: Coalition membership.
        consumption: c_t for this period.
        leisure: l_t for this period (0=full work, 1=no work).
        labor_supply: 1 - l_t.
        savings: a_{t+1} - a_t.
        income: w_t * z_t * labor_supply.
        taxes_paid: Taxes paid this period.
        transfers_received: Transfers received this period.
        realized_utility: u(c_t, l_t, G_t) ground truth.
    """

    id: str
    wealth: float = Field(ge=0.0)
    productivity: float = Field(gt=0.0)
    productivity_index: int = Field(ge=0)
    utility_params: UtilityParams
    value_vector: ValueVector
    role: OccupationalRole = OccupationalRole.WORKER
    firm_id: str | None = None
    coalition_id: str | None = None

    # Per-period decision outcomes (filled after decisions applied)
    consumption: float = Field(default=0.0, ge=0.0)
    leisure: float = Field(default=0.0, ge=0.0, le=1.0)
    labor_supply: float = Field(default=0.0, ge=0.0, le=1.0)
    savings: float = 0.0
    income: float = 0.0
    taxes_paid: float = 0.0
    transfers_received: float = 0.0
    realized_utility: float = 0.0
