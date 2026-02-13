"""Firm state model for the DSGE-HA simulation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FirmState(BaseModel):
    """Active firm entity.

    Args:
        id: Unique firm identifier (e.g. "firm_0000").
        owner_id: HouseholdState.id of the entrepreneur.
        capital: K_f rented from aggregate savings.
        labor_demand: L_f effective labor units hired.
        tfp: A_f firm-specific total factor productivity.
        worker_ids: Agents currently employed.
        rd_spend: R&D expenditure this period.
        output: Y_f this period.
        profit: pi_f this period.
    """

    id: str
    owner_id: str
    owner_ability: float = Field(default=1.0, gt=0.0)
    capital: float = Field(ge=0.0)
    labor_demand: float = Field(ge=0.0)
    tfp: float = Field(gt=0.0)
    worker_ids: list[str] = Field(default_factory=list)
    rd_spend: float = Field(default=0.0, ge=0.0)
    output: float = Field(default=0.0, ge=0.0)
    profit: float = 0.0
