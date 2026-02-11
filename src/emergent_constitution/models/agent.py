"""Agent state and related models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class UtilityParams(BaseModel):
    """Cobb-Douglas utility parameters: U = c^alpha * l^beta * g^gamma.

    Args:
        alpha: Weight on private consumption.
        beta: Weight on leisure.
        gamma: Weight on public goods.
    """

    alpha: float = Field(ge=0.0, le=1.0, description="Weight on private consumption")
    beta: float = Field(ge=0.0, le=1.0, description="Weight on leisure")
    gamma: float = Field(ge=0.0, le=1.0, description="Weight on public goods")

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


class AgentState(BaseModel):
    """State of a single citizen-agent at a point in time.

    Args:
        id: Unique agent identifier.
        wealth: Current wealth (non-negative).
        productivity: Production capacity per tick (positive).
        utility_params: Cobb-Douglas utility weights.
        value_vector: Ideological position.
        coalition_id: Optional coalition membership.
    """

    id: str
    wealth: float = Field(ge=0.0)
    productivity: float = Field(gt=0.0)
    utility_params: UtilityParams
    value_vector: ValueVector
    coalition_id: str | None = None
