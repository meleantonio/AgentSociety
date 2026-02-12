"""Shock state model — realized stochastic shocks for each period."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShockState(BaseModel):
    """Realized shocks for period t.

    Args:
        productivity_grid: Discrete z values from Rouwenhorst discretization.
        transition_matrix: Markov transition probabilities.
        aggregate_tfp: A_t economy-wide total factor productivity.
        preference_shocks: agent_id -> beta perturbation (if enabled).
    """

    productivity_grid: list[float] = Field(default_factory=list)
    transition_matrix: list[list[float]] = Field(default_factory=list)
    aggregate_tfp: float = Field(default=1.0, gt=0.0)
    preference_shocks: dict[str, float] = Field(default_factory=dict)
