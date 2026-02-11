"""Simulation configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Top-level configuration for a simulation run.

    Args:
        num_agents: Number of citizen-agents (≥2).
        max_ticks: Maximum number of ticks to simulate (≥1).
        seed: RNG seed for reproducibility (PROP-001).
        initial_wealth_mean: Mean of the initial wealth distribution.
        initial_wealth_std: Std dev of the initial wealth distribution.
        initial_productivity_mean: Mean of the initial productivity distribution.
        initial_productivity_std: Std dev of the initial productivity distribution.
        proposal_interval: Proposals are collected every K ticks.
        observer_interval: Observer records stats every K ticks.
    """

    num_agents: int = Field(default=50, ge=2)
    max_ticks: int = Field(default=100, ge=1)
    seed: int = 42
    initial_wealth_mean: float = Field(default=100.0, gt=0.0)
    initial_wealth_std: float = Field(default=30.0, ge=0.0)
    initial_productivity_mean: float = Field(default=10.0, gt=0.0)
    initial_productivity_std: float = Field(default=3.0, ge=0.0)
    proposal_interval: int = Field(default=5, ge=1)
    observer_interval: int = Field(default=5, ge=1)
