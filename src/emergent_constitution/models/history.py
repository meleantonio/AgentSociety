"""History and simulation output models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution


class HistoryEntry(BaseModel):
    """Aggregate statistics recorded at an observation tick.

    Args:
        tick: Tick number when this observation was taken.
        gini: Gini coefficient of wealth distribution.
        total_output: Total production output this tick.
        mean_wealth: Mean agent wealth.
        median_wealth: Median agent wealth.
        pareto_score: Pareto efficiency score (1.0 = fully efficient).
        rule_changes: List of rule-change descriptions this period.
        constitution_snapshot: Copy of the constitution at this tick.
    """

    tick: int = Field(ge=0)
    gini: float = Field(ge=0.0, le=1.0)
    total_output: float = Field(ge=0.0)
    mean_wealth: float = Field(ge=0.0)
    median_wealth: float = Field(ge=0.0)
    pareto_score: float = Field(ge=0.0, le=1.0, default=1.0)
    rule_changes: list[str] = Field(default_factory=list)
    constitution_snapshot: Constitution


class SimulationOutput(BaseModel):
    """Final output of a complete simulation run.

    Args:
        constitution: The final constitution.
        history: All observation entries.
        final_agent_states: Agent states at the last tick.
        seed: RNG seed used for this run.
        total_ticks: Number of ticks executed.
    """

    constitution: Constitution
    history: list[HistoryEntry]
    final_agent_states: list[AgentState]
    seed: int
    total_ticks: int = Field(ge=0)
