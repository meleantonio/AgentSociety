"""History, period state, and simulation output models — v1 and v2.

v1 types (HistoryEntry, SimulationOutput) retained for backward compat.
v2 types (PeriodState, HistoryEntryV2, WelfareSummary, SimulationOutputV2)
per spec/design.md sections 2.7-2.8.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution, ConstitutionV2
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal, VoteOutcomeV2
from emergent_constitution.models.shocks import ShockState

# ============================================================================
# v1 types (backward compatibility)
# ============================================================================


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
        num_coalitions: Number of distinct coalitions at this tick.
    """

    tick: int = Field(ge=0)
    gini: float = Field(ge=0.0, le=1.0)
    total_output: float = Field(ge=0.0)
    mean_wealth: float = Field(ge=0.0)
    median_wealth: float = Field(ge=0.0)
    pareto_score: float = Field(ge=0.0, le=1.0, default=1.0)
    rule_changes: list[str] = Field(default_factory=list)
    constitution_snapshot: Constitution
    num_coalitions: int = Field(default=0, ge=0)


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


# ============================================================================
# v2 types (DSGE-HA)
# ============================================================================


class PeriodState(BaseModel):
    """Complete state snapshot for period t. Replaces v1 TickState.

    Args:
        period: Current period number (0-indexed).
        households: State of every household agent.
        firms: State of every active firm.
        market: Market equilibrium for this period.
        shocks: Realized shocks for this period.
        constitution: Active constitutional rules.
        proposals: Proposals submitted this period.
        votes: Vote outcomes this period.
    """

    period: int = Field(ge=0)
    households: list[HouseholdState]
    firms: list[FirmState] = Field(default_factory=list)
    market: MarketState
    shocks: ShockState
    constitution: ConstitutionV2
    proposals: list[ConstitutionalProposal] = Field(default_factory=list)
    votes: list[VoteOutcomeV2] = Field(default_factory=list)


class HistoryEntryV2(BaseModel):
    """Observation record per observation interval (v2).

    Args:
        period: Period number when this observation was taken.
        gini: Gini coefficient of wealth distribution.
        pareto_score: Pareto efficiency score.
        aggregate_output: Y_t total output.
        aggregate_consumption: C_t total consumption.
        aggregate_investment: I_t total investment.
        mean_wealth: Mean agent wealth.
        median_wealth: Median agent wealth.
        wealth_quantiles: Quantile values [p10, p25, p50, p75, p90].
        unemployment_rate: Fraction of agents unemployed.
        num_active_firms: Number of active firms.
        mean_firm_size: Average workers per firm.
        aggregate_rd_spend: Total R&D spending across firms.
        social_welfare: Sum of realized utilities this period.
        cumulative_welfare: Discounted sum of welfare to date.
        wage: Equilibrium wage w_t.
        interest_rate: Equilibrium interest rate r_t.
        rule_changes: List of rule change descriptions this period.
        constitution_snapshot: Copy of the constitution at this period.
    """

    period: int = Field(ge=0)
    gini: float = Field(ge=0.0, le=1.0)
    pareto_score: float = Field(ge=0.0, le=1.0, default=1.0)
    aggregate_output: float = Field(ge=0.0)
    aggregate_consumption: float = Field(ge=0.0, default=0.0)
    aggregate_investment: float = 0.0
    mean_wealth: float = Field(ge=0.0)
    median_wealth: float = Field(ge=0.0)
    wealth_quantiles: list[float] = Field(default_factory=list)
    unemployment_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    num_active_firms: int = Field(ge=0, default=0)
    mean_firm_size: float = Field(ge=0.0, default=0.0)
    aggregate_rd_spend: float = Field(ge=0.0, default=0.0)
    social_welfare: float = 0.0
    cumulative_welfare: float = 0.0
    wage: float = 0.0
    interest_rate: float = 0.0
    rule_changes: list[str] = Field(default_factory=list)
    constitution_snapshot: ConstitutionV2


class WelfareSummary(BaseModel):
    """Welfare comparison between LLM and benchmark.

    Args:
        llm_total_welfare: Total welfare from LLM-driven run.
        benchmark_total_welfare: Total welfare from numerical benchmark.
        per_agent_comparison: Per-agent welfare comparison.
    """

    llm_total_welfare: float
    benchmark_total_welfare: float | None = None
    per_agent_comparison: dict[str, dict] | None = None


class SimulationOutputV2(BaseModel):
    """Final output of the simulation (v2).

    Args:
        constitution: The final constitution.
        history: All observation entries.
        final_households: Household states at the last period.
        final_firms: Firm states at the last period.
        welfare_summary: Welfare comparison summary.
        seed: RNG seed used for this run.
        total_periods: Number of periods executed.
    """

    constitution: ConstitutionV2
    history: list[HistoryEntryV2]
    final_households: list[HouseholdState]
    final_firms: list[FirmState] = Field(default_factory=list)
    welfare_summary: WelfareSummary
    seed: int
    total_periods: int = Field(ge=0)
