"""Lead (Simulation Governor) — the main tick loop.

The Lead advances time in discrete ticks, enforces turn order, and
orchestrates proposals, votes, economic steps, and observation.
"""

from __future__ import annotations

import structlog

from emergent_constitution.config import SimulationConfig
from emergent_constitution.economics import economic_step
from emergent_constitution.initialization import initialize_simulation
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.history import HistoryEntry, SimulationOutput
from emergent_constitution.models.proposal import Proposal, VoteOutcome
from emergent_constitution.models.tick import TickState
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()


class Lead:
    """Simulation governor that runs the main tick loop.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.tick_state: TickState
        self.rng: SimulationRNG
        self.history: list[HistoryEntry] = []
        self.tick_state, self.rng = initialize_simulation(config)
        log.info("simulation.initialized", num_agents=config.num_agents, seed=config.seed)

    def run(self) -> SimulationOutput:
        """Execute all ticks and return the final simulation output.

        Returns:
            SimulationOutput with final constitution, history, and agent states.
        """
        log.info("simulation.started", max_ticks=self.config.max_ticks)

        for tick in range(1, self.config.max_ticks + 1):
            self.tick_state = self._advance_tick(tick)

        log.info("simulation.completed", total_ticks=self.config.max_ticks)
        return SimulationOutput(
            constitution=self.tick_state.constitution.model_copy(deep=True),
            history=list(self.history),
            final_agent_states=[a.model_copy(deep=True) for a in self.tick_state.agent_states],
            seed=self.config.seed,
            total_ticks=self.config.max_ticks,
        )

    def _advance_tick(self, tick: int) -> TickState:
        """Execute a single tick: proposals → votes → constitution update → economics → observe.

        Args:
            tick: Current tick number.

        Returns:
            New TickState for this tick.
        """
        current_agents = [a.model_copy(deep=True) for a in self.tick_state.agent_states]
        current_constitution = self.tick_state.constitution.model_copy(deep=True)

        # Phase 2 stubs: proposals and voting
        proposals = self._collect_proposals(tick, current_agents, current_constitution)
        votes = self._run_votes(proposals, current_agents, current_constitution)
        current_constitution = self._apply_vote_outcomes(votes, current_constitution)

        # Economic step
        updated_agents = economic_step(current_agents, current_constitution)

        # Observation
        self._observe(tick, updated_agents, current_constitution)

        return TickState(
            tick=tick,
            agent_states=updated_agents,
            constitution=current_constitution,
            proposals_this_tick=proposals,
            votes=votes,
        )

    def _collect_proposals(
        self,
        tick: int,
        agents: list,
        constitution: Constitution,
    ) -> list[Proposal]:
        """Collect proposals from citizens. Phase 2 stub: returns empty list.

        Args:
            tick: Current tick number.
            agents: Current agent states.
            constitution: Current constitution.

        Returns:
            List of proposals (empty in Phase 1).
        """
        _ = tick, agents, constitution
        return []

    def _run_votes(
        self,
        proposals: list[Proposal],
        agents: list,
        constitution: Constitution,
    ) -> list[VoteOutcome]:
        """Run votes on proposals. Phase 2 stub: returns empty list.

        Args:
            proposals: Proposals to vote on.
            agents: Current agent states.
            constitution: Current constitution.

        Returns:
            List of vote outcomes (empty in Phase 1).
        """
        _ = proposals, agents, constitution
        return []

    def _apply_vote_outcomes(
        self,
        votes: list[VoteOutcome],
        constitution: Constitution,
    ) -> Constitution:
        """Apply passed proposals to the constitution. Phase 2 stub: no-op.

        Args:
            votes: Vote outcomes to process.
            constitution: Current constitution.

        Returns:
            Updated constitution (unchanged in Phase 1).
        """
        _ = votes
        return constitution

    def _observe(
        self,
        tick: int,
        agents: list,
        constitution: Constitution,
    ) -> None:
        """Record observation statistics. Phase 3 stub: no-op.

        Args:
            tick: Current tick number.
            agents: Current agent states.
            constitution: Current constitution.
        """
        _ = tick, agents, constitution
