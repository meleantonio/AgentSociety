"""Lead (Simulation Governor) — the main tick loop.

The Lead advances time in discrete ticks, enforces turn order, and
orchestrates proposals, votes, economic steps, and observation.
"""

from __future__ import annotations

import structlog

from emergent_constitution.citizen import decide_proposal, decide_votes
from emergent_constitution.config import SimulationConfig
from emergent_constitution.economics import economic_step
from emergent_constitution.initialization import initialize_simulation
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.history import HistoryEntry, SimulationOutput
from emergent_constitution.models.proposal import Proposal, VoteOutcome
from emergent_constitution.models.tick import TickState
from emergent_constitution.observer import observe_tick
from emergent_constitution.rng import SimulationRNG
from emergent_constitution.voting import (
    apply_passed_proposals,
    tally_votes,
    validate_proposal,
)

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
        self.last_observed_constitution: Constitution | None = None
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
        """Execute a single tick: proposals -> votes -> economics -> observe.

        Args:
            tick: Current tick number.

        Returns:
            New TickState for this tick.
        """
        current_agents = [a.model_copy(deep=True) for a in self.tick_state.agent_states]
        current_constitution = self.tick_state.constitution.model_copy(deep=True)

        # Phase 2: proposals and voting
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
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[Proposal]:
        """Collect proposals from citizens on proposal-interval ticks.

        Shuffles agent order via self.rng for fairness. Each agent may produce
        at most one proposal, which is validated before inclusion.

        Args:
            tick: Current tick number.
            agents: Current agent states (read-only copies).
            constitution: Current constitution.

        Returns:
            List of validated proposals.
        """
        if tick % self.config.proposal_interval != 0:
            return []

        # Shuffle for fairness (deterministic via seeded RNG)
        agent_order = list(agents)
        self.rng.shuffle(agent_order)

        proposals: list[Proposal] = []
        for agent in agent_order:
            proposal = decide_proposal(agent, constitution, agents, self.rng)
            if proposal is not None and validate_proposal(proposal):
                proposals.append(proposal)

        log.info("proposals.collected", tick=tick, count=len(proposals))
        return proposals

    def _run_votes(
        self,
        proposals: list[Proposal],
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[VoteOutcome]:
        """Run votes on all proposals.

        Each agent votes on all proposals. Results are tallied according to
        the current voting rule.

        Args:
            proposals: Proposals to vote on.
            agents: Current agent states.
            constitution: Current constitution.

        Returns:
            List of vote outcomes.
        """
        if not proposals:
            return []

        total_eligible = len(agents)

        # Collect each agent's votes on all proposals
        all_agent_votes: dict[str, dict[str, bool]] = {}
        for agent in agents:
            all_agent_votes[agent.id] = decide_votes(
                agent,
                constitution,
                proposals,
                agents,
                self.rng,
            )

        # Tally per proposal
        outcomes: list[VoteOutcome] = []
        for proposal in proposals:
            proposal_votes: dict[str, bool] = {}
            for agent_id, agent_votes in all_agent_votes.items():
                if proposal.proposer_id in agent_votes:
                    proposal_votes[agent_id] = agent_votes[proposal.proposer_id]

            outcome = tally_votes(
                proposal,
                proposal_votes,
                constitution.voting_rule,
                total_eligible,
            )
            outcomes.append(outcome)
            log.debug(
                "vote.tallied",
                rule_key=proposal.rule_key,
                passed=outcome.passed,
                votes_for=outcome.votes_for,
                votes_against=outcome.votes_against,
            )

        return outcomes

    def _apply_vote_outcomes(
        self,
        votes: list[VoteOutcome],
        constitution: Constitution,
    ) -> Constitution:
        """Apply passed proposals to the constitution.

        Args:
            votes: Vote outcomes to process.
            constitution: Current constitution.

        Returns:
            Updated constitution with passed proposals applied.
        """
        return apply_passed_proposals(votes, constitution)

    def _observe(
        self,
        tick: int,
        agents: list[AgentState],
        constitution: Constitution,
    ) -> None:
        """Record observation statistics at observer-interval ticks.

        Args:
            tick: Current tick number.
            agents: Current agent states.
            constitution: Current constitution.
        """
        if tick % self.config.observer_interval != 0:
            return

        entry = observe_tick(tick, agents, constitution, self.last_observed_constitution)
        self.history.append(entry)
        self.last_observed_constitution = constitution.model_copy(deep=True)
        log.info(
            "observation.recorded",
            tick=tick,
            gini=round(entry.gini, 4),
            mean_wealth=round(entry.mean_wealth, 2),
            rule_changes=len(entry.rule_changes),
        )
