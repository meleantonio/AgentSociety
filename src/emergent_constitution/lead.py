"""Lead (Simulation Governor) — the main tick loop.

The Lead advances time in discrete ticks, enforces turn order, and
orchestrates proposals, votes, economic steps, and observation.
"""

from __future__ import annotations

import structlog

from emergent_constitution.citizen import decide_proposal, decide_trade, decide_votes
from emergent_constitution.coalition import form_coalitions
from emergent_constitution.config import SimulationConfig
from emergent_constitution.economics import apply_trades, economic_step
from emergent_constitution.initialization import initialize_simulation
from emergent_constitution.llm_citizen import CitizenLLM, PromptBuilder
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.history import HistoryEntry, SimulationOutput
from emergent_constitution.models.proposal import Proposal, TradeOffer, VoteOutcome
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
        citizen_llm: Optional LLM-based citizen reasoning implementation.
            When provided *and* ``config.use_llm`` is True, a fraction of
            agents (controlled by ``config.llm_fraction``) will use this
            interface for proposal generation and vote reasoning each tick.
    """

    def __init__(
        self,
        config: SimulationConfig,
        citizen_llm: CitizenLLM | None = None,
    ) -> None:
        self.config = config
        self.citizen_llm = citizen_llm
        self.tick_state: TickState
        self.rng: SimulationRNG
        self.history: list[HistoryEntry] = []
        self.last_observed_constitution: Constitution | None = None
        self.tick_state, self.rng = initialize_simulation(config)
        log.info("simulation.initialized", num_agents=config.num_agents, seed=config.seed)

    @property
    def _use_llm(self) -> bool:
        """Whether LLM reasoning is active for this run."""
        return self.config.use_llm and self.citizen_llm is not None

    def _select_llm_agents(self, agents: list[AgentState]) -> set[str]:
        """Select a subset of agent IDs to use LLM reasoning this tick.

        The subset size is ``ceil(len(agents) * config.llm_fraction)``,
        selected via seeded RNG for determinism.

        Args:
            agents: Current agent states.

        Returns:
            Set of agent IDs that should use LLM reasoning.
        """
        if not self._use_llm:
            return set()

        count = max(1, int(len(agents) * self.config.llm_fraction + 0.5))
        count = min(count, len(agents))

        # Deterministic selection: shuffle a copy and take first `count`
        ids = [a.id for a in agents]
        self.rng.shuffle(ids)
        return set(ids[:count])

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

        # Trade step
        trades = self._collect_trades(tick, current_agents, current_constitution)
        if trades:
            current_agents = apply_trades(current_agents, trades)

        # Economic step
        updated_agents = economic_step(current_agents, current_constitution)

        # Coalition formation (periodic)
        if tick % self.config.coalition_interval == 0:
            updated_agents = form_coalitions(updated_agents, self.rng)
            log.info(
                "coalitions.formed",
                tick=tick,
                num_coalitions=len(
                    {a.coalition_id for a in updated_agents if a.coalition_id is not None}
                ),
            )

        # Observation
        self._observe(tick, updated_agents, current_constitution)

        return TickState(
            tick=tick,
            agent_states=updated_agents,
            constitution=current_constitution,
            proposals_this_tick=proposals,
            votes=votes,
            trades_this_tick=trades,
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

        When LLM reasoning is active, a subset of agents (selected via
        ``_select_llm_agents``) use the ``CitizenLLM`` interface; the rest
        use the deterministic rule-based logic.

        Args:
            tick: Current tick number.
            agents: Current agent states (read-only copies).
            constitution: Current constitution.

        Returns:
            List of validated proposals.
        """
        if tick % self.config.proposal_interval != 0:
            return []

        llm_ids = self._select_llm_agents(agents)

        # Shuffle for fairness (deterministic via seeded RNG)
        agent_order = list(agents)
        self.rng.shuffle(agent_order)

        proposals: list[Proposal] = []
        for agent in agent_order:
            if agent.id in llm_ids and self.citizen_llm is not None:
                context = PromptBuilder.build_proposal_prompt(agent, constitution, agents)
                proposal = self.citizen_llm.generate_proposal(agent, constitution, context)
            else:
                proposal = decide_proposal(agent, constitution, agents, self.rng)

            if proposal is not None and validate_proposal(proposal):
                proposals.append(proposal)

        log.info(
            "proposals.collected",
            tick=tick,
            count=len(proposals),
            llm_agents=len(llm_ids),
        )
        return proposals

    def _collect_trades(
        self,
        tick: int,
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[TradeOffer]:
        """Collect bilateral trade offers from citizens on trade-interval ticks.

        Shuffles agent order via self.rng for fairness. Each agent may produce
        at most one trade offer.

        Args:
            tick: Current tick number.
            agents: Current agent states (read-only copies).
            constitution: Current constitution.

        Returns:
            List of trade offers.
        """
        if tick % self.config.trade_interval != 0:
            return []

        # Shuffle for fairness (deterministic via seeded RNG)
        agent_order = list(agents)
        self.rng.shuffle(agent_order)

        trades: list[TradeOffer] = []
        for agent in agent_order:
            trade = decide_trade(agent, agents, constitution, self.rng)
            if trade is not None:
                trades.append(trade)

        log.info("trades.collected", tick=tick, count=len(trades))
        return trades

    def _run_votes(
        self,
        proposals: list[Proposal],
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[VoteOutcome]:
        """Run votes on all proposals.

        Each agent votes on all proposals. When LLM reasoning is active,
        the selected subset uses the ``CitizenLLM`` interface per proposal;
        the rest use the deterministic rule-based logic.

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
        llm_ids = self._select_llm_agents(agents)

        # Collect each agent's votes on all proposals
        all_agent_votes: dict[str, dict[str, bool]] = {}
        for agent in agents:
            if agent.id in llm_ids and self.citizen_llm is not None:
                # LLM path: vote on each proposal individually
                votes_for_agent: dict[str, bool] = {}
                for proposal in proposals:
                    context = PromptBuilder.build_vote_prompt(
                        agent, constitution, proposal, agents
                    )
                    vote = self.citizen_llm.reason_vote(agent, constitution, proposal, context)
                    votes_for_agent[proposal.proposer_id] = vote
                all_agent_votes[agent.id] = votes_for_agent
            else:
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
