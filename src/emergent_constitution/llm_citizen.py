"""Optional LLM-based citizen reasoning — abstract interface and mock implementation.

This module provides an abstract interface (``CitizenLLM``) that users can
implement with any LLM provider.  No external dependencies are added; only
the deterministic ``MockCitizenLLM`` (which delegates to rule-based logic)
and a ``PromptBuilder`` (pure string formatting) are shipped.
"""

from __future__ import annotations

import abc
from statistics import mean

import structlog

from emergent_constitution.citizen import decide_proposal, decide_votes
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.proposal import Proposal
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class CitizenLLM(abc.ABC):
    """Abstract interface for LLM-based citizen reasoning.

    Implementors provide actual LLM calls.  The simulation governor
    (``Lead``) delegates proposal generation and vote reasoning to this
    interface for a configurable fraction of agents each tick.
    """

    @abc.abstractmethod
    def generate_proposal(
        self,
        agent: AgentState,
        constitution: Constitution,
        context: str,
    ) -> Proposal | None:
        """Generate a rule-change proposal (or None to abstain).

        Args:
            agent: The agent's current state.
            constitution: The current constitution.
            context: A formatted prompt string with situational context.

        Returns:
            A Proposal, or None if the agent does not wish to propose.
        """

    @abc.abstractmethod
    def reason_vote(
        self,
        agent: AgentState,
        constitution: Constitution,
        proposal: Proposal,
        context: str,
    ) -> bool:
        """Decide whether to vote *for* a proposal.

        Args:
            agent: The agent's current state.
            constitution: The current constitution.
            proposal: The proposal to vote on.
            context: A formatted prompt string with situational context.

        Returns:
            True to vote in favour, False to vote against.
        """


# ---------------------------------------------------------------------------
# Deterministic mock (for testing)
# ---------------------------------------------------------------------------


class MockCitizenLLM(CitizenLLM):
    """Deterministic mock that delegates to existing rule-based logic.

    This allows integration tests to exercise the LLM code-path without
    requiring real API keys or network access.

    Args:
        rng: Seeded RNG shared with the simulation for determinism.
        all_agents: Current list of all agent states (updated each tick by
            the caller via :pyattr:`all_agents`).
    """

    def __init__(self, rng: SimulationRNG, all_agents: list[AgentState]) -> None:
        self.rng = rng
        self.all_agents = all_agents

    def generate_proposal(
        self,
        agent: AgentState,
        constitution: Constitution,
        context: str,
    ) -> Proposal | None:
        """Delegate to ``citizen.decide_proposal``."""
        _ = context  # prompt not used in mock
        return decide_proposal(agent, constitution, self.all_agents, self.rng)

    def reason_vote(
        self,
        agent: AgentState,
        constitution: Constitution,
        proposal: Proposal,
        context: str,
    ) -> bool:
        """Delegate to ``citizen.decide_votes`` for a single proposal."""
        _ = context  # prompt not used in mock
        votes = decide_votes(agent, constitution, [proposal], self.all_agents, self.rng)
        # decide_votes keys by proposer_id
        return votes.get(proposal.proposer_id, False)


# ---------------------------------------------------------------------------
# Prompt builder (pure string formatting, no API calls)
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds prompts for LLM-based citizen reasoning.

    All methods are static and pure — they format agent state, constitution,
    and context into human-readable prompt strings.  They never call an API.
    """

    @staticmethod
    def build_proposal_prompt(
        agent: AgentState,
        constitution: Constitution,
        all_agents: list[AgentState],
    ) -> str:
        """Build a prompt asking an agent what rule change to propose.

        Args:
            agent: The proposing agent's state.
            constitution: The current constitution.
            all_agents: All agent states (for wealth context).

        Returns:
            A formatted prompt string.
        """
        mean_wealth = mean(a.wealth for a in all_agents) if all_agents else 0.0
        wealth_ratio = agent.wealth / mean_wealth if mean_wealth > 0 else 1.0

        return (
            "You are a citizen in a self-governing society.\n"
            "\n"
            "## Your profile\n"
            f"- ID: {agent.id}\n"
            f"- Wealth: {agent.wealth:.2f} (mean across citizens: {mean_wealth:.2f}, "
            f"ratio to mean: {wealth_ratio:.2f})\n"
            f"- Productivity: {agent.productivity:.2f}\n"
            f"- Values: equality={agent.value_vector.equality:.2f}, "
            f"liberty={agent.value_vector.liberty:.2f}\n"
            f"- Coalition: {agent.coalition_id or 'none'}\n"
            "\n"
            "## Current constitution\n"
            f"- Property rule: {constitution.property_rule.value}\n"
            f"- Tax rate: {constitution.tax_rate:.2%}\n"
            f"- Voting rule: {constitution.voting_rule.value}\n"
            f"- Redistribution: {constitution.redistribution_rule.value}\n"
            "\n"
            "## Society overview\n"
            f"- Number of citizens: {len(all_agents)}\n"
            f"- Mean wealth: {mean_wealth:.2f}\n"
            "\n"
            "## Task\n"
            "Given your values and economic situation, propose a single rule change "
            "to the constitution, or respond with ABSTAIN if you are satisfied.\n"
            "\n"
            "Valid rule keys and values:\n"
            "- property_rule: private | communal | mixed\n"
            "- tax_rate: a float between 0.0 and 1.0\n"
            "- voting_rule: majority | supermajority | unanimity\n"
            "- redistribution_rule: none | flat | progressive\n"
            "\n"
            "Respond in the format:\n"
            "RULE_KEY: proposed_value\n"
            "REASON: one-sentence justification\n"
        )

    @staticmethod
    def build_vote_prompt(
        agent: AgentState,
        constitution: Constitution,
        proposal: Proposal,
        all_agents: list[AgentState],
    ) -> str:
        """Build a prompt asking an agent how to vote on a proposal.

        Args:
            agent: The voting agent's state.
            constitution: The current constitution.
            proposal: The proposal to vote on.
            all_agents: All agent states (for wealth context).

        Returns:
            A formatted prompt string.
        """
        mean_wealth = mean(a.wealth for a in all_agents) if all_agents else 0.0
        wealth_ratio = agent.wealth / mean_wealth if mean_wealth > 0 else 1.0

        return (
            "You are a citizen in a self-governing society.\n"
            "\n"
            "## Your profile\n"
            f"- ID: {agent.id}\n"
            f"- Wealth: {agent.wealth:.2f} (mean: {mean_wealth:.2f}, "
            f"ratio to mean: {wealth_ratio:.2f})\n"
            f"- Productivity: {agent.productivity:.2f}\n"
            f"- Values: equality={agent.value_vector.equality:.2f}, "
            f"liberty={agent.value_vector.liberty:.2f}\n"
            "\n"
            "## Current constitution\n"
            f"- Property rule: {constitution.property_rule.value}\n"
            f"- Tax rate: {constitution.tax_rate:.2%}\n"
            f"- Voting rule: {constitution.voting_rule.value}\n"
            f"- Redistribution: {constitution.redistribution_rule.value}\n"
            "\n"
            "## Proposal under consideration\n"
            f"- Proposed by: {proposal.proposer_id}\n"
            f"- Change: {proposal.rule_key} -> {proposal.proposed_value}\n"
            "\n"
            "## Task\n"
            "Vote YES or NO on this proposal. Consider both your values and "
            "your material self-interest.\n"
            "\n"
            "Respond in the format:\n"
            "VOTE: YES or NO\n"
            "REASON: one-sentence justification\n"
        )
