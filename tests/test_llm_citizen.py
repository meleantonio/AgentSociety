"""Tests for optional LLM-based citizen reasoning."""

from __future__ import annotations

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead
from emergent_constitution.llm_citizen import CitizenLLM, MockCitizenLLM, PromptBuilder
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.proposal import Proposal
from emergent_constitution.rng import SimulationRNG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agents(count: int = 5) -> list[AgentState]:
    """Create a list of agents with varied endowments and values."""
    agents: list[AgentState] = []
    for i in range(count):
        eq = 0.2 + 0.6 * (i / max(count - 1, 1))
        agents.append(
            AgentState(
                id=f"agent_{i:04d}",
                wealth=50.0 + i * 30.0,
                productivity=8.0 + i * 2.0,
                utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
                value_vector=ValueVector(equality=round(eq, 2), liberty=round(1 - eq, 2)),
            )
        )
    return agents


# ---------------------------------------------------------------------------
# TestPromptBuilder
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    """PromptBuilder produces well-formed prompts with agent/constitution context."""

    def test_proposal_prompt_contains_agent_info(self, sample_agent: AgentState) -> None:
        agents = [sample_agent]
        prompt = PromptBuilder.build_proposal_prompt(sample_agent, Constitution(), agents)
        assert sample_agent.id in prompt
        assert str(sample_agent.productivity) in prompt

    def test_proposal_prompt_contains_constitution(self, sample_agent: AgentState) -> None:
        constitution = Constitution(tax_rate=0.25)
        prompt = PromptBuilder.build_proposal_prompt(sample_agent, constitution, [sample_agent])
        assert "25.00%" in prompt
        assert "private" in prompt

    def test_proposal_prompt_is_nonempty_string(self, sample_agent: AgentState) -> None:
        prompt = PromptBuilder.build_proposal_prompt(sample_agent, Constitution(), [sample_agent])
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_proposal_prompt_contains_values(self, sample_agent: AgentState) -> None:
        prompt = PromptBuilder.build_proposal_prompt(sample_agent, Constitution(), [sample_agent])
        assert "equality=" in prompt
        assert "liberty=" in prompt

    def test_vote_prompt_contains_proposal_info(self, sample_agent: AgentState) -> None:
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")
        prompt = PromptBuilder.build_vote_prompt(
            sample_agent, Constitution(), proposal, [sample_agent]
        )
        assert "tax_rate" in prompt
        assert "0.3" in prompt
        assert "agent_0001" in prompt

    def test_vote_prompt_is_nonempty_string(self, sample_agent: AgentState) -> None:
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")
        prompt = PromptBuilder.build_vote_prompt(
            sample_agent, Constitution(), proposal, [sample_agent]
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_vote_prompt_contains_agent_values(self, sample_agent: AgentState) -> None:
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")
        prompt = PromptBuilder.build_vote_prompt(
            sample_agent, Constitution(), proposal, [sample_agent]
        )
        assert "equality=" in prompt
        assert "liberty=" in prompt

    def test_proposal_prompt_wealth_context(self) -> None:
        """Prompt includes wealth relative to mean for multiple agents."""
        agents = _make_agents(5)
        prompt = PromptBuilder.build_proposal_prompt(agents[0], Constitution(), agents)
        assert "ratio to mean" in prompt
        assert "Mean wealth" in prompt

    def test_vote_prompt_includes_vote_instruction(self, sample_agent: AgentState) -> None:
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")
        prompt = PromptBuilder.build_vote_prompt(
            sample_agent, Constitution(), proposal, [sample_agent]
        )
        assert "YES" in prompt
        assert "NO" in prompt


# ---------------------------------------------------------------------------
# TestMockCitizenLLM
# ---------------------------------------------------------------------------


class TestMockCitizenLLM:
    """MockCitizenLLM returns valid proposals and votes deterministically."""

    def test_generate_proposal_returns_valid_or_none(self) -> None:
        rng = SimulationRNG(seed=42)
        agents = _make_agents(5)
        mock = MockCitizenLLM(rng=rng, all_agents=agents)

        results: list[Proposal | None] = []
        for agent in agents:
            result = mock.generate_proposal(agent, Constitution(), "test context")
            results.append(result)

        for r in results:
            assert r is None or isinstance(r, Proposal)

    def test_reason_vote_returns_bool(self) -> None:
        rng = SimulationRNG(seed=42)
        agents = _make_agents(5)
        mock = MockCitizenLLM(rng=rng, all_agents=agents)
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")

        for agent in agents:
            vote = mock.reason_vote(agent, Constitution(), proposal, "test context")
            assert isinstance(vote, bool)

    def test_deterministic_proposals(self) -> None:
        """Same seed produces same proposal outcomes."""
        agents = _make_agents(10)

        results1: list[Proposal | None] = []
        rng1 = SimulationRNG(seed=99)
        mock1 = MockCitizenLLM(rng=rng1, all_agents=agents)
        for agent in agents:
            results1.append(mock1.generate_proposal(agent, Constitution(), "ctx"))

        results2: list[Proposal | None] = []
        rng2 = SimulationRNG(seed=99)
        mock2 = MockCitizenLLM(rng=rng2, all_agents=agents)
        for agent in agents:
            results2.append(mock2.generate_proposal(agent, Constitution(), "ctx"))

        for r1, r2 in zip(results1, results2, strict=True):
            assert r1 == r2

    def test_deterministic_votes(self) -> None:
        """Same seed produces same vote outcomes."""
        agents = _make_agents(10)
        proposal = Proposal(rule_key="tax_rate", proposed_value=0.3, proposer_id="agent_0001")

        rng1 = SimulationRNG(seed=99)
        mock1 = MockCitizenLLM(rng=rng1, all_agents=agents)
        votes1 = [mock1.reason_vote(a, Constitution(), proposal, "ctx") for a in agents]

        rng2 = SimulationRNG(seed=99)
        mock2 = MockCitizenLLM(rng=rng2, all_agents=agents)
        votes2 = [mock2.reason_vote(a, Constitution(), proposal, "ctx") for a in agents]

        assert votes1 == votes2


# ---------------------------------------------------------------------------
# TestCitizenLLMInterface
# ---------------------------------------------------------------------------


class TestCitizenLLMInterface:
    """MockCitizenLLM is a proper implementation of the CitizenLLM interface."""

    def test_mock_is_instance_of_citizen_llm(self) -> None:
        rng = SimulationRNG(seed=42)
        mock = MockCitizenLLM(rng=rng, all_agents=[])
        assert isinstance(mock, CitizenLLM)

    def test_citizen_llm_is_abstract(self) -> None:
        """CitizenLLM cannot be instantiated directly."""
        try:
            CitizenLLM()  # type: ignore[abstract]
            raise AssertionError("Should not be able to instantiate abstract class")  # noqa: TRY301
        except TypeError:
            pass

    def test_mock_has_required_methods(self) -> None:
        rng = SimulationRNG(seed=42)
        mock = MockCitizenLLM(rng=rng, all_agents=[])
        assert hasattr(mock, "generate_proposal")
        assert hasattr(mock, "reason_vote")
        assert callable(mock.generate_proposal)
        assert callable(mock.reason_vote)


# ---------------------------------------------------------------------------
# TestLeadWithLLM
# ---------------------------------------------------------------------------


class TestLeadWithLLM:
    """Lead runs with MockCitizenLLM and produces valid, deterministic output."""

    def test_runs_to_completion(self) -> None:
        config = SimulationConfig(
            num_agents=5,
            max_ticks=10,
            seed=42,
            use_llm=True,
            llm_fraction=0.2,
        )
        lead = Lead(config)
        # Create the mock with shared RNG reference and agents
        mock = MockCitizenLLM(rng=lead.rng, all_agents=lead.tick_state.agent_states)
        lead.citizen_llm = mock
        output = lead.run()
        assert output.total_ticks == 10
        assert len(output.final_agent_states) == 5

    def test_correct_agent_count(self) -> None:
        config = SimulationConfig(
            num_agents=10,
            max_ticks=10,
            seed=42,
            use_llm=True,
            llm_fraction=0.3,
        )
        lead = Lead(config)
        mock = MockCitizenLLM(rng=lead.rng, all_agents=lead.tick_state.agent_states)
        lead.citizen_llm = mock
        output = lead.run()
        assert len(output.final_agent_states) == 10

    def test_deterministic_with_llm(self) -> None:
        """Two runs with same seed and MockCitizenLLM produce identical output."""
        config = SimulationConfig(
            num_agents=5,
            max_ticks=20,
            seed=42,
            use_llm=True,
            llm_fraction=0.4,
            proposal_interval=5,
        )

        lead1 = Lead(config)
        mock1 = MockCitizenLLM(rng=lead1.rng, all_agents=lead1.tick_state.agent_states)
        lead1.citizen_llm = mock1
        output1 = lead1.run()

        lead2 = Lead(config)
        mock2 = MockCitizenLLM(rng=lead2.rng, all_agents=lead2.tick_state.agent_states)
        lead2.citizen_llm = mock2
        output2 = lead2.run()

        assert output1.model_dump_json(indent=2) == output2.model_dump_json(indent=2)

    def test_positive_wealth_preserved(self) -> None:
        """All agents maintain non-negative wealth with LLM reasoning active."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=30,
            seed=42,
            use_llm=True,
            llm_fraction=0.5,
            proposal_interval=5,
        )
        lead = Lead(config)
        mock = MockCitizenLLM(rng=lead.rng, all_agents=lead.tick_state.agent_states)
        lead.citizen_llm = mock
        output = lead.run()
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0

    def test_without_llm_flag_ignores_mock(self) -> None:
        """When use_llm=False, providing a CitizenLLM has no effect."""
        config_no_llm = SimulationConfig(
            num_agents=5,
            max_ticks=10,
            seed=42,
            use_llm=False,
        )
        config_baseline = SimulationConfig(
            num_agents=5,
            max_ticks=10,
            seed=42,
            use_llm=False,
        )

        lead_with_mock = Lead(config_no_llm)
        mock = MockCitizenLLM(
            rng=lead_with_mock.rng,
            all_agents=lead_with_mock.tick_state.agent_states,
        )
        lead_with_mock.citizen_llm = mock
        output_with_mock = lead_with_mock.run()

        output_baseline = Lead(config_baseline).run()

        assert output_with_mock.model_dump_json() == output_baseline.model_dump_json()

    def test_llm_fraction_zero_falls_back(self) -> None:
        """With llm_fraction=0.0, all agents use rule-based logic."""
        config = SimulationConfig(
            num_agents=5,
            max_ticks=10,
            seed=42,
            use_llm=True,
            llm_fraction=0.0,
        )
        lead = Lead(config)
        mock = MockCitizenLLM(rng=lead.rng, all_agents=lead.tick_state.agent_states)
        lead.citizen_llm = mock
        # Should not crash even with fraction=0.0
        output = lead.run()
        assert output.total_ticks == 10

    def test_history_recorded_with_llm(self) -> None:
        """Observer still records history when LLM is active."""
        config = SimulationConfig(
            num_agents=5,
            max_ticks=20,
            seed=42,
            use_llm=True,
            llm_fraction=0.3,
            observer_interval=5,
        )
        lead = Lead(config)
        mock = MockCitizenLLM(rng=lead.rng, all_agents=lead.tick_state.agent_states)
        lead.citizen_llm = mock
        output = lead.run()
        assert len(output.history) == 20 // 5


# ---------------------------------------------------------------------------
# TestSimulationConfig LLM fields
# ---------------------------------------------------------------------------


class TestSimulationConfigLLM:
    """Config accepts and validates the new LLM fields."""

    def test_defaults(self) -> None:
        config = SimulationConfig()
        assert config.use_llm is False
        assert config.llm_fraction == 0.1

    def test_llm_fraction_bounds(self) -> None:
        config = SimulationConfig(llm_fraction=0.0)
        assert config.llm_fraction == 0.0

        config = SimulationConfig(llm_fraction=1.0)
        assert config.llm_fraction == 1.0

    def test_llm_fraction_invalid(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017
            SimulationConfig(llm_fraction=1.5)

        with pytest.raises(Exception):  # noqa: B017
            SimulationConfig(llm_fraction=-0.1)
