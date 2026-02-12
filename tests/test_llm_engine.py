"""Tests for LLM decision engine (Task 10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.llm_engine import (
    DefaultFallbackSolver,
    LLMDecisionEngine,
)
from emergent_constitution.llm_providers import MockProvider
from emergent_constitution.models.constitution import (
    ConstitutionV2,
    create_default_constitution,
)
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.rng import SimulationRNG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> SimulationRNG:
    return SimulationRNG(seed=42)


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(num_agents=5, max_ticks=10, seed=42)


@pytest.fixture
def market() -> MarketState:
    return MarketState(
        wage=1.0,
        interest_rate=0.05,
        aggregate_output=500.0,
        aggregate_consumption=400.0,
        government_spending=50.0,
    )


@pytest.fixture
def constitution() -> ConstitutionV2:
    return create_default_constitution()


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    role: OccupationalRole = OccupationalRole.WORKER,
) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=1.0,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
    )


@pytest.fixture
def households() -> list[HouseholdState]:
    return [_make_household(f"agent_{i:04d}", wealth=50.0 + i * 20) for i in range(5)]


@pytest.fixture
def firms() -> list[FirmState]:
    return [
        FirmState(
            id="firm_0000",
            owner_id="agent_0003",
            capital=100.0,
            labor_demand=2.0,
            tfp=1.0,
            worker_ids=["agent_0000", "agent_0001"],
            output=50.0,
            profit=10.0,
        )
    ]


@pytest.fixture
def engine(config: SimulationConfig, rng: SimulationRNG) -> LLMDecisionEngine:
    """Engine with MockProvider for deterministic testing."""
    provider = MockProvider(rng=rng, config=config)
    return LLMDecisionEngine(config=config, rng=rng, provider=provider)


# ---------------------------------------------------------------------------
# Context Building Tests (REQ-025)
# ---------------------------------------------------------------------------


class TestBuildContext:
    """Test that _build_context produces all 5 required components."""

    def test_context_has_personal_state(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        personal = ctx["personal_state"]
        assert personal["id"] == "agent_0000"
        assert personal["wealth"] == 100.0
        assert personal["productivity"] == 1.0
        assert personal["role"] == "worker"

    def test_context_has_economic_conditions(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        econ = ctx["economic_conditions"]
        assert econ["wage"] == 1.0
        assert econ["interest_rate"] == 0.05
        assert econ["aggregate_output"] == 500.0

    def test_context_has_constitution(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        const = ctx["constitution"]
        assert "rules" in const
        assert "voting_rule" in const
        assert "flat_tax" in const["rules"]

    def test_context_has_history_summary(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "Period 5 summary.")
        assert ctx["history_summary"] == "Period 5 summary."

    def test_context_has_messages_placeholder(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        assert "messages" in ctx
        assert isinstance(ctx["messages"], list)

    def test_context_default_history_text(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        assert ctx["history_summary"] == "No history available."

    def test_context_utility_params_included(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        params = ctx["personal_state"]["utility_params"]
        assert params["alpha"] == 0.5
        assert params["beta"] == 0.3
        assert params["gamma"] == 0.2

    def test_context_value_vector_included(
        self, engine: LLMDecisionEngine, market: MarketState, constitution: ConstitutionV2
    ) -> None:
        agent = _make_household()
        ctx = engine._build_context(agent, market, constitution, "")
        values = ctx["personal_state"]["value_vector"]
        assert values["equality"] == 0.5
        assert values["liberty"] == 0.5


# ---------------------------------------------------------------------------
# Cache Tests (REQ-029)
# ---------------------------------------------------------------------------


class TestCaching:
    """Test response caching for identical state-context pairs."""

    def test_cache_hit_on_identical_context(
        self,
        engine: LLMDecisionEngine,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """Same household state should produce cache hit on second call."""
        households = [_make_household("agent_0000", wealth=100.0)]

        # First call: cache miss
        engine.collect_economic_decisions(households, market, constitution)
        assert engine.cache_misses == 1
        assert engine.cache_hits == 0

        # Second call: same context -> cache hit
        engine.collect_economic_decisions(households, market, constitution)
        assert engine.cache_hits == 1

    def test_cache_miss_on_different_context(
        self,
        engine: LLMDecisionEngine,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """Different wealth should produce cache miss."""
        h1 = [_make_household("agent_0000", wealth=100.0)]
        h2 = [_make_household("agent_0000", wealth=200.0)]

        engine.collect_economic_decisions(h1, market, constitution)
        engine.collect_economic_decisions(h2, market, constitution)
        # Both should be misses (different context)
        assert engine.cache_misses == 2
        assert engine.cache_hits == 0

    def test_clear_cache(
        self,
        engine: LLMDecisionEngine,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        households = [_make_household()]
        engine.collect_economic_decisions(households, market, constitution)
        assert engine.cache_misses == 1

        engine.clear_cache()
        assert engine.cache_hits == 0
        assert engine.cache_misses == 0

        # After clearing, same context should miss again
        engine.collect_economic_decisions(households, market, constitution)
        assert engine.cache_misses == 1


# ---------------------------------------------------------------------------
# Batch Splitting Tests
# ---------------------------------------------------------------------------


class TestBatching:
    """Test that _batch_call splits into correct batch sizes."""

    def test_batch_splitting(self, config: SimulationConfig, rng: SimulationRNG) -> None:
        """Verify batching splits agents into correct chunks."""
        provider = MagicMock()
        provider.generate_batch.return_value = [
            EconomicDecision(consumption=50.0, leisure=0.3).model_dump_json()
        ] * 3

        # Create engine with batch_size=3
        object.__setattr__(config, "llm_batch_size", 3)
        engine = LLMDecisionEngine(config=config, rng=rng, provider=provider)

        # 7 agents should produce 3 batches: [3, 3, 1]
        agent_ids = [f"agent_{i:04d}" for i in range(7)]
        messages_map = {aid: [{"role": "user", "content": f"Agent {aid}"}] for aid in agent_ids}

        # Third batch has 1 agent
        provider.generate_batch.side_effect = [
            [EconomicDecision(consumption=50.0, leisure=0.3).model_dump_json()] * 3,
            [EconomicDecision(consumption=50.0, leisure=0.3).model_dump_json()] * 3,
            [EconomicDecision(consumption=50.0, leisure=0.3).model_dump_json()] * 1,
        ]

        results = engine._batch_call(agent_ids, messages_map, EconomicDecision)
        assert len(results) == 7
        assert provider.generate_batch.call_count == 3

    def test_single_batch(self, config: SimulationConfig, rng: SimulationRNG) -> None:
        """When agents fit in one batch, only one call is made."""
        provider = MagicMock()
        provider.generate_batch.return_value = [
            EconomicDecision(consumption=50.0, leisure=0.3).model_dump_json()
        ] * 3

        engine = LLMDecisionEngine(config=config, rng=rng, provider=provider)

        agent_ids = [f"agent_{i:04d}" for i in range(3)]
        messages_map = {aid: [{"role": "user", "content": f"Agent {aid}"}] for aid in agent_ids}

        results = engine._batch_call(agent_ids, messages_map, EconomicDecision)
        assert len(results) == 3
        assert provider.generate_batch.call_count == 1


# ---------------------------------------------------------------------------
# Economic Decision Tests
# ---------------------------------------------------------------------------


class TestCollectEconomicDecisions:
    """Test collect_economic_decisions returns valid decisions for all agents."""

    def test_returns_decisions_for_all_agents(
        self,
        engine: LLMDecisionEngine,
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        decisions = engine.collect_economic_decisions(households, market, constitution)
        assert len(decisions) == len(households)
        for h in households:
            assert h.id in decisions
            d = decisions[h.id]
            assert isinstance(d, EconomicDecision)
            assert d.consumption >= 0.0
            assert 0.0 <= d.leisure <= 1.0

    def test_deterministic_with_mock(
        self,
        config: SimulationConfig,
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """Same seed -> same decisions (PROP-001)."""
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=42)
        p1 = MockProvider(rng=rng1, config=config)
        p2 = MockProvider(rng=rng2, config=config)
        e1 = LLMDecisionEngine(config=config, rng=rng1, provider=p1)
        e2 = LLMDecisionEngine(config=config, rng=rng2, provider=p2)

        d1 = e1.collect_economic_decisions(households, market, constitution)
        d2 = e2.collect_economic_decisions(households, market, constitution)

        for h in households:
            assert d1[h.id].consumption == d2[h.id].consumption
            assert d1[h.id].leisure == d2[h.id].leisure


# ---------------------------------------------------------------------------
# Entrepreneurial Decision Tests
# ---------------------------------------------------------------------------


class TestCollectEntrepreneurialDecisions:
    """Test collect_entrepreneurial_decisions."""

    def test_returns_decisions_for_all_agents(
        self,
        engine: LLMDecisionEngine,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        decisions = engine.collect_entrepreneurial_decisions(
            households, firms, market, constitution
        )
        assert len(decisions) == len(households)
        for h in households:
            assert h.id in decisions
            assert isinstance(decisions[h.id], EntrepreneurialDecision)

    def test_firm_context_included(
        self,
        engine: LLMDecisionEngine,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """Entrepreneur's owned firms should appear in context."""
        owner = _make_household("agent_0003", role=OccupationalRole.ENTREPRENEUR)
        firms = [
            FirmState(
                id="firm_0000",
                owner_id="agent_0003",
                capital=100.0,
                labor_demand=2.0,
                tfp=1.0,
            )
        ]
        decisions = engine.collect_entrepreneurial_decisions([owner], firms, market, constitution)
        assert "agent_0003" in decisions


# ---------------------------------------------------------------------------
# Political Decision Tests
# ---------------------------------------------------------------------------


class TestCollectPoliticalDecisions:
    """Test collect_political_decisions."""

    def test_returns_decisions_for_all_agents(
        self,
        engine: LLMDecisionEngine,
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        decisions = engine.collect_political_decisions(households, constitution, market)
        assert len(decisions) == len(households)
        for h in households:
            assert h.id in decisions
            assert isinstance(decisions[h.id], PoliticalDecision)

    def test_political_default_is_no_action(
        self,
        engine: LLMDecisionEngine,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """MockProvider returns no proposal and no votes by default."""
        households = [_make_household()]
        decisions = engine.collect_political_decisions(households, constitution, market)
        d = decisions["agent_0000"]
        assert d.proposal is None
        assert d.votes == {}


# ---------------------------------------------------------------------------
# Fallback Tests (REQ-027)
# ---------------------------------------------------------------------------


class TestFallback:
    """Test that fallback solver is used when LLM fails."""

    def test_fallback_on_provider_failure(
        self,
        config: SimulationConfig,
        rng: SimulationRNG,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """When provider returns invalid JSON, fallback should be used."""
        provider = MagicMock()
        # Return invalid JSON that won't parse as EconomicDecision
        provider.generate_batch.return_value = ["not valid json"]

        engine = LLMDecisionEngine(config=config, rng=rng, provider=provider)

        households = [_make_household()]
        decisions = engine.collect_economic_decisions(households, market, constitution)

        # Should still get a valid decision (from fallback)
        assert "agent_0000" in decisions
        d = decisions["agent_0000"]
        assert isinstance(d, EconomicDecision)
        assert d.consumption >= 0.0

    def test_fallback_on_empty_response(
        self,
        config: SimulationConfig,
        rng: SimulationRNG,
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> None:
        """Empty string response triggers fallback."""
        provider = MagicMock()
        provider.generate_batch.return_value = [""]

        engine = LLMDecisionEngine(config=config, rng=rng, provider=provider)

        households = [_make_household()]
        decisions = engine.collect_economic_decisions(households, market, constitution)
        assert "agent_0000" in decisions
        assert isinstance(decisions["agent_0000"], EconomicDecision)

    def test_default_fallback_solver_economic(self) -> None:
        """DefaultFallbackSolver returns valid economic decisions."""
        solver = DefaultFallbackSolver()
        agent = _make_household(wealth=200.0)
        market = MarketState(wage=1.0, interest_rate=0.05)
        decision = solver.solve_economic(agent, market)
        assert decision.consumption >= 0.0
        assert 0.0 <= decision.leisure <= 1.0

    def test_default_fallback_solver_entrepreneurial(self) -> None:
        """DefaultFallbackSolver returns no-op entrepreneurial decisions."""
        solver = DefaultFallbackSolver()
        agent = _make_household()
        market = MarketState(wage=1.0, interest_rate=0.05)
        decision = solver.solve_entrepreneurial(agent, [], market)
        assert not decision.create_firm
        assert not decision.close_firm


# ---------------------------------------------------------------------------
# Engine Initialization Tests
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Test engine initialization and configuration."""

    def test_creates_with_default_provider(
        self, config: SimulationConfig, rng: SimulationRNG
    ) -> None:
        """When no provider given, creates from config (MockProvider for use_llm=False)."""
        engine = LLMDecisionEngine(config=config, rng=rng)
        assert isinstance(engine._provider, MockProvider)

    def test_creates_with_custom_provider(
        self, config: SimulationConfig, rng: SimulationRNG
    ) -> None:
        provider = MockProvider(rng=rng, config=config)
        engine = LLMDecisionEngine(config=config, rng=rng, provider=provider)
        assert engine._provider is provider

    def test_creates_with_custom_fallback(
        self, config: SimulationConfig, rng: SimulationRNG
    ) -> None:
        fallback = DefaultFallbackSolver()
        engine = LLMDecisionEngine(config=config, rng=rng, fallback_solver=fallback)
        assert engine._fallback is fallback
