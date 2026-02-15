"""Tests for LLM provider abstraction (Task 9)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.llm_providers import (
    AnthropicProvider,
    LLMParseError,
    LLMProvider,
    LLMProviderError,
    MockProvider,
    compute_cache_key,
    create_provider,
)
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.rng import SimulationRNG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> SimulationRNG:
    """Seeded RNG for deterministic tests."""
    return SimulationRNG(seed=42)


@pytest.fixture
def config() -> SimulationConfig:
    """Default test config."""
    return SimulationConfig(num_agents=5, max_ticks=10, seed=42)


@pytest.fixture
def mock_provider(rng: SimulationRNG, config: SimulationConfig) -> MockProvider:
    """A mock provider instance."""
    return MockProvider(rng=rng, config=config)


# ---------------------------------------------------------------------------
# LLMProvider Protocol Tests
# ---------------------------------------------------------------------------


class TestLLMProviderProtocol:
    """Test that implementations satisfy the LLMProvider protocol."""

    def test_mock_provider_is_llm_provider(self, mock_provider: MockProvider) -> None:
        assert isinstance(mock_provider, LLMProvider)

    def test_protocol_has_generate_method(self) -> None:
        assert hasattr(LLMProvider, "generate")

    def test_protocol_has_generate_batch_method(self) -> None:
        assert hasattr(LLMProvider, "generate_batch")


# ---------------------------------------------------------------------------
# MockProvider Tests
# ---------------------------------------------------------------------------


class TestMockProvider:
    """Test MockProvider returns valid decisions for all schema types."""

    def test_generate_economic_decision(self, mock_provider: MockProvider) -> None:
        messages = [{"role": "user", "content": "Make an economic decision."}]
        result = mock_provider.generate(messages, EconomicDecision)
        parsed = EconomicDecision.model_validate_json(result)
        assert parsed.consumption >= 0.0
        assert 0.0 <= parsed.leisure <= 1.0

    def test_generate_entrepreneurial_decision(self, mock_provider: MockProvider) -> None:
        messages = [{"role": "user", "content": "Manage your firm."}]
        result = mock_provider.generate(messages, EntrepreneurialDecision)
        parsed = EntrepreneurialDecision.model_validate_json(result)
        assert not parsed.create_firm
        assert not parsed.close_firm

    def test_generate_political_decision(self, mock_provider: MockProvider) -> None:
        messages = [{"role": "user", "content": "Vote on proposals."}]
        result = mock_provider.generate(messages, PoliticalDecision)
        parsed = PoliticalDecision.model_validate_json(result)
        assert parsed.proposal is None
        assert parsed.votes == {}

    def test_generate_batch(self, mock_provider: MockProvider) -> None:
        batch = [[{"role": "user", "content": f"Agent {i}"}] for i in range(5)]
        results = mock_provider.generate_batch(batch, EconomicDecision)
        assert len(results) == 5
        for result in results:
            parsed = EconomicDecision.model_validate_json(result)
            assert parsed.consumption >= 0.0

    def test_determinism(self, config: SimulationConfig) -> None:
        """Same seed produces same output (PROP-001)."""
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=42)
        provider1 = MockProvider(rng=rng1, config=config)
        provider2 = MockProvider(rng=rng2, config=config)

        messages = [{"role": "user", "content": "Decide."}]
        result1 = provider1.generate(messages, EconomicDecision)
        result2 = provider2.generate(messages, EconomicDecision)
        assert result1 == result2


# ---------------------------------------------------------------------------
# AnthropicProvider Tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Test AnthropicProvider with mocked Anthropic client."""

    def _make_mock_response(self, tool_input: dict) -> MagicMock:
        """Create a mock Anthropic API response with a tool_use block."""
        block = MagicMock()
        block.type = "tool_use"
        block.input = tool_input
        response = MagicMock()
        response.content = [block]
        return response

    def _make_text_only_response(self) -> MagicMock:
        """Create a mock response with no tool_use block."""
        block = MagicMock()
        block.type = "text"
        block.text = "I cannot use tools"
        response = MagicMock()
        response.content = [block]
        return response

    def _make_provider(self) -> AnthropicProvider:
        """Create an AnthropicProvider with _init_client mocked out.

        Sets up fake error classes on the instance so _call_with_retry
        can use them in except clauses.
        """
        with patch("emergent_constitution.llm_providers.AnthropicProvider._init_client"):
            config = SimulationConfig(num_agents=5, max_ticks=10, use_llm=True)
            provider = AnthropicProvider(config)

        # Set fake error classes (normally set by _init_client)
        provider._rate_limit_error = type("RateLimitError", (Exception,), {})
        provider._api_status_error = type("APIStatusError", (Exception,), {})
        provider._api_timeout_error = type("APITimeoutError", (Exception,), {})
        provider._client = MagicMock()
        return provider

    def test_generate_success(self) -> None:
        provider = self._make_provider()

        tool_input = {"consumption": 75.0, "leisure": 0.4}
        provider._client.messages.create.return_value = self._make_mock_response(tool_input)

        messages = [{"role": "user", "content": "Make a decision."}]
        result = provider.generate(messages, EconomicDecision)
        parsed = EconomicDecision.model_validate_json(result)
        assert parsed.consumption == 75.0
        assert parsed.leisure == 0.4

    def test_generate_parse_error_no_tool_block(self) -> None:
        provider = self._make_provider()
        provider._client.messages.create.return_value = self._make_text_only_response()

        messages = [{"role": "user", "content": "Make a decision."}]
        with pytest.raises(LLMProviderError):
            provider.generate(messages, EconomicDecision)

    def test_generate_schema_validation_failure(self) -> None:
        provider = self._make_provider()

        # Invalid: leisure > 1.0
        tool_input = {"consumption": 75.0, "leisure": 5.0}
        provider._client.messages.create.return_value = self._make_mock_response(tool_input)

        messages = [{"role": "user", "content": "Make a decision."}]
        with pytest.raises(LLMParseError):
            provider.generate(messages, EconomicDecision)

    @patch("emergent_constitution.llm_providers.time.sleep")
    def test_generate_rate_limit_retry(self, mock_sleep: MagicMock) -> None:
        """Test exponential backoff on rate limit errors."""
        provider = self._make_provider()

        class FakeRateLimitError(Exception):
            pass

        provider._rate_limit_error = FakeRateLimitError

        # First two calls raise rate limit, third succeeds
        tool_input = {"consumption": 50.0, "leisure": 0.3}
        provider._client.messages.create.side_effect = [
            FakeRateLimitError("rate limit"),
            FakeRateLimitError("rate limit"),
            self._make_mock_response(tool_input),
        ]

        messages = [{"role": "user", "content": "Decide."}]
        result = provider.generate(messages, EconomicDecision)

        parsed = EconomicDecision.model_validate_json(result)
        assert parsed.consumption == 50.0
        assert mock_sleep.call_count == 2  # Two retries before success

    @patch("emergent_constitution.llm_providers.time.sleep")
    def test_generate_timeout_retry_then_fail(self, mock_sleep: MagicMock) -> None:
        """Test that timeout errors exhaust retries and raise."""
        provider = self._make_provider()

        class FakeTimeoutError(Exception):
            pass

        provider._api_timeout_error = FakeTimeoutError

        # All calls timeout
        provider._client.messages.create.side_effect = FakeTimeoutError("timeout")

        messages = [{"role": "user", "content": "Decide."}]
        with pytest.raises(LLMProviderError, match="failed after 3 retries"):
            provider.generate(messages, EconomicDecision)

    def test_generate_batch_partial_failure(self) -> None:
        """Test batch generation handles partial failures gracefully."""
        provider = self._make_provider()

        good_input = {"consumption": 60.0, "leisure": 0.2}
        # First call succeeds; second call and its retries all fail parsing
        provider._client.messages.create.side_effect = [
            self._make_mock_response(good_input),
            self._make_text_only_response(),
            self._make_text_only_response(),
            self._make_text_only_response(),
        ]

        batch = [
            [{"role": "user", "content": "Agent 0"}],
            [{"role": "user", "content": "Agent 1"}],
        ]

        results = provider.generate_batch(batch, EconomicDecision)
        assert len(results) == 2
        # First should be valid
        parsed = EconomicDecision.model_validate_json(results[0])
        assert parsed.consumption == 60.0
        # Second should be empty JSON (failed)
        assert results[1] == "{}"

    def test_schema_to_tool(self) -> None:
        """Test Pydantic schema to Anthropic tool definition conversion."""
        tool = AnthropicProvider._schema_to_tool(EconomicDecision)
        assert tool["name"] == "EconomicDecision"
        assert "input_schema" in tool
        assert "properties" in tool["input_schema"]
        assert "consumption" in tool["input_schema"]["properties"]
        assert "leisure" in tool["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Test the provider factory function."""

    def test_create_mock_when_llm_disabled(
        self, rng: SimulationRNG, config: SimulationConfig
    ) -> None:
        # Default config has use_llm=False
        provider = create_provider(config, rng)
        assert isinstance(provider, MockProvider)

    def test_create_mock_in_benchmark_mode(self, rng: SimulationRNG) -> None:
        """benchmark_mode=True forces MockProvider even with use_llm=True."""
        config = SimulationConfig(num_agents=5, max_ticks=10, use_llm=True)
        # Pydantic doesn't allow setting undefined attributes, so we use
        # object.__setattr__ to simulate a v2 config with benchmark_mode.
        object.__setattr__(config, "benchmark_mode", True)
        provider = create_provider(config, rng)
        assert isinstance(provider, MockProvider)

    def test_create_anthropic_falls_back_to_mock_without_package(self, rng: SimulationRNG) -> None:
        """use_llm=True with anthropic provider falls back to MockProvider if package not installed."""
        config = SimulationConfig(num_agents=5, max_ticks=10, use_llm=True)
        # Only run this test if anthropic is not installed
        try:
            import anthropic  # noqa: F401

            pytest.skip("anthropic package is installed")
        except ImportError:
            provider = create_provider(config, rng)
            assert isinstance(provider, MockProvider)


# ---------------------------------------------------------------------------
# Cache Key Tests
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    """Test the response cache key computation."""

    def test_deterministic(self) -> None:
        context = {"wealth": 100.0, "productivity": 10.0, "role": "worker"}
        key1 = compute_cache_key(context)
        key2 = compute_cache_key(context)
        assert key1 == key2

    def test_different_contexts_different_keys(self) -> None:
        ctx1 = {"wealth": 100.0}
        ctx2 = {"wealth": 200.0}
        assert compute_cache_key(ctx1) != compute_cache_key(ctx2)

    def test_key_is_hex_sha256(self) -> None:
        key = compute_cache_key({"a": 1})
        assert len(key) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in key)

    def test_order_independent(self) -> None:
        """Keys should be the same regardless of dict insertion order."""
        ctx1 = {"b": 2, "a": 1}
        ctx2 = {"a": 1, "b": 2}
        assert compute_cache_key(ctx1) == compute_cache_key(ctx2)


# ---------------------------------------------------------------------------
# Error Hierarchy Tests
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """Test the error class hierarchy."""

    def test_timeout_is_provider_error(self) -> None:
        from emergent_constitution.llm_providers import LLMTimeoutError

        assert issubclass(LLMTimeoutError, LLMProviderError)

    def test_parse_is_provider_error(self) -> None:
        assert issubclass(LLMParseError, LLMProviderError)

    def test_rate_limit_is_provider_error(self) -> None:
        from emergent_constitution.llm_providers import LLMRateLimitError

        assert issubclass(LLMRateLimitError, LLMProviderError)
