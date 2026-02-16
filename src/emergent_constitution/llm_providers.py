"""LLM provider abstraction — pluggable interface for decision generation.

Provides a protocol for LLM-based decision making and three implementations:
- MockProvider: deterministic mock for testing (no API calls)
- AnthropicProvider: Anthropic Messages API with structured output
- OpenAICompatibleProvider: OpenAI-compatible local servers (LM Studio, Ollama, vLLM, llama.cpp)

Traceability: REQ-024, REQ-027, REQ-029, PROP-001
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from emergent_constitution.config import SimulationConfig
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM-based agent decision generation.

    Implementations must provide both single and batch generation methods.
    All responses must be valid JSON conforming to the provided Pydantic schema.
    """

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> str:
        """Generate a single response from the LLM.

        Args:
            messages: Chat messages (role + content dicts).
            schema: Pydantic model class for response validation.

        Returns:
            Raw JSON string conforming to the schema.

        Raises:
            LLMProviderError: On unrecoverable generation failure.
        """
        ...

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> list[str]:
        """Generate responses for a batch of message sequences.

        Args:
            batch: List of message sequences, one per agent.
            schema: Pydantic model class for response validation.

        Returns:
            List of raw JSON strings, one per input sequence.

        Raises:
            LLMProviderError: On unrecoverable generation failure.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Base error for LLM provider failures."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM call times out."""


class LLMParseError(LLMProviderError):
    """Raised when an LLM response cannot be parsed as valid JSON/schema."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider returns a rate limit error."""


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class MockProvider:
    """Deterministic mock LLM provider for testing without API calls.

    Returns sensible default decisions based on simple heuristics:
    - Economic decisions: consume half of available resources, moderate leisure.
    - Entrepreneurial decisions: no action (safe default).
    - Political decisions: no proposal, no votes.

    Uses the seeded RNG for any stochastic behavior to maintain determinism
    (PROP-001).

    Args:
        rng: Seeded RNG for deterministic behavior.
        config: Simulation configuration.
    """

    def __init__(self, rng: SimulationRNG, config: SimulationConfig) -> None:
        self._rng = rng
        self._config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> str:
        """Generate a deterministic mock response.

        Args:
            messages: Chat messages (inspected to determine decision type).
            schema: Pydantic model class for response format.

        Returns:
            JSON string conforming to the schema.
        """
        response = self._make_default_response(schema)
        return response.model_dump_json()

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> list[str]:
        """Generate deterministic mock responses for a batch.

        Args:
            batch: List of message sequences.
            schema: Pydantic model class for response format.

        Returns:
            List of JSON strings, one per input.
        """
        return [self.generate(messages, schema) for messages in batch]

    def _make_default_response(self, schema: type[BaseModel]) -> BaseModel:
        """Create a sensible default response based on the schema type.

        Args:
            schema: The Pydantic model class to instantiate.

        Returns:
            A valid instance of the schema with default values.
        """
        if schema is EconomicDecision:
            return EconomicDecision(
                consumption=50.0,
                leisure=0.3,
            )
        if schema is EntrepreneurialDecision:
            # Sometimes create firms (10% chance) for more realistic integration tests
            if self._rng.random() < 0.1:
                return EntrepreneurialDecision(
                    create_firm=True,
                    capital_investment=20.0,
                    labor_demand=5.0,
                    rd_spend=1.0,
                )
            return EntrepreneurialDecision()
        if schema is PoliticalDecision:
            return PoliticalDecision()
        # Fallback: try to instantiate with all defaults
        try:
            return schema()
        except ValidationError:
            log.warning(
                "mock_provider.cannot_create_default",
                schema=schema.__name__,
            )
            raise LLMProviderError(
                f"MockProvider cannot create default for {schema.__name__}"
            ) from None


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

# Maximum retry attempts for transient errors
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0
_BACKOFF_MULTIPLIER = 2.0


class AnthropicProvider:
    """LLM provider using the Anthropic Messages API.

    Features:
    - Structured JSON output via tool_use
    - temperature=0 for determinism (PROP-001)
    - Prompt caching for shared context (REQ-029)
    - Exponential backoff on rate limits and server errors

    Args:
        config: Simulation configuration with LLM parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._model: str = getattr(config, "llm_model", "claude-sonnet-4-5-20250929")
        self._temperature: float = getattr(config, "llm_temperature", 0.0)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client. Deferred import to avoid
        hard dependency when not using Anthropic provider.

        Stores exception classes as instance attributes so _call_with_retry
        can catch them without re-importing.
        """
        try:
            import anthropic  # type: ignore[import-untyped]

            self._client = anthropic.Anthropic()
            self._rate_limit_error: type[Exception] = anthropic.RateLimitError
            self._api_status_error: type[Exception] = anthropic.APIStatusError
            self._api_timeout_error: type[Exception] = anthropic.APITimeoutError
            log.info(
                "anthropic_provider.initialized",
                model=self._model,
                temperature=self._temperature,
            )
        except ImportError as exc:
            log.error("anthropic_provider.import_error")
            raise LLMProviderError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> str:
        """Generate a single response via Anthropic Messages API.

        Args:
            messages: Chat messages with 'role' and 'content' keys.
            schema: Pydantic model class for structured output.

        Returns:
            JSON string conforming to the schema.

        Raises:
            LLMProviderError: On unrecoverable failure after retries.
        """
        tool_definition = self._schema_to_tool(schema)
        return self._call_with_retry(messages, tool_definition, schema)

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> list[str]:
        """Generate responses for a batch of message sequences.

        Calls generate() sequentially for each message sequence.
        Anthropic's API does not natively support batching multiple
        conversations in one request, so we iterate.

        Args:
            batch: List of message sequences.
            schema: Pydantic model class for response validation.

        Returns:
            List of JSON strings.
        """
        results: list[str] = []
        for messages in batch:
            try:
                result = self.generate(messages, schema)
                results.append(result)
            except LLMProviderError as exc:
                log.warning(
                    "anthropic_provider.batch_item_failed",
                    error=str(exc),
                )
                # Return empty JSON for failed items; caller handles fallback
                results.append("{}")
        return results

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        tool_definition: dict[str, Any],
        schema: type[BaseModel],
    ) -> str:
        """Call the API with exponential backoff on transient errors.

        Args:
            messages: Chat messages.
            tool_definition: Tool schema for structured output.
            schema: Pydantic model for validation.

        Returns:
            Validated JSON string.

        Raises:
            LLMProviderError: After exhausting retries.
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    temperature=self._temperature,
                    messages=messages,
                    tools=[tool_definition],
                    tool_choice={"type": "tool", "name": tool_definition["name"]},
                )
                return self._extract_tool_response(response, schema)

            except self._rate_limit_error as exc:
                last_error = exc
                log.warning(
                    "anthropic_provider.rate_limit",
                    attempt=attempt,
                    backoff=backoff,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER

            except self._api_status_error as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    last_error = exc
                    log.warning(
                        "anthropic_provider.server_error",
                        attempt=attempt,
                        status=getattr(exc, "status_code", 0),
                        backoff=backoff,
                    )
                    if attempt < _MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                else:
                    raise LLMProviderError(
                        f"Anthropic API error "
                        f"{getattr(exc, 'status_code', '?')}: "
                        f"{getattr(exc, 'message', str(exc))}"
                    ) from exc

            except self._api_timeout_error as exc:
                last_error = exc
                log.warning(
                    "anthropic_provider.timeout",
                    attempt=attempt,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER

        raise LLMProviderError(f"Anthropic API failed after {_MAX_RETRIES} retries: {last_error}")

    def _extract_tool_response(
        self,
        response: Any,
        schema: type[BaseModel],
    ) -> str:
        """Extract and validate the tool use response.

        Args:
            response: Anthropic API response object.
            schema: Pydantic model for validation.

        Returns:
            Validated JSON string.

        Raises:
            LLMParseError: If response cannot be parsed or validated.
        """
        for block in response.content:
            if block.type == "tool_use":
                tool_input = block.input
                # Validate against schema
                try:
                    validated = schema.model_validate(tool_input)
                    return validated.model_dump_json()
                except ValidationError as exc:
                    raise LLMParseError(f"Response failed schema validation: {exc}") from exc

        raise LLMParseError("No tool_use block found in API response")

    @staticmethod
    def _schema_to_tool(schema: type[BaseModel]) -> dict[str, Any]:
        """Convert a Pydantic schema to an Anthropic tool definition.

        Args:
            schema: Pydantic model class.

        Returns:
            Tool definition dict for the Anthropic API.
        """
        json_schema = schema.model_json_schema()
        return {
            "name": schema.__name__,
            "description": schema.__doc__ or f"Generate a {schema.__name__}",
            "input_schema": json_schema,
        }


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------

_LOCAL_MAX_RETRIES = 3


class OpenAICompatibleProvider:
    """LLM provider for OpenAI-compatible local servers (LM Studio, Ollama, vLLM, llama.cpp).

    Uses httpx to call /chat/completions. Structured output via JSON schema in
    system prompt + response_format=json_object.

    Args:
        config: Simulation configuration with LLM parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._model: str = getattr(config, "llm_model", "lmstudio-community/gpt-oss-20b-GGUF")
        self._temperature: float = getattr(config, "llm_temperature", 0.0)
        self._base_url: str = getattr(config, "llm_base_url", "http://localhost:1234/v1")
        self._client = httpx.Client(base_url=self._base_url, timeout=30.0)

        # Test connectivity (warn but don't crash)
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
            log.info(
                "openai_compatible_provider.initialized",
                model=self._model,
                base_url=self._base_url,
            )
        except Exception as exc:
            log.warning(
                "openai_compatible_provider.connectivity_warning",
                base_url=self._base_url,
                error=str(exc),
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> str:
        """Generate a single response via an OpenAI-compatible /chat/completions endpoint.

        Args:
            messages: Chat messages with 'role' and 'content' keys.
            schema: Pydantic model class for structured output.

        Returns:
            JSON string conforming to the schema.

        Raises:
            LLMProviderError: On unrecoverable failure after retries.
        """
        schema_prompt = self._build_schema_prompt(schema)
        augmented_messages = [
            {"role": "system", "content": schema_prompt},
            *messages,
        ]

        last_error: Exception | None = None
        for attempt in range(1, _LOCAL_MAX_RETRIES + 1):
            try:
                raw_text = self._call_api(augmented_messages, json_mode=True)
                data = json.loads(raw_text)
                validated = schema.model_validate(data)
                return validated.model_dump_json()
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning(
                    "openai_compatible_provider.parse_retry",
                    attempt=attempt,
                    error=str(exc),
                )

        raise LLMParseError(
            f"Failed to parse valid JSON after {_LOCAL_MAX_RETRIES} retries: {last_error}"
        )

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> list[str]:
        """Generate responses for a batch of message sequences.

        Calls generate() sequentially for each message sequence.

        Args:
            batch: List of message sequences.
            schema: Pydantic model class for response validation.

        Returns:
            List of JSON strings, one per input sequence.
        """
        results: list[str] = []
        for messages in batch:
            try:
                result = self.generate(messages, schema)
                results.append(result)
            except LLMProviderError as exc:
                log.warning(
                    "openai_compatible_provider.batch_item_failed",
                    error=str(exc),
                )
                results.append("{}")
        return results

    def _build_schema_prompt(self, schema: type[BaseModel]) -> str:
        """Convert a Pydantic schema to human-readable JSON schema instructions.

        Args:
            schema: Pydantic model class.

        Returns:
            System prompt string instructing the model to output valid JSON.
        """
        json_schema = schema.model_json_schema()
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        lines = [
            "You MUST respond with a single valid JSON object conforming to the following schema.",
            "Do NOT include any text outside the JSON object.",
            "",
            f"Schema: {schema.__name__}",
            f"Description: {schema.__doc__ or 'N/A'}",
            "",
            "Fields:",
        ]
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", field_info.get("anyOf", "unknown"))
            desc = field_info.get("description", "")
            default = field_info.get("default", "REQUIRED" if field_name in required else "N/A")
            lines.append(f"  - {field_name} ({field_type}): {desc} [default: {default}]")

        example_obj = {}
        for field_name, field_info in properties.items():
            if "default" in field_info:
                example_obj[field_name] = field_info["default"]
            elif field_info.get("type") == "number":
                example_obj[field_name] = 0.0
            elif field_info.get("type") == "string":
                example_obj[field_name] = ""
            elif field_info.get("type") == "boolean":
                example_obj[field_name] = False
            else:
                example_obj[field_name] = None

        lines.append("")
        lines.append(f"Example: {json.dumps(example_obj)}")

        return "\n".join(lines)

    def _call_api(self, messages: list[dict[str, str]], json_mode: bool = True) -> str:
        """POST to /chat/completions and return the content text.

        Args:
            messages: Chat messages for the API call.
            json_mode: Whether to request response_format=json_object.

        Returns:
            The text content of the first choice.

        Raises:
            LLMProviderError: On HTTP or parsing errors.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": 1024,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMProviderError(f"OpenAI-compatible API call failed: {exc}") from exc

        try:
            body = resp.json()
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMParseError(f"Unexpected response format: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(
    config: SimulationConfig,
    rng: SimulationRNG,
) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: Simulation configuration.
        rng: Seeded RNG (used by MockProvider for determinism).

    Returns:
        An LLMProvider implementation.

    Raises:
        LLMProviderError: If the provider cannot be created.
    """
    benchmark_mode = getattr(config, "benchmark_mode", False)
    if not config.use_llm or benchmark_mode:
        log.info("provider.mock", reason="use_llm=False or benchmark_mode=True")
        return MockProvider(rng=rng, config=config)

    provider_name: str = getattr(config, "llm_provider", "anthropic")
    provider_name = provider_name.lower()
    if provider_name == "mock":
        log.info("provider.mock", reason="explicit mock provider")
        return MockProvider(rng=rng, config=config)

    if provider_name == "anthropic":
        try:
            return AnthropicProvider(config=config)
        except (LLMProviderError, Exception) as exc:
            log.warning(
                "provider.anthropic_fallback_to_mock",
                error=str(exc),
                reason="Failed to create AnthropicProvider; falling back to MockProvider",
            )
            return MockProvider(rng=rng, config=config)

    if provider_name in ("local", "openai", "openai_compatible", "lmstudio", "ollama", "vllm"):
        try:
            return OpenAICompatibleProvider(config=config)
        except (LLMProviderError, Exception) as exc:
            log.warning(
                "provider.local_fallback_to_mock",
                error=str(exc),
                reason="Failed to create OpenAICompatibleProvider; falling back to MockProvider",
            )
            return MockProvider(rng=rng, config=config)

    raise LLMProviderError(f"Unknown LLM provider: {getattr(config, 'llm_provider', 'unknown')}")


# ---------------------------------------------------------------------------
# Response cache utility
# ---------------------------------------------------------------------------


def compute_cache_key(context: dict[str, Any]) -> str:
    """Compute a deterministic cache key from a context dict.

    Uses JSON serialization and SHA-256 hashing for deterministic,
    collision-resistant keys (REQ-029).

    Args:
        context: The structured context dict to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
