"""LLM decision engine — batching, caching, context building, fallback.

Manages LLM-based agent decision-making for all three decision types:
economic, entrepreneurial, and political. Provides structured context
per REQ-025, batched API calls per REQ-029, and fallback to defaults
on failure per REQ-027.

Traceability: REQ-003, REQ-006, REQ-019, REQ-020, REQ-024, REQ-025,
              REQ-026, REQ-027, REQ-029
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import structlog
from pydantic import BaseModel, ValidationError

from emergent_constitution.config import SimulationConfig
from emergent_constitution.llm_providers import (
    LLMProvider,
    LLMProviderError,
    compute_cache_key,
    create_provider,
)
from emergent_constitution.models.constitution import ConstitutionV2
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState
from emergent_constitution.rng import SimulationRNG

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Fallback solver protocol
# ---------------------------------------------------------------------------


class FallbackSolver(Protocol):
    """Protocol for numerical fallback when LLM calls fail (REQ-027).

    Implementations compute approximately optimal decisions using the
    agent's utility function and budget constraint.
    """

    def solve_economic(self, agent: HouseholdState, market: MarketState) -> EconomicDecision: ...

    def solve_entrepreneurial(
        self, agent: HouseholdState, firms: list[FirmState], market: MarketState
    ) -> EntrepreneurialDecision: ...


class DefaultFallbackSolver:
    """Simple fallback that returns safe default decisions.

    Used when no NumericalSolver is available (Task 7 not yet implemented).
    Returns conservative defaults: moderate consumption, moderate leisure,
    no entrepreneurial action.
    """

    def solve_economic(self, agent: HouseholdState, market: MarketState) -> EconomicDecision:
        """Return a conservative economic decision based on agent state."""
        # Consume a fraction of wealth, moderate leisure
        max_consumption = max(agent.wealth * 0.3, 0.0)
        return EconomicDecision(consumption=max_consumption, leisure=0.3)

    def solve_entrepreneurial(
        self, agent: HouseholdState, firms: list[FirmState], market: MarketState
    ) -> EntrepreneurialDecision:
        """Return a no-op entrepreneurial decision."""
        return EntrepreneurialDecision()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_ECONOMIC_SYSTEM_PROMPT = (
    "You are a household agent in an economic simulation. "
    "Given your personal state and economic conditions, decide how much to "
    "consume and how much leisure to take this period. "
    "Your consumption must be non-negative and your leisure must be "
    "between 0 (full work) and 1 (no work). "
    "Consider your wealth, income potential, and the current market conditions."
)

_ENTREPRENEURIAL_SYSTEM_PROMPT = (
    "You are an agent deciding on entrepreneurial actions. "
    "You may create a firm, invest capital, hire labor, spend on R&D, "
    "or close a firm. Consider market conditions, your wealth, "
    "and potential returns when making your decision."
)

_POLITICAL_SYSTEM_PROMPT = (
    "You are a citizen-agent in a constitutional democracy simulation. "
    "Your goal is to propose and vote on rules that improve societal welfare "
    "according to your values (equality vs liberty preference).\n\n"
    "AVAILABLE RULE TYPES:\n"
    "- tax_schedule: Tax rates and structure. Parameters: rate (0-1). "
    "enforcement_code variables: income, rate, wealth.\n"
    "- transfer_program: Redistribution method. Parameters: method "
    "('equal_share' or 'means_tested').\n"
    "- public_goods: Fraction of revenue for public goods. Parameters: "
    "fraction (0-1).\n"
    "- market_regulation: Market rules. Parameters vary (e.g., minimum_wage).\n"
    "- firm_regulation: Firm constraints. Parameters vary (e.g., max_firm_size).\n"
    "- voting_procedure: Voting thresholds. Parameters: threshold (0-1).\n"
    "- property_rights: Property regime rules.\n"
    "- custom: Any rule with custom enforcement_code. Sandbox functions: "
    "abs, min, max, sqrt, log, exp. Variables: income, rate, wealth.\n\n"
    "HOW TO PROPOSE:\n"
    "- action: 'add' (new rule), 'modify' (change existing), or 'remove'\n"
    "- rule_name: Identifier for the rule\n"
    "- rule_type: One of the types above\n"
    "- parameters: Dict of parameter values\n"
    "- enforcement_code: Optional Python expression for custom logic\n\n"
    "HOW TO VOTE:\n"
    "- Consider your values: high equality preference -> favor redistribution; "
    "high liberty preference -> favor low taxes and free markets\n"
    "- Consider how the rule affects you personally and society overall\n"
    "- Review the societal conditions (Gini, unemployment, output) to identify problems\n"
    "- Review recent proposal outcomes to avoid repeating rejected approaches\n\n"
    "Respond with a PoliticalDecision containing an optional proposal and votes "
    "on existing rules."
)


# ---------------------------------------------------------------------------
# LLMDecisionEngine
# ---------------------------------------------------------------------------


class LLMDecisionEngine:
    """Manages LLM-based agent decision-making with batching and caching.

    Orchestrates the full decision pipeline:
    1. Build structured context per agent (REQ-025)
    2. Check response cache for identical contexts (REQ-029)
    3. Batch uncached prompts to the LLM provider (REQ-029)
    4. Parse and validate responses against schemas (REQ-026)
    5. Fall back to solver on failure (REQ-027)

    Args:
        config: Simulation configuration.
        rng: Seeded RNG for deterministic behavior.
        provider: Optional LLM provider (created from config if not given).
        fallback_solver: Optional fallback solver for failed LLM calls.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: SimulationRNG,
        provider: LLMProvider | None = None,
        fallback_solver: FallbackSolver | None = None,
    ) -> None:
        self._config = config
        self._rng = rng
        self._provider = provider or create_provider(config, rng)
        self._fallback = fallback_solver or DefaultFallbackSolver()
        self._cache: dict[str, str] = {}
        self._batch_size: int = getattr(config, "llm_batch_size", 10)
        self._cache_hits = 0
        self._cache_misses = 0
        log.info(
            "llm_engine.initialized",
            provider_type=type(self._provider).__name__,
            batch_size=self._batch_size,
        )

    # -------------------------------------------------------------------
    # Public API: collect decisions
    # -------------------------------------------------------------------

    def collect_economic_decisions(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
        history_summary: str = "",
    ) -> dict[str, EconomicDecision]:
        """Collect economic decisions for all households via LLM.

        Each agent receives structured context and returns a consumption/
        leisure decision. Falls back to solver on failure (REQ-027).

        Args:
            households: All household agents.
            market: Current market equilibrium.
            constitution: Active constitutional rules.
            history_summary: Recent history summary text.

        Returns:
            Mapping from agent ID to EconomicDecision.
        """
        contexts = {
            h.id: self._build_context(h, market, constitution, history_summary) for h in households
        }

        messages_map = {
            agent_id: self._context_to_messages(ctx, _ECONOMIC_SYSTEM_PROMPT)
            for agent_id, ctx in contexts.items()
        }

        results = self._collect_with_cache(
            agent_ids=[h.id for h in households],
            contexts=contexts,
            messages_map=messages_map,
            schema=EconomicDecision,
        )

        # Apply fallback for missing/failed decisions
        decisions: dict[str, EconomicDecision] = {}
        household_lookup = {h.id: h for h in households}
        for agent_id in [h.id for h in households]:
            raw = results.get(agent_id)
            if raw is not None:
                try:
                    decisions[agent_id] = EconomicDecision.model_validate_json(raw)
                    continue
                except ValidationError:
                    log.warning(
                        "llm_engine.validation_failed",
                        agent_id=agent_id,
                        decision_type="economic",
                    )
            # Fallback
            decisions[agent_id] = self._fallback.solve_economic(household_lookup[agent_id], market)
            log.debug(
                "llm_engine.fallback_used",
                agent_id=agent_id,
                decision_type="economic",
            )

        return decisions

    def collect_entrepreneurial_decisions(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> dict[str, EntrepreneurialDecision]:
        """Collect entrepreneurial decisions for eligible households.

        Only agents with the ENTREPRENEUR role or sufficient wealth to
        create a firm are queried. Others get a no-op default.

        Args:
            households: All household agents.
            firms: Currently active firms.
            market: Current market equilibrium.
            constitution: Active constitutional rules.

        Returns:
            Mapping from agent ID to EntrepreneurialDecision.
        """
        # Build context with firm info
        contexts: dict[str, dict[str, Any]] = {}
        for h in households:
            ctx = self._build_context(h, market, constitution, "")
            # Add firm-specific info for entrepreneurs
            owned_firms = [f for f in firms if f.owner_id == h.id]
            ctx["owned_firms"] = [
                {
                    "id": f.id,
                    "capital": f.capital,
                    "labor_demand": f.labor_demand,
                    "tfp": f.tfp,
                    "output": f.output,
                    "profit": f.profit,
                    "num_workers": len(f.worker_ids),
                }
                for f in owned_firms
            ]
            ctx["total_firms"] = len(firms)
            contexts[h.id] = ctx

        messages_map = {
            agent_id: self._context_to_messages(ctx, _ENTREPRENEURIAL_SYSTEM_PROMPT)
            for agent_id, ctx in contexts.items()
        }

        results = self._collect_with_cache(
            agent_ids=[h.id for h in households],
            contexts=contexts,
            messages_map=messages_map,
            schema=EntrepreneurialDecision,
        )

        decisions: dict[str, EntrepreneurialDecision] = {}
        household_lookup = {h.id: h for h in households}
        for agent_id in [h.id for h in households]:
            raw = results.get(agent_id)
            if raw is not None:
                try:
                    decisions[agent_id] = EntrepreneurialDecision.model_validate_json(raw)
                    continue
                except ValidationError:
                    log.warning(
                        "llm_engine.validation_failed",
                        agent_id=agent_id,
                        decision_type="entrepreneurial",
                    )
            # Fallback
            decisions[agent_id] = self._fallback.solve_entrepreneurial(
                household_lookup[agent_id], firms, market
            )

        return decisions

    def collect_political_decisions(
        self,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
        market: MarketState,
        history: list | None = None,
        recent_outcomes: list[dict] | None = None,
    ) -> dict[str, PoliticalDecision]:
        """Collect political decisions (proposals and votes) from all agents.

        Args:
            households: All household agents.
            constitution: Active constitutional rules.
            market: Current market equilibrium.
            history: Recent HistoryEntryV2 entries (last 3-5).
            recent_outcomes: Recent proposal outcomes (votes + rejections).

        Returns:
            Mapping from agent ID to PoliticalDecision.
        """
        contexts: dict[str, dict[str, Any]] = {}
        for h in households:
            ctx = self._build_political_context(h, market, constitution, history, recent_outcomes)
            # Add voting-specific context
            ctx["active_voting_rule"] = constitution.voting_rule
            rule = constitution.get_active_voting_rule()
            if rule:
                ctx["voting_threshold"] = rule.parameters.get("threshold", 0.5)
            contexts[h.id] = ctx

        messages_map = {
            agent_id: self._context_to_messages(ctx, _POLITICAL_SYSTEM_PROMPT)
            for agent_id, ctx in contexts.items()
        }

        results = self._collect_with_cache(
            agent_ids=[h.id for h in households],
            contexts=contexts,
            messages_map=messages_map,
            schema=PoliticalDecision,
        )

        decisions: dict[str, PoliticalDecision] = {}
        for agent_id in [h.id for h in households]:
            raw = results.get(agent_id)
            if raw is not None:
                try:
                    decisions[agent_id] = PoliticalDecision.model_validate_json(raw)
                    continue
                except ValidationError:
                    log.warning(
                        "llm_engine.validation_failed",
                        agent_id=agent_id,
                        decision_type="political",
                    )
            # Fallback: no proposal, no votes
            decisions[agent_id] = PoliticalDecision()

        return decisions

    # -------------------------------------------------------------------
    # Cache stats
    # -------------------------------------------------------------------

    @property
    def cache_hits(self) -> int:
        """Number of cache hits since initialization."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Number of cache misses since initialization."""
        return self._cache_misses

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # -------------------------------------------------------------------
    # Internal: context building (REQ-025)
    # -------------------------------------------------------------------

    def _build_context(
        self,
        agent: HouseholdState,
        market: MarketState,
        constitution: ConstitutionV2,
        history_summary: str,
    ) -> dict[str, Any]:
        """Build structured context for an agent's LLM prompt.

        Per REQ-025, includes five components:
        (a) personal state, (b) economic conditions,
        (c) current constitution, (d) recent history,
        (e) messages from other agents (placeholder for future).

        Args:
            agent: The household agent state.
            market: Current market equilibrium.
            constitution: Active constitutional rules.
            history_summary: Recent history summary text.

        Returns:
            Structured context dict.
        """
        # (a) Personal state
        personal = {
            "id": agent.id,
            "wealth": agent.wealth,
            "productivity": agent.productivity,
            "role": agent.role.value,
            "firm_id": agent.firm_id,
            "coalition_id": agent.coalition_id,
            "utility_params": {
                "alpha": agent.utility_params.alpha,
                "beta": agent.utility_params.beta,
                "gamma": agent.utility_params.gamma,
                "beta_discount": agent.utility_params.beta_discount,
            },
            "value_vector": {
                "equality": agent.value_vector.equality,
                "liberty": agent.value_vector.liberty,
            },
        }

        # (b) Economic conditions
        economic = {
            "wage": market.wage,
            "interest_rate": market.interest_rate,
            "aggregate_output": market.aggregate_output,
            "aggregate_consumption": market.aggregate_consumption,
            "government_spending": market.government_spending,
        }

        # (c) Current constitution (serialized)
        constitution_ctx = {
            "voting_rule": constitution.voting_rule,
            "rules": {
                name: {
                    "type": rule.rule_type.value,
                    "parameters": rule.parameters,
                    "description": rule.description,
                }
                for name, rule in constitution.rules.items()
            },
        }

        # (d) Recent history summary
        history_ctx = history_summary if history_summary else "No history available."

        # (e) Messages (placeholder for future agent communication)
        messages_ctx: list[str] = []

        return {
            "personal_state": personal,
            "economic_conditions": economic,
            "constitution": constitution_ctx,
            "history_summary": history_ctx,
            "messages": messages_ctx,
        }

    # -------------------------------------------------------------------
    # Internal: political context building
    # -------------------------------------------------------------------

    def _build_political_context(
        self,
        agent: HouseholdState,
        market: MarketState,
        constitution: ConstitutionV2,
        history: list | None = None,
        recent_outcomes: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Build enriched context for political decisions.

        Extends _build_context with societal conditions, history summary,
        rule type guide, and recent proposal outcomes.

        Args:
            agent: The household agent state.
            market: Current market equilibrium.
            constitution: Active constitutional rules.
            history: Recent HistoryEntryV2 entries (last 3-5).
            recent_outcomes: Recent proposal outcomes (votes + rejections).

        Returns:
            Enriched context dict for political LLM prompts.
        """
        ctx = self._build_context(agent, market, constitution, "")

        # Societal conditions from most recent history entry
        ctx["societal_conditions"] = self._extract_societal_conditions(market, history)

        # Recent history summary
        ctx["recent_history"] = self._build_history_summary(history)

        # Rule type guide
        ctx["rule_type_guide"] = self._build_rule_type_guide()

        # Recent proposal outcomes
        ctx["recent_proposal_outcomes"] = recent_outcomes or []

        return ctx

    def _build_history_summary(
        self,
        history: list | None,
    ) -> list[dict[str, Any]]:
        """Build structured summary of recent history entries.

        Args:
            history: Recent HistoryEntryV2 entries.

        Returns:
            List of summary dicts with period, gini, mean_wealth, social_welfare,
            rule_changes, and trend description.
        """
        if not history:
            return []

        summaries: list[dict[str, Any]] = []
        for i, entry in enumerate(history):
            summary: dict[str, Any] = {
                "period": entry.period,
                "gini": round(entry.gini, 4),
                "mean_wealth": round(entry.mean_wealth, 2),
                "social_welfare": round(entry.social_welfare, 2),
                "unemployment_rate": round(entry.unemployment_rate, 4),
                "num_active_firms": entry.num_active_firms,
                "rule_changes": entry.rule_changes,
            }

            # Compute trend relative to previous entry
            if i > 0:
                prev = history[i - 1]
                trends: list[str] = []
                if entry.gini < prev.gini - 0.01:
                    trends.append("Inequality decreased")
                elif entry.gini > prev.gini + 0.01:
                    trends.append("Inequality increased")
                if entry.social_welfare > prev.social_welfare:
                    trends.append("Welfare improved")
                elif entry.social_welfare < prev.social_welfare:
                    trends.append("Welfare declined")
                if entry.mean_wealth > prev.mean_wealth:
                    trends.append("Mean wealth grew")
                summary["trend"] = "; ".join(trends) if trends else "Stable"
            else:
                summary["trend"] = "First observation"

            summaries.append(summary)

        return summaries

    def _extract_societal_conditions(
        self,
        market: MarketState,
        history: list | None,
    ) -> dict[str, Any]:
        """Extract societal condition metrics for political context.

        Sources from most recent history entry if available, otherwise from market.

        Args:
            market: Current market equilibrium.
            history: Recent HistoryEntryV2 entries.

        Returns:
            Dict with Gini, wealth stats, unemployment, firms, output, welfare.
        """
        if history:
            latest = history[-1]
            return {
                "gini_coefficient": round(latest.gini, 4),
                "mean_wealth": round(latest.mean_wealth, 2),
                "median_wealth": round(latest.median_wealth, 2),
                "wealth_quantiles": latest.wealth_quantiles,
                "unemployment_rate": round(latest.unemployment_rate, 4),
                "num_active_firms": latest.num_active_firms,
                "aggregate_output": round(latest.aggregate_output, 2),
                "social_welfare": round(latest.social_welfare, 2),
                "wage": round(latest.wage, 4),
                "interest_rate": round(latest.interest_rate, 4),
            }
        # Fallback: use market data only
        return {
            "gini_coefficient": None,
            "mean_wealth": None,
            "median_wealth": None,
            "wealth_quantiles": [],
            "unemployment_rate": None,
            "num_active_firms": 0,
            "aggregate_output": round(market.aggregate_output, 2),
            "social_welfare": None,
            "wage": round(market.wage, 4),
            "interest_rate": round(market.interest_rate, 4),
        }

    @staticmethod
    def _build_rule_type_guide() -> dict[str, dict[str, Any]]:
        """Build a guide describing available rule types with examples.

        Returns:
            Dict mapping rule type names to descriptions and examples.
        """
        return {
            "tax_schedule": {
                "description": (
                    "Tax rules. Parameters: rate (0-1). "
                    "enforcement_code vars: income, rate, wealth."
                ),
                "example": {
                    "action": "modify",
                    "rule_name": "flat_tax",
                    "parameters": {"rate": 0.15},
                },
            },
            "transfer_program": {
                "description": (
                    "Redistribution. Parameters: method ('equal_share' or 'means_tested')."
                ),
                "example": {
                    "action": "add",
                    "rule_name": "safety_net",
                    "rule_type": "transfer_program",
                    "parameters": {"method": "means_tested"},
                },
            },
            "public_goods": {
                "description": "Public goods allocation. Parameters: fraction (0-1).",
                "example": {
                    "action": "modify",
                    "rule_name": "public_goods",
                    "parameters": {"fraction": 0.3},
                },
            },
            "market_regulation": {
                "description": "Market rules like minimum wage or price controls.",
                "example": {
                    "action": "add",
                    "rule_name": "min_wage",
                    "rule_type": "market_regulation",
                    "parameters": {"minimum_wage": 5.0},
                },
            },
            "firm_regulation": {
                "description": "Firm constraints like max size or R&D requirements.",
                "example": {
                    "action": "add",
                    "rule_name": "rd_mandate",
                    "rule_type": "firm_regulation",
                    "parameters": {"min_rd_fraction": 0.05},
                },
            },
            "voting_procedure": {
                "description": "Voting thresholds and procedures.",
                "example": {
                    "action": "modify",
                    "rule_name": "majority_vote",
                    "parameters": {"threshold": 0.6},
                },
            },
            "property_rights": {
                "description": "Property regime and ownership rules.",
                "example": {
                    "action": "modify",
                    "rule_name": "private_property",
                    "parameters": {"regime": "private"},
                },
            },
            "custom": {
                "description": (
                    "Any rule with custom enforcement_code. "
                    "Sandbox functions: abs, min, max, sqrt, log, exp."
                ),
                "example": {
                    "action": "add",
                    "rule_name": "wealth_cap",
                    "rule_type": "custom",
                    "parameters": {"max_wealth": 500},
                    "enforcement_code": "min(wealth, max_wealth)",
                },
            },
        }

    # -------------------------------------------------------------------
    # Internal: prompt construction
    # -------------------------------------------------------------------

    def _context_to_messages(
        self, context: dict[str, Any], system_prompt: str
    ) -> list[dict[str, str]]:
        """Convert structured context into chat messages for the LLM.

        Args:
            context: Structured context dict from _build_context.
            system_prompt: System prompt for the decision type.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        context_text = json.dumps(context, indent=2, default=str)
        return [
            {"role": "user", "content": f"{system_prompt}\n\nContext:\n{context_text}"},
        ]

    # -------------------------------------------------------------------
    # Internal: caching and batching (REQ-029)
    # -------------------------------------------------------------------

    def _collect_with_cache(
        self,
        agent_ids: list[str],
        contexts: dict[str, dict[str, Any]],
        messages_map: dict[str, list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> dict[str, str | None]:
        """Collect LLM responses with caching and batching.

        1. Check cache for each agent's context
        2. Batch uncached prompts
        3. Store results in cache

        Args:
            agent_ids: Ordered list of agent IDs.
            contexts: Context dicts keyed by agent ID.
            messages_map: Message sequences keyed by agent ID.
            schema: Pydantic model for response validation.

        Returns:
            Mapping from agent ID to raw JSON string, or None on failure.
        """
        results: dict[str, str | None] = {}
        uncached_ids: list[str] = []

        # Step 1: Check cache
        for agent_id in agent_ids:
            cache_key = compute_cache_key(contexts[agent_id])
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[agent_id] = cached
                self._cache_hits += 1
            else:
                uncached_ids.append(agent_id)
                self._cache_misses += 1

        if not uncached_ids:
            return results

        # Step 2: Batch uncached prompts
        batch_results = self._batch_call(
            agent_ids=uncached_ids,
            messages_map=messages_map,
            schema=schema,
        )

        # Step 3: Store in cache and collect results
        for agent_id, raw_response in zip(uncached_ids, batch_results, strict=False):
            if raw_response and raw_response != "{}":
                cache_key = compute_cache_key(contexts[agent_id])
                self._cache[cache_key] = raw_response
                results[agent_id] = raw_response
            else:
                results[agent_id] = None

        return results

    def _batch_call(
        self,
        agent_ids: list[str],
        messages_map: dict[str, list[dict[str, str]]],
        schema: type[BaseModel],
    ) -> list[str]:
        """Send batched LLM requests with retry and fallback.

        Splits agent_ids into batches of configured size, calls the
        provider's generate_batch for each chunk, and collects results.

        Args:
            agent_ids: Agent IDs to process.
            messages_map: Message sequences keyed by agent ID.
            schema: Pydantic model for response validation.

        Returns:
            List of raw JSON strings (one per agent_id), empty string on failure.
        """
        all_results: list[str] = []

        for batch_start in range(0, len(agent_ids), self._batch_size):
            batch_ids = agent_ids[batch_start : batch_start + self._batch_size]
            batch_messages = [messages_map[aid] for aid in batch_ids]

            try:
                batch_responses = self._provider.generate_batch(batch_messages, schema)
                all_results.extend(batch_responses)
            except LLMProviderError as exc:
                log.warning(
                    "llm_engine.batch_failed",
                    batch_start=batch_start,
                    batch_size=len(batch_ids),
                    error=str(exc),
                )
                # Return empty strings for this entire batch
                all_results.extend([""] * len(batch_ids))

        return all_results
