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
from emergent_constitution.models.proposal import ConstitutionalProposal
from emergent_constitution.political_utility import (
    build_bellman_political_context,
    evaluate_proposal,
    log_llm_bellman_deviation,
)
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
    "BUILT-IN RULE TYPES (with dedicated enforcement):\n"
    "- tax_schedule: Tax rates and structure. Parameters: rate (0-1). "
    "enforcement_code variables: income, rate, wealth.\n"
    "- transfer_program: Redistribution method. Parameters: method "
    "('equal_share' or 'means_tested').\n"
    "- public_goods: Fraction of revenue for public goods. Parameters: "
    "fraction_of_revenue (0-1).\n"
    "- market_regulation: Market rules. Parameters vary (e.g., minimum_wage).\n"
    "- firm_regulation: Firm constraints. Parameters vary (e.g., max_workers).\n"
    "- voting_procedure: Voting thresholds. Parameters: threshold (0-1).\n"
    "- property_rights: Property regime rules.\n\n"
    "NOVEL INSTITUTION TYPES (enforced via mechanism_effects):\n"
    "You can also propose entirely NEW types of institutions! Use any rule_type "
    "string (e.g., 'insurance', 'education', 'financial_regulation', 'labor_law', "
    "'environmental', 'social_institution', or invent your own). These are enforced "
    "via structured 'mechanism_effects' — a list of effect declarations.\n\n"
    "MECHANISM EFFECTS FRAMEWORK:\n"
    "Each effect has:\n"
    "- target: 'revenue' (collect money), 'distribution' (pay out), "
    "'productivity' (modify agent productivity), 'wealth_flow' (direct transfer), "
    "'constraint' (bound on decisions), 'utility' (bonus/penalty), "
    "'public_goods' (add to public pool)\n"
    "- scope: 'all', 'workers', 'entrepreneurs', 'wealth_below', "
    "'wealth_above' (with scope_threshold)\n"
    "- magnitude_code: Python expression using variables: income, wealth, "
    "productivity, consumption, leisure, labor_supply, wage, interest_rate, "
    "revenue, num_agents, plus all rule parameters\n"
    "- magnitude_default: fallback value if code fails\n"
    "- direction: 'add' or 'subtract'\n"
    "- priority: ordering (lower = first)\n\n"
    "EXAMPLE 1 — Unemployment Insurance:\n"
    "{\n"
    '  "action": "add",\n'
    '  "rule_name": "unemployment_insurance",\n'
    '  "rule_type": "insurance",\n'
    '  "parameters": {"premium_rate": 0.02, "benefit_rate": 1.5},\n'
    '  "description": "Workers pay 2% premium; low-wealth agents receive '
    '1.5x share of collected premiums",\n'
    '  "mechanism_effects": [\n'
    '    {"target": "revenue", "scope": "workers", '
    '"magnitude_code": "income * premium_rate"},\n'
    '    {"target": "distribution", "scope": "wealth_below", '
    '"scope_threshold": 50, '
    '"magnitude_code": "revenue / num_agents * benefit_rate", '
    '"direction": "add"}\n'
    "  ]\n"
    "}\n\n"
    "EXAMPLE 2 — Education System:\n"
    "{\n"
    '  "action": "add",\n'
    '  "rule_name": "public_education",\n'
    '  "rule_type": "education",\n'
    '  "parameters": {"tax_rate": 0.03, "productivity_boost": 0.05},\n'
    '  "description": "3% education tax funds productivity improvements '
    'for low-productivity workers",\n'
    '  "mechanism_effects": [\n'
    '    {"target": "revenue", "scope": "all", '
    '"magnitude_code": "income * tax_rate"},\n'
    '    {"target": "productivity", "scope": "wealth_below", '
    '"scope_threshold": 100, '
    '"magnitude_code": "productivity_boost", "direction": "add"}\n'
    "  ]\n"
    "}\n\n"
    "Or invent your own category! The simulation will mechanically enforce "
    "whatever effects you specify.\n\n"
    "HOW TO PROPOSE:\n"
    "- action: 'add' (new rule), 'modify' (change existing), or 'remove'\n"
    "- rule_name: Identifier for the rule (snake_case)\n"
    "- rule_type: Any string (built-in or novel)\n"
    "- parameters: Dict of parameter values\n"
    "- description: Natural-language description for voters to read\n"
    "- enforcement_code: Optional Python expression (for built-in types)\n"
    "- mechanism_effects: List of effect declarations (for novel types)\n\n"
    "HOW TO VOTE:\n"
    "- Consider your values: high equality preference -> favor redistribution; "
    "high liberty preference -> favor low taxes and free markets\n"
    "- Consider how the rule affects you personally and society overall\n"
    "- Read the description and mechanism_effects of each proposal carefully\n"
    "- Review societal conditions (Gini, unemployment, output) to identify problems\n"
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
        # Political utility config (REQ-401..405)
        self._political_lambda: float = getattr(config, "political_lambda", 0.05)
        self._pure_bellman_politics: bool = getattr(config, "pure_bellman_politics", False)
        # Value function for Bellman-derived political preferences (set externally)
        self._value_function: list[list[float]] | None = None
        self._a_grid: list[float] | None = None
        self._observed_gini: float | None = None
        # Track LLM-Bellman deviations
        self._bellman_deviation_count = 0
        self._bellman_alignment_count = 0
        log.info(
            "llm_engine.initialized",
            provider_type=type(self._provider).__name__,
            batch_size=self._batch_size,
        )

    # -------------------------------------------------------------------
    # Public API: political state update (REQ-403, REQ-404)
    # -------------------------------------------------------------------

    def set_value_function(
        self,
        value_function: list[list[float]] | None,
        a_grid: list[float] | None,
    ) -> None:
        """Set the value function for Bellman-derived political preferences.

        Called by LeadV2 after the EGM solver runs, providing V(a,z) for
        use in proposal evaluation (REQ-403).

        Args:
            value_function: V(a, z) array, shape [n_a][n_z], or None.
            a_grid: Asset grid points, or None.
        """
        self._value_function = value_function
        self._a_grid = a_grid

    def set_observed_gini(self, gini: float | None) -> None:
        """Set the current observed Gini coefficient.

        Args:
            gini: Observed Gini from the Observer, or None.
        """
        self._observed_gini = gini

    def _build_counterfactual_constitution(
        self,
        agent: HouseholdState,
        constitution: ConstitutionV2,
    ) -> ConstitutionV2:
        """Build a nearby constitutional counterfactual for Bellman context.

        Political contexts are proposal-agnostic at this stage, so we expose
        Bellman preferences using a local tax-rule perturbation aligned with
        the agent's equality/liberty weights.
        """
        proposed = constitution.model_copy(deep=True)
        tax_rules = proposed.get_tax_rules()
        if not tax_rules:
            return proposed

        rule = tax_rules[0]
        current_rate = float(rule.parameters.get("rate", 0.0))
        preferred_rate = max(0.0, min(1.0, agent.value_vector.equality * 0.5))

        if abs(preferred_rate - current_rate) < 1e-8:
            direction = 1.0 if agent.value_vector.equality >= agent.value_vector.liberty else -1.0
            preferred_rate = max(0.0, min(1.0, current_rate + 0.01 * direction))

        params = dict(rule.parameters)
        params["rate"] = preferred_rate
        proposed.rules[rule.name] = rule.model_copy(update={"parameters": params})
        return proposed

    @property
    def bellman_deviation_count(self) -> int:
        """Number of times LLM overrode Bellman recommendation."""
        return self._bellman_deviation_count

    @property
    def bellman_alignment_count(self) -> int:
        """Number of times LLM aligned with Bellman recommendation."""
        return self._bellman_alignment_count

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

        When pure_bellman_politics is enabled (REQ-405), bypasses the LLM
        entirely and uses Bellman value function comparison for all votes.

        When LLM is used, Bellman-derived preferences are included in the
        context (REQ-404), and deviations are logged.

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
        bellman_contexts: dict[str, dict[str, object]] = {}
        for h in households:
            ctx = self._build_political_context(h, market, constitution, history, recent_outcomes)
            # Add voting-specific context
            ctx["active_voting_rule"] = constitution.voting_rule
            rule = constitution.get_active_voting_rule()
            if rule:
                ctx["voting_threshold"] = rule.parameters.get("threshold", 0.5)

            # Add Bellman-derived political context (REQ-404)
            if self._value_function is not None and self._a_grid is not None:
                counterfactual = self._build_counterfactual_constitution(h, constitution)
                bellman_ctx = build_bellman_political_context(
                    value_function=self._value_function,
                    a_grid=self._a_grid,
                    agent=h,
                    proposed_constitution=counterfactual,
                    current_constitution=constitution,
                    political_lambda=self._political_lambda,
                    observed_gini=self._observed_gini,
                )
                ctx["bellman_political_preference"] = bellman_ctx
                bellman_contexts[h.id] = bellman_ctx

            contexts[h.id] = ctx

        # Pure Bellman politics mode (REQ-405): bypass LLM entirely
        if self._pure_bellman_politics:
            decisions: dict[str, PoliticalDecision] = {}
            tax_rules = constitution.get_tax_rules()
            tax_rule_name = tax_rules[0].name if tax_rules else None
            proposal_count = 0

            for h in households:
                counterfactual = self._build_counterfactual_constitution(h, constitution)
                delta_v = evaluate_proposal(
                    value_function=self._value_function or [],
                    a_grid=self._a_grid or [],
                    agent=h,
                    proposed_constitution=counterfactual,
                    current_constitution=constitution,
                    political_lambda=self._political_lambda,
                    observed_gini=self._observed_gini,
                )
                support = delta_v > 0.0

                votes: dict[str, bool] = {}
                if tax_rule_name is not None:
                    votes[tax_rule_name] = support

                proposal: ConstitutionalProposal | None = None
                if tax_rule_name is not None and abs(delta_v) > 1e-8 and self._rng.random() < 0.15:
                    proposed_rule = counterfactual.rules.get(tax_rule_name)
                    if proposed_rule is not None:
                        proposal = ConstitutionalProposal(
                            proposer_id=h.id,
                            action="modify",
                            rule_name=tax_rule_name,
                            parameters=dict(proposed_rule.parameters),
                            description="Bellman-derived tax adjustment proposal",
                        )
                        proposal_count += 1

                decisions[h.id] = PoliticalDecision(proposal=proposal, votes=votes)
            log.info(
                "llm_engine.pure_bellman_politics",
                n_agents=len(households),
                n_proposals=proposal_count,
            )
            return decisions

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

        decisions = {}
        for agent_id in [h.id for h in households]:
            raw = results.get(agent_id)
            if raw is not None:
                try:
                    decision = PoliticalDecision.model_validate_json(raw)
                    decisions[agent_id] = decision

                    # Log LLM-Bellman deviation for votes (REQ-404)
                    bellman_ctx = bellman_contexts.get(agent_id)
                    if bellman_ctx and decision.votes:
                        recommendation = str(bellman_ctx.get("bellman_recommendation", ""))
                        delta_v = float(bellman_ctx.get("value_delta", 0.0))
                        for _proposal_name, llm_vote in decision.votes.items():
                            log_llm_bellman_deviation(
                                agent_id=agent_id,
                                bellman_recommendation=recommendation,
                                llm_vote=llm_vote,
                                delta_v=delta_v,
                            )
                            if (recommendation == "support") != llm_vote:
                                self._bellman_deviation_count += 1
                            else:
                                self._bellman_alignment_count += 1

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

        # (c) Current constitution (serialized, including mechanism_effects)
        rules_ctx: dict[str, dict[str, Any]] = {}
        for name, rule in constitution.rules.items():
            rule_info: dict[str, Any] = {
                "type": str(rule.rule_type),
                "parameters": rule.parameters,
                "description": rule.description,
            }
            if rule.mechanism_effects:
                rule_info["mechanism_effects"] = [
                    {
                        "target": str(e.target),
                        "scope": str(e.scope),
                        "magnitude_code": e.magnitude_code,
                        "direction": e.direction,
                    }
                    for e in rule.mechanism_effects
                ]
            rules_ctx[name] = rule_info
        constitution_ctx = {
            "voting_rule": constitution.voting_rule,
            "rules": rules_ctx,
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
        rule type guide, institutional suggestions, mechanism effects
        summaries, and recent proposal outcomes.

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
        societal = self._extract_societal_conditions(market, history)
        ctx["societal_conditions"] = societal

        # Recent history summary
        ctx["recent_history"] = self._build_history_summary(history)

        # Rule type guide
        ctx["rule_type_guide"] = self._build_rule_type_guide()

        # Institutional suggestions based on societal conditions
        ctx["institutional_suggestions"] = self._generate_institutional_suggestions(societal)

        # Recent proposal outcomes
        ctx["recent_proposal_outcomes"] = recent_outcomes or []

        # Mechanism effects summary for active rules
        ctx["active_mechanism_effects"] = self._summarize_mechanism_effects(constitution)

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

        Includes both built-in types with dedicated enforcement and
        novel types enforced via the mechanism_effects framework.

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
                "description": "Public goods allocation. Parameters: fraction_of_revenue (0-1).",
                "example": {
                    "action": "modify",
                    "rule_name": "public_goods_provision",
                    "parameters": {"fraction_of_revenue": 0.3},
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
            # Novel institution types
            "insurance": {
                "description": (
                    "Insurance programs (health, unemployment, disaster). "
                    "Use mechanism_effects: revenue (premiums) + distribution (payouts)."
                ),
                "example": {
                    "action": "add",
                    "rule_name": "unemployment_insurance",
                    "rule_type": "insurance",
                    "parameters": {"premium_rate": 0.02, "benefit_rate": 1.5},
                    "mechanism_effects": [
                        {
                            "target": "revenue",
                            "scope": "workers",
                            "magnitude_code": "income * premium_rate",
                        },
                        {
                            "target": "distribution",
                            "scope": "wealth_below",
                            "scope_threshold": 50,
                            "magnitude_code": "revenue / num_agents * benefit_rate",
                            "direction": "add",
                        },
                    ],
                },
            },
            "education": {
                "description": (
                    "Education and training programs. "
                    "Use mechanism_effects: revenue (funding) + productivity (boost)."
                ),
                "example": {
                    "action": "add",
                    "rule_name": "public_education",
                    "rule_type": "education",
                    "parameters": {"tax_rate": 0.03, "boost": 0.05},
                    "mechanism_effects": [
                        {
                            "target": "revenue",
                            "scope": "all",
                            "magnitude_code": "income * tax_rate",
                        },
                        {
                            "target": "productivity",
                            "scope": "wealth_below",
                            "scope_threshold": 100,
                            "magnitude_code": "boost",
                            "direction": "add",
                        },
                    ],
                },
            },
        }

    @staticmethod
    def _generate_institutional_suggestions(
        societal_conditions: dict[str, Any],
    ) -> list[str]:
        """Generate targeted institutional suggestions based on conditions.

        Examines societal metrics and suggests novel institutions that
        might address observed problems.

        Args:
            societal_conditions: Dict with gini, unemployment, output, etc.

        Returns:
            List of suggestion strings for the political context.
        """
        suggestions: list[str] = []

        gini = societal_conditions.get("gini_coefficient")
        unemployment = societal_conditions.get("unemployment_rate")
        welfare = societal_conditions.get("social_welfare")
        output = societal_conditions.get("aggregate_output")

        if gini is not None and gini > 0.4:
            suggestions.append(
                f"High inequality (Gini={gini:.3f}). Consider: progressive wealth tax, "
                "means-tested transfers, unemployment insurance, or an education "
                "system to boost low-income productivity."
            )
        if gini is not None and gini < 0.15:
            suggestions.append(
                f"Very low inequality (Gini={gini:.3f}). Consider: reducing redistribution "
                "or adding incentive-based programs to boost output."
            )
        if unemployment is not None and unemployment > 0.15:
            suggestions.append(
                f"High unemployment ({unemployment:.1%}). Consider: job training programs "
                "(education type), wage subsidies (labor_law type), or public "
                "works (public_goods)."
            )
        if output is not None and output < 10.0:
            suggestions.append(
                f"Low aggregate output ({output:.1f}). Consider: R&D incentives "
                "(firm_regulation type), productivity-boosting education, "
                "or entrepreneurship support programs."
            )
        if welfare is not None and welfare < 0.1:
            suggestions.append(
                f"Low social welfare ({welfare:.3f}). Consider: broad-based insurance, "
                "public goods expansion, or progressive transfers."
            )

        if not suggestions:
            suggestions.append(
                "Society is relatively stable. Consider: institutional innovation "
                "like environmental protection, financial regulation, or social "
                "institutions that could improve long-term outcomes."
            )

        return suggestions

    @staticmethod
    def _summarize_mechanism_effects(
        constitution: ConstitutionV2,
    ) -> list[dict[str, Any]]:
        """Summarize active mechanism effects for agent context.

        Args:
            constitution: Active constitutional rules.

        Returns:
            List of dicts summarizing rules with mechanism effects.
        """
        summaries: list[dict[str, Any]] = []
        for name, rule in constitution.rules.items():
            if not rule.mechanism_effects:
                continue
            effects_summary = []
            for e in rule.mechanism_effects:
                effects_summary.append(
                    {
                        "target": str(e.target),
                        "scope": str(e.scope),
                        "direction": e.direction,
                    }
                )
            summaries.append(
                {
                    "rule_name": name,
                    "rule_type": str(rule.rule_type),
                    "description": rule.description,
                    "num_effects": len(rule.mechanism_effects),
                    "effects": effects_summary,
                }
            )
        return summaries

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
