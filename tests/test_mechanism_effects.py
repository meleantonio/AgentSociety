"""Tests for the Mechanism Effects framework.

Covers:
- EffectTarget and EffectScope enums
- MechanismEffect and RuleImpact models
- ConstitutionalRule rule_type normalization and mechanism_effects field
- KNOWN_RULE_TYPES constant
- _normalize_rule_type function
- ConstitutionalProposal rule_type normalization and mechanism_effects
- ConstitutionEngine: _agents_in_scope, _evaluate_magnitude,
  apply_productivity_effects, compute_agent_constraints,
  enforce_mechanism_effects, validate_rule, apply_proposal
- Expanded sandbox builtins (float, int, bool, ceil, floor, log10)
- Integration: full lifecycle and backward compatibility
"""

from __future__ import annotations

import math

import pytest

from emergent_constitution.constitution_engine import (
    ConstitutionEngine,
    SandboxError,
    evaluate_enforcement_code,
)
from emergent_constitution.models.constitution import (
    KNOWN_RULE_TYPES,
    ConstitutionalRule,
    ConstitutionV2,
    EffectScope,
    EffectTarget,
    MechanismEffect,
    RuleImpact,
    RuleType,
    _normalize_rule_type,
    create_default_constitution,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_household(
    id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    role: OccupationalRole = OccupationalRole.WORKER,
    income: float = 10.0,
    labor_supply: float = 0.8,
    consumption: float = 5.0,
    leisure: float = 0.2,
) -> HouseholdState:
    """Create a household with sensible defaults for mechanism effect tests."""
    return HouseholdState(
        id=id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.4, beta=0.3, gamma=0.3),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
        income=income,
        labor_supply=labor_supply,
        consumption=consumption,
        leisure=leisure,
    )


def make_market() -> MarketState:
    """Create a standard market state for tests."""
    return MarketState(
        wage=1.0,
        interest_rate=0.05,
        aggregate_output=100.0,
        aggregate_consumption=80.0,
        government_spending=10.0,
    )


def _make_insurance_rule(
    effects: list[MechanismEffect] | None = None,
) -> ConstitutionalRule:
    """Create a novel 'insurance' rule with mechanism effects."""
    if effects is None:
        effects = [
            MechanismEffect(
                target=EffectTarget.REVENUE,
                scope=EffectScope.ALL,
                magnitude_code="income * premium_rate",
                magnitude_default=1.0,
                direction="subtract",
            ),
            MechanismEffect(
                target=EffectTarget.DISTRIBUTION,
                scope=EffectScope.WEALTH_BELOW,
                scope_threshold=50.0,
                magnitude_code="revenue / num_agents",
                magnitude_default=0.0,
                direction="add",
            ),
        ]
    return ConstitutionalRule(
        name="unemployment_insurance",
        rule_type="insurance",
        parameters={"premium_rate": 0.05},
        description="Unemployment insurance funded by premiums.",
        mechanism_effects=effects,
    )


@pytest.fixture
def engine() -> ConstitutionEngine:
    return ConstitutionEngine()


@pytest.fixture
def market() -> MarketState:
    return make_market()


# ===========================================================================
# 1-2. EffectTarget and EffectScope enums
# ===========================================================================


class TestEffectEnums:
    """Test that EffectTarget and EffectScope contain expected members."""

    @pytest.mark.parametrize(
        "member,value",
        [
            (EffectTarget.REVENUE, "revenue"),
            (EffectTarget.DISTRIBUTION, "distribution"),
            (EffectTarget.PRODUCTIVITY, "productivity"),
            (EffectTarget.WEALTH_FLOW, "wealth_flow"),
            (EffectTarget.CONSTRAINT, "constraint"),
            (EffectTarget.UTILITY, "utility"),
            (EffectTarget.PUBLIC_GOODS, "public_goods"),
        ],
    )
    def test_effect_target_values(self, member: EffectTarget, value: str) -> None:
        assert member.value == value
        assert member == value

    @pytest.mark.parametrize(
        "member,value",
        [
            (EffectScope.ALL, "all"),
            (EffectScope.WORKERS, "workers"),
            (EffectScope.ENTREPRENEURS, "entrepreneurs"),
            (EffectScope.WEALTH_BELOW, "wealth_below"),
            (EffectScope.WEALTH_ABOVE, "wealth_above"),
            (EffectScope.CONDITION, "condition"),
        ],
    )
    def test_effect_scope_values(self, member: EffectScope, value: str) -> None:
        assert member.value == value
        assert member == value


# ===========================================================================
# 3. MechanismEffect model
# ===========================================================================


class TestMechanismEffect:
    """Test MechanismEffect creation, defaults, and validation."""

    def test_minimal_creation(self) -> None:
        effect = MechanismEffect(target=EffectTarget.REVENUE)
        assert effect.target == EffectTarget.REVENUE
        assert effect.scope == EffectScope.ALL
        assert effect.scope_threshold is None
        assert effect.magnitude_code == ""
        assert effect.magnitude_default == 0.0
        assert effect.direction == "subtract"
        assert effect.priority == 0

    def test_full_creation(self) -> None:
        effect = MechanismEffect(
            target=EffectTarget.WEALTH_FLOW,
            scope=EffectScope.WEALTH_BELOW,
            scope_threshold=200.0,
            magnitude_code="income * 0.02",
            magnitude_default=1.5,
            direction="add",
            priority=5,
        )
        assert effect.target == EffectTarget.WEALTH_FLOW
        assert effect.scope == EffectScope.WEALTH_BELOW
        assert effect.scope_threshold == 200.0
        assert effect.magnitude_code == "income * 0.02"
        assert effect.magnitude_default == 1.5
        assert effect.direction == "add"
        assert effect.priority == 5

    def test_direction_literal_constraint(self) -> None:
        """Direction must be 'add' or 'subtract'."""
        with pytest.raises(Exception):
            MechanismEffect(target=EffectTarget.REVENUE, direction="multiply")  # type: ignore[arg-type]


# ===========================================================================
# 4. RuleImpact model
# ===========================================================================


class TestRuleImpact:
    """Test RuleImpact creation and defaults."""

    def test_defaults(self) -> None:
        impact = RuleImpact()
        assert impact.periods_active == 0
        assert impact.total_revenue_collected == 0.0
        assert impact.total_distributed == 0.0
        assert impact.affected_agents_count == 0
        assert impact.gini_delta == 0.0
        assert impact.welfare_delta == 0.0

    def test_custom_values(self) -> None:
        impact = RuleImpact(
            periods_active=10,
            total_revenue_collected=500.0,
            total_distributed=450.0,
            affected_agents_count=25,
            gini_delta=-0.02,
            welfare_delta=3.5,
        )
        assert impact.periods_active == 10
        assert impact.total_revenue_collected == 500.0
        assert impact.total_distributed == 450.0
        assert impact.affected_agents_count == 25
        assert impact.gini_delta == -0.02
        assert impact.welfare_delta == 3.5


# ===========================================================================
# 5-6. ConstitutionalRule rule_type normalization and KNOWN_RULE_TYPES
# ===========================================================================


class TestConstitutionalRuleNormalization:
    """Test rule_type normalization, open string types, backward compat."""

    def test_known_rule_types_contains_8_types(self) -> None:
        assert len(KNOWN_RULE_TYPES) == 8
        expected = {
            "tax_schedule",
            "transfer_program",
            "public_goods",
            "market_regulation",
            "voting_procedure",
            "property_rights",
            "firm_regulation",
            "custom",
        }
        assert KNOWN_RULE_TYPES == expected

    def test_rule_type_accepts_enum_value(self) -> None:
        rule = ConstitutionalRule(
            name="test",
            rule_type=RuleType.TAX_SCHEDULE,
        )
        assert rule.rule_type == "tax_schedule"

    def test_rule_type_accepts_arbitrary_string(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        assert rule.rule_type == "insurance"

    def test_rule_type_normalizes_spaces(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="labor law")
        assert rule.rule_type == "labor_law"

    def test_rule_type_normalizes_dashes(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="financial-regulation")
        assert rule.rule_type == "financial_regulation"

    def test_rule_type_normalizes_uppercase(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="EDUCATION")
        assert rule.rule_type == "education"

    def test_rule_type_normalizes_mixed_case_with_spaces(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="Social Institution")
        assert rule.rule_type == "social_institution"

    def test_rule_type_strips_special_chars(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="carbon_tax!@#$%")
        assert rule.rule_type == "carbon_tax"

    def test_mechanism_effects_field_default_empty(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type="custom")
        assert rule.mechanism_effects == []

    def test_mechanism_effects_field_populated(self) -> None:
        effect = MechanismEffect(target=EffectTarget.REVENUE)
        rule = ConstitutionalRule(
            name="test",
            rule_type="insurance",
            mechanism_effects=[effect],
        )
        assert len(rule.mechanism_effects) == 1
        assert rule.mechanism_effects[0].target == EffectTarget.REVENUE


class TestNormalizeRuleTypeFunction:
    """Test the module-level _normalize_rule_type function directly."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("tax_schedule", "tax_schedule"),
            ("Tax Schedule", "tax_schedule"),
            ("TAX-SCHEDULE", "tax_schedule"),
            ("  insurance  ", "insurance"),
            ("FINANCIAL--REGULATION", "financial_regulation"),
            ("labor   law", "labor_law"),
            ("carbon_tax!@#", "carbon_tax"),
            ("Hello World 123", "hello_world_123"),
            ("a-b--c___d  e", "a_b_c___d_e"),
        ],
    )
    def test_normalize_rule_type(self, input_str: str, expected: str) -> None:
        assert _normalize_rule_type(input_str) == expected


# ===========================================================================
# 7. ConstitutionalProposal rule_type normalization and mechanism_effects
# ===========================================================================


class TestConstitutionalProposalMechanismEffects:
    """Test ConstitutionalProposal accepts str rule_type and mechanism_effects."""

    def test_proposal_normalizes_rule_type(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="my_rule",
            rule_type="Social Institution",
        )
        assert proposal.rule_type == "social_institution"

    def test_proposal_rule_type_none_for_modify(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="flat_tax",
        )
        assert proposal.rule_type is None

    def test_proposal_mechanism_effects_default_empty(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="test",
            rule_type="custom",
        )
        assert proposal.mechanism_effects == []

    def test_proposal_carries_mechanism_effects(self) -> None:
        effects = [
            MechanismEffect(target=EffectTarget.REVENUE, magnitude_default=5.0),
            MechanismEffect(target=EffectTarget.DISTRIBUTION, direction="add"),
        ]
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="test",
            rule_type="insurance",
            mechanism_effects=effects,
        )
        assert len(proposal.mechanism_effects) == 2
        assert proposal.mechanism_effects[0].magnitude_default == 5.0
        assert proposal.mechanism_effects[1].direction == "add"


# ===========================================================================
# 8. _agents_in_scope
# ===========================================================================


class TestAgentsInScope:
    """Test scope filtering logic in ConstitutionEngine._agents_in_scope."""

    def _build_households(self) -> list[HouseholdState]:
        return [
            make_household(id="w1", role=OccupationalRole.WORKER, wealth=30.0),
            make_household(id="w2", role=OccupationalRole.WORKER, wealth=80.0),
            make_household(id="e1", role=OccupationalRole.ENTREPRENEUR, wealth=200.0),
            make_household(id="e2", role=OccupationalRole.ENTREPRENEUR, wealth=40.0),
        ]

    def test_scope_all(self, engine: ConstitutionEngine) -> None:
        hh = self._build_households()
        effect = MechanismEffect(target=EffectTarget.REVENUE, scope=EffectScope.ALL)
        result = engine._agents_in_scope(effect, hh)
        assert len(result) == 4

    def test_scope_workers(self, engine: ConstitutionEngine) -> None:
        hh = self._build_households()
        effect = MechanismEffect(target=EffectTarget.REVENUE, scope=EffectScope.WORKERS)
        result = engine._agents_in_scope(effect, hh)
        assert len(result) == 2
        assert all(h.role == OccupationalRole.WORKER for h in result)

    def test_scope_entrepreneurs(self, engine: ConstitutionEngine) -> None:
        hh = self._build_households()
        effect = MechanismEffect(target=EffectTarget.REVENUE, scope=EffectScope.ENTREPRENEURS)
        result = engine._agents_in_scope(effect, hh)
        assert len(result) == 2
        assert all(h.role == OccupationalRole.ENTREPRENEUR for h in result)

    def test_scope_wealth_below(self, engine: ConstitutionEngine) -> None:
        hh = self._build_households()
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.WEALTH_BELOW,
            scope_threshold=50.0,
        )
        result = engine._agents_in_scope(effect, hh)
        ids = {h.id for h in result}
        assert ids == {"w1", "e2"}

    def test_scope_wealth_above(self, engine: ConstitutionEngine) -> None:
        hh = self._build_households()
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.WEALTH_ABOVE,
            scope_threshold=50.0,
        )
        result = engine._agents_in_scope(effect, hh)
        ids = {h.id for h in result}
        # wealth >= threshold: w2 (80) and e1 (200)
        assert ids == {"w2", "e1"}

    def test_scope_wealth_below_no_threshold_defaults_to_zero(
        self, engine: ConstitutionEngine
    ) -> None:
        hh = self._build_households()
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.WEALTH_BELOW,
            # scope_threshold not set, defaults to None -> 0.0
        )
        result = engine._agents_in_scope(effect, hh)
        # All agents have wealth > 0, so none qualify
        assert len(result) == 0


# ===========================================================================
# 9. _evaluate_magnitude
# ===========================================================================


class TestEvaluateMagnitude:
    """Test magnitude evaluation with code, defaults, and fallbacks."""

    def test_with_code(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="income * 0.05",
            magnitude_default=1.0,
        )
        rule = ConstitutionalRule(
            name="test", rule_type="insurance", parameters={"premium_rate": 0.05}
        )
        h = make_household(income=100.0)
        result = engine._evaluate_magnitude(effect, rule, h)
        assert abs(result - 5.0) < 1e-9

    def test_without_code_returns_default(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="",
            magnitude_default=3.14,
        )
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        h = make_household()
        result = engine._evaluate_magnitude(effect, rule, h)
        assert result == 3.14

    def test_nan_falls_back_to_default(self, engine: ConstitutionEngine) -> None:
        # 0.0 / 0.0 in Python raises ZeroDivisionError, but inf - inf = nan
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="wealth - wealth",  # = 0, which is finite
            magnitude_default=42.0,
        )
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        h = make_household()
        # This returns 0.0 (finite), so it does NOT fall back.
        # Let's use a code that produces NaN through sandbox:
        effect_nan = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="float('nan')",
            magnitude_default=42.0,
        )
        result = engine._evaluate_magnitude(effect_nan, rule, h)
        # float('nan') is not finite, so should fall back
        assert result == 42.0

    def test_with_extra_vars(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.DISTRIBUTION,
            magnitude_code="revenue / num_agents",
            magnitude_default=0.0,
        )
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        h = make_household()
        result = engine._evaluate_magnitude(
            effect, rule, h, extra_vars={"revenue": 100.0, "num_agents": 5}
        )
        assert abs(result - 20.0) < 1e-9

    def test_with_market_variables(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="income * interest_rate",
            magnitude_default=0.0,
        )
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        h = make_household(income=200.0)
        market = make_market()
        result = engine._evaluate_magnitude(effect, rule, h, market)
        assert abs(result - 200.0 * 0.05) < 1e-9

    def test_code_uses_rule_parameters(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="income * premium_rate",
            magnitude_default=0.0,
        )
        rule = ConstitutionalRule(
            name="test", rule_type="insurance", parameters={"premium_rate": 0.08}
        )
        h = make_household(income=100.0)
        result = engine._evaluate_magnitude(effect, rule, h)
        assert abs(result - 8.0) < 1e-9

    def test_sandbox_error_falls_back(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            magnitude_code="undefined_variable * 2",
            magnitude_default=7.77,
        )
        rule = ConstitutionalRule(name="test", rule_type="insurance")
        h = make_household()
        result = engine._evaluate_magnitude(effect, rule, h)
        assert result == 7.77


# ===========================================================================
# 10. apply_productivity_effects
# ===========================================================================


class TestApplyProductivityEffects:
    """Test productivity modification via mechanism effects."""

    def test_add_productivity(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.PRODUCTIVITY,
            scope=EffectScope.ALL,
            magnitude_code="0.5",
            direction="add",
        )
        rule = ConstitutionalRule(
            name="education_subsidy",
            rule_type="education",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"education_subsidy": rule})
        households = [make_household(productivity=1.0)]

        result = engine.apply_productivity_effects(households, constitution)
        assert abs(result[0].productivity - 1.5) < 1e-9

    def test_subtract_productivity(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.PRODUCTIVITY,
            scope=EffectScope.ALL,
            magnitude_code="0.3",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="env_reg",
            rule_type="environmental",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"env_reg": rule})
        households = [make_household(productivity=1.0)]

        result = engine.apply_productivity_effects(households, constitution)
        assert abs(result[0].productivity - 0.7) < 1e-9

    def test_clamp_productivity_lower_bound(self, engine: ConstitutionEngine) -> None:
        """Productivity cannot go below 0.01."""
        effect = MechanismEffect(
            target=EffectTarget.PRODUCTIVITY,
            scope=EffectScope.ALL,
            magnitude_code="999.0",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="destroy_prod",
            rule_type="environmental",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"destroy_prod": rule})
        households = [make_household(productivity=1.0)]

        result = engine.apply_productivity_effects(households, constitution)
        assert result[0].productivity == 0.01

    def test_clamp_productivity_upper_bound(self, engine: ConstitutionEngine) -> None:
        """Productivity cannot exceed 100.0."""
        effect = MechanismEffect(
            target=EffectTarget.PRODUCTIVITY,
            scope=EffectScope.ALL,
            magnitude_code="200.0",
            direction="add",
        )
        rule = ConstitutionalRule(
            name="mega_boost",
            rule_type="education",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"mega_boost": rule})
        households = [make_household(productivity=1.0)]

        result = engine.apply_productivity_effects(households, constitution)
        assert result[0].productivity == 100.0

    def test_no_productivity_effects_returns_original(
        self, engine: ConstitutionEngine
    ) -> None:
        """When no PRODUCTIVITY effects exist, return original list."""
        rule = ConstitutionalRule(
            name="tax_rule",
            rule_type="tax_schedule",
            mechanism_effects=[
                MechanismEffect(target=EffectTarget.REVENUE),
            ],
        )
        constitution = ConstitutionV2(rules={"tax_rule": rule})
        households = [make_household(productivity=1.0)]

        result = engine.apply_productivity_effects(households, constitution)
        # Returns original list, not a copy
        assert result is households

    def test_scoped_productivity_effect(self, engine: ConstitutionEngine) -> None:
        """Productivity boost only for workers."""
        effect = MechanismEffect(
            target=EffectTarget.PRODUCTIVITY,
            scope=EffectScope.WORKERS,
            magnitude_code="0.5",
            direction="add",
        )
        rule = ConstitutionalRule(
            name="worker_training",
            rule_type="education",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"worker_training": rule})
        households = [
            make_household(id="w", role=OccupationalRole.WORKER, productivity=1.0),
            make_household(id="e", role=OccupationalRole.ENTREPRENEUR, productivity=1.0),
        ]

        result = engine.apply_productivity_effects(households, constitution)
        worker = next(h for h in result if h.id == "w")
        entrepreneur = next(h for h in result if h.id == "e")
        assert abs(worker.productivity - 1.5) < 1e-9
        assert abs(entrepreneur.productivity - 1.0) < 1e-9


# ===========================================================================
# 11. compute_agent_constraints
# ===========================================================================


class TestComputeAgentConstraints:
    """Test custom min/max bounds from CONSTRAINT effects."""

    def test_returns_constraints(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.CONSTRAINT,
            scope=EffectScope.ALL,
            magnitude_code="0.1",
            direction="add",  # min bound
        )
        rule = ConstitutionalRule(
            name="min_consumption",
            rule_type="social_institution",
            parameters={"constraint_variable": "consumption"},
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"min_consumption": rule})
        households = [make_household(id="a0")]

        result = engine.compute_agent_constraints(households, constitution)
        assert "a0" in result
        assert "consumption_min" in result["a0"]
        assert abs(result["a0"]["consumption_min"] - 0.1) < 1e-9

    def test_max_bound_via_subtract_direction(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.CONSTRAINT,
            scope=EffectScope.ALL,
            magnitude_code="0.8",
            direction="subtract",  # max bound
        )
        rule = ConstitutionalRule(
            name="max_leisure",
            rule_type="labor_law",
            parameters={"constraint_variable": "leisure"},
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"max_leisure": rule})
        households = [make_household(id="a0")]

        result = engine.compute_agent_constraints(households, constitution)
        assert "leisure_max" in result["a0"]
        assert abs(result["a0"]["leisure_max"] - 0.8) < 1e-9

    def test_conflict_min_greater_than_max_drops_max(
        self, engine: ConstitutionEngine
    ) -> None:
        """When min > max, the max constraint is dropped."""
        effects = [
            MechanismEffect(
                target=EffectTarget.CONSTRAINT,
                scope=EffectScope.ALL,
                magnitude_code="0.9",
                direction="add",  # min = 0.9
                priority=0,
            ),
            MechanismEffect(
                target=EffectTarget.CONSTRAINT,
                scope=EffectScope.ALL,
                magnitude_code="0.3",
                direction="subtract",  # max = 0.3
                priority=1,
            ),
        ]
        rule = ConstitutionalRule(
            name="bad_constraint",
            rule_type="labor_law",
            parameters={"constraint_variable": "consumption"},
            mechanism_effects=effects,
        )
        constitution = ConstitutionV2(rules={"bad_constraint": rule})
        households = [make_household(id="a0")]

        result = engine.compute_agent_constraints(households, constitution)
        assert "consumption_min" in result["a0"]
        assert "consumption_max" not in result["a0"]

    def test_empty_when_no_constraint_effects(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="revenue_only",
            rule_type="insurance",
            mechanism_effects=[MechanismEffect(target=EffectTarget.REVENUE)],
        )
        constitution = ConstitutionV2(rules={"revenue_only": rule})
        households = [make_household()]

        result = engine.compute_agent_constraints(households, constitution)
        assert result == {}


# ===========================================================================
# 12. enforce_mechanism_effects
# ===========================================================================


class TestEnforceMechanismEffects:
    """Test REVENUE, DISTRIBUTION, WEALTH_FLOW, PUBLIC_GOODS enforcement."""

    def test_revenue_collection(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.ALL,
            magnitude_code="income * premium_rate",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="insurance",
            rule_type="insurance",
            parameters={"premium_rate": 0.1},
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"insurance": rule})
        households = [make_household(income=100.0, wealth=500.0)]
        market = make_market()

        updated, pg = engine.enforce_mechanism_effects(households, constitution, market)
        # Premium: 100 * 0.1 = 10
        assert abs(updated[0].wealth - 490.0) < 1e-9
        assert abs(updated[0].taxes_paid - 10.0) < 1e-9
        assert pg == 0.0

    def test_revenue_capped_at_income(self, engine: ConstitutionEngine) -> None:
        """Revenue collected from an agent cannot exceed their income."""
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.ALL,
            magnitude_code="999.0",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="big_tax",
            rule_type="insurance",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"big_tax": rule})
        households = [make_household(income=10.0, wealth=100.0)]
        market = make_market()

        updated, _ = engine.enforce_mechanism_effects(households, constitution, market)
        # Capped at income=10
        assert abs(updated[0].taxes_paid - 10.0) < 1e-9

    def test_distribution_effect(self, engine: ConstitutionEngine) -> None:
        """Distribution is sequential: each agent receives revenue/num_agents
        from the remaining pool, so total distributed < total revenue when
        magnitude_code depends on the shrinking available balance."""
        effects = [
            MechanismEffect(
                target=EffectTarget.REVENUE,
                scope=EffectScope.ALL,
                magnitude_code="income * 0.1",
                direction="subtract",
                priority=0,
            ),
            MechanismEffect(
                target=EffectTarget.DISTRIBUTION,
                scope=EffectScope.ALL,
                magnitude_code="revenue / num_agents",
                direction="add",
                priority=1,
            ),
        ]
        rule = ConstitutionalRule(
            name="insurance",
            rule_type="insurance",
            mechanism_effects=effects,
        )
        constitution = ConstitutionV2(rules={"insurance": rule})
        households = [
            make_household(id="a0", income=100.0, wealth=500.0),
            make_household(id="a1", income=100.0, wealth=500.0),
        ]
        market = make_market()

        updated, _ = engine.enforce_mechanism_effects(households, constitution, market)
        # Revenue: 2 * (100*0.1) = 20 total, each pays 10 -> wealth 490
        # Distribution is sequential:
        #   a0: available=20, gets 20/2=10 -> wealth 500, available=10
        #   a1: available=10, gets 10/2=5  -> wealth 495, available=5
        total_transfers = sum(h.transfers_received for h in updated)
        assert total_transfers > 0
        a0 = next(h for h in updated if h.id == "a0")
        a1 = next(h for h in updated if h.id == "a1")
        assert abs(a0.wealth - 500.0) < 1e-6  # -10 + 10
        assert abs(a1.wealth - 495.0) < 1e-6  # -10 + 5

    def test_wealth_flow_add(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.WEALTH_FLOW,
            scope=EffectScope.ALL,
            magnitude_code="5.0",
            direction="add",
        )
        rule = ConstitutionalRule(
            name="ubi",
            rule_type="social_institution",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"ubi": rule})
        households = [make_household(wealth=100.0)]
        market = make_market()

        updated, _ = engine.enforce_mechanism_effects(households, constitution, market)
        assert abs(updated[0].wealth - 105.0) < 1e-9

    def test_wealth_flow_subtract_with_floor(self, engine: ConstitutionEngine) -> None:
        """Wealth flow cannot drop below a_min."""
        effect = MechanismEffect(
            target=EffectTarget.WEALTH_FLOW,
            scope=EffectScope.ALL,
            magnitude_code="999.0",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="penalty",
            rule_type="social_institution",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"penalty": rule})
        households = [make_household(wealth=10.0)]
        market = make_market()

        updated, _ = engine.enforce_mechanism_effects(
            households, constitution, market, a_min=5.0
        )
        assert updated[0].wealth >= 5.0

    def test_public_goods_contribution(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.PUBLIC_GOODS,
            scope=EffectScope.ALL,
            magnitude_code="2.0",
            direction="add",
        )
        rule = ConstitutionalRule(
            name="public_parks",
            rule_type="environmental",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"public_parks": rule})
        households = [make_household(), make_household(id="a1")]
        market = make_market()

        _, pg = engine.enforce_mechanism_effects(households, constitution, market)
        # 2 agents * 2.0 magnitude each = 4.0
        assert abs(pg - 4.0) < 1e-9

    def test_multi_effect_composition(self, engine: ConstitutionEngine) -> None:
        """A rule with REVENUE + DISTRIBUTION + PUBLIC_GOODS composes correctly."""
        effects = [
            MechanismEffect(
                target=EffectTarget.REVENUE,
                scope=EffectScope.ALL,
                magnitude_code="income * 0.05",
                direction="subtract",
                priority=0,
            ),
            MechanismEffect(
                target=EffectTarget.DISTRIBUTION,
                scope=EffectScope.WEALTH_BELOW,
                scope_threshold=80.0,
                magnitude_code="revenue / num_agents",
                direction="add",
                priority=1,
            ),
            MechanismEffect(
                target=EffectTarget.PUBLIC_GOODS,
                scope=EffectScope.ALL,
                magnitude_code="1.0",
                direction="add",
                priority=2,
            ),
        ]
        rule = ConstitutionalRule(
            name="welfare_system",
            rule_type="social_institution",
            mechanism_effects=effects,
        )
        constitution = ConstitutionV2(rules={"welfare_system": rule})
        households = [
            make_household(id="poor", wealth=50.0, income=40.0),
            make_household(id="rich", wealth=200.0, income=100.0),
        ]
        market = make_market()

        updated, pg = engine.enforce_mechanism_effects(households, constitution, market)
        # Revenue: poor pays 40*0.05=2, rich pays 100*0.05=5, total=7
        # Distribution to WEALTH_BELOW 80: only "poor" qualifies
        # Public goods: 2 agents * 1.0 = 2.0
        assert pg > 0
        total_taxes = sum(h.taxes_paid for h in updated)
        assert total_taxes > 0

    def test_no_mechanism_rules_is_noop(self, engine: ConstitutionEngine) -> None:
        """Constitution with no mechanism_effects returns original list."""
        constitution = create_default_constitution()
        households = [make_household()]
        market = make_market()

        updated, pg = engine.enforce_mechanism_effects(households, constitution, market)
        assert updated is households
        assert pg == 0.0


# ===========================================================================
# 13. validate_rule with mechanism_effects magnitude_code safety
# ===========================================================================


class TestValidateRuleMechanismEffects:
    """Test that validate_rule checks magnitude_code in mechanism_effects."""

    def test_safe_magnitude_code_passes(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="test",
            rule_type="insurance",
            mechanism_effects=[
                MechanismEffect(
                    target=EffectTarget.REVENUE,
                    magnitude_code="income * 0.05",
                ),
            ],
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid
        assert reason == "OK"

    def test_unsafe_magnitude_code_rejected(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="test",
            rule_type="insurance",
            mechanism_effects=[
                MechanismEffect(
                    target=EffectTarget.REVENUE,
                    magnitude_code="__import__('os').system('rm -rf /')",
                ),
            ],
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid
        assert "magnitude_code" in reason.lower() or "unsafe" in reason.lower()

    def test_empty_magnitude_code_passes(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="test",
            rule_type="insurance",
            mechanism_effects=[
                MechanismEffect(
                    target=EffectTarget.REVENUE,
                    magnitude_code="",
                ),
            ],
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid


# ===========================================================================
# 14. apply_proposal passes through mechanism_effects
# ===========================================================================


class TestApplyProposalMechanismEffects:
    """Test that apply_proposal carries mechanism_effects into new rules."""

    def test_add_rule_with_mechanism_effects(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        effects = [
            MechanismEffect(
                target=EffectTarget.REVENUE,
                magnitude_code="income * 0.03",
            ),
        ]
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="unemployment_insurance",
            rule_type="insurance",
            parameters={"premium_rate": 0.03},
            description="UI funded by payroll premiums",
            mechanism_effects=effects,
        )
        updated = engine.apply_proposal(constitution, proposal)
        new_rule = updated.rules["unemployment_insurance"]
        assert len(new_rule.mechanism_effects) == 1
        assert new_rule.mechanism_effects[0].target == EffectTarget.REVENUE
        assert new_rule.rule_type == "insurance"

    def test_modify_rule_with_mechanism_effects(self, engine: ConstitutionEngine) -> None:
        # First add a rule with effects
        constitution = create_default_constitution()
        initial_effects = [MechanismEffect(target=EffectTarget.REVENUE)]
        add_proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="insurance",
            rule_type="insurance",
            mechanism_effects=initial_effects,
        )
        constitution = engine.apply_proposal(constitution, add_proposal)

        # Now modify with new effects
        new_effects = [
            MechanismEffect(target=EffectTarget.REVENUE, magnitude_code="income * 0.1"),
            MechanismEffect(target=EffectTarget.DISTRIBUTION, direction="add"),
        ]
        modify_proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="insurance",
            mechanism_effects=new_effects,
        )
        updated = engine.apply_proposal(constitution, modify_proposal)
        assert len(updated.rules["insurance"].mechanism_effects) == 2


# ===========================================================================
# 15. Rule impact tracking
# ===========================================================================


class TestRuleImpactTracking:
    """Test that enforce_mechanism_effects accumulates rule_impacts."""

    def test_rule_impact_accumulates(self, engine: ConstitutionEngine) -> None:
        effect = MechanismEffect(
            target=EffectTarget.REVENUE,
            scope=EffectScope.ALL,
            magnitude_code="income * 0.1",
            direction="subtract",
        )
        rule = ConstitutionalRule(
            name="premium_tax",
            rule_type="insurance",
            mechanism_effects=[effect],
        )
        constitution = ConstitutionV2(rules={"premium_tax": rule})
        households = [make_household(income=100.0, wealth=500.0)]
        market = make_market()

        # Enforce twice
        engine.enforce_mechanism_effects(households, constitution, market)
        engine.enforce_mechanism_effects(households, constitution, market)

        impact = engine.rule_impacts["premium_tax"]
        assert impact.periods_active == 2
        assert impact.total_revenue_collected > 0
        assert impact.affected_agents_count > 0

    def test_rule_impact_starts_empty(self, engine: ConstitutionEngine) -> None:
        assert engine.rule_impacts == {}


# ===========================================================================
# 16. Expanded sandbox builtins
# ===========================================================================


class TestExpandedSandbox:
    """Test new builtins available in sandbox: float, int, bool, ceil, floor, log10."""

    def test_float_builtin(self) -> None:
        result = evaluate_enforcement_code("float(42)", {})
        assert result == 42.0
        assert isinstance(result, float)

    def test_int_builtin(self) -> None:
        result = evaluate_enforcement_code("int(3.7)", {})
        assert result == 3

    def test_bool_builtin(self) -> None:
        result = evaluate_enforcement_code("bool(1)", {})
        assert result is True

    def test_ceil_function(self) -> None:
        result = evaluate_enforcement_code("ceil(2.3)", {})
        assert result == 3

    def test_floor_function(self) -> None:
        result = evaluate_enforcement_code("floor(2.9)", {})
        assert result == 2

    def test_log10_function(self) -> None:
        result = evaluate_enforcement_code("log10(1000)", {})
        assert abs(result - 3.0) < 1e-9

    def test_compound_expression_with_new_builtins(self) -> None:
        result = evaluate_enforcement_code(
            "float(ceil(log10(income)))", {"income": 500.0}
        )
        # log10(500) ~ 2.699, ceil -> 3, float -> 3.0
        assert result == 3.0


# ===========================================================================
# 17. Integration: full lifecycle
# ===========================================================================


class TestFullLifecycleIntegration:
    """Propose a novel rule with mechanism_effects, enforce, observe impacts."""

    def test_propose_add_enforce_observe(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        market = make_market()

        # Step 1: Propose a novel insurance institution
        effects = [
            MechanismEffect(
                target=EffectTarget.REVENUE,
                scope=EffectScope.ALL,
                magnitude_code="income * premium_rate",
                magnitude_default=1.0,
                direction="subtract",
                priority=0,
            ),
            MechanismEffect(
                target=EffectTarget.DISTRIBUTION,
                scope=EffectScope.WEALTH_BELOW,
                scope_threshold=60.0,
                magnitude_code="revenue / num_agents",
                magnitude_default=0.0,
                direction="add",
                priority=1,
            ),
        ]
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="unemployment_insurance",
            rule_type="insurance",
            parameters={"premium_rate": 0.05},
            description="UI funded by payroll premiums.",
            mechanism_effects=effects,
        )

        # Step 2: Apply (simulates vote passing)
        constitution = engine.apply_proposal(constitution, proposal)
        assert "unemployment_insurance" in constitution.rules

        # Step 3: Enforce mechanism effects
        households = [
            make_household(id="poor", wealth=40.0, income=50.0),
            make_household(id="middle", wealth=80.0, income=70.0),
            make_household(id="rich", wealth=300.0, income=200.0),
        ]
        updated, pg = engine.enforce_mechanism_effects(
            households, constitution, market
        )

        # Verify revenue was collected
        total_tax = sum(h.taxes_paid for h in updated)
        assert total_tax > 0

        # Verify distribution went to WEALTH_BELOW 60 (only "poor")
        poor = next(h for h in updated if h.id == "poor")
        assert poor.transfers_received > 0

        # Step 4: Check rule impact tracking
        assert "unemployment_insurance" in engine.rule_impacts
        impact = engine.rule_impacts["unemployment_insurance"]
        assert impact.periods_active == 1
        assert impact.total_revenue_collected > 0


# ===========================================================================
# 18. Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Existing rules without mechanism_effects work exactly as before."""

    def test_default_constitution_no_mechanism_effects(self) -> None:
        constitution = create_default_constitution()
        for rule in constitution.rules.values():
            assert rule.mechanism_effects == []

    def test_existing_tax_enforcement_unchanged(
        self, engine: ConstitutionEngine
    ) -> None:
        constitution = create_default_constitution()
        market = make_market()
        households = [
            make_household(id="a0", income=100.0, wealth=500.0),
            make_household(id="a1", income=200.0, wealth=500.0),
        ]

        taxed, revenue = engine.enforce_taxes(households, constitution, market)
        # Default flat tax at 10%: 100*0.1=10, 200*0.1=20 = 30
        assert abs(revenue - 30.0) < 1e-6

    def test_mechanism_enforcement_noop_on_default_constitution(
        self, engine: ConstitutionEngine
    ) -> None:
        """enforce_mechanism_effects is a no-op when there are no mechanism_effects."""
        constitution = create_default_constitution()
        households = [make_household()]
        market = make_market()

        updated, pg = engine.enforce_mechanism_effects(
            households, constitution, market
        )
        assert updated is households
        assert pg == 0.0

    def test_rule_type_enum_backward_compat(self) -> None:
        """RuleType enum values still work as rule_type strings."""
        for rt in RuleType:
            rule = ConstitutionalRule(name="test", rule_type=rt)
            assert rule.rule_type == rt.value

    def test_productivity_effects_noop_on_default(
        self, engine: ConstitutionEngine
    ) -> None:
        constitution = create_default_constitution()
        households = [make_household(productivity=1.0)]
        result = engine.apply_productivity_effects(households, constitution)
        assert result is households

    def test_constraints_empty_on_default(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        households = [make_household()]
        result = engine.compute_agent_constraints(households, constitution)
        assert result == {}
