"""Constitution models — v1 fixed ruleset and v2 extensible rules.

v1 types (PropertyRule, VotingRule, RedistributionRule, Constitution) are
retained for backward compatibility until all v1 code is migrated.

v2 types (RuleType, ConstitutionalRule, ConstitutionV2) implement the
extensible rules-based design per spec/design.md section 2.5.

MechanismEffect framework enables LLM-invented institutions to have
mechanical enforcement via structured effect primitives.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# v1 types (backward compatibility)
# ============================================================================


class PropertyRule(StrEnum):
    """How production output is allocated."""

    PRIVATE = "private"
    COMMUNAL = "communal"
    MIXED = "mixed"


class VotingRule(StrEnum):
    """How proposals are decided."""

    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    UNANIMITY = "unanimity"


class RedistributionRule(StrEnum):
    """How tax revenue is redistributed."""

    NONE = "none"
    FLAT = "flat"
    PROGRESSIVE = "progressive"


# Valid Constitution field keys and their allowed types/values.
CONSTITUTION_FIELDS: dict[str, type] = {
    "property_rule": PropertyRule,
    "tax_rate": float,
    "voting_rule": VotingRule,
    "redistribution_rule": RedistributionRule,
}


class Constitution(BaseModel):
    """The active ruleset governing the simulation.

    Args:
        property_rule: How production is allocated.
        tax_rate: Fraction of output collected as tax (0-1).
        voting_rule: How proposals are decided.
        redistribution_rule: How tax revenue is redistributed.
    """

    property_rule: PropertyRule = PropertyRule.PRIVATE
    tax_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    voting_rule: VotingRule = VotingRule.MAJORITY
    redistribution_rule: RedistributionRule = RedistributionRule.FLAT


# ============================================================================
# v2 types (DSGE-HA extensible constitution)
# ============================================================================


class RuleType(StrEnum):
    """Categories of constitutional rules.

    Core types have dedicated enforcement in ConstitutionEngine.
    Novel types (INSURANCE, EDUCATION, etc.) are enforced via
    the generic MechanismEffect framework.
    """

    TAX_SCHEDULE = "tax_schedule"
    TRANSFER_PROGRAM = "transfer_program"
    PUBLIC_GOODS = "public_goods"
    MARKET_REGULATION = "market_regulation"
    VOTING_PROCEDURE = "voting_procedure"
    PROPERTY_RIGHTS = "property_rights"
    FIRM_REGULATION = "firm_regulation"
    CUSTOM = "custom"
    # Novel institution types — enforced via mechanism_effects
    INSURANCE = "insurance"
    EDUCATION = "education"
    FINANCIAL_REGULATION = "financial_regulation"
    LABOR_LAW = "labor_law"
    ENVIRONMENTAL = "environmental"
    SOCIAL_INSTITUTION = "social_institution"


# Rule types with dedicated enforcement in ConstitutionEngine.
# Rules whose type is NOT in this set are enforced via mechanism_effects.
KNOWN_RULE_TYPES: set[str] = {
    RuleType.TAX_SCHEDULE,
    RuleType.TRANSFER_PROGRAM,
    RuleType.PUBLIC_GOODS,
    RuleType.MARKET_REGULATION,
    RuleType.VOTING_PROCEDURE,
    RuleType.PROPERTY_RIGHTS,
    RuleType.FIRM_REGULATION,
    RuleType.CUSTOM,
}


class EffectTarget(StrEnum):
    """What aspect of the simulation a mechanism effect modifies."""

    REVENUE = "revenue"
    DISTRIBUTION = "distribution"
    PRODUCTIVITY = "productivity"
    WEALTH_FLOW = "wealth_flow"
    CONSTRAINT = "constraint"
    UTILITY = "utility"
    PUBLIC_GOODS = "public_goods"


class EffectScope(StrEnum):
    """Which agents a mechanism effect applies to."""

    ALL = "all"
    WORKERS = "workers"
    ENTREPRENEURS = "entrepreneurs"
    WEALTH_BELOW = "wealth_below"
    WEALTH_ABOVE = "wealth_above"
    CONDITION = "condition"


class MechanismEffect(BaseModel):
    """A structured effect declaration for institutional enforcement.

    Args:
        target: What simulation primitive this effect modifies.
        scope: Which agents are affected.
        scope_threshold: Wealth threshold for WEALTH_BELOW/ABOVE scopes.
        magnitude_code: Sandbox expression computing the effect magnitude.
        magnitude_default: Fallback value if magnitude_code fails.
        direction: Whether the effect adds or subtracts.
        priority: Ordering for conflict resolution (lower = first).
    """

    target: EffectTarget
    scope: EffectScope = EffectScope.ALL
    scope_threshold: float | None = None
    magnitude_code: str = ""
    magnitude_default: float = 0.0
    direction: Literal["add", "subtract"] = "subtract"
    priority: int = 0


class RuleImpact(BaseModel):
    """Tracks the observed impact of a constitutional rule over time.

    Args:
        periods_active: Number of periods the rule has been active.
        total_revenue_collected: Cumulative revenue collected by this rule.
        total_distributed: Cumulative amount distributed by this rule.
        affected_agents_count: Number of unique agents affected.
        gini_delta: Change in Gini coefficient since rule enactment.
        welfare_delta: Change in social welfare since rule enactment.
    """

    periods_active: int = 0
    total_revenue_collected: float = 0.0
    total_distributed: float = 0.0
    affected_agents_count: int = 0
    gini_delta: float = 0.0
    welfare_delta: float = 0.0


def _normalize_rule_type(value: str) -> str:
    """Normalize a rule type string to lowercase snake_case."""
    value = value.strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value


class ConstitutionalRule(BaseModel):
    """A single named policy rule within the constitution.

    Args:
        name: Unique identifier (e.g. "income_tax").
        rule_type: Category of the rule (str, not restricted to RuleType enum).
        parameters: Type-specific params (e.g. {"rate": 0.2}).
        description: Natural-language description.
        enforcement_code: Python expression/function body for enforcement.
        mechanism_effects: Structured effect declarations for generic enforcement.
        version: Incremented on modification.
        enacted_period: Period this rule was enacted.
    """

    name: str
    rule_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    enforcement_code: str = ""
    mechanism_effects: list[MechanismEffect] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    enacted_period: int = Field(default=0, ge=0)

    @field_validator("rule_type", mode="before")
    @classmethod
    def _normalize_rule_type(cls, v: Any) -> str:
        if isinstance(v, str):
            return _normalize_rule_type(v)
        return str(v)


class ConstitutionV2(BaseModel):
    """Extensible, structured constitutional document (v2).

    Args:
        rules: Mapping from rule name to ConstitutionalRule.
        voting_rule: Name of the active voting procedure rule.
    """

    rules: dict[str, ConstitutionalRule] = Field(default_factory=dict)
    voting_rule: str = "majority_vote"

    def get_tax_rules(self) -> list[ConstitutionalRule]:
        """Return all active tax schedule rules."""
        return [r for r in self.rules.values() if r.rule_type == RuleType.TAX_SCHEDULE]

    def get_transfer_rules(self) -> list[ConstitutionalRule]:
        """Return all active transfer program rules."""
        return [r for r in self.rules.values() if r.rule_type == RuleType.TRANSFER_PROGRAM]

    def get_active_voting_rule(self) -> ConstitutionalRule | None:
        """Return the active voting procedure rule, or None if not found."""
        return self.rules.get(self.voting_rule)


def create_default_constitution() -> ConstitutionV2:
    """Create the default initial constitution with baseline rules.

    Returns:
        ConstitutionV2 with flat tax (10%), equal-share transfers,
        public goods provision, majority voting, and private property.
    """
    rules = {
        "flat_tax": ConstitutionalRule(
            name="flat_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.1},
            description="Flat income tax at the specified rate.",
            enforcement_code="tax = income * rate",
        ),
        "flat_transfer": ConstitutionalRule(
            name="flat_transfer",
            rule_type=RuleType.TRANSFER_PROGRAM,
            parameters={"method": "equal_share"},
            description="Equal share redistribution of tax revenue.",
            enforcement_code="transfer = revenue / num_agents",
        ),
        "public_goods_provision": ConstitutionalRule(
            name="public_goods_provision",
            rule_type=RuleType.PUBLIC_GOODS,
            parameters={"fraction_of_revenue": 0.3},
            description="Allocate a fraction of tax revenue to public goods.",
            enforcement_code="G = fraction_of_revenue * revenue / num_agents",
        ),
        "majority_vote": ConstitutionalRule(
            name="majority_vote",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.5},
            description="Simple majority voting rule.",
            enforcement_code="passed = votes_for > total * threshold",
        ),
        "private_property": ConstitutionalRule(
            name="private_property",
            rule_type=RuleType.PROPERTY_RIGHTS,
            parameters={"regime": "private"},
            description="Private property rights regime.",
            enforcement_code="",
        ),
    }
    return ConstitutionV2(rules=rules, voting_rule="majority_vote")
