"""Core data models for the simulation — v1 and v2.

v1 types are retained for backward compatibility with existing modules.
v2 types implement the DSGE-HA model hierarchy per spec/design.md.
"""

# v1 imports (backward compatibility)
from emergent_constitution.coalition import CoalitionInfo
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    CONSTITUTION_FIELDS,
    KNOWN_RULE_TYPES,
    Constitution,
    ConstitutionalRule,
    ConstitutionV2,
    EffectScope,
    EffectTarget,
    MechanismEffect,
    PropertyRule,
    RedistributionRule,
    RuleImpact,
    RuleType,
    VotingRule,
    create_default_constitution,
)
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import (
    HistoryEntry,
    HistoryEntryV2,
    PeriodState,
    SimulationOutput,
    SimulationOutputV2,
    WelfareSummary,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
)
from emergent_constitution.models.household import UtilityParams as UtilityParamsV2
from emergent_constitution.models.household import ValueVector as ValueVectorV2
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import (
    ConstitutionalProposal,
    Proposal,
    TradeOffer,
    VoteOutcome,
    VoteOutcomeV2,
)
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.models.tick import TickState

__all__ = [
    # v1
    "AgentState",
    "CoalitionInfo",
    "Constitution",
    "CONSTITUTION_FIELDS",
    "HistoryEntry",
    "PropertyRule",
    "Proposal",
    "RedistributionRule",
    "SimulationOutput",
    "TickState",
    "TradeOffer",
    "UtilityParams",
    "ValueVector",
    "VoteOutcome",
    "VotingRule",
    # v2
    "ConstitutionalProposal",
    "ConstitutionalRule",
    "ConstitutionV2",
    "EconomicDecision",
    "EffectScope",
    "EffectTarget",
    "KNOWN_RULE_TYPES",
    "MechanismEffect",
    "EntrepreneurialDecision",
    "FirmState",
    "HistoryEntryV2",
    "HouseholdState",
    "MarketState",
    "OccupationalRole",
    "PeriodState",
    "PoliticalDecision",
    "RuleImpact",
    "RuleType",
    "ShockState",
    "SimulationOutputV2",
    "UtilityParamsV2",
    "ValueVectorV2",
    "VoteOutcomeV2",
    "WelfareSummary",
    "create_default_constitution",
]
