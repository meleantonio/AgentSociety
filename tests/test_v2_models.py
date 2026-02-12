"""Unit tests for v2 data models: validation, serialization, defaults, edge cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
    PoliticalDecision,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import (
    HistoryEntryV2,
    PeriodState,
    SimulationOutputV2,
    WelfareSummary,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal, VoteOutcomeV2
from emergent_constitution.models.shocks import ShockState

# --- Fixtures ---


@pytest.fixture()
def utility_params() -> UtilityParams:
    return UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.95)


@pytest.fixture()
def value_vector() -> ValueVector:
    return ValueVector(equality=0.6, liberty=0.4)


@pytest.fixture()
def household(utility_params: UtilityParams, value_vector: ValueVector) -> HouseholdState:
    return HouseholdState(
        id="agent_0000",
        wealth=100.0,
        productivity=1.5,
        productivity_index=2,
        utility_params=utility_params,
        value_vector=value_vector,
    )


@pytest.fixture()
def firm() -> FirmState:
    return FirmState(
        id="firm_0000",
        owner_id="agent_0001",
        capital=50.0,
        labor_demand=10.0,
        tfp=1.2,
    )


@pytest.fixture()
def market() -> MarketState:
    return MarketState(wage=5.0, interest_rate=0.05)


@pytest.fixture()
def shocks() -> ShockState:
    return ShockState(
        productivity_grid=[0.5, 1.0, 1.5],
        transition_matrix=[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        aggregate_tfp=1.0,
    )


@pytest.fixture()
def constitution() -> ConstitutionV2:
    return create_default_constitution()


# --- UtilityParams (v2 with beta_discount) ---


class TestUtilityParamsV2:
    def test_valid_params(self) -> None:
        p = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2, beta_discount=0.96)
        assert abs(p.alpha + p.beta + p.gamma - 1.0) < 1e-6

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            UtilityParams(alpha=0.5, beta=0.5, gamma=0.5)

    def test_negative_alpha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UtilityParams(alpha=-0.1, beta=0.6, gamma=0.5)

    def test_beta_discount_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=1.0)
        with pytest.raises(ValidationError):
            UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.0)

    def test_serialization_round_trip(self) -> None:
        p = UtilityParams(alpha=0.4, beta=0.3, gamma=0.3, beta_discount=0.95)
        data = p.model_dump()
        p2 = UtilityParams.model_validate(data)
        assert p == p2


# --- ValueVector (v2) ---


class TestValueVectorV2:
    def test_valid_vector(self) -> None:
        v = ValueVector(equality=0.7, liberty=0.3)
        assert abs(v.equality + v.liberty - 1.0) < 1e-6

    def test_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            ValueVector(equality=0.3, liberty=0.3)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValueVector(equality=-0.1, liberty=1.1)


# --- OccupationalRole ---


class TestOccupationalRole:
    def test_values(self) -> None:
        assert OccupationalRole.WORKER == "worker"
        assert OccupationalRole.ENTREPRENEUR == "entrepreneur"
        assert OccupationalRole.RESEARCHER == "researcher"
        assert OccupationalRole.UNEMPLOYED == "unemployed"


# --- HouseholdState ---


class TestHouseholdState:
    def test_defaults(self, household: HouseholdState) -> None:
        assert household.role == OccupationalRole.WORKER
        assert household.firm_id is None
        assert household.coalition_id is None
        assert household.consumption == 0.0
        assert household.leisure == 0.0
        assert household.labor_supply == 0.0

    def test_negative_wealth_rejected(
        self, utility_params: UtilityParams, value_vector: ValueVector
    ) -> None:
        with pytest.raises(ValidationError):
            HouseholdState(
                id="agent_bad",
                wealth=-10.0,
                productivity=1.0,
                productivity_index=0,
                utility_params=utility_params,
                value_vector=value_vector,
            )

    def test_zero_productivity_rejected(
        self, utility_params: UtilityParams, value_vector: ValueVector
    ) -> None:
        with pytest.raises(ValidationError):
            HouseholdState(
                id="agent_bad",
                wealth=10.0,
                productivity=0.0,
                productivity_index=0,
                utility_params=utility_params,
                value_vector=value_vector,
            )

    def test_invalid_leisure(
        self, utility_params: UtilityParams, value_vector: ValueVector
    ) -> None:
        with pytest.raises(ValidationError):
            HouseholdState(
                id="agent_bad",
                wealth=10.0,
                productivity=1.0,
                productivity_index=0,
                utility_params=utility_params,
                value_vector=value_vector,
                leisure=1.5,
            )

    def test_serialization_round_trip(self, household: HouseholdState) -> None:
        data = household.model_dump()
        h2 = HouseholdState.model_validate(data)
        assert h2.id == household.id
        assert h2.wealth == household.wealth

    def test_entrepreneur_with_firm(
        self, utility_params: UtilityParams, value_vector: ValueVector
    ) -> None:
        h = HouseholdState(
            id="agent_0001",
            wealth=200.0,
            productivity=2.0,
            productivity_index=3,
            utility_params=utility_params,
            value_vector=value_vector,
            role=OccupationalRole.ENTREPRENEUR,
            firm_id="firm_0000",
        )
        assert h.role == OccupationalRole.ENTREPRENEUR
        assert h.firm_id == "firm_0000"


# --- FirmState ---


class TestFirmState:
    def test_valid_firm(self, firm: FirmState) -> None:
        assert firm.id == "firm_0000"
        assert firm.capital == 50.0
        assert firm.worker_ids == []
        assert firm.rd_spend == 0.0

    def test_negative_capital_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FirmState(
                id="firm_bad",
                owner_id="agent_0000",
                capital=-10.0,
                labor_demand=5.0,
                tfp=1.0,
            )

    def test_zero_tfp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FirmState(
                id="firm_bad",
                owner_id="agent_0000",
                capital=10.0,
                labor_demand=5.0,
                tfp=0.0,
            )

    def test_serialization_round_trip(self, firm: FirmState) -> None:
        data = firm.model_dump()
        f2 = FirmState.model_validate(data)
        assert f2 == firm


# --- MarketState ---


class TestMarketState:
    def test_defaults(self, market: MarketState) -> None:
        assert market.wage == 5.0
        assert market.interest_rate == 0.05
        assert market.aggregate_output == 0.0
        assert market.market_clearing_error == 0.0

    def test_serialization(self, market: MarketState) -> None:
        data = market.model_dump()
        m2 = MarketState.model_validate(data)
        assert m2 == market


# --- ShockState ---


class TestShockState:
    def test_defaults(self) -> None:
        s = ShockState()
        assert s.productivity_grid == []
        assert s.transition_matrix == []
        assert s.aggregate_tfp == 1.0
        assert s.preference_shocks == {}

    def test_valid_shocks(self, shocks: ShockState) -> None:
        assert len(shocks.productivity_grid) == 3
        assert len(shocks.transition_matrix) == 3
        assert shocks.aggregate_tfp == 1.0

    def test_negative_tfp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShockState(aggregate_tfp=-0.5)


# --- ConstitutionV2 ---


class TestConstitutionV2:
    def test_default_constitution_has_all_rules(self, constitution: ConstitutionV2) -> None:
        assert "flat_tax" in constitution.rules
        assert "flat_transfer" in constitution.rules
        assert "public_goods_provision" in constitution.rules
        assert "majority_vote" in constitution.rules
        assert "private_property" in constitution.rules

    def test_get_tax_rules(self, constitution: ConstitutionV2) -> None:
        tax_rules = constitution.get_tax_rules()
        assert len(tax_rules) == 1
        assert tax_rules[0].name == "flat_tax"

    def test_get_transfer_rules(self, constitution: ConstitutionV2) -> None:
        transfer_rules = constitution.get_transfer_rules()
        assert len(transfer_rules) == 1
        assert transfer_rules[0].name == "flat_transfer"

    def test_get_active_voting_rule(self, constitution: ConstitutionV2) -> None:
        voting_rule = constitution.get_active_voting_rule()
        assert voting_rule is not None
        assert voting_rule.name == "majority_vote"
        assert voting_rule.parameters["threshold"] == 0.5

    def test_missing_voting_rule_returns_none(self) -> None:
        c = ConstitutionV2(rules={}, voting_rule="nonexistent")
        assert c.get_active_voting_rule() is None

    def test_empty_constitution(self) -> None:
        c = ConstitutionV2()
        assert c.rules == {}
        assert c.get_tax_rules() == []
        assert c.get_transfer_rules() == []

    def test_serialization_round_trip(self, constitution: ConstitutionV2) -> None:
        data = constitution.model_dump()
        c2 = ConstitutionV2.model_validate(data)
        assert c2.rules.keys() == constitution.rules.keys()

    def test_rule_type_enum(self) -> None:
        assert RuleType.TAX_SCHEDULE == "tax_schedule"
        assert RuleType.CUSTOM == "custom"

    def test_constitutional_rule_defaults(self) -> None:
        rule = ConstitutionalRule(name="test", rule_type=RuleType.CUSTOM)
        assert rule.version == 1
        assert rule.enacted_period == 0
        assert rule.parameters == {}


# --- ConstitutionalProposal ---


class TestConstitutionalProposal:
    def test_add_proposal(self) -> None:
        p = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="progressive_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"brackets": [0.1, 0.2, 0.3]},
            description="Progressive tax with three brackets",
        )
        assert p.action == "add"
        assert p.rule_type == RuleType.TAX_SCHEDULE

    def test_modify_proposal(self) -> None:
        p = ConstitutionalProposal(
            proposer_id="agent_0001",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.15},
        )
        assert p.action == "modify"
        assert p.rule_type is None

    def test_remove_proposal(self) -> None:
        p = ConstitutionalProposal(
            proposer_id="agent_0002",
            action="remove",
            rule_name="flat_transfer",
        )
        assert p.action == "remove"

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConstitutionalProposal(
                proposer_id="agent_0000",
                action="invalid",
                rule_name="test",
            )


class TestVoteOutcomeV2:
    def test_valid_outcome(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.1},
        )
        outcome = VoteOutcomeV2(
            proposal=proposal,
            passed=True,
            votes_for=30,
            votes_against=20,
            total_eligible=50,
            voting_rule_used="majority_vote",
        )
        assert outcome.passed is True
        assert outcome.votes_for == 30

    def test_negative_votes_rejected(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="flat_tax",
        )
        with pytest.raises(ValidationError):
            VoteOutcomeV2(
                proposal=proposal,
                passed=False,
                votes_for=-1,
                votes_against=10,
                total_eligible=50,
                voting_rule_used="majority_vote",
            )


# --- Decisions ---


class TestEconomicDecision:
    def test_valid(self) -> None:
        d = EconomicDecision(consumption=50.0, leisure=0.3)
        assert d.consumption == 50.0
        assert d.leisure == 0.3

    def test_negative_consumption_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EconomicDecision(consumption=-1.0, leisure=0.5)

    def test_leisure_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            EconomicDecision(consumption=10.0, leisure=1.5)


class TestEntrepreneurialDecision:
    def test_defaults(self) -> None:
        d = EntrepreneurialDecision()
        assert d.create_firm is False
        assert d.capital_investment == 0.0
        assert d.close_firm is False

    def test_create_firm_decision(self) -> None:
        d = EntrepreneurialDecision(
            create_firm=True,
            capital_investment=100.0,
            labor_demand=5.0,
            rd_spend=10.0,
        )
        assert d.create_firm is True
        assert d.capital_investment == 100.0


class TestPoliticalDecision:
    def test_defaults(self) -> None:
        d = PoliticalDecision()
        assert d.proposal is None
        assert d.votes == {}

    def test_with_proposal_and_votes(self) -> None:
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="min_wage",
            rule_type=RuleType.MARKET_REGULATION,
        )
        d = PoliticalDecision(
            proposal=proposal,
            votes={"flat_tax_increase": True, "remove_transfer": False},
        )
        assert d.proposal is not None
        assert d.votes["flat_tax_increase"] is True


# --- HistoryEntryV2 ---


class TestHistoryEntryV2:
    def test_valid_entry(self, constitution: ConstitutionV2) -> None:
        entry = HistoryEntryV2(
            period=5,
            gini=0.35,
            pareto_score=0.8,
            aggregate_output=5000.0,
            mean_wealth=100.0,
            median_wealth=80.0,
            wealth_quantiles=[20.0, 50.0, 80.0, 150.0, 250.0],
            constitution_snapshot=constitution,
        )
        assert entry.period == 5
        assert entry.gini == 0.35
        assert len(entry.wealth_quantiles) == 5

    def test_defaults(self, constitution: ConstitutionV2) -> None:
        entry = HistoryEntryV2(
            period=0,
            gini=0.0,
            aggregate_output=0.0,
            mean_wealth=100.0,
            median_wealth=100.0,
            constitution_snapshot=constitution,
        )
        assert entry.unemployment_rate == 0.0
        assert entry.num_active_firms == 0
        assert entry.rule_changes == []


# --- PeriodState ---


class TestPeriodState:
    def test_valid(
        self,
        household: HouseholdState,
        market: MarketState,
        shocks: ShockState,
        constitution: ConstitutionV2,
    ) -> None:
        ps = PeriodState(
            period=0,
            households=[household],
            market=market,
            shocks=shocks,
            constitution=constitution,
        )
        assert ps.period == 0
        assert len(ps.households) == 1
        assert ps.firms == []
        assert ps.proposals == []
        assert ps.votes == []


# --- WelfareSummary ---


class TestWelfareSummary:
    def test_llm_only(self) -> None:
        w = WelfareSummary(llm_total_welfare=1000.0)
        assert w.benchmark_total_welfare is None

    def test_with_benchmark(self) -> None:
        w = WelfareSummary(
            llm_total_welfare=1000.0,
            benchmark_total_welfare=950.0,
        )
        assert w.benchmark_total_welfare == 950.0


# --- SimulationOutputV2 ---


class TestSimulationOutputV2:
    def test_valid(
        self,
        household: HouseholdState,
        constitution: ConstitutionV2,
    ) -> None:
        output = SimulationOutputV2(
            constitution=constitution,
            history=[],
            final_households=[household],
            welfare_summary=WelfareSummary(llm_total_welfare=500.0),
            seed=42,
            total_periods=100,
        )
        assert output.seed == 42
        assert output.total_periods == 100
        assert len(output.final_households) == 1
        assert output.final_firms == []

    def test_serialization_round_trip(
        self,
        household: HouseholdState,
        constitution: ConstitutionV2,
    ) -> None:
        output = SimulationOutputV2(
            constitution=constitution,
            history=[],
            final_households=[household],
            welfare_summary=WelfareSummary(llm_total_welfare=500.0),
            seed=42,
            total_periods=100,
        )
        data = output.model_dump()
        o2 = SimulationOutputV2.model_validate(data)
        assert o2.seed == output.seed
        assert len(o2.final_households) == 1
