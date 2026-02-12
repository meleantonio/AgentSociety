"""Tests for constitution enforcement engine (Task 8)."""

from __future__ import annotations

import pytest

from emergent_constitution.constitution_engine import (
    ConstitutionEngine,
    SandboxError,
    evaluate_enforcement_code,
)
from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import (
    HouseholdState,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    income: float = 50.0,
    labor_supply: float = 0.7,
) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=1.0,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        income=income,
        labor_supply=labor_supply,
    )


@pytest.fixture
def engine() -> ConstitutionEngine:
    return ConstitutionEngine()


@pytest.fixture
def constitution() -> ConstitutionV2:
    return create_default_constitution()


@pytest.fixture
def market() -> MarketState:
    return MarketState(wage=1.0, interest_rate=0.05)


@pytest.fixture
def households() -> list[HouseholdState]:
    return [
        _make_household("agent_0000", wealth=100.0, income=50.0),
        _make_household("agent_0001", wealth=200.0, income=80.0),
        _make_household("agent_0002", wealth=50.0, income=30.0),
    ]


# ---------------------------------------------------------------------------
# Sandbox Evaluator Tests
# ---------------------------------------------------------------------------


class TestSandboxEvaluator:
    """Test the restricted sandbox evaluator."""

    def test_simple_arithmetic(self) -> None:
        result = evaluate_enforcement_code("income * rate", {"income": 100.0, "rate": 0.2})
        assert result == 20.0

    def test_comparison(self) -> None:
        result = evaluate_enforcement_code("income > 50", {"income": 100.0})
        assert result is True

    def test_conditional(self) -> None:
        result = evaluate_enforcement_code(
            "income * 0.3 if income > 50 else income * 0.1",
            {"income": 100.0},
        )
        assert result == 30.0

    def test_safe_builtins(self) -> None:
        result = evaluate_enforcement_code("max(income, 0)", {"income": -5.0})
        assert result == 0.0

    def test_safe_math(self) -> None:
        result = evaluate_enforcement_code("sqrt(wealth)", {"wealth": 25.0})
        assert result == 5.0

    def test_empty_code_returns_none(self) -> None:
        assert evaluate_enforcement_code("", {}) is None
        assert evaluate_enforcement_code("  ", {}) is None

    def test_rejects_import(self) -> None:
        # __import__ is not in safe globals, so it fails at runtime
        with pytest.raises(SandboxError, match="execution failed"):
            evaluate_enforcement_code("__import__('os')", {})

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(SandboxError, match="Disallowed AST node"):
            evaluate_enforcement_code("income.__class__", {"income": 1.0})

    def test_rejects_dangerous_function_call(self) -> None:
        # Function calls are allowed for safe builtins, but dangerous
        # functions like eval are not in the safe globals
        with pytest.raises(SandboxError, match="execution failed"):
            evaluate_enforcement_code("eval('1+1')", {})

    def test_rejects_function_definition(self) -> None:
        # Function definitions are parsed in exec mode but rejected as FunctionDef
        with pytest.raises(SandboxError, match="Disallowed AST node"):
            evaluate_enforcement_code("def foo(): pass", {})

    def test_rejects_true_syntax_error(self) -> None:
        with pytest.raises(SandboxError, match="Syntax error"):
            evaluate_enforcement_code("if if if", {})

    def test_execution_error_wrapped(self) -> None:
        with pytest.raises(SandboxError, match="execution failed"):
            evaluate_enforcement_code("income / 0", {"income": 1.0})


# ---------------------------------------------------------------------------
# Rule Validation Tests (REQ-023, PROP-006)
# ---------------------------------------------------------------------------


class TestValidateRule:
    """Test rule validation for internal consistency."""

    def test_valid_tax_rule(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="income_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.2},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid
        assert reason == "OK"

    def test_invalid_tax_rate_negative(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="bad_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": -0.1},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid
        assert "rate" in reason.lower()

    def test_invalid_tax_rate_over_100(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="bad_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 1.5},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid

    def test_valid_transfer_rule(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="transfers",
            rule_type=RuleType.TRANSFER_PROGRAM,
            parameters={"method": "equal_share"},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid

    def test_invalid_transfer_method(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="bad_transfer",
            rule_type=RuleType.TRANSFER_PROGRAM,
            parameters={"method": "random_lottery"},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid

    def test_valid_public_goods_rule(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="public_goods",
            rule_type=RuleType.PUBLIC_GOODS,
            parameters={"fraction_of_revenue": 0.3},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid

    def test_invalid_public_goods_fraction(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="bad_pg",
            rule_type=RuleType.PUBLIC_GOODS,
            parameters={"fraction_of_revenue": 2.0},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid

    def test_valid_voting_rule(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="supermajority",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.67},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid

    def test_invalid_voting_threshold(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="bad_voting",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 1.5},
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid

    def test_unsafe_enforcement_code_rejected(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="dangerous",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.1},
            enforcement_code="__import__('os').system('rm -rf /')",
        )
        is_valid, reason = engine.validate_rule(rule)
        assert not is_valid
        assert "unsafe" in reason.lower()

    def test_safe_enforcement_code_accepted(self, engine: ConstitutionEngine) -> None:
        rule = ConstitutionalRule(
            name="progressive",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.2},
            enforcement_code="income * rate",
        )
        is_valid, reason = engine.validate_rule(rule)
        assert is_valid


# ---------------------------------------------------------------------------
# Tax Enforcement Tests (REQ-022)
# ---------------------------------------------------------------------------


class TestEnforceTaxes:
    """Test tax enforcement."""

    def test_flat_tax(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
        market: MarketState,
    ) -> None:
        constitution = ConstitutionV2(
            rules={
                "flat_tax": ConstitutionalRule(
                    name="flat_tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.2},
                    enforcement_code="income * rate",
                ),
            }
        )

        updated, revenue = engine.enforce_taxes(households, constitution, market)
        assert len(updated) == 3
        # 20% of incomes: 50*0.2=10, 80*0.2=16, 30*0.2=6
        assert abs(revenue - 32.0) < 1e-6
        assert abs(updated[0].taxes_paid - 10.0) < 1e-6
        assert abs(updated[1].taxes_paid - 16.0) < 1e-6
        assert abs(updated[2].taxes_paid - 6.0) < 1e-6

    def test_zero_tax_rate(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
        market: MarketState,
    ) -> None:
        constitution = ConstitutionV2(
            rules={
                "no_tax": ConstitutionalRule(
                    name="no_tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.0},
                ),
            }
        )
        updated, revenue = engine.enforce_taxes(households, constitution, market)
        assert revenue == 0.0
        for h in updated:
            assert h.taxes_paid == 0.0

    def test_no_tax_rules(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
        market: MarketState,
    ) -> None:
        constitution = ConstitutionV2(rules={})
        updated, revenue = engine.enforce_taxes(households, constitution, market)
        assert revenue == 0.0
        assert updated is households  # No copy made when no rules

    def test_tax_cannot_exceed_income(
        self,
        engine: ConstitutionEngine,
        market: MarketState,
    ) -> None:
        """Tax is capped at income (non-negativity)."""
        households = [_make_household(income=10.0)]
        constitution = ConstitutionV2(
            rules={
                "high_tax": ConstitutionalRule(
                    name="high_tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.99},
                    enforcement_code="income * rate",
                ),
            }
        )
        updated, revenue = engine.enforce_taxes(households, constitution, market)
        assert updated[0].taxes_paid <= 10.0
        # enforce_taxes no longer modifies wealth; it only records taxes_paid
        assert updated[0].wealth == 100.0

    def test_original_households_not_mutated(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
        market: MarketState,
    ) -> None:
        constitution = ConstitutionV2(
            rules={
                "tax": ConstitutionalRule(
                    name="tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.2},
                ),
            }
        )
        original_wealth = [h.wealth for h in households]
        engine.enforce_taxes(households, constitution, market)
        for h, w in zip(households, original_wealth, strict=True):
            assert h.wealth == w


# ---------------------------------------------------------------------------
# Transfer Enforcement Tests
# ---------------------------------------------------------------------------


class TestEnforceTransfers:
    """Test transfer distribution."""

    def test_equal_share(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
    ) -> None:
        constitution = ConstitutionV2(
            rules={
                "equal": ConstitutionalRule(
                    name="equal",
                    rule_type=RuleType.TRANSFER_PROGRAM,
                    parameters={"method": "equal_share"},
                ),
            }
        )
        updated = engine.enforce_transfers(households, constitution, revenue=30.0)
        for h in updated:
            assert abs(h.transfers_received - 10.0) < 1e-6

    def test_means_tested(
        self,
        engine: ConstitutionEngine,
    ) -> None:
        """Poorer agents should receive more in means-tested transfers."""
        households = [
            _make_household("poor", wealth=10.0),
            _make_household("rich", wealth=1000.0),
        ]
        constitution = ConstitutionV2(
            rules={
                "means": ConstitutionalRule(
                    name="means",
                    rule_type=RuleType.TRANSFER_PROGRAM,
                    parameters={"method": "means_tested"},
                ),
            }
        )
        updated = engine.enforce_transfers(households, constitution, revenue=100.0)
        # Poor agent (wealth=10) should get more than rich (wealth=1000)
        assert updated[0].transfers_received > updated[1].transfers_received

    def test_zero_revenue_no_transfers(
        self,
        engine: ConstitutionEngine,
        households: list[HouseholdState],
    ) -> None:
        constitution = create_default_constitution()
        updated = engine.enforce_transfers(households, constitution, revenue=0.0)
        assert updated is households  # No copy when nothing to distribute

    def test_budget_constraint(
        self,
        engine: ConstitutionEngine,
    ) -> None:
        """Total transfers should not exceed revenue."""
        households = [_make_household(f"agent_{i}", wealth=50.0) for i in range(5)]
        constitution = ConstitutionV2(
            rules={
                "t1": ConstitutionalRule(
                    name="t1",
                    rule_type=RuleType.TRANSFER_PROGRAM,
                    parameters={"method": "equal_share"},
                ),
            }
        )
        revenue = 100.0
        updated = engine.enforce_transfers(households, constitution, revenue=revenue)
        total_transferred = sum(h.transfers_received for h in updated)
        assert total_transferred <= revenue + 1e-9


# ---------------------------------------------------------------------------
# Public Goods Tests
# ---------------------------------------------------------------------------


class TestEnforcePublicGoods:
    """Test public goods computation."""

    def test_basic_public_goods(self, engine: ConstitutionEngine) -> None:
        constitution = ConstitutionV2(
            rules={
                "pg": ConstitutionalRule(
                    name="pg",
                    rule_type=RuleType.PUBLIC_GOODS,
                    parameters={"fraction_of_revenue": 0.3},
                ),
            }
        )
        g = engine.enforce_public_goods(constitution, revenue=100.0, num_agents=10)
        assert abs(g - 3.0) < 1e-6  # 0.3 * 100 / 10 = 3.0

    def test_no_public_goods_rules(self, engine: ConstitutionEngine) -> None:
        constitution = ConstitutionV2(rules={})
        g = engine.enforce_public_goods(constitution, revenue=100.0, num_agents=10)
        assert g == 0.0

    def test_zero_revenue(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        g = engine.enforce_public_goods(constitution, revenue=0.0, num_agents=10)
        assert g == 0.0

    def test_zero_agents(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        g = engine.enforce_public_goods(constitution, revenue=100.0, num_agents=0)
        assert g == 0.0


# ---------------------------------------------------------------------------
# Regulation Enforcement Tests
# ---------------------------------------------------------------------------


class TestEnforceRegulations:
    """Test market and firm regulation enforcement."""

    def test_no_regulations(self, engine: ConstitutionEngine) -> None:
        firms = [FirmState(id="f1", owner_id="a1", capital=100.0, labor_demand=2.0, tfp=1.0)]
        households = [_make_household()]
        constitution = ConstitutionV2(rules={})
        result_firms, result_hh = engine.enforce_regulations(firms, households, constitution)
        assert result_firms is firms
        assert result_hh is households

    def test_firm_regulation_applied(self, engine: ConstitutionEngine) -> None:
        constitution = ConstitutionV2(
            rules={
                "max_size": ConstitutionalRule(
                    name="max_size",
                    rule_type=RuleType.FIRM_REGULATION,
                    parameters={"max_workers": 5},
                ),
            }
        )
        firms = [
            FirmState(
                id="f1",
                owner_id="a1",
                capital=100.0,
                labor_demand=2.0,
                tfp=1.0,
                worker_ids=[f"w{i}" for i in range(10)],
            )
        ]
        households = [_make_household()]
        result_firms, _ = engine.enforce_regulations(firms, households, constitution)
        # Firms should be returned (regulation is currently informational)
        assert len(result_firms) == 1


# ---------------------------------------------------------------------------
# Proposal Application Tests (REQ-021)
# ---------------------------------------------------------------------------


class TestApplyProposal:
    """Test proposal application (add/modify/remove)."""

    def test_add_rule(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="luxury_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.1},
            description="Tax on luxury goods.",
        )
        updated = engine.apply_proposal(constitution, proposal)
        assert "luxury_tax" in updated.rules
        assert updated.rules["luxury_tax"].rule_type == RuleType.TAX_SCHEDULE
        assert updated.rules["luxury_tax"].parameters["rate"] == 0.1

    def test_add_duplicate_raises(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="flat_tax",  # Already exists
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 0.5},
        )
        with pytest.raises(ValueError, match="already exists"):
            engine.apply_proposal(constitution, proposal)

    def test_add_without_type_raises(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="new_rule",
        )
        with pytest.raises(ValueError, match="rule_type is required"):
            engine.apply_proposal(constitution, proposal)

    def test_modify_rule(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="flat_tax",
            parameters={"rate": 0.3},
            description="Increase flat tax to 30%.",
        )
        updated = engine.apply_proposal(constitution, proposal)
        assert updated.rules["flat_tax"].parameters["rate"] == 0.3
        assert updated.rules["flat_tax"].version == 2  # Incremented

    def test_modify_nonexistent_raises(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="modify",
            rule_name="nonexistent",
            parameters={"rate": 0.5},
        )
        with pytest.raises(ValueError, match="does not exist"):
            engine.apply_proposal(constitution, proposal)

    def test_remove_rule(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        assert "flat_tax" in constitution.rules
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="remove",
            rule_name="flat_tax",
        )
        updated = engine.apply_proposal(constitution, proposal)
        assert "flat_tax" not in updated.rules

    def test_remove_nonexistent_raises(self, engine: ConstitutionEngine) -> None:
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="remove",
            rule_name="nonexistent",
        )
        with pytest.raises(ValueError, match="does not exist"):
            engine.apply_proposal(constitution, proposal)

    def test_remove_voting_rule_raises(self, engine: ConstitutionEngine) -> None:
        """Cannot remove the active voting rule."""
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="remove",
            rule_name="majority_vote",
        )
        with pytest.raises(ValueError, match="active voting rule"):
            engine.apply_proposal(constitution, proposal)

    def test_add_invalid_rule_raises(self, engine: ConstitutionEngine) -> None:
        """Adding a rule that fails validation should raise."""
        constitution = create_default_constitution()
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="bad_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": 5.0},  # Invalid: > 1.0
        )
        with pytest.raises(ValueError, match="validation failed"):
            engine.apply_proposal(constitution, proposal)

    def test_original_constitution_not_mutated(self, engine: ConstitutionEngine) -> None:
        """apply_proposal should not mutate the original."""
        constitution = create_default_constitution()
        original_rules = set(constitution.rules.keys())
        proposal = ConstitutionalProposal(
            proposer_id="agent_0000",
            action="add",
            rule_name="new_rule",
            rule_type=RuleType.CUSTOM,
            parameters={},
        )
        engine.apply_proposal(constitution, proposal)
        assert set(constitution.rules.keys()) == original_rules


# ---------------------------------------------------------------------------
# Integration: Full Enforcement Cycle
# ---------------------------------------------------------------------------


class TestFullEnforcementCycle:
    """Test the full tax -> transfer -> public goods cycle."""

    def test_tax_then_transfer(self, engine: ConstitutionEngine) -> None:
        households = [
            _make_household("a0", wealth=100.0, income=50.0),
            _make_household("a1", wealth=200.0, income=100.0),
        ]
        market = MarketState(wage=1.0, interest_rate=0.05)

        constitution = ConstitutionV2(
            rules={
                "tax": ConstitutionalRule(
                    name="tax",
                    rule_type=RuleType.TAX_SCHEDULE,
                    parameters={"rate": 0.2},
                    enforcement_code="income * rate",
                ),
                "transfer": ConstitutionalRule(
                    name="transfer",
                    rule_type=RuleType.TRANSFER_PROGRAM,
                    parameters={"method": "equal_share"},
                ),
            }
        )

        # Tax
        taxed, revenue = engine.enforce_taxes(households, constitution, market)
        # Revenue: 50*0.2 + 100*0.2 = 10 + 20 = 30
        assert abs(revenue - 30.0) < 1e-6

        # Transfer
        transferred = engine.enforce_transfers(taxed, constitution, revenue)
        # 30 / 2 = 15 each
        for h in transferred:
            assert abs(h.transfers_received - 15.0) < 1e-6

    def test_public_goods_from_revenue(self, engine: ConstitutionEngine) -> None:
        constitution = ConstitutionV2(
            rules={
                "pg": ConstitutionalRule(
                    name="pg",
                    rule_type=RuleType.PUBLIC_GOODS,
                    parameters={"fraction_of_revenue": 0.3},
                ),
            }
        )
        revenue = 100.0
        g = engine.enforce_public_goods(constitution, revenue, num_agents=5)
        # 0.3 * 100 / 5 = 6.0
        assert abs(g - 6.0) < 1e-6
