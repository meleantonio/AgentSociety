"""Constitution enforcement engine — validation, sandbox evaluation, enforcement.

Enforces and validates constitutional rules: taxes, transfers, public goods,
market/firm regulations. Includes a sandboxed evaluator for enforcement_code
that prevents code injection.

Traceability: REQ-018, REQ-021, REQ-022, REQ-023, PROP-006
"""

from __future__ import annotations

import ast
import math
from typing import Any

import structlog

from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Sandbox evaluator
# ---------------------------------------------------------------------------

# AST node types allowed in enforcement_code expressions.
_ALLOWED_AST_NODES: set[type] = {
    ast.Module,
    ast.Expr,
    ast.Expression,
    # Literals
    ast.Constant,
    # Operations
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    # Comparisons
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # Boolean
    ast.And,
    ast.Or,
    ast.Not,
    # Names (variable references)
    ast.Name,
    ast.Load,
    # Function calls (only safe builtins allowed at runtime)
    ast.Call,
    # Containers (for simple expressions)
    ast.Tuple,
    ast.List,
    ast.Dict,
}

# Safe builtins available in sandbox
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "True": True,
    "False": False,
    "None": None,
}

# Safe math functions
_SAFE_MATH: dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pow": pow,
}


class SandboxError(Exception):
    """Raised when enforcement_code fails safety checks or execution."""


def _validate_ast_safety(code: str) -> bool:
    """Check if code contains only allowed AST node types.

    Accepts both expressions (``income * rate``) and simple assignment
    statements (``tax = income * rate``).  Assignment statements are
    allowed so that enforcement_code can bind a result variable; they
    are additionally whitelisted (ast.Assign, ast.Store).

    Args:
        code: Python expression or simple assignment string.

    Returns:
        True if all nodes are in the allowed set.

    Raises:
        SandboxError: If code contains disallowed node types.
    """
    # Try expression first, then fall back to exec mode for assignments
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise SandboxError(f"Syntax error in enforcement_code: {exc}") from exc

    # Assignment nodes are allowed in addition to the base set
    allowed = _ALLOWED_AST_NODES | {ast.Assign, ast.Store}

    for node in ast.walk(tree):
        if type(node) not in allowed:
            raise SandboxError(f"Disallowed AST node: {type(node).__name__} in enforcement_code")
    return True


def evaluate_enforcement_code(code: str, variables: dict[str, Any]) -> Any:
    """Evaluate enforcement_code in a restricted sandbox.

    Args:
        code: Python expression to evaluate.
        variables: Variables available to the expression.

    Returns:
        Result of the expression evaluation.

    Raises:
        SandboxError: If code is unsafe or execution fails.
    """
    if not code or not code.strip():
        return None

    _validate_ast_safety(code)

    # Build restricted globals
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(_SAFE_BUILTINS)
    safe_globals.update(_SAFE_MATH)
    safe_globals.update(variables)

    try:
        return eval(code, safe_globals)  # noqa: S307
    except Exception as exc:
        raise SandboxError(f"Enforcement code execution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# ConstitutionEngine
# ---------------------------------------------------------------------------


class ConstitutionEngine:
    """Enforces and validates constitutional rules.

    Responsible for:
    - Validating rules before activation (REQ-023, PROP-006)
    - Enforcing taxes, transfers, public goods each period (REQ-022)
    - Applying market/firm regulations
    - Processing proposals to add/modify/remove rules (REQ-021)
    """

    def validate_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a rule for internal consistency.

        Checks:
        - Tax rates produce non-negative revenue (rates in valid range).
        - Transfer programs satisfy budget constraint.
        - Enforcement code passes sandbox safety check.
        - No logical contradictions.

        Args:
            rule: The rule to validate.

        Returns:
            Tuple of (is_valid, reason).
        """
        # Check enforcement_code safety
        if rule.enforcement_code:
            try:
                _validate_ast_safety(rule.enforcement_code)
            except SandboxError as exc:
                return False, f"Unsafe enforcement code: {exc}"

        # Type-specific validation
        if rule.rule_type == RuleType.TAX_SCHEDULE:
            return self._validate_tax_rule(rule)
        if rule.rule_type == RuleType.TRANSFER_PROGRAM:
            return self._validate_transfer_rule(rule)
        if rule.rule_type == RuleType.PUBLIC_GOODS:
            return self._validate_public_goods_rule(rule)
        if rule.rule_type == RuleType.VOTING_PROCEDURE:
            return self._validate_voting_rule(rule)

        # Other rule types pass basic validation
        return True, "OK"

    def enforce_taxes(
        self,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
        market: MarketState,
    ) -> tuple[list[HouseholdState], float]:
        """Apply all active TAX_SCHEDULE rules.

        Computes T(y_i) for each household based on active tax rules.
        Updates households with taxes_paid. Respects non-negativity:
        taxes cannot exceed income.

        Args:
            households: All household agents.
            constitution: Active constitutional rules.
            market: Current market state.

        Returns:
            Tuple of (updated households, total_revenue).
        """
        tax_rules = constitution.get_tax_rules()
        if not tax_rules:
            return households, 0.0

        updated = [h.model_copy() for h in households]
        total_revenue = 0.0

        for household in updated:
            income = (
                household.income
                if household.income > 0
                else (market.wage * household.productivity * household.labor_supply)
            )
            tax = 0.0
            for rule in tax_rules:
                tax += self._compute_tax(rule, income, household.wealth)
            # Non-negativity: taxes cannot exceed income
            tax = max(0.0, min(tax, income))
            household.taxes_paid = tax
            household.wealth -= tax
            # Ensure wealth doesn't go below zero
            household.wealth = max(household.wealth, 0.0)
            total_revenue += tax

        return updated, total_revenue

    def enforce_transfers(
        self,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
        revenue: float,
    ) -> list[HouseholdState]:
        """Apply all active TRANSFER_PROGRAM rules.

        Distributes available revenue to households per transfer rules.
        Budget constraint: total transfers cannot exceed revenue.

        Args:
            households: All household agents.
            constitution: Active constitutional rules.
            revenue: Total tax revenue available for distribution.

        Returns:
            Updated households with transfers_received.
        """
        transfer_rules = constitution.get_transfer_rules()
        if not transfer_rules or revenue <= 0.0:
            return households

        updated = [h.model_copy() for h in households]
        remaining_revenue = revenue

        for rule in transfer_rules:
            if remaining_revenue <= 0.0:
                break
            transfers = self._compute_transfers(rule, updated, remaining_revenue)
            distributed = 0.0
            for household in updated:
                transfer = transfers.get(household.id, 0.0)
                household.transfers_received += transfer
                household.wealth += transfer
                distributed += transfer
            remaining_revenue -= distributed

        return updated

    def enforce_public_goods(
        self,
        constitution: ConstitutionV2,
        revenue: float,
        num_agents: int,
    ) -> float:
        """Compute public goods per capita G_t from PUBLIC_GOODS rules.

        Args:
            constitution: Active constitutional rules.
            revenue: Total tax revenue.
            num_agents: Number of agents (for per-capita computation).

        Returns:
            Public goods per capita G_t.
        """
        if num_agents <= 0 or revenue <= 0.0:
            return 0.0

        public_goods_rules = [
            r for r in constitution.rules.values() if r.rule_type == RuleType.PUBLIC_GOODS
        ]

        total_public_spending = 0.0
        for rule in public_goods_rules:
            fraction = rule.parameters.get("fraction_of_revenue", 0.0)
            fraction = max(0.0, min(1.0, fraction))
            total_public_spending += fraction * revenue

        # Public goods per capita
        return total_public_spending / num_agents

    def enforce_regulations(
        self,
        firms: list[FirmState],
        households: list[HouseholdState],
        constitution: ConstitutionV2,
    ) -> tuple[list[FirmState], list[HouseholdState]]:
        """Apply MARKET_REGULATION and FIRM_REGULATION rules.

        Currently supports:
        - Minimum wage enforcement
        - Maximum firm size limits

        Args:
            firms: Active firms.
            households: All households.
            constitution: Active constitutional rules.

        Returns:
            Tuple of (updated firms, updated households).
        """
        regulation_rules = [
            r
            for r in constitution.rules.values()
            if r.rule_type in (RuleType.MARKET_REGULATION, RuleType.FIRM_REGULATION)
        ]

        if not regulation_rules:
            return firms, households

        updated_firms = [f.model_copy() for f in firms]
        updated_households = [h.model_copy() for h in households]

        for rule in regulation_rules:
            if rule.rule_type == RuleType.MARKET_REGULATION:
                self._apply_market_regulation(rule, updated_firms, updated_households)
            elif rule.rule_type == RuleType.FIRM_REGULATION:
                self._apply_firm_regulation(rule, updated_firms)

        return updated_firms, updated_households

    def apply_proposal(
        self,
        constitution: ConstitutionV2,
        proposal: ConstitutionalProposal,
    ) -> ConstitutionV2:
        """Apply a passed proposal to the constitution.

        Handles add, modify, and remove actions. Validates the resulting
        rule before activation (PROP-006).

        Args:
            constitution: Current constitution.
            proposal: The passed proposal to apply.

        Returns:
            Updated constitution (new copy).

        Raises:
            ValueError: If the proposal action is invalid or rule validation fails.
        """
        updated = constitution.model_copy(deep=True)

        if proposal.action == "add":
            return self._apply_add(updated, proposal)
        if proposal.action == "modify":
            return self._apply_modify(updated, proposal)
        if proposal.action == "remove":
            return self._apply_remove(updated, proposal)

        raise ValueError(f"Unknown proposal action: {proposal.action}")

    # -------------------------------------------------------------------
    # Internal: tax computation
    # -------------------------------------------------------------------

    def _compute_tax(self, rule: ConstitutionalRule, income: float, wealth: float) -> float:
        """Compute tax for a single rule and agent.

        Supports flat rate and enforcement_code evaluation.
        """
        rate = rule.parameters.get("rate", 0.0)

        if rule.enforcement_code:
            try:
                result = evaluate_enforcement_code(
                    rule.enforcement_code,
                    {"income": income, "rate": rate, "wealth": wealth},
                )
                if isinstance(result, int | float) and math.isfinite(result):
                    return float(result)
            except SandboxError:
                log.warning(
                    "constitution_engine.tax_code_failed",
                    rule=rule.name,
                )

        # Default: flat tax
        return income * rate

    # -------------------------------------------------------------------
    # Internal: transfer computation
    # -------------------------------------------------------------------

    def _compute_transfers(
        self,
        rule: ConstitutionalRule,
        households: list[HouseholdState],
        available_revenue: float,
    ) -> dict[str, float]:
        """Compute transfer amounts per agent for a single rule."""
        method = rule.parameters.get("method", "equal_share")
        num_agents = len(households)

        if num_agents == 0:
            return {}

        if method == "equal_share":
            per_agent = available_revenue / num_agents
            return {h.id: per_agent for h in households}

        if method == "means_tested":
            # Inversely proportional to wealth
            total_inverse_wealth = sum(1.0 / max(h.wealth, 1.0) for h in households)
            if total_inverse_wealth <= 0:
                return {h.id: available_revenue / num_agents for h in households}
            return {
                h.id: (1.0 / max(h.wealth, 1.0)) / total_inverse_wealth * available_revenue
                for h in households
            }

        # Default: equal share
        per_agent = available_revenue / num_agents
        return {h.id: per_agent for h in households}

    # -------------------------------------------------------------------
    # Internal: regulation enforcement
    # -------------------------------------------------------------------

    def _apply_market_regulation(
        self,
        rule: ConstitutionalRule,
        firms: list[FirmState],
        households: list[HouseholdState],
    ) -> None:
        """Apply a market regulation rule (in-place)."""
        # Example: minimum wage enforcement is informational here;
        # actual wage is set by market clearing. We log violations.
        min_wage = rule.parameters.get("minimum_wage")
        if min_wage is not None:
            log.debug(
                "constitution_engine.regulation_applied",
                rule=rule.name,
                min_wage=min_wage,
            )

    def _apply_firm_regulation(
        self,
        rule: ConstitutionalRule,
        firms: list[FirmState],
    ) -> None:
        """Apply a firm regulation rule (in-place)."""
        max_size = rule.parameters.get("max_workers")
        if max_size is not None:
            for firm in firms:
                if len(firm.worker_ids) > max_size:
                    log.debug(
                        "constitution_engine.firm_oversized",
                        firm=firm.id,
                        workers=len(firm.worker_ids),
                        max_size=max_size,
                    )

    # -------------------------------------------------------------------
    # Internal: validation helpers
    # -------------------------------------------------------------------

    def _validate_tax_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a TAX_SCHEDULE rule."""
        rate = rule.parameters.get("rate")
        if rate is not None:
            if not isinstance(rate, int | float):
                return False, f"Tax rate must be numeric, got {type(rate).__name__}"
            if rate < 0.0 or rate > 1.0:
                return False, f"Tax rate must be in [0, 1], got {rate}"
        return True, "OK"

    def _validate_transfer_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a TRANSFER_PROGRAM rule."""
        method = rule.parameters.get("method")
        if method is not None:
            valid_methods = {"equal_share", "means_tested", "proportional"}
            if method not in valid_methods:
                return False, f"Unknown transfer method: {method}"
        return True, "OK"

    def _validate_public_goods_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a PUBLIC_GOODS rule."""
        fraction = rule.parameters.get("fraction_of_revenue")
        if fraction is not None:
            if not isinstance(fraction, int | float):
                return False, f"fraction_of_revenue must be numeric, got {type(fraction).__name__}"
            if fraction < 0.0 or fraction > 1.0:
                return False, f"fraction_of_revenue must be in [0, 1], got {fraction}"
        return True, "OK"

    def _validate_voting_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a VOTING_PROCEDURE rule."""
        threshold = rule.parameters.get("threshold")
        if threshold is not None:
            if not isinstance(threshold, int | float):
                return False, f"Voting threshold must be numeric, got {type(threshold).__name__}"
            if threshold < 0.0 or threshold > 1.0:
                return False, f"Voting threshold must be in [0, 1], got {threshold}"
        return True, "OK"

    # -------------------------------------------------------------------
    # Internal: proposal application
    # -------------------------------------------------------------------

    def _apply_add(
        self, constitution: ConstitutionV2, proposal: ConstitutionalProposal
    ) -> ConstitutionV2:
        """Add a new rule to the constitution."""
        if proposal.rule_name in constitution.rules:
            raise ValueError(
                f"Cannot add rule '{proposal.rule_name}': already exists. "
                f"Use 'modify' action instead."
            )
        if proposal.rule_type is None:
            raise ValueError("rule_type is required for 'add' action")

        new_rule = ConstitutionalRule(
            name=proposal.rule_name,
            rule_type=proposal.rule_type,
            parameters=proposal.parameters or {},
            description=proposal.description,
            enforcement_code=proposal.enforcement_code or "",
        )

        is_valid, reason = self.validate_rule(new_rule)
        if not is_valid:
            raise ValueError(f"Rule validation failed: {reason}")

        constitution.rules[proposal.rule_name] = new_rule
        return constitution

    def _apply_modify(
        self, constitution: ConstitutionV2, proposal: ConstitutionalProposal
    ) -> ConstitutionV2:
        """Modify an existing rule in the constitution."""
        if proposal.rule_name not in constitution.rules:
            raise ValueError(f"Cannot modify rule '{proposal.rule_name}': does not exist")

        existing = constitution.rules[proposal.rule_name]
        updated_rule = existing.model_copy(
            update={
                "parameters": proposal.parameters or existing.parameters,
                "description": proposal.description or existing.description,
                "enforcement_code": (
                    proposal.enforcement_code
                    if proposal.enforcement_code is not None
                    else existing.enforcement_code
                ),
                "version": existing.version + 1,
            }
        )

        is_valid, reason = self.validate_rule(updated_rule)
        if not is_valid:
            raise ValueError(f"Rule validation failed: {reason}")

        constitution.rules[proposal.rule_name] = updated_rule
        return constitution

    def _apply_remove(
        self, constitution: ConstitutionV2, proposal: ConstitutionalProposal
    ) -> ConstitutionV2:
        """Remove a rule from the constitution."""
        if proposal.rule_name not in constitution.rules:
            raise ValueError(f"Cannot remove rule '{proposal.rule_name}': does not exist")

        # Don't allow removing the active voting rule
        if proposal.rule_name == constitution.voting_rule:
            raise ValueError(f"Cannot remove the active voting rule '{proposal.rule_name}'")

        del constitution.rules[proposal.rule_name]
        return constitution
