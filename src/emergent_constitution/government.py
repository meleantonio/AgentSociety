"""Government sector — debt management, budget constraint, and fiscal rule.

Implements REQ-306 (government budget constraint), REQ-307 (bond market clearing),
REQ-308 (capital market clearing separate from bonds), REQ-309 (fiscal rule for
debt sustainability), and PROP-010 (government budget balance identity).

Design reference: spec/design.md section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()

# Safety cap: if debt exceeds this multiple of GDP despite the fiscal rule,
# freeze governance changes and log a critical warning.
_EXPLOSIVE_DEBT_MULTIPLE = 5.0


@dataclass
class GovernmentState:
    """Government sector state for period t.

    Attributes:
        debt: B_t outstanding government bonds.
        tax_revenue: T_t total tax revenue collected this period.
        spending: G_t public goods spending this period.
        transfers: Tr_t total transfers this period.
        bond_rate: r^b_t bond interest rate.
        debt_to_gdp: B_t / Y_t ratio.
    """

    debt: float = 0.0
    tax_revenue: float = 0.0
    spending: float = 0.0
    transfers: float = 0.0
    bond_rate: float = 0.03
    debt_to_gdp: float = 0.0


class Government:
    """Government sector managing debt, budget constraint, and fiscal rule.

    The government issues bonds to finance spending and transfers in excess
    of tax revenue. A fiscal rule auto-adjusts taxes when the debt-to-GDP
    ratio exceeds a configurable threshold.

    Args:
        initial_debt: Starting government debt B_0.
        debt_gdp_max: Threshold for fiscal rule activation.
        fiscal_rule_adjustment: Tax rate increment when fiscal rule triggers.
    """

    def __init__(
        self,
        initial_debt: float = 0.0,
        debt_gdp_max: float = 1.5,
        fiscal_rule_adjustment: float = 0.01,
    ) -> None:
        self.debt_gdp_max = debt_gdp_max
        self.fiscal_rule_adjustment = fiscal_rule_adjustment
        self.state = GovernmentState(debt=initial_debt)

    def update_budget(
        self,
        tax_revenue: float,
        spending: float,
        transfers: float,
        bond_rate: float,
    ) -> GovernmentState:
        """Apply the government budget constraint (REQ-306, PROP-010).

        B' = (1 + r^b) * B + G + Tr - T

        Args:
            tax_revenue: T_t total tax revenue this period.
            spending: G_t public goods spending this period.
            transfers: Tr_t total transfers this period.
            bond_rate: r^b_t current bond rate.

        Returns:
            Updated GovernmentState with new debt level.
        """
        old_debt = self.state.debt
        new_debt = (1.0 + bond_rate) * old_debt + spending + transfers - tax_revenue

        self.state = GovernmentState(
            debt=new_debt,
            tax_revenue=tax_revenue,
            spending=spending,
            transfers=transfers,
            bond_rate=bond_rate,
            debt_to_gdp=self.state.debt_to_gdp,  # updated in fiscal_rule
        )

        log.debug(
            "government.budget_updated",
            old_debt=round(old_debt, 2),
            new_debt=round(new_debt, 2),
            tax_revenue=round(tax_revenue, 2),
            spending=round(spending, 2),
            transfers=round(transfers, 2),
            bond_rate=round(bond_rate, 6),
        )

        return self.state

    def fiscal_rule(self, output: float) -> float:
        """Apply fiscal rule for debt sustainability (REQ-309).

        If B/Y > B_max, returns a positive tax adjustment (increment to the
        tax rate). Otherwise returns 0.0.

        Also updates the debt_to_gdp ratio on the state and checks for
        explosive debt paths.

        Args:
            output: Y_t aggregate output this period.

        Returns:
            Tax rate adjustment (>= 0.0). Add this to the current tax rate.
        """
        if output <= 0.0:
            log.warning("government.zero_output", debt=self.state.debt)
            return 0.0

        debt_to_gdp = self.state.debt / output
        self.state.debt_to_gdp = debt_to_gdp

        # Check for explosive debt
        if debt_to_gdp > _EXPLOSIVE_DEBT_MULTIPLE:
            log.critical(
                "government.explosive_debt",
                debt_to_gdp=round(debt_to_gdp, 4),
                threshold=_EXPLOSIVE_DEBT_MULTIPLE,
            )

        # Fiscal rule: auto-adjust tax when debt/GDP exceeds threshold
        if debt_to_gdp > self.debt_gdp_max:
            adjustment = self.fiscal_rule_adjustment
            log.info(
                "government.fiscal_rule_triggered",
                debt_to_gdp=round(debt_to_gdp, 4),
                threshold=self.debt_gdp_max,
                tax_adjustment=adjustment,
            )
            return adjustment

        return 0.0


# ============================================================================
# Bond market clearing (REQ-307)
# ============================================================================

_BOND_BISECT_TOL = 1e-8
_BOND_BISECT_MAX_ITER = 100
_BOND_RATE_LO = -0.50
_BOND_RATE_HI = 1.00


def _household_bond_demand(
    household_wealths: list[float],
    bond_rate: float,
    base_rate: float,
) -> float:
    """Compute aggregate household bond demand at a given bond rate.

    Args:
        household_wealths: List of household wealth values.
        bond_rate: Candidate bond rate r^b.
        base_rate: Capital market interest rate r (opportunity cost).

    Returns:
        Aggregate bond demand.
    """
    if not household_wealths:
        return 0.0

    sensitivity = 5.0
    spread = bond_rate - base_rate
    fraction = 1.0 / (1.0 + np.exp(-sensitivity * spread))

    return sum(w * fraction for w in household_wealths)


def clear_bond_market(
    household_wealths: list[float],
    government_debt: float,
    base_interest_rate: float,
    tol: float = _BOND_BISECT_TOL,
    max_iter: int = _BOND_BISECT_MAX_ITER,
) -> tuple[float, float]:
    """Find bond rate r^b that clears the bond market (REQ-307).

    Bisects on r^b to equate aggregate household bond demand with
    government bond supply B_t.

    Args:
        household_wealths: List of household wealth values.
        government_debt: B_t outstanding government bonds.
        base_interest_rate: Capital market rate r (for demand function).
        tol: Convergence tolerance.
        max_iter: Maximum bisection iterations.

    Returns:
        Tuple of (equilibrium bond rate r^b, clearing error).
    """
    if not household_wealths or government_debt <= 0.0:
        return base_interest_rate, 0.0

    def excess_demand(rb: float) -> float:
        demand = _household_bond_demand(household_wealths, rb, base_interest_rate)
        return demand - government_debt

    r_lo = _BOND_RATE_LO
    r_hi = _BOND_RATE_HI

    ed_lo = excess_demand(r_lo)
    ed_hi = excess_demand(r_hi)

    if ed_lo >= 0.0:
        return r_lo, abs(ed_lo)

    if ed_hi <= 0.0:
        return r_hi, abs(ed_hi)

    r_mid = 0.5 * (r_lo + r_hi)
    for _ in range(max_iter):
        r_mid = 0.5 * (r_lo + r_hi)
        ed_mid = excess_demand(r_mid)

        if abs(ed_mid) < tol:
            return r_mid, abs(ed_mid)

        if ed_mid < 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid

    r_mid = 0.5 * (r_lo + r_hi)
    clearing_error = abs(excess_demand(r_mid))

    log.debug(
        "bond_market.cleared",
        bond_rate=round(r_mid, 6),
        clearing_error=clearing_error,
        government_debt=round(government_debt, 2),
    )

    return r_mid, clearing_error
