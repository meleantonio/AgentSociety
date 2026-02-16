"""Market clearing — analytical and Walrasian equilibrium price finding.

Implements REQ-011 (labor market clearing), REQ-012 (capital market clearing),
REQ-013 (goods market identity Y = C + I + G), REQ-014 (iterative price
adjustment), PROP-003 (market clearing error < tolerance).

Walrasian clearing (REQ-101 through REQ-106) uses bisection on excess labor
demand with heterogeneous firms. Falls back to representative-firm analytical
clearing when no firms are present (REQ-105).

Design reference: spec/design.md section 1.1, section 3.3.
"""

from __future__ import annotations

import numpy as np
import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.economics import produce_output
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState, OccupationalRole
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()

# Minimum price floor to prevent division by zero or negative wages
_MIN_WAGE = 1e-6
_MIN_INTEREST = -0.99  # interest rate can be negative but bounded

# Walrasian bisection defaults
_BISECT_TOL = 1e-8
_BISECT_MAX_ITER = 100


def compute_firm_demands(
    firms: list[FirmState],
    wage: float,
    interest_rate: float,
    delta: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute optimal factor demands from Cobb-Douglas FOCs.

    For Y_f = A_f * K_f^alpha * L_f^(1-alpha), the FOCs are:
        FOC capital: alpha * A_f * (K_f/L_f)^(alpha-1) = r + delta
        FOC labor:   (1-alpha) * A_f * (K_f/L_f)^alpha = w

    Dividing: K_f/L_f = (alpha / (1-alpha)) * (w / (r + delta))
    Given firm capital K_f, optimal labor from FOC_L:
    L_f = ((1-alpha) * A_f * K_f^alpha / w)^(1/alpha)

    Args:
        firms: List of active firms.
        wage: Current wage w.
        interest_rate: Current interest rate r.
        delta: Depreciation rate.
        alpha: Capital share in Cobb-Douglas.

    Returns:
        Tuple of (total_labor_demand, total_capital_demand).
    """
    if not firms:
        return 0.0, 0.0

    rental_rate = interest_rate + delta
    if rental_rate <= 0.0:
        rental_rate = 1e-8
    if wage <= 0.0:
        wage = 1e-8

    total_labor_demand = 0.0
    total_capital_demand = 0.0

    for firm in firms:
        # Given firm capital K_f, optimal labor from FOC_L:
        # (1-alpha) * A * K^alpha * L^(-alpha) = w
        # => L = ((1-alpha) * A * K^alpha / w)^(1/alpha)
        a_f = firm.tfp
        k_f = max(firm.capital, 1e-10)

        # L_f = ((1-alpha) * A_f * K_f^alpha / w)^(1/alpha)
        numerator = (1.0 - alpha) * a_f * (k_f**alpha)
        labor_demand = (numerator / wage) ** (1.0 / alpha)

        total_labor_demand += labor_demand
        total_capital_demand += k_f  # firms demand their existing capital

    return total_labor_demand, total_capital_demand


# ============================================================================
# Walrasian clearing (REQ-101 through REQ-106)
# ============================================================================


def _compute_labor_supply(households: list[HouseholdState]) -> float:
    """Compute aggregate effective labor supply, excluding entrepreneurs.

    Entrepreneurs' labor is embedded in their firm's production function
    (REQ-108), so only workers contribute to market labor supply.

    Args:
        households: Current household states.

    Returns:
        Aggregate effective labor supply L^s = sum_{workers} z_i * (1 - l_i).
    """
    total = 0.0
    for h in households:
        if h.role == OccupationalRole.ENTREPRENEUR:
            continue
        ls = h.labor_supply if h.labor_supply > 0.0 else 0.5
        total += h.productivity * ls
    return max(total, 1e-8)


def _firm_labor_demand_at_wage(
    tfp: np.ndarray,
    capital: np.ndarray,
    wage: float,
    alpha: float,
) -> np.ndarray:
    """Vectorized per-firm optimal labor demand at a given wage.

    L_f*(w) = ((1-alpha) * A_f * K_f^alpha / w)^(1/alpha)

    Args:
        tfp: Array of firm TFPs (n_firms,).
        capital: Array of firm capitals (n_firms,).
        wage: Current wage w (must be > 0).
        alpha: Capital share.

    Returns:
        Array of per-firm labor demands (n_firms,).
    """
    w = max(wage, _MIN_WAGE)
    numerator = (1.0 - alpha) * tfp * np.power(capital, alpha)
    return np.power(numerator / w, 1.0 / alpha)


def _bisect_wage(
    firms: list[FirmState],
    labor_supply: float,
    alpha: float,
    tol: float = _BISECT_TOL,
    max_iter: int = _BISECT_MAX_ITER,
) -> tuple[float, int]:
    """Find equilibrium wage via bisection on excess labor demand.

    ELD(w) = sum_f L_f*(w) - L^s is strictly decreasing in w.
    Finds w* such that |ELD(w*)| < tol.

    Args:
        firms: Active firms with heterogeneous TFP and capital.
        labor_supply: Aggregate effective labor supply L^s.
        alpha: Capital share in Cobb-Douglas.
        tol: Convergence tolerance (default 1e-8).
        max_iter: Maximum bisection iterations (default 100).

    Returns:
        Tuple of (equilibrium wage w*, number of iterations used).
    """
    n_firms = len(firms)
    if n_firms == 0:
        return _MIN_WAGE, 0

    # Vectorize firm data
    tfp = np.array([max(f.tfp, 1e-10) for f in firms])
    capital = np.array([max(f.capital, 1e-10) for f in firms])

    def excess_labor_demand(w: float) -> float:
        ld = _firm_labor_demand_at_wage(tfp, capital, w, alpha)
        return float(np.sum(ld)) - labor_supply

    # Bracket finding: w_lo starts low, w_hi doubles until ELD < 0
    w_lo = _MIN_WAGE
    w_hi = 2.0

    # Ensure lower bound has positive excess demand
    eld_lo = excess_labor_demand(w_lo)
    if eld_lo <= 0.0:
        # Even at minimum wage, demand is below supply — return minimum wage
        return w_lo, 0

    # Find upper bound: double w_hi until ELD < 0
    for _ in range(50):
        eld_hi = excess_labor_demand(w_hi)
        if eld_hi < 0.0:
            break
        w_hi *= 2.0
    else:
        # If we never found a negative ELD, return w_hi as best guess
        log.warning(
            "walrasian.bracket_failed",
            w_hi=w_hi,
            eld_hi=excess_labor_demand(w_hi),
        )
        return w_hi, 0

    # Bisection
    iterations = 0
    for iterations in range(1, max_iter + 1):
        w_mid = 0.5 * (w_lo + w_hi)
        eld_mid = excess_labor_demand(w_mid)

        if abs(eld_mid) < tol:
            return w_mid, iterations

        if eld_mid > 0.0:
            w_lo = w_mid
        else:
            w_hi = w_mid

    # Return best midpoint after max iterations
    w_mid = 0.5 * (w_lo + w_hi)
    return w_mid, iterations


def _compute_firm_outputs(
    firms: list[FirmState],
    wage: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Compute per-firm optimal labor and aggregate output at equilibrium wage.

    Per-firm: L_f*(w) from FOC, Y_f = A_f * K_f^alpha * L_f*(w)^(1-alpha).
    Aggregate: Y = sum_f Y_f.

    Args:
        firms: Active firms.
        wage: Equilibrium wage w*.
        alpha: Capital share.

    Returns:
        Tuple of (aggregate output Y, per-firm labor demands array).
    """
    if not firms:
        return 0.0, np.array([])

    tfp = np.array([max(f.tfp, 1e-10) for f in firms])
    capital = np.array([max(f.capital, 1e-10) for f in firms])

    firm_labors = _firm_labor_demand_at_wage(tfp, capital, wage, alpha)
    firm_outputs = tfp * np.power(capital, alpha) * np.power(firm_labors, 1.0 - alpha)

    # NaN/Inf guard
    if not np.all(np.isfinite(firm_outputs)):
        log.warning("walrasian.non_finite_outputs", wage=wage)
        firm_outputs = np.nan_to_num(firm_outputs, nan=0.0, posinf=0.0, neginf=0.0)

    aggregate_output = float(np.sum(firm_outputs))
    return aggregate_output, firm_labors


def clear_markets_walrasian(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Walrasian equilibrium with heterogeneous firms (REQ-101 through REQ-106).

    When firms are present, finds equilibrium wage via bisection on excess labor
    demand, computes firm-level outputs, and derives interest rate from aggregate
    MPK. Falls back to representative-firm clearing when no firms are present.

    Args:
        households: Current household states.
        firms: Active firms.
        aggregate_tfp: Aggregate TFP (used in fallback only).
        config: Simulation configuration.

    Returns:
        MarketState with equilibrium prices and aggregates.
    """
    n_agents = len(households)
    if n_agents == 0:
        return MarketState(wage=1.0, interest_rate=0.05)

    # Fallback: no firms -> representative firm (REQ-105)
    if not firms:
        return _representative_firm_clearing(households, aggregate_tfp, config)

    # Compute labor supply (excluding entrepreneurs, REQ-108)
    labor_supply = _compute_labor_supply(households)

    # Bisection on wage (REQ-102, REQ-106)
    w_star, iterations = _bisect_wage(firms, labor_supply, config.alpha)

    # Compute firm-level outputs at equilibrium wage (REQ-103)
    aggregate_output, firm_labors = _compute_firm_outputs(firms, w_star, config.alpha)

    # Capital market: r from aggregate MPK (REQ-104)
    k_total = sum(max(f.capital, 1e-10) for f in firms)
    k_total = max(k_total, 1e-10)
    r = max(config.alpha * aggregate_output / k_total - config.delta, _MIN_INTEREST)

    # Compute clearing error for diagnostics
    total_labor_demand = float(np.sum(firm_labors)) if len(firm_labors) > 0 else 0.0
    labor_excess = total_labor_demand - labor_supply
    clearing_error = abs(labor_excess)

    # Aggregates
    aggregate_consumption = sum(h.consumption for h in households)
    aggregate_investment = sum(f.capital * config.delta for f in firms)

    log.debug(
        "walrasian.cleared",
        wage=round(w_star, 6),
        interest_rate=round(r, 6),
        aggregate_output=round(aggregate_output, 2),
        clearing_error=clearing_error,
        iterations=iterations,
    )

    return MarketState(
        wage=w_star,
        interest_rate=r,
        aggregate_output=aggregate_output,
        aggregate_consumption=aggregate_consumption,
        aggregate_investment=aggregate_investment,
        government_spending=0.0,
        market_clearing_error=clearing_error,
        labor_excess_demand=labor_excess,
        capital_excess_demand=0.0,
    )


def _representative_firm_clearing(
    households: list[HouseholdState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Representative-firm analytical clearing (REQ-105 fallback).

    Uses Cobb-Douglas FOCs with aggregate household supplies:
        Y = A * K^alpha * L^(1-alpha)
        w = (1-alpha) * Y / L
        r = alpha * Y / K - delta

    Args:
        households: Current household states.
        aggregate_tfp: A_t.
        config: Simulation configuration.

    Returns:
        MarketState with analytical equilibrium prices.
    """
    # Aggregate supply from households (excluding entrepreneurs from labor)
    total_labor_supply = _compute_labor_supply(households)
    total_capital_supply = max(sum(h.wealth for h in households), 1e-8)

    y = (
        aggregate_tfp
        * (total_capital_supply ** config.alpha)
        * (total_labor_supply ** (1.0 - config.alpha))
    )

    w = max((1.0 - config.alpha) * y / total_labor_supply, _MIN_WAGE)
    r = max(config.alpha * y / total_capital_supply - config.delta, _MIN_INTEREST)

    aggregate_consumption = sum(h.consumption for h in households)
    aggregate_investment = total_capital_supply * config.delta

    return MarketState(
        wage=w,
        interest_rate=r,
        aggregate_output=y,
        aggregate_consumption=aggregate_consumption,
        aggregate_investment=aggregate_investment,
        government_spending=0.0,
        market_clearing_error=0.0,
        labor_excess_demand=0.0,
        capital_excess_demand=0.0,
    )


# ============================================================================
# Main dispatch and analytical clearing (backward compatibility)
# ============================================================================


def clear_markets(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
    prev_market: MarketState | None = None,
) -> MarketState:
    """Find equilibrium (w_t, r_t) — dispatches based on config.

    When ``market_clearing_method`` is ``"walrasian"`` and firms are present,
    uses bisection on excess labor demand (REQ-101 through REQ-106).
    Otherwise uses the analytical representative-firm formula.

    Args:
        households: Current household states.
        firms: Active firms.
        aggregate_tfp: A_t aggregate TFP this period.
        config: Simulation configuration.
        prev_market: Previous period market state (unused, kept for API compat).

    Returns:
        MarketState with equilibrium prices and aggregates.

    Implements REQ-011, REQ-012, REQ-013, REQ-014, PROP-003.
    """
    method = getattr(config, "market_clearing_method", "analytical")

    if method == "walrasian":
        return clear_markets_walrasian(households, firms, aggregate_tfp, config)

    # Default: analytical path (backward compatible)
    return _clear_markets_analytical(households, firms, aggregate_tfp, config)


def _clear_markets_analytical(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Original analytical clearing path (backward compatible).

    Uses representative-firm Cobb-Douglas FOCs regardless of firm heterogeneity.
    Preserved for ``market_clearing_method="analytical"`` (default).

    Args:
        households: Current household states.
        firms: Active firms.
        aggregate_tfp: A_t aggregate TFP this period.
        config: Simulation configuration.

    Returns:
        MarketState with equilibrium prices and aggregates.
    """
    n_agents = len(households)
    if n_agents == 0:
        return MarketState(wage=1.0, interest_rate=0.05)

    # Aggregate supply from households
    # Use productivity * 0.5 as labor supply estimate when labor_supply is stale (0)
    # When fix_entrepreneur_budget is active, exclude entrepreneurs from labor
    # supply — their labor is embedded in firm production (REQ-107).
    total_labor_supply = 0.0
    for h in households:
        if (
            config.fix_entrepreneur_budget
            and h.role == OccupationalRole.ENTREPRENEUR
        ):
            continue
        ls = h.labor_supply if h.labor_supply > 0.0 else 0.5
        total_labor_supply += h.productivity * ls
    total_capital_supply = sum(h.wealth for h in households)

    # Ensure positive supply (labor floor prevents division-by-zero in wage)
    total_labor_supply = max(total_labor_supply, 1e-8)
    total_capital_supply = max(total_capital_supply, 1e-8)

    # When firms exist, use their aggregate capital as the capital input
    if firms:
        firm_capital = sum(max(f.capital, 1e-10) for f in firms)
        # Use the larger of firm capital and household wealth as effective K
        k_agg = max(firm_capital, total_capital_supply)
    else:
        k_agg = total_capital_supply

    return _analytical_equilibrium(
        households,
        firms,
        total_labor_supply,
        k_agg,
        aggregate_tfp,
        config,
    )


def _analytical_equilibrium(
    households: list[HouseholdState],
    firms: list[FirmState],
    total_labor_supply: float,
    total_capital_supply: float,
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Compute equilibrium analytically using Cobb-Douglas FOCs.

    Uses a representative firm with Cobb-Douglas technology:
    Y = A * K^alpha * L^(1-alpha)

    FOC: w = (1-alpha) * Y/L, r = alpha * Y/K - delta.

    When labor_supply values are stale (initialized to 0), we use
    productivity * 0.5 as initial guess (assuming 50% leisure).

    Args:
        households: Current household states.
        firms: Active firms (for output computation).
        total_labor_supply: Aggregate effective labor (already corrected for stale values).
        total_capital_supply: Aggregate capital (household wealth or firm capital).
        aggregate_tfp: A_t.
        config: Simulation configuration.

    Returns:
        MarketState with analytical equilibrium prices.
    """
    # Ensure positive supply for numerical stability
    total_labor_supply = max(total_labor_supply, 1e-8)
    total_capital_supply = max(total_capital_supply, 1e-8)

    y = (
        aggregate_tfp
        * (total_capital_supply**config.alpha)
        * (total_labor_supply ** (1.0 - config.alpha))
    )

    w = max((1.0 - config.alpha) * y / total_labor_supply, _MIN_WAGE)
    r = max(config.alpha * y / total_capital_supply - config.delta, _MIN_INTEREST)

    # Compute aggregate output from firms if present, otherwise use Y
    if firms:
        aggregate_output = 0.0
        for firm in firms:
            firm_output = produce_output(firm, config.alpha)
            aggregate_output += firm_output
        # Use max of analytical and firm-based output for consistency
        aggregate_output = max(aggregate_output, y)
    else:
        aggregate_output = y

    aggregate_consumption = sum(h.consumption for h in households)
    if firms:
        aggregate_investment = sum(f.capital * config.delta for f in firms)
    else:
        aggregate_investment = total_capital_supply * config.delta

    return MarketState(
        wage=w,
        interest_rate=r,
        aggregate_output=aggregate_output,
        aggregate_consumption=aggregate_consumption,
        aggregate_investment=aggregate_investment,
        government_spending=0.0,
        market_clearing_error=0.0,
        labor_excess_demand=0.0,
        capital_excess_demand=0.0,
    )
