"""Tatonnement market clearing — find equilibrium prices (w, r) each period.

Implements REQ-011 (labor market clearing), REQ-012 (capital market clearing),
REQ-013 (goods market identity Y = C + I + G), REQ-014 (iterative price
adjustment), PROP-003 (market clearing error < tolerance).

Design reference: spec/design.md section 3.3.
"""

from __future__ import annotations

import math

import structlog

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.economics import produce_output
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.household import HouseholdState
from emergent_constitution.models.market import MarketState

log = structlog.get_logger()

# Minimum price floor to prevent division by zero or negative wages
_MIN_WAGE = 1e-6
_MIN_INTEREST = -0.99  # interest rate can be negative but bounded


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


def clear_markets(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
    prev_market: MarketState | None = None,
) -> MarketState:
    """Find equilibrium (w_t, r_t) via tatonnement price adjustment.

    Algorithm:
    1. Initialize w, r from previous period (or analytical guess).
    2. Loop (max tatonnement_max_iter iterations):
       a. Compute firm optimal demands given (w, r).
       b. Compute aggregate supply from households.
       c. Compute excess demand.
       d. Adjust prices: w += step * excess_labor, r += step * excess_capital.
       e. Check convergence.
    3. Compute aggregates (Y, C, I, G).
    4. Record market_clearing_error.

    If non-convergent, use best prices found (lowest excess demand).

    Args:
        households: Current household states.
        firms: Active firms.
        aggregate_tfp: A_t aggregate TFP this period.
        config: Simulation configuration with tatonnement params.
        prev_market: Previous period market state for warm-starting prices.

    Returns:
        MarketState with equilibrium prices and aggregates.

    Implements REQ-011, REQ-012, REQ-013, REQ-014, PROP-003.
    """
    n_agents = len(households)
    if n_agents == 0:
        return MarketState(wage=1.0, interest_rate=0.05)

    # Aggregate supply from households
    total_labor_supply = sum(h.productivity * h.labor_supply for h in households)
    total_capital_supply = sum(h.wealth for h in households)

    # Ensure positive supply
    total_labor_supply = max(total_labor_supply, 1e-8)
    total_capital_supply = max(total_capital_supply, 1e-8)

    # If no firms, use representative firm approach
    if not firms:
        return _analytical_equilibrium(
            households,
            total_labor_supply,
            total_capital_supply,
            aggregate_tfp,
            config,
        )

    # Initialize prices
    if prev_market is not None:
        w = max(prev_market.wage, _MIN_WAGE)
        r = max(prev_market.interest_rate, _MIN_INTEREST)
    else:
        # Analytical guess from Cobb-Douglas FOCs with aggregate quantities
        y_guess = (
            aggregate_tfp
            * (total_capital_supply**config.alpha)
            * (total_labor_supply ** (1.0 - config.alpha))
        )
        w = max((1.0 - config.alpha) * y_guess / total_labor_supply, _MIN_WAGE)
        r = max(config.alpha * y_guess / total_capital_supply - config.delta, _MIN_INTEREST)

    step = config.tatonnement_step_size
    best_w, best_r = w, r
    best_error = float("inf")

    for iteration in range(config.tatonnement_max_iter):
        # Compute firm demands at current prices
        labor_demand, capital_demand = compute_firm_demands(
            firms,
            w,
            r,
            config.delta,
            config.alpha,
        )

        # Excess demands
        excess_labor = labor_demand - total_labor_supply
        excess_capital = capital_demand - total_capital_supply

        # Market clearing error
        clearing_error = math.sqrt(excess_labor**2 + excess_capital**2)

        # Track best
        if clearing_error < best_error:
            best_error = clearing_error
            best_w = w
            best_r = r

        # Check convergence
        if (
            abs(excess_labor) < config.tatonnement_tolerance
            and abs(excess_capital) < config.tatonnement_tolerance
        ):
            log.debug(
                "market.cleared",
                iteration=iteration,
                wage=w,
                interest_rate=r,
                clearing_error=clearing_error,
            )
            break

        # Adjust prices
        # Normalize excess demands by supply to get relative adjustments
        w += step * (excess_labor / total_labor_supply)
        r += step * (excess_capital / total_capital_supply)

        # Clamp prices
        w = max(w, _MIN_WAGE)
        r = max(r, _MIN_INTEREST)
    else:
        # Did not converge
        w, r = best_w, best_r
        log.warning(
            "market.not_converged",
            max_iter=config.tatonnement_max_iter,
            best_error=best_error,
            wage=w,
            interest_rate=r,
        )

    # Compute aggregates at equilibrium prices
    aggregate_output = 0.0
    for firm in firms:
        firm_output = produce_output(firm, config.alpha)
        aggregate_output += firm_output

    aggregate_consumption = sum(h.consumption for h in households)
    aggregate_investment = sum(f.capital * config.delta for f in firms)  # replacement investment

    # Recompute final excess demands for reporting
    final_ld, final_kd = compute_firm_demands(firms, w, r, config.delta, config.alpha)
    labor_excess = final_ld - total_labor_supply
    capital_excess = final_kd - total_capital_supply
    final_error = math.sqrt(labor_excess**2 + capital_excess**2)

    return MarketState(
        wage=w,
        interest_rate=r,
        aggregate_output=aggregate_output,
        aggregate_consumption=aggregate_consumption,
        aggregate_investment=aggregate_investment,
        government_spending=0.0,
        market_clearing_error=final_error,
        labor_excess_demand=labor_excess,
        capital_excess_demand=capital_excess,
    )


def _analytical_equilibrium(
    households: list[HouseholdState],
    total_labor_supply: float,
    total_capital_supply: float,
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Compute equilibrium analytically when there are no explicit firms.

    Uses a representative firm with Cobb-Douglas technology:
    Y = A * K^alpha * L^(1-alpha)

    FOC: w = (1-alpha) * Y/L, r = alpha * Y/K - delta.

    Args:
        households: Current household states.
        total_labor_supply: Aggregate effective labor.
        total_capital_supply: Aggregate capital (household wealth).
        aggregate_tfp: A_t.
        config: Simulation configuration.

    Returns:
        MarketState with analytical equilibrium prices.
    """
    y = (
        aggregate_tfp
        * (total_capital_supply**config.alpha)
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
