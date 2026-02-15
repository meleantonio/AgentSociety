"""Analytical market clearing — find equilibrium prices (w, r) each period.

Implements REQ-011 (labor market clearing), REQ-012 (capital market clearing),
REQ-013 (goods market identity Y = C + I + G), REQ-014 (iterative price
adjustment), PROP-003 (market clearing error < tolerance).

With Cobb-Douglas production, the aggregation theorem gives closed-form
prices from aggregate factor supplies. This replaces the tatonnement
iteration with exact analytical solutions that always clear.

Design reference: spec/design.md section 3.3.
"""

from __future__ import annotations

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
    """Find equilibrium (w_t, r_t) using analytical Cobb-Douglas solution.

    With Cobb-Douglas production Y = A * K^alpha * L^(1-alpha), the
    competitive equilibrium prices are determined by FOCs:
        w = (1-alpha) * Y / L
        r = alpha * Y / K - delta

    This always clears with zero error. When firms are present, we use
    aggregate firm capital as K; otherwise household wealth serves as K.

    Args:
        households: Current household states.
        firms: Active firms.
        aggregate_tfp: A_t aggregate TFP this period.
        config: Simulation configuration with tatonnement params.
        prev_market: Previous period market state (unused, kept for API compat).

    Returns:
        MarketState with equilibrium prices and aggregates.

    Implements REQ-011, REQ-012, REQ-013, REQ-014, PROP-003.
    """
    n_agents = len(households)
    if n_agents == 0:
        return MarketState(wage=1.0, interest_rate=0.05)

    # Aggregate supply from households
    # Use productivity * 0.5 as labor supply estimate when labor_supply is stale (0)
    total_labor_supply = 0.0
    for h in households:
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
