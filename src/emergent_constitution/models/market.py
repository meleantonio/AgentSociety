"""Market state model — economy-wide equilibrium prices and aggregates."""

from __future__ import annotations

from pydantic import BaseModel


class MarketState(BaseModel):
    """Economy-wide equilibrium prices and aggregates for period t.

    Args:
        wage: w_t labor market clearing price.
        interest_rate: Legacy effective return used by solvers.
        capital_rate: r^k_t return on productive/illiquid capital.
        bond_rate: r^b_t return on liquid government bonds.
        aggregate_output: Y_t total output.
        aggregate_consumption: C_t total consumption.
        aggregate_investment: I_t total investment.
        government_spending: G_t public goods spending.
        market_clearing_error: |excess demand| (PROP-003: < 1e-6).
        labor_excess_demand: Labor demand minus labor supply.
        capital_excess_demand: Capital demand minus capital supply.
        resource_residual: Goods-market residual Y - (C + I + G).
    """

    wage: float
    interest_rate: float
    capital_rate: float | None = None
    bond_rate: float | None = None
    aggregate_output: float = 0.0
    aggregate_consumption: float = 0.0
    aggregate_investment: float = 0.0
    government_spending: float = 0.0
    market_clearing_error: float = 0.0
    labor_excess_demand: float = 0.0
    capital_excess_demand: float = 0.0
    resource_residual: float = 0.0
