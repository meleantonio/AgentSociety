"""Simulation configuration -- v1 and v2.

v1 SimulationConfig retained for backward compat.
v2 SimulationConfigV2 adds DSGE-HA parameters per spec/design.md section 2.9.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ============================================================================
# v1 config (backward compatibility)
# ============================================================================


class SimulationConfig(BaseModel):
    """Top-level configuration for a simulation run (v1).

    Args:
        num_agents: Number of citizen-agents (>=2).
        max_ticks: Maximum number of ticks to simulate (>=1).
        seed: RNG seed for reproducibility (PROP-001).
        initial_wealth_mean: Mean of the initial wealth distribution.
        initial_wealth_std: Std dev of the initial wealth distribution.
        initial_productivity_mean: Mean of the initial productivity distribution.
        initial_productivity_std: Std dev of the initial productivity distribution.
        proposal_interval: Proposals are collected every K ticks.
        observer_interval: Observer records stats every K ticks.
        coalition_interval: Coalitions reform every K ticks.
    """

    num_agents: int = Field(default=50, ge=2)
    max_ticks: int = Field(default=100, ge=1)
    seed: int = 42
    initial_wealth_mean: float = Field(default=100.0, gt=0.0)
    initial_wealth_std: float = Field(default=30.0, ge=0.0)
    initial_productivity_mean: float = Field(default=10.0, gt=0.0)
    initial_productivity_std: float = Field(default=3.0, ge=0.0)
    proposal_interval: int = Field(default=5, ge=1)
    observer_interval: int = Field(default=5, ge=1)
    trade_interval: int = Field(default=5, ge=1)
    coalition_interval: int = Field(default=10, ge=1)
    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM-based citizen logic for a subset of agents.",
    )
    llm_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of agents that use LLM reasoning each tick (0.0-1.0).",
    )


# ============================================================================
# v2 config (DSGE-HA)
# ============================================================================


class SimulationConfigV2(BaseModel):
    """Full DSGE-HA simulation configuration (v2).

    Implements REQ-034 with ~30 fields covering core, shocks, production,
    household, intervals, market clearing, LLM, benchmark, and R&D params.
    """

    # --- Core ---
    num_agents: int = Field(default=50, ge=20, description="N >= 20 (REQ-001)")
    max_periods: int = Field(default=100, ge=1, description="T >= 1")
    seed: int = Field(default=42, description="RNG seed (PROP-001)")

    # --- Shock parameters ---
    rho_z: float = Field(default=0.9, gt=0.0, lt=1.0, description="Idiosyncratic persistence")
    sigma_z: float = Field(default=0.2, gt=0.0, description="Idiosyncratic volatility")
    num_z_states: int = Field(default=7, ge=2, le=50, description="Rouwenhorst grid points")
    rho_a: float = Field(default=0.95, gt=0.0, lt=1.0, description="Aggregate TFP persistence")
    sigma_a: float = Field(default=0.01, gt=0.0, description="Aggregate TFP volatility")
    enable_preference_shocks: bool = Field(
        default=False, description="REQ-017 optional preference shocks"
    )

    # --- Entrepreneurial ---
    rho_e: float = Field(
        default=0.85, gt=0.0, lt=1.0, description="Entrepreneurial ability persistence"
    )
    sigma_e: float = Field(default=0.3, gt=0.0, description="Entrepreneurial ability volatility")
    num_e_states: int = Field(
        default=7, ge=2, le=50, description="Rouwenhorst grid points for ability"
    )
    firm_entry_cost: float = Field(
        default=5.0, gt=0.0, description="Fixed cost to create a firm (sunk)"
    )
    firm_value_horizon: int = Field(
        default=20, ge=1, description="Truncation horizon for firm PDV"
    )
    use_bellman_occ_choice: bool = Field(
        default=False,
        description=(
            "Use Bellman-based occupational choice (REQ-110..114). "
            "Default OFF for backward compat."
        ),
    )

    # --- Production ---
    alpha: float = Field(default=0.33, gt=0.0, lt=1.0, description="Capital share in Cobb-Douglas")
    delta: float = Field(default=0.1, gt=0.0, lt=1.0, description="Depreciation rate")
    min_firm_capital: float = Field(
        default=10.0, gt=0.0, description="Minimum capital for firm creation"
    )

    # --- Household ---
    a_min: float = Field(default=0.0, ge=0.0, description="Borrowing limit (REQ-004)")
    initial_wealth_mean: float = Field(default=100.0, gt=0.0)
    initial_wealth_std: float = Field(default=30.0, ge=0.0)

    # --- Household preferences ---
    homogeneous_preferences: bool = Field(
        default=True, description="All agents share utility params (fast VFI path)"
    )
    utility_alpha: float = Field(
        default=0.4, gt=0.0, lt=1.0, description="Consumption weight (homogeneous)"
    )
    utility_beta: float = Field(
        default=0.35, gt=0.0, lt=1.0, description="Leisure weight (homogeneous)"
    )
    utility_gamma: float = Field(
        default=0.25, gt=0.0, lt=1.0, description="Public goods weight (homogeneous)"
    )
    utility_beta_discount: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Discount factor (homogeneous)"
    )

    # --- Intervals ---
    proposal_interval: int = Field(
        default=5, ge=1, description="Constitutional proposal every K periods"
    )
    observer_interval: int = Field(default=5, ge=1, description="Observation every K periods")

    # --- Market clearing ---
    market_clearing_method: Literal["analytical", "walrasian"] = Field(
        default="analytical",
        description=(
            "'analytical' uses representative-firm FOCs. "
            "'walrasian' uses bisection with heterogeneous firms "
            "(REQ-101..106)."
        ),
    )
    tatonnement_max_iter: int = Field(
        default=100, ge=1, description="Max iterations for price finding"
    )
    tatonnement_tolerance: float = Field(default=1e-6, gt=0.0, description="PROP-003 tolerance")
    tatonnement_step_size: float = Field(default=0.01, gt=0.0, description="Price adjustment step")

    # --- LLM ---
    use_llm: bool = Field(default=True, description="Default: LLM-driven (REQ-024)")
    llm_provider: str = Field(default="anthropic", description="Provider name")
    llm_model: str = Field(default="claude-sonnet-4-5-20250929", description="Model ID")
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Determinism (PROP-001)"
    )
    llm_batch_size: int = Field(default=10, ge=1, description="Agents per batch call (REQ-029)")
    llm_cache_enabled: bool = Field(default=True, description="Cache identical contexts (REQ-029)")

    # --- Benchmark ---
    benchmark_mode: bool = Field(default=False, description="Numerical-only mode (REQ-037)")
    solver_method: Literal["vfi", "vfi_numpy", "egm"] = Field(
        default="egm", description="'vfi', 'vfi_numpy', or 'egm' (REQ-036, REQ-120)"
    )

    # --- R&D ---
    rd_success_base_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Base probability of TFP improvement",
    )
    rd_tfp_improvement_mean: float = Field(
        default=0.05, gt=0.0, description="Mean TFP improvement factor"
    )
    rd_tfp_improvement_std: float = Field(
        default=0.02, ge=0.0, description="Std of TFP improvement factor"
    )

    # --- Distribution mode (Phase 2: KFE) ---
    distribution_mode: Literal["individual", "kfe"] = Field(
        default="individual",
        description=(
            "'individual' tracks N individual agents (default, backward compat). "
            "'kfe' uses Kolmogorov Forward Equation distribution tracking "
            "(REQ-201..205). Feature-gated: default OFF."
        ),
    )
    kfe_convergence_tolerance: float = Field(
        default=1e-10,
        gt=0.0,
        description="L1 convergence tolerance for stationary KFE distribution (REQ-205).",
    )
    kfe_max_iterations: int = Field(
        default=10_000,
        ge=1,
        description="Maximum iterations for KFE stationary distribution (REQ-205).",
    )

    # --- Government (Phase 3: HANK, REQ-306..309) ---
    initial_debt: float = Field(
        default=0.0, ge=0.0, description="Initial government debt B_0 (REQ-306)"
    )
    debt_gdp_max: float = Field(
        default=1.5, gt=0.0, description="Fiscal rule threshold B/Y (REQ-309)"
    )
    fiscal_rule_adjustment: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Tax rate increment when fiscal rule triggers (REQ-309)",
    )

    # --- Feature flags ---
    fix_entrepreneur_budget: bool = Field(
        default=False,
        description=(
            "When True, entrepreneurs receive only firm profit (pi_f) as income, "
            "not labor income. Their labor supply is excluded from market aggregation. "
            "Default OFF preserves v2 backward compatibility."
        ),
    )

    @model_validator(mode="after")
    def _benchmark_disables_llm(self) -> SimulationConfigV2:
        """If benchmark_mode is enabled, force use_llm to False."""
        if self.benchmark_mode and self.use_llm:
            object.__setattr__(self, "use_llm", False)
        return self

    @model_validator(mode="after")
    def _validate_homogeneous_weights(self) -> SimulationConfigV2:
        """When homogeneous_preferences is True, utility weights must sum to ~1.0."""
        if self.homogeneous_preferences:
            total = self.utility_alpha + self.utility_beta + self.utility_gamma
            if abs(total - 1.0) > 1e-6:
                msg = (
                    f"Homogeneous utility weights must sum to 1.0, "
                    f"got {total:.6f} (alpha={self.utility_alpha}, "
                    f"beta={self.utility_beta}, gamma={self.utility_gamma})"
                )
                raise ValueError(msg)
        return self
