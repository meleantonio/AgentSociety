# Configuration Reference

## SimulationConfigV2

**Location**: `src/emergent_constitution/config.py`

The `SimulationConfigV2` class controls all aspects of a v2 DSGE-HA simulation run. It uses Pydantic for validation and provides sensible defaults. In benchmark mode (`benchmark_mode=True`), LLM usage is automatically disabled.

## Complete Field Reference

### Core Parameters

#### num_agents
- **Type**: `int` | **Default**: `50` | **Constraints**: `>= 20`
- Number of household-agents. Minimum 20 per REQ-001.

```python
config = SimulationConfigV2(num_agents=100)
```

#### max_periods
- **Type**: `int` | **Default**: `100` | **Constraints**: `>= 1`
- Number of simulation periods to execute.

#### seed
- **Type**: `int` | **Default**: `42`
- RNG seed for complete reproducibility (PROP-001). Same seed + config = identical results.

### Shock Parameters

#### rho_z
- **Type**: `float` | **Default**: `0.9` | **Constraints**: `(0, 1)`
- Persistence of the idiosyncratic AR(1) productivity process. Higher values mean slower mean reversion.

#### sigma_z
- **Type**: `float` | **Default**: `0.2` | **Constraints**: `> 0`
- Volatility of idiosyncratic productivity shocks. Higher values create more income risk.

#### num_z_states
- **Type**: `int` | **Default**: `7` | **Constraints**: `[2, 50]`
- Number of grid points for the Rouwenhorst discretization of the productivity Markov chain. Changed from 5 to 7 in HANK v3 for better distributional accuracy.

#### rho_a
- **Type**: `float` | **Default**: `0.95` | **Constraints**: `(0, 1)`
- Persistence of the aggregate TFP AR(1) process.

#### sigma_a
- **Type**: `float` | **Default**: `0.01` | **Constraints**: `> 0`
- Volatility of aggregate TFP shocks.

#### enable_preference_shocks
- **Type**: `bool` | **Default**: `False`
- Enable optional REQ-017 preference shocks (noise on utility weights).

### Production Parameters

#### alpha
- **Type**: `float` | **Default**: `0.33` | **Constraints**: `(0, 1)`
- Capital share in Cobb-Douglas production: `Y = A * K^α * L^(1-α)`. Standard calibration is 0.33.

```python
# Higher capital share (capital-intensive economy)
config = SimulationConfigV2(alpha=0.40)
```

#### delta
- **Type**: `float` | **Default**: `0.1` | **Constraints**: `(0, 1)`
- Depreciation rate of capital per period.

#### min_firm_capital
- **Type**: `float` | **Default**: `10.0` | **Constraints**: `> 0`
- Minimum capital required for an agent to create a new firm.

### Household Parameters

#### a_min
- **Type**: `float` | **Default**: `0.0` | **Constraints**: `>= 0`
- Borrowing limit (REQ-004). Wealth cannot fall below this value.

```python
# Allow no borrowing (natural borrowing limit)
config = SimulationConfigV2(a_min=0.0)
```

#### initial_wealth_mean
- **Type**: `float` | **Default**: `100.0` | **Constraints**: `> 0`
- Mean of the initial wealth distribution for households.

#### initial_wealth_std
- **Type**: `float` | **Default**: `30.0` | **Constraints**: `>= 0`
- Standard deviation of initial wealth. Higher values create more initial inequality.

### Simulation Intervals

#### proposal_interval
- **Type**: `int` | **Default**: `5` | **Constraints**: `>= 1`
- Constitutional proposals and votes occur every K periods.

```python
# Rapid institutional change
config = SimulationConfigV2(proposal_interval=2)

# Slow, stable institutions
config = SimulationConfigV2(proposal_interval=20)
```

#### observer_interval
- **Type**: `int` | **Default**: `5` | **Constraints**: `>= 1`
- Observer computes 20+ statistics every K periods.

### Market Clearing Parameters

#### market_clearing_method
- **Type**: `Literal["analytical", "walrasian"]` | **Default**: `"analytical"`
- Market clearing algorithm. `"analytical"` uses closed-form Cobb-Douglas FOCs (zero clearing error). `"walrasian"` uses bisection with heterogeneous firms (REQ-101..106).

```python
# Walrasian clearing with heterogeneous firms
config = SimulationConfigV2(market_clearing_method="walrasian")
```

#### tatonnement_max_iter
- **Type**: `int` | **Default**: `100` | **Constraints**: `>= 1`
- Maximum iterations for the Walrasian price-finding algorithm (only used when `market_clearing_method="walrasian"`).

#### tatonnement_tolerance
- **Type**: `float` | **Default**: `1e-6` | **Constraints**: `> 0`
- Convergence tolerance for market clearing (PROP-003). Applies to Walrasian mode only.

#### tatonnement_step_size
- **Type**: `float` | **Default**: `0.01` | **Constraints**: `> 0`
- Price adjustment step size per iteration. Applies to Walrasian mode only.

### LLM Parameters

#### use_llm
- **Type**: `bool` | **Default**: `True`
- Enable LLM-driven agent decisions. If False, uses numerical solver.

#### llm_provider
- **Type**: `str` | **Default**: `"anthropic"`
- LLM provider name. Currently supports `"anthropic"` and `"mock"`.

#### llm_model
- **Type**: `str` | **Default**: `"claude-sonnet-4-5-20250929"`
- Model ID for the LLM provider.

```python
# Use Haiku for cheaper runs (~75% cost savings)
config = SimulationConfigV2(llm_model="claude-haiku-4-5-20251001")
```

#### llm_temperature
- **Type**: `float` | **Default**: `0.0` | **Constraints**: `[0, 2]`
- LLM sampling temperature. Default 0.0 for deterministic output (PROP-001).

#### llm_batch_size
- **Type**: `int` | **Default**: `10` | **Constraints**: `>= 1`
- Number of agents per batched LLM API call (REQ-029). Higher values reduce overhead but increase per-call latency.

#### llm_cache_enabled
- **Type**: `bool` | **Default**: `True`
- Cache identical (state, context) pairs to skip redundant API calls (REQ-029).

### Benchmark Parameters

#### benchmark_mode
- **Type**: `bool` | **Default**: `False`
- Run entirely on the numerical solver with $0 API cost (REQ-037). Automatically sets `use_llm=False`.

```python
# $0 cost run for testing and comparison
config = SimulationConfigV2(benchmark_mode=True)
```

#### solver_method
- **Type**: `Literal["vfi", "vfi_numpy", "egm"]` | **Default**: `"egm"`
- Numerical solver method: Value Function Iteration, NumPy-vectorized VFI, or Endogenous Grid Method (REQ-036, REQ-120). `"egm"` is fastest and most accurate for standard HANK models.

### R&D Parameters

#### rd_success_base_prob
- **Type**: `float` | **Default**: `0.1` | **Constraints**: `[0, 1]`
- Base probability of R&D investment producing a TFP improvement.

#### rd_tfp_improvement_mean
- **Type**: `float` | **Default**: `0.05` | **Constraints**: `> 0`
- Mean TFP improvement factor when R&D succeeds.

#### rd_tfp_improvement_std
- **Type**: `float` | **Default**: `0.02` | **Constraints**: `>= 0`
- Standard deviation of TFP improvement factor.

### Distribution Mode Parameters (HANK v3: KFE)

#### distribution_mode
- **Type**: `Literal["individual", "kfe"]` | **Default**: `"individual"`
- Distribution tracking method. `"individual"` tracks N discrete agents (default, backward compatible). `"kfe"` uses Kolmogorov Forward Equation with grid-based distribution tracking (REQ-201..205).

```python
# Enable KFE distribution tracking
config = SimulationConfigV2(distribution_mode="kfe")
```

#### kfe_convergence_tolerance
- **Type**: `float` | **Default**: `1e-10` | **Constraints**: `> 0`
- L1 convergence tolerance for stationary KFE distribution (REQ-205). Only used when `distribution_mode="kfe"`.

#### kfe_max_iterations
- **Type**: `int` | **Default**: `10000` | **Constraints**: `>= 1`
- Maximum iterations for KFE stationary distribution computation (REQ-205). Only used when `distribution_mode="kfe"`.

### Two-Asset Household Parameters (HANK v3: Liquid/Illiquid)

#### two_asset_mode
- **Type**: `bool` | **Default**: `False`
- Enable two-asset household model with liquid bonds (b) and illiquid capital (k) with convex adjustment costs (REQ-301..305). Default OFF preserves single-asset backward compatibility.

```python
# Enable two-asset HANK model
config = SimulationConfigV2(two_asset_mode=True, chi_0=0.01, chi_1=0.005)
```

#### chi_0
- **Type**: `float` | **Default**: `0.01` | **Constraints**: `>= 0`
- Linear component of illiquid adjustment cost: `chi(d) = chi_0 * |d| + chi_1 * d^2/k` (REQ-302). Only used when `two_asset_mode=True`.

#### chi_1
- **Type**: `float` | **Default**: `0.005` | **Constraints**: `>= 0`
- Quadratic component of illiquid adjustment cost (REQ-302). Only used when `two_asset_mode=True`.

#### b_min
- **Type**: `float` | **Default**: `0.0`
- Borrowing constraint on liquid assets (REQ-304). Can be negative to allow unsecured credit. Only used when `two_asset_mode=True`.

### Government Sector Parameters (HANK v3: Fiscal Policy)

#### initial_debt
- **Type**: `float` | **Default**: `0.0` | **Constraints**: `>= 0`
- Initial government debt B_0 (REQ-306). When > 0, enables government bond market clearing and fiscal rule.

```python
# Enable government debt and bonds
config = SimulationConfigV2(initial_debt=50.0, debt_gdp_max=1.5)
```

#### debt_gdp_max
- **Type**: `float` | **Default**: `1.5` | **Constraints**: `> 0`
- Fiscal rule threshold for debt-to-GDP ratio (REQ-309). When `B/Y > debt_gdp_max`, taxes automatically increase by `fiscal_rule_adjustment`.

#### fiscal_rule_adjustment
- **Type**: `float` | **Default**: `0.01` | **Constraints**: `(0, 1)`
- Tax rate increment when fiscal rule triggers (REQ-309). Added to current tax rate to stabilize debt.

### Nominal Rigidities Parameters (HANK v3: New Keynesian)

#### nominal_rigidities
- **Type**: `bool` | **Default**: `False`
- Enable New Keynesian sticky prices with Rotemberg adjustment costs, Taylor rule, NKPC, and Fisher equation (REQ-310..315). Default OFF preserves real-economy-only behavior.

```python
# Enable nominal rigidities
config = SimulationConfigV2(
    nominal_rigidities=True,
    rotemberg_cost=100.0,
    taylor_phi_pi=1.5,
    inflation_target=0.02,
)
```

#### rotemberg_cost
- **Type**: `float` | **Default**: `100.0` | **Constraints**: `>= 0`
- Rotemberg price adjustment cost parameter phi_p (REQ-310). Higher values mean more price stickiness. Only used when `nominal_rigidities=True`.

#### taylor_phi_pi
- **Type**: `float` | **Default**: `1.5` | **Constraints**: `> 1.0`
- Taylor rule inflation coefficient (REQ-312). Must be > 1 for Taylor principle (determinacy). Only used when `nominal_rigidities=True`.

#### taylor_phi_y
- **Type**: `float` | **Default**: `0.125` | **Constraints**: `>= 0`
- Taylor rule output gap coefficient (REQ-312). Only used when `nominal_rigidities=True`.

#### inflation_target
- **Type**: `float` | **Default**: `0.02`
- Target inflation rate pi_bar (REQ-312). Default 2% annualized. Only used when `nominal_rigidities=True`.

#### elasticity_sub
- **Type**: `float` | **Default**: `6.0` | **Constraints**: `> 1.0`
- Elasticity of substitution between varieties epsilon in CES aggregator (REQ-310). Determines steady-state markup: `markup = epsilon / (epsilon - 1)`. Only used when `nominal_rigidities=True`.

#### wage_rigidity
- **Type**: `bool` | **Default**: `False`
- Enable optional wage Rotemberg adjustment costs (REQ-315). Only used when `nominal_rigidities=True`.

#### rotemberg_wage_cost
- **Type**: `float` | **Default**: `50.0` | **Constraints**: `>= 0`
- Rotemberg wage adjustment cost parameter phi_w (REQ-315). Only used when `wage_rigidity=True`.

### Political Utility Parameters (HANK v3: Microfounded Politics)

#### political_lambda
- **Type**: `float` | **Default**: `0.05` | **Constraints**: `[0, 1]`
- Weight on political utility in the Bellman equation (REQ-401): `V = u(c,l,G) + lambda * v(C; theta) + beta * E[V']` where `v(C; theta)` is the agent's political utility from the constitution.

```python
# Increase weight on political preferences
config = SimulationConfigV2(political_lambda=0.1)
```

#### pure_bellman_politics
- **Type**: `bool` | **Default**: `False`
- When True, all political decisions (proposals, votes) are made purely by Bellman value function comparison, bypassing the LLM entirely (REQ-405). Useful for benchmark mode with microfounded politics.

```python
# Pure Bellman political economy
config = SimulationConfigV2(
    benchmark_mode=True,
    pure_bellman_politics=True,
    political_lambda=0.05,
)
```

### Occupational Choice Parameters (HANK v3)

#### use_bellman_occ_choice
- **Type**: `bool` | **Default**: `False`
- Use Bellman-based occupational choice comparing worker value `V^W(a,z)` vs entrepreneur value `V^E(a,z,e)` (REQ-110..114). Default OFF for backward compatibility.

```python
# Enable Bellman occupational choice
config = SimulationConfigV2(use_bellman_occ_choice=True)
```

### Feature Flags

#### fix_entrepreneur_budget
- **Type**: `bool` | **Default**: `False`
- When True, entrepreneurs receive only firm profit as income (not labor income), and their labor supply is excluded from market aggregation. Default OFF preserves v2 backward compatibility.

## Configuration Examples

### Quick Test (Benchmark)

```python
config = SimulationConfigV2(
    num_agents=20,
    max_periods=20,
    seed=42,
    benchmark_mode=True,
)
# Runs in seconds, $0 cost
```

### Default LLM Run

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
)
# ~$56 with Sonnet 4.5, ~$14 with Haiku 4.5
```

### High Inequality Scenario

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    initial_wealth_mean=100.0,
    initial_wealth_std=60.0,   # High initial inequality
    sigma_z=0.4,               # High income volatility
)
```

### Volatile Economy

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=200,
    seed=42,
    sigma_a=0.05,              # Large aggregate shocks
    sigma_z=0.3,               # Large idiosyncratic shocks
    rho_a=0.8,                 # Lower persistence (faster mean reversion)
)
```

### Capital-Intensive Economy

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    alpha=0.45,                # Higher capital share
    delta=0.05,                # Lower depreciation (capital lasts longer)
    min_firm_capital=5.0,      # Easier firm entry
)
```

### Rapid Constitutional Evolution

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    proposal_interval=2,       # Proposals every 2 periods
    observer_interval=2,       # Track closely
)
```

### Large-Scale Research Run

```python
config = SimulationConfigV2(
    num_agents=200,
    max_periods=500,
    seed=42,
    proposal_interval=10,
    observer_interval=10,
    llm_model="claude-haiku-4-5-20251001",  # Cheaper model
    llm_batch_size=20,         # Larger batches
)
# ~$230 with Haiku 4.5
```

### HANK Model with Two Assets and Government (HANK v3)

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    benchmark_mode=True,
    solver_method="egm",
    two_asset_mode=True,        # Liquid/illiquid assets
    chi_0=0.01,                 # Linear adjustment cost
    chi_1=0.005,                # Quadratic adjustment cost
    b_min=0.0,                  # No borrowing on liquid assets
    initial_debt=50.0,          # Government debt
    debt_gdp_max=1.5,           # Fiscal rule threshold
    fiscal_rule_adjustment=0.01, # Tax adjustment
)
```

### HANK with Nominal Rigidities (New Keynesian)

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    benchmark_mode=True,
    nominal_rigidities=True,    # Enable NK block
    rotemberg_cost=100.0,       # Price stickiness
    taylor_phi_pi=1.5,          # Taylor rule
    taylor_phi_y=0.125,
    inflation_target=0.02,      # 2% target
    elasticity_sub=6.0,
    initial_debt=50.0,          # Government bonds
)
```

### KFE Distribution Tracking Mode

```python
config = SimulationConfigV2(
    num_agents=50,              # Used only for initialization
    max_periods=100,
    seed=42,
    benchmark_mode=True,
    distribution_mode="kfe",    # Grid-based distribution
    kfe_convergence_tolerance=1e-10,
    kfe_max_iterations=10000,
)
```

### Walrasian Clearing with Heterogeneous Firms

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    benchmark_mode=True,
    market_clearing_method="walrasian",  # Bisection
    tatonnement_max_iter=100,
    tatonnement_tolerance=1e-6,
)
```

### Microfounded Political Economy (Pure Bellman)

```python
config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    benchmark_mode=True,
    pure_bellman_politics=True,  # No LLM for politics
    political_lambda=0.05,       # Weight on political utility
    use_bellman_occ_choice=True, # Bellman occupational choice
)
```

## CLI Configuration

The v2 CLI supports all key configuration flags:

```bash
emergent-constitution-v2 \
    -n 100 \                    # num_agents
    -t 200 \                    # max_periods
    -s 123 \                    # seed
    --rho-z 0.9 \               # rho_z
    --sigma-z 0.2 \             # sigma_z
    --rho-A 0.95 \              # rho_a
    --sigma-A 0.01 \            # sigma_a
    --alpha 0.33 \              # alpha
    --delta 0.1 \               # delta
    --a-min 0.0 \               # a_min
    --proposal-interval 5 \
    --observer-interval 5 \
    --benchmark \               # benchmark_mode
    --llm-provider anthropic \
    --llm-model claude-sonnet-4-5-20250929 \
    -o output.md \              # Output file
    --json \                    # JSON output
    -q                          # Quiet (suppress logging)
```

## Validation

All values are validated by Pydantic on instantiation:

```python
# These raise ValidationError:
SimulationConfigV2(num_agents=10)        # Error: must be >= 20
SimulationConfigV2(alpha=1.5)            # Error: must be in (0, 1)
SimulationConfigV2(rho_z=0.0)            # Error: must be > 0
SimulationConfigV2(tatonnement_tolerance=-1) # Error: must be > 0
```

**Automatic constraint**: `benchmark_mode=True` automatically sets `use_llm=False` via a model validator.

## v1 Configuration (Backward Compatible)

The v1 `SimulationConfig` is still available:

```python
from emergent_constitution import SimulationConfig

config = SimulationConfig(
    num_agents=50,
    max_ticks=100,
    seed=42,
    initial_wealth_mean=100.0,
    initial_wealth_std=30.0,
    proposal_interval=5,
    observer_interval=5,
    trade_interval=5,
    coalition_interval=10,
    use_llm=False,
    llm_fraction=0.1,
)
```
