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
- **Type**: `int` | **Default**: `5` | **Constraints**: `[2, 50]`
- Number of grid points for the Rouwenhorst discretization of the productivity Markov chain.

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

#### tatonnement_max_iter
- **Type**: `int` | **Default**: `100` | **Constraints**: `>= 1`
- Maximum iterations for the tatonnement price-finding algorithm.

#### tatonnement_tolerance
- **Type**: `float` | **Default**: `1e-6` | **Constraints**: `> 0`
- Convergence tolerance for market clearing (PROP-003).

#### tatonnement_step_size
- **Type**: `float` | **Default**: `0.01` | **Constraints**: `> 0`
- Price adjustment step size per iteration.

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
- **Type**: `Literal["vfi", "egm"]` | **Default**: `"egm"`
- Numerical solver method: Value Function Iteration or Endogenous Grid Method (REQ-036).

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
