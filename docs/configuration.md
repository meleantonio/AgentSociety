# Configuration Reference

## SimulationConfig

**Location**: `src/emergent_constitution/config.py`

The `SimulationConfig` class controls all aspects of a simulation run. It uses Pydantic for validation and provides sensible defaults for quick experimentation.

## Complete Field Reference

### Basic Parameters

#### num_agents

- **Type**: `int`
- **Default**: `50`
- **Constraints**: `≥ 2`
- **Description**: Number of citizen-agents in the simulation. Each agent has unique endowments, preferences, and values.

**Example**:
```python
config = SimulationConfig(num_agents=100)
```

**Performance**: O(N²) operations in Pareto computation and some coalition logic. 50-100 agents is typical; 500+ may be slow.

#### max_ticks

- **Type**: `int`
- **Default**: `100`
- **Constraints**: `≥ 1`
- **Description**: Maximum number of simulation ticks to execute. Each tick includes production, voting (if interval), trading (if interval), and potential observation.

**Example**:
```python
config = SimulationConfig(max_ticks=200)
```

**Guidance**: 100 ticks provides meaningful constitutional evolution. Use 200-500 for long-term institutional stability analysis.

#### seed

- **Type**: `int`
- **Default**: `42`
- **Constraints**: Any integer
- **Description**: Random number generator seed for complete reproducibility (PROP-001). Same seed + config = identical results.

**Example**:
```python
config = SimulationConfig(seed=123456)
```

**Important**: All randomness (initialization, tie-breaks, agent selection) uses this seed. Change it to explore different evolutionary paths with the same parameters.

### Initial Distributions

Initial agent endowments are drawn from normal distributions. Values are clamped to ensure validity (e.g., wealth ≥ 0, productivity > 0).

#### initial_wealth_mean

- **Type**: `float`
- **Default**: `100.0`
- **Constraints**: `> 0.0`
- **Description**: Mean of the normal distribution for initial agent wealth.

#### initial_wealth_std

- **Type**: `float`
- **Default**: `30.0`
- **Constraints**: `≥ 0.0`
- **Description**: Standard deviation of initial wealth distribution. Higher values create more initial inequality.

**Example**:
```python
# High initial inequality
config = SimulationConfig(
    initial_wealth_mean=100.0,
    initial_wealth_std=50.0  # ~50% coefficient of variation
)
```

#### initial_productivity_mean

- **Type**: `float`
- **Default**: `10.0`
- **Constraints**: `> 0.0`
- **Description**: Mean of the normal distribution for initial agent productivity (output per tick).

#### initial_productivity_std

- **Type**: `float`
- **Default**: `3.0`
- **Constraints**: `≥ 0.0`
- **Description**: Standard deviation of productivity distribution.

**Example**:
```python
# Homogeneous productivity (low inequality source)
config = SimulationConfig(
    initial_productivity_mean=10.0,
    initial_productivity_std=1.0
)

# Heterogeneous productivity (high inequality source)
config = SimulationConfig(
    initial_productivity_mean=10.0,
    initial_productivity_std=5.0
)
```

### Simulation Intervals

These parameters control the frequency of different actions. All intervals are measured in ticks.

#### proposal_interval

- **Type**: `int`
- **Default**: `5`
- **Constraints**: `≥ 1`
- **Description**: Agents can propose rule changes every K ticks. Lower values = more frequent proposals and faster constitutional evolution.

**Example**:
```python
# Rapid constitutional evolution
config = SimulationConfig(proposal_interval=2)

# Slow, stable evolution
config = SimulationConfig(proposal_interval=20)
```

#### observer_interval

- **Type**: `int`
- **Default**: `5`
- **Constraints**: `≥ 1`
- **Description**: Observer computes statistics every K ticks. Lower values = more detailed history but larger memory footprint.

**Example**:
```python
# Detailed tracking (every tick)
config = SimulationConfig(observer_interval=1)

# Coarse tracking (save memory)
config = SimulationConfig(observer_interval=20)
```

**Performance**: Gini and Pareto computations are O(N) and O(N²) respectively. Larger intervals reduce computational cost.

#### trade_interval

- **Type**: `int`
- **Default**: `5`
- **Constraints**: `≥ 1`
- **Description**: Agents can submit trade offers every K ticks. Trades are bilateral wealth transfers (e.g., buying labor).

**Example**:
```python
# Frequent trading (more market activity)
config = SimulationConfig(trade_interval=2)

# Infrequent trading
config = SimulationConfig(trade_interval=10)
```

#### coalition_interval

- **Type**: `int`
- **Default**: `10`
- **Constraints**: `≥ 1`
- **Description**: Coalitions reform every K ticks based on value vector similarity. Lower values = more dynamic coalition structure.

**Example**:
```python
# Stable coalitions
config = SimulationConfig(coalition_interval=50)

# Dynamic coalitions
config = SimulationConfig(coalition_interval=5)
```

### LLM Integration (Optional)

These parameters enable LLM-based reasoning for a subset of agents. This is experimental and requires a custom `CitizenLLM` implementation.

#### use_llm

- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to use LLM-based citizen logic for proposal generation and vote reasoning.

**Example**:
```python
from emergent_constitution import Lead, SimulationConfig
from my_llm_module import MyCitizenLLM

config = SimulationConfig(use_llm=True, llm_fraction=0.1)
llm_citizen = MyCitizenLLM()
lead = Lead(config, citizen_llm=llm_citizen)
```

**Important**: Requires implementing the `CitizenLLM` protocol from `llm_citizen.py` and passing it to `Lead.__init__()`.

#### llm_fraction

- **Type**: `float`
- **Default**: `0.1`
- **Constraints**: `0.0 ≤ f ≤ 1.0`
- **Description**: Fraction of agents that use LLM reasoning each tick (0.0 = none, 1.0 = all). Used to manage API costs.

**Example**:
```python
# 10% of agents use LLM each tick
config = SimulationConfig(use_llm=True, llm_fraction=0.1)

# All agents use LLM (expensive!)
config = SimulationConfig(use_llm=True, llm_fraction=1.0)
```

**Cost Management**: For 50 agents with `llm_fraction=0.1`, approximately 5 LLM calls per proposal/vote round.

## Configuration Examples

### Default Configuration

```python
from emergent_constitution import SimulationConfig

config = SimulationConfig()
# Equivalent to:
# SimulationConfig(
#     num_agents=50,
#     max_ticks=100,
#     seed=42,
#     initial_wealth_mean=100.0,
#     initial_wealth_std=30.0,
#     initial_productivity_mean=10.0,
#     initial_productivity_std=3.0,
#     proposal_interval=5,
#     observer_interval=5,
#     trade_interval=5,
#     coalition_interval=10,
#     use_llm=False,
#     llm_fraction=0.1,
# )
```

### Small Fast Simulation

```python
config = SimulationConfig(
    num_agents=10,
    max_ticks=20,
    seed=42,
    proposal_interval=2,
    observer_interval=2,
)
# Runs in <100ms, useful for testing
```

### Large-Scale Simulation

```python
config = SimulationConfig(
    num_agents=200,
    max_ticks=500,
    seed=42,
    proposal_interval=10,
    observer_interval=10,
    trade_interval=10,
    coalition_interval=20,
)
# Takes several seconds, produces rich dynamics
```

### High Inequality Scenario

```python
config = SimulationConfig(
    num_agents=50,
    max_ticks=100,
    seed=42,
    initial_wealth_mean=100.0,
    initial_wealth_std=60.0,  # High initial wealth inequality
    initial_productivity_mean=10.0,
    initial_productivity_std=6.0,  # High productivity inequality
)
# Agents start with very unequal endowments
```

### Rapid Constitutional Change

```python
config = SimulationConfig(
    num_agents=50,
    max_ticks=100,
    seed=42,
    proposal_interval=1,  # Proposals every tick
    observer_interval=1,  # Track every change
)
# Constitution evolves very rapidly
```

### Stable Long-Run Evolution

```python
config = SimulationConfig(
    num_agents=50,
    max_ticks=500,
    seed=42,
    proposal_interval=20,  # Infrequent proposals
    observer_interval=10,
    coalition_interval=50,  # Very stable coalitions
)
# Institutions stabilize over long horizon
```

## CLI Configuration

The CLI (`python -m emergent_constitution`) supports a subset of configuration options:

```bash
python -m emergent_constitution \
    -n 100 \                      # num_agents
    -t 200 \                      # max_ticks
    -s 123 \                      # seed
    --proposal-interval 10 \
    --observer-interval 10
```

For full control, use the programmatic API.

## Validation

All configuration values are validated by Pydantic on instantiation:

```python
# This raises ValidationError:
config = SimulationConfig(num_agents=1)  # Error: num_agents must be ≥2

config = SimulationConfig(tax_rate=1.5)  # Error: tax_rate must be in [0,1]

config = SimulationConfig(max_ticks=-10)  # Error: max_ticks must be ≥1
```

## Determinism Guarantees

Given identical `SimulationConfig` (including seed), the simulation produces:

- Identical initial agent states (wealth, productivity, utilities, values)
- Identical proposals and votes at each tick
- Identical trade offers and executions
- Identical final constitution and history

This is guaranteed by:
1. All random operations use the seeded `SimulationRNG`
2. Tie-breaking is deterministic (status quo wins)
3. Agent iteration order is deterministic (by ID)
4. No external inputs or nondeterministic operations

## Performance Guidelines

### Memory Usage

- **Per agent**: ~1KB (state + history)
- **Per tick**: ~50KB for 50 agents (proposals, votes, trades)
- **Total**: O(N × ticks) for full history

Example: 50 agents × 100 ticks × 50KB ≈ 5MB

For memory-constrained environments, increase `observer_interval` to reduce history size.

### Computational Cost

- **Production/taxation**: O(N) per tick
- **Voting**: O(N × proposals) per proposal tick
- **Gini coefficient**: O(N log N) per observer tick
- **Pareto efficiency**: O(N²) per observer tick
- **Coalition formation**: O(N × k) per coalition tick (k = num_coalitions)

Bottleneck is typically Pareto computation. For N > 200, consider disabling it or increasing `observer_interval`.

### Typical Runtimes (M1 MacBook Pro)

- 50 agents × 100 ticks: ~0.5s
- 100 agents × 100 ticks: ~1.5s
- 200 agents × 200 ticks: ~15s
- 500 agents × 500 ticks: ~5 minutes

## Best Practices

1. **Start with defaults**: The default config produces interesting emergent behavior
2. **Use consistent seeds**: For reproducibility in papers/reports, document the seed
3. **Balance intervals**: Proposal and observer intervals should typically be similar
4. **Test small first**: Run with 10 agents × 20 ticks to validate logic before scaling
5. **Monitor inequality**: High initial inequality (std > 50% of mean) creates polarized dynamics
6. **Adjust for goals**: Fast evolution = low proposal_interval; stable dynamics = high proposal_interval

## Extending Configuration

To add new configuration parameters:

1. Add field to `SimulationConfig` with Pydantic constraints
2. Update `Lead.__init__()` to consume the parameter
3. Document the field in this file
4. Add CLI argument in `__main__.py` if appropriate
5. Add tests in `tests/test_config.py`

Example:
```python
class SimulationConfig(BaseModel):
    # ... existing fields ...
    max_proposals_per_agent: int = Field(default=1, ge=1)
```
