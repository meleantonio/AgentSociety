# Data Models Reference

All data models are implemented using Pydantic for validation and serialization.

## Core Models

### AgentState

**Location**: `src/emergent_constitution/models/agent.py`

Represents the complete state of a single citizen-agent at a point in time.

```python
class AgentState(BaseModel):
    id: str
    wealth: float
    productivity: float
    utility_params: UtilityParams
    value_vector: ValueVector
    coalition_id: str | None
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | str | Required | Unique agent identifier (e.g., "agent_001") |
| `wealth` | float | ≥ 0.0 | Current wealth/consumption capacity |
| `productivity` | float | > 0.0 | Production capacity per tick |
| `utility_params` | UtilityParams | Required | Cobb-Douglas utility weights |
| `value_vector` | ValueVector | Required | Ideological position (equality vs liberty) |
| `coalition_id` | str \| None | Optional | ID of coalition this agent belongs to |

**Invariants**:
- Wealth must remain non-negative (enforced by economic engine)
- Productivity must be positive
- Utility params and value vector must sum to 1.0

### UtilityParams

**Location**: `src/emergent_constitution/models/agent.py`

Cobb-Douglas utility function weights: U = c^α × l^β × g^γ

```python
class UtilityParams(BaseModel):
    alpha: float  # Weight on private consumption
    beta: float   # Weight on leisure
    gamma: float  # Weight on public goods
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `alpha` | float | 0.0 ≤ α ≤ 1.0 | Weight on private consumption |
| `beta` | float | 0.0 ≤ β ≤ 1.0 | Weight on leisure (fixed at 1.0 in current implementation) |
| `gamma` | float | 0.0 ≤ γ ≤ 1.0 | Weight on public goods |

**Validation**: α + β + γ must equal 1.0 (within 1e-6 tolerance)

**Example**:
```python
# Agent who values consumption and public goods equally, no leisure preference
UtilityParams(alpha=0.5, beta=0.0, gamma=0.5)
```

### ValueVector

**Location**: `src/emergent_constitution/models/agent.py`

Agent's ideological position on the equality-liberty spectrum.

```python
class ValueVector(BaseModel):
    equality: float  # Preference weight toward equality
    liberty: float   # Preference weight toward liberty
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `equality` | float | 0.0 ≤ e ≤ 1.0 | Weight favoring egalitarian policies (redistribution, communal property) |
| `liberty` | float | 0.0 ≤ l ≤ 1.0 | Weight favoring libertarian policies (private property, low taxes) |

**Validation**: equality + liberty must equal 1.0 (within 1e-6 tolerance)

**Example**:
```python
# Strongly egalitarian agent
ValueVector(equality=0.9, liberty=0.1)

# Balanced agent
ValueVector(equality=0.5, liberty=0.5)

# Strongly libertarian agent
ValueVector(equality=0.1, liberty=0.9)
```

## Constitution Models

### Constitution

**Location**: `src/emergent_constitution/models/constitution.py`

The active ruleset governing the simulation. Mutable through voting.

```python
class Constitution(BaseModel):
    property_rule: PropertyRule
    tax_rate: float
    voting_rule: VotingRule
    redistribution_rule: RedistributionRule
```

**Fields**:

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `property_rule` | PropertyRule | PRIVATE | Enum | How production output is allocated |
| `tax_rate` | float | 0.0 | 0.0 ≤ t ≤ 1.0 | Fraction of output collected as tax |
| `voting_rule` | VotingRule | MAJORITY | Enum | Threshold for proposals to pass |
| `redistribution_rule` | RedistributionRule | FLAT | Enum | How tax revenue is redistributed |

### PropertyRule (Enum)

**Location**: `src/emergent_constitution/models/constitution.py`

```python
class PropertyRule(StrEnum):
    PRIVATE = "private"    # Each agent keeps their own productivity
    COMMUNAL = "communal"  # Total productivity pooled and split equally
    MIXED = "mixed"        # 50% private + 50% pooled
```

### VotingRule (Enum)

**Location**: `src/emergent_constitution/models/constitution.py`

```python
class VotingRule(StrEnum):
    MAJORITY = "majority"            # >50% required
    SUPERMAJORITY = "supermajority"  # ≥66.67% required
    UNANIMITY = "unanimity"          # 100% required
```

### RedistributionRule (Enum)

**Location**: `src/emergent_constitution/models/constitution.py`

```python
class RedistributionRule(StrEnum):
    NONE = "none"              # Tax revenue discarded
    FLAT = "flat"              # Equal distribution to all agents
    PROGRESSIVE = "progressive" # More to below-median wealth agents
```

## Proposal and Voting Models

### Proposal

**Location**: `src/emergent_constitution/models/proposal.py`

A rule-change proposal submitted by a citizen.

```python
class Proposal(BaseModel):
    rule_key: str          # Constitution field to change
    proposed_value: Any    # New value for the field
    proposer_id: str       # Agent who proposed this
```

**Valid rule_keys**: `property_rule`, `tax_rate`, `voting_rule`, `redistribution_rule`

**Example**:
```python
# Proposal to increase tax rate to 20%
Proposal(
    rule_key="tax_rate",
    proposed_value=0.2,
    proposer_id="agent_042"
)
```

### VoteOutcome

**Location**: `src/emergent_constitution/models/proposal.py`

Result of voting on a single proposal.

```python
class VoteOutcome(BaseModel):
    proposal: Proposal
    passed: bool
    votes_for: int
    votes_against: int
    total_eligible: int
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `proposal` | Proposal | Required | The proposal that was voted on |
| `passed` | bool | Required | Whether the proposal was accepted |
| `votes_for` | int | ≥ 0 | Number of yes votes |
| `votes_against` | int | ≥ 0 | Number of no votes |
| `total_eligible` | int | ≥ 0 | Total number of eligible voters |

### TradeOffer

**Location**: `src/emergent_constitution/models/proposal.py`

A bilateral trade offer between two agents (wealth transfer).

```python
class TradeOffer(BaseModel):
    seller_id: str    # Agent receiving wealth
    buyer_id: str     # Agent paying wealth
    amount: float     # Wealth transferred (must be positive)
```

**Validation**: Amount must be > 0.0 and buyer must have sufficient wealth.

## State Models

### TickState

**Location**: `src/emergent_constitution/models/tick.py`

Complete simulation state at a single tick.

```python
class TickState(BaseModel):
    tick: int
    agent_states: list[AgentState]
    constitution: Constitution
    proposals_this_tick: list[Proposal]
    votes: list[VoteOutcome]
    trades_this_tick: list[TradeOffer]
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tick` | int | Required | Current tick number (0-indexed) |
| `agent_states` | list[AgentState] | Required | State of every agent |
| `constitution` | Constitution | Required | Active ruleset |
| `proposals_this_tick` | list[Proposal] | [] | Proposals submitted this tick |
| `votes` | list[VoteOutcome] | [] | Vote outcomes this tick |
| `trades_this_tick` | list[TradeOffer] | [] | Trades executed this tick |

## Output Models

### HistoryEntry

**Location**: `src/emergent_constitution/models/history.py`

Aggregate statistics recorded at an observation tick.

```python
class HistoryEntry(BaseModel):
    tick: int
    gini: float
    total_output: float
    mean_wealth: float
    median_wealth: float
    pareto_score: float
    rule_changes: list[str]
    constitution_snapshot: Constitution
    num_coalitions: int
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `tick` | int | ≥ 0 | Tick when this observation was taken |
| `gini` | float | 0.0 ≤ g ≤ 1.0 | Gini coefficient (0=perfect equality, 1=perfect inequality) |
| `total_output` | float | ≥ 0.0 | Total production output this tick |
| `mean_wealth` | float | ≥ 0.0 | Mean agent wealth |
| `median_wealth` | float | ≥ 0.0 | Median agent wealth |
| `pareto_score` | float | 0.0 ≤ p ≤ 1.0 | Pareto efficiency estimate (1.0=fully efficient) |
| `rule_changes` | list[str] | Default [] | Rule change descriptions this period |
| `constitution_snapshot` | Constitution | Required | Constitution copy at this tick |
| `num_coalitions` | int | ≥ 0 | Number of distinct coalitions |

### SimulationOutput

**Location**: `src/emergent_constitution/models/history.py`

Final output of a complete simulation run.

```python
class SimulationOutput(BaseModel):
    constitution: Constitution
    history: list[HistoryEntry]
    final_agent_states: list[AgentState]
    seed: int
    total_ticks: int
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `constitution` | Constitution | Required | Final constitution after all ticks |
| `history` | list[HistoryEntry] | Required | All observation entries |
| `final_agent_states` | list[AgentState] | Required | Agent states at last tick |
| `seed` | int | Required | RNG seed used for this run (for reproducibility) |
| `total_ticks` | int | ≥ 0 | Number of ticks executed |

**Serialization**:
```python
# Convert to JSON
json_str = output.model_dump_json(indent=2)

# Convert from JSON
output = SimulationOutput.model_validate_json(json_str)
```

## Configuration Models

### SimulationConfig

**Location**: `src/emergent_constitution/config.py`

Top-level configuration for a simulation run.

```python
class SimulationConfig(BaseModel):
    num_agents: int
    max_ticks: int
    seed: int
    initial_wealth_mean: float
    initial_wealth_std: float
    initial_productivity_mean: float
    initial_productivity_std: float
    proposal_interval: int
    observer_interval: int
    trade_interval: int
    coalition_interval: int
    use_llm: bool
    llm_fraction: float
```

See [Configuration Reference](configuration.md) for complete details.

## Coalition Models

### CoalitionInfo

**Location**: `src/emergent_constitution/coalition.py`

Information about a single coalition.

```python
class CoalitionInfo(BaseModel):
    id: str              # Coalition identifier
    member_ids: list[str]  # Agent IDs in this coalition
    mean_equality: float   # Mean equality value of members
    mean_liberty: float    # Mean liberty value of members
```

## Validation and Constraints

### Pydantic Features Used

1. **Field Constraints**: `ge`, `gt`, `le`, `lt` for numeric bounds
2. **Model Validators**: Custom validation logic (e.g., weights sum to 1.0)
3. **Type Coercion**: Automatic conversion when safe
4. **Serialization**: JSON export/import with `model_dump_json()` / `model_validate_json()`
5. **Deep Copying**: `model_copy(deep=True)` for immutable state management

### Custom Validators

**UtilityParams and ValueVector**: Both enforce that their weights sum to 1.0 within 1e-6 tolerance using `@model_validator(mode="after")`.

**Constitution**: Field values are constrained by enum types and numeric bounds.

**Proposal**: Validated against `CONSTITUTION_FIELDS` dictionary to ensure rule_key exists and proposed_value is compatible.

## Type Annotations

All models use modern Python type hints (requires Python 3.10+):

- `str | None` instead of `Optional[str]`
- `list[AgentState]` instead of `List[AgentState]`
- `dict[str, float]` instead of `Dict[str, float]`

## Immutability Pattern

Models are designed for functional-style updates:

```python
# Create new state with updated wealth
new_agent = agent.model_copy(update={"wealth": agent.wealth + 10.0})

# Deep copy for nested structures
new_constitution = constitution.model_copy(deep=True)
```

This ensures state immutability within each tick, supporting determinism and testability.
