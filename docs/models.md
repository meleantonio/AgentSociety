# Data Models Reference

All data models use Pydantic for validation and serialization. v2 models extend the simulation with DSGE-HA entities (households, firms, markets, shocks, decisions).

## v2 Core Models

### HouseholdState

**Location**: `src/emergent_constitution/models/household.py`

Replaces v1 `AgentState` with full DSGE-HA household model.

```python
class HouseholdState(BaseModel):
    id: str
    wealth: float                    # Assets a_t (>= 0)
    productivity: float              # Idiosyncratic z_t from Markov chain
    productivity_index: int          # Index into transition matrix
    utility_params: UtilityParams    # Cobb-Douglas weights
    value_vector: ValueVector        # Equality vs liberty
    role: OccupationalRole           # worker/entrepreneur/researcher/unemployed
    firm_id: str | None              # Firm owned (if entrepreneur)
    coalition_id: str | None         # Coalition membership
    # Per-period outcomes (filled after decisions applied)
    consumption: float               # c_t for this period
    leisure: float                   # l_t (0=full work, 1=no work)
    labor_supply: float              # 1 - l_t
    savings: float                   # a_{t+1} - a_t
    income: float                    # w_t * z_t * labor_supply
    taxes_paid: float
    transfers_received: float
    realized_utility: float          # u(c_t, l_t, G_t)
```

### UtilityParams

**Location**: `src/emergent_constitution/models/household.py`

```python
class UtilityParams(BaseModel):
    alpha: float   # Weight on consumption (0-1)
    beta: float    # Weight on leisure (0-1)
    gamma: float   # Weight on public goods (0-1)
    beta_discount: float = 0.95  # Intertemporal discount factor (0-1)
```

Validation: `alpha + beta + gamma` must equal 1.0 (within 1e-6 tolerance).

### ValueVector

**Location**: `src/emergent_constitution/models/household.py`

```python
class ValueVector(BaseModel):
    equality: float  # 0-1, preference toward egalitarian policies
    liberty: float   # 0-1, preference toward libertarian policies
```

Validation: `equality + liberty` must equal 1.0.

### OccupationalRole

```python
class OccupationalRole(StrEnum):
    WORKER = "worker"
    ENTREPRENEUR = "entrepreneur"
    RESEARCHER = "researcher"
    UNEMPLOYED = "unemployed"
```

### FirmState

**Location**: `src/emergent_constitution/models/firm.py`

```python
class FirmState(BaseModel):
    id: str                          # e.g. "firm_0000"
    owner_id: str                    # HouseholdState.id of entrepreneur
    capital: float                   # K_f rented from savings (>= 0)
    labor_demand: float              # L_f effective labor units (>= 0)
    tfp: float                       # A_f firm-specific TFP (> 0)
    worker_ids: list[str]            # Agents currently employed
    rd_spend: float                  # R&D expenditure this period (>= 0)
    output: float                    # Y_f = A_f * K_f^α * L_f^(1-α) (>= 0)
    profit: float                    # π_f = Y_f - w*L_f - (r+δ)*K_f
```

### MarketState

**Location**: `src/emergent_constitution/models/market.py`

```python
class MarketState(BaseModel):
    wage: float                      # w_t labor market clearing price
    interest_rate: float             # r_t capital market clearing price
    aggregate_output: float          # Y_t total output
    aggregate_consumption: float     # C_t total consumption
    aggregate_investment: float      # I_t total investment
    government_spending: float       # G_t public goods spending
    market_clearing_error: float     # |excess demand| (PROP-003: < 1e-6)
    labor_excess_demand: float
    capital_excess_demand: float
```

### ShockState

**Location**: `src/emergent_constitution/models/shocks.py`

```python
class ShockState(BaseModel):
    productivity_grid: list[float]           # Rouwenhorst grid points
    transition_matrix: list[list[float]]     # Markov transition probabilities
    aggregate_tfp: float                     # Current A_t
    preference_shocks: dict[str, float]      # Agent-specific preference perturbations
```

## Decision Models

**Location**: `src/emergent_constitution/models/decisions.py`

### EconomicDecision

```python
class EconomicDecision(BaseModel):
    consumption: float  # c_t >= 0
    leisure: float      # l_t in [0, 1]
```

### EntrepreneurialDecision

```python
class EntrepreneurialDecision(BaseModel):
    create_firm: bool = False
    capital_investment: float = 0.0   # K_f to rent (>= 0)
    labor_demand: float = 0.0        # L_f to hire (>= 0)
    rd_spend: float = 0.0            # R&D expenditure (>= 0)
    close_firm: bool = False
```

### PoliticalDecision

```python
class PoliticalDecision(BaseModel):
    proposal: ConstitutionalProposal | None = None
    votes: dict[str, bool] = {}  # rule_name -> for/against
```

## Constitution Models

### ConstitutionV2

**Location**: `src/emergent_constitution/models/constitution.py`

Extensible, structured constitutional document. Unlike v1's fixed 4 fields, v2 supports an open-ended collection of named rules.

```python
class ConstitutionV2(BaseModel):
    rules: dict[str, ConstitutionalRule]  # name -> rule
    voting_rule: str = "majority_vote"    # Active voting procedure
```

**Key methods**:
- `get_tax_rules()` — Return all rules with `rule_type == TAX_SCHEDULE`
- `get_transfer_rules()` — Return all transfer program rules
- `get_active_voting_rule()` — Return the active voting procedure

### ConstitutionalRule

```python
class ConstitutionalRule(BaseModel):
    name: str                            # Unique identifier (e.g. "income_tax")
    rule_type: RuleType                  # Category
    parameters: dict[str, Any] = {}      # Type-specific params (e.g. {"rate": 0.2})
    description: str = ""                # Natural-language description
    enforcement_code: str = ""           # Python expression for enforcement (AST-sandboxed)
    version: int = 1                     # Incremented on modification
    enacted_period: int = 0              # Period this rule was enacted
```

### RuleType

```python
class RuleType(StrEnum):
    TAX_SCHEDULE = "tax_schedule"
    TRANSFER_PROGRAM = "transfer_program"
    PUBLIC_GOODS = "public_goods"
    MARKET_REGULATION = "market_regulation"
    VOTING_PROCEDURE = "voting_procedure"
    PROPERTY_RIGHTS = "property_rights"
    FIRM_REGULATION = "firm_regulation"
    CUSTOM = "custom"
```

### Default Constitution

`create_default_constitution()` returns a ConstitutionV2 with 5 baseline rules:

| Rule | Type | Parameters |
|------|------|------------|
| `flat_tax` | TAX_SCHEDULE | `{"rate": 0.0}` |
| `flat_transfer` | TRANSFER_PROGRAM | `{"method": "equal_share"}` |
| `public_goods_provision` | PUBLIC_GOODS | `{"fraction_of_revenue": 0.3}` |
| `majority_vote` | VOTING_PROCEDURE | `{"threshold": 0.5}` |
| `private_property` | PROPERTY_RIGHTS | `{"regime": "private"}` |

## Proposal and Voting Models

### ConstitutionalProposal (v2)

**Location**: `src/emergent_constitution/models/proposal.py`

```python
class ConstitutionalProposal(BaseModel):
    rule_name: str           # Rule to modify (or new rule name)
    rule_type: RuleType      # Category
    parameters: dict         # New parameters
    description: str = ""    # Explanation
    proposer_id: str = ""    # Agent who proposed
```

### VoteOutcomeV2

```python
class VoteOutcomeV2(BaseModel):
    proposal: ConstitutionalProposal
    passed: bool
    votes_for: int
    votes_against: int
    total_eligible: int
```

## Output Models

### HistoryEntryV2

**Location**: `src/emergent_constitution/models/history.py`

```python
class HistoryEntryV2(BaseModel):
    period: int
    gini: float                              # Wealth Gini (0-1)
    pareto_score: float                      # Pareto efficiency (0-1)
    aggregate_output: float                  # Y_t
    aggregate_consumption: float             # C_t
    aggregate_investment: float              # I_t
    mean_wealth: float
    median_wealth: float
    wealth_quantiles: list[float]            # [p10, p25, p50, p75, p90]
    unemployment_rate: float
    num_active_firms: int
    mean_firm_size: float
    aggregate_rd_spend: float
    social_welfare: float                    # Sum of realized utilities
    cumulative_welfare: float                # Discounted sum to date
    wage: float                              # w_t
    interest_rate: float                     # r_t
    rule_changes: list[str]
    constitution_snapshot: ConstitutionV2
```

### WelfareSummary

```python
class WelfareSummary(BaseModel):
    llm_total_welfare: float
    benchmark_total_welfare: float | None = None
    per_agent_comparison: dict[str, dict] | None = None
```

### SimulationOutputV2

```python
class SimulationOutputV2(BaseModel):
    constitution: ConstitutionV2
    history: list[HistoryEntryV2]
    final_households: list[HouseholdState]
    final_firms: list[FirmState]
    welfare_summary: WelfareSummary
    seed: int
    total_periods: int
```

### PeriodState

Internal state container used by LeadV2 during the simulation loop.

```python
class PeriodState(BaseModel):
    period: int
    households: list[HouseholdState]
    firms: list[FirmState]
    market: MarketState
    shocks: ShockState
    constitution: ConstitutionV2
    proposals: list[ConstitutionalProposal]
    votes: list[VoteOutcomeV2]
```

## v1 Models (Backward Compatible)

### AgentState

**Location**: `src/emergent_constitution/models/agent.py`

```python
class AgentState(BaseModel):
    id: str
    wealth: float            # >= 0.0
    productivity: float      # > 0.0
    utility_params: UtilityParams  # (v1 version without beta_discount)
    value_vector: ValueVector
    coalition_id: str | None
```

### Constitution (v1)

```python
class Constitution(BaseModel):
    property_rule: PropertyRule = PropertyRule.PRIVATE
    tax_rate: float = 0.0           # 0-1
    voting_rule: VotingRule = VotingRule.MAJORITY
    redistribution_rule: RedistributionRule = RedistributionRule.FLAT
```

### SimulationOutput (v1)

```python
class SimulationOutput(BaseModel):
    constitution: Constitution
    history: list[HistoryEntry]
    final_agent_states: list[AgentState]
    seed: int
    total_ticks: int
```

## Serialization

All models support JSON serialization via Pydantic:

```python
# Serialize
json_str = output.model_dump_json(indent=2)

# Deserialize
output = SimulationOutputV2.model_validate_json(json_str)

# Deep copy for immutability
new_household = household.model_copy(update={"wealth": new_wealth})
new_constitution = constitution.model_copy(deep=True)
```

## Validation Summary

| Model | Validators |
|-------|-----------|
| UtilityParams | `alpha + beta + gamma == 1.0` |
| ValueVector | `equality + liberty == 1.0` |
| HouseholdState | `wealth >= 0`, `productivity > 0`, `0 <= leisure <= 1` |
| FirmState | `capital >= 0`, `labor_demand >= 0`, `tfp > 0` |
| SimulationConfigV2 | `benchmark_mode => use_llm=False`, all numeric bounds |
| ConstitutionalRule | AST validation of `enforcement_code` |
