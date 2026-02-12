# Design Specification v2: DSGE-HA with LLM Agents

> **Traceability:** Every section references the requirement IDs it implements. See `spec/requirements.md` for the full EARS specification.

---

## 1. Architecture Overview

The system is a single-process, discrete-time simulation where heterogeneous agents make economic, entrepreneurial, and political decisions via LLM reasoning, within a DSGE-HA market-clearing framework.

```
SimulationConfig + RNG seed
        |
        v
+------------------+
| Initialization   |  Create N agents, default constitution, empty firm set
+------------------+
        |
        v
+--------------------------------------------+
| Lead (Governor) — period loop              |
|                                            |
| For each period t = 1..T:                  |
|   (1) Draw shocks (idiosyncratic + agg.)   |
|   (2) Market clearing (find w_t, r_t)      |
|   (3) LLM agent decisions (batched)        |
|   (4) Validate & enforce constraints       |
|   (5) Execute production & distribution    |
|   (6) Enforce constitution (tax/transfer)  |
|   (7) Proposals & votes (on interval)      |
|   (8) Update agent & firm states           |
|   (9) Observe & record statistics          |
+--------------------------------------------+
        |
        v
+------------------+
| Observer         |  Macro stats, welfare, history log
+------------------+
        |
        v
SimulationOutput: constitution + history + final states + welfare summary
```

**Key architectural change from v1:** The tick loop expands from 5 ad-hoc phases to a 9-step period lifecycle (REQ-033) with market clearing, stochastic shocks, and LLM-driven decisions as first-class steps.

### 1.1 Execution Model

- **Single process**, synchronous period loop. (Steering: single-process tick loop.)
- **LLM calls batched** per decision type within each period (REQ-029). All agents' economic decisions collected in one batch, then all entrepreneurial decisions, etc.
- **Market clearing** uses an inner iterative loop (tatonnement) to find equilibrium prices before agent decisions (REQ-014).
- **State immutability per period:** Agents receive read-only views of current state. Decisions are collected and applied atomically by the Lead.
- **Determinism:** Single `SimulationRNG` instance seeded once (PROP-001). LLM calls use temperature=0 and fixed seeds where the provider supports it.

### 1.2 Module Dependency Graph

```
config.py ──────────────────────┐
                                v
initialization.py ──> models/ ──> lead.py
                                    |
                 ┌──────────────────┼──────────────────┐
                 v                  v                   v
          shocks.py          market_clearing.py    llm_engine.py
                                    |                   |
                                    v                   v
                              economics.py        numerical_solver.py
                                    |
                 ┌──────────────────┼──────────────────┐
                 v                  v                   v
         constitution.py      observer.py         reporter.py
```

- **No circular dependencies.** Each module depends only on `models/` and modules above it.
- `llm_engine.py` and `numerical_solver.py` are interchangeable decision backends.

---

## 2. Data Models

All models are Pydantic `BaseModel` subclasses with validation. Frozen models use `model_config = ConfigDict(frozen=True)` for immutability where needed.

### 2.1 Household / Agent State

*Traceability:* REQ-001, REQ-002, REQ-003, REQ-004

```python
class UtilityParams(BaseModel):
    """Cobb-Douglas utility parameters: u(c,l,G) = c^alpha * l^beta * G^gamma."""
    alpha: float          # consumption weight, in (0,1)
    beta: float           # leisure weight, in (0,1)
    gamma: float          # public goods weight, in (0,1)
    beta_discount: float  # intertemporal discount factor, in (0,1)
    # Invariant: alpha + beta + gamma = 1.0 (enforced by validator)

class OccupationalRole(StrEnum):
    WORKER = "worker"
    ENTREPRENEUR = "entrepreneur"
    RESEARCHER = "researcher"
    UNEMPLOYED = "unemployed"

class HouseholdState(BaseModel):
    """Per-agent state each period. Replaces v1 AgentState."""
    id: str                              # e.g. "agent_0000"
    wealth: float                        # assets a_t (≥ a_min)
    productivity: float                  # idiosyncratic z_t (from Markov chain)
    productivity_index: int              # index into Markov transition matrix
    utility_params: UtilityParams
    value_vector: ValueVector            # retained from v1 (equality/liberty)
    role: OccupationalRole              # current occupational role
    firm_id: str | None = None           # firm owned (if entrepreneur)
    coalition_id: str | None = None      # coalition membership

    # Per-period decision outcomes (filled after decisions applied)
    consumption: float = 0.0             # c_t
    leisure: float = 0.0                 # l_t (0=full work, 1=no work)
    labor_supply: float = 0.0           # 1 - l_t
    savings: float = 0.0                # a_{t+1} - a_t
    income: float = 0.0                 # w_t * z_t * labor_supply
    taxes_paid: float = 0.0
    transfers_received: float = 0.0
    realized_utility: float = 0.0       # u(c_t, l_t, G_t) — ground truth (REQ-028)
```

### 2.2 Firm State

*Traceability:* REQ-006, REQ-007, REQ-008, REQ-009, REQ-010

```python
class FirmState(BaseModel):
    """Active firm entity."""
    id: str                              # e.g. "firm_0000"
    owner_id: str                        # HouseholdState.id of the entrepreneur
    capital: float                       # K_f (rented from aggregate savings)
    labor_demand: float                  # L_f (effective labor units hired)
    tfp: float                           # A_f (firm-specific total factor productivity)
    worker_ids: list[str] = []           # agents currently employed
    rd_spend: float = 0.0               # R&D expenditure this period
    output: float = 0.0                  # Y_f this period
    profit: float = 0.0                  # pi_f this period
```

### 2.3 Market State

*Traceability:* REQ-011, REQ-012, REQ-013, REQ-014

```python
class MarketState(BaseModel):
    """Economy-wide equilibrium prices and aggregates for period t."""
    wage: float                          # w_t (labor market clearing price)
    interest_rate: float                 # r_t (capital market clearing price)
    aggregate_output: float              # Y_t
    aggregate_consumption: float         # C_t
    aggregate_investment: float          # I_t
    government_spending: float           # G_t
    market_clearing_error: float         # |excess demand| (PROP-003: < 1e-6)
    labor_excess_demand: float           # labor demand - labor supply
    capital_excess_demand: float         # capital demand - capital supply
```

### 2.4 Shock State

*Traceability:* REQ-015, REQ-016, REQ-017

```python
class ShockState(BaseModel):
    """Realized shocks for period t."""
    productivity_grid: list[float]       # discrete z values (Rouwenhorst)
    transition_matrix: list[list[float]] # Markov transition probabilities
    aggregate_tfp: float                 # A_t (economy-wide TFP)
    preference_shocks: dict[str, float]  # agent_id -> beta perturbation (if enabled)
```

### 2.5 Constitution v2

*Traceability:* REQ-018, REQ-019, REQ-020, REQ-021, REQ-022, REQ-023

The constitution is no longer a fixed 4-field struct. It is an **extensible collection of named policy rules**.

```python
class RuleType(StrEnum):
    TAX_SCHEDULE = "tax_schedule"
    TRANSFER_PROGRAM = "transfer_program"
    PUBLIC_GOODS = "public_goods"
    MARKET_REGULATION = "market_regulation"
    VOTING_PROCEDURE = "voting_procedure"
    PROPERTY_RIGHTS = "property_rights"
    FIRM_REGULATION = "firm_regulation"
    CUSTOM = "custom"                    # open-ended for LLM-invented rules

class ConstitutionalRule(BaseModel):
    """A single named policy rule within the constitution."""
    name: str                            # unique identifier, e.g. "income_tax"
    rule_type: RuleType
    parameters: dict[str, Any]           # type-specific params (e.g. {"rate": 0.2, "brackets": [...]})
    description: str                     # natural-language description
    enforcement_code: str                # Python expression/function body for enforcement
    version: int = 1                     # incremented on modification
    enacted_period: int = 0              # period this rule was enacted

class Constitution(BaseModel):
    """Extensible, structured constitutional document."""
    rules: dict[str, ConstitutionalRule] = {}  # name -> rule
    voting_rule: str = "majority"        # active voting procedure name (references a rule)

    def get_tax_rules(self) -> list[ConstitutionalRule]: ...
    def get_transfer_rules(self) -> list[ConstitutionalRule]: ...
    def get_active_voting_rule(self) -> ConstitutionalRule: ...
```

**Default initial constitution** includes:
- `"flat_tax"`: RuleType.TAX_SCHEDULE, `{"rate": 0.0}` (no tax initially)
- `"flat_transfer"`: RuleType.TRANSFER_PROGRAM, `{"method": "equal_share"}`
- `"public_goods_provision"`: RuleType.PUBLIC_GOODS, `{"fraction_of_revenue": 0.3}`
- `"majority_vote"`: RuleType.VOTING_PROCEDURE, `{"threshold": 0.5}`
- `"private_property"`: RuleType.PROPERTY_RIGHTS, `{"regime": "private"}`

### 2.6 Proposal and Vote

*Traceability:* REQ-019, REQ-020, REQ-021

```python
class ConstitutionalProposal(BaseModel):
    """A proposal to add, modify, or remove a constitutional rule."""
    proposer_id: str
    action: Literal["add", "modify", "remove"]
    rule_name: str                       # target rule name
    rule_type: RuleType | None = None    # required for "add"
    parameters: dict[str, Any] | None = None
    description: str = ""                # natural-language rationale
    enforcement_code: str | None = None  # enforcement specification

class VoteOutcome(BaseModel):
    """Result of a constitutional vote."""
    proposal: ConstitutionalProposal
    passed: bool
    votes_for: int
    votes_against: int
    total_eligible: int
    voting_rule_used: str                # name of the voting rule applied
```

### 2.7 Agent Decision Schemas

*Traceability:* REQ-024, REQ-025, REQ-026

Each decision type has a dedicated schema for LLM structured output:

```python
class EconomicDecision(BaseModel):
    """Household consumption-savings-labor decision."""
    consumption: float                   # c_t ≥ 0
    leisure: float                       # l_t ∈ [0, 1]
    # savings derived: a_{t+1} = budget - consumption

class EntrepreneurialDecision(BaseModel):
    """Firm management decisions."""
    create_firm: bool = False            # start a new firm?
    capital_investment: float = 0.0      # K_f to rent
    labor_demand: float = 0.0           # L_f to hire
    rd_spend: float = 0.0              # R&D expenditure
    close_firm: bool = False            # liquidate?

class PoliticalDecision(BaseModel):
    """Proposal and voting decisions."""
    proposal: ConstitutionalProposal | None = None
    votes: dict[str, bool] = {}         # proposal_name -> for/against
```

### 2.8 Period State and History

*Traceability:* REQ-030, REQ-031, REQ-032

```python
class PeriodState(BaseModel):
    """Complete state snapshot for period t. Replaces v1 TickState."""
    period: int
    households: list[HouseholdState]
    firms: list[FirmState]
    market: MarketState
    shocks: ShockState
    constitution: Constitution
    proposals: list[ConstitutionalProposal] = []
    votes: list[VoteOutcome] = []

class HistoryEntry(BaseModel):
    """Observation record per observation interval."""
    period: int
    gini: float
    pareto_score: float
    aggregate_output: float
    aggregate_consumption: float
    aggregate_investment: float
    mean_wealth: float
    median_wealth: float
    wealth_quantiles: list[float]        # e.g. [p10, p25, p50, p75, p90]
    unemployment_rate: float
    num_active_firms: int
    mean_firm_size: float
    aggregate_rd_spend: float
    social_welfare: float                # sum of realized utilities
    cumulative_welfare: float            # discounted sum to date
    wage: float
    interest_rate: float
    rule_changes: list[str] = []
    constitution_snapshot: Constitution

class SimulationOutput(BaseModel):
    """Final output of the simulation. (REQ-032)"""
    constitution: Constitution
    history: list[HistoryEntry]
    final_households: list[HouseholdState]
    final_firms: list[FirmState]
    welfare_summary: WelfareSummary
    seed: int
    total_periods: int

class WelfareSummary(BaseModel):
    """Welfare comparison between LLM and benchmark. (REQ-032e)"""
    llm_total_welfare: float
    benchmark_total_welfare: float | None = None  # if benchmark mode was run
    per_agent_comparison: dict[str, dict] | None = None
```

### 2.9 Simulation Configuration

*Traceability:* REQ-034, REQ-035

```python
class SimulationConfig(BaseModel):
    # --- Core ---
    num_agents: int = 50                 # N ≥ 20 (REQ-001)
    max_periods: int = 100               # T ≥ 1
    seed: int = 42                       # RNG seed (PROP-001)

    # --- Shock parameters ---
    rho_z: float = 0.9                   # idiosyncratic persistence
    sigma_z: float = 0.2                 # idiosyncratic volatility
    num_z_states: int = 5                # Rouwenhorst grid points
    rho_A: float = 0.95                  # aggregate TFP persistence
    sigma_A: float = 0.01                # aggregate TFP volatility
    enable_preference_shocks: bool = False  # REQ-017 optional

    # --- Production ---
    alpha: float = 0.33                  # capital share in Cobb-Douglas
    delta: float = 0.1                   # depreciation rate
    min_firm_capital: float = 10.0       # minimum capital for firm creation

    # --- Household ---
    a_min: float = 0.0                   # borrowing limit (REQ-004)
    initial_wealth_mean: float = 100.0
    initial_wealth_std: float = 30.0

    # --- Intervals ---
    proposal_interval: int = 5           # constitutional proposal every K periods
    observer_interval: int = 5           # observation every K periods

    # --- Market clearing ---
    tatonnement_max_iter: int = 100      # max iterations for price finding
    tatonnement_tolerance: float = 1e-6  # PROP-003 tolerance
    tatonnement_step_size: float = 0.01  # price adjustment step

    # --- LLM ---
    use_llm: bool = True                 # default: LLM-driven (REQ-024)
    llm_provider: str = "anthropic"      # provider name
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_temperature: float = 0.0         # determinism (PROP-001)
    llm_batch_size: int = 10             # agents per batch call (REQ-029)
    llm_cache_enabled: bool = True       # cache identical contexts (REQ-029)

    # --- Benchmark ---
    benchmark_mode: bool = False         # numerical-only mode (REQ-037)
    solver_method: str = "egm"           # "vfi" or "egm" (REQ-036)

    # --- R&D ---
    rd_success_base_prob: float = 0.1    # base probability of TFP improvement
    rd_tfp_improvement_mean: float = 0.05
    rd_tfp_improvement_std: float = 0.02
```

---

## 3. Component Interfaces

### 3.1 Lead (Governor)

*Traceability:* REQ-033

```python
class Lead:
    def __init__(self, config: SimulationConfig) -> None: ...
    def run(self) -> SimulationOutput: ...

    # --- Period lifecycle (9 steps) ---
    def _advance_period(self, t: int) -> PeriodState:
        """Execute one period in order:
        1. _draw_shocks(t)
        2. _clear_markets()
        3. _collect_decisions()       # LLM or numerical
        4. _validate_constraints()    # project onto feasible set
        5. _execute_production()      # firms produce output
        6. _enforce_constitution()    # taxes, transfers, public goods
        7. _process_governance(t)     # proposals + votes (on interval)
        8. _update_states()           # apply all changes atomically
        9. _observe(t)               # record statistics (on interval)
        """
```

**Step details:**

| Step | Method | Requirements |
|------|--------|-------------|
| 1. Draw shocks | `_draw_shocks(t)` | REQ-015, REQ-016, REQ-017 |
| 2. Clear markets | `_clear_markets()` | REQ-011, REQ-012, REQ-013, REQ-014 |
| 3. Collect decisions | `_collect_decisions()` | REQ-024, REQ-025, REQ-026, REQ-029 |
| 4. Validate constraints | `_validate_constraints()` | REQ-004, REQ-005, PROP-002, PROP-004 |
| 5. Execute production | `_execute_production()` | REQ-007, REQ-008, REQ-009 |
| 6. Enforce constitution | `_enforce_constitution()` | REQ-022 |
| 7. Process governance | `_process_governance(t)` | REQ-019, REQ-020, REQ-021, REQ-023 |
| 8. Update states | `_update_states()` | REQ-001, REQ-003, REQ-006, REQ-010 |
| 9. Observe | `_observe(t)` | REQ-030, REQ-031 |

### 3.2 Shock Generator

*Traceability:* REQ-015, REQ-016, REQ-017

```python
# shocks.py

def rouwenhorst_discretize(
    rho: float, sigma: float, num_states: int
) -> tuple[list[float], list[list[float]]]:
    """Discretize AR(1) process into Markov chain using Rouwenhorst method.
    Returns (grid_values, transition_matrix).
    """

def draw_idiosyncratic_shocks(
    households: list[HouseholdState],
    transition_matrix: list[list[float]],
    grid: list[float],
    rng: SimulationRNG,
) -> list[HouseholdState]:
    """Transition each agent's productivity_index via Markov chain.
    Update productivity = grid[new_index].
    """

def draw_aggregate_tfp(
    prev_log_A: float, rho_A: float, sigma_A: float, rng: SimulationRNG
) -> float:
    """Draw A_t from log(A_t) = rho_A * log(A_{t-1}) + epsilon."""

def draw_preference_shocks(
    households: list[HouseholdState], rng: SimulationRNG, sigma: float
) -> dict[str, float]:
    """Optional: small perturbation to each agent's beta_discount."""
```

### 3.3 Market Clearing

*Traceability:* REQ-011, REQ-012, REQ-013, REQ-014, PROP-003

```python
# market_clearing.py

def clear_markets(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfig,
    rng: SimulationRNG,
) -> MarketState:
    """Find equilibrium (w_t, r_t) via tatonnement.

    Algorithm:
    1. Initialize w, r from previous period (or analytical guess).
    2. Loop (max tatonnement_max_iter iterations):
       a. Given (w, r), compute each firm's optimal K_f, L_f.
       b. Aggregate labor demand = sum(L_f), capital demand = sum(K_f).
       c. Aggregate labor supply = sum(z_i * (1 - l_i)) from agent decisions
          (use previous period's labor supply as proxy, or solve agent FOCs).
       d. Aggregate capital supply = sum(a_i) from household assets.
       e. Compute excess demand for labor and capital.
       f. Adjust: w += step * excess_labor_demand,
                  r += step * excess_capital_demand.
       g. If |excess_labor| < tol and |excess_capital| < tol: converge.
    3. Compute aggregate Y, C, I, G from equilibrium.
    4. Log market_clearing_error (PROP-003).
    """

def compute_firm_demands(
    firms: list[FirmState], wage: float, interest_rate: float,
    delta: float, alpha: float,
) -> tuple[float, float]:
    """Given prices, compute each firm's optimal K_f, L_f from FOCs
    of Y_f = A_f * K_f^alpha * L_f^(1-alpha).

    FOC capital: alpha * A_f * (K_f/L_f)^(alpha-1) = r + delta
    FOC labor:   (1-alpha) * A_f * (K_f/L_f)^alpha = w

    Returns (total_labor_demand, total_capital_demand).
    """
```

**Design choice — price-finding approach:** We use a simple tatonnement (iterative price adjustment) rather than a full general-equilibrium solver. This keeps the implementation tractable and aligns with the Aiyagari (1994) tradition of iterating on prices until supply equals demand. If agents' labor supply decisions depend on prices (they do, via the budget constraint), we use the **previous period's labor supply as initial proxy** during price adjustment, then re-solve agent decisions at final prices. This introduces a small approximation that is standard in computational DSGE-HA literature.

### 3.4 LLM Decision Engine

*Traceability:* REQ-024, REQ-025, REQ-026, REQ-027, REQ-029

```python
# llm_engine.py

class LLMDecisionEngine:
    """Manages LLM-based agent decision-making."""

    def __init__(self, config: SimulationConfig) -> None: ...

    def collect_economic_decisions(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution: Constitution,
        history_summary: str,
    ) -> dict[str, EconomicDecision]:
        """Batch LLM calls for consumption/savings/labor decisions.
        Each agent receives structured context (REQ-025).
        Returns validated JSON responses (REQ-026).
        Falls back to numerical solver on failure (REQ-027).
        Caches identical state-context pairs (REQ-029).
        """

    def collect_entrepreneurial_decisions(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        constitution: Constitution,
    ) -> dict[str, EntrepreneurialDecision]: ...

    def collect_political_decisions(
        self,
        households: list[HouseholdState],
        constitution: Constitution,
        market: MarketState,
    ) -> dict[str, PoliticalDecision]: ...

    # --- Internal ---

    def _build_context(
        self, agent: HouseholdState, market: MarketState,
        constitution: Constitution, history_summary: str,
    ) -> dict:
        """Build structured context per REQ-025:
        (a) personal state, (b) economic conditions,
        (c) current constitution, (d) recent history,
        (e) messages from other agents.
        """

    def _batch_call(
        self, prompts: list[dict], schema: type[BaseModel],
    ) -> list[BaseModel]:
        """Send batched LLM requests. Parse and validate JSON responses
        against the provided schema. Retry/fallback on failure.
        """

    def _cache_key(self, context: dict) -> str:
        """Deterministic hash of context for response caching."""
```

**LLM provider abstraction:** The engine uses a pluggable `LLMProvider` protocol:

```python
class LLMProvider(Protocol):
    def generate(self, messages: list[dict], schema: type[BaseModel]) -> str: ...
    def generate_batch(self, batch: list[list[dict]], schema: type[BaseModel]) -> list[str]: ...
```

Implementations: `AnthropicProvider`, `MockProvider` (deterministic, for testing).

### 3.5 Numerical Benchmark Solver

*Traceability:* REQ-036, REQ-037, REQ-027

```python
# numerical_solver.py

class NumericalSolver:
    """Computes approximately optimal household decisions via
    value function iteration (VFI) or endogenous grid method (EGM).
    Used as (a) fallback when LLM fails and (b) benchmark mode.
    """

    def __init__(self, config: SimulationConfig) -> None: ...

    def solve_household(
        self,
        agent: HouseholdState,
        wage: float,
        interest_rate: float,
        public_goods: float,
        tax_function: Callable[[float], float],
        transfer: float,
    ) -> EconomicDecision:
        """Solve single-agent Bellman equation:

        V(a, z) = max_{c, l} { u(c, l, G) + beta * E[V(a', z')] }
        s.t. a' = (1+r)*a + w*z*(1-l) - c - T(y) + Tr
             a' >= a_min, c >= 0, 0 <= l <= 1

        Returns optimal (consumption, leisure).
        """

    def solve_all(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution: Constitution,
    ) -> dict[str, EconomicDecision]:
        """Solve for all agents. Used in benchmark mode (REQ-037)."""
```

**EGM implementation sketch:**
1. Discretize asset grid (a) and productivity grid (z, from Rouwenhorst).
2. For each (a, z) point, use the Euler equation to find optimal consumption on an endogenous grid.
3. Interpolate back to the exogenous grid.
4. Iterate until value function converges (|V_new - V_old| < epsilon).

### 3.6 Economics Engine

*Traceability:* REQ-003, REQ-007, REQ-008, REQ-009, REQ-010

```python
# economics.py — pure functions, no side effects

def compute_budget(
    agent: HouseholdState, wage: float, interest_rate: float,
    tax: float, transfer: float,
) -> float:
    """Available resources: (1 + r) * a + w * z * labor_supply - tax + transfer.
    This is the maximum the agent can consume (saving nothing).
    """

def produce_output(firm: FirmState, alpha: float) -> float:
    """Y_f = A_f * K_f^alpha * L_f^(1-alpha). (REQ-007)"""

def distribute_firm_income(
    firm: FirmState, wage: float, interest_rate: float, delta: float,
) -> tuple[float, float, float]:
    """Returns (wages_paid, capital_cost, profit).
    wages = w * L_f
    capital_cost = (r + delta) * K_f
    profit = Y_f - wages - capital_cost  (REQ-008)
    """

def apply_rd_shock(
    firm: FirmState, rng: SimulationRNG, config: SimulationConfig,
) -> FirmState:
    """Stochastic TFP improvement from R&D spending. (REQ-009)
    prob = config.rd_success_base_prob * sqrt(rd_spend / mean_rd_spend)
    If success: A_f *= (1 + drawn_improvement).
    """

def liquidate_firm(firm: FirmState) -> float:
    """Return remaining capital to owner. (REQ-010)"""

def enforce_budget_constraint(
    decision: EconomicDecision, agent: HouseholdState,
    budget: float, a_min: float,
) -> EconomicDecision:
    """Project decision onto feasible set. (REQ-005, PROP-002, PROP-004)
    1. Clamp leisure to [0, 1].
    2. Recompute available budget with clamped labor supply.
    3. Clamp consumption to [0, budget - a_min + current_wealth].
    4. Derive savings = budget - consumption.
    5. Ensure a_{t+1} = wealth + savings >= a_min.
    Log any correction made.
    """

def compute_realized_utility(
    agent: HouseholdState, public_goods_per_capita: float,
) -> float:
    """u(c, l, G) = c^alpha * l^beta * G^gamma. (REQ-028, PROP-005)
    Uses realized (c_t, l_t, G_t) — independent of decision method.
    """
```

### 3.7 Constitution Engine

*Traceability:* REQ-018, REQ-022, REQ-023

```python
# constitution_engine.py

class ConstitutionEngine:
    """Enforces and validates constitutional rules."""

    def enforce_taxes(
        self, households: list[HouseholdState], constitution: Constitution,
        market: MarketState,
    ) -> tuple[list[HouseholdState], float]:
        """Apply all active TAX_SCHEDULE rules. Returns updated agents and total revenue."""

    def enforce_transfers(
        self, households: list[HouseholdState], constitution: Constitution,
        revenue: float,
    ) -> list[HouseholdState]:
        """Apply all active TRANSFER_PROGRAM rules. Distribute revenue per rule."""

    def enforce_public_goods(
        self, constitution: Constitution, revenue: float, num_agents: int,
    ) -> float:
        """Compute public goods per capita G_t from PUBLIC_GOODS rules."""

    def enforce_regulations(
        self, firms: list[FirmState], households: list[HouseholdState],
        constitution: Constitution,
    ) -> tuple[list[FirmState], list[HouseholdState]]:
        """Apply MARKET_REGULATION and FIRM_REGULATION rules."""

    def validate_rule(self, rule: ConstitutionalRule) -> tuple[bool, str]:
        """Validate a rule for internal consistency. (REQ-023)
        Checks: tax rates produce non-negative revenue, transfers satisfy
        budget constraint, no logical contradictions.
        Returns (is_valid, reason).
        """

    def apply_proposal(
        self, constitution: Constitution, proposal: ConstitutionalProposal,
    ) -> Constitution:
        """Add, modify, or remove a rule. Only called after validation passes."""
```

**Rule enforcement strategy:** Each `ConstitutionalRule` has an `enforcement_code` field containing a Python expression. The engine evaluates this in a **restricted sandbox** (no imports, no file/network access, limited builtins) using a safe evaluator. This enables LLM-proposed rules to be executable while preventing code injection.

### 3.8 Observer

*Traceability:* REQ-030, REQ-031, REQ-032

```python
# observer.py

class Observer:
    """Computes and records macro statistics."""

    def __init__(self, config: SimulationConfig) -> None:
        self.history: list[HistoryEntry] = []
        self.cumulative_welfare: float = 0.0

    def observe(
        self, period_state: PeriodState, prev_constitution: Constitution | None,
    ) -> HistoryEntry:
        """Compute all statistics per REQ-030:
        - Gini coefficient
        - Pareto efficiency score
        - Aggregate Y, C, I
        - Mean/median wealth, wealth quantiles
        - Unemployment rate
        - Firm statistics (count, mean size, aggregate R&D)
        - Social welfare (sum of realized utilities, REQ-031)
        - Cumulative discounted welfare
        - Constitutional changes
        """

    def finalize(self, period_state: PeriodState) -> SimulationOutput:
        """Produce final output per REQ-032."""
```

### 3.9 Initialization

*Traceability:* REQ-001, REQ-034

```python
# initialization.py

def initialize_simulation(config: SimulationConfig) -> tuple[PeriodState, SimulationRNG]:
    """Bootstrap simulation at period 0.
    1. Create SimulationRNG from seed.
    2. Discretize productivity process (Rouwenhorst).
    3. Create N households with:
       - wealth ~ Normal(mean, std), clamped >= a_min
       - productivity_index drawn uniformly from stationary distribution
       - productivity = grid[index]
       - utility_params: alpha, beta, gamma ~ Dirichlet; beta_discount ~ Uniform(0.9, 0.99)
       - value_vector: random split
       - role = WORKER (default)
    4. Create default constitution.
    5. Create initial MarketState with analytical guesses for w, r.
    6. Return (PeriodState(period=0, ...), rng).
    """
```

---

## 4. Key Algorithms

### 4.1 Tatonnement Price Adjustment

```
Input: households, firms, config
Output: (w*, r*) such that markets clear

1. w, r ← initial_guess(households, firms)  # e.g. MPL, MPK from aggregate production
2. for iter in 1..max_iter:
   a. For each firm: compute optimal K_f, L_f from FOCs given (w, r)
   b. L_demand = sum(L_f), K_demand = sum(K_f)
   c. L_supply = sum(z_i * labor_supply_i) for all households  # from last period or FOC
   d. K_supply = sum(a_i) for all households
   e. excess_L = L_demand - L_supply
   f. excess_K = K_demand - K_supply
   g. w += step * excess_L
   h. r += step * excess_K
   i. Clamp w > 0, r > -delta  # prices must be economically meaningful
   j. if |excess_L| < tol and |excess_K| < tol: break
3. Compute Y = sum(Y_f), C = sum(c_i), I = Y - C - G
4. Check |Y - C - I - G| < tol  (PROP-003)
5. Return MarketState(w, r, Y, C, I, G, error)
```

### 4.2 Rouwenhorst Discretization

```
Input: rho, sigma, n_states
Output: (grid, transition_matrix)

1. sigma_z = sigma / sqrt(1 - rho^2)  # unconditional std dev
2. z_max = sigma_z * sqrt(n_states - 1)
3. grid = linspace(-z_max, z_max, n_states)  # in logs
4. p = (1 + rho) / 2
5. Build transition matrix recursively:
   P_2 = [[p, 1-p], [1-p, p]]
   P_n = p*[P_{n-1}, 0; 0, 0] + (1-p)*[0, P_{n-1}; 0, 0]
         + (1-p)*[0, 0; P_{n-1}, 0] + p*[0, 0; 0, P_{n-1}]
   Normalize rows to sum to 1.
6. grid_values = exp(grid)  # convert from log to level
7. Return (grid_values, P_n)
```

### 4.3 Constraint Projection (REQ-005)

```
Input: decision (c, l), agent state, prices, a_min
Output: feasible (c', l')

1. l' = clamp(l, 0, 1)
2. labor_income = w * z * (1 - l')
3. total_resources = (1 + r) * a + labor_income - tax + transfer
4. max_consumption = total_resources - a_min  # leaves a_{t+1} = a_min
5. c' = clamp(c, 0, max(0, max_consumption))
6. a_{t+1} = total_resources - c'
7. Assert a_{t+1} >= a_min  # by construction
8. If c' != c or l' != l: log correction
9. Return (c', l')
```

---

## 5. Module Inventory

| Module | Responsibility | New/Refactor | Key Requirements |
|--------|---------------|-------------|-----------------|
| `models/household.py` | HouseholdState, UtilityParams, OccupationalRole | **New** (replaces agent.py) | REQ-001, REQ-002 |
| `models/firm.py` | FirmState | **New** | REQ-006 |
| `models/market.py` | MarketState | **New** | REQ-011–014 |
| `models/constitution.py` | Constitution v2, ConstitutionalRule, RuleType | **Refactor** (extensible) | REQ-018 |
| `models/proposal.py` | ConstitutionalProposal, VoteOutcome | **Refactor** | REQ-019–021 |
| `models/decisions.py` | EconomicDecision, EntrepreneurialDecision, PoliticalDecision | **New** | REQ-024–026 |
| `models/history.py` | HistoryEntry, SimulationOutput, WelfareSummary, PeriodState | **Refactor** | REQ-030–032 |
| `models/shocks.py` | ShockState | **New** | REQ-015–017 |
| `config.py` | SimulationConfig v2 | **Refactor** (many new fields) | REQ-034 |
| `initialization.py` | initialize_simulation, create_households | **Refactor** | REQ-001, REQ-034 |
| `lead.py` | Lead — 9-step period loop | **Refactor** (major) | REQ-033 |
| `shocks.py` | Rouwenhorst, shock drawing | **New** | REQ-015–017 |
| `market_clearing.py` | Tatonnement, firm FOCs | **New** | REQ-011–014, PROP-003 |
| `economics.py` | Budget, production, distribution, constraints | **Refactor** (major) | REQ-003–010, PROP-002, PROP-004 |
| `constitution_engine.py` | Rule enforcement, validation, sandbox eval | **New** | REQ-018, REQ-022–023, PROP-006 |
| `llm_engine.py` | LLM decision collection, batching, caching | **New** (replaces llm_citizen.py) | REQ-024–029 |
| `numerical_solver.py` | VFI/EGM household solver | **New** | REQ-036–037 |
| `observer.py` | Macro stats, welfare, history | **Refactor** | REQ-030–031 |
| `reporter.py` | Markdown/JSON output | **Refactor** | REQ-032 |
| `voting.py` | Vote tallying (generalized for extensible rules) | **Refactor** | REQ-020 |
| `coalition.py` | Coalition formation | **Minor update** | — |
| `rng.py` | SimulationRNG | **No change** | PROP-001 |
| `__main__.py` | CLI entry point | **Refactor** (new config flags) | REQ-035 |

---

## 6. Error Handling

### 6.1 Constraint Violations (REQ-005, PROP-002, PROP-004)

- **Budget violation:** `enforce_budget_constraint()` projects onto feasible set. Never raises — always produces a valid decision. Logs every correction with agent ID, original values, and corrected values.
- **Borrowing violation:** Consumption clamped so `a_{t+1} >= a_min`. If agent wealth is already at `a_min` and income < 0 (impossible under non-negative constraints but guarded), consumption set to 0.
- **Non-negativity:** Consumption and labor supply clamped by construction.

### 6.2 Market Clearing Failure (PROP-003)

- If tatonnement does not converge within `max_iter`:
  - Log a `MarketClearingWarning` with the final excess demand values.
  - Use the best prices found (lowest excess demand) and record the error in `MarketState.market_clearing_error`.
  - Do **not** abort the simulation — proceed with approximate prices.

### 6.3 LLM Failures (REQ-027)

| Failure Type | Response |
|-------------|----------|
| Timeout | Fall back to `NumericalSolver.solve_household()` for that agent |
| Parse error (invalid JSON) | Retry once with a reprompt; if still fails, fall back |
| Schema validation error | Attempt to fix (clamp values); if irrecoverable, fall back |
| Provider error (rate limit, 5xx) | Exponential backoff (3 retries); then fall back for entire batch |

All fallbacks logged with `structlog` at WARNING level.

### 6.4 Constitutional Rule Validation (REQ-023, PROP-006)

- `validate_rule()` checks before activation:
  - Tax rates produce non-negative revenue (rates in valid range).
  - Transfer programs do not exceed available revenue.
  - Enforcement code passes sandbox safety check (no dangerous operations).
  - No logical contradictions with existing rules (e.g., two conflicting tax schedules).
- Invalid rules rejected with logged reason. The proposal is marked as failed.

### 6.5 Numerical Instability

- All wealth/utility computations checked for NaN/Inf after each period.
- If detected: raise `NumericalInstabilityError` with full context (agent ID, period, values).
- Production function uses `max(K_f, epsilon)` and `max(L_f, epsilon)` to avoid 0^alpha.

### 6.6 Firm Lifecycle Errors (REQ-010)

- Firm with negative net worth (profit < -capital): automatically liquidated.
- Owner receives `max(0, remaining_capital)` — no negative wealth injection.
- Workers released (role set to UNEMPLOYED).

---

## 7. Security Considerations

### 7.1 LLM Output Sandboxing

- All LLM responses are parsed as JSON and validated against Pydantic schemas. No raw string evaluation.
- Constitutional `enforcement_code` is evaluated in a **restricted sandbox**:
  - Allowed: arithmetic, comparisons, conditionals, basic math functions.
  - Blocked: imports, file I/O, network, `exec`, `eval`, `__builtins__`, `os`, `sys`.
  - Implementation: Use `ast.literal_eval` for simple expressions, or a restricted `eval` with curated `__builtins__`.
  - Maximum execution time: 100ms per rule evaluation.

### 7.2 Configuration Validation

- All `SimulationConfig` fields have Pydantic validators with range constraints.
- No user-provided code is executed outside the constitutional enforcement sandbox.
- CLI inputs are parsed via `argparse` (no shell injection).

### 7.3 State Isolation

- Citizens receive **read-only copies** of state (frozen Pydantic models).
- No citizen logic has access to the RNG, file system, or network.
- The Lead is the only component that mutates state.

---

## 8. Testing Strategy

### 8.1 Unit Tests

- **Models:** Validation, serialization, constraint enforcement (e.g., UtilityParams sum to 1).
- **Economics:** Budget computation, production function, constraint projection. Test against analytical solutions.
- **Market clearing:** Convergence on known equilibria (e.g., single-firm economy with closed-form solution).
- **Shocks:** Rouwenhorst output matches known values from Kopecky & Suen (2010).
- **Constitution engine:** Rule validation (valid/invalid cases), enforcement correctness.
- **Numerical solver:** Compare VFI/EGM output to known analytical solutions (e.g., log utility with no labor choice).

### 8.2 Integration Tests

- **Full period lifecycle:** Run 1 period, verify all 9 steps executed, state consistent.
- **Market clearing + agent decisions:** Verify Y = C + I + G within tolerance.
- **LLM fallback chain:** Mock LLM failures, verify fallback to numerical solver.
- **Constitutional evolution:** Run multiple periods, verify proposals collected, votes tallied, rules enacted.

### 8.3 Property-Based Tests

- **PROP-001 (Determinism):** Same seed → same output (run twice, assert equal).
- **PROP-002 (Budget consistency):** For every agent every period, verify accounting identity.
- **PROP-003 (Market clearing):** Assert `market_clearing_error < tolerance` every period.
- **PROP-004 (Non-negativity):** Assert `c >= 0`, `a >= a_min`, `K >= 0`, `L >= 0` every period.
- **PROP-005 (Welfare measurability):** Verify `realized_utility` computable from (c, l, G, params) alone.
- **PROP-006 (Constitutional validity):** Assert no invalid rule in active constitution at any period.

---

## 9. Migration from v1

The v2 design is a **major refactor**, not a rewrite from scratch. Key migration paths:

| v1 Component | v2 Replacement | Migration Notes |
|-------------|---------------|----------------|
| `AgentState` | `HouseholdState` | Add wealth dynamics, productivity Markov chain, role, firm ownership. Remove simple productivity. |
| `Constitution` (4 fixed fields) | `Constitution` (extensible rules dict) | Fundamental redesign. Initial constitution provides equivalent defaults. |
| `Proposal` (rule_key/value) | `ConstitutionalProposal` (add/modify/remove rules) | Expand from field mutation to rule lifecycle. |
| `economics.py` (simple production/tax) | `economics.py` + `market_clearing.py` | Replace heuristic production with Cobb-Douglas firms + market clearing. |
| `citizen.py` (rule-based) | `llm_engine.py` + `numerical_solver.py` | Rule-based citizen logic becomes the test mock, not the default. |
| `llm_citizen.py` (proposal/vote only) | `llm_engine.py` (all decisions) | Expand LLM to economic + entrepreneurial + political decisions. |
| `observer.py` | `observer.py` (expanded) | Add firm stats, welfare computation, wealth quantiles. |
| `coalition.py` | `coalition.py` (minor) | Retain greedy clustering; coalition formation now also an LLM decision. |
| `rng.py` | `rng.py` (no change) | Unchanged. |
| `voting.py` | `voting.py` (generalized) | Support extensible voting rules from constitution. |
| `reporter.py` | `reporter.py` (expanded) | Add firm section, welfare comparison, expanded statistics. |

**Backward compatibility:** v1 test fixtures for models will need updating to match new schemas. The v1 test structure (unit → integration → E2E) is retained.

---

## 10. Performance Considerations

- **LLM calls** dominate cost and latency. Mitigation:
  - Batch calls per decision type (REQ-029): 50 agents → 5 batches of 10.
  - Cache identical contexts (REQ-029): agents with same state hash skip LLM call.
  - Use `temperature=0` for determinism and caching effectiveness.
- **Market clearing** tatonnement: O(num_firms * max_iter) per period. With 10-50 firms and 100 iterations max, this is negligible.
- **Numerical solver** (VFI/EGM): O(n_a * n_z * max_vfi_iter) per agent. Pre-compute value function on grid once, then interpolate for each agent. Amortize across agents with same utility params.
- **Observer** statistics: Gini is O(N log N), Pareto is O(N²) — acceptable for N ≤ 200. For larger N, sample-based Pareto approximation.

---

## 11. Cost Estimation

LLM API calls are the dominant operational cost. All estimates below use **Anthropic API pricing** (as of early 2025).

### 11.1 LLM Call Volume per Simulation Run

Each period, agents make up to 3 categories of LLM calls:

| Decision Type | Frequency | Calls per Occurrence |
|--------------|-----------|---------------------|
| Economic (consumption/savings/labor) | Every period | N agents |
| Entrepreneurial (firm mgmt, R&D) | Every period (entrepreneurs only) | ~0.2N agents (est. 20% entrepreneurs) |
| Political (proposals + votes) | Every `proposal_interval` periods | N agents |

**Call count formula** for N agents, T periods, proposal interval K:

```
Total calls = T × N                          # economic decisions
            + T × 0.2 × N                    # entrepreneurial (est. 20% entrepreneurs)
            + (T / K) × N                    # proposals
            + (T / K) × P × N               # votes (P proposals per round, est. P ≈ 3)
```

### 11.2 Token Estimates per Call

| Component | Input Tokens | Notes |
|-----------|-------------|-------|
| System prompt + instructions | ~500 | Shared across all agents (cacheable) |
| Constitution (current rules) | ~300–800 | Grows as rules are added; cacheable within batch |
| Economic conditions (prices, aggregates) | ~150 | Same for all agents; cacheable |
| Agent-specific state | ~200 | Unique per agent |
| Recent history summary | ~200 | Same for all agents; cacheable |
| **Total input per call** | **~1,350–1,850** | ~900 cacheable + ~400 unique |
| **Output per call** | **~100–200** | Structured JSON decision |

### 11.3 Cost per Call (Claude Models)

| Model | Input ($/MTok) | Cached Input ($/MTok) | Output ($/MTok) | Est. Cost per Call |
|-------|---------------|----------------------|-----------------|-------------------|
| Claude Sonnet 4.5 | $3.00 | $0.30 | $15.00 | ~$0.004 |
| Claude Haiku 4.5 | $0.80 | $0.08 | $4.00 | ~$0.001 |

**Per-call cost breakdown (Sonnet 4.5 with prompt caching):**
- Cached input (900 tokens): 900 × $0.30/MTok = $0.00027
- Unique input (450 tokens): 450 × $3.00/MTok = $0.00135
- Output (150 tokens): 150 × $15.00/MTok = $0.00225
- **Total: ~$0.004/call**

### 11.4 Scenario Cost Estimates

| Scenario | Agents | Periods | Proposal Int. | Est. Calls | Sonnet 4.5 | Haiku 4.5 |
|----------|--------|---------|--------------|-----------|------------|-----------|
| **Minimal** (dev/test) | 20 | 50 | 5 | ~3,200 | **~$13** | **~$3** |
| **Default** | 50 | 100 | 5 | ~14,000 | **~$56** | **~$14** |
| **Large** | 100 | 200 | 5 | ~56,000 | **~$224** | **~$56** |
| **Research** | 200 | 500 | 10 | ~230,000 | **~$920** | **~$230** |

### 11.5 Cost Reduction Strategies

| Strategy | Savings | Implementation |
|----------|---------|---------------|
| **Prompt caching** (REQ-029) | ~60% on input cost | Share system prompt + constitution + conditions across batch |
| **Response caching** (REQ-029) | 10–30% fewer calls | Hash (state, context) → skip call for identical situations |
| **Use Haiku for routine decisions** | ~75% | Economic decisions via Haiku; political via Sonnet (better reasoning) |
| **Selective LLM use** | Up to 90% | Use numerical solver for most agents; LLM for a configurable fraction (`llm_fraction`) |
| **Benchmark mode** (REQ-037) | 100% (no LLM) | Run entirely on numerical solver; $0 API cost |

### 11.6 Recommended Development Workflow

| Phase | Mode | Est. Cost |
|-------|------|----------|
| Unit testing | `MockProvider` (no API calls) | $0 |
| Integration testing | `MockProvider` or Haiku, 5 agents, 10 periods | $0–$0.50 |
| Validation runs | Haiku, 20 agents, 50 periods | ~$3 |
| Full runs | Sonnet, 50 agents, 100 periods | ~$56 |
| Benchmark comparison | Numerical solver (no LLM) | $0 |

### 11.7 Compute Costs (Non-LLM)

Non-LLM compute is negligible for the default configuration:

- **Market clearing** (tatonnement): <1s per period (50 firms × 100 iterations)
- **Numerical solver** (VFI/EGM): ~5–30s for initial value function convergence (one-time), <0.1s per agent after
- **Observer statistics**: <0.01s per observation
- **Total for 100 periods**: ~2–5 minutes of CPU time (excluding LLM latency)
- **LLM latency**: ~1–3s per batch call → ~30–90 minutes wall time for default config (dominant factor)
