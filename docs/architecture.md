# Architecture Documentation

## Overview

The Emergent Constitution is built around a single-process tick loop architecture where a Lead (Governor) orchestrates the simulation, with citizen agents making decisions and an Observer tracking aggregate statistics.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SimulationConfig                         │
│  (num_agents, max_ticks, seed, intervals, distributions)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                         Lead                                 │
│  - Initialize simulation state                               │
│  - Run main tick loop                                        │
│  - Orchestrate all components                                │
│  - Manage deterministic RNG                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       v               v               v
┌────────────┐  ┌────────────┐  ┌────────────┐
│  Citizen   │  │ Economics  │  │  Observer  │
│  Logic     │  │  Engine    │  │            │
└────────────┘  └────────────┘  └────────────┘
       │               │               │
       v               v               v
┌────────────┐  ┌────────────┐  ┌────────────┐
│ Proposals  │  │ Production │  │ Statistics │
│ Votes      │  │ Taxation   │  │ History    │
│ Trades     │  │ Redistrib. │  │ Reporting  │
└────────────┘  └────────────┘  └────────────┘
```

## Component Responsibilities

### Lead (Simulation Governor)

**Location**: `src/emergent_constitution/lead.py`

The Lead is the central orchestrator that manages the entire simulation lifecycle.

**Responsibilities**:
- Initialize simulation state from configuration
- Advance time in discrete ticks
- Enforce turn order and timing (proposal intervals, observer intervals)
- Collect proposals from citizens
- Run votes according to the current voting rule
- Apply economic steps (production, taxation, redistribution)
- Trigger Observer at specified intervals
- Maintain and return simulation history

**Key Methods**:
- `__init__(config, citizen_llm)` - Initialize with configuration
- `run() -> SimulationOutput` - Execute all ticks and return results
- `_advance_tick(tick) -> TickState` - Process one tick
- `_collect_proposals(tick, agents) -> list[Proposal]` - Gather proposals from agents
- `_collect_votes(proposals, agents) -> list[VoteOutcome]` - Run voting on proposals
- `_collect_trades(tick, agents) -> list[TradeOffer]` - Gather trade offers

### Citizen Logic

**Location**: `src/emergent_constitution/citizen.py`, `src/emergent_constitution/llm_citizen.py`

Implements agent decision-making for proposals, voting, and trading.

**Responsibilities**:
- Generate rule-change proposals based on agent preferences and values
- Cast votes on proposals (yes/no decisions)
- Generate trade offers based on wealth and productivity
- Compute utility and coalition membership
- Calculate Gini coefficient for inequality measurement

**Key Functions**:
- `decide_proposal(agent, constitution, rng) -> Proposal | None` - Generate proposal
- `decide_votes(agent, proposals, constitution, rng) -> dict[Proposal, bool]` - Vote on proposals
- `decide_trade(agent, other_agents, constitution, rng) -> TradeOffer | None` - Generate trade
- `compute_gini(agents) -> float` - Calculate Gini coefficient

**Decision Heuristics**:
1. Proposals favor changes aligned with agent values (equality vs liberty)
2. Votes support proposals that improve expected utility
3. Trades occur when wealth/productivity asymmetries create gains from exchange

### Economics Engine

**Location**: `src/emergent_constitution/economics.py`

Pure functions for production, taxation, and redistribution.

**Responsibilities**:
- Compute production output based on property rules (private/communal/mixed)
- Apply taxation at the constitutional rate
- Redistribute tax revenue according to redistribution rules
- Execute bilateral trades between agents
- Validate agent states for numerical stability (NaN/Inf detection)

**Key Functions**:
- `compute_production(agents, constitution) -> dict[str, float]` - Calculate output per agent
- `apply_taxation(agents, production, constitution) -> tuple[list[AgentState], float]` - Tax and accumulate revenue
- `apply_redistribution(agents, tax_revenue, constitution) -> list[AgentState]` - Distribute revenue
- `apply_trades(agents, trades) -> list[AgentState]` - Execute trade offers
- `economic_step(agents, constitution) -> list[AgentState]` - Complete economic update

**Property Rules**:
- **Private**: Each agent keeps their own productivity output
- **Communal**: Total productivity pooled and split equally
- **Mixed**: 50% private, 50% pooled

**Redistribution Rules**:
- **None**: Tax revenue is discarded
- **Flat**: Equal distribution to all agents
- **Progressive**: More to agents with below-median wealth

### Observer/Statistician

**Location**: `src/emergent_constitution/observer.py`

Computes aggregate statistics and detects rule changes.

**Responsibilities**:
- Calculate Gini coefficient (wealth inequality)
- Measure total economic output
- Compute mean and median wealth
- Estimate Pareto efficiency
- Detect constitutional changes between ticks
- Count coalition membership

**Key Functions**:
- `observe_tick(tick, agents, constitution, last_constitution, votes) -> HistoryEntry` - Create history entry
- `compute_pareto_efficiency(agents) -> float` - Estimate efficiency (0-1 scale)
- `detect_rule_changes(old_const, new_const, votes) -> list[str]` - Identify changes

**Pareto Efficiency Algorithm**:
For each pair of agents, test if a small wealth transfer from richer to poorer can improve the recipient's utility without harming the donor. The efficiency score is the fraction of pairs with no such improvement (1.0 = fully efficient).

### Voting System

**Location**: `src/emergent_constitution/voting.py`

Handles proposal validation, vote tallying, and constitution updates.

**Responsibilities**:
- Validate proposals against constitution schema
- Check proposal value ranges (e.g., tax_rate in [0,1])
- Tally votes according to voting rules (majority/supermajority/unanimity)
- Apply passed proposals to constitution
- Handle tie-breaking deterministically (status quo wins)

**Key Functions**:
- `validate_proposal(proposal) -> bool` - Check if proposal is valid
- `tally_votes(proposals, votes, voting_rule) -> list[VoteOutcome]` - Count votes
- `apply_passed_proposals(constitution, outcomes) -> Constitution` - Update rules

**Voting Rules**:
- **Majority**: >50% votes required
- **Supermajority**: ≥66.67% votes required
- **Unanimity**: 100% votes required
- Ties go to status quo (proposal fails)

### Coalition Formation

**Location**: `src/emergent_constitution/coalition.py`

Groups agents with similar values.

**Responsibilities**:
- Cluster agents based on value vectors (equality vs liberty)
- Assign coalition IDs using k-means-style clustering
- Support coalition-based voting and coordination

**Key Functions**:
- `form_coalitions(agents, num_coalitions, rng) -> list[AgentState]` - Cluster agents
- `_compute_distance(v1, v2) -> float` - Value vector distance

### Reporter

**Location**: `src/emergent_constitution/reporter.py`

Generates human-readable markdown reports.

**Responsibilities**:
- Format simulation output as markdown
- Summarize constitutional evolution
- Present statistical trends
- Display final agent states

**Key Functions**:
- `generate_report(output) -> str` - Create markdown report

## Data Flow

### Tick Lifecycle

Each tick follows this sequence:

```
1. Check intervals (proposal, trade, coalition, observer)
2. Form coalitions (if coalition_interval)
3. Collect proposals (if proposal_interval)
   - Each agent calls decide_proposal()
   - Lead validates proposals
4. Collect votes
   - Each agent calls decide_votes() on all proposals
   - Lead tallies votes per voting rule
5. Apply passed proposals
   - Update constitution with winning proposals
6. Collect trades (if trade_interval)
   - Each agent calls decide_trade()
   - Lead validates trade amounts
7. Economic step
   - Compute production (based on property_rule)
   - Apply taxation (based on tax_rate)
   - Apply redistribution (based on redistribution_rule)
   - Execute trades
8. Observer (if observer_interval)
   - Compute statistics
   - Detect rule changes
   - Append HistoryEntry
9. Update TickState
   - Store new agent states
   - Store proposals, votes, trades
10. Increment tick
```

### State Management

**Immutability Principle**: Agent states are immutable within a tick. All economic and voting operations create new states rather than mutating existing ones.

**State Flow**:
1. Lead maintains current `TickState`
2. Citizens receive read-only views of state
3. Citizens return actions (proposals, votes, trades)
4. Lead applies actions and creates new `TickState`
5. No agent can directly mutate global state

## Determinism (PROP-001)

All randomness is controlled by a single seeded RNG:

**Location**: `src/emergent_constitution/rng.py`

**Features**:
- Single `SimulationRNG` instance per simulation
- All random operations go through this RNG
- Seed is stored in configuration and output
- Same seed + config = identical results

**Random Operations**:
- Initial wealth/productivity distributions
- Tie-breaking in votes
- Proposal ordering
- Agent selection for LLM reasoning
- Coalition initialization
- Trade partner selection

## Consistency (PROP-002)

Budget constraints and rule enforcement:

**Wealth Conservation**: Total wealth is conserved across ticks (except for production/consumption).

**Tax Constraints**: `0 ≤ tax_rate ≤ 1` enforced by Pydantic validation.

**Trade Constraints**: Trades only execute if buyer has sufficient wealth.

**Production**: Always non-negative, based on productivity.

**Numerical Stability**: All agent wealth values checked for NaN/Inf after each economic step.

## Extensibility

### Adding New Constitution Rules

1. Define enum in `models/constitution.py`
2. Add field to `Constitution` with validation
3. Update `CONSTITUTION_FIELDS` dict
4. Implement rule logic in relevant component (e.g., `economics.py`)
5. Add proposal generation logic in `citizen.py`

### Adding New Statistics

1. Add computation function to `observer.py`
2. Add field to `HistoryEntry` model
3. Update `observe_tick()` to compute new statistic
4. Update report formatting in `reporter.py`

### Custom Citizen Logic

1. Implement `CitizenLLM` protocol from `llm_citizen.py`
2. Pass instance to `Lead.__init__()`
3. Set `config.use_llm = True`
4. Adjust `config.llm_fraction` for API cost management

## Performance Considerations

- **Single Process**: All computation in one Python process
- **No Parallelism**: Sequential agent decisions for determinism
- **O(N²) Operations**: Pareto efficiency, some coalition logic
- **Memory**: Full state history kept in memory (O(N × ticks))

For 50 agents × 100 ticks, typical runtime is under 1 second.

For larger simulations (500+ agents, 1000+ ticks), consider:
- Reducing observer_interval to limit history size
- Disabling Pareto computation (most expensive statistic)
- Using history summarization for long runs
