# The Emergent Constitution

An agent-based political economy simulation built on a DSGE-HA (Dynamic Stochastic General Equilibrium with Heterogeneous Agents) framework. 50+ citizen-agents with heterogeneous preferences, endowments, and values self-organize governance from scratch. An LLM (Claude) makes all economic, entrepreneurial, and political decisions for each agent, while a numerical solver provides a benchmark. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes such as Pareto efficiency, Gini coefficient, and social welfare.

## Overview

**The Emergent Constitution** combines a rigorous economic model (Bewley-Huggett-Aiyagari) with LLM-driven agent behavior to study how institutions emerge from individual decision-making.

Each citizen-agent (household) has:
- **Economic state**: wealth (assets), idiosyncratic productivity from a Markov chain
- **Preferences**: Cobb-Douglas utility over consumption, leisure, and public goods
- **Ideological values**: equality vs liberty spectrum
- **Occupational role**: worker, entrepreneur, researcher, or unemployed

Every period, agents make three types of decisions via LLM:
1. **Economic**: how much to consume vs save, how much to work vs leisure
2. **Entrepreneurial**: whether to create/manage/liquidate firms, invest capital, conduct R&D
3. **Political**: propose constitutional rules, vote on others' proposals

A tatonnement algorithm clears labor and capital markets each period, finding equilibrium wages and interest rates. Firms produce output via Cobb-Douglas technology. The constitution is an extensible set of rules (taxes, transfers, public goods, regulations) that agents can modify through democratic voting.

## Key Features

- **DSGE-HA economic model**: Cobb-Douglas production, Rouwenhorst productivity shocks, tatonnement market clearing
- **LLM-driven decisions**: Claude makes all agent decisions with structured JSON output, context-aware prompts, and batched API calls
- **Numerical benchmark**: Value Function Iteration (VFI) solver for comparison with LLM decisions
- **Extensible constitution**: open-ended rule collection (tax schedules, transfer programs, public goods, regulations, voting procedures)
- **Firm dynamics**: endogenous firm creation, R&D investment, productivity shocks, liquidation
- **20+ tracked statistics**: Gini, Pareto efficiency, Y/C/I aggregates, wealth quantiles, unemployment, firm stats, social welfare
- **Deterministic**: same seed produces identical outcomes (seeded RNG for all randomness)
- **892 tests** including property-based invariant tests for 6 formal properties

## Architecture

The simulation follows a **9-step period lifecycle**:

```
For each period t = 1, ..., T:
  1. Draw shocks      → Idiosyncratic productivity (Rouwenhorst), aggregate TFP, preference shocks
  2. Clear markets     → Tatonnement finds equilibrium wage w_t and interest rate r_t
  3. Collect decisions → LLM (or numerical solver) returns economic + entrepreneurial decisions
  4. Validate          → Project decisions onto feasible set (budget constraints, borrowing limits)
  5. Execute production→ Firms produce Y = A * K^α * L^(1-α), distribute income, process R&D
  6. Enforce constitution → Apply taxes, transfers, public goods, regulations
  7. Process governance → Collect proposals, run votes, apply passed rules
  8. Update states     → Consumption, savings, utility computation for all households
  9. Observe           → Record Gini, Pareto, welfare, aggregates (every K periods)
```

### Components

| Component | Module | Description |
|-----------|--------|-------------|
| **LeadV2** | `lead.py` | Simulation governor: runs the 9-step lifecycle |
| **LLMDecisionEngine** | `llm_engine.py` | Batched LLM calls with caching and fallback |
| **LLMProvider** | `llm_providers.py` | Pluggable LLM interface (Anthropic, Mock) |
| **NumericalSolver** | `numerical_solver.py` | VFI benchmark solver |
| **ConstitutionEngine** | `constitution_engine.py` | Rule enforcement with AST-sandboxed code |
| **Market clearing** | `market_clearing.py` | Tatonnement price finding |
| **ObserverV2** | `observer.py` | 20+ macro statistics computation |
| **Reporter** | `reporter.py` | Markdown and JSON report generation |

## Installation

### Requirements

- Python 3.10+
- Pydantic 2.0+
- structlog 24.0+
- anthropic SDK (for LLM mode)

### Using pip

```bash
pip install -e .
```

### Using rye (recommended)

```bash
rye sync
```

## Quick Start

### CLI Usage

#### Benchmark mode (no LLM, $0 cost)

```bash
# Run with numerical solver only — no API calls
emergent-constitution-v2 --benchmark -n 50 -t 100

# Short run for testing
emergent-constitution-v2 --benchmark -n 20 -t 20 -q
```

#### LLM mode (Claude-powered agents)

```bash
# Default: 50 agents, 100 periods, Claude Sonnet 4.5
export ANTHROPIC_API_KEY="your-key-here"
emergent-constitution-v2 -n 50 -t 100

# Use Haiku for cheaper runs (~75% cost reduction)
emergent-constitution-v2 --llm-model claude-haiku-4-5-20251001

# Custom economic parameters
emergent-constitution-v2 \
    -n 100 -t 200 \
    --alpha 0.33 --delta 0.1 \
    --rho-z 0.9 --sigma-z 0.2 \
    --rho-A 0.95 --sigma-A 0.01
```

#### Output options

```bash
# Save markdown report
emergent-constitution-v2 --benchmark -o report.md

# Output raw JSON (for programmatic analysis)
emergent-constitution-v2 --benchmark --json -o results.json

# Quiet mode (suppress logging)
emergent-constitution-v2 --benchmark -q
```

### Python API

#### Benchmark mode (numerical solver)

```python
from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.lead import LeadV2

config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    benchmark_mode=True,  # No LLM calls, $0 cost
)

lead = LeadV2(config)
output = lead.run()

# Access results
print(f"Periods simulated: {output.total_periods}")
print(f"Final Gini: {output.history[-1].gini:.4f}")
print(f"Final welfare: {output.welfare_summary.llm_total_welfare:.2f}")
print(f"Active firms: {output.history[-1].num_active_firms}")
print(f"Unemployment: {output.history[-1].unemployment_rate:.2%}")

# Examine the final constitution
for name, rule in output.constitution.rules.items():
    print(f"  {name}: {rule.parameters}")

# Wealth distribution
last = output.history[-1]
print(f"Wealth quantiles (p10/p25/p50/p75/p90): {last.wealth_quantiles}")
```

#### LLM mode (Claude-powered agents)

```python
from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.lead import LeadV2

config = SimulationConfigV2(
    num_agents=50,
    max_periods=100,
    seed=42,
    # LLM settings
    use_llm=True,
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929",
    llm_batch_size=10,       # Agents per API batch call
    llm_cache_enabled=True,  # Cache identical contexts (~60% input cost savings)
    # Economic parameters
    alpha=0.33,     # Capital share (Cobb-Douglas)
    delta=0.1,      # Depreciation rate
    rho_z=0.9,      # Idiosyncratic productivity persistence
    sigma_z=0.2,    # Idiosyncratic productivity volatility
    rho_a=0.95,     # Aggregate TFP persistence
    sigma_a=0.01,   # Aggregate TFP volatility
    a_min=0.0,      # Borrowing limit
)

lead = LeadV2(config)
output = lead.run()
```

#### Generating reports

```python
from emergent_constitution.reporter import generate_report_v2, generate_json_v2

# Markdown report (7 sections: summary, constitution, households, firms, welfare, timeline, stats)
report = generate_report_v2(output)
with open("report.md", "w") as f:
    f.write(report)

# JSON output (full serialization for analysis)
json_str = generate_json_v2(output)
```

#### v1 API (backward compatible)

```python
from emergent_constitution import Lead, SimulationConfig

config = SimulationConfig(num_agents=50, max_ticks=100, seed=42)
lead = Lead(config)
output = lead.run()
```

## Configuration

### SimulationConfigV2 Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **Core** | | | |
| `num_agents` | int | 50 | Number of households (min: 20) |
| `max_periods` | int | 100 | Simulation length |
| `seed` | int | 42 | RNG seed for reproducibility |
| **Shocks** | | | |
| `rho_z` | float | 0.9 | Idiosyncratic productivity persistence |
| `sigma_z` | float | 0.2 | Idiosyncratic productivity volatility |
| `num_z_states` | int | 5 | Rouwenhorst grid points |
| `rho_a` | float | 0.95 | Aggregate TFP persistence |
| `sigma_a` | float | 0.01 | Aggregate TFP volatility |
| **Production** | | | |
| `alpha` | float | 0.33 | Capital share (Cobb-Douglas) |
| `delta` | float | 0.1 | Depreciation rate |
| `min_firm_capital` | float | 10.0 | Minimum capital for firm creation |
| **Household** | | | |
| `a_min` | float | 0.0 | Borrowing limit |
| `initial_wealth_mean` | float | 100.0 | Mean initial wealth |
| `initial_wealth_std` | float | 30.0 | Std dev initial wealth |
| **Intervals** | | | |
| `proposal_interval` | int | 5 | Constitutional proposals every K periods |
| `observer_interval` | int | 5 | Statistics computed every K periods |
| **Market Clearing** | | | |
| `tatonnement_max_iter` | int | 100 | Max price-finding iterations |
| `tatonnement_tolerance` | float | 1e-6 | Convergence tolerance |
| **LLM** | | | |
| `use_llm` | bool | True | Enable LLM-driven decisions |
| `llm_provider` | str | "anthropic" | LLM provider name |
| `llm_model` | str | "claude-sonnet-4-5-20250929" | Model ID |
| `llm_batch_size` | int | 10 | Agents per batch API call |
| `llm_cache_enabled` | bool | True | Cache identical contexts |
| **Benchmark** | | | |
| `benchmark_mode` | bool | False | Numerical-only mode (no LLM, $0) |
| `solver_method` | str | "egm" | "vfi" or "egm" |
| **R&D** | | | |
| `rd_success_base_prob` | float | 0.1 | Base R&D success probability |
| `rd_tfp_improvement_mean` | float | 0.05 | Mean TFP improvement from R&D |

## Cost of Running Simulations

### LLM API costs

Each agent makes ~3 LLM calls per period (economic, entrepreneurial, political decisions). With prompt caching, ~60% of input tokens are cached at reduced rates.

**Per-call cost (Claude Sonnet 4.5 with prompt caching): ~$0.004**

| Scenario | Agents | Periods | Est. API Calls | Sonnet 4.5 Cost | Haiku 4.5 Cost |
|----------|--------|---------|----------------|-----------------|----------------|
| **Dev/test** | 20 | 50 | ~3,200 | **~$13** | **~$3** |
| **Default** | 50 | 100 | ~14,000 | **~$56** | **~$14** |
| **Large** | 100 | 200 | ~56,000 | **~$224** | **~$56** |
| **Research** | 200 | 500 | ~230,000 | **~$920** | **~$230** |

### Cost reduction strategies

| Strategy | Savings | How |
|----------|---------|-----|
| **Benchmark mode** | 100% | `--benchmark` flag — uses numerical solver, $0 API cost |
| **Use Haiku** | ~75% | `--llm-model claude-haiku-4-5-20251001` for routine decisions |
| **Prompt caching** | ~60% input | Enabled by default (`llm_cache_enabled=True`) |
| **Response caching** | 10-30% fewer calls | Identical agent states skip API calls |

### Compute costs (non-LLM)

Non-LLM compute is negligible:
- Market clearing: <1s per period
- Numerical solver: ~5-30s initial convergence, <0.1s/agent after
- Full benchmark run (50 agents, 100 periods): ~2-5 minutes CPU time

### Recommended development workflow

| Phase | Mode | Cost |
|-------|------|------|
| Unit tests | `MockProvider` | $0 |
| Integration tests | Mock or Haiku, 5 agents, 10 periods | $0-$0.50 |
| Validation runs | Haiku, 20 agents, 50 periods | ~$3 |
| Full runs | Sonnet, 50 agents, 100 periods | ~$56 |
| Benchmark comparison | `--benchmark` | $0 |

## Example Research Questions

This simulation can help investigate questions like:

**Institutional emergence**
- What tax rates emerge when agents have heterogeneous inequality preferences?
- Do agents converge on progressive or flat taxation? Under what conditions?
- How does initial wealth inequality affect the voting rules that emerge?

**LLM vs optimal behavior**
- Do LLM-driven agents achieve higher or lower social welfare than the numerical benchmark?
- Where do LLM decisions deviate most from optimal policy functions — consumption, labor, or entrepreneurship?
- How does the welfare gap between LLM and benchmark change with the number of agents?

**Economic dynamics**
- How does the Gini coefficient evolve over time under different constitutional regimes?
- What is the relationship between R&D spending and long-run TFP growth?
- How do aggregate productivity shocks propagate through the wealth distribution?

**Political economy**
- Do coalitions form along economic lines (rich vs poor) or ideological lines (equality vs liberty)?
- How does the voting rule (majority vs supermajority) affect the stability of institutions?
- Can a minority coalition block redistributive taxation when supermajority rules are in place?

**Firm dynamics**
- Under what conditions do agents choose entrepreneurship over employment?
- How does firm entry/exit respond to constitutional changes (e.g., higher taxes)?
- Is there a relationship between firm R&D investment and unemployment rates?

### Example analysis workflow

```python
from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.lead import LeadV2

# Run LLM simulation
config_llm = SimulationConfigV2(
    num_agents=50, max_periods=100, seed=42,
    use_llm=True,
)
output_llm = LeadV2(config_llm).run()

# Run benchmark with same seed
config_bench = SimulationConfigV2(
    num_agents=50, max_periods=100, seed=42,
    benchmark_mode=True,
)
output_bench = LeadV2(config_bench).run()

# Compare welfare
llm_welfare = output_llm.welfare_summary.llm_total_welfare
bench_welfare = output_bench.welfare_summary.llm_total_welfare
print(f"LLM welfare:       {llm_welfare:.2f}")
print(f"Benchmark welfare: {bench_welfare:.2f}")
print(f"Gap:               {(llm_welfare - bench_welfare) / bench_welfare:.2%}")

# Compare inequality trajectories
for entry_l, entry_b in zip(output_llm.history, output_bench.history):
    print(f"Period {entry_l.period}: LLM Gini={entry_l.gini:.4f}, Bench Gini={entry_b.gini:.4f}")
```

## Project Structure

```
AgentSociety/
├── src/emergent_constitution/
│   ├── __init__.py                 # Package exports (v1)
│   ├── __main__.py                 # CLI entry points (v1 + v2)
│   ├── config.py                   # SimulationConfig (v1) + SimulationConfigV2
│   ├── lead.py                     # Lead (v1) + LeadV2 (9-step lifecycle)
│   ├── citizen.py                  # Rule-based citizen logic (v1)
│   ├── llm_citizen.py              # LLM citizen interface (v1)
│   ├── llm_engine.py               # LLM decision engine (v2): batching, caching, fallback
│   ├── llm_providers.py            # LLM provider abstraction: Anthropic, Mock
│   ├── numerical_solver.py         # VFI benchmark solver (v2)
│   ├── constitution_engine.py      # Rule enforcement with AST sandbox (v2)
│   ├── market_clearing.py          # Tatonnement market clearing (v2)
│   ├── shock_generators.py         # Rouwenhorst + shock drawing (v2)
│   ├── economics.py                # Production, budget, utility (v1 + v2)
│   ├── voting.py                   # Proposal validation and tallying (v1 + v2)
│   ├── coalition.py                # Coalition formation (v1 + v2)
│   ├── initialization.py           # State initialization (v1 + v2)
│   ├── observer.py                 # Statistics (v1) + ObserverV2 (20+ stats)
│   ├── reporter.py                 # Report generation (v1 + v2)
│   ├── rng.py                      # Seeded RNG wrapper
│   ├── logging.py                  # structlog configuration
│   └── models/
│       ├── __init__.py             # Model exports
│       ├── agent.py                # AgentState (v1)
│       ├── household.py            # HouseholdState (v2)
│       ├── firm.py                 # FirmState (v2)
│       ├── market.py               # MarketState (v2)
│       ├── shocks.py               # ShockState (v2)
│       ├── decisions.py            # EconomicDecision, EntrepreneurialDecision, PoliticalDecision
│       ├── constitution.py         # Constitution (v1) + ConstitutionV2 (extensible rules)
│       ├── proposal.py             # Proposal (v1) + ConstitutionalProposal (v2)
│       ├── tick.py                 # TickState (v1)
│       └── history.py              # HistoryEntry (v1) + HistoryEntryV2, SimulationOutputV2
├── tests/                          # 35 test modules, 892 tests
├── docs/                           # Architecture, models, configuration, development guides
├── AgentSocietyPlanning/
│   ├── spec/                       # Specification documents (intent, requirements, design, tasks)
│   └── steering/                   # Coding standards
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=emergent_constitution --cov-report=term-missing

# Fast tests only (exclude slow integration/scale tests)
pytest -m "not slow"

# Run v2-specific tests
pytest tests/test_lead_v2.py tests/test_v2_models.py tests/test_v2_config.py

# Run property-based invariant tests
pytest tests/test_properties.py -v
```

### Property invariants tested

| Property | Description |
|----------|-------------|
| PROP-001 | Determinism: same seed + config = identical results |
| PROP-002 | Budget consistency: wealth >= borrowing limit for all agents |
| PROP-003 | Market clearing: excess demand < tolerance |
| PROP-004 | Non-negativity: consumption, wealth, output >= 0 |
| PROP-005 | Welfare measurability: social welfare is finite and non-negative |
| PROP-006 | Constitutional validity: all rules pass schema validation |

## Code Quality

```bash
ruff format .    # Format
ruff check .     # Lint
```

## Documentation

- [Architecture](docs/architecture.md) - 9-step lifecycle, component interactions, data flow
- [Data Models](docs/models.md) - Complete reference for v1 and v2 Pydantic models
- [Configuration](docs/configuration.md) - All configuration fields with examples
- [Development Guide](docs/development.md) - Contributing, testing, coding conventions

## Credits

Developed by meleantonio (meleantonio@gmail.com)

This project showcases Agent Teams at scale, leveraging Claude's 1M context window to maintain full simulation state and history. The entire v2 DSGE-HA implementation (17 tasks, 86 subtasks, ~13,200 lines of code) was built by a team of 4 Claude Code agents working in parallel.

## License

See pyproject.toml for license information.
