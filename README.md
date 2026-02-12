# The Emergent Constitution

An agent-based political economy simulation where 50+ citizen-agents with heterogeneous preferences, endowments, and values self-organize governance from scratch. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes such as Pareto efficiency and Gini coefficient. The result is a readable "constitutional" document and comprehensive history log.

## Overview

**The Emergent Constitution** demonstrates how institutions and governance structures can emerge from the bottom-up through agent interactions. Each citizen-agent has:

- Economic endowments (wealth, productivity)
- Preferences (utility function over consumption, leisure, public goods)
- Ideological values (equality vs liberty)
- Capabilities to propose rules, vote, trade, and form coalitions

The simulation runs in discrete time steps (ticks), with agents continuously proposing and voting on constitutional changes. Over time, a stable governance structure emerges, shaped by the competing preferences and values of the agent population.

## Key Features

- **50+ heterogeneous agents** with unique economic endowments and preferences
- **Emergent institutions**: property rights, taxation, voting rules, redistribution policies
- **Coalition dynamics**: agents form and dissolve coalitions based on shared values
- **Economic simulation**: production, taxation, redistribution with budget constraints
- **Statistical tracking**: Gini coefficient, Pareto efficiency, wealth distribution
- **Deterministic**: same seed produces identical outcomes (PROP-001)
- **Comprehensive output**: constitutional document and tick-by-tick history
- **96% test coverage** with 186 tests

## Architecture

```
Initial config (N agents, endowments, preferences, seed)
       |
       v
+------------------+
| Lead (Governor)  |  tick loop: update state, collect proposals, run votes, apply rules
+------------------+
       |
       +---> Citizen agents (50+): propose, vote, trade
       |
       +---> Constitution (mutable ruleset): property_rule, tax_rate, voting_rule, redistribution_rule
       |
       v (every K ticks)
+------------------+
| Observer         |  compute Gini, total output, Pareto; append to history
+------------------+
       |
       v
   Output: constitution snapshot + history log
```

### Components

**Lead (Simulation Governor)**: Advances time in discrete ticks, enforces turn order, collects proposals, runs votes, applies economic rules, and triggers the Observer.

**Citizen Agents**: Each agent can propose constitutional changes, vote on proposals, engage in bilateral trades, and join coalitions. Agent behavior is driven by utility maximization given their preferences and values.

**Constitution**: The mutable ruleset governing the simulation, including property allocation, tax rates, voting thresholds, and redistribution policies.

**Observer**: Periodically computes aggregate statistics (Gini coefficient, total output, Pareto efficiency) and records them in the history log.

## Installation

### Requirements

- Python 3.10 or higher
- Pydantic 2.0+
- structlog 24.0+

### Using pip

```bash
pip install -e .
```

### Using rye (recommended)

```bash
rye sync
```

## Quick Start

### Basic Usage

Run a simulation with default parameters (50 agents, 100 ticks, seed 42):

```bash
python -m emergent_constitution
```

### Custom Parameters

```bash
# 100 agents, 200 ticks, custom seed
python -m emergent_constitution -n 100 -t 200 -s 123

# Custom proposal and observation intervals
python -m emergent_constitution --proposal-interval 10 --observer-interval 10
```

### Output Options

```bash
# Save markdown report to file
python -m emergent_constitution -o report.md

# Output raw JSON
python -m emergent_constitution --json

# Quiet mode (suppress logging)
python -m emergent_constitution -q
```

### Programmatic Usage

```python
from emergent_constitution import Lead, SimulationConfig

config = SimulationConfig(
    num_agents=50,
    max_ticks=100,
    seed=42,
    initial_wealth_mean=100.0,
    initial_wealth_std=30.0,
    proposal_interval=5,
    observer_interval=5,
)

lead = Lead(config)
output = lead.run()

# Access results
print(f"Final constitution: {output.constitution}")
print(f"Gini coefficient: {output.history[-1].gini}")
print(f"Total output: {output.history[-1].total_output}")
```

## Configuration

### SimulationConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_agents` | int | 50 | Number of citizen-agents (≥2) |
| `max_ticks` | int | 100 | Maximum simulation ticks (≥1) |
| `seed` | int | 42 | RNG seed for reproducibility |
| `initial_wealth_mean` | float | 100.0 | Mean of initial wealth distribution |
| `initial_wealth_std` | float | 30.0 | Std dev of initial wealth distribution |
| `initial_productivity_mean` | float | 10.0 | Mean of initial productivity distribution |
| `initial_productivity_std` | float | 3.0 | Std dev of initial productivity distribution |
| `proposal_interval` | int | 5 | Proposals collected every K ticks |
| `observer_interval` | int | 5 | Observer records stats every K ticks |
| `trade_interval` | int | 5 | Agents can trade every K ticks |
| `coalition_interval` | int | 10 | Coalitions reform every K ticks |
| `use_llm` | bool | False | Enable LLM-based reasoning for subset of agents |
| `llm_fraction` | float | 0.1 | Fraction of agents using LLM (0.0-1.0) |

### Constitution Rules

| Rule | Options | Default | Description |
|------|---------|---------|-------------|
| `property_rule` | private, communal, mixed | private | How production output is allocated |
| `tax_rate` | 0.0 - 1.0 | 0.0 | Fraction of output collected as tax |
| `voting_rule` | majority, supermajority, unanimity | majority | Threshold for proposals to pass |
| `redistribution_rule` | none, flat, progressive | flat | How tax revenue is redistributed |

## Project Structure

```
AgentSociety/
├── src/emergent_constitution/
│   ├── __init__.py                 # Package exports
│   ├── __main__.py                 # CLI entry point
│   ├── config.py                   # SimulationConfig
│   ├── lead.py                     # Lead (Governor) tick loop
│   ├── citizen.py                  # Citizen decision logic (rule-based)
│   ├── llm_citizen.py              # LLM-based citizen reasoning (optional)
│   ├── voting.py                   # Proposal validation, vote tallying
│   ├── economics.py                # Production, taxation, redistribution
│   ├── coalition.py                # Coalition formation logic
│   ├── observer.py                 # Statistics computation (Gini, Pareto)
│   ├── reporter.py                 # Markdown report generation
│   ├── initialization.py           # Agent/state initialization
│   ├── rng.py                      # Seeded RNG wrapper (determinism)
│   ├── logging.py                  # structlog configuration
│   └── models/
│       ├── __init__.py             # Model exports
│       ├── agent.py                # AgentState, UtilityParams, ValueVector
│       ├── constitution.py         # Constitution, PropertyRule, VotingRule, RedistributionRule
│       ├── proposal.py             # Proposal, TradeOffer, VoteOutcome
│       ├── tick.py                 # TickState
│       └── history.py              # HistoryEntry, SimulationOutput
├── tests/                          # 19 test modules, 186 tests, 96% coverage
├── AgentSocietyPlanning/
│   ├── spec/                       # Original specification documents
│   │   ├── intent.md               # Project goals and motivation
│   │   ├── requirements.md         # EARS-formatted requirements
│   │   ├── design.md               # Architecture and data models
│   │   └── tasks.md                # Implementation plan
│   └── steering/
│       └── coding-standards.md     # Coding patterns and constraints
├── pyproject.toml                  # Project metadata and dependencies
├── CLAUDE.md                       # Claude Code guidance
└── README.md                       # This file
```

## Testing

Run all tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=emergent_constitution --cov-report=term-missing
```

Run only fast tests (exclude slow integration tests):

```bash
pytest -m "not slow"
```

## Code Quality

Format code with Ruff:

```bash
ruff format .
```

Lint with Ruff:

```bash
ruff check .
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture](docs/architecture.md) - Detailed component interactions and data flow
- [Data Models](docs/models.md) - Complete reference for all Pydantic models
- [Configuration](docs/configuration.md) - Full configuration options and valid ranges
- [Development Guide](docs/development.md) - Contributing, testing, and coding conventions

## Requirements Traceability

The implementation satisfies all six core requirements and both properties:

- **REQ-001**: Maintains N≥50 agents with endowments, preferences, and values
- **REQ-002**: Advances time in discrete ticks with state updates
- **REQ-003**: Collects and votes on agent proposals
- **REQ-004**: Updates constitution based on vote outcomes
- **REQ-005**: Observer computes statistics every K ticks
- **REQ-006**: Outputs constitutional summary and history log
- **PROP-001**: Deterministic (fixed seed → identical outcomes)
- **PROP-002**: Consistent (all states respect budget constraints and rules)

## Design Principles

- **Determinism**: Single RNG seed for all randomness; same seed produces identical results
- **State immutability per tick**: Agents receive read-only state views; all mutations go through Lead
- **Schema validation**: Constitution fields are validated; invalid proposals are rejected
- **Numerical stability**: NaN/Inf values are detected and handled gracefully
- **Comprehensive logging**: Structured logging with context for debugging

## Credits

Developed by meleantonio (meleantonio@gmail.com)

This project showcases Agent Teams at scale, leveraging Claude's 1M context window to maintain full simulation state and history.

## License

See pyproject.toml for license information.
