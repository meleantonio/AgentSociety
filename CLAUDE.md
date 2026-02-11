# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Emergent Constitution** is an agent-based political economy simulation. 50+ citizen-agents with heterogeneous preferences, endowments, and value vectors self-organize governance from scratch. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes (Pareto efficiency, Gini coefficient). It produces a readable "constitutional" document and history log.

This project is designed to showcase Agent Teams at scale, leveraging Claude's 1M context window to maintain full state and history.

## Project Status

**Implementation is complete.** All 6 requirements (REQ-001 through REQ-006) and both properties (PROP-001, PROP-002) are implemented. 186 tests, 96% coverage.

### Implemented Phases

- **Phase 1** (PR #1) — Core simulation loop and state: Pydantic data models, agent initialization, Lead tick loop, economics engine (production, taxation, redistribution). REQ-001, REQ-002, PROP-002.
- **Phase 2** (PR #2) — Proposals and voting: Proposal collection/validation, voting mechanisms (majority/supermajority/unanimity), citizen decision logic (rule-based). REQ-003, REQ-004, PROP-001.
- **Phase 3** (PR #3) — Observer and output: Observer statistics (Gini, total output, Pareto), history logging. REQ-005, REQ-006.
- **Phase 4** (PR #4) — Reporter and CLI: Markdown report generator, CLI entry point (`python -m emergent_constitution`).

### Spec Documents

Original specification documents live under `AgentSocietyPlanning/`:

- `spec/intent.md` — Project goals and motivation
- `spec/requirements.md` — EARS-formatted requirements
- `spec/design.md` — Architecture, data models, component interfaces, error handling
- `spec/tasks.md` — Implementation plan with traceability to requirements
- `steering/coding-standards.md` — Coding patterns and constraints

## Architecture

```
Initial config (N agents, endowments, preferences, seed)
       |
       v
+------------------+
| Lead (Governor)  |  tick loop: update state, collect proposals, run votes, apply rules
+------------------+
       |
       +---> Citizen agents (50+): propose, vote, trade (decisions per tick)
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

**Lead (Simulation Governor):** Advances time in discrete ticks, enforces turn order, resolves voting conflicts, aggregates global state, triggers Observer.

**Citizen Agents:** Each has endowments (wealth, productivity), preferences (utility function over consumption/leisure/public goods), and a value vector (equality vs liberty). Capabilities: propose rules, vote, trade, form coalitions. Behavior driven by micro-rules to maximize utility.

**Observer/Statistician:** Reads state every K ticks, computes Gini coefficient, total output, Pareto dominance checks, coalition sizes. Maintains history log.

**Execution model:** Single process with clear tick loop. Citizen logic is code-based (deterministic), with LLM agents used selectively for proposal generation and vote reasoning to manage API costs.

## Data Models

Seven core Pydantic models in `src/emergent_constitution/models/`:

- **AgentState** — id, wealth, productivity, utility_params, value_vector, coalition_id
- **Constitution** — property_rule, tax_rate, voting_rule, redistribution_rule (mutable; schema-validated)
- **Proposal** — rule_key, proposed_value, proposer_id
- **VoteOutcome** — proposal, passed, votes_for
- **TickState** — tick, agent_states, constitution, proposals_this_tick, votes
- **HistoryEntry** — tick, gini, total_output, rule_changes
- **Output** — constitution, history, final_agent_states

## Key Source Files

- `src/emergent_constitution/models/` — Pydantic data models (agent, constitution, history, proposal, tick)
- `src/emergent_constitution/config.py` — `SimulationConfig`
- `src/emergent_constitution/initialization.py` — Agent/state creation
- `src/emergent_constitution/economics.py` — Production, tax, redistribution
- `src/emergent_constitution/citizen.py` — Proposal/vote decision logic
- `src/emergent_constitution/voting.py` — Vote tallying, proposal validation
- `src/emergent_constitution/lead.py` — Lead tick loop (Governor)
- `src/emergent_constitution/observer.py` — Statistics computation (Gini, Pareto)
- `src/emergent_constitution/reporter.py` — Markdown report generation
- `src/emergent_constitution/rng.py` — Seeded RNG wrapper (PROP-001)
- `src/emergent_constitution/logging.py` — structlog configuration
- `src/emergent_constitution/__main__.py` — CLI entry point

## Critical Design Constraints

- **Determinism (PROP-001):** Single RNG seed for all randomness (proposal order, tie-breaks, stochastic behavior). Same seed + config = same output. Store seed in config and log it.
- **State immutability per tick:** Pass copies or read-only views to citizens. Citizens return actions; they never mutate global state directly.
- **Constitution schema validation:** Define valid keys and value types (e.g., tax_rate in [0,1]). Reject invalid proposals without crashing.
- **Deterministic tie-breaks:** Status quo wins ties, or use seeded randomness. Never leave tie-break behavior undefined.
- **Numerical stability:** Cap or abort on NaN/Inf values in wealth or utility calculations.
- **Sandboxed citizen logic:** No network or file access from citizen logic except via Lead-provided state.

## Running

```bash
# Default: 50 agents, 100 ticks, seed 42
python -m emergent_constitution

# Custom parameters
python -m emergent_constitution -n 100 -t 200 -s 123

# Save report to file
python -m emergent_constitution -o report.md

# JSON output
python -m emergent_constitution --json

# Quiet mode (suppress logging)
python -m emergent_constitution -q
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=emergent_constitution --cov-report=term-missing
```

## Tech Stack

- **Language:** Python (3.10+)
- **State models:** Pydantic
- **Formatter/linter:** Ruff (`ruff format .`, `ruff check .`)
- **Testing:** pytest (186 tests, 96% coverage)
- **Orchestration:** Single-process tick loop
