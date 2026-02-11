# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Emergent Constitution** is an agent-based political economy simulation. 50+ citizen-agents with heterogeneous preferences, endowments, and value vectors self-organize governance from scratch. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes (Pareto efficiency, Gini coefficient). It produces a readable "constitutional" document and history log.

This project is designed to showcase Agent Teams at scale, leveraging Claude's 1M context window to maintain full state and history.

## Project Status

The project is in the **specification/planning phase**. All specification documents live under `AgentSocietyPlanning/`:

- `spec/intent.md` — Project goals and motivation
- `spec/requirements.md` — EARS-formatted requirements (REQ-001 through REQ-006, PROP-001, PROP-002)
- `spec/design.md` — Architecture, data models, component interfaces, error handling
- `spec/tasks.md` — Three-phase implementation plan with traceability to requirements
- `steering/coding-standards.md` — Coding patterns and constraints

**Always consult these spec documents before implementing.** Implementation must trace back to the requirements.

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

Seven core models (implement as Python dataclasses or Pydantic):

- **AgentState** — id, wealth, productivity, utility_params, value_vector, coalition_id
- **Constitution** — property_rule, tax_rate, voting_rule, redistribution_rule (mutable; schema-validated)
- **Proposal** — rule_key, proposed_value, proposer_id
- **VoteOutcome** — proposal, passed, votes_for
- **TickState** — tick, agent_states, constitution, proposals_this_tick, votes
- **HistoryEntry** — tick, gini, total_output, rule_changes
- **Output** — constitution, history, final_agent_states

## Implementation Phases

**Phase 1 — Core simulation loop and state** (Tasks 1-2): Data models, initialization, Lead tick loop with economic step (production, tax, redistribution). Traces to REQ-001, REQ-002, PROP-002.

**Phase 2 — Proposals and voting** (Tasks 3-4): Proposal collection/validation, voting mechanism (majority/supermajority), citizen decision logic (rule-based or LLM). Traces to REQ-003, REQ-004, PROP-001.

**Phase 3 — Observer and output** (Tasks 5-6): Observer statistics, history logging, constitutional summary generation. Traces to REQ-005, REQ-006.

## Critical Design Constraints

- **Determinism (PROP-001):** Single RNG seed for all randomness (proposal order, tie-breaks, stochastic behavior). Same seed + config = same output. Store seed in config and log it.
- **State immutability per tick:** Pass copies or read-only views to citizens. Citizens return actions; they never mutate global state directly.
- **Constitution schema validation:** Define valid keys and value types (e.g., tax_rate in [0,1]). Reject invalid proposals without crashing.
- **Deterministic tie-breaks:** Status quo wins ties, or use seeded randomness. Never leave tie-break behavior undefined.
- **Numerical stability:** Cap or abort on NaN/Inf values in wealth or utility calculations.
- **Sandboxed citizen logic:** No network or file access from citizen logic except via Lead-provided state.

## Tech Stack

- **Language:** Python (3.10+)
- **State models:** dataclasses or Pydantic
- **Formatter/linter:** Ruff (`ruff format .`, `ruff check .`)
- **Testing:** pytest
- **Orchestration:** Single-process tick loop
