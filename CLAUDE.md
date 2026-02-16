# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Emergent Constitution** is a HANK (Heterogeneous Agent New Keynesian) political economy simulation. Citizen-agents with heterogeneous preferences, endowments, and value vectors self-organize governance from scratch. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes (Pareto efficiency, Gini coefficient). Economic decisions are solved via EGM/VFI on the Bellman equation; political decisions are delegated to LLMs with Bellman-derived preferences.

## Project Status

**v3 HANK engine complete.** 1412 tests passing across 50 test modules. All features from v1 through v3 are implemented and merged.

### Completed Phases

- **v1 Phases 1-4** (PRs #1-#4) — Core simulation, proposals/voting, observer, reporter/CLI
- **v2 Phase 5** (PRs #52-#57) — DSGE-HA engine: Cobb-Douglas production, Rouwenhorst shocks, VFI household solver, entrepreneurial choice, analytical market clearing, LLM governance, constitution engine
- **v2.1** (`feat/mechanism-effects-framework`) — Generic mechanism effects for novel LLM-proposed institutions
- **v3 Phase 1** (PR #79) — EGM solver (Carroll 2006), Walrasian market clearing, corrected entrepreneur budget, Bellman occupational choice
- **v3 Phase 2** (PR #80) — KFE distribution tracking (Young 2010), 7-state Rouwenhorst grids, SMM calibration
- **v3 Phase 3** (PR #81) — Two-asset households (liquid/illiquid), government debt/bond market, nominal rigidities (Rotemberg, NKPC, Taylor rule)
- **v3 Phase 4** (PR #82) — Political utility in Bellman equation, HANK integration tests

### Spec Documents

**v3 HANK upgrade specs** (active, use these for new work):

- `spec/intent.md` — HANK upgrade vision, motivation, success criteria
- `spec/requirements.md` — 50 EARS requirements (REQ-1xx through REQ-4xx) + 12 properties
- `spec/design.md` — Full technical design: EGM algorithm, Walrasian clearing, KFE, two-asset model, nominal block
- `spec/tasks.md` — 84 subtasks across 4 phases with requirement traceability
- `steering/coding-standards.md` — Numerical code, backward compat, testing standards

**v2 original specs** (historical reference):

- `AgentSocietyPlanning/spec/` — Original intent, requirements (REQ-001 to REQ-037), design, tasks

## Architecture

### v2 Period Lifecycle (9 steps + mechanism effect hooks)

```
Step 1:  Draw shocks (Rouwenhorst Markov transitions, aggregate TFP)
Step 1b: Apply productivity mechanism effects (from novel constitutional rules)
Step 2:  Clear markets (find equilibrium w, r)
Step 3:  Collect decisions (VFI/EGM household solver + entrepreneurial solver)
Step 4:  Validate constraints (clamp c, l, a' to feasible set)
Step 4b: Apply mechanism constraints (min/max bounds from novel rules)
Step 5:  Execute production (firm outputs, profits, R&D)
Step 6:  Enforce constitution (taxes, transfers, public goods)
Step 6b: Enforce mechanism effects (revenue, distribution, wealth flows)
Step 7:  Process governance (proposals, voting — LLM or benchmark)
Step 8:  Update states (budget constraint: a' = (1+r)a + income - c - T + Tr)
Step 9:  Observe (Gini, Pareto, aggregates, welfare)
```

**v3 additions** (implemented): Steps 2b (nominal block), 3b (portfolio choice), 7b (government budget), 8b (KFE forward)

### Key Components

- **Lead (Governor):** Advances time in discrete periods, enforces the 9-step lifecycle, triggers Observer
- **Household Solver:** VFI on Bellman equation (v2) or EGM (v3) for optimal (c, l, savings)
- **Entrepreneurial Solver:** Utility-based occupational choice, firm value, optimal capital
- **Market Clearing:** Analytical Cobb-Douglas FOC (v2) or Walrasian bisection (v3)
- **Constitution Engine:** Rule enforcement with sandboxed expressions + mechanism effects framework
- **LLM Engine:** Structured context → LLM → political decisions (proposals, votes)
- **Observer:** Gini, Pareto, aggregates, welfare summary

### Data Models (Pydantic, in `src/emergent_constitution/models/`)

**v1:** AgentState, Constitution, Proposal, VoteOutcome, TickState, HistoryEntry, SimulationOutput

**v2:** HouseholdState, FirmState, MarketState, ShockState, ConstitutionV2, ConstitutionalRule, ConstitutionalProposal, EconomicDecision, EntrepreneurialDecision, PoliticalDecision, PeriodState, SimulationOutputV2

**v2.1:** MechanismEffect, EffectTarget, EffectScope, RuleImpact (open rule types, mechanism effects framework)

## Key Source Files

### V1 (basic simulation)
- `src/emergent_constitution/models/` — Pydantic data models (v1 and v2)
- `src/emergent_constitution/config.py` — `SimulationConfig` and `SimulationConfigV2`
- `src/emergent_constitution/initialization.py` — Agent/state creation
- `src/emergent_constitution/economics.py` — Production, tax, redistribution
- `src/emergent_constitution/citizen.py` — Proposal/vote decision logic (v1)
- `src/emergent_constitution/voting.py` — Vote tallying, proposal validation
- `src/emergent_constitution/lead.py` — Lead tick loop (Governor, v1 and v2)
- `src/emergent_constitution/observer.py` — Statistics computation (Gini, Pareto)
- `src/emergent_constitution/reporter.py` — Markdown report generation
- `src/emergent_constitution/rng.py` — Seeded RNG wrapper (PROP-001)
- `src/emergent_constitution/logging.py` — structlog configuration
- `src/emergent_constitution/__main__.py` — CLI entry point

### V2 (DSGE-HA engine)
- `src/emergent_constitution/numerical_solver.py` — VFI household solver (NumPy-vectorized, policy cache)
- `src/emergent_constitution/entrepreneurial_solver.py` — Firm value, utility-based entry/exit, optimal capital
- `src/emergent_constitution/market_clearing.py` — Analytical Cobb-Douglas equilibrium prices
- `src/emergent_constitution/constitution_engine.py` — Rule enforcement, taxation, transfers, public goods, mechanism effects
- `src/emergent_constitution/llm_engine.py` — LLM decision engine (political prompts, context building, institutional suggestions)
- `src/emergent_constitution/citizen_v2.py` — Rule-based proposal/voting for benchmark mode
- `src/emergent_constitution/llm_providers.py` — LLM provider abstraction (Anthropic, Mock)
- `docs/model_paper.tex` — LaTeX academic paper documenting the DSGE-HA model

### V3 (HANK engine)
- `src/emergent_constitution/egm_solver.py` — EGM household solver (Carroll 2006 with Fella 2014 upper envelope)
- `src/emergent_constitution/distribution.py` — KFE distribution tracking (Young 2010 lottery)
- `src/emergent_constitution/calibration.py` — Moment-matching calibration (SMM via Nelder-Mead)
- `src/emergent_constitution/government.py` — Government budget constraint, debt dynamics, fiscal rule, bond market clearing
- `src/emergent_constitution/nominal.py` — Nominal rigidities: Rotemberg pricing, NKPC, Taylor rule, Fisher equation
- `src/emergent_constitution/political_utility.py` — Political utility in Bellman equation, proposal evaluation

## Critical Design Constraints

- **Determinism (PROP-001):** Single RNG seed for all randomness. Same seed + config = same output.
- **State immutability per tick:** Citizens return actions; they never mutate global state directly.
- **Constitution schema validation:** Reject invalid proposals without crashing. Sandbox enforcement code at AST level.
- **Deterministic tie-breaks:** Status quo wins ties, or use seeded randomness.
- **Numerical stability:** Cap or abort on NaN/Inf values. All division guards against zero denominators.
- **Sandboxed citizen logic:** No network or file access from citizen logic except via Lead-provided state.
- **Backward compatibility (PROP-012):** All new v3 features gated behind config flags. Default config reproduces v2 behavior. All existing tests pass.
- **Euler equation accuracy (PROP-007):** EGM solver achieves Euler residual < 1e-6. Policy functions are monotone in wealth.
- **Market clearing tolerance (PROP-003):** Excess demand in all markets < 1e-8 after equilibrium computation.
- **Distribution conservation (PROP-008):** KFE distribution sums to 1 at every period (tolerance 1e-12).

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
- **Numerics:** NumPy (vectorized VFI/EGM, market clearing, KFE), SciPy (v3: optimization, linear algebra)
- **Formatter/linter:** Ruff (`ruff format .`, `ruff check .`)
- **Testing:** pytest (1412 tests across 50 modules)
- **Orchestration:** Single-process tick loop

## Key Economic Concepts (Implemented)

- **EGM (Endogenous Grid Method):** Inverts Euler equation for consumption at each savings grid point. 200-point exponential asset grid, 51-point leisure grid. Fella (2014) upper envelope for non-monotonicities. See `egm_solver.py`.
- **Walrasian clearing:** Bisects on wage to equate `sum_f L_f*(w)` with `L^s`. Interest rate from aggregate MPK. See `market_clearing.py`.
- **KFE (Kolmogorov Forward Equation):** Evolves distribution forward using policy functions + transition matrix. Young (2010) lottery for non-grid-aligned savings. See `distribution.py`.
- **Two-asset HANK:** Liquid bonds + illiquid capital with convex adjustment cost `chi(d,k) = chi_0|d| + chi_1*d^2/k`. See `models/household.py`.
- **Rotemberg pricing:** Quadratic price adjustment cost, NKPC, Taylor rule, Fisher equation. See `nominal.py`.
- **Political utility:** `v(C;theta) = theta_eq*(-Gini) + theta_lib*(-tau)` enters Bellman as constant flow bonus. See `political_utility.py`.
- **Government bonds:** Budget constraint `B' = (1+r^b)B + G + Tr - T`, fiscal rule for debt sustainability. See `government.py`.
