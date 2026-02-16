# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Emergent Constitution** is a DSGE-HA (Dynamic Stochastic General Equilibrium with Heterogeneous Agents) political economy simulation. Citizen-agents with heterogeneous preferences, endowments, and value vectors self-organize governance from scratch. The simulation tracks the emergence of property rights, voting rules, taxation, coalition dynamics, and measures outcomes (Pareto efficiency, Gini coefficient). Economic decisions are solved via VFI on the Bellman equation; political decisions are delegated to LLMs.

## Project Status

**v2 engine complete.** 977 tests passing. A **HANK upgrade** (v3) is planned to address economic inconsistencies and add nominal rigidities, two-asset portfolio choice, and proper distribution tracking.

### Completed Phases (v1 + v2)

- **v1 Phases 1-4** (PRs #1-#4) — Core simulation, proposals/voting, observer, reporter/CLI
- **v2 Phase 5** (PRs #52-#57) — DSGE-HA engine: Cobb-Douglas production, Rouwenhorst shocks, VFI household solver, entrepreneurial choice, analytical market clearing, LLM governance, constitution engine
- **v2.1** (`feat/mechanism-effects-framework`) — Generic mechanism effects for novel LLM-proposed institutions (revenue, distribution, productivity, constraints, wealth flows, public goods, utility effects)

### HANK Upgrade (v3) — In Progress

The v3 upgrade addresses critiques from a review against Kaplan-Moll-Violante (2018), Auclert (2019), Cagetti-De Nardi (2006). Four phases, 14 tasks, 84 subtasks:

- **Phase 1: Fix Economic Foundations** — EGM solver (Carroll 2006) replacing 11-point grid search, Walrasian market clearing with firm-level FOC bisection, corrected entrepreneur budget constraint (profit only, no labor income), Bellman-based occupational choice
- **Phase 2: Scale and Calibrate** — KFE distribution tracking (Young 2010), 7-point Rouwenhorst grids, moment-matching calibration (wealth Gini 0.80, entrepreneur share 10%)
- **Phase 3: HANK Features** — Two-asset structure (liquid bonds + illiquid capital), government debt/bond market, nominal rigidities (Rotemberg pricing, NKPC, Taylor rule)
- **Phase 4: Microfound Politics** — Political utility in the Bellman equation, Bellman-derived political preferences

All new features gated behind config flags (defaults preserve v2 behavior). See `spec/tasks.md` for detailed subtask checklist.

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

**v3 additions** (planned): Steps 2b (nominal block), 3b (portfolio choice), 7b (government budget), 8b (KFE forward)

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

### V3 (HANK upgrade — planned, see `spec/design.md`)
- `src/emergent_constitution/egm_solver.py` — EGM household solver (Carroll 2006), replaces brute-force VFI
- `src/emergent_constitution/distribution.py` — KFE distribution tracking (Young 2010 lottery)
- `src/emergent_constitution/calibration.py` — Moment-matching calibration (SMM)
- `src/emergent_constitution/government.py` — Government budget, debt, fiscal rule
- `src/emergent_constitution/nominal.py` — Nominal rigidities, Taylor rule, NKPC
- `src/emergent_constitution/political_utility.py` — Political utility function for Bellman equation

## Critical Design Constraints

- **Determinism (PROP-001):** Single RNG seed for all randomness. Same seed + config = same output.
- **State immutability per tick:** Citizens return actions; they never mutate global state directly.
- **Constitution schema validation:** Reject invalid proposals without crashing. Sandbox enforcement code at AST level.
- **Deterministic tie-breaks:** Status quo wins ties, or use seeded randomness.
- **Numerical stability:** Cap or abort on NaN/Inf values. All division guards against zero denominators.
- **Sandboxed citizen logic:** No network or file access from citizen logic except via Lead-provided state.
- **Backward compatibility (PROP-012):** All new v3 features gated behind config flags. Default config reproduces v2 behavior. All 977 existing tests must pass at each phase boundary.
- **Euler equation accuracy (PROP-007, v3):** EGM solver must achieve Euler residual < 1e-6. Policy functions must be monotone in wealth.
- **Market clearing tolerance (PROP-003):** Excess demand in all markets < 1e-8 after equilibrium computation.
- **Distribution conservation (PROP-008, v3):** KFE distribution sums to 1 at every period (tolerance 1e-12).

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
- **Testing:** pytest (977 tests, target 90%+ coverage for new code)
- **Orchestration:** Single-process tick loop

## Key Economic Concepts (Reference)

When implementing v3 tasks, refer to these:

- **EGM (Endogenous Grid Method):** Invert Euler equation to get consumption analytically at each savings grid point, then interpolate back to exogenous grid. Avoids root-finding. See `spec/design.md` §1.4.
- **Walrasian clearing:** Bisect on wage to equate `sum_f L_f*(w)` with `L^s`. Interest rate from aggregate MPK. See `spec/design.md` §1.1.
- **KFE (Kolmogorov Forward Equation):** Evolve distribution forward using policy functions + transition matrix. Young (2010) lottery for non-grid-aligned savings. See `spec/design.md` §2.1.
- **Two-asset HANK:** Liquid (bonds) + illiquid (capital) with convex adjustment cost. Nested EGM: outer loop over deposit, inner EGM for liquid savings. See `spec/design.md` §3.1.
- **Rotemberg pricing:** Quadratic price adjustment cost creates sticky prices. NKPC links inflation to marginal cost. Taylor rule sets nominal rate. Fisher equation links nominal to real. See `spec/design.md` §3.3.
