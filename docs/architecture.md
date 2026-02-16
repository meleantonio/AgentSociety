# Architecture Documentation

## Overview

The Emergent Constitution v2 is built on a DSGE-HA (Dynamic Stochastic General Equilibrium with Heterogeneous Agents) framework. A LeadV2 governor orchestrates a 9-step period lifecycle where citizen-agents make economic, entrepreneurial, and political decisions via LLM (or a numerical solver benchmark). Markets clear via tatonnement, firms produce output, and an extensible constitution governs taxation, transfers, and regulations.

## System Architecture

```
SimulationConfigV2 (~50 fields)
       │
       v
┌─────────────────────────────────────────────────────────┐
│                       LeadV2                             │
│  9-step period lifecycle × max_periods                   │
│  Manages: households, firms, market, shocks, constitution│
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────────┐
    │          │          │              │
    v          v          v              v
┌────────┐ ┌────────┐ ┌─────────────┐ ┌──────────┐
│ LLM    │ │Numeric │ │Constitution │ │ Observer  │
│Decision│ │Solver  │ │   Engine    │ │   V2     │
│Engine  │ │EGM/VFI │ │(AST sandbox)│ │(20+ stats)│
└───┬────┘ └───┬────┘ └──────┬──────┘ └────┬─────┘
    │          │              │              │
    v          v              v              v
┌────────┐ ┌────────┐ ┌──────────┐  ┌──────────┐
│Economic│ │Optimal │ │Taxes     │  │Gini,     │
│Entrep. │ │Policy  │ │Transfers │  │Pareto,   │
│Politic.│ │Function│ │Public    │  │Welfare,  │
│Decisns │ │        │ │Goods     │  │Firms,    │
└────────┘ └────────┘ └──────────┘  │Quantiles │
                                     └──────────┘
```

### Supporting Components

```
┌──────────────┐  ┌──────────────────┐  ┌───────────────┐
│ Market       │  │ Shock Generators │  │ LLM Providers │
│ Clearing     │  │                  │  │               │
│ (analytical/ │  │ - Rouwenhorst    │  │ - Anthropic   │
│  walrasian)  │  │ - Aggregate TFP  │  │ - Mock        │
│ w_t, r_t     │  │ - Preference     │  │               │
└──────────────┘  └──────────────────┘  └───────────────┘

┌──────────────┐  ┌──────────────────┐  ┌───────────────┐
│ Government   │  │ Distribution     │  │ Nominal Block │
│              │  │                  │  │               │
│ - Debt       │  │ - KFE tracking   │  │ - Taylor rule │
│ - Bonds      │  │ - Stationary     │  │ - NKPC        │
│ - Fiscal     │  │ - Aggregation    │  │ - Fisher eq.  │
│   rule       │  │                  │  │               │
└──────────────┘  └──────────────────┘  └───────────────┘
```

## Component Responsibilities

### LeadV2 (Simulation Governor)

**Location**: `src/emergent_constitution/lead.py`

The LeadV2 orchestrates the entire simulation, executing the 9-step period lifecycle for each period from 1 to `max_periods`.

**Key methods**:
- `__init__(config: SimulationConfigV2)` — Initialize RNG, period-0 state, engines, solver, observer
- `run() -> SimulationOutputV2` — Execute all periods, return final output with welfare summary
- `_advance_period(t) -> PeriodState` — Execute one period (9 steps)
- `_draw_shocks()` — Step 1: idiosyncratic, aggregate TFP, preference shocks
- `_clear_markets()` — Step 2: tatonnement for equilibrium prices
- `_collect_decisions()` — Step 3: LLM or numerical solver decisions
- `_validate_constraints()` — Step 4: project onto feasible set
- `_execute_production()` — Step 5: firm production, R&D, creation, liquidation
- `_enforce_constitution()` — Step 6: taxes, transfers, public goods, regulations
- `_process_governance()` — Step 7: proposals, votes, rule changes
- `_update_states()` — Step 8: consumption, savings, utility
- `_observe_period()` — Step 9: record statistics

**Traceability**: REQ-033

### LLM Decision Engine

**Location**: `src/emergent_constitution/llm_engine.py`

Manages LLM-based agent decision-making for all three decision types. Provides structured context per REQ-025, batched API calls per REQ-029, and fallback to defaults on failure per REQ-027.

**Key methods**:
- `collect_economic_decisions(households, market, constitution)` — Consumption/leisure decisions for all agents
- `collect_entrepreneurial_decisions(households, firms, market, constitution)` — Firm management decisions
- `collect_political_decisions(households, constitution, market)` — Proposals and votes

**Context building** (REQ-025): Each LLM prompt includes 5 components:
1. System prompt with instructions (~500 tokens, cacheable)
2. Current constitution rules (~300-800 tokens, cacheable)
3. Economic conditions — prices, aggregates (~150 tokens, cacheable)
4. Agent-specific state — wealth, productivity, role (~200 tokens, unique)
5. Recent history summary (~200 tokens, cacheable)

**Batching** (REQ-029): Agents are grouped into batches of `llm_batch_size` for parallel API calls.

**Caching** (REQ-029): Identical (state, context) pairs produce a cache key; duplicate situations skip API calls.

**Fallback** (REQ-027): On LLM failure, falls back to `DefaultFallbackSolver` (conservative defaults) or `NumericalSolver` if available.

**Traceability**: REQ-003, REQ-006, REQ-019, REQ-020, REQ-024-029

### LLM Providers

**Location**: `src/emergent_constitution/llm_providers.py`

Pluggable protocol for LLM-based decision generation.

**Protocol**: `LLMProvider`
- `generate(messages, schema) -> str` — Single response
- `generate_batch(batch, schema) -> list[str]` — Batch of responses

**Implementations**:
- `MockProvider` — Deterministic mock for testing ($0 cost, no API calls)
- `AnthropicProvider` — Anthropic Messages API with structured output, tool_use format, temperature=0, exponential backoff on rate limits

**Traceability**: REQ-024, REQ-027, REQ-029, PROP-001

### Numerical Solver

**Location**: `src/emergent_constitution/numerical_solver.py` (VFI), `src/emergent_constitution/egm_solver.py` (EGM)

Solves household Bellman equations via Value Function Iteration (VFI) or Endogenous Grid Method (EGM) on a discrete grid. Provides the benchmark optimal policy functions against which LLM decisions can be compared.

**EGM Solver** (default, REQ-115..120):
- 200-point exponential asset grid (finer near borrowing constraint)
- 51-point leisure grid
- Carroll (2006) endogenous grid method with Fella (2014) upper envelope for non-monotonicity
- Policy cache keyed on market parameters
- Supports two-asset extension (liquid/illiquid) per REQ-301..305

**VFI Solver** (legacy):
- Discrete grid search over consumption and leisure
- NumPy-vectorized for speed

**Key methods**:
- `solve_all(households, market, constitution_tax_rate, public_goods, transfer, political_flow_bonus) -> dict[str, EconomicDecision]`
- `get_value_function() -> tuple[list[list[float]], list[float]]` — Exposes V(a,z) for occupational choice

The solver computes:
```
V(a, z) = max_{c, l} { u(c, l, G) + λ * v(C; θ) + β * E[V(a', z')] }
s.t.  a' = (1+r)*a + w*z*(1-l) - c - tax + transfer
      a' >= a_min, c >= 0, 0 <= l <= 1
```

**Traceability**: REQ-036, REQ-037, REQ-115..120, REQ-301..305, REQ-401

### Constitution Engine

**Location**: `src/emergent_constitution/constitution_engine.py`

Enforces constitutional rules with AST-sandboxed code execution. Rules contain Python expressions (`enforcement_code`) that are parsed and validated at the AST level to prevent unsafe operations.

**Key methods**:
- `validate_rule(rule) -> tuple[bool, str]` — AST-check enforcement code
- `enforce_taxes(households, constitution, market)` — Collect tax revenue
- `enforce_transfers(households, constitution, revenue)` — Distribute transfers
- `enforce_public_goods(constitution, revenue, num_agents)` — Compute public goods per capita
- `enforce_regulations(firms, households, constitution)` — Apply firm/market regulations
- `apply_proposal(proposal, constitution)` — Modify constitution via passed proposal

**Traceability**: REQ-022

### Market Clearing

**Location**: `src/emergent_constitution/market_clearing.py`

Finds equilibrium prices (wage `w_t`, interest rate `r_t`) via analytical FOCs or Walrasian bisection.

**Analytical mode** (default, REQ-101):
- Closed-form Cobb-Douglas FOCs: `w = (1-α) * Y/L`, `r = α * Y/K - δ`
- Zero market-clearing error every period

**Walrasian mode** (REQ-101..106):
- Bisection on prices with heterogeneous firm demands
- Iteratively adjusts prices until excess demand in both labor and capital markets falls below `tatonnement_tolerance`

**Key function**:
- `clear_markets(households, firms, aggregate_tfp, config, prev_market) -> MarketState`

**Traceability**: REQ-011 through REQ-014, REQ-101..106, PROP-003

### Shock Generators

**Location**: `src/emergent_constitution/shock_generators.py`

**Rouwenhorst discretization**: Converts the continuous AR(1) process `log(z') = ρ * log(z) + ε` into a discrete Markov chain with `num_z_states` grid points and a transition matrix.

**Key functions**:
- `rouwenhorst(n, rho, sigma)` — Build grid and transition matrix
- `draw_idiosyncratic_shocks(households, transition_matrix, grid, rng)` — Advance each agent's productivity index
- `draw_aggregate_tfp(prev_log_a, rho_a, sigma_a, rng)` — Draw aggregate TFP
- `draw_preference_shocks(households, rng, scale)` — Optional preference perturbations

**Traceability**: REQ-015, REQ-016, REQ-017

### EGMSolver

**Location**: `src/emergent_constitution/egm_solver.py`

Endogenous Grid Method solver for consumption-savings-labor problems using Carroll (2006) EGM with Fella (2014) upper envelope.

**Grid specification** (REQ-118):
- 200-point exponential asset grid: finer spacing near borrowing constraint
- 51-point leisure grid
- 7-state Rouwenhorst productivity Markov chain (default)

**Features**:
- Fast convergence: no inner maximization loop
- Policy cache: skip re-solving when market prices unchanged
- Supports political utility flow bonus (REQ-401)
- Two-asset extension: nested EGM for liquid/illiquid with convex adjustment costs (REQ-301..305)

**Key methods**:
- `solve_egm_cached(alpha, beta, gamma, beta_discount, wage, interest_rate, public_goods, tax_function, transfer, political_flow_bonus) -> (c_policy, l_policy)`
- `euler_residual(c_policy, l_policy, ...) -> float` — Convergence check (REQ-119)
- `solve_all(households, market, constitution_tax_rate, public_goods, transfer, political_flow_bonus) -> dict[str, EconomicDecision]`

**Traceability**: REQ-115..120, REQ-301..305, REQ-401

### Distribution

**Location**: `src/emergent_constitution/distribution.py`

Tracks the joint wealth-productivity distribution via Kolmogorov Forward Equation (KFE) using the Young (2010) lottery allocation method.

**Representation**:
- Grid-based probability mass array: `mu(a, z)` of shape `(n_a, n_z)`
- Each cell stores `Pr(a = a_grid[i], z = z_grid[j])`

**Operations**:
- `forward(policy_savings, transition_matrix)` — Advance distribution one period via KFE (REQ-201, REQ-202)
- `stationary(a_grid, z_grid, policy_savings, transition_matrix)` — Iterate to convergence (REQ-205)
- `aggregate(policy_fn)` — Compute aggregate quantities (REQ-203)
- `sample_agents(n_agents, rng)` — Draw individual agents from distribution (REQ-204)
- `gini()`, `mean_wealth()`, `percentiles()`, `top_share()` — Distributional statistics

**Traceability**: REQ-201..205, PROP-008

### Government

**Location**: `src/emergent_constitution/government.py`

Government sector managing debt, bonds, budget constraint, and fiscal rule.

**Budget constraint** (REQ-306, PROP-010):
```
B' = (1 + r^b) * B + G + Tr - T
```

**Bond market clearing** (REQ-307):
- Bisection on bond rate `r^b` to equate household bond demand with government supply
- Separate from capital market rate `r^k`

**Fiscal rule** (REQ-309):
- Automatic tax adjustment when `B/Y > debt_gdp_max` (default 1.5)
- Stabilizes debt dynamics

**Key components**:
- `GovernmentState` — tracks debt, tax revenue, spending, transfers, bond rate, debt-to-GDP
- `Government.update_budget()` — Apply budget constraint
- `Government.fiscal_rule()` — Auto-adjust taxes for debt sustainability
- `clear_bond_market()` — Find equilibrium bond rate

**Traceability**: REQ-306..309, PROP-010

### NominalBlock

**Location**: `src/emergent_constitution/nominal.py`

New Keynesian sticky prices with Rotemberg adjustment costs, Taylor rule, and Fisher equation.

**Taylor rule** (REQ-312):
```
i_t = r_bar + phi_pi * (pi_t - pi_bar) + phi_y * gap_t
```

**Fisher equation** (REQ-313):
```
r^b_t = (1 + i_t) / (1 + E[pi_{t+1}]) - 1
```

**NKPC** (REQ-311):
```
phi_p * pi_t * (pi_t - pi_bar) = (1 - epsilon) + epsilon * mc_t
    + beta * phi_p * E[pi_{t+1}] * (E[pi_{t+1}] - pi_bar) * Y_{t+1}/Y_t
```

**Algorithm**:
- Outer iteration to find inflation consistent with Taylor-Fisher-NKPC system
- Adaptive expectations: `E[pi'] = rho * pi_{t-1}`

**Traceability**: REQ-310..315

### PoliticalUtility

**Location**: `src/emergent_constitution/political_utility.py`

Microfounded political preferences integrated into the Bellman equation.

**Political utility function** (REQ-402):
```
v(C; theta) = theta_eq * f_eq(C) + theta_lib * f_lib(C)
  where f_eq(C) = -Gini(C), f_lib(C) = -tau(C)
```

**Bellman integration** (REQ-401):
```
V(a, z) = max_{c,l} { u(c,l,G) + lambda * v(C; theta) + beta * E[V(a',z')] }
```

**Proposal evaluation** (REQ-403, REQ-404):
- Compare Bellman values under proposed vs current constitution
- LLM receives Bellman-derived recommendation as structured context
- Log when LLM overrides Bellman recommendation

**Traceability**: REQ-401..405

### Calibrator

**Location**: `src/emergent_constitution/calibration.py`

Simulated Method of Moments (SMM) calibration for model parameters.

**Target moments** (REQ-208):
- Wealth Gini coefficient
- Entrepreneur share
- Top 10% wealth share
- Median MPC (Phase 3, two-asset)
- Liquid/illiquid wealth ratio (Phase 3, two-asset)

**SMM calibration** (REQ-209):
- Nelder-Mead derivative-free optimization
- Moment weights for relative importance
- Parameter bounds enforcement

**Key methods**:
- `compute_model_moments(agents)` — Extract moments from simulation
- `smm_objective(params, targets, simulate_fn)` — Weighted moment distance
- `calibrate(initial_params, targets, simulate_fn, bounds)` — Full SMM calibration (REQ-210)

**Traceability**: REQ-208..210

### ObserverV2

**Location**: `src/emergent_constitution/observer.py`

Computes 20+ macroeconomic statistics at each observer interval.

**Statistics computed** (REQ-030):
- Gini coefficient (wealth inequality)
- Pareto efficiency score
- Aggregate output (Y_t), consumption (C_t), investment (I_t)
- Mean and median wealth
- Wealth quantiles (p10, p25, p50, p75, p90)
- Unemployment rate
- Number of active firms and mean firm size
- Aggregate R&D spending
- Social welfare (sum of realized utilities)
- Cumulative discounted welfare
- Equilibrium wage and interest rate
- Rule changes from votes

**Key methods**:
- `observe(period_state) -> HistoryEntryV2`
- `finalize(final_state) -> SimulationOutputV2`

**Traceability**: REQ-030, REQ-031, REQ-032

### Reporter

**Location**: `src/emergent_constitution/reporter.py`

Generates human-readable output from simulation results.

**v2 functions**:
- `generate_report_v2(output) -> str` — Markdown report with 7 sections:
  1. Simulation summary (seed, periods, agents, mode)
  2. Final constitution (all rules with parameters)
  3. Household wealth distribution (quantiles, Gini, top/bottom shares)
  4. Firm summary (count, mean capital, total output, R&D)
  5. Welfare summary (total welfare, per-agent comparison if available)
  6. Constitutional timeline (rule changes by period)
  7. Statistics evolution (time series table)
- `generate_json_v2(output) -> str` — Full JSON serialization

## Data Flow: 9-Step Period Lifecycle (with HANK Extensions)

```
Period t begins
│
├─ Step 1: DRAW SHOCKS
│  ├─ Idiosyncratic: Rouwenhorst Markov chain transition for each household
│  ├─ Aggregate: AR(1) process for TFP: log(A_t) = ρ_A * log(A_{t-1}) + ε_t
│  └─ Preference: Optional noise on utility weights
│
├─ Step 2: CLEAR MARKETS
│  ├─ Input: households (labor supply), firms (capital + labor demand), A_t
│  ├─ Analytical mode: closed-form FOCs w = (1-α)Y/L, r = αY/K - δ
│  ├─ Walrasian mode: iterate w_t, r_t until |excess demand| < tolerance
│  ├─ HANK: Government bond market clearing (if government debt > 0)
│  │   └─ Bisect on r^b to equate bond demand = bond supply
│  └─ Output: MarketState(wage, interest_rate, bond_rate, aggregates)
│
├─ Step 3: COLLECT DECISIONS
│  ├─ LLM path: build context → batch API call → parse JSON → validate
│  │   ├─ EconomicDecision: consumption c_t, leisure l_t (deposit d_t in two-asset mode)
│  │   ├─ EntrepreneurialDecision: create/close firm, capital, R&D
│  │   └─ PoliticalDecision: proposal, votes (with Bellman context if enabled)
│  └─ Benchmark path: EGMSolver.solve_all() / NumericalSolver.solve_all() → optimal policy
│      ├─ EGM: Carroll (2006) + Fella (2014) upper envelope, 200-point grid
│      ├─ Political utility: lambda * v(C; theta) added to flow utility (REQ-401)
│      └─ Two-asset: nested EGM for liquid/illiquid with adjustment costs (REQ-301..305)
│
├─ Step 4: VALIDATE CONSTRAINTS
│  ├─ Budget: c_t ≤ (1+r)*a_t + w*z*(1-l) - tax + transfer
│  ├─ Borrowing: a_{t+1} ≥ a_min (b_{t+1} ≥ b_min in two-asset mode)
│  └─ Non-negativity: c_t ≥ 0, 0 ≤ l_t ≤ 1, k_{t+1} ≥ 0
│
├─ Step 5: EXECUTE PRODUCTION
│  ├─ Each firm: Y_f = A_f * K_f^α * L_f^(1-α)
│  ├─ Income distribution: wages to workers, capital returns, profit to owner
│  ├─ R&D: stochastic TFP improvement based on spending
│  └─ Firm creation/liquidation from entrepreneurial decisions
│
├─ Step 6: ENFORCE CONSTITUTION
│  ├─ Taxes: collect revenue per tax_schedule rules
│  ├─ Public goods: allocate fraction of revenue
│  ├─ Transfers: distribute remaining revenue per transfer_program rules
│  ├─ Regulations: apply market_regulation and firm_regulation rules
│  ├─ HANK: Government budget constraint B' = (1+r^b)*B + G + Tr - T (REQ-306)
│  ├─ HANK: Fiscal rule — auto-adjust taxes if B/Y > threshold (REQ-309)
│  └─ HANK: Nominal block update (if enabled, REQ-310..315)
│      ├─ Compute marginal cost mc_t = w / A
│      ├─ Taylor rule: i_t = r_bar + phi_pi*(pi - pi_bar) + phi_y*gap
│      ├─ Fisher equation: r^b = (1 + i) / (1 + E[pi']) - 1
│      └─ NKPC: solve for inflation pi_t
│
├─ Step 7: PROCESS GOVERNANCE (if proposal_interval)
│  ├─ Collect proposals from political decisions
│  ├─ Validate proposals against constitution schema
│  ├─ Tally votes per voting_procedure rule (majority/supermajority)
│  ├─ HANK: Bellman-derived vote recommendations (if pure_bellman_politics=True, REQ-405)
│  └─ Apply passed proposals to constitution
│
├─ Step 8: UPDATE STATES
│  ├─ New wealth: a_{t+1} = a_t - c_t + income - taxes + transfers
│  │   └─ Two-asset: b_{t+1}, k_{t+1} updated with deposit d_t and adjustment cost chi(d_t, k_t)
│  ├─ Realized utility: u(c_t, l_t, G_t) = c^α * l^β * G^γ
│  ├─ HANK: Distribution tracking (if distribution_mode="kfe")
│  │   ├─ Forward KFE step: mu_{t+1} = T(mu_t, policy_savings, transition_matrix)
│  │   └─ Aggregate moments from distribution (REQ-203)
│  └─ Validate: NaN/Inf checks, a_{t+1} ≥ a_min
│
└─ Step 9: OBSERVE (if observer_interval)
   ├─ Compute 20+ statistics (Gini, Pareto, welfare, etc.)
   ├─ HANK: Government state (debt, debt/GDP, bond rate, fiscal rule triggers)
   ├─ HANK: Nominal state (inflation, nominal rate, price level, output gap)
   ├─ HANK: Calibration moment comparison (if calibration_targets set, REQ-210)
   ├─ Detect rule changes vs previous observation
   └─ Append HistoryEntryV2 to history
```

## State Management

**Immutability principle**: All state objects (`HouseholdState`, `FirmState`, `MarketState`, `ConstitutionV2`) are deep-copied at the start of each period. All updates create new objects via `model_copy(update={...})`. No agent or component directly mutates shared state.

**State hierarchy**:
```
PeriodState
├── period: int
├── households: list[HouseholdState]    # 50+ agents with wealth, productivity, decisions
├── firms: list[FirmState]              # Active firms with capital, labor, output
├── market: MarketState                 # Equilibrium prices and aggregates
├── shocks: ShockState                  # Markov grid, TFP, preference shocks
├── constitution: ConstitutionV2        # Named rules with parameters
├── proposals: list[ConstitutionalProposal]
└── votes: list[VoteOutcomeV2]
```

## Determinism (PROP-001)

All randomness is controlled by a single seeded `SimulationRNG`:
- Rouwenhorst Markov chain transitions
- Aggregate TFP draws
- Preference shock draws
- R&D success/improvement draws
- Agent ordering in governance steps
- Coalition initialization

Same seed + config = identical results (tested in `test_properties.py`).

**Note**: LLM mode uses `temperature=0` for determinism, but API-level non-determinism means strict PROP-001 only holds for benchmark mode.

## v1 Backward Compatibility

The v1 API (`Lead`, `SimulationConfig`, `SimulationOutput`) is fully retained. v1 components remain in the same modules alongside v2 classes. The `__init__.py` exports v1 only; v2 classes must be imported from their specific modules.
