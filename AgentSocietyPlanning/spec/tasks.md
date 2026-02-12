# Implementation Tasks — v2: DSGE-HA with LLM Agents

> **Rules:** Two-level hierarchy (Task > Subtask). Sequential order. Each item links to requirement IDs.

---

## Phase 1 — Data Models and Configuration (Foundation)

### Task 1: Implement v2 data models

Replace v1 models with the DSGE-HA model hierarchy. All subsequent tasks depend on these types.

- [ ] **1.1** Create `models/household.py` with `UtilityParams` (alpha+beta+gamma=1 validator, beta_discount), `OccupationalRole` enum, `ValueVector`, and `HouseholdState` (wealth, productivity, productivity_index, role, firm_id, coalition_id, per-period outcomes: consumption, leisure, labor_supply, savings, income, taxes_paid, transfers_received, realized_utility). *Traceability:* `REQ-001`, `REQ-002`, `REQ-003`, `REQ-004`
- [ ] **1.2** Create `models/firm.py` with `FirmState` (id, owner_id, capital, labor_demand, tfp, worker_ids, rd_spend, output, profit). *Traceability:* `REQ-006`
- [ ] **1.3** Create `models/market.py` with `MarketState` (wage, interest_rate, aggregate_output, aggregate_consumption, aggregate_investment, government_spending, market_clearing_error, labor_excess_demand, capital_excess_demand). *Traceability:* `REQ-011`–`REQ-014`
- [ ] **1.4** Create `models/shocks.py` with `ShockState` (productivity_grid, transition_matrix, aggregate_tfp, preference_shocks). *Traceability:* `REQ-015`–`REQ-017`
- [ ] **1.5** Refactor `models/constitution.py` with `RuleType` enum, `ConstitutionalRule` (name, rule_type, parameters, description, enforcement_code, version, enacted_period), `Constitution` (rules dict, voting_rule, helper methods get_tax_rules/get_transfer_rules/get_active_voting_rule). Create default initial constitution factory. *Traceability:* `REQ-018`
- [ ] **1.6** Refactor `models/proposal.py` with `ConstitutionalProposal` (proposer_id, action literal, rule_name, rule_type, parameters, description, enforcement_code) and `VoteOutcome` (proposal, passed, votes_for, votes_against, total_eligible, voting_rule_used). *Traceability:* `REQ-019`–`REQ-021`
- [ ] **1.7** Create `models/decisions.py` with `EconomicDecision` (consumption, leisure), `EntrepreneurialDecision` (create_firm, capital_investment, labor_demand, rd_spend, close_firm), `PoliticalDecision` (proposal, votes dict). *Traceability:* `REQ-024`–`REQ-026`
- [ ] **1.8** Refactor `models/history.py` with expanded `HistoryEntry` (period, gini, pareto_score, aggregate_output/consumption/investment, mean/median_wealth, wealth_quantiles, unemployment_rate, num_active_firms, mean_firm_size, aggregate_rd_spend, social_welfare, cumulative_welfare, wage, interest_rate, rule_changes, constitution_snapshot), `WelfareSummary`, `SimulationOutput`, and `PeriodState`. *Traceability:* `REQ-030`–`REQ-032`
- [ ] **1.9** Write unit tests for all v2 models: validation constraints, serialization round-trip, default values, edge cases (UtilityParams sum != 1, negative wealth, etc.).

### Task 2: Refactor SimulationConfig

Expand configuration to support all DSGE-HA parameters.

- [ ] **2.1** Refactor `config.py` with full v2 `SimulationConfig`: core params (num_agents >= 20, max_periods, seed), shock params (rho_z, sigma_z, num_z_states, rho_A, sigma_A, enable_preference_shocks), production params (alpha, delta, min_firm_capital), household params (a_min, initial_wealth_mean/std), intervals (proposal_interval, observer_interval), market clearing params (tatonnement_max_iter, tatonnement_tolerance, tatonnement_step_size), LLM params (use_llm, llm_provider, llm_model, llm_temperature, llm_batch_size, llm_cache_enabled), benchmark params (benchmark_mode, solver_method), R&D params. Add Pydantic validators for all range constraints. *Traceability:* `REQ-034`
- [ ] **2.2** Write unit tests for config validation: valid defaults, invalid values rejected (num_agents=0, alpha>1, etc.), serialization.

### Task 3: Refactor initialization

Bootstrap the simulation with DSGE-HA initial conditions.

- [ ] **3.1** Refactor `initialization.py`: `initialize_simulation(config)` creates SimulationRNG, calls Rouwenhorst to get productivity grid/matrix, creates N `HouseholdState` agents (wealth from Normal clamped >= a_min, productivity_index from stationary distribution, utility_params from Dirichlet + Uniform beta_discount, value_vector random, role=WORKER), creates default constitution, creates initial `MarketState` with analytical guesses, creates initial `ShockState`, returns `(PeriodState, SimulationRNG)`. *Traceability:* `REQ-001`, `REQ-034`, `PROP-001`
- [ ] **3.2** Write unit tests: correct agent count, wealth >= a_min, productivity indices valid, determinism (same seed = same agents), constitution defaults present.

---

## Phase 2 — Stochastic Shocks

### Task 4: Implement shock generators

Build the Rouwenhorst discretization and shock-drawing functions.

- [ ] **4.1** Create `shocks.py` with `rouwenhorst_discretize(rho, sigma, num_states)` that returns `(grid_values, transition_matrix)`. Implement the recursive algorithm: build P_2, then P_n from P_{n-1}, normalize rows, convert grid from log to level. *Traceability:* `REQ-015`
- [ ] **4.2** Implement `draw_idiosyncratic_shocks(households, transition_matrix, grid, rng)`: for each agent, draw new productivity_index using cumulative transition probabilities and rng, update productivity = grid[new_index]. Return updated households. *Traceability:* `REQ-015`
- [ ] **4.3** Implement `draw_aggregate_tfp(prev_log_A, rho_A, sigma_A, rng)`: compute log(A_t) = rho_A * log(A_{t-1}) + epsilon, return exp(log_A_t). *Traceability:* `REQ-016`
- [ ] **4.4** Implement `draw_preference_shocks(households, rng, sigma)`: for each agent, draw beta perturbation from N(0, sigma^2), return dict of agent_id -> perturbation. *Traceability:* `REQ-017`
- [ ] **4.5** Write unit tests: Rouwenhorst grid matches known values (Kopecky & Suen 2010 Table 1), transition matrix rows sum to 1, matrix is stochastic, idiosyncratic shocks are deterministic with fixed seed, aggregate TFP stays positive, preference shocks have correct distribution.

---

## Phase 3 — Economics Core

### Task 5: Implement DSGE-HA economics engine

Replace v1 heuristic economics with microfounded production, budget constraints, and utility.

- [ ] **5.1** Implement `compute_budget(agent, wage, interest_rate, tax, transfer)`: returns `(1 + r) * a + w * z * labor_supply - tax + transfer`. *Traceability:* `REQ-003`
- [ ] **5.2** Implement `produce_output(firm, alpha)`: returns `A_f * K_f^alpha * L_f^(1-alpha)` with epsilon guards on K_f, L_f to prevent 0^alpha. *Traceability:* `REQ-007`
- [ ] **5.3** Implement `distribute_firm_income(firm, wage, interest_rate, delta)`: compute wages_paid, capital_cost, profit; return tuple. *Traceability:* `REQ-008`
- [ ] **5.4** Implement `apply_rd_shock(firm, rng, config)`: stochastic TFP improvement based on rd_spend. Probability = base_prob * sqrt(rd_spend / mean_rd_spend), capped at 0.5. On success, A_f *= (1 + drawn_factor). *Traceability:* `REQ-009`
- [ ] **5.5** Implement `liquidate_firm(firm)`: return max(0, remaining_capital). *Traceability:* `REQ-010`
- [ ] **5.6** Implement `enforce_budget_constraint(decision, agent, budget, a_min)`: clamp leisure to [0,1], recompute budget with clamped labor, clamp consumption to [0, max_affordable], derive savings, assert a_{t+1} >= a_min, log corrections. *Traceability:* `REQ-004`, `REQ-005`, `PROP-002`, `PROP-004`
- [ ] **5.7** Implement `compute_realized_utility(agent, public_goods_per_capita)`: u = c^alpha * l^beta * G^gamma with guards for c=0 or l=0 (return 0 or use log utility with epsilon). *Traceability:* `REQ-002`, `REQ-028`, `PROP-005`
- [ ] **5.8** Implement `validate_agent_states(households)`: check for NaN/Inf in wealth, consumption, utility; raise `NumericalInstabilityError` with context.
- [ ] **5.9** Write unit tests: budget computation against manual calculation, Cobb-Douglas output for known K/L/A, profit = Y - wL - (r+d)K, constraint projection cases (over-consuming agent, agent at borrowing limit, negative leisure), utility for known values, NaN detection.

---

## Phase 4 — Market Clearing

### Task 6: Implement tatonnement market clearing

Find equilibrium prices (w, r) each period via iterative price adjustment.

- [ ] **6.1** Implement `compute_firm_demands(firms, wage, interest_rate, delta, alpha)`: solve FOCs analytically for each firm's optimal K_f, L_f given prices. FOC capital: K_f = L_f * ((alpha * A_f) / (r + delta))^(1/(1-alpha)). FOC labor: L_f from (1-alpha) * A_f * (K_f/L_f)^alpha = w. Return (total_labor_demand, total_capital_demand). *Traceability:* `REQ-011`, `REQ-012`
- [ ] **6.2** Implement `clear_markets(households, firms, aggregate_tfp, config, prev_market, rng)`: tatonnement loop — initialize from prev_market or analytical guess, iterate (compute firm demands, compute supply from households, adjust prices by step * excess demand), clamp w > 0 and r > -delta, converge when both excess demands < tolerance. Compute Y, C, I, G. Return MarketState. Log if non-convergent. *Traceability:* `REQ-011`–`REQ-014`, `PROP-003`
- [ ] **6.3** Write unit tests: single-firm analytical equilibrium (known closed-form w*, r*), convergence within tolerance, non-convergence logging, prices stay positive, aggregate resource constraint Y = C + I + G verified.

---

## Phase 5 — Numerical Benchmark Solver

### Task 7: Implement VFI/EGM household solver

Build the fallback and benchmark decision engine.

- [ ] **7.1** Implement `NumericalSolver.__init__(config)`: set up asset grid (e.g., 100 points from a_min to a_max), store productivity grid and transition matrix from config/shocks, set convergence tolerance. *Traceability:* `REQ-036`
- [ ] **7.2** Implement `NumericalSolver.solve_household(agent, wage, interest_rate, public_goods, tax_function, transfer)`: solve Bellman equation via EGM. Discretize labor choice (or use FOC for interior solution). Iterate value function until convergence. Interpolate policy function at agent's current (a, z). Return `EconomicDecision(consumption, leisure)`. *Traceability:* `REQ-036`, `REQ-027`
- [ ] **7.3** Implement `NumericalSolver.solve_all(households, market, constitution)`: extract tax function and transfer amount from constitution, call solve_household for each agent, return dict. *Traceability:* `REQ-037`
- [ ] **7.4** Write unit tests: compare to analytical solution for log utility with no labor choice (c = (1-beta) * resources), verify policy function monotonicity (higher wealth → higher consumption), verify borrowing constraint binds for low-wealth agents, determinism.

---

## Phase 6 — Constitution Engine

### Task 8: Implement extensible constitution enforcement

Build the rule validation, enforcement, and sandboxed evaluation engine.

- [ ] **8.1** Implement `ConstitutionEngine.validate_rule(rule)`: check tax rates in valid range, transfer budgets feasible, enforcement_code passes AST safety check (no imports, no builtins abuse, no exec/eval), no contradictions with existing rules. Return (is_valid, reason). *Traceability:* `REQ-023`, `PROP-006`
- [ ] **8.2** Implement sandbox evaluator for `enforcement_code`: parse with `ast.parse`, whitelist allowed node types (BinOp, Compare, IfExp, Num, Str, Name, Attribute), reject dangerous nodes, evaluate with restricted globals (only math functions), enforce 100ms timeout. *Traceability:* Design §7.1
- [ ] **8.3** Implement `ConstitutionEngine.enforce_taxes(households, constitution, market)`: iterate tax rules, compute T(y_i) per rule parameters, deduct from agent wealth, accumulate revenue. Return (updated households, total_revenue). *Traceability:* `REQ-022`
- [ ] **8.4** Implement `ConstitutionEngine.enforce_transfers(households, constitution, revenue)`: iterate transfer rules, distribute revenue per rule (equal share, means-tested, etc.). Return updated households. *Traceability:* `REQ-022`
- [ ] **8.5** Implement `ConstitutionEngine.enforce_public_goods(constitution, revenue, num_agents)`: compute G_t = fraction_of_revenue * remaining_revenue / num_agents. Return public_goods_per_capita. *Traceability:* `REQ-022`
- [ ] **8.6** Implement `ConstitutionEngine.enforce_regulations(firms, households, constitution)`: apply market regulation and firm regulation rules (e.g., minimum wage, price controls). Return (updated firms, updated households). *Traceability:* `REQ-022`
- [ ] **8.7** Implement `ConstitutionEngine.apply_proposal(constitution, proposal)`: handle add/modify/remove actions. Validate before applying. Increment version on modify. Return new Constitution. *Traceability:* `REQ-021`
- [ ] **8.8** Write unit tests: valid/invalid rule detection, sandbox rejects dangerous code, flat tax computes correctly, progressive tax brackets, equal-share transfer, means-tested transfer, public goods calculation, proposal add/modify/remove, PROP-006 invariant.

---

## Phase 7 — LLM Decision Engine

### Task 9: Implement LLM provider abstraction

Create the pluggable provider interface and implementations.

- [ ] **9.1** Define `LLMProvider` protocol with `generate(messages, schema)` and `generate_batch(batch, schema)` methods. *Traceability:* `REQ-024`
- [ ] **9.2** Implement `MockProvider`: deterministic mock that delegates to `NumericalSolver` for economic decisions and uses rule-based logic for political decisions. Essential for testing without API calls. *Traceability:* `REQ-027`
- [ ] **9.3** Implement `AnthropicProvider`: calls Anthropic Messages API with structured output (tool_use or JSON mode), temperature=0, prompt caching enabled. Handles rate limits with exponential backoff. *Traceability:* `REQ-024`, `REQ-029`, `PROP-001`
- [ ] **9.4** Write unit tests: MockProvider returns valid decisions, AnthropicProvider (mocked HTTP) handles success/timeout/rate-limit/parse-error.

### Task 10: Implement LLM decision engine

Build the batching, caching, context-building, and fallback logic.

- [ ] **10.1** Implement `LLMDecisionEngine.__init__(config)`: instantiate provider (Mock or Anthropic based on config), instantiate NumericalSolver for fallback, initialize response cache (dict). *Traceability:* `REQ-024`, `REQ-029`
- [ ] **10.2** Implement `_build_context(agent, market, constitution, history_summary)`: format structured context per REQ-025 — (a) personal state, (b) prices + aggregates + distribution stats, (c) constitution rules summary, (d) recent history, (e) placeholder for agent messages. Return dict. *Traceability:* `REQ-025`
- [ ] **10.3** Implement `_cache_key(context)`: deterministic hash (JSON-serialize context, SHA-256). Check cache before calling provider. *Traceability:* `REQ-029`
- [ ] **10.4** Implement `_batch_call(prompts, schema)`: split prompts into batches of `llm_batch_size`, call provider.generate_batch for each, parse responses against schema, handle failures per Design §6.3 (retry once on parse error, fallback on persistent failure). *Traceability:* `REQ-026`, `REQ-027`, `REQ-029`
- [ ] **10.5** Implement `collect_economic_decisions(households, market, constitution, history_summary)`: build context per agent, check cache, batch uncached calls, validate responses, fall back to numerical solver on failure, return dict[agent_id, EconomicDecision]. *Traceability:* `REQ-003`, `REQ-024`–`REQ-027`
- [ ] **10.6** Implement `collect_entrepreneurial_decisions(households, firms, market, constitution)`: similar pattern, only for agents with role ENTREPRENEUR or sufficient wealth. Schema: `EntrepreneurialDecision`. *Traceability:* `REQ-006`, `REQ-024`
- [ ] **10.7** Implement `collect_political_decisions(households, constitution, market)`: collect proposals and votes. Schema: `PoliticalDecision`. *Traceability:* `REQ-019`, `REQ-020`, `REQ-024`
- [ ] **10.8** Write unit tests: context building includes all 5 components, cache hit skips LLM call, batch splitting correct, fallback triggered on mock failure, all decision types return valid schemas.

---

## Phase 8 — Lead Orchestrator (9-Step Period Loop)

### Task 11: Refactor Lead with 9-step period lifecycle

Rewrite the Lead to orchestrate all DSGE-HA components in the correct order.

- [ ] **11.1** Refactor `Lead.__init__(config)`: initialize SimulationRNG, call `initialize_simulation`, create `LLMDecisionEngine` (or `NumericalSolver` if benchmark_mode), create `ConstitutionEngine`, create `Observer`. Store period state, firm list, market state. *Traceability:* `REQ-033`, `REQ-034`
- [ ] **11.2** Implement `Lead._draw_shocks(t)` (step 1): call `draw_idiosyncratic_shocks` to transition all agents' productivity, call `draw_aggregate_tfp`, optionally call `draw_preference_shocks`. Update ShockState. *Traceability:* `REQ-015`–`REQ-017`
- [ ] **11.3** Implement `Lead._clear_markets()` (step 2): call `clear_markets(households, firms, aggregate_tfp, config, prev_market, rng)`. Store resulting MarketState. *Traceability:* `REQ-011`–`REQ-014`
- [ ] **11.4** Implement `Lead._collect_decisions()` (step 3): call `llm_engine.collect_economic_decisions` for all agents, call `collect_entrepreneurial_decisions` for entrepreneurs, store decisions. If benchmark_mode, use `numerical_solver.solve_all` instead. *Traceability:* `REQ-024`–`REQ-029`
- [ ] **11.5** Implement `Lead._validate_constraints()` (step 4): for each agent, call `enforce_budget_constraint` on their economic decision. Log all corrections. *Traceability:* `REQ-004`, `REQ-005`, `PROP-002`, `PROP-004`
- [ ] **11.6** Implement `Lead._execute_production()` (step 5): for each firm, call `produce_output` and `distribute_firm_income`. Apply R&D shocks via `apply_rd_shock`. Process firm creation (from entrepreneurial decisions) and liquidation. *Traceability:* `REQ-006`–`REQ-010`
- [ ] **11.7** Implement `Lead._enforce_constitution()` (step 6): call `constitution_engine.enforce_taxes`, `enforce_transfers`, `enforce_public_goods`, `enforce_regulations`. Compute realized utility for all agents. *Traceability:* `REQ-022`, `REQ-028`
- [ ] **11.8** Implement `Lead._process_governance(t)` (step 7): if `t % proposal_interval == 0`, collect political decisions, validate proposals via `constitution_engine.validate_rule`, tally votes via updated `voting.py`, apply passed proposals via `constitution_engine.apply_proposal`. *Traceability:* `REQ-019`–`REQ-023`, `PROP-006`
- [ ] **11.9** Implement `Lead._update_states()` (step 8): apply economic decisions to household wealth (a_{t+1}), update firm states, update agent roles (new entrepreneurs, released workers), validate all states. *Traceability:* `REQ-001`, `REQ-003`
- [ ] **11.10** Implement `Lead._observe(t)` (step 9): if `t % observer_interval == 0`, call `observer.observe(period_state, prev_constitution)`. *Traceability:* `REQ-030`, `REQ-031`
- [ ] **11.11** Implement `Lead.run()`: loop `_advance_period(t)` for t = 1..max_periods, call `observer.finalize()`, return `SimulationOutput`. *Traceability:* `REQ-033`
- [ ] **11.12** Write integration tests: run 1 period with MockProvider, verify all 9 steps executed (shocks drawn, prices found, decisions collected, constraints enforced, production computed, constitution enforced, states updated, observation recorded). Verify PROP-002 (budget consistency) and PROP-004 (non-negativity) hold after the period.

---

## Phase 9 — Observer and Reporter

### Task 12: Expand Observer with DSGE-HA statistics

Add firm statistics, welfare computation, and expanded wealth distribution.

- [ ] **12.1** Refactor `Observer.observe()`: compute all REQ-030 statistics — Gini, Pareto score, Y/C/I aggregates, mean/median wealth, wealth quantiles (p10/p25/p50/p75/p90), unemployment rate (UNEMPLOYED / total), num_active_firms, mean_firm_size (workers per firm), aggregate_rd_spend, social_welfare (sum of realized utilities), cumulative discounted welfare, wage, interest_rate, rule changes vs prev constitution. *Traceability:* `REQ-030`, `REQ-031`
- [ ] **12.2** Implement `Observer.finalize(period_state)`: create `WelfareSummary` and `SimulationOutput` per REQ-032. Include welfare comparison if benchmark data available. *Traceability:* `REQ-032`
- [ ] **12.3** Write unit tests: Gini = 0 for equal wealth, Gini > 0 for unequal, correct quantile computation, unemployment rate = fraction with UNEMPLOYED role, firm stats correct.

### Task 13: Expand Reporter

Update Markdown and JSON output for v2 data structures.

- [ ] **13.1** Refactor `reporter.py` to handle v2 `SimulationOutput`: add firm summary section, welfare summary section, expanded statistics table (all HistoryEntry fields), expanded wealth distribution (quantiles). *Traceability:* `REQ-032`
- [ ] **13.2** Update JSON output to serialize all v2 models (firms, welfare, market state). *Traceability:* `REQ-032`
- [ ] **13.3** Write unit tests: Markdown has all expected sections, JSON is parseable with all expected keys.

---

## Phase 10 — Voting and Coalitions

### Task 14: Generalize voting for extensible rules

Update voting to support constitution-defined voting procedures.

- [ ] **14.1** Refactor `voting.py`: `tally_votes` reads voting rule from constitution (not a fixed enum). Support majority, supermajority, unanimity, and custom threshold from rule parameters. *Traceability:* `REQ-020`
- [ ] **14.2** Update `validate_proposal` for v2 `ConstitutionalProposal` format (action, rule_name, rule_type, parameters). *Traceability:* `REQ-019`
- [ ] **14.3** Write unit tests: majority passes at >50%, supermajority at >=2/3, custom threshold from constitution, invalid proposals rejected.

### Task 15: Update coalition formation

Minor updates to coalition.py for v2 models.

- [ ] **15.1** Update `form_coalitions` and `compute_coalition_stats` to work with `HouseholdState` (instead of AgentState). No algorithmic changes needed. *Traceability:* Design §9 migration table
- [ ] **15.2** Write unit tests: coalition formation produces valid coalition_ids, stats computed correctly.

---

## Phase 11 — CLI and End-to-End

### Task 16: Refactor CLI entry point

Add v2 configuration flags and execution modes.

- [ ] **16.1** Refactor `__main__.py` with flags for all key SimulationConfig fields: `--agents`, `--periods`, `--seed`, `--rho-z`, `--sigma-z`, `--rho-A`, `--sigma-A`, `--alpha`, `--delta`, `--a-min`, `--proposal-interval`, `--observer-interval`, `--benchmark` (enables benchmark_mode), `--llm-provider`, `--llm-model`, `--output`, `--json`, `--quiet`, `--version`. *Traceability:* `REQ-035`
- [ ] **16.2** Write end-to-end tests: module runs via subprocess, `--benchmark` flag produces output without LLM calls, `--json` flag produces valid JSON, `--version` prints version, invalid config exits nonzero.

---

## Phase 12 — Property Tests and Validation

### Task 17: Implement property-based invariant tests

Verify all 6 properties hold across multi-period simulation runs.

- [ ] **17.1** Test PROP-001 (Determinism): run simulation twice with same seed and MockProvider, assert outputs are identical byte-for-byte. *Traceability:* `PROP-001`
- [ ] **17.2** Test PROP-002 (Budget consistency): run 20-period simulation, for every agent every period assert `a_{t+1} = (1+r)*a_t + income - c - taxes + transfers` within epsilon. *Traceability:* `PROP-002`
- [ ] **17.3** Test PROP-003 (Market clearing): run 20-period simulation, assert `market_clearing_error < tolerance` every period. Assert `|Y - C - I - G| < tolerance`. *Traceability:* `PROP-003`
- [ ] **17.4** Test PROP-004 (Non-negativity): run 20-period simulation, assert `c >= 0`, `a >= a_min`, `K >= 0`, `L >= 0` for all agents and firms every period. *Traceability:* `PROP-004`
- [ ] **17.5** Test PROP-005 (Welfare measurability): for every agent, compute `u(c, l, G)` from realized values and utility params, assert it equals `realized_utility` stored on agent. *Traceability:* `PROP-005`
- [ ] **17.6** Test PROP-006 (Constitutional validity): run simulation with governance periods, assert every active rule in the constitution passes `validate_rule()` at every observation point. *Traceability:* `PROP-006`
- [ ] **17.7** Write a "smoke test" that runs 50 agents for 50 periods in benchmark mode and verifies all 6 properties hold simultaneously. Mark as `@pytest.mark.slow`.

---

## Summary

| Phase | Tasks | Subtasks | Key Requirements |
|-------|-------|----------|-----------------|
| 1. Foundation | 1–3 | 13 | REQ-001–004, REQ-006, REQ-011–021, REQ-024–026, REQ-030–034 |
| 2. Shocks | 4 | 5 | REQ-015–017 |
| 3. Economics | 5 | 9 | REQ-002–010, PROP-002, PROP-004, PROP-005 |
| 4. Market Clearing | 6 | 3 | REQ-011–014, PROP-003 |
| 5. Numerical Solver | 7 | 4 | REQ-027, REQ-036–037 |
| 6. Constitution | 8 | 8 | REQ-018–023, PROP-006 |
| 7. LLM Engine | 9–10 | 12 | REQ-024–029, PROP-001 |
| 8. Lead Orchestrator | 11 | 12 | REQ-033, all integration |
| 9. Observer/Reporter | 12–13 | 6 | REQ-030–032 |
| 10. Voting/Coalitions | 14–15 | 5 | REQ-019–020 |
| 11. CLI | 16 | 2 | REQ-035 |
| 12. Property Tests | 17 | 7 | PROP-001–006 |

**Total: 17 tasks, 86 subtasks**

Estimated implementation order allows each phase to build on tested foundations from previous phases. Phase 8 (Lead orchestrator) is the integration point where all components come together.
