# Implementation Tasks — HANK Upgrade

> **Convention:** `[ ]` = pending, `[x]` = complete. Tasks are sequential within each phase.
> **Dependency:** Phase 2 depends on Phase 1. Phase 3 depends on Phase 2. Phase 4 depends on Phase 3.

---

## Phase 1: Fix Economic Foundations

### Task 1.1: Implement EGM Solver
*Traceability: REQ-115, REQ-116, REQ-117, REQ-118, REQ-119, REQ-120*

- [ ] **1.1.1** Create `src/emergent_constitution/egm_solver.py` with `EGMSolver` class. Implement 200-point exponential asset grid, Euler equation inversion for Cobb-Douglas utility, intratemporal FOC for leisure `l* = (beta_u * c) / (alpha_u * w * z)`, and endogenous-to-exogenous grid interpolation (monotone piecewise cubic).
- [ ] **1.1.2** Implement the constrained region handler: for wealth below the lowest endogenous grid point, set `a' = a_min` and compute consumption from the budget constraint.
- [ ] **1.1.3** Implement the upper envelope algorithm (Fella 2014) to handle non-monotonicity in the endogenous grid near kinks in the policy function.
- [ ] **1.1.4** Add Euler equation residual computation: `max |1 - c_euler/c_policy|` across the grid. Verify residual < `1e-6`.
- [ ] **1.1.5** Implement `EGMSolver.solve_all()` with the same interface as `NumericalSolver.solve_all()` — returns `dict[str, EconomicDecision]`.
- [ ] **1.1.6** Add `solver_method` config parameter (`"egm"`, `"vfi"`, `"vfi_numpy"`). Wire `LeadV2` to dispatch to EGM or VFI based on config.
- [ ] **1.1.7** Create `tests/test_egm_solver.py`: EGM accuracy vs brute-force VFI (max deviation < 1e-4), Euler residual < 1e-6, monotonicity of policy functions, constrained region correctness, homogeneous-preference speedup, cache behavior.
- [ ] **1.1.8** Verify all 977 existing tests still pass with `solver_method="vfi_numpy"` (backward compat).

### Task 1.2: Walrasian Market Clearing
*Traceability: REQ-101, REQ-102, REQ-103, REQ-104, REQ-105, REQ-106*

- [ ] **1.2.1** Implement `_bisect_wage(firms, labor_supply, alpha)` in `market_clearing.py`: bisection on excess labor demand `ELD(w) = sum_f L_f*(w) - L^s`, tolerance `1e-8`, max 100 iterations. Includes bracket-finding loop (double `w_hi` until `ELD < 0`).
- [ ] **1.2.2** Implement `_compute_firm_outputs(firms, wage, alpha)` that computes per-firm optimal labor `L_f*(w)`, per-firm output `Y_f`, and returns aggregate output `Y = sum Y_f`.
- [ ] **1.2.3** Implement `clear_markets_walrasian()` that calls bisection for wage, computes aggregate output from firm-level production, derives interest rate from aggregate MPK.
- [ ] **1.2.4** Add fallback: when `len(firms) == 0`, dispatch to `_representative_firm_clearing()` (existing analytical method, renamed). *Traceability: REQ-105.*
- [ ] **1.2.5** Add `market_clearing_method` config parameter. Wire `LeadV2` to dispatch based on config.
- [ ] **1.2.6** Create `tests/test_walrasian_clearing.py`: clearing error < 1e-8 with heterogeneous firms, convergence within 100 iterations, fallback to representative firm, edge cases (single firm, all same TFP, zero labor supply).
- [ ] **1.2.7** Verify all existing tests still pass with `market_clearing_method="analytical"`.

### Task 1.3: Fix Entrepreneur Budget Constraint
*Traceability: REQ-107, REQ-108, REQ-109*

- [ ] **1.3.1** In `lead.py` `_update_household_states()`: for entrepreneurs, compute income as `pi_f` only (no `w*z*(1-l)` term). Tax is on profit, not labor income.
- [ ] **1.3.2** In `market_clearing.py` `_compute_labor_supply()`: exclude households with `role == ENTREPRENEUR` from labor supply aggregation.
- [ ] **1.3.3** In the period lifecycle Step 3: set `labor_supply = 0` for entrepreneurs on the household state, so their time doesn't enter the labor market.
- [ ] **1.3.4** Update `economics.py` `compute_budget()` (or equivalent) to dispatch on role: workers get labor income, entrepreneurs get firm profit.
- [ ] **1.3.5** Create `tests/test_entrepreneur_fix.py`: verify entrepreneur income = profit only, entrepreneur labor supply = 0, tax base = profit, budget constraint holds for both roles.
- [ ] **1.3.6** Verify all existing tests still pass. Update any tests that explicitly check entrepreneur labor income.

### Task 1.4: Fix Occupational Choice (Bellman-Based)
*Traceability: REQ-110, REQ-111, REQ-112, REQ-113, REQ-114*

- [ ] **1.4.1** In `entrepreneurial_solver.py`, implement `compute_firm_value_bellman(capital, wage, interest_rate)`: solve `V^F = (I - beta*Pi)^{-1} * pi` using `numpy.linalg.solve()`. Returns firm value for all ability grid points at given capital.
- [ ] **1.4.2** Implement `_optimal_capital_golden_section(ability_idx, wage, interest_rate, max_capital)`: golden-section search over 50+ evaluations to find `K*` maximizing `V^F(e, K)`.
- [ ] **1.4.3** Modify `compute_worker_value()` to accept the VFI/EGM value function `V^W(a, z)` and interpolate at the agent's actual state `(a_i, z_i)` — replacing the perpetuity formula.
- [ ] **1.4.4** Modify `compute_entrepreneur_value()` to use: firm profit from `V^F(e, K*)`, entrepreneur's leisure from intratemporal FOC (not hardcoded 0.2), consumption from budget constraint (not hardcoded 0.7).
- [ ] **1.4.5** Update `solve_entry_exit()` to pass the VFI value function to `compute_worker_value()`. Update the `solve_all()` interface to accept the value function.
- [ ] **1.4.6** Update `LeadV2` to pass the VFI/EGM value function to the entrepreneurial solver in Step 3.
- [ ] **1.4.7** Create `tests/test_occupational_choice.py`: firm value Bellman vs iterative solution (max diff < 1e-6), golden-section vs grid search (same optimum), worker value from VFI matches interpolation, entry/exit decisions consistent with value comparison, no hardcoded constants in value computation.
- [ ] **1.4.8** Verify all existing tests still pass.

---

## Phase 2: Scale and Calibrate

### Task 2.1: KFE Distribution Tracking
*Traceability: REQ-201, REQ-202, REQ-203, REQ-204, REQ-205*

- [ ] **2.1.1** Create `src/emergent_constitution/distribution.py` with `Distribution` class. Implement histogram on `(n_a, n_z)` grid with `forward()` method using Young (2010) lottery allocation.
- [ ] **2.1.2** Implement `stationary()` class method: iterate KFE forward until `||mu_{n+1} - mu_n||_1 < 1e-10` or 10,000 iterations.
- [ ] **2.1.3** Implement aggregate computation methods: `aggregate(policy_fn)`, `gini()`, `mean_wealth()`, `percentiles([10, 25, 50, 75, 90])`, `top_share(fraction)`.
- [ ] **2.1.4** Implement `sample_agents(n, rng)` for LLM mode: sample `n` individual agents from the distribution using inverse CDF.
- [ ] **2.1.5** Add `distribution_mode` config parameter (`"individual"`, `"kfe"`). Wire into `LeadV2`: in KFE mode, use distribution-based aggregates; in individual mode, use agent-based (existing behavior).
- [ ] **2.1.6** Create `tests/test_distribution.py`: mass conservation after forward, stationary distribution convergence, Gini computation matches individual-agent Gini (50,000 samples, KS < 0.05), lottery allocation correctness.
- [ ] **2.1.7** Verify all existing tests still pass with `distribution_mode="individual"`.

### Task 2.2: Expand Rouwenhorst Grids
*Traceability: REQ-206, REQ-207*

- [ ] **2.2.1** Change default `num_z_states` from 5 to 7 in `SimulationConfigV2`.
- [ ] **2.2.2** Change default `num_e_states` from 5 to 7 in `SimulationConfigV2`.
- [ ] **2.2.3** Update EGM solver and VFI solver to handle variable grid sizes (they already should, but verify).
- [ ] **2.2.4** Update tests that hardcode 5-state grids to parameterize on grid size.
- [ ] **2.2.5** Verify all existing tests pass (some may need updated expected values due to finer grid).

### Task 2.3: Calibration Module
*Traceability: REQ-208, REQ-209, REQ-210*

- [ ] **2.3.1** Create `src/emergent_constitution/calibration.py` with `CalibrationTargets` dataclass and `Calibrator` class.
- [ ] **2.3.2** Implement `compute_model_moments()`: wealth Gini, entrepreneur share, top 10% share from distribution or agent states.
- [ ] **2.3.3** Implement `smm_objective()`: weighted squared distance between model and target moments.
- [ ] **2.3.4** Implement `calibrate()` using `scipy.optimize.minimize` (Nelder-Mead) on the SMM objective. Parameters to calibrate: `beta_d`, `sigma_z`, `entry_cost`, `a_min`.
- [ ] **2.3.5** Add moment reporting to Observer: at each observation interval, report model moments alongside targets.
- [ ] **2.3.6** Add calibration targets to the reporter output.
- [ ] **2.3.7** Create `tests/test_calibration.py`: moment computation correctness, SMM objective gradient direction (perturbing params moves moments toward targets), calibration convergence on a simple test case.

---

## Phase 3: HANK Features

### Task 3.1: Two-Asset Household Structure
*Traceability: REQ-301, REQ-302, REQ-303, REQ-304, REQ-305*

- [ ] **3.1.1** Add `liquid: float = 0.0` and `illiquid: float = 0.0` fields to `HouseholdState`. Add `total_wealth` property. Ensure `wealth` field backward compat (when both are 0, use `wealth` as single asset).
- [ ] **3.1.2** Implement `adjustment_cost(deposit, illiquid_stock, chi_0, chi_1)` utility function.
- [ ] **3.1.3** Extend `EGMSolver` with `solve_two_asset()`: outer loop over deposit grid (30 points), inner EGM for liquid savings conditional on deposit, golden-section refinement on deposit choice.
- [ ] **3.1.4** Implement upper envelope for two-asset non-convexities.
- [ ] **3.1.5** Add `two_asset_mode` config parameter. When True, use two-asset solver; when False, single-asset (backward compat).
- [ ] **3.1.6** Update `LeadV2` Step 3/3b to collect portfolio decisions (deposit `d`) alongside consumption/leisure.
- [ ] **3.1.7** Update Step 8 (state update) to evolve `liquid` and `illiquid` separately: `b' = (1+r^b)*b + income - c - d - chi(d) - T + Tr`, `k' = (1+r^k)*k + d`.
- [ ] **3.1.8** Create `tests/test_two_asset.py`: budget constraint holds with two assets, adjustment cost computation, wealthy hand-to-mouth detection (high `k`, low `b`, high MPC), portfolio choice responds to return differential.

### Task 3.2: Government and Bond Market
*Traceability: REQ-306, REQ-307, REQ-308, REQ-309*

- [ ] **3.2.1** Create `src/emergent_constitution/government.py` with `GovernmentState` dataclass and `Government` class.
- [ ] **3.2.2** Implement `update_budget(state, bond_rate)`: `B' = (1+r^b)*B + G + Tr - T`.
- [ ] **3.2.3** Implement `fiscal_rule(state, output)`: auto-adjust tax when `B/Y > B_max`.
- [ ] **3.2.4** Add bond market clearing to `market_clearing.py`: bisect `r^b` to equate `sum_i b_i(r^b) = B_t`.
- [ ] **3.2.5** Wire Government into `LeadV2` lifecycle (new Step 7b).
- [ ] **3.2.6** Create `tests/test_government.py`: budget constraint identity, fiscal rule triggers at threshold, bond market clears, debt dynamics are stable with fiscal rule.

### Task 3.3: Nominal Rigidities Block
*Traceability: REQ-310, REQ-311, REQ-312, REQ-313, REQ-314, REQ-315*

- [ ] **3.3.1** Create `src/emergent_constitution/nominal.py` with `NominalState` dataclass and `NominalBlock` class.
- [ ] **3.3.2** Implement `taylor_rule(inflation, output_gap, r_natural)`.
- [ ] **3.3.3** Implement `fisher_equation(nominal_rate, expected_inflation)`.
- [ ] **3.3.4** Implement `nkpc(inflation, marginal_cost, expected_inflation, ...)` — the Rotemberg NKPC.
- [ ] **3.3.5** Implement `update()` method: given real economy outcomes, compute nominal state via outer iteration (adaptive expectations → Taylor → Fisher → NKPC → check convergence).
- [ ] **3.3.6** Add `nominal_rigidities` config flag. Wire NominalBlock into `LeadV2` (new Step 2b). Pass real bond rate to household solver.
- [ ] **3.3.7** Update all market clearing to use real rates when nominal block is active.
- [ ] **3.3.8** Create `tests/test_nominal.py`: Taylor rule computation, Fisher equation identity, NKPC at steady state (zero inflation → mc = (epsilon-1)/epsilon), nominal block convergence, interaction with bond market clearing.

### Task 3.4: Extended Market Clearing Integration
*Traceability: REQ-307, REQ-308, REQ-313*

- [ ] **3.4.1** Implement the joint market clearing loop: labor (bisect w) → capital (MPK) → nominal (Taylor → Fisher) → bonds (bisect r^b) → iterate on inflation expectations until convergence.
- [ ] **3.4.2** Add convergence monitoring: log iteration count, inflation convergence error, and market clearing errors per period.
- [ ] **3.4.3** Create integration test: full period with two assets, government, and nominal block. Verify all markets clear, Fisher holds, government budget balances.

---

## Phase 4: Microfound Political Decisions

### Task 4.1: Political Utility Function
*Traceability: REQ-401, REQ-402*

- [ ] **4.1.1** Create `src/emergent_constitution/political_utility.py` with `compute_political_utility(constitution, theta_eq, theta_lib)`. Implement `f_eq(C) = -Gini(C)` and `f_lib(C) = -tau(C)` components.
- [ ] **4.1.2** Add `political_lambda` config parameter (default 0.05).
- [ ] **4.1.3** Integrate political utility into EGM solver as a constant flow utility bonus (added to per-period utility between governance periods).
- [ ] **4.1.4** Create `tests/test_political_utility.py`: political utility computation, integration with Bellman equation, sensitivity to `theta` values.

### Task 4.2: Bellman-Derived Political Preferences
*Traceability: REQ-403, REQ-404, REQ-405*

- [ ] **4.2.1** Implement `evaluate_proposal(value_function, agent_state, proposed_constitution, current_constitution)`: returns `V(s; C_proposed) - V(s; C_current)` by re-solving the Bellman at the proposed constitution parameters.
- [ ] **4.2.2** Add Bellman-derived preference to LLM political context: "Your economic analysis suggests this proposal would [increase/decrease] your lifetime welfare by X%."
- [ ] **4.2.3** Add `pure_bellman_politics` config flag. When True, all political decisions use Bellman comparison, bypassing LLM.
- [ ] **4.2.4** Log deviation when LLM overrides Bellman recommendation (for analysis).
- [ ] **4.2.5** Create tests: Bellman preference matches manual value function comparison, LLM deviation logging works, pure-Bellman mode produces same decisions as manual comparison.

### Task 4.3: Integration and Regression Tests
*Traceability: PROP-001 through PROP-012*

- [ ] **4.3.1** Create comprehensive integration test: 100-period simulation with all Phase 1-4 features enabled. Verify all properties hold.
- [ ] **4.3.2** Create backward compatibility test: run with all new features disabled (default config minus new features), verify output matches v2 baseline to tolerance `1e-6`.
- [ ] **4.3.3** Run full test suite (`pytest`), verify all tests pass.
- [ ] **4.3.4** Run `ruff check .` and `ruff format .`, fix any issues.
- [ ] **4.3.5** Update `docs/model_paper.tex` with the HANK extensions (new sections on two-asset model, nominal rigidities, EGM, KFE).

---

## Summary

| Phase | Tasks | Subtasks | New Files | Modified Files |
|-------|-------|----------|-----------|----------------|
| 1 | 4 | 30 | 5 | 5 |
| 2 | 3 | 17 | 3 | 4 |
| 3 | 4 | 23 | 4 | 3 |
| 4 | 3 | 14 | 2 | 3 |
| **Total** | **14** | **84** | **14** | **~10 unique** |
