# Publication-Grade Audit Report
**Target:** `/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex` vs executable codebase  
**Date:** 2026-02-22  
**Standard:** Top-journal referee + replication engineer  
**Output format:** Severity-ranked engineering ledger (`P0`..`P3`)

## 1) Reproducibility and Truth Baseline
### 1.1 Commands run and outcomes
1. `pytest -q tests/test_hank_integration.py::TestNominalRigidities::test_nominal_block_integration`  
Result: **FAIL** with `AttributeError: 'LeadV2' object has no attribute '_compute_nominal'` at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:573`.
2. `pytest -q tests/test_hank_integration.py::TestTwoAssetMode::test_two_asset_simulation`  
Result: **PASS** (integration test only checks run completion and non-negative wealth/consumption).
3. `pytest -q tests/test_distribution.py -k "distribution_mode or kfe"`  
Result: **PASS** (config-level checks; not lifecycle integration checks).
4. `pytest -q tests/test_political_utility.py::TestLLMEnginePoliticalIntegration::test_pure_bellman_mode_returns_empty_decisions`  
Result: **PASS** and confirms pure Bellman mode returns no proposals and empty votes.
5. Custom diagnostic: entrepreneurial stationary distribution vs independent power iteration  
Result: solver `pi=[0.35, 0.333333, 0.316667]`; true `pi=[0.409091, 0.318182, 0.272727]`; `max_abs_diff=0.05909`.
6. Custom diagnostic: `distribution_mode="kfe"` simulation  
Result: simulation runs, but `LeadV2` has no `_distribution` state (`has__distribution_attr=False`).
7. Custom diagnostic: same seed, `two_asset_mode=False` vs `True`  
Result: identical final outcomes (`max_abs_wealth_diff=0.0`, `max_abs_consumption_diff=0.0`, `max_abs_labor_diff=0.0`, and zero liquid/illiquid differences).
8. Custom diagnostic: governance proposals at period 5 with duplicate `rule_name`  
Result: duplicated proposals on same `rule_name` receive identical vote totals (collision via `rule_name` keyed vote map).
9. Custom diagnostic: accounting identity check (period 5 sample run)  
Result: `market.aggregate_output` differs from `sum_firm_output`; large resource gaps under both measures (`Y-(C+I+G)` far from zero).

### 1.2 Baseline verdict
The current implementation does not support several central claims in the paper as executable facts. Core feature flags (nominal, KFE, two-asset, pure Bellman politics) are either broken, inert, or behaviorally inconsistent.

---

## 2) Claim-to-Implementation Matrix
Status legend: `implemented`, `approximation`, `not implemented`, `contradicted`, `untested`

| ID | Paper claim (line anchor) | Status | Code anchor(s) | Evidence |
|---|---|---|---|---|
| C1 | Full HANK stack integrated (abstract, `/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:64`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:573`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1389`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:793` | Nominal crash; KFE not wired; two-asset inert. |
| C2 | Bellman occupational mode replaces perpetuity mode (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:700`) | approximation | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:700`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:446` | Mixed implementation: Bellman branch exists, but continuation uses `max(VW,VF)` approximation and entrepreneurial utility still stylized. |
| C3 | Worker value in Bellman mode ensures exact consistency (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:708`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:251`; search shows no call from `LeadV2` | `set_value_function` is never called in runtime pipeline; Bellman political context usually absent. |
| C4 | Step 3 occupational comparison uses current public goods (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:733`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:791`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:813` | Step 3 uses `revenue=0` and `transfer=0`, not period-consistent fiscal quantities. |
| C5 | KFE mode tracks cross-sectional distribution instead of individual agents (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:751`) | not implemented | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1389`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py` (no call sites) | `_update_distribution` exists but is never invoked; no `_distribution` initialized. |
| C6 | Two-asset nested EGM operational (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:832`) | not implemented | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:138`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:793`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/models/household.py:106` | Grids/flags exist; no two-asset solve path and no state transition updates for `liquid`/`illiquid`. |
| C7 | Government fiscal rule adjusts taxes and can freeze governance (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:861`; `:869`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1266`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/government.py:135` | Fiscal adjustment is computed then only logged; no tax policy mutation and no governance freeze logic. |
| C8 | Bond rate enters household budget constraints (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:902`; `:824`) | approximation | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/models/market.py:24`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1250` | `MarketState` has one `interest_rate`; bond and capital rates not first-class separated in state transition. |
| C9 | Nominal block integrated and solved each period (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:908`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:573` | Runtime crash due missing method `_compute_nominal`. |
| C10 | Political utility enters Bellman and guides political choices (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:940`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:800`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:795` | `political_flow_bonus` exists in solver API but never passed by `LeadV2` (always default `0.0`). |
| C11 | `pure_bellman_politics` yields fully microfounded governance (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:959`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:480`; tests `/Users/antoniomele/Dropbox/github/AgentSociety/tests/test_political_utility.py:570` | Mode returns empty proposals and empty vote maps, not Bellman-generated governance actions. |
| C12 | LLM receives Bellman-derived context (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:963`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:464`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:469` | Context compares constitution to itself; plus value function is not set by lead pipeline. |
| C13 | Analytical clearing exact and zero-error (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:646`) | approximation | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:465`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:528` | Uses ad hoc `max()` aggregators (`K` and `Y`) for consistency, not exact equilibrium identities. |
| C14 | Governance voting maps to proposals (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:986`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1181`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/citizen_v2.py:169` | Votes keyed by `rule_name`, causing collisions when multiple proposals target same rule. |
| C15 | Resource identity representation (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:361`) | approximation | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:629`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/observer.py:285` | Market aggregates are partially stale post-production; no strict closure checks/enforcement. |
| C16 | Stationary ability distribution converges to tolerance (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:224`) | contradicted | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:144`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:149` | Convergence check is degenerate (difference computed after overwrite). |
| C17 | Legacy occupational approximations documented (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:232`; `:240`) | implemented | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:610`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:620`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:701` | Hardcoded leisure/consumption constants are present exactly as described. |
| C18 | Nominal-state equations solved jointly with convergence (`/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:908`) | untested | `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/nominal.py:214` | Unit tests for `NominalBlock` exist, but lead integration is broken. |

---

## 3) Math Integrity Ledger
### 3.1 Household Bellman and political utility
1. **Equation (Bellman with political flow bonus) is not active in lead lifecycle.**  
Paper: `/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:940`.  
Code: solver supports `political_flow_bonus` (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:800`), but `LeadV2` never supplies it (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:795`).  
Verdict: **core identification mismatch**.
2. **Proposal evaluation formula is explicitly approximate and not a full re-solve under proposed constitution.**  
Code comments admit approximation (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/political_utility.py:158`).  
Paper language around "structurally consistent" overstates this.

### 3.2 Entrepreneur block
1. **Stationary distribution convergence implementation is incorrect.**  
At `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:149`, the "difference" compares `new_pi/total` to `pi` after `pi` has been set to that same normalized vector (`:144`), forcing immediate pseudo-convergence.
2. **Bellman entrepreneur continuation is approximated via `max(V^W, V^F)` rather than a state-contingent expected value under transitions.**  
Code: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:443`.  
This can bias entry/exit thresholds.

### 3.3 Market clearing and resource closure
1. **Analytical path uses ad hoc reconciliation (`max`) for aggregate capital/output.**  
Code: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:465`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:528`.  
This is not a strict equilibrium derivation.
2. **Observer aggregates can be stale relative to executed production and state updates.**  
`market.aggregate_output` is set at Step 2 and not refreshed after Step 5; only consumption is refreshed (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:629`).

### 3.4 Government and bond block
1. **Fiscal rule is not operative in policy state transition.**  
`tax_adjustment` computed then logged, not applied (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1266`).  
2. **Governance freeze under explosive debt is not implemented despite paper text.**  
Only critical log exists (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/government.py:135`).
3. **Bond/capital rate separation is incomplete in state types.**  
`MarketState` has only `interest_rate` (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/models/market.py:24`), while paper equations distinguish `r` vs `r^b`.

### 3.5 Nominal block
1. **Nominal system cannot be solved in lead lifecycle because integration method is missing.**  
Call exists (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:573`), method absent.

---

## 4) Code Architecture Consistency Ledger
### 4.1 9-step lifecycle ordering and timing
1. Step 3 solves household and entrepreneurial decisions using `public_goods` computed from zero revenue and `transfer=0` (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:791`; `:813`).  
2. Taxes are computed in Step 6 from incomes set in Step 4b, before current-period firm production profits are consistently allocated.
3. Step 5 computes output/profits using old firm labor demand before applying owner decision labor (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:953`; `:968`).

### 4.2 Stock-flow and accounting
1. Household wealth transition has no explicit capital investment debit for new firm creation and no explicit liquidation credit (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1333`).  
2. Liquidation path comments that capital will be returned, but update logic does not carry this flow (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:945`; `:1333`).

### 4.3 Feature-flag integrity
1. `nominal_rigidities=True` => crash (hard failure).
2. `distribution_mode="kfe"` => flag accepted but inert in lifecycle.
3. `two_asset_mode=True` => flag accepted but behavior unchanged.
4. `pure_bellman_politics=True` => deterministic empty politics, not Bellman governance.

### 4.4 Governance interface consistency
1. Vote maps are keyed by `rule_name` not proposal identity (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1181`), causing collisions with multiple proposals on the same rule.

---

## 5) Literature Gap Ledger (2016-Feb 2026)
Tag legend: `core`, `important`, `optional`

### 5.1 Missing or underused references
1. **`core`** Auclert et al., Sequence-Space Jacobian (NBER w26123) and JPE publication framing for solving and estimating HA models at scale.  
2. **`core`** Auclert, Rognlie, Straub synthesis on fiscal/monetary policy with heterogeneity (NBER w32991 and Annual Review 2025).  
3. **`core`** Deficits/Inflation in HA environments and FTPL interactions (NBER w33102, rev. Jan 2026).  
4. **`important`** RANK-to-HANK without FIRE restrictions (NBER w34596, 2025).  
5. **`important`** Joint household and firm heterogeneity frontier (NBER w34611, 2025).  
6. **`important`** Monetary policy under Okun's hypothesis in HA settings (NBER w33488, 2025).  
7. **`important`** HANK-squared monetary union framework (JME 2024, DOI: 10.1016/j.jmoneco.2024.103579).  
8. **`important`** THANK analytical framework in ReStud (2025 issue).

### 5.2 Overreach vs frontier
1. The paper claims "full HANK" with two-asset/KFE/nominal/political Bellman integration, but executable implementation is not at frontier-comparable completeness.
2. Bibliography is heavily pre-2018 and under-represents 2020s canonical computational/equilibrium contributions.

### 5.3 Primary sources consulted
1. SSJ foundation: https://www.nber.org/papers/w26123
2. Fiscal and monetary policy with heterogeneity (NBER + review pointer): https://www.nber.org/papers/w32991
3. Household and firm heterogeneity frontier: https://www.nber.org/papers/w34611
4. RANK to HANK without FIRE: https://www.nber.org/papers/w34596
5. Deficits and inflation, HA + FTPL interactions (rev. Jan 2026): https://www.nber.org/papers/w33102
6. Monetary policy under Okun's hypothesis: https://www.nber.org/papers/w33488
7. HANK-squared monetary unions (JME): https://doi.org/10.1016/j.jmoneco.2024.103579
8. THANK framework (ReStud): https://academic.oup.com/restud/article-abstract/92/4/2398/7699676
9. Annual Review (2025): https://www.annualreviews.org/content/journals/10.1146/annurev-economics-091624-044646

---

## 6) Prioritized Findings (`P0`..`P3`)
Each item includes: what is wrong, economic importance, evidence, exact adjustment in paper/code, and suggested test.

### P0-1 Nominal block integration is broken (hard runtime failure)
- What is wrong: lead lifecycle calls missing method `_compute_nominal`.
- Why it matters economically: all nominal results are non-executable; any NK claims are invalid.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:573`; failing test `tests/test_hank_integration.py::TestNominalRigidities::test_nominal_block_integration`.
- Exact adjustment required:
  - Paper: qualify nominal section as "module-level only" until lifecycle integration is fixed.
  - Code: implement `_compute_nominal(market, shocks)` in `LeadV2` calling `NominalBlock.update`, then persist and expose `NominalState`.
- Suggested validation test: rerun nominal integration test plus 100-period nominal simulation with finite-state assertions and no `AttributeError`.

### P0-2 `distribution_mode="kfe"` is not integrated
- What is wrong: KFE update function exists but is never invoked; `_distribution` is never initialized.
- Why it matters economically: distribution dynamics/aggregates in KFE mode are absent; paper section is non-operative.
- Evidence: `_update_distribution` only definition at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1389`; no call sites in file; runtime diagnostic `has__distribution_attr=False`.
- Exact adjustment required:
  - Paper: downgrade to "partial implementation" unless lifecycle wiring added.
  - Code: initialize `Distribution` when `distribution_mode=="kfe"` and call `_update_distribution` each period with correct policies and transitions.
- Suggested validation test: assert KFE mass conservation and period-to-period movement in `mu`; compare KFE aggregates to sampled-agent aggregates on same policy.

### P0-3 Two-asset mode is flag-only (no behavioral effect)
- What is wrong: `two_asset_mode=True` produces identical outcomes to single-asset mode under identical seed.
- Why it matters economically: wealthy-HtM channel, policy transmission heterogeneity, and portfolio margins are absent.
- Evidence: solver scaffolding `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:138`; no two-asset solve branch in `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/egm_solver.py:793`; diagnostics show zero differences.
- Exact adjustment required:
  - Paper: remove claims about operational two-asset nested EGM unless implemented.
  - Code: implement `(b,k,z)` policy solve and update `HouseholdState.liquid/illiquid` transitions in lead lifecycle.
- Suggested validation test: fixed-seed A/B run should produce statistically meaningful differences in liquid/illiquid holdings and MPC distribution.

### P0-4 Bellman political integration is non-operative and internally degenerate
- What is wrong: (i) lead never supplies value function/Gini to LLM engine; (ii) political context compares constitution to itself; (iii) pure Bellman mode returns empty decisions.
- Why it matters economically: "fully microfounded governance" claim fails; political block does not represent Bellman-based institutional choice.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:469`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/llm_engine.py:480`; no runtime call sites for `set_value_function`/`set_observed_gini`.
- Exact adjustment required:
  - Paper: rewrite Bellman-politics section as prototype unless pipeline is wired.
  - Code: in `LeadV2`, set value function and observed gini each period; pass distinct current/proposed constitutions during proposal evaluation; in pure mode, generate proposal-level Bellman votes and proposal generation logic.
- Suggested validation test: non-empty proposal/vote outputs in pure mode; Bellman context changes when proposal parameters change.

### P0-5 Ability stationary distribution calculation is wrong
- What is wrong: convergence check is computed after overwrite, creating near-zero difference regardless of true convergence.
- Why it matters economically: stationary ability moments and firm value approximations are biased.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:144`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:149`; diagnostic max error `0.05909`.
- Exact adjustment required:
  - Paper: remove convergence-precision claim until fix.
  - Code: compute `max_diff` from pre-update `pi_old` vs normalized `new_pi`, then assign `pi=new_pi_norm`.
- Suggested validation test: compare against eigenvector/power-iteration benchmark across random stochastic matrices; enforce `max_abs_diff < 1e-10`.

### P0-6 Entrepreneur capital stock-flow accounting is inconsistent
- What is wrong: firm capital flows are not explicitly debited/credited in household budget transition despite firm creation/liquidation semantics.
- Why it matters economically: wealth dynamics, entry barriers, and entrepreneurial returns are mismeasured.
- Evidence: creation/liquidation in `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:987` and `:942`; wealth transition lacks capital-flow terms at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1333`.
- Exact adjustment required:
  - Paper: qualify budget equation implementation as currently partial.
  - Code: add explicit state variables for entrepreneurial capital accounts, debit on entry/investment, credit on liquidation/disinvestment, and include depreciation consistently.
- Suggested validation test: single-agent deterministic accounting test over entry-operate-exit sequence with exact stock-flow reconciliation.

### P1-1 Governance vote collisions due to `rule_name` keyed votes
- What is wrong: multiple proposals targeting same rule share the same vote key.
- Why it matters economically: proposal-level selection is corrupted; vote outcomes can be duplicated mechanically.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1181`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/citizen_v2.py:169`; reproduced duplicate outcomes.
- Exact adjustment required:
  - Paper: specify proposal-level identifiers in vote schema.
  - Code: add unique `proposal_id` and key vote maps by `proposal_id`.
- Suggested validation test: period with two proposals on same rule must permit different vote tallies.

### P1-2 Decision timing uses wrong fiscal inputs in Step 3
- What is wrong: solver decisions use `public_goods` from zero revenue and `transfer=0`, while fiscal quantities are realized later.
- Why it matters economically: household and occupational choices are based on inconsistent policy environment.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:791`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:813`.
- Exact adjustment required:
  - Paper: clarify timing assumptions or remove claim that current `G` enters step-3 optimization.
  - Code: introduce expected-policy mapping for Step 3 or solve households after fiscal realization with fixed-point iteration.
- Suggested validation test: deterministic fiscal regime where expected and realized `G` diverge should trigger predictable behavioral shifts.

### P1-3 Production uses stale labor demand to compute output/profits
- What is wrong: output/profit computed before applying owner's updated labor choice.
- Why it matters economically: profit signal driving entry/exit and wealth is misaligned with chosen firm controls.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:953` then labor update at `:968`.
- Exact adjustment required:
  - Paper: remove strict claim of period-internal firm optimization consistency.
  - Code: apply decision-updated labor demand before `produce_output` and income distribution.
- Suggested validation test: controlled case where owner changes labor demand sharply should alter same-period output accordingly.

### P1-4 Fiscal rule has no policy feedback
- What is wrong: tax adjustment is computed but never applied.
- Why it matters economically: debt sustainability channel is absent; government block mostly observational.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1266`.
- Exact adjustment required:
  - Paper: change wording from active fiscal stabilization to monitored indicator unless fixed.
  - Code: mutate effective tax rule (or dedicated fiscal instrument) when fiscal rule triggers.
- Suggested validation test: with high initial debt and low taxes, debt-to-GDP should eventually stabilize under rule-on configuration.

### P1-5 Income/tax timing misses current-period entrepreneurial profits
- What is wrong: tax base uses pre-production incomes, often excluding contemporaneous firm profits.
- Why it matters economically: redistribution and budget feedback are biased.
- Evidence: incomes set at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:600`; taxes at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/constitution_engine.py:284`.
- Exact adjustment required:
  - Paper: qualify tax timing and incidence assumptions.
  - Code: either tax lagged income explicitly in model equations or update income after production before tax enforcement.
- Suggested validation test: entrepreneur profit shock should map to predictable tax revenue change in same modeled tax period.

### P1-6 Aggregate identities are not closed in recorded state
- What is wrong: recorded `Y`, `I`, and `C` can be internally inconsistent post-updates.
- Why it matters economically: welfare/accounting moments used in calibration can be distorted.
- Evidence: market aggregate updates partial (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:629`); observer uses `market.aggregate_output` (`/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/observer.py:285`); diagnostics show large `Y-(C+I+G)` gaps.
- Exact adjustment required:
  - Paper: remove strict goods-market identity language unless enforced.
  - Code: recompute full aggregate ledger after Step 8 and enforce/report residual explicitly.
- Suggested validation test: assertion on resource residual magnitude each period with tolerance gates.

### P1-7 Bond-vs-capital return architecture is underspecified
- What is wrong: single `interest_rate` in `MarketState` cannot cleanly track both `r^k` and `r^b` paths.
- Why it matters economically: two-asset and fiscal-monetary transmission channels are confounded.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/models/market.py:24`; government bond rate computed separately at `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/lead.py:1250`.
- Exact adjustment required:
  - Paper: avoid claiming separated rates until state interfaces are explicit.
  - Code: extend `MarketState` with `capital_rate` and `bond_rate`; plumb through solvers and budget equations.
- Suggested validation test: scenario with active debt should generate `r^b != r^k` and propagate into household portfolios when two-asset mode is active.

### P2-1 Paper claims overstate "exactness" where code uses heuristics
- What is wrong: multiple "exact" phrases conflict with fallback constants and approximations.
- Why it matters economically: weakens identification and replicability claims.
- Evidence: paper lines around `/Users/antoniomele/Dropbox/github/AgentSociety/docs/model_paper.tex:708`; legacy constants in `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/entrepreneurial_solver.py:610`.
- Exact adjustment required:
  - Paper: relabel these pieces as approximations or legacy modes, with explicit switch conditions.
  - Code: expose approximation flags in output metadata for transparent empirical use.
- Suggested validation test: metadata snapshot test ensuring active approximation modes are reported.

### P2-2 Analytical clearing path uses ad hoc reconciliation
- What is wrong: uses `max(firm_capital, household_wealth)` and `max(firm_output, analytical_output)`.
- Why it matters economically: can inflate/deflate implied factor returns and output.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:465`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/market_clearing.py:528`.
- Exact adjustment required:
  - Paper: remove "exact" analytical equilibrium language.
  - Code: pick one coherent aggregation regime; report discrepancy diagnostics instead of taking maxima.
- Suggested validation test: compare analytical and Walrasian aggregates under controlled firm heterogeneity.

### P2-3 Test suite validates flags more than mechanisms
- What is wrong: key tests pass while mechanisms are inert or degenerate.
- Why it matters economically: false confidence in feature completeness.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/tests/test_two_asset.py:347`; `/Users/antoniomele/Dropbox/github/AgentSociety/tests/test_distribution.py:490`; `/Users/antoniomele/Dropbox/github/AgentSociety/tests/test_political_utility.py:570`.
- Exact adjustment required:
  - Paper: avoid implying full validation coverage.
  - Code/tests: add behavior-differentiation and identity-consistency regression tests.
- Suggested validation test: enforce non-zero behavioral distance metrics between active/inactive feature modes.

### P3-1 Code hygiene issue: duplicate method definition
- What is wrong: `get_value_function` appears twice.
- Why it matters economically: low direct effect; raises maintenance and audit risk.
- Evidence: `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/numerical_solver.py:83`; `/Users/antoniomele/Dropbox/github/AgentSociety/src/emergent_constitution/numerical_solver.py:96`.
- Exact adjustment required:
  - Paper: none.
  - Code: remove duplicate block and keep one canonical docstring.
- Suggested validation test: static lint rule for duplicate definitions.

---

## 7) Immediate Paper Revisions Required
1. Replace all "full/fully integrated" language for nominal, KFE, and two-asset blocks with current implementation status.
2. Explicitly distinguish legacy approximations vs Bellman modes and state which results use which path.
3. Remove claim that pure Bellman politics currently yields operative governance.
4. Qualify government and debt-stabilization claims as partial until fiscal feedback is implemented.
5. Add a transparent "implementation limitations" subsection with feature-gate truth table.

## 8) Immediate Code Priorities (ordered)
1. Fix nominal integration (`_compute_nominal`) and unblock runtime.
2. Repair stationary distribution convergence bug.
3. Implement proposal-level vote identifiers.
4. Wire Bellman political context pipeline (`set_value_function`, `set_observed_gini`, distinct proposed/current constitutions).
5. Make fiscal rule operative in policy state.
6. Implement real two-asset state transitions and solver branch.
7. Wire KFE mode into lifecycle with proper initialization and aggregate handoff.
8. Enforce stock-flow-consistent entrepreneurial accounting.

---

## 9) Bottom Line
As of 2026-02-22, the repository is **not publishable as a full HANK + endogenous institutions platform** under top-journal standards. The paper substantially overstates executable completeness in several core dimensions. The strongest path forward is to narrow claims to what currently runs, then prioritize the `P0` fixes before any quantitative interpretation.
