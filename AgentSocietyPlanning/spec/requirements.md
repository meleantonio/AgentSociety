# Requirements (EARS) — v2: DSGE-HA with LLM Agents

> **Format:** EARS. **IDs:** [REQ-NNN], [PROP-NNN].

---

## 1. Household (Citizen-Agent) Economic Structure

1. **[REQ-001]** [UBIQUITOUS] The system shall maintain N citizen-agents (N >= 20), each with individual state: wealth (assets `a_t`), idiosyncratic productivity (`z_t`), utility parameters (`alpha`, `beta_discount`, `gamma`), a value vector, and an occupational role (worker, entrepreneur, researcher, or unemployed).

2. **[REQ-002]** [UBIQUITOUS] Each citizen-agent shall have an explicit utility function `u(c, l, G)` over consumption (`c`), leisure (`l`), and public goods per capita (`G`), parameterized per agent (Cobb-Douglas or CES form).

3. **[REQ-003]** [STATE-DRIVEN] WHILE the simulation runs, each citizen-agent shall make per-period economic decisions: consumption (`c_t`), savings (`a_{t+1} - a_t`), and labor supply (`1 - l_t`), subject to its budget constraint: `a_{t+1} = (1 + r_t) * a_t + w_t * z_t * (1 - l_t) - c_t - T(y_t) + Tr_t`.

4. **[REQ-004]** [UBIQUITOUS] The system shall enforce a borrowing constraint `a_t >= a_min` and non-negativity constraints `c_t >= 0`, `0 <= l_t <= 1` on every agent every period. No agent's wealth shall fall below `a_min`.

5. **[REQ-005]** [EVENT-DRIVEN] WHEN an agent's economic decisions violate budget or borrowing constraints, the system shall project the decisions onto the feasible set (clamp consumption to affordable level, enforce `a_{t+1} >= a_min`) and log the correction.

## 2. Firms and Production

6. **[REQ-006]** [EVENT-DRIVEN] WHEN a citizen-agent decides to create a firm, the system shall instantiate a Firm entity with: owner ID, capital stock, labor demand, and firm-specific TFP. Firm creation requires a minimum capital investment from the owner's assets.

7. **[REQ-007]** [STATE-DRIVEN] WHILE a firm is active, it shall produce output each period using a Cobb-Douglas production function: `Y_f = A_f * K_f^alpha * L_f^(1-alpha)`, where `A_f` is firm-specific TFP, `K_f` is rented capital, and `L_f` is hired labor.

8. **[REQ-008]** [STATE-DRIVEN] WHILE a firm is active, it shall pay wages `w_t * L_f` to its workers, pay capital rental cost `(r_t + delta) * K_f` to capital owners, and distribute remaining profits `pi_f = Y_f - w_t * L_f - (r_t + delta) * K_f` to the firm owner.

9. **[REQ-009]** [EVENT-DRIVEN] WHEN a citizen-agent allocates resources to R&D, the system shall apply a stochastic productivity improvement to their firm's TFP: with probability `p(R&D_spend)`, `A_f` increases by a drawn factor. The probability function and draw distribution shall be configurable.

10. **[REQ-010]** [EVENT-DRIVEN] WHEN a firm's net worth falls below zero or the owner decides to close it, the system shall liquidate the firm: return remaining capital to the owner, release workers, and remove the firm from the active set.

## 3. Markets and Price Clearing

11. **[REQ-011]** [STATE-DRIVEN] WHILE the simulation runs, the system shall clear the labor market each period by finding a wage `w_t` such that aggregate labor supply equals aggregate labor demand, using an iterative price adjustment (tatonnement) algorithm.

12. **[REQ-012]** [STATE-DRIVEN] WHILE the simulation runs, the system shall clear the capital market each period by finding an interest rate `r_t` such that aggregate savings (total household assets) equals aggregate capital demand (from all active firms), using an iterative price adjustment algorithm.

13. **[REQ-013]** [UBIQUITOUS] The system shall enforce the aggregate resource constraint each period: total output `Y_t` equals aggregate consumption `C_t` plus aggregate investment `I_t` plus government spending `G_t`. Any residual imbalance shall be logged as a market-clearing error.

14. **[REQ-014]** [UBIQUITOUS] The system shall compute and record equilibrium prices (`w_t`, `r_t`) each period before agent decisions that depend on them.

## 4. Stochastic Shocks

15. **[REQ-015]** [STATE-DRIVEN] WHILE the simulation runs, each agent's idiosyncratic productivity `z_{i,t}` shall evolve according to a discrete Markov chain (Rouwenhorst discretization of `log(z') = rho_z * log(z) + epsilon`, `epsilon ~ N(0, sigma_z^2)`), with configurable persistence `rho_z` and volatility `sigma_z`.

16. **[REQ-016]** [STATE-DRIVEN] WHILE the simulation runs, aggregate TFP `A_t` shall evolve according to `log(A_t) = rho_A * log(A_{t-1}) + epsilon_t^A`, `epsilon^A ~ N(0, sigma_A^2)`, with configurable persistence and volatility.

17. **[REQ-017]** [OPTIONAL] WHERE preference shocks are enabled, each agent's discount factor `beta_i` shall receive a small stochastic perturbation each period, drawn from a configurable distribution.

## 5. Constitution and Governance

18. **[REQ-018]** [UBIQUITOUS] The system shall maintain a constitution as a structured, extensible document: a collection of named policy rules, each with a type, parameters, and an enforcement function. The constitution shall not be limited to a fixed set of rule types.

19. **[REQ-019]** [EVENT-DRIVEN] WHEN a constitutional proposal period occurs (every K ticks, configurable), the system shall collect proposals from citizen-agents via LLM reasoning. Each proposal shall specify: a rule name, rule type, parameters, a natural-language rationale, and an executable enforcement specification.

20. **[REQ-020]** [EVENT-DRIVEN] WHEN proposals have been collected, the system shall conduct a vote among all citizen-agents using the currently active voting rule (default: majority). The voting rule itself is part of the constitution and may be changed by vote.

21. **[REQ-021]** [EVENT-DRIVEN] WHEN a vote passes, the system shall update the constitution by adding, modifying, or removing the specified rule, and apply it starting from the next period.

22. **[REQ-022]** [UBIQUITOUS] The system shall enforce all active constitutional rules each period. Enforcement includes: computing taxes `T(y_i)` per the active tax schedule, computing transfers `Tr_i` per the active transfer program, providing public goods `G_t` per the active provision rule, and applying any active market regulations.

23. **[REQ-023]** [UBIQUITOUS] The system shall validate every constitutional rule for internal consistency (e.g., tax rates produce non-negative revenue, transfer programs satisfy a budget constraint, regulations do not create logical contradictions) before activation. Invalid rules shall be rejected with a logged reason.

## 6. LLM Agent Decision-Making

24. **[REQ-024]** [UBIQUITOUS] The system shall use LLM-based reasoning as the default decision-making mechanism for all citizen-agent decisions: economic (consumption, savings, labor supply), entrepreneurial (firm creation, hiring, investment, R&D), political (proposals, votes), and social (coalition formation, communication).

25. **[REQ-025]** [UBIQUITOUS] Each LLM decision call shall provide the agent with structured context: (a) personal state (wealth, productivity, utility params, role, firm ownership), (b) economic conditions (prices, aggregate output, employment, distribution statistics), (c) current constitution, (d) recent history summary, (e) messages from other agents if applicable.

26. **[REQ-026]** [UBIQUITOUS] Every LLM decision shall be returned as structured data (JSON) conforming to a per-decision-type schema. The system shall parse and validate the response against the schema before applying it.

27. **[REQ-027]** [EVENT-DRIVEN] WHEN an LLM call fails (timeout, parse error, invalid response), the system shall fall back to a numerical solver that computes the approximately optimal decision given the agent's state, using the agent's explicit utility function and budget constraint.

28. **[REQ-028]** [UBIQUITOUS] The system shall compute and record each agent's realized utility `u(c_t, l_t, G_t)` each period, providing a ground-truth welfare measure independent of how the decision was made (LLM or fallback).

29. **[REQ-029]** [UBIQUITOUS] The system shall support batching of LLM calls (multiple agents per API request where the LLM provider supports it) and caching of responses for identical state-context pairs, to manage cost and latency.

## 7. Observer and Statistics

30. **[REQ-030]** [STATE-DRIVEN] WHILE the simulation runs, the Observer shall compute at least every K periods (configurable): Gini coefficient, Pareto efficiency score, total output `Y_t`, aggregate consumption `C_t`, aggregate investment `I_t`, mean and median wealth, the full wealth distribution (histogram/quantiles), unemployment rate, number of active firms, mean firm size, aggregate R&D spending, and the current constitution snapshot.

31. **[REQ-031]** [UBIQUITOUS] The Observer shall compute social welfare as the (weighted or unweighted) sum of all agents' realized per-period utilities, and cumulative discounted welfare over the simulation horizon.

32. **[REQ-032]** [EVENT-DRIVEN] WHEN the simulation ends, the system shall output: (a) the final constitution, (b) the full history log (statistics per observation period), (c) final agent states, (d) final firm states, (e) a welfare summary comparing LLM-agent welfare to the numerical-solver benchmark (if computed).

## 8. Simulation Orchestration

33. **[REQ-033]** [STATE-DRIVEN] WHILE the simulation runs, the Lead (Governor) shall advance time in discrete periods. Each period shall execute in order: (1) draw shocks, (2) compute/clear markets (find equilibrium prices), (3) collect LLM agent decisions, (4) validate and enforce constraints, (5) execute production and distribution, (6) enforce constitutional rules (taxes, transfers, public goods), (7) process proposals and votes (on proposal-interval periods), (8) update agent and firm states, (9) observe and record statistics.

34. **[REQ-034]** [UBIQUITOUS] The system shall accept a simulation configuration specifying at minimum: number of agents, maximum periods, RNG seed, shock parameters (rho_z, sigma_z, rho_A, sigma_A), production parameters (alpha, delta), borrowing limit (a_min), proposal interval, observer interval, LLM provider configuration, and initial distribution parameters.

35. **[REQ-035]** [UBIQUITOUS] The system shall provide a CLI entry point with flags for all key configuration parameters, supporting both interactive (with LLM) and benchmark (numerical-only) execution modes.

## 9. Numerical Benchmark Solver

36. **[REQ-036]** [UBIQUITOUS] The system shall include a numerical benchmark solver that computes approximately optimal household decisions (consumption, savings, labor supply) given current prices and the agent's state, using value function iteration or the endogenous grid method (EGM) on the agent's Bellman equation.

37. **[REQ-037]** [OPTIONAL] WHERE benchmark mode is enabled, the system shall run all agents using the numerical solver instead of LLM, producing a baseline simulation for comparison against LLM-driven runs.

---

## Properties (Invariants)

1. **[PROP-001]** **Determinism:** For a fixed seed, initial configuration, and LLM provider responses (or deterministic LLM settings: temperature=0), the simulation outcome shall be fully reproducible.

2. **[PROP-002]** **Budget consistency:** At every period, every agent's realized state shall satisfy its budget constraint. Wealth, consumption, and labor supply shall be consistent with the accounting identity: `a_{t+1} = (1 + r_t) * a_t + income_t - c_t - taxes_t + transfers_t`.

3. **[PROP-003]** **Market clearing:** At every period, the aggregate resource constraint `Y_t = C_t + I_t + G_t` shall hold to within a configurable numerical tolerance (default: 1e-6). Excess demand in labor and capital markets shall be below the same tolerance after price adjustment.

4. **[PROP-004]** **Non-negativity:** No agent shall have negative consumption. No agent's wealth shall fall below `a_min`. No firm shall operate with negative capital or labor inputs.

5. **[PROP-005]** **Welfare measurability:** Every agent's realized utility shall be computable from observed quantities (consumption, leisure, public goods) and the agent's utility parameters, without requiring knowledge of the agent's decision process.

6. **[PROP-006]** **Constitutional validity:** Every active rule in the constitution shall have passed the validation check (REQ-023). No invalid or internally inconsistent rule shall be enforceable.
