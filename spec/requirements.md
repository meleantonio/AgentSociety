# Requirements (EARS) — HANK Upgrade

> **Format:** EARS. **IDs:** [REQ-1xx] for Phase 1, [REQ-2xx] for Phase 2, [REQ-3xx] for Phase 3, [REQ-4xx] for Phase 4.
> **Supersedes:** Requirements in `AgentSocietyPlanning/spec/requirements.md` where conflicting.

---

## Phase 1: Fix Economic Foundations

### 1.1 Proper Walrasian Market Clearing

1. **[REQ-101]** [UBIQUITOUS] The system shall compute equilibrium prices `(w_t, r_t)` by solving for prices that equate aggregate firm-level factor demands (from individual firm FOCs) with aggregate household factor supplies. The system shall NOT use a representative-firm production function when heterogeneous firms are active.

2. **[REQ-102]** [STATE-DRIVEN] WHILE heterogeneous firms are active, the system shall compute aggregate labor demand as `L^d(w) = sum_f L_f*(w)` where each `L_f*(w) = ((1-alpha) * A_f * K_f^alpha / w)^(1/alpha)` from the individual firm's FOC. The equilibrium wage `w*` shall satisfy `L^d(w*) = L^s`.

3. **[REQ-103]** [STATE-DRIVEN] WHILE heterogeneous firms are active, the system shall compute aggregate output as `Y = sum_f A_f * K_f^alpha * L_f*(w*)^(1-alpha)`, the true sum of heterogeneous firm outputs — not a representative-firm formula.

4. **[REQ-104]** [UBIQUITOUS] The capital market shall clear by equating total firm capital demand `K^d = sum_f K_f` with the capital supplied by households. The interest rate `r` shall be determined by the marginal product of aggregate capital in the properly-aggregated production: `r = dY/dK - delta`, computed numerically from the firm-level outputs.

5. **[REQ-105]** [UBIQUITOUS] When no heterogeneous firms are active (all agents are workers), the system shall fall back to a representative-firm clearing using aggregate household wealth as capital and aggregate labor supply as labor. This preserves backward compatibility with the pre-entrepreneurial model.

6. **[REQ-106]** [UBIQUITOUS] The market-clearing algorithm shall use a bisection method on the wage `w` to find the root of `L^d(w) - L^s = 0`, with tolerance `1e-8` and maximum 100 iterations. The interest rate shall be derived from the zero-profit condition for the marginal firm.

### 1.2 Fix Entrepreneur Budget Constraint

7. **[REQ-107]** [UBIQUITOUS] An entrepreneur's budget constraint shall be `a_{t+1} = (1+r_t)*a_t + pi_f - c_t - T(pi_f) + Tr_t`, where `pi_f` is firm profit (the residual claim). Entrepreneurs shall NOT receive separate labor income `w*z*(1-l)`. Their labor is embedded in the firm's production function.

8. **[REQ-108]** [UBIQUITOUS] An entrepreneur's effective labor supply to the market shall be zero. Their time allocation affects firm management (productivity multiplier) but does not earn a market wage. The leisure choice `l_E` enters utility but not the labor market.

9. **[REQ-109]** [STATE-DRIVEN] WHILE a household is an entrepreneur, the system shall compute their income as firm profit only: `y_i = pi_f(i)`. Tax obligations shall apply to profit, not to imputed labor income.

### 1.3 Fix Occupational Choice

10. **[REQ-110]** [UBIQUITOUS] Worker lifetime value `V^W(a, z)` shall be computed from the VFI-solved value function at the household's actual state `(a_it, z_it)`, not from a perpetuity formula with hardcoded leisure and consumption fractions.

11. **[REQ-111]** [UBIQUITOUS] Firm value `V^F(e, K)` shall be computed by solving the firm's Bellman equation: `V^F(e, K) = pi(e, K) + beta_f * sum_{e'} Pi(e, e') * V^F(e', K)`, accounting for the full Markov transition of ability — not a perpetuity at the stationary mean.

12. **[REQ-112]** [UBIQUITOUS] Entrepreneur lifetime value `V^E(a, z, e)` shall solve a Bellman equation that includes: (a) firm profit as income, (b) the option value of exiting to become a worker, (c) the continuation value with stochastic ability transitions. No hardcoded leisure or consumption values.

13. **[REQ-113]** [UBIQUITOUS] Occupational choice shall compare `V^W(a, z)` vs `V^E(a, z, e)` where both are properly-solved Bellman value functions. Entry requires `V^E > V^W` AND `a >= K_min + F`. Exit requires `V^W > V^E` (without entry cost).

14. **[REQ-114]** [UBIQUITOUS] Optimal firm capital `K*` shall be chosen to maximize the firm Bellman value `V^F(e, K)` subject to `K <= a - F` (for entrants) or `K <= a + K_current` (for continuing entrepreneurs), using a golden-section search with 50+ evaluations.

### 1.4 Endogenous Grid Method (EGM)

15. **[REQ-115]** [UBIQUITOUS] The household consumption-savings problem shall be solved using the Endogenous Grid Method (Carroll 2006). For each asset grid point `a'` on the savings grid and each productivity state `z`, the EGM shall invert the Euler equation to find optimal consumption `c*(a', z)` analytically.

16. **[REQ-116]** [UBIQUITOUS] The Euler equation for Cobb-Douglas utility `u = c^alpha * l^beta * G^gamma` shall be: `alpha * c^(alpha-1) * l^beta * G^gamma = beta_d * (1+r') * E[alpha * c'^(alpha-1) * l'^beta * G'^gamma | z]`. The EGM inverts this to get `c` as a function of the RHS.

17. **[REQ-117]** [UBIQUITOUS] The leisure choice shall be solved jointly with consumption using the intratemporal FOC: `beta_u/l = alpha_u * w * z / c`, giving `l* = (beta_u * c) / (alpha_u * w * z)`, clamped to `[0, 1]`. This replaces the discrete leisure grid search with an analytical solution conditional on consumption.

18. **[REQ-118]** [UBIQUITOUS] The EGM asset grid shall use at least 200 exponentially-spaced points on `[a_min, a_max]`. The endogenous consumption grid shall have machine-precision accuracy at each grid point.

19. **[REQ-119]** [UBIQUITOUS] VFI convergence shall be measured by the Euler equation residual `max_i |1 - c_euler(a_i, z_j) / c_policy(a_i, z_j)|` in addition to the sup-norm on the value function. The Euler equation residual shall be below `1e-6` at convergence.

20. **[REQ-120]** [UBIQUITOUS] The existing brute-force VFI solver shall be retained as a fallback (configurable). The EGM solver shall be the default. Both shall produce compatible policy function interfaces.

---

## Phase 2: Scale and Calibrate

### 2.1 Distribution Tracking (KFE / Histogram)

21. **[REQ-201]** [UBIQUITOUS] The system shall track the cross-sectional wealth distribution using a histogram on the asset grid (probability mass at each grid point), not individual agent states. The distribution shall evolve via the Kolmogorov Forward Equation (KFE): `mu_{t+1}(a', z') = sum_{a,z} mu_t(a, z) * Pi(z, z') * 1{g(a,z) = a'}`, where `g(a,z)` is the savings policy function.

22. **[REQ-202]** [UBIQUITOUS] The KFE implementation shall use the Young (2010) lottery method: for each `(a, z)` cell, the policy `a' = g(a, z)` is allocated probabilistically to the two nearest grid points. This avoids the need for `a'` to land exactly on a grid point.

23. **[REQ-203]** [UBIQUITOUS] Aggregate quantities (mean wealth, consumption, labor supply, Gini) shall be computed from the distribution `mu_t` and the policy functions, not from individual agent states: `C_t = sum_{a,z} mu_t(a,z) * c(a,z)`, etc.

24. **[REQ-204]** [OPTIONAL] WHERE individual-agent mode is enabled (for LLM-driven governance), the system shall support 5000+ individual agents sampled from the KFE distribution. Each agent's state `(a_i, z_i)` shall be drawn from `mu_t` at initialization.

25. **[REQ-205]** [UBIQUITOUS] The stationary distribution `mu*` shall be computed at initialization by iterating the KFE forward until `||mu_{n+1} - mu_n||_1 < 1e-10`, with a maximum of 10,000 iterations.

### 2.2 Rouwenhorst Grid Expansion

26. **[REQ-206]** [UBIQUITOUS] The default Rouwenhorst grid for idiosyncratic productivity shall use 7 grid points (previously 5). The configuration shall allow up to 15 grid points. With 7 points, the tails of the earnings distribution are better captured.

27. **[REQ-207]** [UBIQUITOUS] The entrepreneurial ability Rouwenhorst grid shall also default to 7 grid points (previously 5).

### 2.3 Moment-Matching Calibration

28. **[REQ-208]** [UBIQUITOUS] The system shall define calibration targets as a configuration object: wealth Gini (target: 0.80), entrepreneur share of population (target: 0.10), top 10% wealth share (target: 0.70), median MPC (target: 0.25, for Phase 3), liquid/illiquid wealth ratio (target: 0.25, for Phase 3).

29. **[REQ-209]** [OPTIONAL] WHERE auto-calibration mode is enabled, the system shall run a Simulated Method of Moments (SMM) loop that adjusts key parameters (`beta_d`, `sigma_z`, `entry_cost`, `a_min`) to minimize the distance between model moments and calibration targets.

30. **[REQ-210]** [UBIQUITOUS] The system shall report model-implied moments alongside calibration targets at each simulation run, in the observer output and the final report.

---

## Phase 3: HANK Features

### 3.1 Two-Asset Structure (Liquid/Illiquid)

31. **[REQ-301]** [UBIQUITOUS] Each household shall hold two assets: liquid assets `b_t` (government bonds) and illiquid assets `k_t` (physical capital / housing). Total wealth is `a_t = b_t + k_t`. The liquid asset earns return `r^b_t` (bond rate); the illiquid asset earns return `r^k_t` (capital return).

32. **[REQ-302]** [UBIQUITOUS] Adjusting illiquid assets shall incur a transaction cost `chi(d)` where `d = k_{t+1} - k_t` is the deposit/withdrawal. The cost function shall be convex: `chi(d) = chi_0 * |d| + chi_1 * d^2 / k_t`, with `chi(0) = 0`. This creates wealthy hand-to-mouth agents who hold illiquid wealth but do not adjust it frequently.

33. **[REQ-303]** [UBIQUITOUS] The household Bellman equation shall be two-dimensional in `(b, k)` for a given `z`: `V(b, k, z) = max_{c, l, d} { u(c, l, G) + beta * E[V(b', k', z') | z] }` subject to `b' = (1+r^b)*b + w*z*(1-l) - c - T(y) + Tr - d - chi(d)` and `k' = (1+r^k)*k + d` and `b' >= b_min`.

34. **[REQ-304]** [UBIQUITOUS] The borrowing constraint shall apply to liquid assets only: `b_t >= b_min` (with `b_min <= 0` allowed, representing unsecured credit). Illiquid assets shall satisfy `k_t >= 0`.

35. **[REQ-305]** [UBIQUITOUS] The EGM solver shall be extended to two assets. For a given illiquid deposit choice `d`, the liquid savings problem reduces to a one-dimensional EGM. The outer loop over `d` uses a coarse grid (20-50 points) with golden-section refinement.

### 3.2 Government Debt and Bond Market

36. **[REQ-306]** [UBIQUITOUS] The government shall maintain a stock of outstanding debt `B_t` (government bonds). The government budget constraint is: `B_{t+1} = (1+r^b_t)*B_t + G_t + Tr_t - T_t`, where `G_t` is public goods spending, `Tr_t` is total transfers, and `T_t` is total tax revenue.

37. **[REQ-307]** [UBIQUITOUS] The bond market shall clear: `sum_i b_i = B_t` (total household liquid asset holdings equal government debt). The bond rate `r^b_t` shall be determined to clear this market.

38. **[REQ-308]** [UBIQUITOUS] The capital market shall clear separately: `sum_i k_i = K_t` where `K_t = sum_f K_f` is total firm capital demand. The capital return `r^k_t` shall be the marginal product of capital minus depreciation.

39. **[REQ-309]** [UBIQUITOUS] The government shall maintain a fiscal rule to ensure debt sustainability: if `B_t / Y_t > B_max` (configurable, default 1.5), the tax rate shall automatically increase by `0.01` per period until the debt-to-GDP ratio declines. This prevents explosive debt paths.

### 3.3 Nominal Rigidities

40. **[REQ-310]** [UBIQUITOUS] The model shall include a final goods sector with sticky prices. Intermediate goods firms set prices subject to Rotemberg quadratic adjustment costs: `Phi(pi_t) = (phi_p / 2) * (pi_t - pi_bar)^2 * Y_t`, where `pi_t` is the inflation rate, `pi_bar` is the target inflation rate, and `phi_p` is the adjustment cost parameter.

41. **[REQ-311]** [UBIQUITOUS] The New Keynesian Phillips Curve (NKPC) shall hold in equilibrium: `phi_p * pi_t * (pi_t - pi_bar) = (1 - epsilon) + epsilon * mc_t + beta * phi_p * E[pi_{t+1} * (pi_{t+1} - pi_bar) * Y_{t+1}/Y_t]`, where `mc_t` is the real marginal cost and `epsilon` is the elasticity of substitution between varieties.

42. **[REQ-312]** [UBIQUITOUS] The central bank shall set the nominal interest rate `i_t` according to a Taylor rule: `i_t = r_bar + phi_pi * (pi_t - pi_bar) + phi_y * (Y_t - Y_bar) / Y_bar`, where `r_bar` is the natural rate, `phi_pi > 1` (Taylor principle), `phi_y >= 0`, `pi_bar` is the inflation target, and `Y_bar` is potential output.

43. **[REQ-313]** [UBIQUITOUS] The Fisher equation shall link nominal and real rates: `1 + r^b_t = (1 + i_t) / (1 + E[pi_{t+1}])`. Households make savings decisions based on the real bond rate `r^b_t`.

44. **[REQ-314]** [UBIQUITOUS] The system shall track the price level `P_t`, inflation rate `pi_t = P_t/P_{t-1} - 1`, nominal wage `W_t = w_t * P_t`, and nominal interest rate `i_t` as state variables.

45. **[REQ-315]** [OPTIONAL] WHERE wage rigidity is enabled, the system shall apply Rotemberg wage adjustment costs analogously to price adjustment costs, with a separate parameter `phi_w`.

---

## Phase 4: Microfound Political Decisions

### 4.1 Unified Political-Economic Utility

46. **[REQ-401]** [UBIQUITOUS] Each agent's lifetime objective shall include a political utility component: `max E_0 sum_t beta^t [ u(c_t, l_t, G_t) + lambda * v(C_t; theta_i) ]`, where `v(C_t; theta_i)` is a political utility function over the constitution `C_t` parameterized by the agent's ideological type `theta_i`, and `lambda` is the weight on political utility (configurable, default 0.05).

47. **[REQ-402]** [UBIQUITOUS] The political utility function shall be: `v(C_t; theta_i) = theta_i^eq * f_eq(C_t) + theta_i^lib * f_lib(C_t)`, where `f_eq(C_t) = -Gini(C_t)` measures the equality payoff (lower inequality = higher utility) and `f_lib(C_t) = -tau(C_t)` measures the liberty payoff (lower tax rate = higher utility). The functions `f_eq` and `f_lib` shall be configurable.

48. **[REQ-403]** [UBIQUITOUS] Political preferences for proposals and voting shall derive from the agent's value function: a proposal is supported if `E[V(s'; C_proposed)] > E[V(s'; C_current)]`, where `V` is the Bellman value function that includes both economic and political utility.

49. **[REQ-404]** [UBIQUITOUS] The system shall provide the Bellman-derived political preferences to the LLM as structured context, alongside the ideological value vector. The LLM may override the Bellman recommendation but the deviation shall be logged and measured.

50. **[REQ-405]** [OPTIONAL] WHERE pure-Bellman politics mode is enabled, all political decisions (proposals, votes) shall be made by the Bellman value function comparison, bypassing the LLM entirely. This provides a game-theoretic baseline for political dynamics.

---

## Properties (Invariants)

### Preserved from v2

1. **[PROP-001]** **Determinism:** For a fixed seed, configuration, and LLM responses (or temperature=0), the simulation shall be fully reproducible.

2. **[PROP-002]** **Budget consistency:** At every period, every agent's realized state shall satisfy its budget constraint (single-asset or two-asset as applicable).

3. **[PROP-003]** **Market clearing:** Excess demand in all markets (labor, capital, bonds) shall be below tolerance `1e-8` after equilibrium computation.

4. **[PROP-004]** **Non-negativity:** No agent shall have negative consumption. Liquid wealth shall satisfy `b >= b_min`. Illiquid wealth shall satisfy `k >= 0`. No firm shall have negative capital or labor.

5. **[PROP-005]** **Welfare measurability:** Realized utility is computable from observed `(c, l, G)` and agent utility parameters.

6. **[PROP-006]** **Constitutional validity:** All active rules pass validation before enforcement.

### New Properties

7. **[PROP-007]** **Euler equation accuracy:** The maximum Euler equation residual across the asset grid shall be below `1e-6` for the EGM solver. Policy functions shall be monotone in wealth.

8. **[PROP-008]** **Distribution conservation:** The KFE distribution `mu_t` shall satisfy `sum_{a,z} mu_t(a,z) = 1` at every period, to within tolerance `1e-12`.

9. **[PROP-009]** **Fisher consistency:** The nominal rate, real rate, and expected inflation shall satisfy the Fisher equation `1 + r^b = (1 + i) / (1 + E[pi'])` at every period.

10. **[PROP-010]** **Government budget balance:** The government debt `B_{t+1}` shall exactly satisfy the government budget constraint at every period.

11. **[PROP-011]** **Walras' law:** With N markets, only N-1 need explicit clearing. The goods market identity `Y = C + I + G + chi` (where `chi` is aggregate adjustment costs) shall hold by Walras' law, verified to tolerance `1e-6`.

12. **[PROP-012]** **Backward compatibility:** All existing v2 tests (977 tests) shall pass after each phase. New functionality is additive; existing interfaces are preserved or have compatible replacements.
