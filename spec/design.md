# Technical Design — HANK Upgrade

> **Traceability:** Each section links to requirements `[REQ-xxx]`.
> **Convention:** New files are prefixed with module purpose. Existing files are modified in-place.

---

## 1. Phase 1: Fix Economic Foundations

### 1.1 Proper Walrasian Market Clearing

**File:** `src/emergent_constitution/market_clearing.py` (major rewrite)

*Traceability: REQ-101, REQ-102, REQ-103, REQ-104, REQ-105, REQ-106*

#### Current Problem

```python
# WRONG: representative firm with heterogeneous firms
y = aggregate_tfp * (total_capital_supply**config.alpha) * (total_labor_supply**(1 - config.alpha))
w = (1 - config.alpha) * y / total_labor_supply
r = config.alpha * y / total_capital_supply - config.delta
```

This ignores firm heterogeneity. With different `A_f` and `K_f`, the wage that clears the labor market is NOT the representative-firm FOC.

#### New Design: Bisection on Excess Labor Demand

**Algorithm:**

1. Compute aggregate labor supply: `L^s = sum_{workers} z_i * (1 - l_i)`
2. Define excess labor demand as a function of wage `w`:
   ```
   ELD(w) = sum_f L_f*(w) - L^s
   where L_f*(w) = ((1-alpha) * A_f * K_f^alpha / w)^(1/alpha)
   ```
3. `ELD(w)` is strictly decreasing in `w` (higher wage → lower demand). Find `w*` such that `ELD(w*) = 0` via bisection:
   - Lower bound: `w_lo = 1e-6` (at very low wage, demand exceeds supply)
   - Upper bound: `w_hi` such that `ELD(w_hi) < 0` (start at 2× current wage, double until negative)
   - Bisect until `|ELD(w)| < 1e-8` or 100 iterations
4. Compute aggregate output: `Y = sum_f A_f * K_f^alpha * L_f*(w*)^(1-alpha)`
5. Compute interest rate from capital market: `r = alpha * Y / K^s - delta` where `K^s = sum_f K_f`

**Fallback:** When no firms exist, use the existing representative-firm formula (REQ-105).

```python
def clear_markets_walrasian(
    households: list[HouseholdState],
    firms: list[FirmState],
    aggregate_tfp: float,
    config: SimulationConfigV2,
) -> MarketState:
    """Walrasian equilibrium with heterogeneous firms."""
    if not firms:
        return _representative_firm_clearing(households, aggregate_tfp, config)

    labor_supply = _compute_labor_supply(households)

    # Bisection on wage
    w_star = _bisect_wage(firms, labor_supply, config.alpha)

    # Compute firm-level outputs at equilibrium wage
    Y, firm_labors = _compute_firm_outputs(firms, w_star, config.alpha)

    # Capital market: r from aggregate MPK
    K_total = sum(f.capital for f in firms)
    r = config.alpha * Y / K_total - config.delta

    return MarketState(wage=w_star, interest_rate=r, ...)
```

**Interface change:** `clear_markets()` signature remains the same. Internal dispatch based on `len(firms) > 0`.

#### Data Model Changes

None — `MarketState` already has all needed fields.

---

### 1.2 Fix Entrepreneur Budget Constraint

**Files:** `src/emergent_constitution/lead.py`, `src/emergent_constitution/models/household.py`, `src/emergent_constitution/economics.py`

*Traceability: REQ-107, REQ-108, REQ-109*

#### Current Problem

In `lead.py` Step 8 (update states), entrepreneurs receive:
```
a' = (1+r)*a + w*z*(1-l) + pi_f - c - T(y) + Tr
```
The `w*z*(1-l)` term is labor income that entrepreneurs should NOT receive.

#### Fix

**Entrepreneur budget constraint:**
```
a' = (1+r)*a + pi_f - c - T(pi_f) + Tr
```

**Implementation:**
- In `lead.py` `_update_household_states()`: check `household.role == OccupationalRole.ENTREPRENEUR`. If so, set labor income to 0 and use firm profit as income.
- In `market_clearing.py` `_compute_labor_supply()`: exclude entrepreneurs from labor supply aggregation.
- Entrepreneur's `labor_supply` field is set to 0 in the household state during Step 3.
- Tax is on firm profit, not imputed labor income.

**Backward compat:** The `HouseholdState` model doesn't change. The behavioral change is in how income is computed for entrepreneurs.

---

### 1.3 Fix Occupational Choice — Proper Bellman Values

**File:** `src/emergent_constitution/entrepreneurial_solver.py` (major rewrite)

*Traceability: REQ-110, REQ-111, REQ-112, REQ-113, REQ-114*

#### Current Problem

```python
# WRONG: hardcoded approximations
leisure = 0.35  # hardcoded
consumption = max(available * 0.7, EPSILON)  # hardcoded
V_worker = u(c, l, G) / (1 - beta)  # perpetuity, no dynamics

# WRONG: perpetuity at stationary mean
V_firm = pi_0 + beta * pi_mean / (1 - beta)  # ignores Markov transitions
```

#### New Design: Coupled Bellman Equations

**Firm Value Bellman (solve once per period):**

```
V^F(e_j, K) = pi(e_j, K, w, r) + beta_f * sum_{j'} Pi_e(j, j') * V^F(e_{j'}, K)
```

This is a linear system for each K: `V^F = pi + beta_f * Pi_e * V^F`, or `(I - beta_f * Pi_e) * V^F = pi`. Solve via direct linear algebra: `V^F = (I - beta_f * Pi_e)^{-1} * pi`.

Do this for a grid of K values (50 points in `[K_min, K_max]`).

**Worker Value:** Read directly from the VFI/EGM-solved value function `V^W(a, z)` at the agent's actual state.

**Entrepreneur Value:** For an entrepreneur with state `(a, z, e)` and firm capital K:
```
V^E(a, z, e, K) = u(c_E, l_E, G) + beta_d * E[max(V^W(a', z'), V^E(a', z', e', K'))]
```
where:
- `c_E = max(pi(e, K) + r*a_remaining - savings, 0)` from the entrepreneur's budget constraint
- `l_E` is the entrepreneur's optimal leisure (from FOC, not hardcoded)
- The expectation is over `(z', e')` transitions

**Practical approximation:** Since solving the full two-dimensional Bellman `V^E(a, z, e, K)` is expensive, we use a **one-step lookahead**:
1. Compute `V^F(e, K)` for all `(e, K)` on the grid (linear system, fast)
2. Compute optimal `K*` for each `e` by maximizing `V^F(e, K)` over K grid
3. Compute entrepreneur consumption and leisure from budget constraint + FOC
4. `V^E = u(c_E, l_E, G) + beta_d * E[max(V^W(a', z'), V^F_perpetuity)]`

This is still an approximation but much better than the current one because:
- Worker value comes from VFI (exact up to grid)
- Firm value solves the correct Markov Bellman (not perpetuity at mean)
- Entrepreneur consumption/leisure use FOC, not hardcoded values

**Optimal capital search:** Golden-section search over 50+ points in `[K_min, a - F]` (REQ-114), replacing the 20-point grid search.

```python
class EntrepreneurialSolver:
    def __init__(self, config, ability_grid, ability_trans):
        ...
        # Pre-compute (I - beta * Pi)^{-1} for firm value Bellman
        self._firm_value_matrix = np.linalg.inv(
            np.eye(len(ability_grid)) - self._beta * np.array(ability_trans)
        )

    def compute_firm_value_bellman(self, capital, wage, interest_rate):
        """Solve V^F(e, K) = (I - beta*Pi)^{-1} * pi(e, K) for all e."""
        profits = np.array([
            self._compute_profit(e, capital, wage, interest_rate)
            for e in self._ability_grid
        ])
        return self._firm_value_matrix @ profits  # shape: (n_e,)
```

---

### 1.4 Endogenous Grid Method (EGM)

**File:** `src/emergent_constitution/egm_solver.py` (NEW file)

*Traceability: REQ-115, REQ-116, REQ-117, REQ-118, REQ-119, REQ-120*

#### Algorithm: EGM with Cobb-Douglas Utility

**Setup:**
- Exogenous savings grid: `a'_grid` with 200 exponentially-spaced points on `[a_min, a_max]`
- Productivity grid: `z_grid` with n_z states and transition matrix `Pi`
- For each `(a', z)`, compute optimal `(c, l)` by inverting the Euler equation

**Step 1: Expected marginal utility of savings**

For each `(a'_j, z_s)`:
```
RHS(a'_j, z_s) = beta * (1 + r) * sum_{s'} Pi(s, s') * u_c(c*(a'_j, z_{s'}), l*(a'_j, z_{s'}), G)
```
where `u_c = alpha * c^(alpha-1) * l^beta * G^gamma` is the marginal utility of consumption.

On the first iteration, initialize with the steady-state policy. On subsequent iterations, use the previous iteration's policy.

**Step 2: Invert the Euler equation**

From the Euler equation `u_c(c, l, G) = RHS`:
```
alpha * c^(alpha-1) * l^beta * G^gamma = RHS
```

Using the intratemporal FOC (REQ-117):
```
l = (beta_u * c) / (alpha_u * w * z)    [clamped to [0, 1]]
```

Substitute into the Euler equation:
```
alpha * c^(alpha-1) * ((beta_u * c) / (alpha_u * w * z))^beta_u * G^gamma = RHS
```

This gives `c` as a function of `RHS`, solvable by Newton's method or bisection on `c`.

**Step 3: Endogenous grid**

Given `(c, l)` at savings point `a'_j`, back out current assets:
```
a_j = (a'_j + c + T(w*z*(1-l)) - Tr - w*z*(1-l)) / (1 + r)
```

**Step 4: Interpolate back to exogenous grid**

The EGM produces `(a_j, c_j)` pairs on an endogenous grid. Interpolate `c(a)` back onto the exogenous asset grid using monotone piecewise cubic interpolation.

**Step 5: Handle the constraint region**

For `a` below the lowest endogenous grid point, the agent is constrained: `a' = a_min`, and consumption is `c = (1+r)*a + w*z*(1-l) - T + Tr - a_min`.

**Convergence:** Iterate Steps 1-5 until the Euler equation residual (REQ-119) is below `1e-6`.

```python
class EGMSolver:
    """Endogenous Grid Method for Cobb-Douglas utility."""

    def __init__(self, config, productivity_grid, transition_matrix):
        self.n_a = 200  # savings grid points
        self.a_grid = self._exponential_grid(config.a_min, 1000.0, self.n_a)
        ...

    def solve(self, wage, interest_rate, public_goods, tax_fn, transfer):
        """Solve for policy functions c(a, z) and l(a, z) via EGM."""
        ...
        return PolicyFunctions(consumption=c_policy, leisure=l_policy,
                              savings=a_prime_policy)

    def euler_residual(self, policy):
        """Compute max Euler equation residual for convergence check."""
        ...
```

**Interface compatibility:** `EGMSolver.solve_all()` returns `dict[str, EconomicDecision]`, same as `NumericalSolver.solve_all()`. The `NumericalSolver` class is retained as fallback per REQ-120.

---

## 2. Phase 2: Scale and Calibrate

### 2.1 KFE Distribution Tracking

**File:** `src/emergent_constitution/distribution.py` (NEW file)

*Traceability: REQ-201, REQ-202, REQ-203, REQ-204, REQ-205*

#### Design: Histogram Distribution on Asset Grid

```python
class Distribution:
    """Cross-sectional wealth-productivity distribution on a grid."""

    def __init__(self, a_grid, z_grid):
        self.a_grid = a_grid  # shape (n_a,)
        self.z_grid = z_grid  # shape (n_z,)
        self.mu = np.zeros((len(a_grid), len(z_grid)))  # probability mass

    def forward(self, policy_savings, transition_matrix):
        """Advance distribution one period via KFE (Young 2010 lottery).

        For each (a_i, z_j) cell with mass mu(a_i, z_j):
          1. Policy gives a' = g(a_i, z_j)
          2. Find bracket: a_grid[lo] <= a' <= a_grid[hi]
          3. Weight: w = (a' - a_lo) / (a_hi - a_lo)
          4. Allocate mass: mu'(a_hi, z') += w * Pi(z_j, z') * mu(a_i, z_j)
                           mu'(a_lo, z') += (1-w) * Pi(z_j, z') * mu(a_i, z_j)
        """
        mu_new = np.zeros_like(self.mu)
        ...
        self.mu = mu_new

    def aggregate(self, policy_fn):
        """Compute aggregate from distribution: sum mu(a,z) * f(a,z)."""
        return np.sum(self.mu * policy_fn)

    def gini(self):
        """Compute Gini coefficient from the distribution."""
        ...

    def sample_agents(self, n_agents, rng):
        """Sample n individual agents from the distribution for LLM mode."""
        ...
```

#### Integration with Lead

In the period lifecycle:
- **KFE mode (benchmark):** No individual agents. Use `Distribution.forward()` after policy functions are computed. Aggregate quantities from distribution.
- **Individual mode (LLM):** Sample 5000+ agents from the distribution at initialization. Individual agents make decisions. Distribution is updated from realized agent transitions.

### 2.2 Calibration Module

**File:** `src/emergent_constitution/calibration.py` (NEW file)

*Traceability: REQ-208, REQ-209, REQ-210*

```python
@dataclass
class CalibrationTargets:
    wealth_gini: float = 0.80
    entrepreneur_share: float = 0.10
    top10_wealth_share: float = 0.70
    median_mpc: float | None = None      # Phase 3 only
    liquid_illiquid_ratio: float | None = None  # Phase 3 only

class Calibrator:
    def compute_model_moments(self, distribution, ...) -> dict[str, float]:
        """Compute model-implied moments from stationary distribution."""
        ...

    def smm_objective(self, params, targets) -> float:
        """Simulated Method of Moments objective function."""
        ...

    def calibrate(self, initial_params, targets) -> dict[str, float]:
        """Run SMM to find parameters matching data targets."""
        ...
```

---

## 3. Phase 3: HANK Features

### 3.1 Two-Asset Household Model

**Files:** `src/emergent_constitution/models/household.py` (modify), `src/emergent_constitution/egm_solver.py` (extend)

*Traceability: REQ-301, REQ-302, REQ-303, REQ-304, REQ-305*

#### Household State Extension

```python
class HouseholdState(BaseModel):
    # Existing fields preserved
    wealth: float  # = liquid + illiquid (backward compat)

    # New two-asset fields
    liquid: float = 0.0       # b_t: government bonds
    illiquid: float = 0.0     # k_t: physical capital / housing

    @property
    def total_wealth(self) -> float:
        return self.liquid + self.illiquid
```

**Backward compat:** When `liquid == 0 and illiquid == 0`, the model uses `wealth` as the single asset (Phase 1/2 behavior). When two-asset mode is active, `wealth = liquid + illiquid`.

#### Transaction Cost Function

```python
def adjustment_cost(deposit: float, illiquid_stock: float,
                    chi_0: float = 0.01, chi_1: float = 0.005) -> float:
    """Convex adjustment cost for illiquid deposits/withdrawals."""
    if abs(deposit) < 1e-10:
        return 0.0
    k = max(illiquid_stock, 1e-10)
    return chi_0 * abs(deposit) + chi_1 * deposit**2 / k
```

#### Two-Asset EGM

The two-asset problem is solved via **nested EGM** (Kaplan-Moll-Violante 2018, online appendix):

1. **Outer loop:** For each illiquid deposit `d` on a coarse grid (30 points):
   - Compute adjustment cost `chi(d)`
   - Solve the liquid-asset EGM conditional on `d` (standard one-asset EGM with modified budget: subtract `d + chi(d)` from resources)
2. **Inner solve:** Standard EGM for `(c, l)` given `d`
3. **Outer optimization:** For each `(b, k, z)` state, pick the `d` that maximizes value
4. **Upper envelope:** Handle non-convexities from the adjustment cost using the upper envelope algorithm

### 3.2 Government and Bond Market

**File:** `src/emergent_constitution/government.py` (NEW file)

*Traceability: REQ-306, REQ-307, REQ-308, REQ-309*

```python
@dataclass
class GovernmentState:
    debt: float = 0.0          # B_t: outstanding debt
    tax_revenue: float = 0.0   # T_t
    spending: float = 0.0      # G_t
    transfers: float = 0.0     # Tr_t
    bond_rate: float = 0.03    # r^b_t
    debt_to_gdp: float = 0.0

class Government:
    def update_budget(self, state, bond_rate):
        """Government budget constraint: B' = (1+r^b)*B + G + Tr - T."""
        ...

    def fiscal_rule(self, state, output):
        """Auto-adjust taxes if debt/GDP exceeds threshold."""
        ...
```

**Bond market clearing:** Bisection on `r^b` to equate household bond demand `sum_i b_i(r^b)` with government supply `B_t`.

### 3.3 Nominal Block

**File:** `src/emergent_constitution/nominal.py` (NEW file)

*Traceability: REQ-310, REQ-311, REQ-312, REQ-313, REQ-314, REQ-315*

```python
@dataclass
class NominalState:
    price_level: float = 1.0
    inflation: float = 0.0        # pi_t
    nominal_rate: float = 0.05    # i_t
    real_bond_rate: float = 0.03  # r^b_t
    expected_inflation: float = 0.0
    marginal_cost: float = 1.0    # mc_t

class NominalBlock:
    """New Keynesian nominal rigidities block."""

    def __init__(self, config):
        self.phi_p = config.rotemberg_cost      # price adjustment cost
        self.phi_pi = config.taylor_phi_pi      # Taylor rule inflation weight
        self.phi_y = config.taylor_phi_y        # Taylor rule output weight
        self.pi_bar = config.inflation_target   # target inflation
        self.epsilon = config.elasticity_sub    # elasticity of substitution

    def taylor_rule(self, inflation, output_gap, r_natural):
        """i_t = r_bar + phi_pi*(pi - pi_bar) + phi_y*gap."""
        return r_natural + self.phi_pi * (inflation - self.pi_bar) + \
               self.phi_y * output_gap

    def fisher_equation(self, nominal_rate, expected_inflation):
        """r^b = (1 + i) / (1 + E[pi']) - 1."""
        return (1 + nominal_rate) / (1 + expected_inflation) - 1

    def nkpc(self, inflation, marginal_cost, expected_inflation,
             output, expected_output, beta):
        """NKPC residual (should be zero in equilibrium)."""
        ...

    def update(self, output, steady_state_output, wage, ...):
        """Compute nominal state for this period."""
        ...
```

### 3.4 Extended Market Clearing (Phase 3)

**File:** `src/emergent_constitution/market_clearing.py` (extend)

With two assets and nominal rigidities, market clearing becomes:

1. **Labor market:** `L^d(w) = L^s` → bisect for `w`
2. **Capital market:** `K^d = sum_i k_i` → `r^k = MPK - delta`
3. **Bond market:** `sum_i b_i(r^b) = B_t` → bisect for `r^b`
4. **Nominal block:** Taylor rule → `i_t`, Fisher equation → `r^b`

**Iteration:** The nominal block and bond market clearing interact (Fisher equation links `i_t` to `r^b` via expected inflation). Solve via outer iteration:
1. Guess `pi_{t+1}` (use adaptive expectations: `E[pi'] = rho_pi * pi_t`)
2. Taylor rule → `i_t`
3. Fisher → `r^b`
4. Solve household problem conditional on `(w, r^k, r^b)`
5. Clear bond market → check `sum b_i = B_t`
6. Compute actual `pi_t` from NKPC
7. Update `E[pi']` and iterate until `|pi_t - pi_t^{old}| < 1e-6`

---

## 4. Phase 4: Microfound Politics

### 4.1 Political Utility in the Bellman Equation

**Files:** `src/emergent_constitution/models/household.py`, `src/emergent_constitution/egm_solver.py`, `src/emergent_constitution/llm_engine.py`

*Traceability: REQ-401, REQ-402, REQ-403, REQ-404, REQ-405*

#### Model Extension

The Bellman equation becomes:
```
V(a, z; C) = max_{c,l} { u(c, l, G) + lambda * v(C; theta) + beta * E[V(a', z'; C')] }
```

Since the constitution `C` changes infrequently (every `proposal_interval` periods), the political utility `v(C; theta)` is constant between governance periods. This means:
- Between governance periods: solve the standard Bellman (economic optimization)
- At governance periods: compare `V(s; C_proposed)` vs `V(s; C_current)` for each proposal

**Implementation:**
- Add `political_lambda: float = 0.05` to `SimulationConfigV2`
- Add `compute_political_utility(constitution, theta)` to a new utility module
- In the EGM solver, add the political utility term as a constant flow utility bonus
- In the LLM engine, provide the Bellman-implied preference for each proposal

```python
def compute_political_utility(constitution, theta_eq, theta_lib):
    """v(C; theta) = theta_eq * (-Gini) + theta_lib * (-tax_rate)."""
    gini_component = -constitution.effective_gini_impact()
    liberty_component = -constitution.effective_tax_rate()
    return theta_eq * gini_component + theta_lib * liberty_component
```

---

## 5. Configuration Extensions

**File:** `src/emergent_constitution/config.py`

```python
class SimulationConfigV2(BaseModel):
    # ... existing fields ...

    # Phase 1: EGM
    solver_method: Literal["egm", "vfi", "vfi_numpy"] = "egm"
    egm_n_asset_points: int = 200
    euler_residual_tolerance: float = 1e-6

    # Phase 1: Market clearing
    market_clearing_method: Literal["walrasian", "analytical"] = "walrasian"
    wage_bisection_tolerance: float = 1e-8
    wage_bisection_max_iter: int = 100

    # Phase 2: Distribution
    distribution_mode: Literal["individual", "kfe"] = "individual"
    kfe_convergence_tolerance: float = 1e-10
    kfe_max_iterations: int = 10000
    num_z_states: int = 7  # was 5
    num_e_states: int = 7  # was 5

    # Phase 2: Calibration
    calibration_targets: CalibrationTargets | None = None

    # Phase 3: Two-asset
    two_asset_mode: bool = False
    chi_0: float = 0.01          # linear adjustment cost
    chi_1: float = 0.005         # quadratic adjustment cost
    b_min: float = 0.0           # liquid borrowing limit

    # Phase 3: Government
    initial_debt: float = 0.0
    debt_gdp_max: float = 1.5
    fiscal_rule_adjustment: float = 0.01

    # Phase 3: Nominal
    nominal_rigidities: bool = False
    rotemberg_cost: float = 100.0
    taylor_phi_pi: float = 1.5
    taylor_phi_y: float = 0.125
    inflation_target: float = 0.02
    elasticity_sub: float = 6.0
    wage_rigidity: bool = False
    rotemberg_wage_cost: float = 50.0

    # Phase 4: Political utility
    political_lambda: float = 0.05
    pure_bellman_politics: bool = False
```

---

## 6. Period Lifecycle Changes

**File:** `src/emergent_constitution/lead.py`

The 9-step lifecycle is extended to 12 steps:

```
Step 1:  Draw shocks (unchanged)
Step 1b: Apply productivity mechanism effects (existing, from mechanism-effects branch)
Step 2:  Clear markets (REQ-101: Walrasian with firm-level FOCs)
Step 2b: Compute nominal block (Phase 3: Taylor rule, Fisher, NKPC)
Step 3:  Collect decisions (REQ-115: EGM solver, REQ-110: proper occ choice)
Step 3b: Portfolio choice (Phase 3: two-asset deposit/withdrawal decisions)
Step 4:  Validate constraints (REQ-107: corrected entrepreneur budget)
Step 4b: Apply mechanism constraints (existing, from mechanism-effects branch)
Step 5:  Execute production (unchanged, but use corrected firm profits)
Step 6:  Enforce constitution (unchanged)
Step 6b: Enforce mechanism effects (existing, from mechanism-effects branch)
Step 7:  Process governance (Phase 4: Bellman-derived political preferences)
Step 7b: Update government budget (Phase 3: debt dynamics, fiscal rule)
Step 8:  Update states (REQ-107: separate entrepreneur/worker budget constraints)
Step 8b: Update distribution (Phase 2: KFE forward step)
Step 9:  Observe (REQ-210: report model moments vs calibration targets)
```

---

## 7. File Inventory

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| `src/emergent_constitution/egm_solver.py` | 1 | EGM household solver |
| `src/emergent_constitution/distribution.py` | 2 | KFE distribution tracking |
| `src/emergent_constitution/calibration.py` | 2 | Moment-matching calibration |
| `src/emergent_constitution/government.py` | 3 | Government budget, debt, fiscal rule |
| `src/emergent_constitution/nominal.py` | 3 | Nominal rigidities, Taylor rule, NKPC |
| `src/emergent_constitution/political_utility.py` | 4 | Political utility function |
| `tests/test_egm_solver.py` | 1 | EGM tests |
| `tests/test_walrasian_clearing.py` | 1 | Walrasian market clearing tests |
| `tests/test_entrepreneur_fix.py` | 1 | Corrected budget constraint tests |
| `tests/test_occupational_choice.py` | 1 | Bellman-based occ choice tests |
| `tests/test_distribution.py` | 2 | KFE distribution tests |
| `tests/test_calibration.py` | 2 | Calibration module tests |
| `tests/test_two_asset.py` | 3 | Two-asset EGM tests |
| `tests/test_government.py` | 3 | Government budget tests |
| `tests/test_nominal.py` | 3 | Nominal block tests |
| `tests/test_political_utility.py` | 4 | Political utility tests |

### Modified Files

| File | Phase | Changes |
|------|-------|---------|
| `market_clearing.py` | 1, 3 | Walrasian bisection, two-asset clearing |
| `entrepreneurial_solver.py` | 1 | Bellman firm value, proper occ choice |
| `lead.py` | 1, 2, 3, 4 | Extended lifecycle, KFE integration |
| `models/household.py` | 3 | Two-asset fields (liquid, illiquid) |
| `config.py` | 1, 2, 3, 4 | New configuration parameters |
| `economics.py` | 1 | Separate entrepreneur income computation |
| `llm_engine.py` | 4 | Bellman-derived political context |
| `observer.py` | 2 | Distribution-based statistics |
| `reporter.py` | 2 | Moment reporting |
| `models/__init__.py` | all | Export new types |

---

## 8. Error Handling and Safety

### Numerical Guards (all phases)

- **Bisection divergence:** If bisection doesn't converge in 100 iterations, log warning and use midpoint of remaining bracket. Fall back to representative-firm clearing if bracket is malformed.
- **Singular matrix:** If `(I - beta * Pi)` is singular in firm value computation, fall back to perpetuity formula with warning.
- **EGM non-monotonicity:** If endogenous grid is non-monotone (can happen near constraints), apply upper envelope algorithm (Fella 2014) to restore monotonicity.
- **Negative consumption from EGM:** Clamp to `1e-10` and log.
- **KFE mass leak:** After each forward step, verify `sum(mu) = 1` to tolerance `1e-12`. Renormalize if violated.
- **Explosive debt:** Fiscal rule caps debt/GDP. If debt exceeds 5× GDP despite fiscal rule, freeze governance changes and log critical warning.
- **NKPC non-convergence:** If nominal block doesn't converge in 50 iterations, use previous period's inflation as current and log.

### Backward Compatibility Guards

- All new features gated behind configuration flags (default: off)
- `solver_method="vfi_numpy"` reproduces exact v2 behavior
- `market_clearing_method="analytical"` reproduces exact v2 behavior
- `two_asset_mode=False` (default) keeps single-asset behavior
- `nominal_rigidities=False` (default) keeps real economy
- Test suite runs both old and new code paths

---

## 9. Testing Strategy

### Phase 1 Tests
- **EGM accuracy:** Compare EGM policy functions against brute-force VFI on fine grid. Max deviation < `1e-4`.
- **Euler residual:** Verify Euler equation residual < `1e-6` for EGM solutions.
- **Walrasian clearing:** Verify `|L^d(w*) - L^s| < 1e-8` with heterogeneous firms.
- **Entrepreneur budget:** Verify entrepreneurs receive only firm profit, no labor income.
- **Firm value Bellman:** Compare `(I - beta*Pi)^{-1} * pi` against iterative solution.
- **Occupational choice:** Verify entry/exit uses VFI values, not hardcoded approximations.

### Phase 2 Tests
- **KFE conservation:** Verify `sum(mu) = 1` after 10,000 forward steps.
- **KFE vs. simulation:** Run 50,000 individual agents, compute empirical distribution, compare to KFE. KS-statistic < 0.05.
- **Moment computation:** Verify Gini, mean, percentiles from distribution match individual-agent computation.
- **Rouwenhorst 7-point:** Verify transition matrix properties (row sums = 1, stationary distribution exists).

### Phase 3 Tests
- **Two-asset budget:** Verify `b' + k' + c + chi(d) + T = (1+r^b)*b + (1+r^k)*k + w*z*(1-l) + Tr`.
- **Bond market clearing:** Verify `sum b_i = B_t` to tolerance `1e-8`.
- **Fisher equation:** Verify `r^b = (1+i)/(1+E[pi']) - 1` every period.
- **Taylor rule:** Verify `i = r_bar + 1.5*(pi - pi_bar) + 0.125*gap`.
- **Government budget:** Verify `B' = (1+r^b)*B + G + Tr - T` exactly.
- **Fiscal rule:** Verify tax adjustment triggers when debt/GDP > threshold.

### Phase 4 Tests
- **Political utility:** Verify `v(C; theta)` computation matches expected values.
- **Bellman politics:** Verify Bellman-derived proposal preference matches value function comparison.
- **Integration:** Full 100-period simulation with political utility, verify welfare = economic + political.
