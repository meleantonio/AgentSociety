# Intent: HANK Upgrade — From Toy Bewley to Publication-Grade DSGE-HA

## Vision

Transform the Emergent Constitution simulation from a prototype Bewley economy into a **publication-grade Heterogeneous Agent New Keynesian (HANK) model** with endogenous governance. The upgrade addresses ten fundamental critiques identified in a detailed review against the state-of-the-art heterogeneous-agent macro literature (Kaplan-Moll-Violante 2018, Auclert 2019, Cagetti-De Nardi 2006).

The result: a model that can credibly claim to produce quantitatively meaningful predictions about the co-evolution of institutions and inequality, grounded in the same economic foundations as the leading HANK models.

## Why This Upgrade

The current v2 engine, while functional, has critical deficiencies that undermine its economic credibility:

### Economic Inconsistencies (Must Fix)

1. **Market clearing is internally inconsistent.** The model features heterogeneous firms (different TFPs, different capitals) but computes prices using a representative-firm formula `Y = A*K^a*L^(1-a)`. With heterogeneous firms, aggregate output does NOT equal a representative firm production function unless specific aggregation conditions hold — conditions that are violated because firms have fixed capital within a period. The `max(firm_capital, household_wealth)` in the code is economically meaningless.

2. **Entrepreneur budget constraint double-counts labor income.** Entrepreneurs earn both labor income `w*z*(1-l)` AND firm profit, but in Cagetti-De Nardi (2006), entrepreneurs are purely residual claimants. Their labor is embedded in firm production, not separately compensated.

3. **Occupational choice uses hardcoded approximations.** Worker lifetime value uses `leisure=0.35` and `consumption=0.7*income` instead of VFI-computed policy functions. The perpetuity formula ignores transition dynamics and option values. This defeats the purpose of solving the Bellman equation.

4. **VFI is absurdly coarse.** 11 consumption grid points and 21 leisure grid points produce massive approximation error. Modern practice uses the Endogenous Grid Method (Carroll 2006) for exact policy functions.

### Missing Model Features (HANK Gap)

5. **No nominal rigidities.** No sticky prices, no Phillips Curve, no monetary policy. The interest rate is the marginal product of capital, not a policy instrument. This makes the model a real Bewley economy that cannot speak to the HANK literature it claims connection to.

6. **Single-asset economy.** No liquid/illiquid asset distinction. Cannot capture "wealthy hand-to-mouth" agents (high illiquid wealth, low liquid wealth, high MPC) that drive HANK fiscal multipliers.

7. **No government debt or bond market.** Fiscal policy operates through taxes and transfers but government cannot borrow. No bond market to determine the liquid return.

8. **50 agents is not a continuum.** The law of large numbers doesn't hold. Gini coefficients from 50 draws have standard deviation ~0.14. Need 5000+ agents or KFE-based distribution tracking.

### Weak Foundations

9. **Political decisions have no microfoundation.** The value vector `(v_eq, v_lib)` enters politics but not the Bellman equation. Agents are schizophrenic: maximize Cobb-Douglas utility for economics, use separate ideology for politics.

10. **No moment-matching calibration.** Parameters are listed without data targets. No wealth Gini target, no entrepreneur share target, no MPC distribution target.

## Upgrade Architecture

The upgrade is organized into four phases with strict dependency ordering:

### Phase 1: Fix Economic Foundations (No New Features)
Fix the market clearing, entrepreneur budget constraint, occupational choice, and VFI solver. After Phase 1, the model produces a correctly-computed competitive equilibrium with heterogeneous firms.

### Phase 2: Scale and Calibrate
Scale from 50 agents to a continuum (KFE or 5000+ histogram). Add moment-matching calibration with data targets. Increase Rouwenhorst grid points.

### Phase 3: HANK Features
Add the two-asset structure (liquid bonds + illiquid capital), government debt, bond market clearing, nominal rigidities (Rotemberg pricing), and Taylor rule monetary policy.

### Phase 4: Microfound Politics
Unify political preferences with the utility function. Political utility derives from economic self-interest plus ideological term, entering the Bellman equation directly.

## Relationship to Literature

After this upgrade, the model will be comparable to:
- **Kaplan, Moll, Violante (2018 AER):** Two-asset HANK with liquid/illiquid distinction
- **Auclert (2019 AER):** Fiscal multipliers with heterogeneous MPCs
- **Cagetti and De Nardi (2006 JPE):** Entrepreneurship with wealth constraints
- **Krusell and Smith (1998 JPE):** Distribution tracking with aggregate shocks
- **Carroll (2006 EL):** Endogenous Grid Method for exact policy functions
- **Auclert, Bardoczy, Rognlie, Straub (2021 Econometrica):** Sequence-space Jacobian methods

The novel contribution remains the endogenous governance layer — no existing HANK model has institutions that evolve through agent proposals and voting.

## Success Criteria

1. **Zero market-clearing error** with properly aggregated heterogeneous firm output
2. **EGM-precision policy functions** with Euler equation residuals < 1e-8
3. **5000+ agents** or KFE-based distribution with smooth wealth density
4. **Two-asset portfolio choice** reproducing liquid/illiquid wealth ratio ~0.25
5. **Nominal rigidities** with Phillips Curve and Taylor rule
6. **Moment-matched calibration** targeting wealth Gini ~0.80, entrepreneur share ~10%, median MPC ~0.25
7. **Unified political-economic utility** where political preferences derive from the Bellman equation
8. **All existing 977 tests pass** at each phase boundary (backward compatibility)
9. **New test coverage** for every added component (target: 90%+)
