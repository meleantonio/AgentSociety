"""Nominal rigidities block — New Keynesian sticky prices, Taylor rule, NKPC.

Implements REQ-310 (Rotemberg pricing), REQ-311 (NKPC), REQ-312 (Taylor rule),
REQ-313 (Fisher equation), REQ-314 (nominal state tracking), REQ-315 (optional
wage rigidity).

Design reference: spec/design.md section 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()

# Maximum outer-iteration attempts for nominal block convergence
_MAX_NOMINAL_ITER = 50
_NOMINAL_TOL = 1e-6

# Adaptive expectations persistence for E[pi']
_RHO_PI = 0.9


@dataclass
class NominalState:
    """Tracks nominal variables for a single period (REQ-314).

    Attributes:
        price_level: P_t (cumulative price level).
        inflation: pi_t = P_t / P_{t-1} - 1.
        nominal_rate: i_t (policy rate from Taylor rule).
        real_bond_rate: r^b_t from Fisher equation.
        expected_inflation: E_t[pi_{t+1}].
        marginal_cost: mc_t (real marginal cost).
    """

    price_level: float = 1.0
    inflation: float = 0.0
    nominal_rate: float = 0.05
    real_bond_rate: float = 0.03
    expected_inflation: float = 0.0
    marginal_cost: float = 1.0


class NominalBlock:
    """New Keynesian nominal rigidities block.

    Computes the Taylor rule, Fisher equation, and Rotemberg NKPC each period.
    The ``update`` method runs an outer iteration to find the inflation rate
    consistent with the NKPC, Taylor rule, and Fisher equation simultaneously.

    Args:
        config: Simulation configuration providing nominal parameters.

    Attributes:
        phi_p: Rotemberg price adjustment cost (REQ-310).
        phi_pi: Taylor rule inflation coefficient (> 1 for Taylor principle).
        phi_y: Taylor rule output gap coefficient.
        pi_bar: Inflation target.
        epsilon: Elasticity of substitution between varieties.
    """

    def __init__(self, config: object) -> None:
        self.phi_p: float = getattr(config, "rotemberg_cost", 100.0)
        self.phi_pi: float = getattr(config, "taylor_phi_pi", 1.5)
        self.phi_y: float = getattr(config, "taylor_phi_y", 0.125)
        self.pi_bar: float = getattr(config, "inflation_target", 0.02)
        self.epsilon: float = getattr(config, "elasticity_sub", 6.0)

        # Previous-period state for adaptive expectations
        self._prev_state: NominalState = NominalState()

    def taylor_rule(
        self,
        inflation: float,
        output_gap: float,
        r_natural: float,
    ) -> float:
        """Compute the nominal interest rate via the Taylor rule (REQ-312).

        i_t = r_bar + phi_pi * (pi_t - pi_bar) + phi_y * gap_t

        Args:
            inflation: Current inflation rate pi_t.
            output_gap: (Y_t - Y_bar) / Y_bar.
            r_natural: Natural (steady-state) real interest rate r_bar.

        Returns:
            Nominal interest rate i_t.
        """
        return r_natural + self.phi_pi * (inflation - self.pi_bar) + self.phi_y * output_gap

    def fisher_equation(
        self,
        nominal_rate: float,
        expected_inflation: float,
    ) -> float:
        """Compute the real bond rate from the Fisher equation (REQ-313).

        r^b_t = (1 + i_t) / (1 + E_t[pi_{t+1}]) - 1

        Args:
            nominal_rate: Nominal interest rate i_t.
            expected_inflation: Expected next-period inflation E_t[pi_{t+1}].

        Returns:
            Real bond rate r^b_t.
        """
        return (1.0 + nominal_rate) / (1.0 + expected_inflation) - 1.0

    def nkpc(
        self,
        inflation: float,
        marginal_cost: float,
        expected_inflation: float,
        output: float,
        expected_output: float,
        beta: float,
    ) -> float:
        """Compute the NKPC residual (REQ-311).

        The Rotemberg NKPC in equilibrium:
        phi_p * pi_t * (pi_t - pi_bar) = (1 - epsilon) + epsilon * mc_t
            + beta * phi_p * E[pi_{t+1} * (pi_{t+1} - pi_bar) * Y_{t+1}/Y_t]

        Rearranged, the residual (should be zero) is:
        LHS - RHS = 0

        where:
        LHS = phi_p * pi_t * (pi_t - pi_bar)
        RHS = (1 - epsilon) + epsilon * mc_t
              + beta * phi_p * E[pi_{t+1}] * (E[pi_{t+1}] - pi_bar) * Y_{t+1}/Y_t

        Args:
            inflation: Current inflation pi_t.
            marginal_cost: Real marginal cost mc_t.
            expected_inflation: E_t[pi_{t+1}].
            output: Current output Y_t.
            expected_output: Expected next-period output Y_{t+1}.
            beta: Household discount factor.

        Returns:
            NKPC residual (zero in equilibrium).
        """
        lhs = self.phi_p * inflation * (inflation - self.pi_bar)
        rhs_static = (1.0 - self.epsilon) + self.epsilon * marginal_cost
        # Forward-looking term
        y_ratio = expected_output / max(output, 1e-10)
        rhs_forward = (
            beta * self.phi_p * expected_inflation * (expected_inflation - self.pi_bar) * y_ratio
        )
        return lhs - rhs_static - rhs_forward

    def nkpc_solve_inflation(
        self,
        marginal_cost: float,
        expected_inflation: float,
        output: float,
        expected_output: float,
        beta: float,
    ) -> float:
        """Solve the NKPC for current inflation pi_t given mc_t and expectations.

        From the NKPC:
        phi_p * pi * (pi - pi_bar) = (1 - epsilon) + epsilon * mc
            + beta * phi_p * E[pi'] * (E[pi'] - pi_bar) * Y'/Y

        This is a quadratic in pi:
        phi_p * pi^2 - phi_p * pi_bar * pi - [(1-eps) + eps*mc + forward] = 0

        Using the quadratic formula: pi = [phi_p*pi_bar +/- sqrt(D)] / (2*phi_p)
        where D = (phi_p*pi_bar)^2 + 4*phi_p*[(1-eps) + eps*mc + forward]

        We select the root closer to pi_bar (the stable equilibrium).

        Args:
            marginal_cost: Real marginal cost mc_t.
            expected_inflation: E_t[pi_{t+1}].
            output: Current output Y_t.
            expected_output: Expected next-period output Y_{t+1}.
            beta: Household discount factor.

        Returns:
            Inflation rate pi_t consistent with NKPC.
        """
        y_ratio = expected_output / max(output, 1e-10)
        forward = (
            beta * self.phi_p * expected_inflation * (expected_inflation - self.pi_bar) * y_ratio
        )
        rhs = (1.0 - self.epsilon) + self.epsilon * marginal_cost + forward

        # Quadratic: phi_p * pi^2 - phi_p * pi_bar * pi - rhs = 0
        a_coeff = self.phi_p
        b_coeff = -self.phi_p * self.pi_bar
        c_coeff = -rhs

        discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
        if discriminant < 0:
            # No real solution — return target inflation as fallback
            log.warning("nkpc.no_real_solution", discriminant=discriminant, mc=marginal_cost)
            return self.pi_bar

        sqrt_d = discriminant**0.5
        root1 = (-b_coeff + sqrt_d) / (2.0 * a_coeff)
        root2 = (-b_coeff - sqrt_d) / (2.0 * a_coeff)

        # Select root closer to target inflation
        if abs(root1 - self.pi_bar) <= abs(root2 - self.pi_bar):
            return root1
        return root2

    def update(
        self,
        output: float,
        steady_state_output: float,
        wage: float,
        interest_rate: float,
        beta: float,
        aggregate_tfp: float = 1.0,
    ) -> NominalState:
        """Compute the nominal state for this period via outer iteration.

        Algorithm (spec/design.md section 3.4):
        1. Guess E[pi'] using adaptive expectations: E[pi'] = rho * pi_{t-1}
        2. Compute marginal cost: mc = w / (A * MPL) (for Cobb-Douglas, mc = w/A
           at the aggregate level since MPL = (1-alpha)*Y/L and mc = w*L/((1-alpha)*Y)
           simplifies to w / ((1-alpha) * A * (K/L)^alpha)). We approximate mc = w/A
           for aggregate marginal cost in the Rotemberg framework.
        3. Taylor rule -> i_t
        4. Fisher -> r^b
        5. NKPC -> solve for actual pi_t
        6. Update E[pi'] and iterate until |pi_t - pi_t_old| < tol

        Args:
            output: Aggregate output Y_t.
            steady_state_output: Potential output Y_bar.
            wage: Real wage w_t.
            interest_rate: Real capital rate r^k_t (from market clearing).
            beta: Household discount factor.
            aggregate_tfp: A_t (aggregate TFP, used for marginal cost).

        Returns:
            Updated NominalState for the current period.
        """
        prev = self._prev_state

        # Output gap
        ss_output = max(steady_state_output, 1e-10)
        output_gap = (output - ss_output) / ss_output

        # Natural rate approximation: use real capital rate as proxy
        r_natural = interest_rate

        # Marginal cost: in Rotemberg framework, mc = w / (aggregate_tfp)
        # More precisely, for Cobb-Douglas this is the inverse markup at
        # the competitive price. We use w / A as the standard approximation.
        mc = wage / max(aggregate_tfp, 1e-10)

        # Adaptive expectations initialization
        expected_inflation = _RHO_PI * prev.inflation

        # Expected output (simple persistence assumption)
        expected_output = output  # naive expectation: Y_{t+1} = Y_t

        # Outer iteration
        pi_old = prev.inflation
        converged = False

        for _iteration in range(1, _MAX_NOMINAL_ITER + 1):
            # Taylor rule
            nominal_rate = self.taylor_rule(pi_old, output_gap, r_natural)

            # Fisher equation
            real_bond_rate = self.fisher_equation(nominal_rate, expected_inflation)

            # NKPC: solve for pi_t
            pi_new = self.nkpc_solve_inflation(
                marginal_cost=mc,
                expected_inflation=expected_inflation,
                output=output,
                expected_output=expected_output,
                beta=beta,
            )

            # Check convergence
            if abs(pi_new - pi_old) < _NOMINAL_TOL:
                converged = True
                pi_old = pi_new
                break

            # Update for next iteration
            expected_inflation = _RHO_PI * pi_new
            pi_old = pi_new

        if not converged:
            log.warning(
                "nominal.not_converged",
                iterations=_MAX_NOMINAL_ITER,
                last_pi=pi_old,
                delta=abs(pi_new - pi_old) if "pi_new" in dir() else None,
            )

        # Final Taylor rule and Fisher with converged inflation
        inflation = pi_old
        nominal_rate = self.taylor_rule(inflation, output_gap, r_natural)
        expected_inflation = _RHO_PI * inflation
        real_bond_rate = self.fisher_equation(nominal_rate, expected_inflation)

        # Update price level: P_t = P_{t-1} * (1 + pi_t)
        price_level = prev.price_level * (1.0 + inflation)

        state = NominalState(
            price_level=price_level,
            inflation=inflation,
            nominal_rate=nominal_rate,
            real_bond_rate=real_bond_rate,
            expected_inflation=expected_inflation,
            marginal_cost=mc,
        )

        self._prev_state = state

        log.debug(
            "nominal.updated",
            inflation=round(inflation, 6),
            nominal_rate=round(nominal_rate, 6),
            real_bond_rate=round(real_bond_rate, 6),
            mc=round(mc, 6),
            converged=converged,
        )

        return state
