"""Calibration module -- moment-matching targets and SMM calibration.

Provides CalibrationTargets for storing empirical calibration targets,
a Calibrator class for computing model moments from household data and
running Simulated Method of Moments (SMM) calibration to match model
moments to data targets.

Traceability: REQ-208, REQ-209, REQ-210.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()

# Default moment weights for SMM objective (equal weighting).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "wealth_gini": 1.0,
    "entrepreneur_share": 1.0,
    "top10_wealth_share": 1.0,
}


@dataclass
class CalibrationTargets:
    """Empirical calibration targets for moment matching.

    Args:
        wealth_gini: Target wealth Gini coefficient.
        entrepreneur_share: Target share of entrepreneurs in population.
        top10_wealth_share: Target share of wealth held by top 10%.
        median_mpc: Target median marginal propensity to consume (Phase 3).
        liquid_illiquid_ratio: Target liquid-to-illiquid wealth ratio (Phase 3).
    """

    wealth_gini: float = 0.80
    entrepreneur_share: float = 0.10
    top10_wealth_share: float = 0.70
    median_mpc: float | None = None
    liquid_illiquid_ratio: float | None = None


# Parameter names used in SMM calibration.
_PARAM_NAMES: list[str] = ["beta_d", "sigma_z", "entry_cost", "a_min"]

# Default bounds for calibration parameters.
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "beta_d": (0.80, 0.999),
    "sigma_z": (0.05, 1.0),
    "entry_cost": (0.1, 50.0),
    "a_min": (0.0, 10.0),
}


class Calibrator:
    """Computes model moments and runs SMM calibration.

    Args:
        weights: Optional moment weights for the SMM objective.
            Keys are moment names, values are positive weights.
            Defaults to equal weighting of the three core moments.
        tolerance: Convergence tolerance for the optimizer.
        max_iterations: Maximum iterations for the optimizer.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        tolerance: float = 1e-6,
        max_iterations: int = 500,
    ) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def compute_model_moments(
        self,
        agents: list[Any],
    ) -> dict[str, float]:
        """Compute model-implied moments from agent/household data.

        Accepts any objects with ``wealth`` and ``role`` attributes (both
        HouseholdState and AgentState are supported). For objects without
        a ``role`` attribute, all agents are treated as non-entrepreneurs.

        Args:
            agents: List of agent/household state objects.

        Returns:
            Dictionary with keys:
                - ``wealth_gini``: Gini coefficient of wealth.
                - ``entrepreneur_share``: Fraction of entrepreneurs.
                - ``top10_wealth_share``: Share of wealth held by top 10%.
        """
        n = len(agents)
        if n == 0:
            return {
                "wealth_gini": 0.0,
                "entrepreneur_share": 0.0,
                "top10_wealth_share": 0.0,
            }

        wealths = np.array([a.wealth for a in agents], dtype=np.float64)

        # Gini coefficient
        gini = self._compute_gini(wealths)

        # Entrepreneur share
        n_entre = 0
        for a in agents:
            role = getattr(a, "role", None)
            if role is not None and str(role) == "entrepreneur":
                n_entre += 1
        entrepreneur_share = n_entre / n if n > 0 else 0.0

        # Top 10% wealth share
        top10_share = self._compute_top_pct_share(wealths, 0.10)

        return {
            "wealth_gini": float(gini),
            "entrepreneur_share": float(entrepreneur_share),
            "top10_wealth_share": float(top10_share),
        }

    def smm_objective(
        self,
        params: dict[str, float],
        targets: CalibrationTargets,
        model_moments: dict[str, float] | None = None,
        simulate_fn: Callable[[dict[str, float]], dict[str, float]] | None = None,
    ) -> float:
        """Compute weighted squared distance between model and target moments.

        Either ``model_moments`` or ``simulate_fn`` must be provided.
        If ``simulate_fn`` is given, it is called with ``params`` to produce
        the model moments.

        Args:
            params: Current parameter values (keys from _PARAM_NAMES).
            targets: Calibration target values.
            model_moments: Pre-computed model moments (if available).
            simulate_fn: Function that takes params dict and returns moments.

        Returns:
            Weighted sum of squared moment deviations.
        """
        if model_moments is None:
            if simulate_fn is None:
                msg = "Either model_moments or simulate_fn must be provided"
                raise ValueError(msg)
            model_moments = simulate_fn(params)

        target_dict = {
            "wealth_gini": targets.wealth_gini,
            "entrepreneur_share": targets.entrepreneur_share,
            "top10_wealth_share": targets.top10_wealth_share,
        }

        objective = 0.0
        for moment_name, target_val in target_dict.items():
            if target_val is None:
                continue
            model_val = model_moments.get(moment_name, 0.0)
            weight = self.weights.get(moment_name, 1.0)
            # Guard against zero target to avoid division issues in relative error
            denom = max(abs(target_val), 1e-10)
            objective += weight * ((model_val - target_val) / denom) ** 2

        return float(objective)

    def calibrate(
        self,
        initial_params: dict[str, float],
        targets: CalibrationTargets,
        simulate_fn: Callable[[dict[str, float]], dict[str, float]],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Run SMM calibration to find parameters matching data targets.

        Uses scipy.optimize.minimize with Nelder-Mead (derivative-free).

        Args:
            initial_params: Starting parameter values.
            targets: Calibration target values.
            simulate_fn: Function that takes a params dict and returns
                a dict of model moments.
            bounds: Optional parameter bounds (overrides defaults).

        Returns:
            Calibrated parameter dict with optimized values.
        """
        from scipy.optimize import minimize

        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[k] for k in param_names], dtype=np.float64)

        effective_bounds = bounds or _PARAM_BOUNDS

        def objective(x: np.ndarray) -> float:
            params = dict(zip(param_names, x, strict=True))
            # Enforce bounds by projecting
            for name, val in params.items():
                if name in effective_bounds:
                    lo, hi = effective_bounds[name]
                    params[name] = float(np.clip(val, lo, hi))
            try:
                moments = simulate_fn(params)
                return self.smm_objective(params, targets, model_moments=moments)
            except Exception:
                log.warning("calibration.simulation_failed", params=params)
                return 1e10  # Penalty for failed simulation

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": self.tolerance,
                "fatol": self.tolerance,
                "maxiter": self.max_iterations,
                "adaptive": True,
            },
        )

        calibrated = dict(zip(param_names, result.x, strict=True))
        # Project to bounds
        for name, val in calibrated.items():
            if name in effective_bounds:
                lo, hi = effective_bounds[name]
                calibrated[name] = float(np.clip(val, lo, hi))

        log.info(
            "calibration.completed",
            success=result.success,
            objective=round(result.fun, 8),
            iterations=result.nit,
            params={k: round(v, 6) for k, v in calibrated.items()},
        )

        return calibrated

    def format_comparison(
        self,
        model_moments: dict[str, float],
        targets: CalibrationTargets,
    ) -> str:
        """Format a human-readable comparison of model moments vs targets.

        Args:
            model_moments: Model-implied moment values.
            targets: Calibration target values.

        Returns:
            Markdown-formatted comparison string.
        """
        target_dict = {
            "wealth_gini": targets.wealth_gini,
            "entrepreneur_share": targets.entrepreneur_share,
            "top10_wealth_share": targets.top10_wealth_share,
        }

        lines = [
            "| Moment | Target | Model | Deviation |",
            "| --- | --- | --- | --- |",
        ]

        for name, target in target_dict.items():
            model_val = model_moments.get(name, 0.0)
            dev = model_val - target if target is not None else 0.0
            target_str = f"{target:.4f}" if target is not None else "N/A"
            lines.append(f"| {name} | {target_str} | {model_val:.4f} | {dev:+.4f} |")

        # Phase 3 moments (show as N/A if not available)
        if targets.median_mpc is not None:
            model_mpc = model_moments.get("median_mpc", 0.0)
            lines.append(
                f"| median_mpc | {targets.median_mpc:.4f} "
                f"| {model_mpc:.4f} | {model_mpc - targets.median_mpc:+.4f} |"
            )

        if targets.liquid_illiquid_ratio is not None:
            model_ratio = model_moments.get("liquid_illiquid_ratio", 0.0)
            lines.append(
                f"| liquid_illiquid_ratio | {targets.liquid_illiquid_ratio:.4f} "
                f"| {model_ratio:.4f} "
                f"| {model_ratio - targets.liquid_illiquid_ratio:+.4f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Compute the Gini coefficient from an array of values.

        Uses the sorted-values formula consistent with ObserverV2._compute_gini.

        Args:
            values: 1D array of non-negative values.

        Returns:
            Gini coefficient in [0, 1].
        """
        n = len(values)
        if n < 2:
            return 0.0
        sorted_v = np.sort(values)
        total = sorted_v.sum()
        if total <= 0.0:
            return 0.0
        indices = np.arange(1, n + 1, dtype=np.float64)
        weighted_sum = (indices * sorted_v).sum()
        return float((2.0 * weighted_sum - (n + 1) * total) / (n * total))

    @staticmethod
    def _compute_top_pct_share(values: np.ndarray, pct: float) -> float:
        """Compute the share of total wealth held by the top ``pct`` fraction.

        Args:
            values: 1D array of non-negative wealth values.
            pct: Fraction defining the top group (e.g. 0.10 for top 10%).

        Returns:
            Share of total wealth held by the top group, in [0, 1].
        """
        n = len(values)
        if n == 0:
            return 0.0
        total = values.sum()
        if total <= 0.0:
            return 0.0
        sorted_v = np.sort(values)
        # Number of agents in the top group (at least 1)
        top_count = max(1, int(np.ceil(n * pct)))
        top_wealth = sorted_v[-top_count:].sum()
        return float(top_wealth / total)
