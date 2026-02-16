"""Political utility functions — microfounded political preferences.

Implements REQ-401 through REQ-405: unified political-economic utility that
enters the Bellman equation as a constant flow bonus, and Bellman-derived
proposal evaluation for structurally consistent political decisions.

Traceability: REQ-401, REQ-402, REQ-403, REQ-404, REQ-405.
Design reference: spec/design.md section 4.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from emergent_constitution.models.constitution import ConstitutionV2
    from emergent_constitution.models.household import HouseholdState

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Core political utility (REQ-401, REQ-402)
# ---------------------------------------------------------------------------


def _effective_tax_rate(constitution: ConstitutionV2) -> float:
    """Extract the effective flat tax rate from the constitution.

    Searches for tax_schedule rules and returns the average rate parameter.
    Falls back to 0.0 if no tax rules exist.

    Args:
        constitution: The active constitutional ruleset.

    Returns:
        Effective tax rate in [0, 1].
    """
    tax_rules = constitution.get_tax_rules()
    if not tax_rules:
        return 0.0
    rates = [r.parameters.get("rate", 0.0) for r in tax_rules]
    avg_rate = sum(rates) / len(rates)
    return max(0.0, min(1.0, avg_rate))


def _effective_gini_proxy(constitution: ConstitutionV2) -> float:
    """Estimate the Gini impact from constitutional redistribution rules.

    Uses the constitutional structure as a proxy for inequality: higher tax
    rates and more redistribution imply lower Gini. This is a structural
    proxy since actual Gini requires simulation state.

    The proxy is: Gini_proxy = 1 - (tax_rate * redistribution_intensity)
    where redistribution_intensity depends on transfer rules.

    Args:
        constitution: The active constitutional ruleset.

    Returns:
        Gini proxy in [0, 1]. Lower means more equal.
    """
    tax_rate = _effective_tax_rate(constitution)

    # Check transfer rules for redistribution intensity
    transfer_rules = constitution.get_transfer_rules()
    redistribution_intensity = 0.0
    if transfer_rules:
        for rule in transfer_rules:
            method = rule.parameters.get("method", "equal_share")
            if method == "means_tested":
                redistribution_intensity = max(redistribution_intensity, 1.0)
            elif method == "equal_share":
                redistribution_intensity = max(redistribution_intensity, 0.5)
            else:
                redistribution_intensity = max(redistribution_intensity, 0.3)

    # Proxy: higher tax + redistribution -> lower Gini
    # Base Gini of ~0.4 reduced by policy
    gini_proxy = max(0.0, 0.4 - tax_rate * redistribution_intensity * 0.4)
    return gini_proxy


def compute_political_utility(
    constitution: ConstitutionV2,
    theta_eq: float,
    theta_lib: float,
    observed_gini: float | None = None,
) -> float:
    """Compute the political utility v(C; theta) for an agent.

    REQ-402: v(C; theta) = theta_eq * f_eq(C) + theta_lib * f_lib(C)
    where f_eq(C) = -Gini(C) and f_lib(C) = -tau(C).

    When observed_gini is provided (from the Observer), it is used directly.
    Otherwise, a structural proxy is computed from the constitution.

    Args:
        constitution: The active constitutional ruleset.
        theta_eq: Agent's equality preference weight (from ValueVector).
        theta_lib: Agent's liberty preference weight (from ValueVector).
        observed_gini: Optional observed Gini coefficient from simulation.

    Returns:
        Political utility value (higher is better from agent's perspective).
    """
    # f_eq(C) = -Gini(C): equality payoff (lower Gini = higher utility)
    gini = observed_gini if observed_gini is not None else _effective_gini_proxy(constitution)
    f_eq = -gini

    # f_lib(C) = -tau(C): liberty payoff (lower tax = higher utility)
    tau = _effective_tax_rate(constitution)
    f_lib = -tau

    v = theta_eq * f_eq + theta_lib * f_lib

    log.debug(
        "political_utility.computed",
        theta_eq=theta_eq,
        theta_lib=theta_lib,
        gini=gini,
        tau=tau,
        f_eq=f_eq,
        f_lib=f_lib,
        v=v,
    )

    return v


# ---------------------------------------------------------------------------
# Bellman-derived proposal evaluation (REQ-403, REQ-404, REQ-405)
# ---------------------------------------------------------------------------


def evaluate_proposal(
    value_function: list[list[float]],
    a_grid: list[float],
    agent: HouseholdState,
    proposed_constitution: ConstitutionV2,
    current_constitution: ConstitutionV2,
    political_lambda: float = 0.05,
    observed_gini: float | None = None,
    proposed_gini: float | None = None,
) -> float:
    """Evaluate a proposal by comparing Bellman values under both constitutions.

    REQ-403: Returns V(s; C_proposed) - V(s; C_current), where V includes
    both economic and political utility components.

    Since the constitution is fixed between governance periods, the political
    utility difference acts as a constant shift to the value function:
        Delta_V = lambda / (1 - beta) * [v(C_proposed) - v(C_current)]
    plus any change in the economic value from altered tax/transfer policies.

    For the full Bellman comparison, we would need to re-solve the household
    problem under both constitutions. As an approximation, we use:
    1. The current value function V(a, z; C_current) directly
    2. The political utility difference as the constitutional shift

    Args:
        value_function: V(a, z) array, shape [n_a][n_z].
        a_grid: Asset grid points.
        agent: The household agent state.
        proposed_constitution: The proposed constitutional rules.
        current_constitution: The current constitutional rules.
        political_lambda: Weight on political utility (lambda).
        observed_gini: Current observed Gini (used for current constitution).
        proposed_gini: Projected Gini under the proposed constitution.

    Returns:
        V(s; C_proposed) - V(s; C_current). Positive means the proposal
        is preferred.
    """
    if not value_function or not a_grid:
        log.warning("political_utility.no_value_function")
        return 0.0

    theta_eq = agent.value_vector.equality
    theta_lib = agent.value_vector.liberty
    beta_discount = agent.utility_params.beta_discount

    # Political utility under current and proposed constitutions
    v_current = compute_political_utility(
        current_constitution, theta_eq, theta_lib, observed_gini=observed_gini
    )
    v_proposed = compute_political_utility(
        proposed_constitution, theta_eq, theta_lib, observed_gini=proposed_gini
    )

    # The political utility difference, capitalized as a perpetuity
    # Since v(C) is constant between governance periods:
    # Delta_political = lambda * (v_proposed - v_current) / (1 - beta)
    denom = max(1.0 - beta_discount, 1e-10)
    delta_political = political_lambda * (v_proposed - v_current) / denom

    # For a full Bellman comparison, we would re-solve under C_proposed.
    # As an approximation, the economic value difference comes from the
    # tax rate change affecting disposable income:
    tau_current = _effective_tax_rate(current_constitution)
    tau_proposed = _effective_tax_rate(proposed_constitution)
    tau_delta = tau_proposed - tau_current

    # Economic value change from tax rate shift:
    # Approximate as: Delta_V_econ ~ -tau_delta * income / (1 - beta)
    # (higher tax reduces disposable income and thus value)
    income = agent.income if agent.income > 0 else agent.productivity * 0.5
    delta_economic = -tau_delta * income / denom

    total_delta = delta_political + delta_economic

    log.debug(
        "political_utility.proposal_evaluated",
        agent_id=agent.id,
        v_pol_current=v_current,
        v_pol_proposed=v_proposed,
        delta_political=delta_political,
        delta_economic=delta_economic,
        total_delta=total_delta,
    )

    return total_delta


def bellman_vote(
    value_function: list[list[float]],
    a_grid: list[float],
    agent: HouseholdState,
    proposed_constitution: ConstitutionV2,
    current_constitution: ConstitutionV2,
    political_lambda: float = 0.05,
    observed_gini: float | None = None,
) -> bool:
    """Determine a vote based purely on Bellman value comparison (REQ-405).

    Args:
        value_function: V(a, z) array.
        a_grid: Asset grid points.
        agent: The household agent state.
        proposed_constitution: The proposed constitutional rules.
        current_constitution: The current constitutional rules.
        political_lambda: Weight on political utility.
        observed_gini: Current observed Gini.

    Returns:
        True if agent supports the proposal (Delta_V > 0), False otherwise.
    """
    delta = evaluate_proposal(
        value_function=value_function,
        a_grid=a_grid,
        agent=agent,
        proposed_constitution=proposed_constitution,
        current_constitution=current_constitution,
        political_lambda=political_lambda,
        observed_gini=observed_gini,
    )
    return delta > 0.0


def build_bellman_political_context(
    value_function: list[list[float]],
    a_grid: list[float],
    agent: HouseholdState,
    proposed_constitution: ConstitutionV2,
    current_constitution: ConstitutionV2,
    political_lambda: float = 0.05,
    observed_gini: float | None = None,
) -> dict[str, object]:
    """Build structured context for LLM with Bellman-derived preferences (REQ-404).

    Provides the LLM with:
    - The Bellman-derived preference (support/oppose)
    - The magnitude of the value function difference
    - The breakdown into political and economic components

    Args:
        value_function: V(a, z) array.
        a_grid: Asset grid points.
        agent: The household agent state.
        proposed_constitution: Proposed rules.
        current_constitution: Current rules.
        political_lambda: Weight on political utility.
        observed_gini: Current observed Gini.

    Returns:
        Dict with Bellman-derived political context for LLM prompts.
    """
    theta_eq = agent.value_vector.equality
    theta_lib = agent.value_vector.liberty

    v_current = compute_political_utility(
        current_constitution, theta_eq, theta_lib, observed_gini=observed_gini
    )
    v_proposed = compute_political_utility(
        proposed_constitution, theta_eq, theta_lib, observed_gini=None
    )

    delta_v = evaluate_proposal(
        value_function=value_function,
        a_grid=a_grid,
        agent=agent,
        proposed_constitution=proposed_constitution,
        current_constitution=current_constitution,
        political_lambda=political_lambda,
        observed_gini=observed_gini,
    )

    bellman_recommendation = "support" if delta_v > 0 else "oppose"

    return {
        "bellman_recommendation": bellman_recommendation,
        "value_delta": round(delta_v, 6),
        "political_utility_current": round(v_current, 6),
        "political_utility_proposed": round(v_proposed, 6),
        "tax_rate_current": round(_effective_tax_rate(current_constitution), 4),
        "tax_rate_proposed": round(_effective_tax_rate(proposed_constitution), 4),
        "note": (
            "The Bellman recommendation is based on your value function "
            "which accounts for both economic welfare and political preferences. "
            "You may override this if you have strategic or coalition reasons."
        ),
    }


def log_llm_bellman_deviation(
    agent_id: str,
    bellman_recommendation: str,
    llm_vote: bool,
    delta_v: float,
) -> None:
    """Log when the LLM overrides the Bellman recommendation (REQ-404).

    Args:
        agent_id: Agent identifier.
        bellman_recommendation: "support" or "oppose".
        llm_vote: The LLM's actual vote (True=for, False=against).
        delta_v: The Bellman value difference.
    """
    bellman_vote_bool = bellman_recommendation == "support"
    if llm_vote != bellman_vote_bool:
        log.warning(
            "political_utility.llm_deviation",
            agent_id=agent_id,
            bellman_recommendation=bellman_recommendation,
            llm_vote=llm_vote,
            delta_v=delta_v,
        )
    else:
        log.debug(
            "political_utility.llm_aligned",
            agent_id=agent_id,
            bellman_recommendation=bellman_recommendation,
            delta_v=delta_v,
        )
