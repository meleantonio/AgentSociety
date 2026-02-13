"""Shock generators: Rouwenhorst discretization and shock drawing.

Implements REQ-015 (idiosyncratic shocks), REQ-016 (aggregate TFP),
REQ-017 (preference shocks) per spec/design.md section 3.2.
"""

from __future__ import annotations

import math

from emergent_constitution.models.household import HouseholdState
from emergent_constitution.rng import SimulationRNG

# ============================================================================
# Rouwenhorst discretization (spec section 4.2)
# ============================================================================


def rouwenhorst_discretize(
    rho: float,
    sigma: float,
    n_states: int,
) -> tuple[list[float], list[list[float]]]:
    """Discretize AR(1) process into Markov chain using Rouwenhorst method.

    Implements the algorithm from Kopecky & Suen (2010) as described in
    spec/design.md section 4.2.

    Args:
        rho: Persistence parameter (0 < rho < 1).
        sigma: Innovation volatility (sigma > 0).
        n_states: Number of grid points (>= 2).

    Returns:
        Tuple of (grid_values, transition_matrix) where grid_values are
        exp(z) levels and transition_matrix is row-stochastic.
    """
    if n_states < 2:
        msg = f"n_states must be >= 2, got {n_states}"
        raise ValueError(msg)
    if not (0.0 < rho < 1.0):
        msg = f"rho must be in (0, 1), got {rho}"
        raise ValueError(msg)
    if sigma <= 0.0:
        msg = f"sigma must be > 0, got {sigma}"
        raise ValueError(msg)

    # Step 1: unconditional std dev
    sigma_z = sigma / math.sqrt(1.0 - rho**2)

    # Step 2: grid bounds
    z_max = sigma_z * math.sqrt(n_states - 1)

    # Step 3: log grid (equally spaced)
    step = 2.0 * z_max / (n_states - 1)
    log_grid = [-z_max + i * step for i in range(n_states)]

    # Step 4: transition probability parameter
    p = (1.0 + rho) / 2.0

    # Step 5: build transition matrix recursively
    # Base case: 2x2
    trans = [[p, 1.0 - p], [1.0 - p, p]]

    for n in range(3, n_states + 1):
        prev = trans
        new_trans = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                # Four quadrant contributions
                if i < n - 1 and j < n - 1:
                    new_trans[i][j] += p * prev[i][j]
                if i < n - 1 and j > 0:
                    new_trans[i][j] += (1.0 - p) * prev[i][j - 1]
                if i > 0 and j < n - 1:
                    new_trans[i][j] += (1.0 - p) * prev[i - 1][j]
                if i > 0 and j > 0:
                    new_trans[i][j] += p * prev[i - 1][j - 1]

        # Normalize rows to sum to 1
        for i in range(n):
            row_sum = sum(new_trans[i])
            if row_sum > 0.0:
                new_trans[i] = [x / row_sum for x in new_trans[i]]

        trans = new_trans

    # Step 6: convert from log to level
    grid_values = [math.exp(z) for z in log_grid]

    return grid_values, trans


def stationary_distribution(
    transition_matrix: list[list[float]],
) -> list[float]:
    """Compute stationary distribution of a Markov chain.

    Uses power iteration: start from uniform, multiply by P^T repeatedly.

    Args:
        transition_matrix: Row-stochastic Markov transition matrix.

    Returns:
        Stationary distribution as a list of probabilities.
    """
    n = len(transition_matrix)
    dist = [1.0 / n] * n

    for _ in range(1000):
        new_dist = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_dist[j] += dist[i] * transition_matrix[i][j]
        # Check convergence
        max_diff = max(abs(new_dist[k] - dist[k]) for k in range(n))
        dist = new_dist
        if max_diff < 1e-12:
            break

    # Normalize
    total = sum(dist)
    return [d / total for d in dist]


# ============================================================================
# Shock drawing functions
# ============================================================================


def _draw_from_cdf(rng: SimulationRNG, probabilities: list[float]) -> int:
    """Draw an index from a discrete probability distribution.

    Args:
        rng: Seeded RNG instance.
        probabilities: Probability of each index (must sum to ~1).

    Returns:
        Drawn index.
    """
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if u < cumulative:
            return i
    return len(probabilities) - 1


def draw_idiosyncratic_shocks(
    households: list[HouseholdState],
    transition_matrix: list[list[float]],
    grid: list[float],
    rng: SimulationRNG,
) -> list[HouseholdState]:
    """Transition each agent's productivity_index via Markov chain.

    For each household, draw the next productivity state using the
    transition probabilities from their current state. Returns new
    HouseholdState objects with updated productivity fields.

    Args:
        households: Current household states.
        transition_matrix: Markov transition matrix (row-stochastic).
        grid: Productivity grid values (exp(z) levels).
        rng: Seeded RNG instance.

    Returns:
        List of HouseholdState with updated productivity and productivity_index.

    Implements REQ-015 (idiosyncratic shocks).
    """
    updated: list[HouseholdState] = []
    for h in households:
        row = transition_matrix[h.productivity_index]
        new_idx = _draw_from_cdf(rng, row)
        new_productivity = grid[new_idx]
        updated.append(
            h.model_copy(
                update={
                    "productivity_index": new_idx,
                    "productivity": new_productivity,
                }
            )
        )
    return updated


def draw_entrepreneurial_ability_shocks(
    households: list[HouseholdState],
    transition_matrix: list[list[float]],
    grid: list[float],
    rng: SimulationRNG,
) -> list[HouseholdState]:
    """Transition each agent's entrepreneurial_ability_index via Markov chain.

    Same pattern as draw_idiosyncratic_shocks but using the ability Markov chain.

    Args:
        households: Current household states.
        transition_matrix: Ability Markov transition matrix (row-stochastic).
        grid: Ability grid values (exp(z) levels).
        rng: Seeded RNG instance.

    Returns:
        List of HouseholdState with updated entrepreneurial_ability and
        entrepreneurial_ability_index.
    """
    updated: list[HouseholdState] = []
    for h in households:
        row = transition_matrix[h.entrepreneurial_ability_index]
        new_idx = _draw_from_cdf(rng, row)
        new_ability = grid[new_idx]
        updated.append(
            h.model_copy(
                update={
                    "entrepreneurial_ability_index": new_idx,
                    "entrepreneurial_ability": new_ability,
                }
            )
        )
    return updated


def draw_aggregate_tfp(
    prev_log_a: float,
    rho_a: float,
    sigma_a: float,
    rng: SimulationRNG,
) -> float:
    """Draw aggregate TFP from AR(1) process in logs.

    log(A_t) = rho_A * log(A_{t-1}) + epsilon,
    epsilon ~ N(0, sigma_A^2)

    Args:
        prev_log_a: log(A_{t-1}).
        rho_a: Aggregate TFP persistence (0 < rho_a < 1).
        sigma_a: Aggregate TFP innovation volatility.
        rng: Seeded RNG instance.

    Returns:
        A_t = exp(log(A_t)), guaranteed > 0.

    Implements REQ-016 (aggregate shocks).
    """
    epsilon = rng.gauss(0.0, sigma_a)
    log_a_t = rho_a * prev_log_a + epsilon
    return math.exp(log_a_t)


def draw_preference_shocks(
    households: list[HouseholdState],
    rng: SimulationRNG,
    sigma: float,
) -> dict[str, float]:
    """Draw preference shocks for each agent.

    Each agent receives a perturbation to their discount factor drawn
    from N(0, sigma^2). The perturbation is additive; the caller is
    responsible for clamping beta_discount to valid range.

    Args:
        households: Current household states.
        rng: Seeded RNG instance.
        sigma: Standard deviation of the preference shock.

    Returns:
        Dict mapping agent_id -> beta perturbation.

    Implements REQ-017 (preference shocks).
    """
    shocks: dict[str, float] = {}
    for h in households:
        perturbation = rng.gauss(0.0, sigma)
        shocks[h.id] = perturbation
    return shocks
