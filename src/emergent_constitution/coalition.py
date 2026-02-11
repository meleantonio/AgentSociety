"""Coalition formation — greedy clustering by value vector similarity.

All functions are pure: they take state and return results without
mutating any inputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from emergent_constitution.models.agent import AgentState
from emergent_constitution.rng import SimulationRNG


class CoalitionInfo(BaseModel):
    """Aggregate statistics for a single coalition.

    Args:
        coalition_id: Unique coalition identifier.
        size: Number of agents in the coalition.
        mean_wealth: Average wealth of coalition members.
        mean_equality: Average equality preference of coalition members.
    """

    coalition_id: str
    size: int = Field(ge=1)
    mean_wealth: float = Field(ge=0.0)
    mean_equality: float = Field(ge=0.0, le=1.0)


def form_coalitions(
    agents: list[AgentState],
    rng: SimulationRNG,
    similarity_threshold: float = 0.3,
) -> list[AgentState]:
    """Assign agents to coalitions based on value vector similarity.

    Algorithm:
        1. Compute pairwise value distance (|eq_i - eq_j|) for all agent pairs.
        2. Agents with distance < similarity_threshold can form a coalition.
        3. Use greedy clustering: iterate agents in rng-shuffled order,
           assign to existing coalition if similar to coalition centroid,
           or create new coalition.
        4. Return new agent list with updated coalition_id fields.

    Coalition IDs are formatted as "coalition_XXXX" with sequential numbering.

    Args:
        agents: Current agent states (not mutated).
        rng: Seeded RNG for deterministic shuffling.
        similarity_threshold: Maximum value distance for coalition membership.

    Returns:
        New list of AgentState copies with updated coalition_id fields.
    """
    if not agents:
        return []

    # Shuffle agent order for fairness (deterministic via seeded RNG)
    order = list(range(len(agents)))
    rng.shuffle(order)

    # Track coalitions: coalition_id -> list of equality values (centroid computation)
    coalition_members: dict[str, list[float]] = {}
    # Map agent index -> assigned coalition_id
    agent_coalition: dict[int, str] = {}
    next_coalition_num = 0

    for idx in order:
        agent_eq = agents[idx].value_vector.equality
        assigned = False

        # Try to join an existing coalition (check centroid distance)
        for cid, eq_values in coalition_members.items():
            centroid = sum(eq_values) / len(eq_values)
            if abs(agent_eq - centroid) < similarity_threshold:
                coalition_members[cid].append(agent_eq)
                agent_coalition[idx] = cid
                assigned = True
                break

        # Create a new coalition if no existing one is similar enough
        if not assigned:
            cid = f"coalition_{next_coalition_num:04d}"
            next_coalition_num += 1
            coalition_members[cid] = [agent_eq]
            agent_coalition[idx] = cid

    # Build new agent list with updated coalition_ids, preserving original order
    return [
        agents[i].model_copy(update={"coalition_id": agent_coalition[i]})
        for i in range(len(agents))
    ]


def compute_coalition_stats(agents: list[AgentState]) -> dict[str, CoalitionInfo]:
    """Compute statistics for each coalition.

    Args:
        agents: Agent states with coalition_id fields populated.

    Returns:
        Mapping of coalition_id to CoalitionInfo. Agents with coalition_id=None
        are excluded.
    """
    # Group agents by coalition
    groups: dict[str, list[AgentState]] = {}
    for agent in agents:
        if agent.coalition_id is not None:
            groups.setdefault(agent.coalition_id, []).append(agent)

    stats: dict[str, CoalitionInfo] = {}
    for cid, members in sorted(groups.items()):
        size = len(members)
        mean_wealth = sum(a.wealth for a in members) / size
        mean_equality = sum(a.value_vector.equality for a in members) / size
        stats[cid] = CoalitionInfo(
            coalition_id=cid,
            size=size,
            mean_wealth=mean_wealth,
            mean_equality=mean_equality,
        )

    return stats
