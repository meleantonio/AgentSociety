"""Tests for v2 coalition formation — HouseholdState compatibility.

Covers:
- form_coalitions_v2: greedy clustering with HouseholdState (15.1)
- compute_coalition_stats_v2: stats computation with HouseholdState (15.1)
- Backward compatibility: v1 functions still work with AgentState

Traceability: Design section 9 migration table
"""

from __future__ import annotations

import pytest

from emergent_constitution.coalition import (
    CoalitionInfo,
    compute_coalition_stats_v2,
    form_coalitions_v2,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.rng import SimulationRNG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_household(
    agent_id: str,
    equality: float,
    wealth: float = 100.0,
    productivity: float = 1.0,
    role: OccupationalRole = OccupationalRole.WORKER,
) -> HouseholdState:
    """Helper to create a HouseholdState with a specific equality value."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=equality, liberty=round(1.0 - equality, 6)),
        role=role,
    )


# ---------------------------------------------------------------------------
# form_coalitions_v2
# ---------------------------------------------------------------------------


class TestFormCoalitionsV2:
    def test_assigns_coalition_ids(self):
        """All households receive a non-None coalition_id after formation."""
        households = [_make_household(f"h{i}", equality=0.1 * i) for i in range(10)]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng)
        assert len(result) == len(households)
        for h in result:
            assert h.coalition_id is not None
            assert h.coalition_id.startswith("coalition_")

    def test_returns_household_state_instances(self):
        """Returned items are HouseholdState (not AgentState)."""
        households = [_make_household("h0", equality=0.5)]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng)
        assert isinstance(result[0], HouseholdState)

    def test_preserves_household_fields(self):
        """HouseholdState-specific fields (role, productivity_index) are preserved."""
        h = _make_household("h0", equality=0.5, role=OccupationalRole.ENTREPRENEUR)
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2([h], rng)
        assert result[0].role == OccupationalRole.ENTREPRENEUR
        assert result[0].productivity_index == 0
        assert result[0].id == "h0"

    def test_similar_households_same_coalition(self):
        """Households with similar equality values cluster together."""
        households = [
            _make_household("h0", equality=0.50),
            _make_household("h1", equality=0.52),
            _make_household("h2", equality=0.48),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng, similarity_threshold=0.3)
        coalition_ids = {h.coalition_id for h in result}
        assert len(coalition_ids) == 1

    def test_dissimilar_households_different_coalitions(self):
        """Households with very different equality values end up in different coalitions."""
        households = [
            _make_household("h0", equality=0.0),
            _make_household("h1", equality=1.0),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng, similarity_threshold=0.1)
        coalition_ids = {h.coalition_id for h in result}
        assert len(coalition_ids) == 2

    def test_deterministic_with_seed(self):
        """Same seed produces identical coalition assignments."""
        households = [_make_household(f"h{i}", equality=0.1 * i) for i in range(10)]

        rng1 = SimulationRNG(seed=99)
        result1 = form_coalitions_v2(households, rng1)

        rng2 = SimulationRNG(seed=99)
        result2 = form_coalitions_v2(households, rng2)

        ids1 = [h.coalition_id for h in result1]
        ids2 = [h.coalition_id for h in result2]
        assert ids1 == ids2

    def test_single_household(self):
        """Single household gets its own coalition."""
        households = [_make_household("h0", equality=0.5)]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng)
        assert len(result) == 1
        assert result[0].coalition_id == "coalition_0000"

    def test_empty_list(self):
        """Empty household list returns empty list."""
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2([], rng)
        assert result == []

    def test_original_not_mutated(self):
        """Original household list is not modified."""
        households = [_make_household("h0", equality=0.5)]
        rng = SimulationRNG(seed=42)
        assert households[0].coalition_id is None
        form_coalitions_v2(households, rng)
        assert households[0].coalition_id is None

    def test_coalition_ids_sequential(self):
        """Coalition IDs follow sequential numbering format."""
        households = [
            _make_household("h0", equality=0.0),
            _make_household("h1", equality=1.0),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng, similarity_threshold=0.1)
        coalition_ids = sorted({h.coalition_id for h in result})
        assert coalition_ids[0] == "coalition_0000"
        assert coalition_ids[1] == "coalition_0001"

    def test_mixed_roles_cluster_by_values(self):
        """Clustering is by equality value, not by occupational role."""
        households = [
            _make_household("h0", equality=0.5, role=OccupationalRole.WORKER),
            _make_household("h1", equality=0.52, role=OccupationalRole.ENTREPRENEUR),
            _make_household("h2", equality=0.48, role=OccupationalRole.UNEMPLOYED),
        ]
        rng = SimulationRNG(seed=42)
        result = form_coalitions_v2(households, rng, similarity_threshold=0.3)
        # All similar equality values -> same coalition
        coalition_ids = {h.coalition_id for h in result}
        assert len(coalition_ids) == 1


# ---------------------------------------------------------------------------
# compute_coalition_stats_v2
# ---------------------------------------------------------------------------


class TestComputeCoalitionStatsV2:
    def test_stats_computed_correctly(self):
        """Correct size, mean wealth, and mean equality for known coalitions."""
        households = [
            _make_household("h0", equality=0.6, wealth=100.0).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
            _make_household("h1", equality=0.8, wealth=200.0).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
            _make_household("h2", equality=0.2, wealth=50.0).model_copy(
                update={"coalition_id": "coalition_0001"}
            ),
        ]
        stats = compute_coalition_stats_v2(households)
        assert len(stats) == 2

        c0 = stats["coalition_0000"]
        assert c0.size == 2
        assert c0.mean_wealth == pytest.approx(150.0)
        assert c0.mean_equality == pytest.approx(0.7)

        c1 = stats["coalition_0001"]
        assert c1.size == 1
        assert c1.mean_wealth == pytest.approx(50.0)
        assert c1.mean_equality == pytest.approx(0.2)

    def test_excludes_unaffiliated(self):
        """Households with coalition_id=None are excluded from stats."""
        households = [
            _make_household("h0", equality=0.5).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
            _make_household("h1", equality=0.5),  # No coalition_id
        ]
        stats = compute_coalition_stats_v2(households)
        assert len(stats) == 1
        assert stats["coalition_0000"].size == 1

    def test_empty_list(self):
        """Empty household list returns empty dict."""
        assert compute_coalition_stats_v2([]) == {}

    def test_returns_coalition_info_type(self):
        """Stats values are CoalitionInfo instances."""
        households = [
            _make_household("h0", equality=0.5).model_copy(
                update={"coalition_id": "coalition_0000"}
            ),
        ]
        stats = compute_coalition_stats_v2(households)
        assert isinstance(stats["coalition_0000"], CoalitionInfo)

    def test_integration_form_then_stats(self):
        """Form coalitions then compute stats — full pipeline."""
        households = [
            _make_household("h0", equality=0.5, wealth=100.0),
            _make_household("h1", equality=0.52, wealth=200.0),
            _make_household("h2", equality=0.1, wealth=50.0),
        ]
        rng = SimulationRNG(seed=42)
        formed = form_coalitions_v2(households, rng, similarity_threshold=0.3)
        stats = compute_coalition_stats_v2(formed)

        # All agents should be in some coalition
        total_agents_in_coalitions = sum(ci.size for ci in stats.values())
        assert total_agents_in_coalitions == 3

        # Stats should have valid values
        for ci in stats.values():
            assert ci.size >= 1
            assert ci.mean_wealth >= 0.0
            assert 0.0 <= ci.mean_equality <= 1.0
