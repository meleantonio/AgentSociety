"""Tests for agent snapshot recording in observer.

Verifies that ``record_agent_snapshots=True`` populates
``household_snapshots`` in HistoryEntryV2 with the correct agent count,
and that default (False) leaves the field as None.
"""

from __future__ import annotations

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.constitution import ConstitutionV2, create_default_constitution
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import PeriodState
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.observer import ObserverV2


def _make_household(agent_id: str, wealth: float = 100.0) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=1.0,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=OccupationalRole.WORKER,
        consumption=10.0,
        leisure=0.3,
        labor_supply=0.7,
        realized_utility=1.0,
    )


def _make_period_state(num_agents: int = 20, period: int = 5) -> PeriodState:
    households = [
        _make_household(f"agent_{i:04d}", wealth=50.0 + i * 5) for i in range(num_agents)
    ]
    constitution = create_default_constitution()
    return PeriodState(
        period=period,
        households=households,
        firms=[],
        market=MarketState(
            wage=1.0,
            interest_rate=0.05,
            aggregate_output=100.0,
            aggregate_investment=10.0,
        ),
        shocks=ShockState(aggregate_tfp=1.0),
        constitution=constitution,
    )


class TestAgentSnapshots:
    """Tests for the record_agent_snapshots config flag."""

    def test_snapshots_off_by_default(self) -> None:
        """Default config leaves household_snapshots as None."""
        config = SimulationConfigV2(num_agents=20, observer_interval=5)
        observer = ObserverV2(config)
        ps = _make_period_state()
        entry = observer.observe(ps)
        assert entry.household_snapshots is None

    def test_snapshots_on_populates_data(self) -> None:
        """record_agent_snapshots=True stores household data."""
        config = SimulationConfigV2(
            num_agents=20,
            observer_interval=5,
            record_agent_snapshots=True,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        entry = observer.observe(ps)
        assert entry.household_snapshots is not None
        assert len(entry.household_snapshots) == 20

    def test_snapshot_contains_correct_ids(self) -> None:
        """Snapshot agent IDs match the period state households."""
        config = SimulationConfigV2(
            num_agents=20,
            observer_interval=5,
            record_agent_snapshots=True,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        entry = observer.observe(ps)
        snapshot_ids = {h.id for h in entry.household_snapshots}
        expected_ids = {f"agent_{i:04d}" for i in range(20)}
        assert snapshot_ids == expected_ids

    def test_snapshots_are_deep_copies(self) -> None:
        """Modifying source households after observe doesn't affect snapshots."""
        config = SimulationConfigV2(
            num_agents=20,
            observer_interval=5,
            record_agent_snapshots=True,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        entry = observer.observe(ps)
        original_wealth = entry.household_snapshots[0].wealth

        # Mutate the source
        ps.households[0].wealth = 99999.0

        assert entry.household_snapshots[0].wealth == original_wealth

    def test_config_field_defaults_to_false(self) -> None:
        """The record_agent_snapshots field defaults to False."""
        config = SimulationConfigV2(num_agents=20)
        assert config.record_agent_snapshots is False
