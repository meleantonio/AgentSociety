"""Unit tests for calibration module — CalibrationTargets, Calibrator, SMM.

Tests: moment computation correctness, SMM objective gradient direction,
CalibrationTargets validation, backward compatibility.

Traceability: REQ-208, REQ-209, REQ-210.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from scipy.optimize import minimize  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from emergent_constitution.calibration import (
    _DEFAULT_WEIGHTS,
    _PARAM_NAMES,
    CalibrationTargets,
    Calibrator,
)
from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.constitution import create_default_constitution
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
from emergent_constitution.reporter import generate_report_v2

# ============================================================================
# Helpers
# ============================================================================


def _make_household(
    agent_id: str,
    wealth: float = 100.0,
    role: OccupationalRole = OccupationalRole.WORKER,
    consumption: float = 10.0,
    leisure: float = 0.3,
    realized_utility: float = 1.0,
) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=1.0,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2, beta_discount=0.95),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
        consumption=consumption,
        leisure=leisure,
        labor_supply=1.0 - leisure,
        realized_utility=realized_utility,
    )


def _make_period_state(
    period: int = 1,
    households: list[HouseholdState] | None = None,
    constitution=None,
) -> PeriodState:
    if households is None:
        households = [_make_household(f"agent_{i:04d}") for i in range(10)]
    if constitution is None:
        constitution = create_default_constitution()
    return PeriodState(
        period=period,
        households=households,
        firms=[],
        market=MarketState(
            wage=1.0,
            interest_rate=0.04,
            aggregate_output=100.0,
            aggregate_investment=20.0,
        ),
        shocks=ShockState(
            productivity_grid=[0.5, 1.0, 1.5],
            transition_matrix=[[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
        ),
        constitution=constitution,
    )


# ============================================================================
# Tests: CalibrationTargets
# ============================================================================


class TestCalibrationTargets:
    """Test CalibrationTargets dataclass."""

    def test_default_values(self) -> None:
        """Default targets match spec values."""
        targets = CalibrationTargets()
        assert targets.wealth_gini == 0.80
        assert targets.entrepreneur_share == 0.10
        assert targets.top10_wealth_share == 0.70
        assert targets.median_mpc is None
        assert targets.liquid_illiquid_ratio is None

    def test_custom_values(self) -> None:
        """Custom targets are stored correctly."""
        targets = CalibrationTargets(
            wealth_gini=0.85,
            entrepreneur_share=0.15,
            top10_wealth_share=0.65,
            median_mpc=0.25,
            liquid_illiquid_ratio=0.30,
        )
        assert targets.wealth_gini == 0.85
        assert targets.entrepreneur_share == 0.15
        assert targets.top10_wealth_share == 0.65
        assert targets.median_mpc == 0.25
        assert targets.liquid_illiquid_ratio == 0.30

    def test_phase3_fields_optional(self) -> None:
        """Phase 3 fields are optional (None)."""
        targets = CalibrationTargets(wealth_gini=0.75)
        assert targets.median_mpc is None
        assert targets.liquid_illiquid_ratio is None


# ============================================================================
# Tests: Moment computation
# ============================================================================


class TestComputeModelMoments:
    """Test Calibrator.compute_model_moments()."""

    def test_equal_wealth_gini_zero(self) -> None:
        """All agents have equal wealth -> Gini = 0."""
        agents = [_make_household(f"a{i}", wealth=100.0) for i in range(10)]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["wealth_gini"] == pytest.approx(0.0, abs=1e-10)

    def test_known_gini(self) -> None:
        """Known wealth distribution -> correct Gini.

        For values [0, 0, 0, 0, 100], Gini should be high.
        """
        agents = [_make_household(f"a{i}", wealth=0.0) for i in range(4)]
        agents.append(_make_household("a4", wealth=100.0))
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        # With 4 zeros and 1 value, Gini = (2*5*100 - 6*100)/(5*100) = 0.8
        assert moments["wealth_gini"] == pytest.approx(0.8, abs=0.01)

    def test_entrepreneur_share(self) -> None:
        """Mixed roles -> correct entrepreneur share."""
        agents = [_make_household(f"a{i}", role=OccupationalRole.WORKER) for i in range(8)]
        agents.extend(
            [_make_household(f"e{i}", role=OccupationalRole.ENTREPRENEUR) for i in range(2)]
        )
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["entrepreneur_share"] == pytest.approx(0.2)

    def test_no_entrepreneurs(self) -> None:
        """All workers -> entrepreneur share = 0."""
        agents = [_make_household(f"a{i}", role=OccupationalRole.WORKER) for i in range(10)]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["entrepreneur_share"] == pytest.approx(0.0)

    def test_top10_share_equal_wealth(self) -> None:
        """Equal wealth -> top 10% share = 10%."""
        agents = [_make_household(f"a{i}", wealth=100.0) for i in range(100)]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["top10_wealth_share"] == pytest.approx(0.10, abs=0.01)

    def test_top10_share_concentrated(self) -> None:
        """All wealth in one agent -> top 10% share high."""
        agents = [_make_household(f"a{i}", wealth=0.0) for i in range(9)]
        agents.append(_make_household("a9", wealth=1000.0))
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["top10_wealth_share"] == pytest.approx(1.0, abs=0.01)

    def test_empty_agents(self) -> None:
        """Empty agent list -> zero moments."""
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments([])
        assert moments["wealth_gini"] == 0.0
        assert moments["entrepreneur_share"] == 0.0
        assert moments["top10_wealth_share"] == 0.0

    def test_single_agent(self) -> None:
        """Single agent -> Gini 0, no entrepreneur."""
        agents = [_make_household("a0", wealth=100.0)]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["wealth_gini"] == 0.0
        assert moments["top10_wealth_share"] == pytest.approx(1.0)


# ============================================================================
# Tests: Gini consistency with ObserverV2
# ============================================================================


class TestGiniConsistency:
    """Verify Gini from Calibrator matches ObserverV2."""

    def test_gini_matches_observer(self) -> None:
        """Calibrator Gini should match ObserverV2 Gini for same data."""
        wealths = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # ObserverV2 Gini
        observer_gini = ObserverV2._compute_gini(wealths)
        # Calibrator Gini
        calibrator = Calibrator()
        calibrator_gini = calibrator._compute_gini(np.array(wealths))
        assert calibrator_gini == pytest.approx(observer_gini, abs=1e-10)


# ============================================================================
# Tests: SMM objective
# ============================================================================


class TestSMMObjective:
    """Test Calibrator.smm_objective()."""

    def test_zero_distance_at_target(self) -> None:
        """When model moments = targets, objective = 0."""
        targets = CalibrationTargets(
            wealth_gini=0.80,
            entrepreneur_share=0.10,
            top10_wealth_share=0.70,
        )
        model_moments = {
            "wealth_gini": 0.80,
            "entrepreneur_share": 0.10,
            "top10_wealth_share": 0.70,
        }
        calibrator = Calibrator()
        obj = calibrator.smm_objective({}, targets, model_moments=model_moments)
        assert obj == pytest.approx(0.0, abs=1e-10)

    def test_positive_distance_away_from_target(self) -> None:
        """When model moments != targets, objective > 0."""
        targets = CalibrationTargets(
            wealth_gini=0.80,
            entrepreneur_share=0.10,
            top10_wealth_share=0.70,
        )
        model_moments = {
            "wealth_gini": 0.50,
            "entrepreneur_share": 0.20,
            "top10_wealth_share": 0.50,
        }
        calibrator = Calibrator()
        obj = calibrator.smm_objective({}, targets, model_moments=model_moments)
        assert obj > 0.0

    def test_gradient_direction_gini(self) -> None:
        """Moving Gini toward target reduces objective."""
        targets = CalibrationTargets(wealth_gini=0.80)
        calibrator = Calibrator()

        # Far from target
        m_far = {"wealth_gini": 0.50, "entrepreneur_share": 0.10, "top10_wealth_share": 0.70}
        # Closer to target
        m_near = {"wealth_gini": 0.75, "entrepreneur_share": 0.10, "top10_wealth_share": 0.70}

        obj_far = calibrator.smm_objective({}, targets, model_moments=m_far)
        obj_near = calibrator.smm_objective({}, targets, model_moments=m_near)
        assert obj_near < obj_far

    def test_gradient_direction_entrepreneur(self) -> None:
        """Moving entrepreneur share toward target reduces objective."""
        targets = CalibrationTargets(entrepreneur_share=0.10)
        calibrator = Calibrator()

        m_far = {"wealth_gini": 0.80, "entrepreneur_share": 0.30, "top10_wealth_share": 0.70}
        m_near = {"wealth_gini": 0.80, "entrepreneur_share": 0.15, "top10_wealth_share": 0.70}

        obj_far = calibrator.smm_objective({}, targets, model_moments=m_far)
        obj_near = calibrator.smm_objective({}, targets, model_moments=m_near)
        assert obj_near < obj_far

    def test_custom_weights(self) -> None:
        """Custom weights change objective magnitude."""
        targets = CalibrationTargets(wealth_gini=0.80)
        model_moments = {
            "wealth_gini": 0.50,
            "entrepreneur_share": 0.10,
            "top10_wealth_share": 0.70,
        }

        low_weights = {
            "wealth_gini": 0.1,
            "entrepreneur_share": 1.0,
            "top10_wealth_share": 1.0,
        }
        high_weights = {
            "wealth_gini": 10.0,
            "entrepreneur_share": 1.0,
            "top10_wealth_share": 1.0,
        }
        cal_low = Calibrator(weights=low_weights)
        cal_high = Calibrator(weights=high_weights)

        obj_low = cal_low.smm_objective({}, targets, model_moments=model_moments)
        obj_high = cal_high.smm_objective({}, targets, model_moments=model_moments)
        assert obj_high > obj_low

    def test_requires_moments_or_simulate_fn(self) -> None:
        """Raises ValueError when neither model_moments nor simulate_fn given."""
        targets = CalibrationTargets()
        calibrator = Calibrator()
        with pytest.raises(ValueError, match="Either model_moments or simulate_fn"):
            calibrator.smm_objective({}, targets)

    def test_simulate_fn_called(self) -> None:
        """simulate_fn is called when model_moments is None."""
        targets = CalibrationTargets(wealth_gini=0.80)
        called = {"count": 0}

        def mock_simulate(params):
            called["count"] += 1
            return {"wealth_gini": 0.80, "entrepreneur_share": 0.10, "top10_wealth_share": 0.70}

        calibrator = Calibrator()
        obj = calibrator.smm_objective({"beta_d": 0.95}, targets, simulate_fn=mock_simulate)
        assert called["count"] == 1
        assert obj == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Tests: SMM calibrate()
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestSMMCalibrate:
    """Test Calibrator.calibrate() optimization."""

    def test_calibrate_finds_known_solution(self) -> None:
        """When simulate_fn returns target moments at a known param, calibrate converges."""
        targets = CalibrationTargets(
            wealth_gini=0.80,
            entrepreneur_share=0.10,
            top10_wealth_share=0.70,
        )

        # Simulate function: moments depend on beta_d
        # At beta_d=0.95 -> exact target
        def simulate(params):
            beta = params.get("beta_d", 0.95)
            return {
                "wealth_gini": 0.80 + (beta - 0.95) * 2.0,
                "entrepreneur_share": 0.10 + (beta - 0.95) * 0.5,
                "top10_wealth_share": 0.70 + (beta - 0.95) * 1.0,
            }

        initial = {"beta_d": 0.90}
        calibrator = Calibrator(tolerance=1e-8, max_iterations=200)
        result = calibrator.calibrate(
            initial,
            targets,
            simulate,
            bounds={"beta_d": (0.80, 0.999)},
        )
        assert abs(result["beta_d"] - 0.95) < 0.02

    def test_calibrate_respects_bounds(self) -> None:
        """Calibrated parameters stay within bounds."""
        targets = CalibrationTargets()

        def simulate(params):
            return {
                "wealth_gini": 0.50,
                "entrepreneur_share": 0.05,
                "top10_wealth_share": 0.50,
            }

        initial = {"beta_d": 0.95, "sigma_z": 0.2}
        bounds = {"beta_d": (0.85, 0.99), "sigma_z": (0.05, 0.5)}
        calibrator = Calibrator(max_iterations=50)
        result = calibrator.calibrate(initial, targets, simulate, bounds=bounds)
        assert 0.85 <= result["beta_d"] <= 0.99
        assert 0.05 <= result["sigma_z"] <= 0.50

    def test_calibrate_handles_simulation_failure(self) -> None:
        """Calibrate continues when simulate_fn raises."""
        targets = CalibrationTargets()
        call_count = {"n": 0}

        def sometimes_fails(params):
            call_count["n"] += 1
            if call_count["n"] % 3 == 0:
                raise RuntimeError("Simulation failed")
            return {
                "wealth_gini": 0.80,
                "entrepreneur_share": 0.10,
                "top10_wealth_share": 0.70,
            }

        initial = {"beta_d": 0.95}
        calibrator = Calibrator(max_iterations=50)
        result = calibrator.calibrate(
            initial,
            targets,
            sometimes_fails,
            bounds={"beta_d": (0.80, 0.999)},
        )
        assert "beta_d" in result


# ============================================================================
# Tests: format_comparison
# ============================================================================


class TestFormatComparison:
    """Test Calibrator.format_comparison()."""

    def test_contains_all_moments(self) -> None:
        """Output contains all three core moments."""
        targets = CalibrationTargets()
        model_moments = {
            "wealth_gini": 0.75,
            "entrepreneur_share": 0.12,
            "top10_wealth_share": 0.65,
        }
        calibrator = Calibrator()
        text = calibrator.format_comparison(model_moments, targets)
        assert "wealth_gini" in text
        assert "entrepreneur_share" in text
        assert "top10_wealth_share" in text

    def test_contains_table_headers(self) -> None:
        """Output contains Markdown table headers."""
        targets = CalibrationTargets()
        model_moments = {
            "wealth_gini": 0.80,
            "entrepreneur_share": 0.10,
            "top10_wealth_share": 0.70,
        }
        calibrator = Calibrator()
        text = calibrator.format_comparison(model_moments, targets)
        assert "| Moment |" in text
        assert "| Target |" in text
        assert "| Model |" in text

    def test_phase3_moments_shown_when_present(self) -> None:
        """Phase 3 moments shown when targets are provided."""
        targets = CalibrationTargets(median_mpc=0.25, liquid_illiquid_ratio=0.30)
        model_moments = {
            "wealth_gini": 0.80,
            "entrepreneur_share": 0.10,
            "top10_wealth_share": 0.70,
            "median_mpc": 0.20,
            "liquid_illiquid_ratio": 0.25,
        }
        calibrator = Calibrator()
        text = calibrator.format_comparison(model_moments, targets)
        assert "median_mpc" in text
        assert "liquid_illiquid_ratio" in text


# ============================================================================
# Tests: Config integration
# ============================================================================


class TestConfigIntegration:
    """Test calibration_targets in SimulationConfigV2."""

    def test_default_no_targets(self) -> None:
        """Default config has no calibration targets."""
        config = SimulationConfigV2(num_agents=20, benchmark_mode=True)
        assert config.calibration_targets is None

    def test_config_with_targets(self) -> None:
        """Config accepts CalibrationTargets."""
        targets = CalibrationTargets(wealth_gini=0.85)
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            calibration_targets=targets,
        )
        assert config.calibration_targets is not None
        assert config.calibration_targets.wealth_gini == 0.85


# ============================================================================
# Tests: Observer integration (REQ-210)
# ============================================================================


class TestObserverCalibrationIntegration:
    """Test that ObserverV2 logs calibration moments when targets are set."""

    def test_observer_logs_moments_with_targets(self) -> None:
        """Observer logs calibration moments when config has targets."""
        targets = CalibrationTargets()
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
            calibration_targets=targets,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        # Should not raise
        entry = observer.observe(ps)
        assert entry.gini >= 0.0

    def test_observer_no_error_without_targets(self) -> None:
        """Observer works normally without calibration targets."""
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        entry = observer.observe(ps)
        assert entry.gini >= 0.0


# ============================================================================
# Tests: Reporter integration (REQ-210)
# ============================================================================


class TestReporterCalibrationIntegration:
    """Test that generate_report_v2 includes calibration when targets given."""

    def test_report_includes_calibration_section(self) -> None:
        """Report includes calibration comparison when targets provided."""
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
        )
        observer = ObserverV2(config)
        ps = _make_period_state(period=1)
        observer.observe(ps)
        output = observer.finalize(ps)

        targets = CalibrationTargets()
        report = generate_report_v2(output, calibration_targets=targets)
        assert "Calibration Comparison" in report
        assert "wealth_gini" in report

    def test_report_without_calibration(self) -> None:
        """Report works without calibration targets (backward compat)."""
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
        )
        observer = ObserverV2(config)
        ps = _make_period_state(period=1)
        observer.observe(ps)
        output = observer.finalize(ps)

        report = generate_report_v2(output)
        assert "Calibration Comparison" not in report
        assert "## Wealth Distribution" in report


# ============================================================================
# Tests: Backward compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test that calibration module doesn't break existing functionality."""

    def test_observer_v2_without_calibration(self) -> None:
        """ObserverV2 works identically without calibration targets."""
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
        )
        households = [_make_household(f"a{i}", wealth=float((i + 1) * 10)) for i in range(10)]
        ps = _make_period_state(households=households)

        observer = ObserverV2(config)
        entry = observer.observe(ps)

        # Standard statistics should be computed correctly
        assert entry.gini > 0.0
        assert entry.mean_wealth > 0.0
        assert entry.median_wealth > 0.0

    def test_reporter_v2_backward_compat(self) -> None:
        """generate_report_v2 without targets produces same output as before."""
        config = SimulationConfigV2(
            num_agents=20,
            benchmark_mode=True,
            observer_interval=1,
        )
        observer = ObserverV2(config)
        ps = _make_period_state()
        observer.observe(ps)
        output = observer.finalize(ps)

        report = generate_report_v2(output)
        # Verify standard sections are present
        assert "Simulation Report (v2)" in report
        assert "Final Constitution" in report
        assert "Wealth Distribution" in report
        assert "Statistics Over Time" in report

    def test_param_names_match_spec(self) -> None:
        """Calibration parameter names match spec design.md."""
        assert "beta_d" in _PARAM_NAMES
        assert "sigma_z" in _PARAM_NAMES
        assert "entry_cost" in _PARAM_NAMES
        assert "a_min" in _PARAM_NAMES

    def test_default_weights_defined(self) -> None:
        """Default weights cover all three core moments."""
        assert "wealth_gini" in _DEFAULT_WEIGHTS
        assert "entrepreneur_share" in _DEFAULT_WEIGHTS
        assert "top10_wealth_share" in _DEFAULT_WEIGHTS


# ============================================================================
# Tests: Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases in calibration computation."""

    def test_all_zero_wealth(self) -> None:
        """All agents with zero wealth -> Gini 0, top 10% share 0."""
        agents = [_make_household(f"a{i}", wealth=0.0) for i in range(10)]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert moments["wealth_gini"] == 0.0
        assert moments["top10_wealth_share"] == 0.0

    def test_very_large_wealth_disparity(self) -> None:
        """Extreme wealth disparity doesn't cause numerical issues."""
        agents = [_make_household(f"a{i}", wealth=0.01) for i in range(99)]
        agents.append(_make_household("a99", wealth=1e8))
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert 0.0 <= moments["wealth_gini"] <= 1.0
        assert 0.0 <= moments["top10_wealth_share"] <= 1.0

    def test_two_agents(self) -> None:
        """Two agents produce valid moments."""
        agents = [
            _make_household("a0", wealth=10.0),
            _make_household("a1", wealth=90.0),
        ]
        calibrator = Calibrator()
        moments = calibrator.compute_model_moments(agents)
        assert 0.0 < moments["wealth_gini"] < 1.0
        assert moments["top10_wealth_share"] == pytest.approx(0.9)
