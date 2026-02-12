"""Unit tests for v2 SimulationConfigV2: defaults, validation, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emergent_constitution.config import SimulationConfigV2


class TestSimulationConfigV2Defaults:
    def test_defaults_valid(self) -> None:
        config = SimulationConfigV2()
        assert config.num_agents == 50
        assert config.max_periods == 100
        assert config.seed == 42
        assert config.alpha == 0.33
        assert config.a_min == 0.0
        assert config.use_llm is True
        assert config.benchmark_mode is False
        assert config.solver_method == "egm"

    def test_shock_defaults(self) -> None:
        config = SimulationConfigV2()
        assert config.rho_z == 0.9
        assert config.sigma_z == 0.2
        assert config.num_z_states == 5
        assert config.rho_a == 0.95
        assert config.sigma_a == 0.01
        assert config.enable_preference_shocks is False

    def test_market_clearing_defaults(self) -> None:
        config = SimulationConfigV2()
        assert config.tatonnement_max_iter == 100
        assert config.tatonnement_tolerance == 1e-6
        assert config.tatonnement_step_size == 0.01

    def test_llm_defaults(self) -> None:
        config = SimulationConfigV2()
        assert config.llm_provider == "anthropic"
        assert config.llm_temperature == 0.0
        assert config.llm_batch_size == 10
        assert config.llm_cache_enabled is True

    def test_rd_defaults(self) -> None:
        config = SimulationConfigV2()
        assert config.rd_success_base_prob == 0.1
        assert config.rd_tfp_improvement_mean == 0.05
        assert config.rd_tfp_improvement_std == 0.02


class TestSimulationConfigV2Validation:
    def test_num_agents_too_small(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(num_agents=19)

    def test_num_agents_zero(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(num_agents=0)

    def test_max_periods_zero(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(max_periods=0)

    def test_alpha_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(alpha=1.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(alpha=0.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(alpha=1.5)

    def test_delta_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(delta=0.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(delta=1.0)

    def test_rho_z_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(rho_z=0.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(rho_z=1.0)

    def test_sigma_z_not_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(sigma_z=0.0)

    def test_num_z_states_too_small(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(num_z_states=1)

    def test_num_z_states_too_large(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(num_z_states=51)

    def test_a_min_negative(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(a_min=-1.0)

    def test_tatonnement_tolerance_not_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(tatonnement_tolerance=0.0)

    def test_llm_temperature_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(llm_temperature=2.5)
        with pytest.raises(ValidationError):
            SimulationConfigV2(llm_temperature=-0.1)

    def test_llm_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(llm_batch_size=0)

    def test_solver_method_invalid(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(solver_method="invalid")

    def test_solver_method_vfi(self) -> None:
        config = SimulationConfigV2(solver_method="vfi")
        assert config.solver_method == "vfi"

    def test_rd_success_prob_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(rd_success_base_prob=1.5)
        with pytest.raises(ValidationError):
            SimulationConfigV2(rd_success_base_prob=-0.1)

    def test_initial_wealth_mean_not_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(initial_wealth_mean=0.0)

    def test_min_firm_capital_not_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(min_firm_capital=0.0)


class TestSimulationConfigV2BenchmarkMode:
    def test_benchmark_disables_llm(self) -> None:
        config = SimulationConfigV2(benchmark_mode=True, use_llm=True)
        assert config.benchmark_mode is True
        assert config.use_llm is False

    def test_benchmark_without_llm(self) -> None:
        config = SimulationConfigV2(benchmark_mode=True, use_llm=False)
        assert config.benchmark_mode is True
        assert config.use_llm is False

    def test_non_benchmark_keeps_llm(self) -> None:
        config = SimulationConfigV2(benchmark_mode=False, use_llm=True)
        assert config.use_llm is True


class TestSimulationConfigV2Serialization:
    def test_round_trip(self) -> None:
        config = SimulationConfigV2(
            num_agents=30,
            max_periods=50,
            seed=123,
            alpha=0.4,
            rho_z=0.85,
        )
        data = config.model_dump()
        config2 = SimulationConfigV2.model_validate(data)
        assert config2.num_agents == 30
        assert config2.max_periods == 50
        assert config2.seed == 123
        assert config2.alpha == 0.4
        assert config2.rho_z == 0.85

    def test_json_round_trip(self) -> None:
        config = SimulationConfigV2()
        json_str = config.model_dump_json()
        config2 = SimulationConfigV2.model_validate_json(json_str)
        assert config == config2

    def test_custom_config(self) -> None:
        config = SimulationConfigV2(
            num_agents=100,
            max_periods=200,
            seed=999,
            rho_z=0.95,
            sigma_z=0.3,
            num_z_states=7,
            alpha=0.36,
            delta=0.08,
            a_min=0.1,
            tatonnement_max_iter=200,
            use_llm=False,
            benchmark_mode=True,
            solver_method="vfi",
        )
        assert config.num_agents == 100
        assert config.num_z_states == 7
        assert config.use_llm is False
        assert config.solver_method == "vfi"


class TestHomogeneousPreferencesConfig:
    def test_defaults(self) -> None:
        config = SimulationConfigV2()
        assert config.homogeneous_preferences is True
        assert config.utility_alpha == 0.4
        assert config.utility_beta == 0.35
        assert config.utility_gamma == 0.25
        assert config.utility_beta_discount == 0.95

    def test_weights_sum_validated(self) -> None:
        """Weights that don't sum to 1.0 should raise when homogeneous=True."""
        with pytest.raises(ValidationError):
            SimulationConfigV2(
                homogeneous_preferences=True,
                utility_alpha=0.5,
                utility_beta=0.5,
                utility_gamma=0.5,
            )

    def test_weights_not_validated_when_heterogeneous(self) -> None:
        """When homogeneous_preferences=False, utility weights are not enforced."""
        config = SimulationConfigV2(
            homogeneous_preferences=False,
            utility_alpha=0.1,
            utility_beta=0.1,
            utility_gamma=0.1,
        )
        assert config.homogeneous_preferences is False

    def test_custom_weights(self) -> None:
        config = SimulationConfigV2(
            utility_alpha=0.5,
            utility_beta=0.3,
            utility_gamma=0.2,
            utility_beta_discount=0.98,
        )
        assert config.utility_alpha == 0.5
        assert config.utility_beta == 0.3
        assert config.utility_gamma == 0.2
        assert config.utility_beta_discount == 0.98

    def test_out_of_range_alpha(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(utility_alpha=0.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(utility_alpha=1.0)

    def test_out_of_range_discount(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfigV2(utility_beta_discount=0.0)
        with pytest.raises(ValidationError):
            SimulationConfigV2(utility_beta_discount=1.0)
