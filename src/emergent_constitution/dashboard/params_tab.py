"""Tab 1: Simulation parameter form."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.dashboard.runner import run_simulation


def render(key_prefix: str = "params") -> None:
    """Render the parameter form and run-simulation button.

    Args:
        key_prefix: Widget key prefix to avoid collisions.
    """
    with st.form("simulation_params"):
        st.subheader("Simulation Parameters")

        # --- Core (always visible) ---
        c1, c2, c3 = st.columns(3)
        num_agents = c1.number_input("Agents", 20, 500, 50, key=f"{key_prefix}_n")
        max_periods = c2.number_input("Periods", 1, 1000, 100, key=f"{key_prefix}_t")
        seed = c3.number_input("Seed", 0, 999999, 42, key=f"{key_prefix}_seed")

        # --- Shock Parameters ---
        with st.expander("Shock Parameters"):
            s1, s2 = st.columns(2)
            rho_z = s1.number_input("rho_z (idio. persistence)", 0.01, 0.99, 0.9, 0.01)
            sigma_z = s2.number_input("sigma_z (idio. volatility)", 0.01, 2.0, 0.2, 0.01)
            num_z_states = s1.number_input("num_z_states", 2, 50, 7)
            rho_a = s2.number_input("rho_a (agg. TFP persist.)", 0.01, 0.99, 0.95, 0.01)
            sigma_a = s1.number_input("sigma_a (agg. TFP vol.)", 0.001, 1.0, 0.01, 0.001)

        # --- Production ---
        with st.expander("Production"):
            p1, p2 = st.columns(2)
            alpha = p1.number_input("alpha (capital share)", 0.01, 0.99, 0.33, 0.01)
            delta = p2.number_input("delta (depreciation)", 0.01, 0.99, 0.1, 0.01)
            min_firm_capital = p1.number_input("min_firm_capital", 0.1, 100.0, 10.0, 0.1)

        # --- Entrepreneurial ---
        with st.expander("Entrepreneurial"):
            e1, e2 = st.columns(2)
            rho_e = e1.number_input("rho_e", 0.01, 0.99, 0.85, 0.01)
            sigma_e = e2.number_input("sigma_e", 0.01, 2.0, 0.3, 0.01)
            firm_entry_cost = e1.number_input("firm_entry_cost", 0.1, 100.0, 5.0, 0.1)
            use_bellman_occ = e2.checkbox("use_bellman_occ_choice", value=False)

        # --- Household ---
        with st.expander("Household"):
            h1, h2 = st.columns(2)
            a_min = h1.number_input("a_min (borrowing limit)", 0.0, 100.0, 0.0, 0.1)
            iw_mean = h2.number_input("initial_wealth_mean", 1.0, 1000.0, 100.0, 1.0)
            iw_std = h1.number_input("initial_wealth_std", 0.0, 500.0, 30.0, 1.0)

        # --- Preferences ---
        with st.expander("Preferences"):
            homogeneous = st.checkbox("homogeneous_preferences", value=True)
            pr1, pr2, pr3 = st.columns(3)
            u_alpha = pr1.number_input("utility_alpha", 0.01, 0.99, 0.4, 0.01)
            u_beta = pr2.number_input("utility_beta", 0.01, 0.99, 0.35, 0.01)
            u_gamma = pr3.number_input("utility_gamma", 0.01, 0.99, 0.25, 0.01)
            u_discount = st.number_input("utility_beta_discount", 0.01, 0.99, 0.95, 0.01)

            weight_sum = u_alpha + u_beta + u_gamma
            if abs(weight_sum - 1.0) > 1e-6:
                st.warning(
                    f"Utility weights sum to {weight_sum:.4f} (should be 1.0). "
                    "Validation will fail if homogeneous_preferences is enabled."
                )

        # --- Market Clearing ---
        with st.expander("Market Clearing"):
            mc_method = st.selectbox(
                "market_clearing_method", ["analytical", "walrasian"], index=0
            )
            mc1, mc2 = st.columns(2)
            tat_max = mc1.number_input("tatonnement_max_iter", 1, 10000, 100)
            tat_tol = mc2.number_input("tatonnement_tolerance", 1e-12, 1e-2, 1e-6, format="%.2e")
            tat_step = mc1.number_input("tatonnement_step_size", 0.001, 1.0, 0.01, 0.001)

        # --- Execution Mode ---
        with st.expander("Execution Mode"):
            benchmark_mode = st.checkbox("benchmark_mode", value=True)
            solver_method = st.selectbox("solver_method", ["egm", "vfi", "vfi_numpy"], index=0)

        # --- LLM Settings ---
        with st.expander("LLM Settings"):
            use_llm = st.checkbox("use_llm", value=False)
            llm1, llm2 = st.columns(2)
            llm_provider = llm1.text_input("llm_provider", value="local")
            llm_model = llm2.text_input("llm_model", value="lmstudio-community/gpt-oss-20b-GGUF")
            llm_base_url = st.text_input("llm_base_url", value="http://localhost:1234/v1")
            llm_temp = st.number_input("llm_temperature", 0.0, 2.0, 0.0, 0.1)

        # --- Intervals ---
        with st.expander("Intervals"):
            i1, i2 = st.columns(2)
            prop_interval = i1.number_input("proposal_interval", 1, 100, 5)
            obs_interval = i2.number_input("observer_interval", 1, 100, 5)

        # --- Government / Fiscal ---
        with st.expander("Government / Fiscal"):
            g1, g2 = st.columns(2)
            init_debt = g1.number_input("initial_debt", 0.0, 1000.0, 0.0, 1.0)
            debt_gdp = g2.number_input("debt_gdp_max", 0.1, 10.0, 1.5, 0.1)
            fisc_adj = g1.number_input("fiscal_rule_adjustment", 0.001, 0.5, 0.01, 0.001)

        # --- Two-Asset HANK ---
        with st.expander("Two-Asset HANK"):
            two_asset = st.checkbox("two_asset_mode", value=False)
            ta1, ta2 = st.columns(2)
            chi_0 = ta1.number_input("chi_0", 0.0, 1.0, 0.01, 0.001)
            chi_1 = ta2.number_input("chi_1", 0.0, 1.0, 0.005, 0.001)
            b_min = ta1.number_input("b_min", -10.0, 10.0, 0.0, 0.1)

        # --- Nominal Rigidities ---
        with st.expander("Nominal Rigidities"):
            nominal_rig = st.checkbox("nominal_rigidities", value=False)
            n1, n2 = st.columns(2)
            rot_cost = n1.number_input("rotemberg_cost", 0.0, 1000.0, 100.0, 1.0)
            tay_pi = n2.number_input("taylor_phi_pi", 1.01, 5.0, 1.5, 0.01)
            tay_y = n1.number_input("taylor_phi_y", 0.0, 2.0, 0.125, 0.01)
            inf_target = n2.number_input("inflation_target", -0.05, 0.2, 0.02, 0.005)

        # --- Political Utility ---
        with st.expander("Political Utility"):
            pol_lambda = st.slider("political_lambda", 0.0, 1.0, 0.05, 0.01)
            pure_bellman = st.checkbox("pure_bellman_politics", value=False)

        # --- R&D ---
        with st.expander("R&D"):
            r1, r2 = st.columns(2)
            rd_prob = r1.number_input("rd_success_base_prob", 0.0, 1.0, 0.1, 0.01)
            rd_mean = r2.number_input("rd_tfp_improvement_mean", 0.001, 1.0, 0.05, 0.001)
            rd_std = r1.number_input("rd_tfp_improvement_std", 0.0, 1.0, 0.02, 0.001)

        submitted = st.form_submit_button(
            "Run Simulation", type="primary", use_container_width=True
        )

    if submitted:
        try:
            config = SimulationConfigV2(
                num_agents=int(num_agents),
                max_periods=int(max_periods),
                seed=int(seed),
                rho_z=float(rho_z),
                sigma_z=float(sigma_z),
                num_z_states=int(num_z_states),
                rho_a=float(rho_a),
                sigma_a=float(sigma_a),
                rho_e=float(rho_e),
                sigma_e=float(sigma_e),
                firm_entry_cost=float(firm_entry_cost),
                use_bellman_occ_choice=use_bellman_occ,
                alpha=float(alpha),
                delta=float(delta),
                min_firm_capital=float(min_firm_capital),
                a_min=float(a_min),
                initial_wealth_mean=float(iw_mean),
                initial_wealth_std=float(iw_std),
                homogeneous_preferences=homogeneous,
                utility_alpha=float(u_alpha),
                utility_beta=float(u_beta),
                utility_gamma=float(u_gamma),
                utility_beta_discount=float(u_discount),
                market_clearing_method=mc_method,
                tatonnement_max_iter=int(tat_max),
                tatonnement_tolerance=float(tat_tol),
                tatonnement_step_size=float(tat_step),
                benchmark_mode=benchmark_mode,
                solver_method=solver_method,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_temperature=float(llm_temp),
                proposal_interval=int(prop_interval),
                observer_interval=int(obs_interval),
                initial_debt=float(init_debt),
                debt_gdp_max=float(debt_gdp),
                fiscal_rule_adjustment=float(fisc_adj),
                two_asset_mode=two_asset,
                chi_0=float(chi_0),
                chi_1=float(chi_1),
                b_min=float(b_min),
                nominal_rigidities=nominal_rig,
                rotemberg_cost=float(rot_cost),
                taylor_phi_pi=float(tay_pi),
                taylor_phi_y=float(tay_y),
                inflation_target=float(inf_target),
                political_lambda=float(pol_lambda),
                pure_bellman_politics=pure_bellman,
                rd_success_base_prob=float(rd_prob),
                rd_tfp_improvement_mean=float(rd_mean),
                rd_tfp_improvement_std=float(rd_std),
            )
        except ValidationError as exc:
            st.error(f"Configuration validation failed:\n\n{exc}")
            return

        st.session_state["sim_config"] = config
        output = run_simulation(config)
        st.session_state["sim_output"] = output
        st.rerun()
