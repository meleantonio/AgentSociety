"""Lead (Simulation Governor) — the main tick/period loop.

v1 Lead retained for backward compat (tick-based simulation).
v2 LeadV2 implements the 9-step period lifecycle per spec/design.md section 3.1:
1. Draw shocks, 2. Clear markets, 3. Collect decisions,
4. Validate constraints, 5. Execute production, 6. Enforce constitution,
7. Process governance, 8. Update states, 9. Observe.

Traceability: REQ-033
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import structlog

from emergent_constitution.citizen import decide_proposal, decide_trade, decide_votes
from emergent_constitution.citizen_v2 import decide_proposal_v2, decide_votes_v2
from emergent_constitution.coalition import form_coalitions
from emergent_constitution.config import SimulationConfig, SimulationConfigV2
from emergent_constitution.constitution_engine import ConstitutionEngine
from emergent_constitution.distribution import Distribution
from emergent_constitution.economics import (
    apply_rd_shock,
    apply_trades,
    compute_budget,
    compute_realized_utility,
    distribute_firm_income,
    economic_step,
    enforce_budget_constraint,
    liquidate_firm,
    produce_output,
    validate_household_states,
)
from emergent_constitution.egm_solver import EGMSolver, adjustment_cost
from emergent_constitution.entrepreneurial_solver import EntrepreneurialSolver
from emergent_constitution.government import Government, clear_bond_market
from emergent_constitution.initialization import initialize_simulation, initialize_simulation_v2
from emergent_constitution.llm_citizen import CitizenLLM, PromptBuilder
from emergent_constitution.llm_engine import LLMDecisionEngine
from emergent_constitution.market_clearing import clear_markets
from emergent_constitution.models.agent import AgentState
from emergent_constitution.models.constitution import Constitution, ConstitutionV2
from emergent_constitution.models.decisions import (
    EconomicDecision,
    EntrepreneurialDecision,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import (
    HistoryEntry,
    PeriodState,
    SimulationOutput,
    SimulationOutputV2,
)
from emergent_constitution.models.household import HouseholdState, OccupationalRole
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import (
    ConstitutionalProposal,
    Proposal,
    TradeOffer,
    VoteOutcome,
    VoteOutcomeV2,
)
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.models.tick import TickState
from emergent_constitution.nominal import NominalBlock, NominalState
from emergent_constitution.numerical_solver import NumericalSolver
from emergent_constitution.observer import ObserverV2, observe_tick
from emergent_constitution.rng import SimulationRNG
from emergent_constitution.shock_generators import (
    draw_aggregate_tfp,
    draw_entrepreneurial_ability_shocks,
    draw_idiosyncratic_shocks,
    draw_preference_shocks,
)
from emergent_constitution.voting import (
    apply_passed_proposals,
    apply_passed_proposals_v2,
    tally_votes,
    tally_votes_v2,
    validate_proposal,
    validate_proposal_v2,
)

log = structlog.get_logger()


# ============================================================================
# v1 Lead (backward compatibility)
# ============================================================================


class Lead:
    """Simulation governor that runs the main tick loop.

    Args:
        config: Simulation configuration.
        citizen_llm: Optional LLM-based citizen reasoning implementation.
            When provided *and* ``config.use_llm`` is True, a fraction of
            agents (controlled by ``config.llm_fraction``) will use this
            interface for proposal generation and vote reasoning each tick.
    """

    def __init__(
        self,
        config: SimulationConfig,
        citizen_llm: CitizenLLM | None = None,
    ) -> None:
        self.config = config
        self.citizen_llm = citizen_llm
        self.tick_state: TickState
        self.rng: SimulationRNG
        self.history: list[HistoryEntry] = []
        self.last_observed_constitution: Constitution | None = None
        self.tick_state, self.rng = initialize_simulation(config)
        log.info("simulation.initialized", num_agents=config.num_agents, seed=config.seed)

    @property
    def _use_llm(self) -> bool:
        """Whether LLM reasoning is active for this run."""
        return self.config.use_llm and self.citizen_llm is not None

    def _select_llm_agents(self, agents: list[AgentState]) -> set[str]:
        """Select a subset of agent IDs to use LLM reasoning this tick.

        The subset size is ``ceil(len(agents) * config.llm_fraction)``,
        selected via seeded RNG for determinism.

        Args:
            agents: Current agent states.

        Returns:
            Set of agent IDs that should use LLM reasoning.
        """
        if not self._use_llm:
            return set()

        count = max(1, int(len(agents) * self.config.llm_fraction + 0.5))
        count = min(count, len(agents))

        # Deterministic selection: shuffle a copy and take first `count`
        ids = [a.id for a in agents]
        self.rng.shuffle(ids)
        return set(ids[:count])

    def run(self) -> SimulationOutput:
        """Execute all ticks and return the final simulation output.

        Returns:
            SimulationOutput with final constitution, history, and agent states.
        """
        log.info("simulation.started", max_ticks=self.config.max_ticks)

        for tick in range(1, self.config.max_ticks + 1):
            self.tick_state = self._advance_tick(tick)

        log.info("simulation.completed", total_ticks=self.config.max_ticks)
        return SimulationOutput(
            constitution=self.tick_state.constitution.model_copy(deep=True),
            history=list(self.history),
            final_agent_states=[a.model_copy(deep=True) for a in self.tick_state.agent_states],
            seed=self.config.seed,
            total_ticks=self.config.max_ticks,
        )

    def _advance_tick(self, tick: int) -> TickState:
        """Execute a single tick: proposals -> votes -> economics -> observe.

        Args:
            tick: Current tick number.

        Returns:
            New TickState for this tick.
        """
        current_agents = [a.model_copy(deep=True) for a in self.tick_state.agent_states]
        current_constitution = self.tick_state.constitution.model_copy(deep=True)

        # Phase 2: proposals and voting
        proposals = self._collect_proposals(tick, current_agents, current_constitution)
        votes = self._run_votes(proposals, current_agents, current_constitution)
        current_constitution = self._apply_vote_outcomes(votes, current_constitution)

        # Trade step
        trades = self._collect_trades(tick, current_agents, current_constitution)
        if trades:
            current_agents = apply_trades(current_agents, trades)

        # Economic step
        updated_agents = economic_step(current_agents, current_constitution)

        # Coalition formation (periodic)
        if tick % self.config.coalition_interval == 0:
            updated_agents = form_coalitions(updated_agents, self.rng)
            log.info(
                "coalitions.formed",
                tick=tick,
                num_coalitions=len(
                    {a.coalition_id for a in updated_agents if a.coalition_id is not None}
                ),
            )

        # Observation
        self._observe(tick, updated_agents, current_constitution)

        return TickState(
            tick=tick,
            agent_states=updated_agents,
            constitution=current_constitution,
            proposals_this_tick=proposals,
            votes=votes,
            trades_this_tick=trades,
        )

    def _collect_proposals(
        self,
        tick: int,
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[Proposal]:
        """Collect proposals from citizens on proposal-interval ticks.

        Shuffles agent order via self.rng for fairness. Each agent may produce
        at most one proposal, which is validated before inclusion.

        When LLM reasoning is active, a subset of agents (selected via
        ``_select_llm_agents``) use the ``CitizenLLM`` interface; the rest
        use the deterministic rule-based logic.

        Args:
            tick: Current tick number.
            agents: Current agent states (read-only copies).
            constitution: Current constitution.

        Returns:
            List of validated proposals.
        """
        if tick % self.config.proposal_interval != 0:
            return []

        llm_ids = self._select_llm_agents(agents)

        # Shuffle for fairness (deterministic via seeded RNG)
        agent_order = list(agents)
        self.rng.shuffle(agent_order)

        proposals: list[Proposal] = []
        for agent in agent_order:
            if agent.id in llm_ids and self.citizen_llm is not None:
                context = PromptBuilder.build_proposal_prompt(agent, constitution, agents)
                proposal = self.citizen_llm.generate_proposal(agent, constitution, context)
            else:
                proposal = decide_proposal(agent, constitution, agents, self.rng)

            if proposal is not None and validate_proposal(proposal):
                proposals.append(proposal)

        log.info(
            "proposals.collected",
            tick=tick,
            count=len(proposals),
            llm_agents=len(llm_ids),
        )
        return proposals

    def _collect_trades(
        self,
        tick: int,
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[TradeOffer]:
        """Collect bilateral trade offers from citizens on trade-interval ticks.

        Shuffles agent order via self.rng for fairness. Each agent may produce
        at most one trade offer.

        Args:
            tick: Current tick number.
            agents: Current agent states (read-only copies).
            constitution: Current constitution.

        Returns:
            List of trade offers.
        """
        if tick % self.config.trade_interval != 0:
            return []

        # Shuffle for fairness (deterministic via seeded RNG)
        agent_order = list(agents)
        self.rng.shuffle(agent_order)

        trades: list[TradeOffer] = []
        for agent in agent_order:
            trade = decide_trade(agent, agents, constitution, self.rng)
            if trade is not None:
                trades.append(trade)

        log.info("trades.collected", tick=tick, count=len(trades))
        return trades

    def _run_votes(
        self,
        proposals: list[Proposal],
        agents: list[AgentState],
        constitution: Constitution,
    ) -> list[VoteOutcome]:
        """Run votes on all proposals.

        Each agent votes on all proposals. When LLM reasoning is active,
        the selected subset uses the ``CitizenLLM`` interface per proposal;
        the rest use the deterministic rule-based logic.

        Args:
            proposals: Proposals to vote on.
            agents: Current agent states.
            constitution: Current constitution.

        Returns:
            List of vote outcomes.
        """
        if not proposals:
            return []

        total_eligible = len(agents)
        llm_ids = self._select_llm_agents(agents)

        # Collect each agent's votes on all proposals
        all_agent_votes: dict[str, dict[str, bool]] = {}
        for agent in agents:
            if agent.id in llm_ids and self.citizen_llm is not None:
                # LLM path: vote on each proposal individually
                votes_for_agent: dict[str, bool] = {}
                for proposal in proposals:
                    context = PromptBuilder.build_vote_prompt(
                        agent, constitution, proposal, agents
                    )
                    vote = self.citizen_llm.reason_vote(agent, constitution, proposal, context)
                    votes_for_agent[proposal.proposer_id] = vote
                all_agent_votes[agent.id] = votes_for_agent
            else:
                all_agent_votes[agent.id] = decide_votes(
                    agent,
                    constitution,
                    proposals,
                    agents,
                    self.rng,
                )

        # Tally per proposal
        outcomes: list[VoteOutcome] = []
        for proposal in proposals:
            proposal_votes: dict[str, bool] = {}
            for agent_id, agent_votes in all_agent_votes.items():
                if proposal.proposer_id in agent_votes:
                    proposal_votes[agent_id] = agent_votes[proposal.proposer_id]

            outcome = tally_votes(
                proposal,
                proposal_votes,
                constitution.voting_rule,
                total_eligible,
            )
            outcomes.append(outcome)
            log.debug(
                "vote.tallied",
                rule_key=proposal.rule_key,
                passed=outcome.passed,
                votes_for=outcome.votes_for,
                votes_against=outcome.votes_against,
            )

        return outcomes

    def _apply_vote_outcomes(
        self,
        votes: list[VoteOutcome],
        constitution: Constitution,
    ) -> Constitution:
        """Apply passed proposals to the constitution.

        Args:
            votes: Vote outcomes to process.
            constitution: Current constitution.

        Returns:
            Updated constitution with passed proposals applied.
        """
        return apply_passed_proposals(votes, constitution)

    def _observe(
        self,
        tick: int,
        agents: list[AgentState],
        constitution: Constitution,
    ) -> None:
        """Record observation statistics at observer-interval ticks.

        Args:
            tick: Current tick number.
            agents: Current agent states.
            constitution: Current constitution.
        """
        if tick % self.config.observer_interval != 0:
            return

        entry = observe_tick(tick, agents, constitution, self.last_observed_constitution)
        self.history.append(entry)
        self.last_observed_constitution = constitution.model_copy(deep=True)
        log.info(
            "observation.recorded",
            tick=tick,
            gini=round(entry.gini, 4),
            mean_wealth=round(entry.mean_wealth, 2),
            rule_changes=len(entry.rule_changes),
        )


# ============================================================================
# v2 LeadV2 — 9-step period lifecycle (DSGE-HA)
# ============================================================================


class LeadV2:
    """DSGE-HA simulation governor with 9-step period lifecycle.

    Implements the main simulation loop per spec/design.md section 3.1.
    Each period executes nine steps in order:
    1. Draw shocks (REQ-015, 016, 017)
    2. Clear markets (REQ-011..014)
    3. Collect decisions (REQ-024..026, 029)
    4. Validate constraints (REQ-004, 005, PROP-002, 004)
    5. Execute production (REQ-007..009)
    6. Enforce constitution (REQ-022)
    7. Process governance (REQ-019..021, 023)
    8. Update states (REQ-001, 003, 006, 010)
    9. Observe (REQ-030, 031)

    Args:
        config: v2 simulation configuration.

    Traceability: REQ-033
    """

    def __init__(self, config: SimulationConfigV2) -> None:
        self.config = config

        # Initialize RNG and period-0 state
        period_state, self.rng = initialize_simulation_v2(config)
        self.period_state: PeriodState = period_state

        # Constitution engine
        self.constitution_engine = ConstitutionEngine()

        # Decision engine: LLM or numerical solver.
        # pure_bellman_politics requires the political engine even when
        # benchmark mode is active or use_llm is False.
        if config.pure_bellman_politics or (config.use_llm and not config.benchmark_mode):
            # LLMDecisionEngine accepts duck-typed config (uses getattr for v2 fields)
            self._llm_engine: LLMDecisionEngine | None = LLMDecisionEngine(  # type: ignore[arg-type]
                config=config,
                rng=self.rng,
            )
        else:
            self._llm_engine = None
        self._solver = self._create_solver(config, period_state)

        # Entrepreneurial solver (used in both benchmark and LLM modes)
        self._entre_solver = EntrepreneurialSolver(
            config=config,
            ability_grid=period_state.shocks.ability_grid,
            ability_transition_matrix=period_state.shocks.ability_transition_matrix,
        )

        # KFE distribution state (Phase 2, REQ-201..205)
        self._distribution: Distribution | None = None
        if config.distribution_mode == "kfe":
            self._distribution = Distribution(
                a_grid=self._solver.a_grid,
                z_grid=self._solver.productivity_grid,
                kfe_tolerance=config.kfe_convergence_tolerance,
                kfe_max_iter=config.kfe_max_iterations,
            )
            self._distribution.initialize_uniform()

        # Observer (REQ-030, REQ-031, REQ-032)
        self.observer = ObserverV2(config)
        # Convenience alias — shares same list object as observer.history
        self.history = self.observer.history

        # Track log(A) for aggregate TFP AR(1) process
        self._prev_log_a: float = 0.0  # log(1.0) = 0

        # Firm ID counter
        self._next_firm_id: int = 0

        # Per-period entrepreneurial capital flows for stock-flow consistency.
        self._period_capital_debits: dict[str, float] = {}
        self._period_capital_credits: dict[str, float] = {}

        # Government sector (REQ-306..309)
        self._government = Government(
            initial_debt=config.initial_debt,
            debt_gdp_max=config.debt_gdp_max,
            fiscal_rule_adjustment=config.fiscal_rule_adjustment,
        )

        # Nominal rigidities block (REQ-310..315)
        if config.nominal_rigidities:
            self._nominal_block: NominalBlock | None = NominalBlock(config)
            self._nominal_state: NominalState = NominalState()
            # Track steady-state output for output gap computation
            self._steady_state_output: float = period_state.market.aggregate_output
        else:
            self._nominal_block = None
            self._nominal_state = NominalState()
            self._steady_state_output = period_state.market.aggregate_output

        # Track recent proposal outcomes for LLM governance context
        self._recent_rejections: list[tuple[str, str, str]] = []  # (agent_id, rule_name, reason)
        self._recent_vote_outcomes: list[dict] = []  # Structured vote outcome records

        log.info(
            "simulation_v2.initialized",
            num_agents=config.num_agents,
            seed=config.seed,
            benchmark_mode=config.benchmark_mode,
            use_llm=config.use_llm,
        )

    def run(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        observe_callback: Callable[[SimulationOutputV2], None] | None = None,
    ) -> SimulationOutputV2:
        """Execute all periods and return the final simulation output.

        Args:
            progress_callback: Optional callback invoked after each period with
                (current_period, total_periods). Used by the dashboard for
                progress bar updates.
            observe_callback: Optional callback invoked at each observer interval
                with a partial SimulationOutputV2 snapshot. Used by the dashboard
                to stream intermediate results to the UI.

        Returns:
            SimulationOutputV2 with final state, history, and welfare summary.
        """
        log.info("simulation_v2.started", max_periods=self.config.max_periods)

        for t in range(1, self.config.max_periods + 1):
            self.period_state = self._advance_period(t)
            if progress_callback is not None:
                progress_callback(t, self.config.max_periods)
            if observe_callback is not None and t % self.config.observer_interval == 0:
                partial = self.observer.finalize(self.period_state)
                observe_callback(partial)

        log.info("simulation_v2.completed", total_periods=self.config.max_periods)

        return self.observer.finalize(self.period_state)

    def _advance_period(self, t: int) -> PeriodState:
        """Execute one period with the 9-step lifecycle.

        Args:
            t: Period number (1-indexed).

        Returns:
            Updated PeriodState for this period.
        """
        # Deep copy current state for immutability
        households = [h.model_copy(deep=True) for h in self.period_state.households]
        firms = [f.model_copy(deep=True) for f in self.period_state.firms]
        constitution = self.period_state.constitution.model_copy(deep=True)
        shocks = self.period_state.shocks.model_copy(deep=True)

        # Reset period-level capital flow ledger.
        self._period_capital_debits = {}
        self._period_capital_credits = {}

        # Step 1: Draw shocks
        households, shocks = self._draw_shocks(t, households, shocks)

        # Step 1b: Apply productivity effects from novel institutional rules
        # before market clearing so prices reflect institutional effects
        households = self.constitution_engine.apply_productivity_effects(
            households, constitution, self.period_state.market
        )

        # Step 2: Clear markets
        market = self._clear_markets(households, firms, shocks)

        # Step 2b: Nominal block (REQ-310..315)
        if self._nominal_block is not None:
            self._nominal_state = self._compute_nominal(market, shocks)
            # Keep capital and bond returns explicit; legacy interest_rate
            # mirrors the effective bond return used by household solvers.
            capital_rate = (
                market.capital_rate if market.capital_rate is not None else market.interest_rate
            )
            market = market.model_copy(
                update={
                    "interest_rate": self._nominal_state.real_bond_rate,
                    "bond_rate": self._nominal_state.real_bond_rate,
                    "capital_rate": capital_rate,
                }
            )

        # Step 3: Collect decisions
        econ_decisions, entre_decisions = self._collect_decisions(
            households, firms, market, constitution
        )

        # Step 4: Validate constraints (including mechanism effect constraints)
        mechanism_constraints = self.constitution_engine.compute_agent_constraints(
            households, constitution, market
        )
        econ_decisions = self._validate_constraints(
            econ_decisions, households, market, constitution, mechanism_constraints
        )

        # Step 4b: Set labor_supply and income on households from validated decisions
        # so that Step 6 (enforce_taxes) can compute taxes on actual income.
        # Build firm profit lookup for entrepreneur income (when fix is active)
        _firm_profit_by_owner: dict[str, float] = {}
        if self.config.fix_entrepreneur_budget:
            for f in firms:
                _firm_profit_by_owner[f.owner_id] = f.profit

        for h in households:
            decision = econ_decisions.get(h.id, EconomicDecision(consumption=0.0, leisure=0.5))
            if self.config.fix_entrepreneur_budget and h.role == OccupationalRole.ENTREPRENEUR:
                # Entrepreneurs do not supply labor; income = firm profit only
                h.labor_supply = 0.0
                h.income = _firm_profit_by_owner.get(h.id, 0.0)
            else:
                h.labor_supply = 1.0 - decision.leisure
                h.income = market.wage * h.productivity * h.labor_supply

        # Step 5: Execute production
        firms = self._execute_production(firms, entre_decisions, households, market)
        self._refresh_household_incomes_after_production(households, firms, market, econ_decisions)

        # Step 6: Enforce constitution
        households, public_goods = self._enforce_constitution(
            households, firms, constitution, market
        )

        # Sync value function and current inequality context for Bellman
        # political evaluation before governance step.
        self._sync_bellman_political_context(
            households=households,
            market=market,
            constitution=constitution,
            public_goods=public_goods,
        )

        # Step 7: Process governance
        proposals, votes, constitution = self._process_governance(
            t, households, constitution, market
        )

        # Step 7b: Update government budget (REQ-306..309)
        constitution, market = self._update_government(
            households, constitution, market, public_goods
        )

        # Step 8: Update states
        households = self._update_states(households, econ_decisions, market, public_goods, firms)

        # Step 8b: Update KFE distribution if enabled
        if self.config.distribution_mode == "kfe":
            avg_transfer = (
                sum(h.transfers_received for h in households) / len(households)
                if households
                else 0.0
            )
            self._update_distribution(
                econ_decisions=econ_decisions,
                households=households,
                market=market,
                constitution=constitution,
                public_goods=public_goods,
                transfer=avg_transfer,
            )

        # Reconcile and persist aggregate accounting after all within-period
        # household and fiscal updates.
        market = self._reconcile_market_aggregates(market, households, firms, public_goods)

        # Step 9: Observe
        self._observe_period(t, households, firms, market, constitution, proposals, votes)

        return PeriodState(
            period=t,
            households=households,
            firms=firms,
            market=market,
            shocks=shocks,
            constitution=constitution,
            proposals=proposals,
            votes=votes,
        )

    # ------------------------------------------------------------------
    # Step 1: Draw shocks (REQ-015, REQ-016, REQ-017)
    # ------------------------------------------------------------------

    def _draw_shocks(
        self,
        t: int,
        households: list[HouseholdState],
        shocks: ShockState,
    ) -> tuple[list[HouseholdState], ShockState]:
        """Draw idiosyncratic, aggregate, and preference shocks.

        Args:
            t: Current period.
            households: Current household states.
            shocks: Previous shock state (carries grid and matrix).

        Returns:
            Tuple of (updated households, updated shock state).
        """
        # Idiosyncratic productivity shocks (REQ-015)
        households = draw_idiosyncratic_shocks(
            households,
            shocks.transition_matrix,
            shocks.productivity_grid,
            self.rng,
        )

        # Entrepreneurial ability shocks
        if shocks.ability_grid and shocks.ability_transition_matrix:
            households = draw_entrepreneurial_ability_shocks(
                households,
                shocks.ability_transition_matrix,
                shocks.ability_grid,
                self.rng,
            )

        # Aggregate TFP shock (REQ-016)
        new_tfp = draw_aggregate_tfp(
            self._prev_log_a,
            self.config.rho_a,
            self.config.sigma_a,
            self.rng,
        )
        self._prev_log_a = math.log(new_tfp)

        # Preference shocks (REQ-017, optional)
        preference_shocks: dict[str, float] = {}
        if self.config.enable_preference_shocks:
            preference_shocks = draw_preference_shocks(
                households, self.rng, self.config.sigma_z * 0.1
            )

        updated_shocks = shocks.model_copy(
            update={
                "aggregate_tfp": new_tfp,
                "preference_shocks": preference_shocks,
            }
        )

        log.debug(
            "step1.shocks_drawn",
            period=t,
            aggregate_tfp=round(new_tfp, 4),
            num_preference_shocks=len(preference_shocks),
        )

        return households, updated_shocks

    # ------------------------------------------------------------------
    # Step 2: Clear markets (REQ-011..014, PROP-003)
    # ------------------------------------------------------------------

    def _clear_markets(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        shocks: ShockState,
    ) -> MarketState:
        """Find equilibrium prices via tatonnement.

        Args:
            households: Current household states.
            firms: Active firms.
            shocks: Current shock state.

        Returns:
            MarketState with equilibrium prices.
        """
        prev_market = self.period_state.market
        market = clear_markets(
            households=households,
            firms=firms,
            aggregate_tfp=shocks.aggregate_tfp,
            config=self.config,
            prev_market=prev_market,
        )

        log.debug(
            "step2.markets_cleared",
            wage=round(market.wage, 4),
            interest_rate=round(market.interest_rate, 4),
            clearing_error=round(market.market_clearing_error, 8),
        )

        return market

    # ------------------------------------------------------------------
    # Step 3: Collect decisions (REQ-024..026, REQ-029)
    # ------------------------------------------------------------------

    def _collect_decisions(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> tuple[dict[str, EconomicDecision], dict[str, EntrepreneurialDecision]]:
        """Collect economic and entrepreneurial decisions from agents.

        Uses LLM engine if available, otherwise numerical solver.

        Args:
            households: Current household states.
            firms: Active firms.
            market: Current market equilibrium.
            constitution: Current constitution.

        Returns:
            Tuple of (economic_decisions, entrepreneurial_decisions).
        """
        use_llm_for_economic = (
            self._llm_engine is not None and self.config.use_llm and not self.config.benchmark_mode
        )
        expected_public_goods, expected_transfer = self._expected_fiscal_inputs(
            households, constitution
        )

        # Economic decisions
        if use_llm_for_economic:
            econ_decisions = self._llm_engine.collect_economic_decisions(
                households=households,
                market=market,
                constitution=constitution,
            )
        else:
            # Extract flat tax rate from constitution
            tax_rate = self._get_tax_rate(constitution)
            econ_decisions = self._solver.solve_all(
                households=households,
                market=market,
                constitution_tax_rate=tax_rate,
                public_goods=expected_public_goods,
                transfer=expected_transfer,
            )

        # Entrepreneurial decisions
        if use_llm_for_economic:
            entre_decisions = self._llm_engine.collect_entrepreneurial_decisions(
                households=households,
                firms=firms,
                market=market,
                constitution=constitution,
            )
        else:
            # Benchmark mode: solver-driven entrepreneurial decisions
            # Pass VFI value function for Bellman-based occupational choice (REQ-110)
            vfi_value_func, vfi_a_grid = self._solver.get_value_function()
            entre_decisions = self._entre_solver.solve_all(
                households,
                firms,
                market,
                public_goods=expected_public_goods,
                vfi_value_func=vfi_value_func,
                a_grid=vfi_a_grid,
            )

        log.debug(
            "step3.decisions_collected",
            num_economic=len(econ_decisions),
            num_entrepreneurial=len(entre_decisions),
        )

        return econ_decisions, entre_decisions

    def _expected_fiscal_inputs(
        self,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
    ) -> tuple[float, float]:
        """Build lagged fiscal expectations for Step 3 decision solving."""
        n_households = len(households)
        if n_households == 0:
            return 0.0, 0.0

        lagged_revenue = sum(max(h.taxes_paid, 0.0) for h in households)
        expected_public_goods = self.constitution_engine.enforce_public_goods(
            constitution, lagged_revenue, n_households
        )
        if expected_public_goods <= 0.0 and self.period_state.market.government_spending > 0.0:
            expected_public_goods = self.period_state.market.government_spending / n_households

        expected_transfer = sum(max(h.transfers_received, 0.0) for h in households) / n_households
        return expected_public_goods, expected_transfer

    # ------------------------------------------------------------------
    # Step 4: Validate constraints (REQ-004, 005, PROP-002, PROP-004)
    # ------------------------------------------------------------------

    def _validate_constraints(
        self,
        decisions: dict[str, EconomicDecision],
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
        mechanism_constraints: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, EconomicDecision]:
        """Project economic decisions onto the feasible set.

        Ensures consumption and leisure are within budget constraints,
        including any custom bounds from mechanism effects.

        Args:
            decisions: Raw economic decisions from step 3.
            households: Current household states.
            market: Current market equilibrium.
            constitution: Current constitution.
            mechanism_constraints: Optional custom bounds from mechanism effects.

        Returns:
            Constrained economic decisions.
        """
        tax_rate = self._get_tax_rate(constitution)

        # Build firm profit lookup for entrepreneur budget validation
        _firm_profit_by_owner: dict[str, float] = {}
        if self.config.fix_entrepreneur_budget:
            active_firms = self.period_state.firms
            for f in active_firms:
                _firm_profit_by_owner[f.owner_id] = f.profit

        constrained: dict[str, EconomicDecision] = {}
        for h in households:
            decision = decisions.get(h.id, EconomicDecision(consumption=0.0, leisure=0.5))
            if self.config.fix_entrepreneur_budget and h.role == OccupationalRole.ENTREPRENEUR:
                # Entrepreneur income = firm profit only (no labor income)
                income = _firm_profit_by_owner.get(h.id, 0.0)
            else:
                income = market.wage * h.productivity * (1.0 - decision.leisure)
            tax = income * tax_rate
            transfer = 0.0  # Conservative; actual transfers computed in step 6
            budget = compute_budget(
                h,
                market.wage,
                market.interest_rate,
                tax,
                transfer,
                firm_profit=_firm_profit_by_owner.get(h.id, 0.0)
                if self.config.fix_entrepreneur_budget
                else None,
            )
            clamped = enforce_budget_constraint(decision, h, budget, self.config.a_min)

            # Apply mechanism effect constraints
            if mechanism_constraints and h.id in mechanism_constraints:
                bounds = mechanism_constraints[h.id]
                cons = clamped.consumption
                leis = clamped.leisure
                cons = max(cons, bounds.get("consumption_min", cons))
                cons = min(cons, bounds.get("consumption_max", cons))
                leis = max(leis, bounds.get("leisure_min", leis))
                leis = min(leis, bounds.get("leisure_max", leis))
                cons = max(0.0, cons)
                leis = max(0.0, min(1.0, leis))
                clamped = EconomicDecision(consumption=cons, leisure=leis)

            constrained[h.id] = clamped

        log.debug("step4.constraints_validated", num_agents=len(constrained))
        return constrained

    def _sync_bellman_political_context(
        self,
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
        public_goods: float,
    ) -> None:
        """Push latest value-function and inequality context into LLM engine.

        This enables Bellman-derived political context (REQ-404) and pure
        Bellman governance mode (REQ-405) to use current simulation state.
        """
        if self._llm_engine is None:
            return

        value_func, a_grid = self._solver.get_value_function()
        if value_func is None and self.config.pure_bellman_politics and households:
            avg_transfer = sum(h.transfers_received for h in households) / len(households)
            # Ensure Bellman politics has a current value function even when
            # economic decisions came from the LLM path this period.
            self._solver.solve_all(
                households=households,
                market=market,
                constitution_tax_rate=self._get_tax_rate(constitution),
                public_goods=public_goods,
                transfer=avg_transfer,
            )
            value_func, a_grid = self._solver.get_value_function()
        self._llm_engine.set_value_function(value_func, a_grid)

        gini = ObserverV2._compute_gini([h.wealth for h in households])
        self._llm_engine.set_observed_gini(gini)

    # ------------------------------------------------------------------
    # Step 5: Execute production (REQ-007..009)
    # ------------------------------------------------------------------

    def _execute_production(
        self,
        firms: list[FirmState],
        entre_decisions: dict[str, EntrepreneurialDecision],
        households: list[HouseholdState],
        market: MarketState,
    ) -> list[FirmState]:
        """Execute firm production, R&D, creation, and liquidation.

        Args:
            firms: Current active firms.
            entre_decisions: Entrepreneurial decisions from step 3.
            households: Current household states.
            market: Current market equilibrium.

        Returns:
            Updated list of active firms.
        """
        household_map = {h.id: h for h in households}
        updated_firms: list[FirmState] = []

        # Process existing firms
        for firm in firms:
            owner_decision = entre_decisions.get(firm.owner_id, EntrepreneurialDecision())

            # Check for liquidation (REQ-010)
            if owner_decision.close_firm:
                returned_capital = liquidate_firm(firm)
                owner = household_map.get(firm.owner_id)
                if owner is not None:
                    # Firm capital is treated as rented from aggregate savings
                    # (see FirmState docs), so liquidation does not move principal
                    # onto owner balance sheets here.
                    log.debug(
                        "step5.firm_liquidated",
                        firm_id=firm.id,
                        returned_capital=returned_capital,
                    )
                continue

            labor_demand = (
                owner_decision.labor_demand
                if owner_decision.labor_demand > 0.0
                else firm.labor_demand
            )
            priced_firm = firm.model_copy(update={"labor_demand": labor_demand})

            # Produce output (REQ-007) using decision-updated labor demand.
            output = produce_output(priced_firm, self.config.alpha)

            # Distribute income (REQ-008)
            capital_rate = (
                market.capital_rate if market.capital_rate is not None else market.interest_rate
            )
            wages_paid, capital_cost, profit = distribute_firm_income(
                priced_firm.model_copy(update={"output": output}),
                market.wage,
                capital_rate,
                self.config.delta,
            )
            _ = (wages_paid, capital_cost)

            # Update firm with results
            firm_update = {
                "output": output,
                "profit": profit,
                "labor_demand": labor_demand,
                "rd_spend": owner_decision.rd_spend,
            }
            updated_firm = firm.model_copy(update=firm_update)

            # Apply R&D shock (REQ-009)
            mean_rd = sum(f.rd_spend for f in firms) / len(firms) if firms else 1.0
            updated_firm = apply_rd_shock(updated_firm, self.rng, self.config, mean_rd)

            updated_firms.append(updated_firm)

        # Process new firm creation
        aggregate_tfp = self.period_state.shocks.aggregate_tfp
        for h in households:
            decision = entre_decisions.get(h.id, EntrepreneurialDecision())
            if decision.create_firm and h.wealth >= self.config.min_firm_capital:
                firm_tfp = h.entrepreneurial_ability * aggregate_tfp
                new_firm = FirmState(
                    id=f"firm_{self._next_firm_id:04d}",
                    owner_id=h.id,
                    owner_ability=h.entrepreneurial_ability,
                    capital=decision.capital_investment,
                    labor_demand=decision.labor_demand,
                    tfp=firm_tfp,
                    rd_spend=decision.rd_spend,
                )
                self._next_firm_id += 1
                updated_firms.append(new_firm)

                # Entry cost is sunk and paid from household resources.
                # Capital itself is modeled as rented, not principal-financed.
                capital_debit = self.config.firm_entry_cost
                self._period_capital_debits[h.id] = (
                    self._period_capital_debits.get(h.id, 0.0) + capital_debit
                )

                log.debug(
                    "step5.firm_created",
                    firm_id=new_firm.id,
                    owner_id=h.id,
                    capital=decision.capital_investment,
                    tfp=round(firm_tfp, 4),
                )

        log.debug(
            "step5.production_executed",
            num_firms=len(updated_firms),
            total_output=sum(f.output for f in updated_firms),
        )

        return updated_firms

    def _refresh_household_incomes_after_production(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        econ_decisions: dict[str, EconomicDecision],
    ) -> None:
        """Refresh household income/labor fields so taxes see current profits."""
        firm_profit_by_owner: dict[str, float] = {}
        for firm in firms:
            firm_profit_by_owner[firm.owner_id] = (
                firm_profit_by_owner.get(firm.owner_id, 0.0) + firm.profit
            )

        for household in households:
            decision = econ_decisions.get(
                household.id, EconomicDecision(consumption=0.0, leisure=0.5)
            )
            labor_supply = 1.0 - decision.leisure
            labor_income = market.wage * household.productivity * labor_supply
            firm_profit = firm_profit_by_owner.get(household.id, 0.0)

            if self.config.fix_entrepreneur_budget and household.id in firm_profit_by_owner:
                household.labor_supply = 0.0
                household.income = firm_profit
            else:
                household.labor_supply = labor_supply
                household.income = labor_income + firm_profit

    # ------------------------------------------------------------------
    # Step 6: Enforce constitution (REQ-022)
    # ------------------------------------------------------------------

    def _enforce_constitution(
        self,
        households: list[HouseholdState],
        firms: list[FirmState],
        constitution: ConstitutionV2,
        market: MarketState,
    ) -> tuple[list[HouseholdState], float]:
        """Apply taxes, transfers, public goods, and regulations.

        Args:
            households: Current household states.
            firms: Active firms.
            constitution: Current constitution.
            market: Current market equilibrium.

        Returns:
            Tuple of (updated households, public_goods_per_capita).
        """
        # Apply taxes (collect revenue)
        households, revenue = self.constitution_engine.enforce_taxes(
            households, constitution, market
        )

        # Compute public goods allocation
        public_goods = self.constitution_engine.enforce_public_goods(
            constitution, revenue, len(households)
        )

        # Distribute transfers (from remaining revenue after public goods)
        transfer_revenue = revenue - public_goods * len(households)
        transfer_revenue = max(0.0, transfer_revenue)
        households = self.constitution_engine.enforce_transfers(
            households, constitution, transfer_revenue
        )

        # Apply regulations
        _, households = self.constitution_engine.enforce_regulations(
            firms, households, constitution
        )

        # Apply mechanism effects from novel institution rules
        households, pg_extra = self.constitution_engine.enforce_mechanism_effects(
            households, constitution, market, self.config.a_min
        )
        public_goods += pg_extra

        log.debug(
            "step6.constitution_enforced",
            revenue=round(revenue, 2),
            public_goods=round(public_goods, 4),
            transfer_revenue=round(transfer_revenue, 2),
            mechanism_pg_extra=round(pg_extra, 4),
        )

        return households, public_goods

    @staticmethod
    def _assign_proposal_ids(proposals: list[ConstitutionalProposal], period: int) -> None:
        """Ensure each proposal has a unique identifier for vote tracking."""
        for idx, proposal in enumerate(proposals):
            if proposal.proposal_id:
                continue
            proposals[idx] = proposal.model_copy(
                update={"proposal_id": f"p{period:04d}_{idx:04d}_{proposal.proposer_id}"}
            )

    # ------------------------------------------------------------------
    # Step 7: Process governance (REQ-019..021, REQ-023)
    # ------------------------------------------------------------------

    def _process_governance(
        self,
        t: int,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
        market: MarketState,
    ) -> tuple[list[ConstitutionalProposal], list[VoteOutcomeV2], ConstitutionV2]:
        """Collect proposals, run votes, apply passed proposals.

        Only runs on proposal_interval periods.

        Args:
            t: Current period number.
            households: Current household states.
            constitution: Current constitution.
            market: Current market equilibrium.

        Returns:
            Tuple of (proposals, vote_outcomes, updated_constitution).
        """
        if t % self.config.proposal_interval != 0:
            return [], [], constitution

        proposals: list[ConstitutionalProposal] = []
        all_votes: dict[str, dict[str, bool]] = {}

        if self._llm_engine is not None:
            # LLM-driven governance
            political_decisions = self._llm_engine.collect_political_decisions(
                households=households,
                constitution=constitution,
                market=market,
                history=self.observer.history[-5:] if self.observer.history else None,
                recent_outcomes=self._recent_vote_outcomes[-20:]
                if self._recent_vote_outcomes
                else None,
            )

            # Extract proposals
            for agent_id, decision in political_decisions.items():
                if decision.proposal is not None:
                    is_valid, reason = validate_proposal_v2(decision.proposal, constitution)
                    if is_valid:
                        proposals.append(decision.proposal)
                    else:
                        self._recent_rejections.append(
                            (agent_id, decision.proposal.rule_name, reason or "validation_failed")
                        )
                        self._recent_vote_outcomes.append(
                            {
                                "period": t,
                                "rule_name": decision.proposal.rule_name,
                                "action": decision.proposal.action,
                                "outcome": "rejected_validation",
                                "reason": reason or "validation_failed",
                            }
                        )
                        log.debug(
                            "step7.proposal_rejected",
                            agent_id=agent_id,
                            reason=reason,
                        )

            self._assign_proposal_ids(proposals, t)

            # Extract votes from political decisions
            for agent_id, decision in political_decisions.items():
                if decision.votes:
                    all_votes[agent_id] = decision.votes
        else:
            # Benchmark mode: rule-based governance (mirrors v1 citizen logic)
            # Shuffle for fairness
            h_order = list(households)
            self.rng.shuffle(h_order)

            for h in h_order:
                proposal = decide_proposal_v2(h, constitution, self.config.num_agents, self.rng)
                if proposal is not None:
                    is_valid, reason = validate_proposal_v2(proposal, constitution)
                    if is_valid:
                        proposals.append(proposal)
                    else:
                        self._recent_rejections.append(
                            (h.id, proposal.rule_name, reason or "validation_failed")
                        )
                        log.debug(
                            "step7.benchmark_proposal_rejected",
                            agent_id=h.id,
                            reason=reason,
                        )

            self._assign_proposal_ids(proposals, t)

            # Collect votes from all households
            if proposals:
                for h in households:
                    agent_votes = decide_votes_v2(h, proposals, constitution)
                    if agent_votes:
                        all_votes[h.id] = agent_votes

        # Tally votes for each proposal
        vote_outcomes: list[VoteOutcomeV2] = []
        total_eligible = len(households)
        legacy_rule_vote_consumed: dict[str, set[str]] = {}

        for proposal in proposals:
            proposal_vote_key = proposal.proposal_id or proposal.rule_name
            proposal_votes: dict[str, bool] = {}
            for agent_id, agent_vote_map in all_votes.items():
                vote = agent_vote_map.get(proposal_vote_key)
                if vote is None:
                    # Backward-compatible fallback for legacy vote maps keyed by
                    # rule name. Consume at most one vote per (agent, rule) to
                    # avoid proposal-collision double counting.
                    legacy_vote = agent_vote_map.get(proposal.rule_name)
                    if legacy_vote is not None:
                        used_rules = legacy_rule_vote_consumed.setdefault(agent_id, set())
                        if proposal.rule_name in used_rules:
                            continue
                        used_rules.add(proposal.rule_name)
                        vote = legacy_vote
                if vote is not None:
                    proposal_votes[agent_id] = vote

            outcome = tally_votes_v2(proposal, proposal_votes, constitution, total_eligible)
            vote_outcomes.append(outcome)
            self._recent_vote_outcomes.append(
                {
                    "period": t,
                    "proposal_id": proposal.proposal_id,
                    "rule_name": proposal.rule_name,
                    "action": proposal.action,
                    "outcome": "passed" if outcome.passed else "voted_down",
                    "votes_for": outcome.votes_for,
                    "votes_against": outcome.votes_against,
                }
            )

        # Apply passed proposals
        constitution = apply_passed_proposals_v2(vote_outcomes, constitution)

        # Keep only recent outcomes (last 20) to avoid unbounded growth
        if len(self._recent_vote_outcomes) > 50:
            self._recent_vote_outcomes = self._recent_vote_outcomes[-20:]
        if len(self._recent_rejections) > 50:
            self._recent_rejections = self._recent_rejections[-20:]

        log.debug(
            "step7.governance_processed",
            period=t,
            num_proposals=len(proposals),
            num_passed=sum(1 for v in vote_outcomes if v.passed),
        )

        return proposals, vote_outcomes, constitution

    # ------------------------------------------------------------------
    # Step 7b: Government budget update (REQ-306..309)
    # ------------------------------------------------------------------

    def _update_government(
        self,
        households: list[HouseholdState],
        constitution: ConstitutionV2,
        market: MarketState,
        public_goods: float,
    ) -> tuple[ConstitutionV2, MarketState]:
        """Update government budget and apply fiscal rule.

        Computes bond market clearing rate, updates the government budget
        constraint B' = (1+r^b)*B + G + Tr - T, and checks the fiscal rule.

        When initial_debt is 0 (default), this is effectively a no-op since
        the budget constraint with zero debt and balanced budget produces
        zero new debt.

        Args:
            households: Current household states (with taxes_paid, transfers_received).
            constitution: Current constitution (for tax-rule feedback).
            market: Current market equilibrium.
            public_goods: Public goods per capita this period.
        """
        # Skip if no government debt to manage
        if self.config.initial_debt <= 0.0 and self._government.state.debt <= 0.0:
            return constitution, market

        # Compute fiscal aggregates from household states
        tax_revenue = sum(h.taxes_paid for h in households)
        transfers = sum(h.transfers_received for h in households)
        spending = public_goods * len(households)

        base_bond_rate = market.bond_rate if market.bond_rate is not None else market.interest_rate

        # Clear bond market to find equilibrium bond rate (REQ-307)
        bond_rate, clearing_error = clear_bond_market(
            households=households,
            government_debt=max(self._government.state.debt, 0.0),
            base_interest_rate=base_bond_rate,
        )

        # Update government budget constraint (REQ-306, PROP-010)
        self._government.update_budget(
            tax_revenue=tax_revenue,
            spending=spending,
            transfers=transfers,
            bond_rate=bond_rate,
        )

        # Apply fiscal rule (REQ-309)
        output = market.aggregate_output if market.aggregate_output > 0.0 else 1.0
        tax_adjustment = self._government.fiscal_rule(output)

        market = market.model_copy(update={"bond_rate": bond_rate})

        if tax_adjustment > 0.0:
            constitution = self._apply_fiscal_tax_adjustment(constitution, tax_adjustment)
            log.info(
                "step7b.fiscal_rule_active",
                debt_to_gdp=round(self._government.state.debt_to_gdp, 4),
                tax_adjustment=tax_adjustment,
                bond_clearing_error=round(clearing_error, 8),
            )

        return constitution, market

    def _apply_fiscal_tax_adjustment(
        self,
        constitution: ConstitutionV2,
        tax_adjustment: float,
    ) -> ConstitutionV2:
        """Apply fiscal-rule tax feedback to the first active tax rule."""
        tax_rules = constitution.get_tax_rules()
        if not tax_rules:
            log.warning(
                "step7b.fiscal_rule_no_tax_rule",
                tax_adjustment=round(tax_adjustment, 6),
            )
            return constitution

        target_rule = tax_rules[0]
        old_rate = float(target_rule.parameters.get("rate", 0.0))
        new_rate = max(0.0, min(1.0, old_rate + tax_adjustment))
        if abs(new_rate - old_rate) < 1e-12:
            return constitution

        updated_rule = target_rule.model_copy(
            update={
                "parameters": {**target_rule.parameters, "rate": new_rate},
                "version": target_rule.version + 1,
            }
        )
        updated_constitution = constitution.model_copy(deep=True)
        updated_constitution.rules[target_rule.name] = updated_rule
        log.info(
            "step7b.fiscal_rule_applied_to_tax",
            rule=target_rule.name,
            old_rate=round(old_rate, 6),
            new_rate=round(new_rate, 6),
        )
        return updated_constitution

    def _reconcile_market_aggregates(
        self,
        market: MarketState,
        households: list[HouseholdState],
        firms: list[FirmState],
        public_goods: float,
    ) -> MarketState:
        """Recompute aggregate ledger and expose resource residual explicitly."""
        aggregate_consumption = sum(h.consumption for h in households)
        if firms:
            aggregate_output = sum(max(f.output, 0.0) for f in firms)
            aggregate_investment = sum(max(f.capital, 0.0) * self.config.delta for f in firms)
        else:
            aggregate_output = max(market.aggregate_output, 0.0)
            aggregate_investment = max(market.aggregate_investment, 0.0)

        government_spending = max(public_goods * len(households), 0.0)
        resource_residual = (
            aggregate_output - aggregate_consumption - aggregate_investment - government_spending
        )

        if abs(resource_residual) > 1e-6:
            log.debug(
                "step8.aggregate_residual",
                residual=round(resource_residual, 6),
                output=round(aggregate_output, 4),
                consumption=round(aggregate_consumption, 4),
                investment=round(aggregate_investment, 4),
                government_spending=round(government_spending, 4),
            )

        return market.model_copy(
            update={
                "aggregate_output": aggregate_output,
                "aggregate_consumption": aggregate_consumption,
                "aggregate_investment": aggregate_investment,
                "government_spending": government_spending,
                "resource_residual": resource_residual,
            }
        )

    # ------------------------------------------------------------------
    # Step 8: Update states (REQ-001, 003, 006, 010)
    # ------------------------------------------------------------------

    def _update_states(
        self,
        households: list[HouseholdState],
        econ_decisions: dict[str, EconomicDecision],
        market: MarketState,
        public_goods: float,
        firms: list[FirmState] | None = None,
    ) -> list[HouseholdState]:
        """Apply all changes atomically: consumption, savings, utility.

        Args:
            households: Household states after constitution enforcement.
            econ_decisions: Validated economic decisions.
            market: Market equilibrium.
            public_goods: Public goods per capita.
            firms: Current period's active firms (for role/profit attribution).

        Returns:
            Updated household states with all per-period fields set.
        """
        # Build firm lookup for entrepreneur profit attribution
        active_firms = firms if firms is not None else self.period_state.firms
        firm_profit_by_owner: dict[str, float] = {}
        firm_id_by_owner: dict[str, str] = {}
        owners_with_firms: set[str] = set()
        for f in active_firms:
            firm_profit_by_owner[f.owner_id] = f.profit
            firm_id_by_owner[f.owner_id] = f.id
            owners_with_firms.add(f.owner_id)

        updated: list[HouseholdState] = []

        for h in households:
            decision = econ_decisions.get(h.id, EconomicDecision(consumption=0.0, leisure=0.5))

            consumption = decision.consumption
            leisure = decision.leisure
            labor_supply = 1.0 - leisure

            # Add firm profit for entrepreneurs
            firm_profit = firm_profit_by_owner.get(h.id, 0.0)
            capital_debit = self._period_capital_debits.get(h.id, 0.0)
            capital_credit = self._period_capital_credits.get(h.id, 0.0)

            if self.config.fix_entrepreneur_budget and h.id in owners_with_firms:
                # Entrepreneurs are residual claimants: income = pi_f only.
                # Their labor is embedded in firm production, not separately
                # compensated at market wage. Labor supply = 0.
                labor_supply = 0.0
                total_income = firm_profit
            else:
                # Workers: income from labor (+ any leftover firm profit for
                # backward compat when flag is off)
                labor_income = market.wage * h.productivity * labor_supply
                total_income = labor_income + firm_profit

            # Single-asset and two-asset transitions share current-period
            # income/tax/transfer accounting, but two-asset mode tracks
            # explicit liquid/illiquid positions.
            if self.config.two_asset_mode:
                bond_rate = (
                    market.bond_rate if market.bond_rate is not None else market.interest_rate
                )
                capital_rate = (
                    market.capital_rate
                    if market.capital_rate is not None
                    else market.interest_rate
                )
                liquid_prev = h.liquid
                illiquid_prev = h.illiquid

                # Fallback split for legacy snapshots where two-asset fields
                # were not initialized.
                if liquid_prev <= 0.0 and illiquid_prev <= 0.0 and h.wealth > 0.0:
                    liquid_prev = max(self.config.b_min, 0.3 * h.wealth)
                    illiquid_prev = max(0.0, h.wealth - liquid_prev)

                liquid_resources = (
                    (1.0 + bond_rate) * liquid_prev
                    + total_income
                    - consumption
                    - h.taxes_paid
                    + h.transfers_received
                    - capital_debit
                    + capital_credit
                )

                # Heuristic target illiquid share used by the reduced-form
                # two-asset transition until full nested-EGM is active.
                target_illiquid = max(0.0, 0.7 * max(h.wealth, 0.0))
                deposit = target_illiquid - illiquid_prev

                # Constrain deposits/withdrawals to feasible ranges.
                if deposit > 0.0:
                    deposit = min(deposit, max(liquid_resources, 0.0))
                else:
                    deposit = max(deposit, -illiquid_prev)

                adjust_cost = adjustment_cost(
                    deposit=deposit,
                    illiquid_stock=illiquid_prev,
                    chi_0=self.config.chi_0,
                    chi_1=self.config.chi_1,
                )

                # If liquid resources cannot finance deposit+cost, shrink deposit.
                if deposit > 0.0 and deposit + adjust_cost > liquid_resources:
                    available = max(liquid_resources, 0.0)
                    deposit = max(0.0, available / (1.0 + self.config.chi_0))
                    adjust_cost = adjustment_cost(
                        deposit=deposit,
                        illiquid_stock=illiquid_prev,
                        chi_0=self.config.chi_0,
                        chi_1=self.config.chi_1,
                    )

                new_liquid = liquid_resources - deposit - adjust_cost
                new_illiquid = max(
                    0.0,
                    (1.0 + capital_rate) * illiquid_prev + deposit,
                )

                # Enforce borrowing constraint on liquid assets with internal
                # rebalancing from illiquid holdings.
                if new_liquid < self.config.b_min:
                    shortfall = self.config.b_min - new_liquid
                    withdrawal = min(shortfall, new_illiquid)
                    new_illiquid -= withdrawal
                    new_liquid += withdrawal

                new_liquid = max(new_liquid, self.config.b_min)
                new_wealth = max(new_liquid + new_illiquid, self.config.a_min)
            else:
                # Full DSGE-HA budget constraint (single-asset):
                # a' = (1+r)*a + income - c - taxes + transfers
                new_wealth = (
                    (1.0 + market.interest_rate) * h.wealth
                    + total_income
                    - consumption
                    - h.taxes_paid
                    + h.transfers_received
                    - capital_debit
                    + capital_credit
                )
                new_wealth = max(new_wealth, self.config.a_min)
                new_liquid = h.liquid
                new_illiquid = h.illiquid
            savings = new_wealth - h.wealth

            # Update role based on firm ownership
            role = h.role
            if h.id in owners_with_firms:
                role = OccupationalRole.ENTREPRENEUR
            elif h.role == OccupationalRole.ENTREPRENEUR:
                # Was entrepreneur but firm no longer exists (liquidated)
                role = OccupationalRole.WORKER

            # Update household with per-period outcomes
            updated_h = h.model_copy(
                update={
                    "consumption": consumption,
                    "leisure": leisure,
                    "labor_supply": labor_supply,
                    "income": total_income,
                    "savings": savings,
                    "wealth": new_wealth,
                    "liquid": new_liquid,
                    "illiquid": new_illiquid,
                    "role": role,
                    "firm_id": firm_id_by_owner.get(h.id, None),
                }
            )

            # Compute realized utility (REQ-028)
            utility = compute_realized_utility(updated_h, max(public_goods, 1e-10))
            updated_h = updated_h.model_copy(update={"realized_utility": utility})

            updated.append(updated_h)

        # Validate household states for numerical stability
        validate_household_states(updated)

        log.debug(
            "step8.states_updated",
            num_households=len(updated),
            total_consumption=round(sum(h.consumption for h in updated), 2),
            total_wealth=round(sum(h.wealth for h in updated), 2),
        )

        return updated

    # ------------------------------------------------------------------
    # Step 8b: Update KFE distribution (Phase 2, REQ-201..205)
    # ------------------------------------------------------------------

    def _update_distribution(
        self,
        econ_decisions: dict[str, EconomicDecision],
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
        public_goods: float,
        transfer: float,
    ) -> None:
        """Advance KFE distribution one period using current policy.

        Constructs the savings policy grid from the solver's policy
        functions and calls Distribution.forward().

        Args:
            econ_decisions: Economic decisions (not used directly; policy
                comes from solver grid).
            households: Current household states.
            market: Current market equilibrium.
        """
        if self._distribution is None:
            return

        solver = self._solver
        a_grid_np = solver._a_grid_np if hasattr(solver, "_a_grid_np") else np.array(solver.a_grid)
        z_grid_np = (
            solver._z_grid_np
            if hasattr(solver, "_z_grid_np")
            else np.array(solver.productivity_grid)
        )
        trans_np = (
            solver._trans_np
            if hasattr(solver, "_trans_np")
            else np.array(solver.transition_matrix)
        )

        # Build savings policy a'(a, z) on the grid from budget constraint
        tax_rate = self._get_tax_rate(constitution)
        n_a = len(a_grid_np)
        n_z = len(z_grid_np)

        # Get policy functions from the solver (last-solved)
        ref = households[0].utility_params if households else None
        if ref is None:
            return

        def tax_fn(income: float) -> float:
            return income * tax_rate

        if hasattr(solver, "solve_egm_cached"):
            # EGM solver path
            c_policy, lei_policy = solver.solve_egm_cached(
                alpha_u=ref.alpha,
                beta_u=ref.beta,
                gamma_u=ref.gamma,
                beta_discount=ref.beta_discount,
                wage=market.wage,
                interest_rate=market.interest_rate,
                public_goods=max(1e-10, public_goods),
                tax_function=tax_fn,
                transfer=transfer,
            )
        else:
            # VFI solver path
            c_policy_list, lei_policy_list = solver.solve_vfi_shared(
                alpha=ref.alpha,
                beta_param=ref.beta,
                gamma=ref.gamma,
                beta_discount=ref.beta_discount,
                wage=market.wage,
                interest_rate=market.interest_rate,
                public_goods=max(1e-10, public_goods),
                tax_function=tax_fn,
                transfer=transfer,
            )
            c_policy = np.array(c_policy_list)
            lei_policy = np.array(lei_policy_list)

        # Compute savings policy: a' = (1+r)*a + w*z*(1-l) - c - T(y) + Tr
        a_col = a_grid_np.reshape(n_a, 1)
        z_row = z_grid_np.reshape(1, n_z)
        labor = 1.0 - np.asarray(lei_policy)
        income = market.wage * z_row * labor
        tax = income * tax_rate
        policy_savings = (
            (1.0 + market.interest_rate) * a_col + income - tax - np.asarray(c_policy) + transfer
        )
        policy_savings = np.maximum(policy_savings, self.config.a_min)

        self._distribution.forward(policy_savings, trans_np)

        log.debug(
            "step8b.distribution_updated",
            mass_total=round(self._distribution.mass_total(), 12),
            mean_wealth=round(self._distribution.mean_wealth(), 2),
        )

    # ------------------------------------------------------------------
    # Step 9: Observe (REQ-030, REQ-031)
    # ------------------------------------------------------------------

    def _observe_period(
        self,
        t: int,
        households: list[HouseholdState],
        firms: list[FirmState],
        market: MarketState,
        constitution: ConstitutionV2,
        proposals: list[ConstitutionalProposal],
        votes: list[VoteOutcomeV2],
    ) -> None:
        """Record observation statistics at observer_interval periods.

        Delegates to ObserverV2.observe() which computes all REQ-030 stats
        including Pareto efficiency, Gini, welfare, firm stats, etc.

        Args:
            t: Current period.
            households: Updated household states.
            firms: Active firms.
            market: Market equilibrium.
            constitution: Current constitution.
            proposals: Proposals this period.
            votes: Vote outcomes this period.
        """
        if t % self.config.observer_interval != 0:
            return

        period_snapshot = PeriodState(
            period=t,
            households=households,
            firms=firms,
            market=market,
            shocks=self.period_state.shocks,
            constitution=constitution,
            proposals=proposals,
            votes=votes,
        )

        self.observer.observe(period_snapshot)

    # ------------------------------------------------------------------
    # Step 2b: Nominal block (REQ-310..315)
    # ------------------------------------------------------------------

    def _compute_nominal(
        self,
        market: MarketState,
        shocks: ShockState,
    ) -> NominalState:
        """Compute nominal state from current real aggregates and prices.

        Args:
            market: Current real market state from Step 2.
            shocks: Current shock state (for aggregate TFP).

        Returns:
            Updated nominal state for this period.
        """
        if self._nominal_block is None:
            return self._nominal_state

        # Use current output as fallback steady-state anchor if needed.
        output = max(market.aggregate_output, 1e-10)
        if self._steady_state_output <= 0.0:
            self._steady_state_output = output

        # Beta in NK block: use cross-sectional average if available.
        if self.period_state.households:
            beta_disc = sum(
                h.utility_params.beta_discount for h in self.period_state.households
            ) / len(self.period_state.households)
        else:
            beta_disc = self.config.utility_beta_discount

        state = self._nominal_block.update(
            output=output,
            steady_state_output=max(self._steady_state_output, 1e-10),
            wage=market.wage,
            interest_rate=market.interest_rate,
            beta=beta_disc,
            aggregate_tfp=shocks.aggregate_tfp,
        )

        # Slowly update the output reference level to avoid permanent drift.
        self._steady_state_output = 0.99 * self._steady_state_output + 0.01 * output
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_solver(
        config: SimulationConfigV2,
        period_state: PeriodState,
    ) -> NumericalSolver | EGMSolver:
        """Create the appropriate household solver based on config.solver_method.

        Args:
            config: Simulation configuration.
            period_state: Initial period state (for grids).

        Returns:
            NumericalSolver or EGMSolver instance.

        Implements REQ-120 (EGM default, VFI fallback).
        """
        if config.solver_method == "egm":
            return EGMSolver(
                config=config,
                productivity_grid=period_state.shocks.productivity_grid,
                transition_matrix=period_state.shocks.transition_matrix,
            )
        # "vfi" and "vfi_numpy" both use NumericalSolver (numpy-vectorized VFI)
        return NumericalSolver(
            config=config,
            productivity_grid=period_state.shocks.productivity_grid,
            transition_matrix=period_state.shocks.transition_matrix,
        )

    def _get_tax_rate(self, constitution: ConstitutionV2) -> float:
        """Extract a flat tax rate from the constitution.

        Args:
            constitution: Current constitution.

        Returns:
            Effective flat tax rate in [0, 1].
        """
        tax_rules = constitution.get_tax_rules()
        if not tax_rules:
            return 0.0
        # Use the rate from the first tax rule
        rate = tax_rules[0].parameters.get("rate", 0.0)
        return max(0.0, min(1.0, float(rate)))
