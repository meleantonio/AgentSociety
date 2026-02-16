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

import structlog

from emergent_constitution.citizen import decide_proposal, decide_trade, decide_votes
from emergent_constitution.citizen_v2 import decide_proposal_v2, decide_votes_v2
from emergent_constitution.coalition import form_coalitions
from emergent_constitution.config import SimulationConfig, SimulationConfigV2
from emergent_constitution.constitution_engine import ConstitutionEngine
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
from emergent_constitution.egm_solver import EGMSolver
from emergent_constitution.entrepreneurial_solver import EntrepreneurialSolver
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

        # Decision engine: LLM or numerical solver
        if config.benchmark_mode or not config.use_llm:
            self._llm_engine: LLMDecisionEngine | None = None
            self._solver = self._create_solver(config, period_state)
        else:
            # LLMDecisionEngine accepts duck-typed config (uses getattr for v2 fields)
            self._llm_engine = LLMDecisionEngine(config=config, rng=self.rng)  # type: ignore[arg-type]
            self._solver = self._create_solver(config, period_state)

        # Entrepreneurial solver (used in both benchmark and LLM modes)
        self._entre_solver = EntrepreneurialSolver(
            config=config,
            ability_grid=period_state.shocks.ability_grid,
            ability_transition_matrix=period_state.shocks.ability_transition_matrix,
        )

        # Observer (REQ-030, REQ-031, REQ-032)
        self.observer = ObserverV2(config)
        # Convenience alias — shares same list object as observer.history
        self.history = self.observer.history

        # Track log(A) for aggregate TFP AR(1) process
        self._prev_log_a: float = 0.0  # log(1.0) = 0

        # Firm ID counter
        self._next_firm_id: int = 0

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

    def run(self) -> SimulationOutputV2:
        """Execute all periods and return the final simulation output.

        Returns:
            SimulationOutputV2 with final state, history, and welfare summary.
        """
        log.info("simulation_v2.started", max_periods=self.config.max_periods)

        for t in range(1, self.config.max_periods + 1):
            self.period_state = self._advance_period(t)

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

        # Step 1: Draw shocks
        households, shocks = self._draw_shocks(t, households, shocks)

        # Step 2: Clear markets
        market = self._clear_markets(households, firms, shocks)

        # Step 3: Collect decisions
        econ_decisions, entre_decisions = self._collect_decisions(
            households, firms, market, constitution
        )

        # Step 4: Validate constraints
        econ_decisions = self._validate_constraints(
            econ_decisions, households, market, constitution
        )

        # Step 4b: Set labor_supply and income on households from validated decisions
        # so that Step 6 (enforce_taxes) can compute taxes on actual income.
        for h in households:
            decision = econ_decisions.get(h.id, EconomicDecision(consumption=0.0, leisure=0.5))
            h.labor_supply = 1.0 - decision.leisure
            h.income = market.wage * h.productivity * h.labor_supply

        # Step 5: Execute production
        firms = self._execute_production(firms, entre_decisions, households, market)

        # Step 6: Enforce constitution
        households, public_goods = self._enforce_constitution(
            households, firms, constitution, market
        )

        # Step 7: Process governance
        proposals, votes, constitution = self._process_governance(
            t, households, constitution, market
        )

        # Step 8: Update states
        households = self._update_states(households, econ_decisions, market, public_goods, firms)

        # Update market aggregates with actual post-Step-8 values
        market = market.model_copy(
            update={
                "aggregate_consumption": sum(h.consumption for h in households),
            }
        )

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
        # Economic decisions
        if self._llm_engine is not None:
            econ_decisions = self._llm_engine.collect_economic_decisions(
                households=households,
                market=market,
                constitution=constitution,
            )
        else:
            # Extract flat tax rate from constitution
            tax_rate = self._get_tax_rate(constitution)
            public_goods = self.constitution_engine.enforce_public_goods(
                constitution, 0.0, len(households)
            )
            transfer = 0.0  # Will be computed in step 6
            econ_decisions = self._solver.solve_all(
                households=households,
                market=market,
                constitution_tax_rate=tax_rate,
                public_goods=public_goods,
                transfer=transfer,
            )

        # Entrepreneurial decisions
        if self._llm_engine is not None:
            entre_decisions = self._llm_engine.collect_entrepreneurial_decisions(
                households=households,
                firms=firms,
                market=market,
                constitution=constitution,
            )
        else:
            # Benchmark mode: solver-driven entrepreneurial decisions
            entre_public_goods = self.constitution_engine.enforce_public_goods(
                constitution, 0.0, len(households)
            )
            # Pass VFI value function for Bellman occ choice (REQ-110)
            vfi_vf, vfi_ag = self._solver.get_value_function()
            entre_decisions = self._entre_solver.solve_all(
                households, firms, market,
                public_goods=entre_public_goods,
                vfi_value_func=vfi_vf, a_grid=vfi_ag,
            )

        log.debug(
            "step3.decisions_collected",
            num_economic=len(econ_decisions),
            num_entrepreneurial=len(entre_decisions),
        )

        return econ_decisions, entre_decisions

    # ------------------------------------------------------------------
    # Step 4: Validate constraints (REQ-004, 005, PROP-002, PROP-004)
    # ------------------------------------------------------------------

    def _validate_constraints(
        self,
        decisions: dict[str, EconomicDecision],
        households: list[HouseholdState],
        market: MarketState,
        constitution: ConstitutionV2,
    ) -> dict[str, EconomicDecision]:
        """Project economic decisions onto the feasible set.

        Ensures consumption and leisure are within budget constraints.

        Args:
            decisions: Raw economic decisions from step 3.
            households: Current household states.
            market: Current market equilibrium.
            constitution: Current constitution.

        Returns:
            Constrained economic decisions.
        """
        tax_rate = self._get_tax_rate(constitution)

        constrained: dict[str, EconomicDecision] = {}
        for h in households:
            decision = decisions.get(h.id, EconomicDecision(consumption=0.0, leisure=0.5))
            income = market.wage * h.productivity * (1.0 - decision.leisure)
            tax = income * tax_rate
            transfer = 0.0  # Conservative; actual transfers computed in step 6
            budget = compute_budget(h, market.wage, market.interest_rate, tax, transfer)
            constrained[h.id] = enforce_budget_constraint(decision, h, budget, self.config.a_min)

        log.debug("step4.constraints_validated", num_agents=len(constrained))
        return constrained

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
                    # Capital will be returned in step 8
                    log.debug(
                        "step5.firm_liquidated",
                        firm_id=firm.id,
                        returned_capital=returned_capital,
                    )
                continue

            # Produce output (REQ-007)
            output = produce_output(firm, self.config.alpha)

            # Distribute income (REQ-008)
            wages_paid, capital_cost, profit = distribute_firm_income(
                firm.model_copy(update={"output": output}),
                market.wage,
                market.interest_rate,
                self.config.delta,
            )

            # Update firm with results
            firm_update = {
                "output": output,
                "profit": profit,
                "labor_demand": owner_decision.labor_demand
                if owner_decision.labor_demand > 0
                else firm.labor_demand,
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

        log.debug(
            "step6.constitution_enforced",
            revenue=round(revenue, 2),
            public_goods=round(public_goods, 4),
            transfer_revenue=round(transfer_revenue, 2),
        )

        return households, public_goods

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
                proposal = decide_proposal_v2(
                    h, constitution, self.config.num_agents, self.rng
                )
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

            # Collect votes from all households
            if proposals:
                for h in households:
                    agent_votes = decide_votes_v2(h, proposals, constitution)
                    if agent_votes:
                        all_votes[h.id] = agent_votes

        # Tally votes for each proposal
        vote_outcomes: list[VoteOutcomeV2] = []
        total_eligible = len(households)

        for proposal in proposals:
            proposal_votes: dict[str, bool] = {}
            for agent_id, agent_vote_map in all_votes.items():
                vote = agent_vote_map.get(proposal.rule_name)
                if vote is not None:
                    proposal_votes[agent_id] = vote

            outcome = tally_votes_v2(proposal, proposal_votes, constitution, total_eligible)
            vote_outcomes.append(outcome)
            self._recent_vote_outcomes.append(
                {
                    "period": t,
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

            # Income from labor
            labor_income = market.wage * h.productivity * labor_supply

            # Add firm profit for entrepreneurs
            firm_profit = firm_profit_by_owner.get(h.id, 0.0)
            total_income = labor_income + firm_profit

            # Full DSGE-HA budget constraint:
            # a' = (1+r)*a + w*z*(1-l) + firm_profit - c - taxes + transfers
            new_wealth = (
                (1.0 + market.interest_rate) * h.wealth
                + total_income
                - consumption
                - h.taxes_paid
                + h.transfers_received
            )
            new_wealth = max(new_wealth, self.config.a_min)
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
