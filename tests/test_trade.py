"""Tests for bilateral trade — model, decision logic, application, and integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emergent_constitution.citizen import decide_trade
from emergent_constitution.config import SimulationConfig
from emergent_constitution.economics import apply_trades
from emergent_constitution.lead import Lead
from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import Constitution
from emergent_constitution.models.proposal import TradeOffer
from emergent_constitution.rng import SimulationRNG


def _make_agent(agent_id: str, wealth: float, productivity: float) -> AgentState:
    """Helper to create an agent with minimal boilerplate."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


# ---------------------------------------------------------------------------
# TradeOffer model validation
# ---------------------------------------------------------------------------


class TestTradeOffer:
    def test_valid_trade_offer(self):
        offer = TradeOffer(seller_id="a", buyer_id="b", amount=10.0)
        assert offer.seller_id == "a"
        assert offer.buyer_id == "b"
        assert offer.amount == 10.0

    def test_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            TradeOffer(seller_id="a", buyer_id="b", amount=0.0)

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            TradeOffer(seller_id="a", buyer_id="b", amount=-5.0)

    def test_small_positive_amount_accepted(self):
        offer = TradeOffer(seller_id="a", buyer_id="b", amount=0.001)
        assert offer.amount == 0.001


# ---------------------------------------------------------------------------
# decide_trade logic
# ---------------------------------------------------------------------------


class TestDecideTrade:
    def test_returns_valid_offer_or_none(self):
        """decide_trade returns a TradeOffer or None for each agent."""
        # Set up: seller has high productivity, low wealth; buyer has high wealth
        agents = [
            _make_agent("seller", wealth=30.0, productivity=20.0),
            _make_agent("buyer", wealth=200.0, productivity=5.0),
        ]
        constitution = Constitution()

        # Run many attempts — should eventually get both outcomes
        results = []
        for i in range(100):
            result = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=i))
            results.append(result)

        non_none = [r for r in results if r is not None]
        nones = [r for r in results if r is None]

        # With probabilistic gating, we expect both outcomes
        assert len(non_none) > 0, "Should produce at least one trade offer"
        assert len(nones) > 0, "Should skip some trades"

        # Validate returned offers
        for offer in non_none:
            assert isinstance(offer, TradeOffer)
            assert offer.seller_id == "seller"
            assert offer.buyer_id == "buyer"
            assert offer.amount > 0.0

    def test_deterministic_with_seed(self):
        """Same seed produces identical trade decisions."""
        agents = [
            _make_agent("seller", wealth=30.0, productivity=20.0),
            _make_agent("buyer", wealth=200.0, productivity=5.0),
        ]
        constitution = Constitution()

        result1 = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=42))
        result2 = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=42))
        assert result1 == result2

    def test_respects_budget(self):
        """Trade amount should not exceed buyer's wealth."""
        agents = [
            _make_agent("seller", wealth=10.0, productivity=20.0),
            _make_agent("buyer", wealth=50.0, productivity=5.0),
        ]
        constitution = Constitution()

        for i in range(100):
            result = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=i))
            if result is not None:
                assert result.amount <= agents[1].wealth

    def test_agent_not_both_buyer_and_seller(self):
        """An agent should not trade with itself."""
        agents = [
            _make_agent("a", wealth=30.0, productivity=20.0),
            _make_agent("b", wealth=200.0, productivity=5.0),
        ]
        constitution = Constitution()

        for i in range(100):
            result = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=i))
            if result is not None:
                assert result.seller_id != result.buyer_id

    def test_single_agent_returns_none(self):
        """With only one agent, no trade is possible."""
        agents = [_make_agent("solo", wealth=30.0, productivity=20.0)]
        constitution = Constitution()
        rng = SimulationRNG(seed=42)

        result = decide_trade(agents[0], agents, constitution, rng)
        assert result is None

    def test_no_eligible_seller_returns_none(self):
        """Agent with below-median productivity should not sell."""
        agents = [
            _make_agent("low_prod", wealth=30.0, productivity=3.0),
            _make_agent("high_prod", wealth=200.0, productivity=20.0),
        ]
        constitution = Constitution()

        for i in range(100):
            result = decide_trade(agents[0], agents, constitution, SimulationRNG(seed=i))
            assert result is None, "Low-productivity agent should not initiate trade"


# ---------------------------------------------------------------------------
# apply_trades
# ---------------------------------------------------------------------------


class TestApplyTrades:
    def test_wealth_conservation(self):
        """Total wealth is preserved after applying valid trades."""
        agents = [
            _make_agent("seller", wealth=50.0, productivity=10.0),
            _make_agent("buyer", wealth=200.0, productivity=10.0),
        ]
        trades = [TradeOffer(seller_id="seller", buyer_id="buyer", amount=20.0)]

        total_before = sum(a.wealth for a in agents)
        result = apply_trades(agents, trades)
        total_after = sum(a.wealth for a in result)

        assert total_after == pytest.approx(total_before)

    def test_correct_transfer(self):
        """Buyer loses and seller gains the trade amount."""
        agents = [
            _make_agent("seller", wealth=50.0, productivity=10.0),
            _make_agent("buyer", wealth=200.0, productivity=10.0),
        ]
        trades = [TradeOffer(seller_id="seller", buyer_id="buyer", amount=20.0)]

        result = apply_trades(agents, trades)
        result_map = {a.id: a for a in result}

        assert result_map["seller"].wealth == pytest.approx(70.0)
        assert result_map["buyer"].wealth == pytest.approx(180.0)

    def test_invalid_trade_insufficient_funds_skipped(self):
        """Trade is skipped when buyer cannot afford the amount."""
        agents = [
            _make_agent("seller", wealth=50.0, productivity=10.0),
            _make_agent("buyer", wealth=5.0, productivity=10.0),
        ]
        trades = [TradeOffer(seller_id="seller", buyer_id="buyer", amount=20.0)]

        result = apply_trades(agents, trades)
        result_map = {a.id: a for a in result}

        # No transfer happened
        assert result_map["seller"].wealth == pytest.approx(50.0)
        assert result_map["buyer"].wealth == pytest.approx(5.0)

    def test_unknown_agent_skipped(self):
        """Trade with nonexistent agent ID is silently skipped."""
        agents = [_make_agent("a", wealth=100.0, productivity=10.0)]
        trades = [TradeOffer(seller_id="a", buyer_id="ghost", amount=10.0)]

        result = apply_trades(agents, trades)
        assert result[0].wealth == pytest.approx(100.0)

    def test_no_mutation_of_inputs(self):
        """Original agent list is not mutated."""
        agents = [
            _make_agent("seller", wealth=50.0, productivity=10.0),
            _make_agent("buyer", wealth=200.0, productivity=10.0),
        ]
        original_wealths = [a.wealth for a in agents]
        trades = [TradeOffer(seller_id="seller", buyer_id="buyer", amount=20.0)]

        apply_trades(agents, trades)

        for orig_w, agent in zip(original_wealths, agents, strict=True):
            assert agent.wealth == orig_w

    def test_empty_trades_no_change(self):
        """Empty trade list returns copies with same wealth."""
        agents = [_make_agent("a", wealth=100.0, productivity=10.0)]
        result = apply_trades(agents, [])
        assert result[0].wealth == pytest.approx(100.0)

    def test_multiple_trades_applied_sequentially(self):
        """Multiple trades are applied in order."""
        agents = [
            _make_agent("a", wealth=100.0, productivity=10.0),
            _make_agent("b", wealth=100.0, productivity=10.0),
            _make_agent("c", wealth=100.0, productivity=10.0),
        ]
        trades = [
            TradeOffer(seller_id="a", buyer_id="b", amount=10.0),
            TradeOffer(seller_id="b", buyer_id="c", amount=20.0),
        ]

        result = apply_trades(agents, trades)
        result_map = {a.id: a for a in result}

        # a: 100 + 10 = 110
        # b: 100 - 10 + 20 = 110 (wait, b sells to c in trade 2, so b receives)
        # Actually: trade 1: a gets 10 from b. b: 100-10=90, a: 100+10=110
        # trade 2: b gets 20 from c. b: 90+20=110, c: 100-20=80
        assert result_map["a"].wealth == pytest.approx(110.0)
        assert result_map["b"].wealth == pytest.approx(110.0)
        assert result_map["c"].wealth == pytest.approx(80.0)

    def test_second_trade_fails_if_first_drains_buyer(self):
        """Sequential application means a later trade can fail if funds drained."""
        agents = [
            _make_agent("a", wealth=50.0, productivity=10.0),
            _make_agent("b", wealth=30.0, productivity=10.0),
        ]
        trades = [
            TradeOffer(seller_id="a", buyer_id="b", amount=25.0),
            TradeOffer(seller_id="a", buyer_id="b", amount=25.0),  # b only has 5 left
        ]

        result = apply_trades(agents, trades)
        result_map = {a.id: a for a in result}

        # First trade succeeds: a=75, b=5
        # Second trade fails (b has 5 < 25): no change
        assert result_map["a"].wealth == pytest.approx(75.0)
        assert result_map["b"].wealth == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Integration with Lead tick loop
# ---------------------------------------------------------------------------


class TestTradeIntegration:
    def test_lead_runs_with_trades(self):
        """Lead completes with trade_interval enabled."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=20,
            seed=42,
            trade_interval=5,
        )
        lead = Lead(config)
        output = lead.run()
        assert output.total_ticks == 20
        assert len(output.final_agent_states) == 10

    def test_wealth_stays_positive(self):
        """All agents maintain non-negative wealth with active trading."""
        config = SimulationConfig(
            num_agents=20,
            max_ticks=50,
            seed=42,
            trade_interval=3,
            proposal_interval=5,
        )
        output = Lead(config).run()
        for agent in output.final_agent_states:
            assert agent.wealth >= 0.0, f"Agent {agent.id} has negative wealth: {agent.wealth}"

    def test_deterministic_with_trades(self):
        """Two runs with same seed produce identical output including trades."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=30,
            seed=42,
            trade_interval=5,
        )
        output1 = Lead(config).run()
        output2 = Lead(config).run()
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 == json2

    def test_trades_on_interval_ticks_only(self):
        """Trades should only be collected on trade-interval ticks."""
        config = SimulationConfig(
            num_agents=10,
            max_ticks=20,
            seed=42,
            trade_interval=5,
        )
        lead = Lead(config)
        for tick in range(1, 21):
            lead.tick_state = lead._advance_tick(tick)
            if tick % 5 != 0:
                assert lead.tick_state.trades_this_tick == [], (
                    f"Trades found on non-interval tick {tick}"
                )

    def test_trade_interval_config_default(self):
        """Default trade_interval is 5."""
        config = SimulationConfig()
        assert config.trade_interval == 5

    def test_trade_interval_config_validation(self):
        """trade_interval must be >= 1."""
        with pytest.raises(ValidationError):
            SimulationConfig(trade_interval=0)
