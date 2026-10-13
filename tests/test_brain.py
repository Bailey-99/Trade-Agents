from __future__ import annotations

from types import SimpleNamespace

import pytest

from kraken_agent import strategy
from kraken_agent.brain import ClaudeBrain, DecisionContext, LLMOrder, LLMVerdict, RulesBrain

from .conftest import make_candles, trend_then_breakout


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(verdict=None, stop_reason="end_turn", error=None):
    response = SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        parsed_output=verdict,
        usage=SimpleNamespace(input_tokens=2000, output_tokens=800),
    )
    messages = FakeMessages(response, error)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


@pytest.fixture
def ctx(settings):
    f = strategy.compute_features("XBTEUR", make_candles(trend_then_breakout()), settings)
    signal = strategy.entry_signal(f, settings)
    assert signal is not None
    return DecisionContext(
        now=1_700_000_000,
        mode="paper",
        equity=10.0,
        cash=10.0,
        initial_capital=10.0,
        survival_floor=7.5,
        can_open=True,
        open_block_reason="",
        free_slots=1,
        candidates=[signal],
        positions=[{"pair": "ETHEUR", "pnl_pct": -1.0}],
        soft_exits={"ETHEUR": ["tendance cassée"]},
        recent_trades=[],
        settings=settings,
        llm_usage={},
    )


def order(action, pair, size=1.0):
    return LLMOrder(action=action, pair=pair, size_fraction=size, confidence=0.7, reasoning="test")


def test_rules_brain_sells_soft_exits_and_buys_best_candidate(ctx):
    decisions = RulesBrain().decide(ctx)
    assert [(d.action, d.pair) for d in decisions] == [("sell", "ETHEUR"), ("buy", "XBTEUR")]


def test_claude_request_shape_and_validation(ctx, settings):
    verdict = LLMVerdict(
        market_view="vue",
        orders=[order("buy", "XBT/EUR", 3.0), order("buy", "DOGEEUR"), order("sell", "ETHEUR"), order("sell", "SOLEUR")],
    )
    client, messages = fake_client(verdict)
    brain = ClaudeBrain(settings, client=client)
    decisions = brain.decide(ctx)
    assert [(d.action, d.pair) for d in decisions] == [("buy", "XBTEUR"), ("sell", "ETHEUR")]
    assert decisions[0].size_fraction == 1.0  # borné à [0.25, 1]
    call = messages.calls[0]
    assert call["model"] == settings.claude_model
    assert call["output_format"] is LLMVerdict
    assert call["output_config"] == {"effort": settings.claude_effort}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "XBTEUR" in call["messages"][0]["content"]
    assert ctx.llm_usage["calls_today"] == 1
    assert ctx.llm_usage["est_cost_usd"] > 0


def test_claude_cannot_buy_when_risk_manager_blocks(ctx, settings):
    ctx.can_open = False
    ctx.open_block_reason = "pause"
    client, _ = fake_client(LLMVerdict(market_view="", orders=[order("buy", "XBTEUR")]))
    decisions = ClaudeBrain(settings, client=client).decide(ctx)
    assert all(d.action != "buy" for d in decisions)


def test_claude_refusal_or_error_means_no_new_buys(ctx, settings):
    client, _ = fake_client(None, stop_reason="refusal")
    decisions = ClaudeBrain(settings, client=client).decide(ctx)
    assert [(d.action, d.pair) for d in decisions] == [("sell", "ETHEUR")]
    client, _ = fake_client(error=RuntimeError("réseau"))
    decisions = ClaudeBrain(settings, client=client).decide(ctx)
    assert [(d.action, d.pair) for d in decisions] == [("sell", "ETHEUR")]


def test_claude_daily_budget(ctx, settings):
    settings.max_llm_calls_per_day = 1
    client, messages = fake_client(LLMVerdict(market_view="", orders=[]))
    brain = ClaudeBrain(settings, client=client)
    brain.decide(ctx)
    brain.decide(ctx)
    assert len(messages.calls) == 1


def test_claude_not_called_when_nothing_to_decide(ctx, settings):
    ctx.candidates = []
    ctx.soft_exits = {}
    client, messages = fake_client(LLMVerdict(market_view="", orders=[]))
    assert ClaudeBrain(settings, client=client).decide(ctx) == []
    assert messages.calls == []


def test_haiku_requests_skip_server_fallback(ctx, settings):
    settings.claude_model = "claude-haiku-5-5"
    client, messages = fake_client(LLMVerdict(market_view="", orders=[]))
    ClaudeBrain(settings, client=client).decide(ctx)
    assert "fallbacks" not in messages.calls[0]
