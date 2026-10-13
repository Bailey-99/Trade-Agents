from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_agent.agent import TradingAgent
from kraken_agent.backtest import SimulatedKraken, run_backtest, synthetic_candles
from kraken_agent.brain import RulesBrain
from kraken_agent.broker import LiveBroker
from kraken_agent.state import StateStore

from .conftest import make_candles, trend_then_breakout
from .fakes import FakeLiveKraken


@pytest.fixture
def candles():
    return {"XBTEUR": make_candles(trend_then_breakout(260, start=100.0))}


def close_time(candles, index=-1):
    c = candles["XBTEUR"][index]
    return c.time + 3600


def paper_agent(settings, sim, tmp_path=None):
    s = replace(settings, universe=["XBTEUR"])
    store = StateStore(tmp_path, "paper")
    return TradingAgent(s, sim, store, brain=RulesBrain(), clock=lambda: sim.now)


def test_paper_agent_buys_on_signal_and_stops_out(settings, candles, tmp_path):
    sim = SimulatedKraken(candles, interval=60)
    sim.now = close_time(candles)
    agent = paper_agent(settings, sim, tmp_path)
    agent.step()
    assert len(agent.state.positions) == 1
    pos = agent.state.positions[0]
    assert pos.stop < pos.entry_price
    # Le risque réel au stop reste proche du budget (2% de 10€).
    loss_at_stop = pos.entry_cost - pos.volume * pos.stop * (1 - settings.taker_fee_pct / 100)
    assert loss_at_stop <= 10.0 * settings.risk_per_trade_pct / 100 * 2
    cash_after_buy = agent.state.paper_cash
    assert cash_after_buy < 10.0

    sim.price_override["XBTEUR"] = pos.stop * 0.99
    agent.step()
    assert agent.state.positions == []
    assert agent.state.stats["trades"] == 1 and agent.state.stats["losses"] == 1
    assert 9.5 < agent.state.paper_cash < 10.0
    assert agent.state.paper_holdings == {}
    # L'état est persisté sur disque et rechargeable.
    reloaded = StateStore(tmp_path, "paper").load()
    assert reloaded.stats["trades"] == 1


def test_survival_floor_liquidates_and_halts(settings, candles):
    sim = SimulatedKraken(candles, interval=60)
    sim.now = close_time(candles)
    agent = paper_agent(replace(settings, max_position_pct=100, risk_per_trade_pct=10), sim)
    agent.step()
    assert agent.state.positions
    # Simule un krach brutal sans passage par le stop (gap) : le capital passe sous le plancher.
    pos = agent.state.positions[0]
    pos.stop = 0.0
    sim.price_override["XBTEUR"] = pos.entry_price * 0.5
    agent.step()
    assert agent.state.status == "halted"
    assert agent.state.positions == []
    assert agent.state.paper_holdings == {}
    agent.step()  # un agent arrêté ne fait plus rien
    assert agent.state.stats["trades"] == 1


def test_live_agent_places_server_stop_and_detects_its_execution(settings, candles, tmp_path):
    kraken = FakeLiveKraken(candles, interval=60, eur=10.0)
    kraken.now = close_time(candles)
    s = replace(settings, mode="live", universe=["XBTEUR"])
    agent = TradingAgent(
        s, kraken, StateStore(tmp_path, "live"), brain=RulesBrain(),
        broker=LiveBroker(kraken, s, sleep=lambda x: None), clock=lambda: kraken.now,
    )  # fmt: skip
    agent.step()
    assert len(agent.state.positions) == 1
    pos = agent.state.positions[0]
    buy, stop = kraken.added[0], kraken.added[1]
    assert buy["ordertype"] == "market" and buy["oflags"] == "fciq"
    assert stop["ordertype"] == "stop-loss" and stop["type"] == "sell"
    assert float(stop["price"]) == pytest.approx(pos.stop, rel=1e-3)
    assert pos.stop_txid

    # Le prix chute : Kraken exécute le stop pendant que l'agent est « hors ligne ».
    kraken.price_override["XBTEUR"] = pos.stop * 0.995
    kraken.trigger_stops()
    agent.step()
    assert agent.state.positions == []
    assert agent.state.stats["trades"] == 1
    assert kraken.wallet.get("XBT", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_live_close_cancels_server_stop_before_selling(settings, candles, tmp_path):
    kraken = FakeLiveKraken(candles, interval=60, eur=10.0)
    kraken.now = close_time(candles)
    s = replace(settings, mode="live", universe=["XBTEUR"])
    agent = TradingAgent(
        s, kraken, StateStore(tmp_path, "live"), brain=RulesBrain(),
        broker=LiveBroker(kraken, s, sleep=lambda x: None), clock=lambda: kraken.now,
    )  # fmt: skip
    agent.step()
    pos = agent.state.positions[0]
    stop_txid = pos.stop_txid
    agent.panic()
    assert stop_txid in kraken.canceled
    assert kraken.added[-1]["ordertype"] == "market" and kraken.added[-1]["type"] == "sell"
    assert agent.state.status == "halted"
    assert kraken.wallet["ZEUR"] == pytest.approx(10.0, rel=0.02)


def test_live_trailing_stop_is_replaced_on_kraken(settings, candles, tmp_path):
    kraken = FakeLiveKraken(candles, interval=60, eur=10.0)
    kraken.now = close_time(candles)
    s = replace(settings, mode="live", universe=["XBTEUR"])
    agent = TradingAgent(
        s, kraken, StateStore(tmp_path, "live"), brain=RulesBrain(),
        broker=LiveBroker(kraken, s, sleep=lambda x: None), clock=lambda: kraken.now,
    )  # fmt: skip
    agent.step()
    pos = agent.state.positions[0]
    old_txid, old_stop = pos.stop_txid, pos.stop
    kraken.price_override["XBTEUR"] = pos.entry_price * 1.10
    agent.step()
    assert pos.stop > old_stop
    assert old_txid in kraken.canceled
    assert kraken.orders[pos.stop_txid]["status"] == "open"
    assert float(kraken.orders[pos.stop_txid]["stopprice"]) == pytest.approx(pos.stop, rel=1e-3)


def test_backtest_runs_and_never_breaks_accounting(settings):
    data = {f"S{i}EUR": synthetic_candles(900, start_price=50 + 30 * i, seed=i) for i in range(3)}
    result = run_backtest(data, settings)
    assert result.candles > 700
    assert result.end_equity > 0
    assert result.max_drawdown_pct < 100 - settings.survival_floor_pct + 5
    for trade in result.trades:
        assert trade["fee"] > 0


def test_simulator_never_leaks_future_candles(candles):
    sim = SimulatedKraken(candles, interval=60)
    sim.now = candles["XBTEUR"][100].time + 3600
    rows = sim.ohlc("XBTEUR")["XBTEUR"]
    assert rows[-1][0] == candles["XBTEUR"][100].time
