from __future__ import annotations

import pandas as pd
import pytest

from kraken_agent import indicators as ind
from kraken_agent import strategy
from kraken_agent.models import Position

from .conftest import make_candles, trend_then_breakout


def test_ema_matches_pandas():
    values = [float(x) for x in [10, 11, 12, 11, 13, 15, 14, 16, 18, 17]]
    expected = pd.Series(values).ewm(span=4, adjust=False).mean().tolist()
    assert ind.ema(values, 4) == pytest.approx(expected)


def test_rsi_extremes_and_range():
    up = [float(i) for i in range(1, 40)]
    assert ind.rsi(up, 14)[-1] == pytest.approx(100.0)
    down = list(reversed(up))
    assert ind.rsi(down, 14)[-1] == pytest.approx(0.0)
    zigzag = [100 + (i % 2) * 2 for i in range(40)]
    value = ind.rsi(zigzag, 14)[-1]
    assert 30 < value < 70
    assert ind.rsi(up, 14)[13] is None


def test_atr_matches_wilder_formula():
    candles = make_candles([100, 102, 101, 104, 103, 105, 107, 106], wick=0.01)
    tr = ind.true_ranges(candles)
    period = 3
    expected = sum(tr[:period]) / period
    for value in tr[period:]:
        expected = (expected * (period - 1) + value) / period
    assert ind.atr(candles, period)[-1] == pytest.approx(expected)


def test_entry_signal_on_uptrend_breakout(settings):
    f = strategy.compute_features("XBTEUR", make_candles(trend_then_breakout()), settings)
    assert f is not None and f.breakout and f.trend_up
    signal = strategy.entry_signal(f, settings)
    assert signal is not None
    assert settings.min_stop_pct <= signal.stop_pct <= settings.max_stop_pct
    assert signal.target_pct >= 4 * settings.round_trip_cost_pct


def test_no_entry_in_downtrend(settings):
    closes = [100 * (0.998**i) for i in range(260)]
    f = strategy.compute_features("XBTEUR", make_candles(closes), settings)
    assert f is not None
    assert strategy.entry_signal(f, settings) is None


def test_not_enough_history(settings):
    assert strategy.compute_features("XBTEUR", make_candles([100.0] * 50), settings) is None


def make_position(**kw):
    base = dict(
        pair="XBTEUR", base="XXBT", volume=1.0, entry_price=100.0, entry_cost=100.4, entry_fee=0.4,
        stop=97.0, initial_stop=97.0, highest=100.0, opened_at=0.0, atr=1.0,
    )  # fmt: skip
    base.update(kw)
    return Position(**base)


def test_trailing_stop_only_moves_up(settings):
    pos = make_position()
    assert strategy.trailing_stop(pos, 99.0, settings) == 97.0
    # +1R atteint : stop au point mort (frais compris) ou trailing ATR, le plus haut des deux.
    new_stop = strategy.trailing_stop(pos, 104.0, settings)
    assert new_stop >= 100.0 * (1 + settings.round_trip_cost_pct / 100)
    pos.stop = new_stop
    pos.highest = 104.0
    assert strategy.trailing_stop(pos, 101.5, settings) == new_stop


def test_trailing_stop_stays_below_price(settings):
    pos = make_position(stop=99.0, highest=110.0, atr=0.1)
    assert strategy.trailing_stop(pos, 99.5, settings) < 99.5


def test_soft_exit_on_stagnation(settings):
    pos = make_position(opened_at=0.0)
    now = settings.max_hold_bars * settings.interval_seconds + 1
    reasons = strategy.soft_exit_reasons(pos, None, now, settings)
    assert any("stagnante" in r for r in reasons)
