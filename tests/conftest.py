from __future__ import annotations

import math

import pytest

from kraken_agent.config import Settings
from kraken_agent.models import Candle, PairInfo


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, candle_interval=60, brain="rules")


@pytest.fixture
def btc_pair() -> PairInfo:
    return PairInfo(
        name="XXBTZEUR",
        altname="XBTEUR",
        wsname="XBT/EUR",
        base="XXBT",
        quote="ZEUR",
        pair_decimals=1,
        lot_decimals=8,
        ordermin=0.00005,
        costmin=0.5,
    )


def make_candles(closes: list[float], *, interval: int = 60, start: int = 1_700_000_400, wick: float = 0.002) -> list[Candle]:
    t0 = start - start % (interval * 60)
    candles = []
    prev = closes[0]
    for i, close in enumerate(closes):
        high = max(prev, close) * (1 + wick)
        low = min(prev, close) * (1 - wick)
        candles.append(Candle(t0 + i * interval * 60, prev, high, low, close, (high + low + close) / 3, 100.0, 10))
        prev = close
    return candles


def trend_then_breakout(n: int = 260, start: float = 100.0) -> list[float]:
    """Hausse avec respirations (RSI ~70), puis cassure nette sur la dernière bougie."""
    closes = []
    price = start
    for i in range(n - 1):
        price *= 1.001 + 0.012 * math.sin(i / 2.0)
        closes.append(price)
    closes.append(max(closes[-21:]) * 1.01)
    return closes
