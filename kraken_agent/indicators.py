"""Indicateurs techniques en Python pur (pas de dépendance à numpy/pandas)."""

from __future__ import annotations

from typing import Sequence

from .models import Candle


def ema(values: Sequence[float], period: int) -> list[float]:
    """Moyenne mobile exponentielle (équivalent pandas `ewm(span=period, adjust=False)`)."""
    if not values:
        return []
    k = 2 / (period + 1)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(value * k + out[-1] * (1 - k))
    return out


def rsi(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """RSI de Wilder. Les `period` premières valeurs sont None."""
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def value(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100 - 100 / (1 + g / l)

    out[period] = value(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = value(avg_gain, avg_loss)
    return out


def true_ranges(candles: Sequence[Candle]) -> list[float]:
    out = []
    for i, c in enumerate(candles):
        if i == 0:
            out.append(c.high - c.low)
        else:
            prev = candles[i - 1].close
            out.append(max(c.high - c.low, abs(c.high - prev), abs(c.low - prev)))
    return out


def atr(candles: Sequence[Candle], period: int = 14) -> list[float | None]:
    """Average True Range de Wilder. Les `period - 1` premières valeurs sont None."""
    out: list[float | None] = [None] * len(candles)
    if len(candles) < period:
        return out
    tr = true_ranges(candles)
    current = sum(tr[:period]) / period
    out[period - 1] = current
    for i in range(period, len(candles)):
        current = (current * (period - 1) + tr[i]) / period
        out[i] = current
    return out


def pct_change(values: Sequence[float], lookback: int) -> float:
    """Variation en % entre la dernière valeur et celle `lookback` périodes avant."""
    if len(values) <= lookback or values[-1 - lookback] == 0:
        return 0.0
    return (values[-1] / values[-1 - lookback] - 1) * 100
