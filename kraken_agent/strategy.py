"""Scanner de signaux : suivi de tendance + cassure, stops basés sur l'ATR.

Le scanner ne passe aucun ordre. Il propose des candidats à l'achat et
signale les positions à surveiller ; le « cerveau » (règles ou Claude) décide,
puis le gestionnaire de risque valide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from . import indicators as ind
from .config import Settings
from .models import Candle, Position

FEATURE_WINDOW = 300  # bougies utilisées pour les calculs (suffisant pour une EMA 100)


@dataclass
class Features:
    pair: str
    candle_time: int
    close: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    rsi: float
    atr: float
    atr_pct: float
    ret_short_pct: float  # variation sur 24 bougies
    ret_long_pct: float  # variation sur 168 bougies
    donchian_high: float  # plus haut des N bougies précédentes
    breakout: bool
    volume_ratio: float  # volume de la dernière bougie / moyenne
    recent_closes: list[float] = field(default_factory=list)

    @property
    def trend_up(self) -> bool:
        return self.close > self.ema_trend and self.ema_fast > self.ema_slow

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        last = self.close
        # Trajectoire récente exprimée en % par rapport au dernier prix : compact et lisible.
        data["recent_closes_pct"] = [round((c / last - 1) * 100, 2) for c in self.recent_closes]
        del data["recent_closes"]
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 6 if value < 1 else 4)
        return data


@dataclass
class EntrySignal:
    pair: str
    score: float
    price: float
    stop_pct: float
    target_pct: float
    reasons: list[str]
    features: Features


def compute_features(pair: str, candles: Sequence[Candle], s: Settings) -> Features | None:
    """Calcule les indicateurs sur des bougies CLÔTURÉES. None si historique insuffisant."""
    candles = list(candles)[-FEATURE_WINDOW:]
    needed = max(s.ema_trend, s.breakout_lookback + 1, s.atr_period + 1, 30)
    if len(candles) < needed:
        return None
    closes = [c.close for c in candles]
    atr_values = ind.atr(candles, s.atr_period)
    rsi_values = ind.rsi(closes, s.rsi_period)
    if atr_values[-1] is None or rsi_values[-1] is None:
        return None
    last = candles[-1]
    previous = candles[-1 - s.breakout_lookback : -1]
    donchian_high = max(c.high for c in previous)
    volumes = [c.volume for c in candles[-21:-1]]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
    atr_value = float(atr_values[-1])
    return Features(
        pair=pair,
        candle_time=last.time,
        close=last.close,
        ema_fast=ind.ema(closes, s.ema_fast)[-1],
        ema_slow=ind.ema(closes, s.ema_slow)[-1],
        ema_trend=ind.ema(closes, s.ema_trend)[-1],
        rsi=float(rsi_values[-1]),
        atr=atr_value,
        atr_pct=atr_value / last.close * 100 if last.close else 0.0,
        ret_short_pct=ind.pct_change(closes, 24),
        ret_long_pct=ind.pct_change(closes, 168),
        donchian_high=donchian_high,
        breakout=last.close > donchian_high,
        volume_ratio=last.volume / avg_volume if avg_volume > 0 else 1.0,
        recent_closes=closes[-24:],
    )


def stop_distance_pct(f: Features, s: Settings) -> float:
    return min(max(s.atr_stop_mult * f.atr_pct, s.min_stop_pct), s.max_stop_pct)


def entry_signal(f: Features, s: Settings) -> EntrySignal | None:
    """Signal d'achat : tendance haussière + cassure + momentum raisonnable."""
    if not f.trend_up:
        return None
    if not (s.rsi_min <= f.rsi <= s.rsi_max):
        return None
    if not f.breakout or f.ret_long_pct <= 0:
        return None
    if not (s.min_atr_pct <= f.atr_pct <= s.max_atr_pct):
        return None
    stop_pct = stop_distance_pct(f, s)
    target_pct = s.reward_risk * stop_pct
    # Le gain visé doit couvrir largement les frais d'un aller-retour.
    if target_pct < 4 * s.round_trip_cost_pct:
        return None
    # Score = momentum ajusté du risque + bonus de volume (plafonné).
    score = f.ret_long_pct / max(f.atr_pct, 0.1) + 0.5 * min(f.volume_ratio, 3.0)
    reasons = [
        f"tendance haussière (EMA{s.ema_fast} > EMA{s.ema_slow}, prix > EMA{s.ema_trend})",
        f"cassure du plus haut {s.breakout_lookback} bougies",
        f"RSI {f.rsi:.0f}",
        f"ATR {f.atr_pct:.2f}%",
    ]
    return EntrySignal(
        pair=f.pair,
        score=round(score, 3),
        price=f.close,
        stop_pct=round(stop_pct, 3),
        target_pct=round(target_pct, 3),
        reasons=reasons,
        features=f,
    )


def trailing_stop(pos: Position, price: float, s: Settings) -> float:
    """Nouveau niveau de stop (ne descend jamais)."""
    highest = max(pos.highest, price)
    candidate = pos.stop
    if pos.atr > 0:
        candidate = max(candidate, highest - s.atr_trail_mult * pos.atr)
    # Dès +1R de gain latent, le stop remonte au point mort (frais compris).
    one_r = pos.entry_price - pos.initial_stop
    if one_r > 0 and highest >= pos.entry_price + one_r:
        breakeven = pos.entry_price * (1 + s.round_trip_cost_pct / 100)
        candidate = max(candidate, breakeven)
    # Le stop doit rester sous le prix courant, sinon il serait déclenché immédiatement.
    if candidate >= price:
        candidate = pos.stop
    return max(pos.stop, candidate)


def soft_exit_reasons(pos: Position, f: Features | None, now: float, s: Settings) -> list[str]:
    """Signaux de sortie « discutables » (le cerveau peut choisir de garder la position)."""
    reasons: list[str] = []
    if f is not None:
        if f.ema_fast < f.ema_slow:
            reasons.append(f"tendance cassée (EMA{s.ema_fast} < EMA{s.ema_slow})")
        if f.close < f.ema_trend:
            reasons.append(f"prix sous l'EMA{s.ema_trend}")
        price = f.close
    else:
        price = pos.entry_price
    held_bars = (now - pos.opened_at) / s.interval_seconds
    if held_bars >= s.max_hold_bars and pos.pnl_pct(price) < s.round_trip_cost_pct:
        reasons.append(f"position stagnante depuis {held_bars:.0f} bougies")
    return reasons
