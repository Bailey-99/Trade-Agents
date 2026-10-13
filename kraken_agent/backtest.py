"""Backtest : rejoue l'agent (même code que le mode réel) sur des bougies historiques.

Le backtest utilise le cerveau à règles : Claude connaît l'histoire des marchés,
un backtest avec lui serait biaisé (et coûteux).
"""

from __future__ import annotations

import bisect
import csv
import logging
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .agent import TradingAgent
from .brain import RulesBrain
from .config import Settings
from .models import Candle, Ticker
from .state import StateStore

log = logging.getLogger(__name__)


class SimulatedKraken:
    """Imite les endpoints publics de Kraken à partir de bougies en mémoire."""

    def __init__(
        self,
        candles: dict[str, list[Candle]],
        *,
        interval: int = 60,
        pair_specs: dict[str, dict[str, Any]] | None = None,
        spread_pct: float = 0.06,
        max_rows: int = 720,
    ):
        self.interval = interval
        self.spread_pct = spread_pct
        self.max_rows = max_rows
        self.now = 0.0
        self.price_override: dict[str, float] = {}
        self.specs: dict[str, dict[str, Any]] = {}
        self._rows: dict[str, list[list[Any]]] = {}
        self._times: dict[str, list[int]] = {}
        self._closes: dict[str, list[float]] = {}
        self._volumes: dict[str, list[float]] = {}
        for alt, series in candles.items():
            series = sorted(series, key=lambda c: c.time)
            spec = dict((pair_specs or {}).get(alt) or default_spec(alt, series[-1].close if series else 1.0))
            spec.setdefault("altname", alt)
            self.specs[spec.get("name", alt)] = spec
            name = spec.get("name", alt)
            self._rows[name] = [
                [c.time, c.open, c.high, c.low, c.close, c.vwap, c.volume, c.count] for c in series
            ]
            self._times[name] = [c.time for c in series]
            self._closes[name] = [c.close for c in series]
            self._volumes[name] = [c.volume for c in series]
        self._by_alt = {spec["altname"]: name for name, spec in self.specs.items()}

    def _closed_index(self, name: str) -> int:
        """Nombre de bougies clôturées à l'instant `now`."""
        return bisect.bisect_right(self._times[name], self.now - self.interval * 60)

    def _name(self, pair: str) -> str:
        return self._by_alt.get(pair, pair)

    # --- endpoints ---------------------------------------------------------

    def asset_pairs(self) -> dict[str, Any]:
        return {name: {k: v for k, v in spec.items() if k != "name"} for name, spec in self.specs.items()}

    def ticker(self, pairs: Iterable[str] | None = None) -> dict[str, Any]:
        names = [self._name(p) for p in pairs] if pairs else list(self.specs)
        out = {}
        for name in names:
            if name not in self._rows:
                continue
            n = self._closed_index(name)
            if n == 0:
                continue
            alt = self.specs[name]["altname"]
            price = self.price_override.get(alt, self._closes[name][n - 1])
            half = price * self.spread_pct / 200
            volume = sum(self._volumes[name][max(0, n - 24) : n])
            out[name] = {
                "a": [str(price + half), "1", "1"],
                "b": [str(price - half), "1", "1"],
                "c": [str(price), "0"],
                "v": [str(volume), str(volume)],
                "p": [str(price), str(price)],
            }
        return out

    def ohlc(self, pair: str, interval: int = 60, since: int | None = None) -> dict[str, Any]:
        name = self._name(pair)
        n = self._closed_index(name)
        rows = self._rows[name][max(0, n - self.max_rows) : n]
        return {name: rows, "last": rows[-1][0] if rows else 0}

    def candle_at(self, alt: str, time: int) -> list[Any] | None:
        name = self._name(alt)
        times = self._times.get(name, [])
        i = bisect.bisect_left(times, time)
        return self._rows[name][i] if i < len(times) and times[i] == time else None


def default_spec(alt: str, price: float) -> dict[str, Any]:
    """Caractéristiques plausibles pour une paire inconnue (minimum ~5€ comme sur Kraken)."""
    symbol = alt[:-3] if alt.endswith("EUR") else alt
    lot_decimals = 8
    ordermin = 10 ** math.floor(math.log10(max(5.0 / price, 1e-8)))
    pair_decimals = max(1, min(8, 5 - int(math.floor(math.log10(max(price, 1e-8))))))
    return {
        "name": alt,
        "altname": alt,
        "wsname": f"{symbol}/EUR",
        "base": symbol,
        "quote": "ZEUR",
        "pair_decimals": pair_decimals,
        "lot_decimals": lot_decimals,
        "ordermin": f"{ordermin:.8f}",
        "costmin": "0.5",
        "status": "online",
    }


# --- données ---------------------------------------------------------------------


def synthetic_candles(
    n: int = 1500,
    *,
    start_price: float = 100.0,
    interval: int = 60,
    seed: int = 0,
    start_time: int = 1_700_000_000,
) -> list[Candle]:
    """Marché synthétique alternant phases haussières, baissières et sans tendance."""
    rng = random.Random(seed)
    candles = []
    price = start_price
    regimes = [(0.0012, 0.008), (-0.0010, 0.010), (0.0, 0.006)]
    drift, vol = regimes[0]
    t0 = start_time - start_time % (interval * 60)
    for i in range(n):
        if i % 150 == 0:
            drift, vol = rng.choice(regimes)
        open_ = price
        close = max(open_ * math.exp(drift + vol * rng.gauss(0, 1)), 1e-6)
        high = max(open_, close) * (1 + abs(rng.gauss(0, vol / 2)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, vol / 2)))
        volume = 100 * (1 + abs(rng.gauss(0, 1)))
        candles.append(Candle(t0 + i * interval * 60, open_, high, low, close, (high + low + close) / 3, volume, 10))
        price = close
    return candles


def load_kraken_csv(path: Path) -> list[Candle]:
    """Fichier OHLCVT téléchargé depuis Kraken : timestamp,open,high,low,close,volume,trades."""
    candles = []
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip().isdigit():
                continue
            t, o, h, l, c, v = (float(x) for x in row[:6])
            trades = int(float(row[6])) if len(row) > 6 else 0
            candles.append(Candle(int(t), o, h, l, c, (h + l + c) / 3, v, trades))
    return candles


# --- moteur ----------------------------------------------------------------------


@dataclass
class BacktestResult:
    start_equity: float
    end_equity: float
    max_drawdown_pct: float
    trades: list[dict[str, Any]]
    buy_hold_pct: float
    halted: bool
    halt_reason: str
    fees_paid: float
    candles: int
    equity_curve: list[tuple[float, float]] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        return (self.end_equity / self.start_equity - 1) * 100 if self.start_equity else 0.0

    @property
    def wins(self) -> list[dict[str, Any]]:
        return [t for t in self.trades if (t.get("pnl") or 0) > 0]

    @property
    def losses(self) -> list[dict[str, Any]]:
        return [t for t in self.trades if (t.get("pnl") or 0) <= 0]

    @property
    def profit_factor(self) -> float:
        gains = sum(t["pnl"] for t in self.wins)
        losses = -sum(t["pnl"] for t in self.losses)
        return gains / losses if losses > 0 else math.inf if gains > 0 else 0.0

    def report(self) -> str:
        n = len(self.trades)
        win_rate = len(self.wins) / n * 100 if n else 0.0
        lines = [
            "=== Résultat du backtest ===",
            f"Bougies simulées      : {self.candles}",
            f"Capital               : {self.start_equity:.2f}€ -> {self.end_equity:.2f}€ ({self.return_pct:+.2f}%)",
            f"Acheter et conserver  : {self.buy_hold_pct:+.2f}% (moyenne des paires)",
            f"Drawdown maximum      : -{self.max_drawdown_pct:.2f}%",
            f"Trades                : {n} (gagnants {win_rate:.0f}%)",
            f"Profit factor         : {self.profit_factor:.2f}",
            f"Frais payés           : {self.fees_paid:.3f}€",
        ]
        if n:
            avg_win = sum(t["pnl_pct"] for t in self.wins) / len(self.wins) if self.wins else 0.0
            avg_loss = sum(t["pnl_pct"] for t in self.losses) / len(self.losses) if self.losses else 0.0
            lines.append(f"Gain moyen / perte moy.: {avg_win:+.2f}% / {avg_loss:+.2f}%")
        if self.halted:
            lines.append(f"ARRÊT DE SURVIE déclenché : {self.halt_reason}")
        return "\n".join(lines)


def run_backtest(
    candles: dict[str, list[Candle]],
    settings: Settings,
    *,
    pair_specs: dict[str, dict[str, Any]] | None = None,
) -> BacktestResult:
    s = replace(
        settings,
        mode="paper",
        brain="rules",
        universe=list(candles),
        universe_refresh_hours=1e9,
        min_volume_24h=0.0,
    )
    interval_s = s.interval_seconds
    sim = SimulatedKraken(candles, interval=s.candle_interval, pair_specs=pair_specs, max_rows=300)
    timeline = sorted({c.time for series in candles.values() for c in series})
    warmup = s.ema_trend + 20
    if len(timeline) <= warmup + 1:
        raise ValueError(f"historique trop court : il faut plus de {warmup + 1} bougies")

    store = StateStore(None, "paper")
    agent = TradingAgent(s, sim, store, brain=RulesBrain(), clock=lambda: sim.now)
    sim.now = timeline[warmup] + interval_s
    agent.start()
    start_equity = agent.state.last_equity
    curve = [(sim.now, start_equity)]
    first_close = {alt: sim.candle_at(alt, timeline[warmup]) for alt in candles}

    for t in timeline[warmup + 1 :]:
        now = t + interval_s
        # Stop touché pendant la bougie : exécution au stop, ou à l'ouverture en cas de gap.
        for pos in list(agent.state.positions):
            row = sim.candle_at(pos.pair, t)
            if row is not None and row[3] <= pos.stop:
                price = min(pos.stop, row[1])
                agent.close_position(pos, Ticker(pos.pair, price, price, price, 0, 0), "stop-loss (intra-bougie)", now)
        sim.now = now
        agent.step()
        curve.append((now, agent.state.last_equity))
        if agent.state.status != "alive":
            break

    peak, max_dd = curve[0][1], 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (1 - eq / peak) * 100)
    buy_hold = []
    for alt, row in first_close.items():
        last = sim.candle_at(alt, int(sim.now - interval_s))
        if row and last:
            buy_hold.append((last[4] / row[4] - 1) * 100)
    trades = [r for journal, r in store.memory_log if journal == "trades" and r["side"] == "sell"]
    return BacktestResult(
        start_equity=start_equity,
        end_equity=curve[-1][1],
        max_drawdown_pct=max_dd,
        trades=trades,
        buy_hold_pct=sum(buy_hold) / len(buy_hold) if buy_hold else 0.0,
        halted=agent.state.status != "alive",
        halt_reason=agent.state.halt_reason,
        fees_paid=agent.state.stats["fees_paid"],
        candles=len(curve) - 1,
        equity_curve=curve,
    )
