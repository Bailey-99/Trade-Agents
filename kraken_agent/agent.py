"""Boucle principale de l'agent de trading."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from . import market, strategy
from .brain import Brain, DecisionContext, make_brain
from .broker import Broker, BrokerError, LiveBroker, PaperBroker
from .config import Settings
from .kraken_client import KrakenError
from .models import Decision, Fill, Position, Ticker, TradeRecord
from .risk import RiskManager
from .state import StateStore
from .strategy import EntrySignal, Features

log = logging.getLogger(__name__)

MAX_CANDIDATES = 5
# Erreurs Kraken indiquant qu'une paire n'est pas négociable pour ce compte (pays, statut…).
_PAIR_BLOCKING_ERRORS = ("Invalid permissions", "Permission denied", "restricted", "Unknown asset pair")


class TradingAgent:
    def __init__(
        self,
        settings: Settings,
        client: Any,
        store: StateStore,
        *,
        brain: Brain | None = None,
        broker: Broker | None = None,
        risk: RiskManager | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.s = settings
        self.client = client
        self.store = store
        self.state = store.load()
        self.state.mode = settings.mode
        self.brain = brain or make_brain(settings)
        self.risk = risk or RiskManager(settings)
        if broker is None:
            broker = LiveBroker(client, settings) if settings.mode == "live" else PaperBroker(self.state, settings)
        self.broker = broker
        self.clock = clock
        self.pairs: dict[str, Any] = {}
        self.assets: dict[str, Any] = {}
        self.universe: list[str] = []
        self.tickers: dict[str, Ticker] = {}
        self.features: dict[str, Features] = {}
        self._universe_at = float("-inf")
        self._started = False

    # --- démarrage ------------------------------------------------------------

    def start(self) -> None:
        self.pairs = market.load_pairs(self.client.asset_pairs(), self.s.quote)
        self.assets = market.asset_index(self.pairs)
        if self.state.initial_capital is None and self.s.mode == "paper":
            self.state.paper_cash = self.s.paper_start_eur
        self._refresh_universe(self.clock())
        self.broker.refresh()
        self.tickers = self._fetch_tickers()
        equity = self.equity()
        now = self.clock()
        if self.state.initial_capital is None:
            self.state.initial_capital = equity
            self.state.peak_equity = equity
            log.info("capital de départ : %.2f %s", equity, self.s.quote)
        self._track_equity(equity, now)
        self._started = True
        self.save()

    def _refresh_universe(self, now: float) -> None:
        if self.s.universe:
            self.universe = [p for p in self.s.universe if p in self.pairs]
            missing = set(self.s.universe) - set(self.universe)
            if missing:
                log.warning("paires inconnues ignorées : %s", ", ".join(sorted(missing)))
        else:
            self.universe = market.discover_universe(
                self.client,
                self.pairs,
                size=self.s.universe_size,
                min_volume=self.s.min_volume_24h,
                max_spread_pct=self.s.max_spread_pct,
                exclude=self.state.blacklist,
            )
        self._universe_at = now
        log.info("univers surveillé (%d) : %s", len(self.universe), ", ".join(self.universe))

    # --- valorisation ---------------------------------------------------------

    def _watched_pairs(self) -> set[str]:
        watched = set(self.universe) | {p.pair for p in self.state.positions}
        for asset, amount in self.broker.balances().items():
            pair = self.assets.get(asset) or self.assets.get(market.normalize_asset(asset))
            if amount > 0 and pair is not None:
                watched.add(pair.altname)
        return watched

    def _fetch_tickers(self) -> dict[str, Ticker]:
        return market.fetch_tickers(self.client, self.pairs, self._watched_pairs())

    def equity(self) -> float:
        quote = market.quote_codes(self.s.quote)
        total = 0.0
        for asset, amount in self.broker.balances().items():
            if amount <= 0:
                continue
            code = market.normalize_asset(asset)
            if asset in quote or code in quote:
                total += amount
                continue
            pair = self.assets.get(asset) or self.assets.get(code)
            ticker = self.tickers.get(pair.altname) if pair else None
            if ticker is not None:
                total += amount * ticker.bid
        return total

    def _track_equity(self, equity: float, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        if self.state.day != day:
            self.state.day = day
            self.state.day_start_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.last_equity = equity

    def save(self) -> None:
        self.store.save(self.state)

    # --- tour de boucle -------------------------------------------------------

    def step(self) -> None:
        if not self._started:
            self.start()
        if self.state.status != "alive":
            return
        now = self.clock()
        if now - self._universe_at >= self.s.universe_refresh_hours * 3600:
            self._refresh_universe(now)
        self.broker.refresh()
        self.tickers = self._fetch_tickers()

        self._sync_positions(now)
        equity = self.equity()
        self._track_equity(equity, now)

        if self.risk.breached_floor(self.state, equity):
            floor = self.risk.survival_floor(self.state)
            self.halt(f"capital {equity:.2f}€ sous le plancher de survie {floor:.2f}€")
            return

        self._manage_stops(now)

        candle_id = int(now // self.s.interval_seconds)
        if candle_id != self.state.last_analysis_candle:
            self._analyze_and_trade(now)
            self.state.last_analysis_candle = candle_id
        self.state.last_equity = self.equity()
        self.save()

    def _sync_positions(self, now: float) -> None:
        """Mode réel : détecte les stops exécutés par Kraken et repose les stops manquants."""
        for pos in list(self.state.positions):
            pair = self.pairs.get(pos.pair)
            if pair is None:
                continue
            try:
                fill = self.broker.sync(pos, pair)
            except (KrakenError, BrokerError, requests.RequestException) as exc:
                held = self.broker.balances().get(pos.base, 0.0)
                if isinstance(exc, KrakenError) and exc.has("Insufficient funds") and held < pos.volume * 0.5:
                    # L'actif n'est plus sur le compte (vendu à la main ?) : on oublie la position.
                    log.warning("%s : actif absent du compte, position retirée du suivi", pos.pair)
                    self.state.positions.remove(pos)
                else:
                    log.error("synchronisation de %s impossible : %s", pos.pair, exc)
                continue
            if fill is not None and fill.volume > 0:
                self._record_close(pos, fill, "stop-loss exécuté par Kraken", now)

    def _manage_stops(self, now: float) -> None:
        for pos in list(self.state.positions):
            ticker = self.tickers.get(pos.pair)
            pair = self.pairs.get(pos.pair)
            if ticker is None or pair is None:
                continue
            if ticker.bid <= pos.stop:
                self.close_position(pos, ticker, f"stop-loss touché ({ticker.bid:g} <= {pos.stop:g})", now)
                continue
            new_stop = strategy.trailing_stop(pos, ticker.bid, self.s)
            pos.highest = max(pos.highest, ticker.bid)
            if new_stop > pos.stop * 1.002:
                old = pos.stop
                try:
                    self.broker.move_stop(pos, pair, new_stop)
                    log.info("%s : stop remonté %g -> %g", pos.pair, old, pos.stop)
                except (KrakenError, BrokerError, requests.RequestException) as exc:
                    log.error("%s : mise à jour du stop impossible (%s), on sort par sécurité", pos.pair, exc)
                    self.close_position(pos, ticker, "stop impossible à mettre à jour", now)

    # --- analyse et décisions ---------------------------------------------------

    def _analyze_and_trade(self, now: float) -> None:
        held = {p.pair for p in self.state.positions}
        features: dict[str, Features] = {}
        for alt in sorted(set(self.universe) | held):
            pair = self.pairs.get(alt)
            if pair is None:
                continue
            try:
                candles = market.fetch_candles(self.client, pair, self.s.candle_interval, now)
            except (KrakenError, requests.RequestException) as exc:
                log.warning("bougies %s indisponibles : %s", alt, exc)
                continue
            f = strategy.compute_features(alt, candles, self.s)
            if f is not None:
                features[alt] = f
        self.features = features

        for pos in self.state.positions:
            if pos.pair in features:
                pos.atr = features[pos.pair].atr

        candidates: list[EntrySignal] = []
        for alt, f in features.items():
            ticker = self.tickers.get(alt)
            if alt in held or alt in self.state.blacklist or ticker is None:
                continue
            if ticker.spread_pct > self.s.max_spread_pct:
                continue
            signal = strategy.entry_signal(f, self.s)
            if signal is not None:
                candidates.append(signal)
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:MAX_CANDIDATES]

        soft_exits = {
            pos.pair: reasons
            for pos in self.state.positions
            if (reasons := strategy.soft_exit_reasons(pos, features.get(pos.pair), now, self.s))
        }

        ctx = self._context(now, candidates, soft_exits)
        decisions = self.brain.decide(ctx)
        self.store.append(
            "decisions",
            {
                "time": now,
                "brain": self.brain.name,
                "equity": round(ctx.equity, 4),
                "candidates": [c.pair for c in candidates],
                "soft_exits": soft_exits,
                "decisions": [d.__dict__ for d in decisions],
                "market_view": getattr(self.brain, "last_view", ""),
            },
        )
        if decisions:
            log.info(
                "candidats : %s | décisions : %s",
                ", ".join(f"{c.pair}({c.score})" for c in candidates) or "aucun",
                ", ".join(f"{d.action} {d.pair}" for d in decisions) or "rien",
            )
        self._execute(decisions, candidates, now)

    def _context(self, now: float, candidates: list[EntrySignal], soft_exits: dict[str, list[str]]) -> DecisionContext:
        equity = self.equity()
        can_open, why = self.risk.can_open(self.state, equity, now)
        positions = []
        for pos in self.state.positions:
            ticker = self.tickers.get(pos.pair)
            price = ticker.bid if ticker else pos.entry_price
            positions.append(
                {
                    "pair": pos.pair,
                    "entry_price": pos.entry_price,
                    "price": price,
                    "pnl_pct": round(pos.pnl_pct(price), 2),
                    "stop": pos.stop,
                    "stop_distance_pct": round((1 - pos.stop / price) * 100, 2) if price else None,
                    "held_candles": round((now - pos.opened_at) / self.s.interval_seconds, 1),
                    "value_eur": round(pos.volume * price, 4),
                    "soft_exit_signals": soft_exits.get(pos.pair, []),
                }
            )
        return DecisionContext(
            now=now,
            mode=self.s.mode,
            equity=equity,
            cash=self.broker.cash(),
            initial_capital=self.state.initial_capital or equity,
            survival_floor=self.risk.survival_floor(self.state),
            can_open=can_open,
            open_block_reason=why,
            free_slots=max(self.s.max_open_positions - len(self.state.positions), 0),
            candidates=candidates,
            positions=positions,
            soft_exits=soft_exits,
            recent_trades=self.state.recent_trades,
            settings=self.s,
            llm_usage=self.state.llm,
        )

    def _execute(self, decisions: list[Decision], candidates: list[EntrySignal], now: float) -> None:
        signals = {c.pair: c for c in candidates}
        for d in decisions:
            if d.action != "sell":
                continue
            pos = self.state.position(d.pair)
            ticker = self.tickers.get(d.pair)
            if pos is not None and ticker is not None:
                self.close_position(pos, ticker, f"[{d.source}] {d.reasoning}", now)
        for d in decisions:
            if d.action != "buy" or d.pair not in signals:
                continue
            ticker = self.tickers.get(d.pair)
            if ticker is not None:
                self.open_position(signals[d.pair], d, ticker, now)

    # --- ordres -----------------------------------------------------------------

    def open_position(self, signal: EntrySignal, decision: Decision, ticker: Ticker, now: float) -> Position | None:
        pair = self.pairs[signal.pair]
        equity = self.equity()
        ok, why = self.risk.can_open(self.state, equity, now)
        if not ok:
            log.info("achat de %s annulé : %s", signal.pair, why)
            return None
        plan, why = self.risk.size_position(
            state=self.state,
            equity=equity,
            cash=self.broker.cash(),
            price=ticker.ask,
            stop_pct=signal.stop_pct,
            pair=pair,
            size_fraction=decision.size_fraction,
        )
        if plan is None:
            log.info("achat de %s ignoré : %s", signal.pair, why)
            return None
        try:
            fill = self.broker.market_buy(pair, plan.volume, ticker)
        except KrakenError as exc:
            if any(exc.has(e) for e in _PAIR_BLOCKING_ERRORS):
                self.state.blacklist.append(signal.pair)
                log.warning("%s non négociable sur ce compte, paire exclue : %s", signal.pair, exc)
            else:
                log.error("achat de %s échoué : %s", signal.pair, exc)
            return None
        except (BrokerError, requests.RequestException) as exc:
            log.error("achat de %s échoué : %s", signal.pair, exc)
            return None

        stop = fill.price * (1 - signal.stop_pct / 100)
        pos = Position(
            pair=pair.altname,
            base=pair.base,
            volume=fill.volume,
            entry_price=fill.price,
            entry_cost=fill.cost + fill.fee,
            entry_fee=fill.fee,
            stop=stop,
            initial_stop=stop,
            highest=fill.price,
            opened_at=now,
            atr=signal.features.atr,
            entry_txid=fill.txid,
            reason=decision.reasoning,
        )
        self.state.positions.append(pos)
        self.state.stats["fees_paid"] += fill.fee
        self.save()
        self.store.append(
            "trades",
            TradeRecord(
                pair=pos.pair,
                side="buy",
                volume=fill.volume,
                price=fill.price,
                cost=fill.cost,
                fee=fill.fee,
                time=now,
                reason=decision.reasoning,
                extra={"stop": stop, "brain": decision.source, "note": plan.note},
            ).to_dict(),
        )
        log.info(
            "ACHAT %s : %g @ %g (%.2f€ + %.3f€ de frais), stop %g (-%.2f%%) — %s",
            pos.pair, fill.volume, fill.price, fill.cost, fill.fee, stop, signal.stop_pct, decision.reasoning,
        )  # fmt: skip
        try:
            self.broker.protect(pos, pair)
        except (KrakenError, BrokerError, requests.RequestException) as exc:
            log.error("stop-loss impossible à poser (%s) : revente immédiate par sécurité", exc)
            self.close_position(pos, ticker, "stop-loss impossible à poser", now)
            return None
        self.save()
        return pos

    def close_position(self, pos: Position, ticker: Ticker, reason: str, now: float) -> Fill | None:
        pair = self.pairs[pos.pair]
        try:
            fill = self.broker.close(pos, pair, ticker)
        except (KrakenError, BrokerError, requests.RequestException) as exc:
            log.error("vente de %s impossible : %s", pos.pair, exc)
            return None
        self._record_close(pos, fill, reason, now)
        return fill

    def _record_close(self, pos: Position, fill: Fill, reason: str, now: float) -> None:
        proceeds = fill.cost - fill.fee
        pnl = proceeds - pos.entry_cost
        pnl_pct = pnl / pos.entry_cost * 100 if pos.entry_cost else 0.0
        stats = self.state.stats
        stats["trades"] += 1
        stats["wins" if pnl > 0 else "losses"] += 1
        stats["realized_pnl"] += pnl
        stats["fees_paid"] += fill.fee
        self.risk.record_result(self.state, pnl, now)
        if pos in self.state.positions:
            self.state.positions.remove(pos)
        record = TradeRecord(
            pair=pos.pair,
            side="sell",
            volume=fill.volume,
            price=fill.price,
            cost=fill.cost,
            fee=fill.fee,
            time=now,
            reason=reason,
            pnl=round(pnl, 6),
            pnl_pct=round(pnl_pct, 3),
            extra={"entry_price": pos.entry_price, "held_hours": round((now - pos.opened_at) / 3600, 2)},
        )
        self.store.append("trades", record.to_dict())
        self.state.recent_trades.append(
            {
                "pair": pos.pair,
                "pnl_eur": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 2),
                "held_hours": record.extra["held_hours"],
                "exit_reason": reason[:120],
            }
        )
        self.state.recent_trades = self.state.recent_trades[-20:]
        self.save()
        log.info("VENTE %s : %g @ %g, résultat %+.4f€ (%+.2f%%) — %s", pos.pair, fill.volume, fill.price, pnl, pnl_pct, reason)

    # --- arrêt ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """Règle de survie : tout revendre en EUR et arrêter l'agent."""
        log.critical("ARRÊT DE SURVIE : %s. Liquidation des positions.", reason)
        now = self.clock()
        for pos in list(self.state.positions):
            ticker = self.tickers.get(pos.pair)
            if ticker is None:
                ticker = Ticker(pos.pair, pos.entry_price, pos.entry_price, pos.entry_price, 0, 0)
            self.close_position(pos, ticker, f"arrêt : {reason}", now)
        self.state.status = "halted"
        self.state.halt_reason = reason
        self.save()

    def panic(self) -> None:
        if not self._started:
            self.start()
        self.broker.refresh()
        self.tickers = self._fetch_tickers()
        self.halt("arrêt manuel (panic)")

    def revive(self) -> None:
        if not self._started:
            self.start()
        self.broker.refresh()
        self.tickers = self._fetch_tickers()
        equity = self.equity()
        self.state.status = "alive"
        self.state.halt_reason = ""
        self.state.cooldown_until = 0.0
        self.state.consecutive_losses = 0
        self.state.initial_capital = equity
        self.state.peak_equity = equity
        self.save()
        log.info("agent réactivé, nouveau capital de référence : %.2f€", equity)

    # --- affichage --------------------------------------------------------------

    def status_line(self) -> str:
        equity = self.state.last_equity
        initial = self.state.initial_capital or equity or 1
        parts = [
            f"[{self.s.mode}] capital {equity:.2f}€ ({(equity / initial - 1) * 100:+.2f}%)",
            f"liquidités {self.broker.cash():.2f}€",
        ]
        for pos in self.state.positions:
            ticker = self.tickers.get(pos.pair)
            price = ticker.bid if ticker else pos.entry_price
            parts.append(f"{pos.pair} {pos.pnl_pct(price):+.2f}% (stop {pos.stop:g})")
        if not self.state.positions:
            parts.append("aucune position")
        llm = self.state.llm
        if llm.get("total_calls"):
            parts.append(f"IA {llm['total_calls']} appels ~{llm['est_cost_usd']:.3f}$")
        if self.state.status != "alive":
            parts.append(f"ARRÊTÉ : {self.state.halt_reason}")
        return " | ".join(parts)

    def run_forever(self, *, max_steps: int | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        while not self._started:
            try:
                self.start()
            except (KrakenError, requests.RequestException) as exc:
                if max_steps is not None:
                    raise
                log.error("démarrage impossible (%s), nouvel essai dans %ss", exc, self.s.loop_seconds)
                sleep(self.s.loop_seconds)
        steps = 0
        while self.state.status == "alive":
            try:
                self.step()
            except (KrakenError, BrokerError, requests.RequestException) as exc:
                log.error("tour interrompu : %s", exc)
            except Exception:
                # On continue : les stops serveur protègent les positions, et la
                # boucle doit continuer à les surveiller.
                log.exception("erreur inattendue")
            log.info(self.status_line())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if self.state.status == "alive":
                sleep(self.s.loop_seconds)
        if self.state.status != "alive":
            log.critical("agent arrêté : %s (relancer avec la commande `revive` après analyse)", self.state.halt_reason)
