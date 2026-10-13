"""Exécution des ordres : simulation (paper) ou Kraken réel (live)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from .config import Settings
from .kraken_client import KrakenClient, KrakenError
from .market import quote_codes
from .models import Fill, PairInfo, Position, Ticker
from .state import AgentState

log = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    mode: str = ""

    def refresh(self) -> None:
        """Recharge les soldes (une fois par tour de boucle)."""

    @abstractmethod
    def balances(self) -> dict[str, float]:
        """Soldes totaux par code d'actif (pour la valorisation)."""

    @abstractmethod
    def cash(self) -> float:
        """EUR disponibles pour acheter."""

    @abstractmethod
    def market_buy(self, pair: PairInfo, volume: float, ticker: Ticker) -> Fill: ...

    @abstractmethod
    def close(self, pos: Position, pair: PairInfo, ticker: Ticker) -> Fill:
        """Vend toute la position (annule d'abord le stop serveur s'il existe)."""

    def protect(self, pos: Position, pair: PairInfo) -> None:
        """Pose le stop-loss côté serveur (mode réel uniquement)."""

    def move_stop(self, pos: Position, pair: PairInfo, new_stop: float) -> None:
        pos.stop = new_stop

    def sync(self, pos: Position, pair: PairInfo) -> Fill | None:
        """Renvoie l'exécution du stop serveur s'il a été déclenché."""
        return None


class PaperBroker(Broker):
    """Simulation : exécution au meilleur prix + glissement, frais taker Kraken."""

    mode = "paper"

    def __init__(self, state: AgentState, settings: Settings):
        self.state = state
        self.s = settings
        self.quote_code = "Z" + settings.quote.upper()

    def balances(self) -> dict[str, float]:
        return {self.quote_code: self.state.paper_cash, **self.state.paper_holdings}

    def cash(self) -> float:
        return self.state.paper_cash

    def market_buy(self, pair: PairInfo, volume: float, ticker: Ticker) -> Fill:
        fee_rate = self.s.taker_fee_pct / 100
        price = ticker.ask * (1 + self.s.slippage_pct / 100)
        max_volume = pair.round_volume(self.state.paper_cash / (price * (1 + fee_rate)))
        volume = min(pair.round_volume(volume), max_volume)
        if volume <= 0 or volume < pair.ordermin:
            raise BrokerError("fonds insuffisants (simulation)")
        cost = volume * price
        fee = cost * fee_rate
        self.state.paper_cash -= cost + fee
        self.state.paper_holdings[pair.base] = self.state.paper_holdings.get(pair.base, 0.0) + volume
        return Fill(volume=volume, price=price, cost=cost, fee=fee, txid=f"PAPER-{time.time_ns()}")

    def market_sell(self, pair: PairInfo, volume: float, ticker: Ticker, price: float | None = None) -> Fill:
        held = self.state.paper_holdings.get(pair.base, 0.0)
        volume = min(volume, held)
        if volume <= 0:
            raise BrokerError(f"aucun {pair.symbol} à vendre (simulation)")
        price = price if price is not None else ticker.bid * (1 - self.s.slippage_pct / 100)
        cost = volume * price
        fee = cost * self.s.taker_fee_pct / 100
        self.state.paper_cash += cost - fee
        remaining = held - volume
        if remaining <= 1e-12:
            self.state.paper_holdings.pop(pair.base, None)
        else:
            self.state.paper_holdings[pair.base] = remaining
        return Fill(volume=volume, price=price, cost=cost, fee=fee, txid=f"PAPER-{time.time_ns()}")

    def close(self, pos: Position, pair: PairInfo, ticker: Ticker) -> Fill:
        return self.market_sell(pair, pos.volume, ticker)


class LiveBroker(Broker):
    """Ordres réels sur Kraken. Chaque position est protégée par un stop-loss côté serveur,
    qui reste actif même si l'agent s'arrête ou perd la connexion."""

    mode = "live"

    def __init__(
        self,
        client: KrakenClient,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        fill_timeout: float = 30.0,
    ):
        self.client = client
        self.s = settings
        self._sleep = sleep
        self.fill_timeout = fill_timeout
        self._balances: dict[str, dict[str, float]] = {}
        self._quote_codes = quote_codes(settings.quote)

    def refresh(self) -> None:
        raw = self.client.balance_ex()
        self._balances = {
            asset: {
                "balance": float(info.get("balance", 0) or 0),
                "hold": float(info.get("hold_trade", 0) or 0),
            }
            for asset, info in raw.items()
        }

    def balances(self) -> dict[str, float]:
        return {asset: b["balance"] for asset, b in self._balances.items()}

    def available(self, asset: str) -> float:
        b = self._balances.get(asset)
        return max(b["balance"] - b["hold"], 0.0) if b else 0.0

    def cash(self) -> float:
        return sum(self.available(code) for code in self._quote_codes)

    # --- ordres ------------------------------------------------------------

    def _order_fill(self, txid: str) -> tuple[str, Fill]:
        info = self.client.query_orders([txid]).get(txid)
        if info is None:
            raise BrokerError(f"ordre {txid} introuvable")
        fill = Fill(
            volume=float(info.get("vol_exec", 0) or 0),
            price=float(info.get("price", 0) or 0),
            cost=float(info.get("cost", 0) or 0),
            fee=float(info.get("fee", 0) or 0),
            txid=txid,
        )
        return info.get("status", ""), fill

    def _wait_fill(self, txid: str) -> Fill:
        deadline = time.monotonic() + self.fill_timeout
        while True:
            status, fill = self._order_fill(txid)
            if status == "closed" or (status in {"canceled", "expired"} and fill.volume > 0):
                return fill
            if status in {"canceled", "expired"}:
                raise BrokerError(f"ordre {txid} {status} sans exécution")
            if time.monotonic() > deadline:
                raise BrokerError(f"ordre {txid} toujours '{status}' après {self.fill_timeout:.0f}s")
            self._sleep(1.0)

    def _market(self, side: str, pair: PairInfo, volume: float) -> Fill:
        result: dict[str, Any] = self.client.add_order(
            pair=pair.altname,
            type=side,
            ordertype="market",
            volume=pair.format_volume(volume),
            oflags="fciq",  # frais prélevés en EUR : on garde la totalité de l'actif acheté
        )
        txid = result["txid"][0]
        log.info("ordre %s envoyé : %s (%s)", side, result.get("descr", {}).get("order", ""), txid)
        return self._wait_fill(txid)

    def market_buy(self, pair: PairInfo, volume: float, ticker: Ticker) -> Fill:
        fill = self._market("buy", pair, volume)
        self.refresh()
        return fill

    def protect(self, pos: Position, pair: PairInfo) -> None:
        volume = min(pos.volume, self.available(pos.base) or pos.volume)
        result = self.client.add_order(
            pair=pair.altname,
            type="sell",
            ordertype="stop-loss",
            price=pair.format_price(pos.stop),
            volume=pair.format_volume(volume),
            oflags="fciq",
        )
        pos.stop_txid = result["txid"][0]
        log.info("stop-loss serveur posé à %s pour %s (%s)", pair.format_price(pos.stop), pos.pair, pos.stop_txid)

    def _cancel(self, txid: str) -> None:
        try:
            self.client.cancel_order(txid)
        except KrakenError as exc:
            if not exc.has("Unknown order"):
                raise

    def move_stop(self, pos: Position, pair: PairInfo, new_stop: float) -> None:
        # Annuler puis reposer : Kraken bloque les fonds d'un ordre ouvert,
        # on ne peut pas avoir deux stops sur le même volume.
        if pos.stop_txid:
            self._cancel(pos.stop_txid)
            status, fill = self._order_fill(pos.stop_txid)
            if status == "closed":
                raise BrokerError("le stop a été exécuté pendant sa mise à jour")
            pos.stop_txid = None
        old = pos.stop
        pos.stop = new_stop
        self.refresh()
        try:
            self.protect(pos, pair)
        except KrakenError as exc:
            pos.stop = old
            raise BrokerError(f"impossible de reposer le stop : {exc}") from exc

    def sync(self, pos: Position, pair: PairInfo) -> Fill | None:
        if not pos.stop_txid:
            self.protect(pos, pair)
            return None
        status, fill = self._order_fill(pos.stop_txid)
        if status == "closed":
            return fill
        if status in {"canceled", "expired"}:
            if fill.volume > 0:
                return fill
            log.warning("stop %s %s : on le repose", pos.stop_txid, status)
            pos.stop_txid = None
            self.refresh()
            self.protect(pos, pair)
        return None

    def close(self, pos: Position, pair: PairInfo, ticker: Ticker) -> Fill:
        partial: Fill | None = None
        if pos.stop_txid:
            self._cancel(pos.stop_txid)
            status, stop_fill = self._order_fill(pos.stop_txid)
            if status == "closed" or stop_fill.volume >= pos.volume * 0.99:
                return stop_fill  # le stop serveur a déjà vendu
            partial = stop_fill if stop_fill.volume > 0 else None
            pos.stop_txid = None
        self.refresh()
        volume = min(pos.volume - (partial.volume if partial else 0.0), self.available(pos.base))
        if volume < pair.ordermin:
            if partial:
                return partial
            raise BrokerError(f"solde {pair.symbol} insuffisant pour vendre ({volume})")
        fill = self._market("sell", pair, volume)
        self.refresh()
        if partial:
            total = fill.volume + partial.volume
            fill = Fill(
                volume=total,
                price=(fill.cost + partial.cost) / total,
                cost=fill.cost + partial.cost,
                fee=fill.fee + partial.fee,
                txid=fill.txid,
            )
        return fill
