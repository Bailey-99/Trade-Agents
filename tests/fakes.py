"""Faux Kraken (public simulé + endpoints privés) pour tester le mode réel sans argent."""

from __future__ import annotations

from typing import Any

from kraken_agent.backtest import SimulatedKraken
from kraken_agent.kraken_client import KrakenError


class FakeLiveKraken(SimulatedKraken):
    def __init__(self, candles, *, eur: float = 10.0, fee_pct: float = 0.4, **kwargs):
        super().__init__(candles, **kwargs)
        self.fee = fee_pct / 100
        self.wallet: dict[str, float] = {"ZEUR": eur}
        self.orders: dict[str, dict[str, Any]] = {}
        self.added: list[dict[str, Any]] = []
        self.canceled: list[str] = []
        self._n = 0

    def _base(self, pair: str) -> str:
        return self.specs[self._name(pair)]["base"]

    def _price(self, pair: str, side: str) -> float:
        t = self.ticker([pair])[self._name(pair)]
        return float(t["a"][0] if side == "buy" else t["b"][0])

    def _execute(self, order: dict[str, Any], price: float) -> None:
        vol = float(order["vol"])
        base = self._base(order["pair"])
        cost = vol * price
        fee = cost * self.fee
        if order["type"] == "buy":
            if self.wallet["ZEUR"] < cost + fee:
                raise KrakenError(["EOrder:Insufficient funds"], "AddOrder")
            self.wallet["ZEUR"] -= cost + fee
            self.wallet[base] = self.wallet.get(base, 0.0) + vol
        else:
            self.wallet[base] = self.wallet.get(base, 0.0) - vol
            self.wallet["ZEUR"] += cost - fee
        order.update(status="closed", vol_exec=str(vol), price=str(price), cost=str(cost), fee=str(fee))

    def _hold(self, asset: str) -> float:
        return sum(
            float(o["vol"])
            for o in self.orders.values()
            if o["status"] == "open" and o["type"] == "sell" and self._base(o["pair"]) == asset
        )

    # --- endpoints privés ----------------------------------------------------

    def balance_ex(self) -> dict[str, Any]:
        return {a: {"balance": str(v), "hold_trade": str(self._hold(a))} for a, v in self.wallet.items()}

    def add_order(self, **p: Any) -> dict[str, Any]:
        self.added.append(p)
        self._n += 1
        txid = f"O{self._n}"
        order = {"pair": p["pair"], "type": p["type"], "ordertype": p["ordertype"], "vol": p["volume"],
                 "status": "open", "vol_exec": "0", "price": "0", "cost": "0", "fee": "0",
                 "stopprice": p.get("price")}  # fmt: skip
        if p["type"] == "sell":
            available = self.wallet.get(self._base(p["pair"]), 0.0) - self._hold(self._base(p["pair"]))
            if float(p["volume"]) > available + 1e-12:
                raise KrakenError(["EOrder:Insufficient funds"], "AddOrder")
        if p["ordertype"] == "market":
            self._execute(order, self._price(p["pair"], p["type"]))
        self.orders[txid] = order
        return {"txid": [txid], "descr": {"order": f"{p['type']} {p['volume']} {p['pair']} @ {p['ordertype']}"}}

    def query_orders(self, txids) -> dict[str, Any]:
        return {t: dict(self.orders[t]) for t in txids if t in self.orders}

    def cancel_order(self, txid: str) -> dict[str, Any]:
        order = self.orders.get(txid)
        if order is None or order["status"] != "open":
            raise KrakenError(["EOrder:Unknown order"], "CancelOrder")
        order["status"] = "canceled"
        self.canceled.append(txid)
        return {"count": 1}

    def trigger_stops(self) -> None:
        """Exécute les stops dont le prix de déclenchement est atteint (comme le moteur Kraken)."""
        for order in self.orders.values():
            if order["status"] == "open" and order["ordertype"] == "stop-loss":
                bid = self._price(order["pair"], "sell")
                if bid <= float(order["stopprice"]):
                    self._execute(order, bid)
