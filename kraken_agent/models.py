"""Structures de données partagées par tous les modules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PairInfo:
    """Caractéristiques d'une paire Kraken (issues de /0/public/AssetPairs)."""

    name: str  # clé Kraken, ex. "XXBTZEUR"
    altname: str  # ex. "XBTEUR"
    wsname: str  # ex. "XBT/EUR"
    base: str  # code d'actif de base, ex. "XXBT"
    quote: str  # code d'actif de cotation, ex. "ZEUR"
    pair_decimals: int  # précision du prix
    lot_decimals: int  # précision du volume
    ordermin: float  # volume minimum d'un ordre (en actif de base)
    costmin: float  # coût minimum d'un ordre (en devise de cotation)
    status: str = "online"

    @property
    def symbol(self) -> str:
        """Symbole lisible de l'actif de base, ex. "XBT"."""
        return self.wsname.split("/")[0] if "/" in self.wsname else self.base

    def round_volume(self, volume: float) -> float:
        """Arrondit un volume vers le bas à la précision autorisée."""
        factor = 10**self.lot_decimals
        return math.floor(volume * factor + 1e-9) / factor

    def format_volume(self, volume: float) -> str:
        return f"{self.round_volume(volume):.{self.lot_decimals}f}"

    def format_price(self, price: float) -> str:
        return f"{round(price, self.pair_decimals):.{self.pair_decimals}f}"

    def min_notional(self, price: float) -> float:
        """Montant minimum (en EUR) d'un ordre sur cette paire au prix donné."""
        return max(self.costmin, self.ordermin * price)


@dataclass(frozen=True)
class Candle:
    time: int  # horodatage d'ouverture de la bougie (secondes)
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    count: int = 0


@dataclass(frozen=True)
class Ticker:
    pair: str  # altname
    ask: float
    bid: float
    last: float
    volume_24h: float  # en actif de base
    vwap_24h: float

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid * 100 if self.mid > 0 else math.inf

    @property
    def volume_24h_quote(self) -> float:
        return self.volume_24h * self.vwap_24h


@dataclass
class Fill:
    """Résultat d'une exécution (réelle ou simulée)."""

    volume: float
    price: float  # prix moyen d'exécution
    cost: float  # volume * prix, hors frais (EUR)
    fee: float  # frais payés (EUR)
    txid: str | None = None


@dataclass
class Position:
    pair: str  # altname, ex. "XBTEUR"
    base: str  # code d'actif détenu, ex. "XXBT"
    volume: float
    entry_price: float
    entry_cost: float  # EUR dépensés, frais inclus
    entry_fee: float
    stop: float  # stop-loss courant
    initial_stop: float
    highest: float  # plus haut prix observé depuis l'entrée
    opened_at: float
    atr: float = 0.0  # dernier ATR connu (pour le stop suiveur)
    stop_txid: str | None = None  # ordre stop-loss côté Kraken (mode réel)
    entry_txid: str | None = None
    reason: str = ""

    def pnl_pct(self, price: float) -> float:
        return (price / self.entry_price - 1) * 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Decision:
    """Décision émise par un « cerveau » (règles ou IA)."""

    action: str  # "buy" | "sell" | "hold"
    pair: str = ""
    size_fraction: float = 1.0
    confidence: float = 0.5
    reasoning: str = ""
    source: str = ""


@dataclass
class TradeRecord:
    pair: str
    side: str
    volume: float
    price: float
    cost: float
    fee: float
    time: float
    reason: str = ""
    pnl: float | None = None
    pnl_pct: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
