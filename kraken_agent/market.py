"""Lecture et normalisation des données de marché Kraken."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .models import Candle, PairInfo, Ticker

log = logging.getLogger(__name__)

# Stablecoins et devises : pas de tendance exploitable, on les exclut.
EXCLUDED_SYMBOLS = {
    "USDT", "USDC", "DAI", "EURT", "EURQ", "EURR", "EURC", "EUROP", "PYUSD", "USDG",
    "USDQ", "USDR", "RLUSD", "TUSD", "USDS", "FDUSD", "USDE", "USD1", "UST", "USTC",
    "USD", "GBP", "CHF", "AUD", "CAD", "JPY", "EUR", "ZUSD", "ZGBP", "ZEUR",
}  # fmt: skip

QUOTE_CODES = {"EUR": {"EUR", "ZEUR"}, "USD": {"USD", "ZUSD"}}


def quote_codes(quote: str) -> set[str]:
    return QUOTE_CODES.get(quote.upper(), {quote.upper(), "Z" + quote.upper(), "X" + quote.upper()})


def load_pairs(raw: dict[str, Any], quote: str = "EUR") -> dict[str, PairInfo]:
    """Construit les PairInfo (indexées par altname) à partir de /AssetPairs."""
    codes = quote_codes(quote)
    pairs: dict[str, PairInfo] = {}
    for name, info in raw.items():
        if info.get("quote") not in codes or name.endswith(".d"):
            continue
        altname = info.get("altname", name)
        pairs[altname] = PairInfo(
            name=name,
            altname=altname,
            wsname=info.get("wsname", altname),
            base=info.get("base", ""),
            quote=info.get("quote", ""),
            pair_decimals=int(info.get("pair_decimals", 5)),
            lot_decimals=int(info.get("lot_decimals", 8)),
            ordermin=float(info.get("ordermin", 0) or 0),
            costmin=float(info.get("costmin", 0) or 0),
            status=info.get("status", "online"),
        )
    return pairs


def asset_index(pairs: dict[str, PairInfo]) -> dict[str, PairInfo]:
    """Associe un code d'actif (« XXBT », « XBT »…) à sa paire contre la devise de cotation."""
    index: dict[str, PairInfo] = {}
    for p in pairs.values():
        index.setdefault(p.base, p)
        index.setdefault(p.symbol, p)
    return index


def normalize_asset(code: str) -> str:
    """« XBT.F » -> « XBT », « EUR.HOLD » -> « EUR » (soldes Earn / en attente)."""
    return code.split(".")[0]


def parse_ticker(altname: str, raw: dict[str, Any]) -> Ticker:
    return Ticker(
        pair=altname,
        ask=float(raw["a"][0]),
        bid=float(raw["b"][0]),
        last=float(raw["c"][0]),
        volume_24h=float(raw["v"][1]),
        vwap_24h=float(raw["p"][1]),
    )


def parse_tickers(raw: dict[str, Any], pairs: dict[str, PairInfo]) -> dict[str, Ticker]:
    by_name = {p.name: p for p in pairs.values()}
    out: dict[str, Ticker] = {}
    for key, data in raw.items():
        pair = by_name.get(key) or pairs.get(key)
        if pair is None:
            continue
        try:
            out[pair.altname] = parse_ticker(pair.altname, data)
        except (KeyError, IndexError, ValueError, TypeError):
            log.debug("ticker illisible pour %s", key)
    return out


def fetch_tickers(client: Any, pairs: dict[str, PairInfo], altnames: Iterable[str]) -> dict[str, Ticker]:
    names = sorted({a for a in altnames if a in pairs})
    if not names:
        return {}
    return parse_tickers(client.ticker(names), pairs)


def parse_candles(raw: dict[str, Any]) -> list[Candle]:
    rows = next((v for k, v in raw.items() if k != "last"), [])
    return [
        Candle(
            time=int(r[0]),
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            vwap=float(r[5]),
            volume=float(r[6]),
            count=int(r[7]) if len(r) > 7 else 0,
        )
        for r in rows
    ]


def fetch_candles(client: Any, pair: PairInfo, interval: int, now: float) -> list[Candle]:
    """Bougies clôturées uniquement (Kraken renvoie aussi la bougie en cours)."""
    candles = parse_candles(client.ohlc(pair.altname, interval))
    return [c for c in candles if c.time + interval * 60 <= now]


def discover_universe(
    client: Any,
    pairs: dict[str, PairInfo],
    *,
    size: int,
    min_volume: float,
    max_spread_pct: float,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Sélectionne les paires les plus liquides (volume 24h en EUR, spread serré)."""
    excluded = set(exclude)
    eligible = {
        alt: p
        for alt, p in pairs.items()
        if p.status == "online" and p.symbol not in EXCLUDED_SYMBOLS and alt not in excluded
    }
    tickers = parse_tickers(client.ticker(), eligible)
    ranked = sorted(
        (
            t
            for t in tickers.values()
            if t.volume_24h_quote >= min_volume and t.spread_pct <= max_spread_pct and t.bid > 0
        ),
        key=lambda t: t.volume_24h_quote,
        reverse=True,
    )
    return [t.pair for t in ranked[:size]]
