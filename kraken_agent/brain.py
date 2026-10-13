"""Les « cerveaux » de l'agent : règles déterministes ou Claude (IA).

Quel que soit le cerveau, ses décisions passent ensuite par le gestionnaire de
risque, qui peut les refuser. Claude ne peut acheter que parmi les candidats du
scanner et ne peut ni supprimer ni élargir un stop-loss.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .models import Decision
from .strategy import EntrySignal

log = logging.getLogger(__name__)

# Prix publics en $ par million de tokens (entrée, sortie), pour estimer le coût de l'IA.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass
class DecisionContext:
    now: float
    mode: str
    equity: float
    cash: float
    initial_capital: float
    survival_floor: float
    can_open: bool
    open_block_reason: str
    free_slots: int
    candidates: list[EntrySignal]
    positions: list[dict[str, Any]]
    soft_exits: dict[str, list[str]]
    recent_trades: list[dict[str, Any]]
    settings: Settings
    llm_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def entries_possible(self) -> bool:
        return self.can_open and self.free_slots > 0 and bool(self.candidates)


class Brain(ABC):
    name = "brain"

    @abstractmethod
    def decide(self, ctx: DecisionContext) -> list[Decision]: ...


class RulesBrain(Brain):
    """Applique les signaux du scanner tels quels."""

    name = "rules"

    def __init__(self, allow_entries: bool = True):
        self.allow_entries = allow_entries

    def decide(self, ctx: DecisionContext) -> list[Decision]:
        decisions = [
            Decision("sell", pair, reasoning="; ".join(reasons), source=self.name)
            for pair, reasons in ctx.soft_exits.items()
        ]
        if self.allow_entries and ctx.entries_possible:
            for signal in ctx.candidates[: ctx.free_slots]:
                decisions.append(
                    Decision(
                        "buy",
                        signal.pair,
                        confidence=0.5,
                        reasoning=", ".join(signal.reasons),
                        source=self.name,
                    )
                )
        return decisions


# --- Claude --------------------------------------------------------------------

SYSTEM_PROMPT = """You are the decision module of an autonomous trading agent running a small Kraken spot account denominated in EUR.

Objective: grow the account over time. The account is shut down for good if equity falls to the survival floor, so protecting capital comes first and chasing gains second. Holding cash is a legitimate decision and often the right one.

How the system around you works:
- Every closed candle, a rule-based scanner (trend + breakout) proposes buy candidates with indicators and a proposed stop distance. You can only buy pairs from that list, at most as many as the free position slots, or none.
- A risk manager sizes every buy so that hitting the stop costs about risk_per_trade_pct of equity. size_fraction (0.25 to 1.0) only scales that size down when you are less convinced.
- Every position gets a stop-loss on the exchange and a trailing stop that only moves up. You cannot remove or widen them.
- Open positions are shown with their soft exit signals (broken trend, stagnation). Decide whether to sell them or keep them. You can also sell a position that has no exit signal.
- Every trade pays the taker fee on both sides plus slippage (round_trip_cost_pct). A trade is only worth taking if the expected move clearly exceeds that cost.

Indicator conventions: returns are in percent; ret_short_pct covers the last 24 candles and ret_long_pct the last 168; recent_closes_pct are the last 24 closes relative to the current price.

Return your decision as structured output; orders may be empty. Write market_view and each reasoning in French, in one or two sentences."""


class LLMOrder(BaseModel):
    action: Literal["buy", "sell"]
    pair: str = Field(description="Paire exactement comme dans les données, ex. XBTEUR")
    size_fraction: float = Field(description="0.25 à 1.0 : fraction de la taille autorisée par le risk manager")
    confidence: float = Field(description="0.0 à 1.0")
    reasoning: str


class LLMVerdict(BaseModel):
    market_view: str
    orders: list[LLMOrder]


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


class ClaudeBrain(Brain):
    """Claude choisit parmi les candidats du scanner et arbitre les sorties.

    Appelé seulement quand il y a quelque chose à décider, avec un plafond
    d'appels par jour. Sans réponse exploitable de l'IA (erreur, budget épuisé,
    refus), l'agent n'ouvre aucune nouvelle position mais applique les sorties.
    """

    name = "claude"

    def __init__(self, settings: Settings, client: Any = None):
        self.s = settings
        self._client = client
        self.fallback = RulesBrain(allow_entries=False)
        self.last_view = ""

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def decide(self, ctx: DecisionContext) -> list[Decision]:
        if not (ctx.soft_exits or ctx.entries_possible):
            return []
        usage = ctx.llm_usage
        today = _day(ctx.now)
        if usage.get("day") != today:
            usage["day"] = today
            usage["calls_today"] = 0
        if usage.get("calls_today", 0) >= self.s.max_llm_calls_per_day:
            log.warning("budget d'appels IA du jour épuisé : sorties selon les règles, pas de nouvel achat")
            return self.fallback.decide(ctx)
        try:
            verdict = self._ask(ctx)
        except Exception as exc:  # erreur réseau/API : on reste prudent
            log.error("appel à Claude impossible (%s) : sorties selon les règles, pas de nouvel achat", exc)
            return self.fallback.decide(ctx)
        if verdict is None:
            return self.fallback.decide(ctx)
        self.last_view = verdict.market_view
        log.info("analyse IA : %s", verdict.market_view)
        return self._validate(verdict, ctx)

    def _ask(self, ctx: DecisionContext) -> LLMVerdict | None:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.s.claude_model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": self.render(ctx)}],
            "output_config": {"effort": self.s.claude_effort},
            "output_format": LLMVerdict,
        }
        if not self.s.claude_model.startswith("claude-haiku"):
            # En cas de refus du modèle principal, l'API relance la requête sur un modèle de repli.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            response = self.client.beta.messages.parse(**kwargs)
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"API Claude {exc.status_code}: {exc.message}") from exc
        self._record_usage(response, ctx.llm_usage)
        if response.stop_reason == "refusal":
            log.warning("Claude a refusé de répondre (%s)", getattr(response, "stop_details", None))
            return None
        if response.stop_reason == "max_tokens":
            log.warning("réponse IA tronquée (max_tokens)")
            return None
        return response.parsed_output

    def _record_usage(self, response: Any, usage: dict[str, Any]) -> None:
        tokens_in = getattr(response.usage, "input_tokens", 0) or 0
        tokens_out = getattr(response.usage, "output_tokens", 0) or 0
        price_in, price_out = PRICING_PER_MTOK.get(self.s.claude_model, (0.0, 0.0))
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        usage["calls_today"] = usage.get("calls_today", 0) + 1
        usage["total_calls"] = usage.get("total_calls", 0) + 1
        usage["input_tokens"] = usage.get("input_tokens", 0) + tokens_in
        usage["output_tokens"] = usage.get("output_tokens", 0) + tokens_out
        usage["est_cost_usd"] = round(usage.get("est_cost_usd", 0.0) + cost, 6)
        log.info("appel IA : %d tokens entrée, %d sortie, ~%.4f $", tokens_in, tokens_out, cost)

    def render(self, ctx: DecisionContext) -> str:
        s = ctx.settings
        payload = {
            "time_utc": datetime.fromtimestamp(ctx.now, timezone.utc).isoformat(timespec="minutes"),
            "mode": ctx.mode,
            "account": {
                "equity_eur": round(ctx.equity, 4),
                "cash_eur": round(ctx.cash, 4),
                "initial_capital_eur": round(ctx.initial_capital, 4),
                "survival_floor_eur": round(ctx.survival_floor, 4),
                "pnl_since_start_pct": round((ctx.equity / ctx.initial_capital - 1) * 100, 2)
                if ctx.initial_capital
                else 0.0,
            },
            "rules": {
                "candle_interval_min": s.candle_interval,
                "taker_fee_pct": s.taker_fee_pct,
                "round_trip_cost_pct": round(s.round_trip_cost_pct, 3),
                "risk_per_trade_pct": s.risk_per_trade_pct,
                "max_open_positions": s.max_open_positions,
                "free_slots": ctx.free_slots,
                "new_positions_allowed": ctx.can_open and ctx.free_slots > 0,
                "blocked_reason": ctx.open_block_reason or None,
            },
            "open_positions": ctx.positions,
            "candidates": [
                {
                    "pair": c.pair,
                    "rule_score": c.score,
                    "proposed_stop_pct": c.stop_pct,
                    "target_pct": c.target_pct,
                    "signals": c.reasons,
                    "indicators": c.features.summary(),
                }
                for c in ctx.candidates
            ],
            "recent_trades": ctx.recent_trades[-10:],
        }
        return "Current situation:\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    def _validate(self, verdict: LLMVerdict, ctx: DecisionContext) -> list[Decision]:
        candidates = {c.pair for c in ctx.candidates}
        held = {p["pair"] for p in ctx.positions}
        decisions: list[Decision] = []
        buys = 0
        for order in verdict.orders:
            pair = order.pair.upper().replace("/", "").strip()
            decision = Decision(
                order.action,
                pair,
                size_fraction=min(max(order.size_fraction, 0.25), 1.0),
                confidence=min(max(order.confidence, 0.0), 1.0),
                reasoning=order.reasoning,
                source=self.name,
            )
            if order.action == "sell":
                if pair in held and all(d.pair != pair for d in decisions):
                    decisions.append(decision)
                else:
                    log.warning("IA : vente refusée, aucune position sur %s", pair)
            elif not ctx.can_open or buys >= ctx.free_slots:
                log.warning("IA : achat de %s refusé (%s)", pair, ctx.open_block_reason or "plus de place")
            elif pair not in candidates:
                log.warning("IA : achat de %s refusé (pas dans les candidats du scanner)", pair)
            else:
                decisions.append(decision)
                buys += 1
        return decisions


def make_brain(settings: Settings) -> Brain:
    import os

    choice = settings.brain
    if choice == "auto":
        choice = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "rules"
    if choice == "claude":
        return ClaudeBrain(settings)
    return RulesBrain()
