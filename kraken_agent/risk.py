"""Règles de survie et dimensionnement des positions.

Ces règles s'appliquent quoi que décide le cerveau (règles ou IA) :
- spot uniquement, aucun effet de levier ni vente à découvert ;
- chaque achat est dimensionné pour qu'un stop-loss ne coûte qu'un petit % du capital ;
- plancher de survie : si le capital passe sous X % du capital initial, tout est
  vendu et l'agent s'arrête définitivement (réactivation manuelle uniquement) ;
- pause des achats après une grosse perte journalière ou une série de pertes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
from .models import PairInfo
from .state import AgentState


@dataclass
class SizePlan:
    volume: float
    notional: float  # EUR, hors frais
    max_loss: float  # perte estimée si le stop est touché (frais compris)
    note: str = ""


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings

    # --- survie ------------------------------------------------------------

    def survival_floor(self, state: AgentState) -> float:
        if not state.initial_capital:
            return 0.0
        return state.initial_capital * self.s.survival_floor_pct / 100

    def breached_floor(self, state: AgentState, equity: float) -> bool:
        return bool(state.initial_capital) and equity <= self.survival_floor(state)

    def can_open(self, state: AgentState, equity: float, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now
        if state.status != "alive":
            return False, f"agent arrêté ({state.halt_reason})"
        if len(state.positions) >= self.s.max_open_positions:
            return False, "nombre maximum de positions atteint"
        if state.cooldown_until > now:
            hours = (state.cooldown_until - now) / 3600
            return False, f"pause après {self.s.max_consecutive_losses} pertes d'affilée ({hours:.1f} h restantes)"
        if state.day_start_equity > 0:
            day_loss = (1 - equity / state.day_start_equity) * 100
            if day_loss >= self.s.daily_loss_limit_pct:
                return False, f"limite de perte journalière atteinte (-{day_loss:.1f}%)"
        return True, ""

    def record_result(self, state: AgentState, pnl: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if pnl < 0:
            state.consecutive_losses += 1
            if state.consecutive_losses >= self.s.max_consecutive_losses:
                state.cooldown_until = now + self.s.cooldown_hours * 3600
                state.consecutive_losses = 0
        else:
            state.consecutive_losses = 0

    # --- dimensionnement ---------------------------------------------------

    def loss_per_unit_pct(self, stop_pct: float) -> float:
        """Perte réelle (en %) si le stop est touché : distance + frais + glissement."""
        return stop_pct + self.s.round_trip_cost_pct

    def size_position(
        self,
        *,
        state: AgentState,
        equity: float,
        cash: float,
        price: float,
        stop_pct: float,
        pair: PairInfo,
        size_fraction: float = 1.0,
    ) -> tuple[SizePlan | None, str]:
        if price <= 0 or equity <= 0:
            return None, "prix ou capital invalide"
        fee = self.s.taker_fee_pct / 100
        loss_pct = self.loss_per_unit_pct(stop_pct) / 100
        risk_budget = equity * self.s.risk_per_trade_pct / 100
        fraction = min(max(size_fraction, 0.25), 1.0)

        cap = min(equity * self.s.max_position_pct / 100, cash / (1 + fee) * 0.995)
        # Un stop touché ne doit jamais faire passer le capital sous le plancher de survie.
        headroom = equity - self.survival_floor(state)
        if headroom <= 0:
            return None, "capital déjà au plancher de survie"
        cap = min(cap, headroom / loss_pct * 0.98)

        notional = min(risk_budget / loss_pct * fraction, cap)
        note = ""
        minimum = pair.min_notional(price) * 1.01
        if notional < minimum:
            # Avec un petit compte, le minimum Kraken peut dépasser la taille idéale :
            # on l'accepte seulement si le risque reste <= 2x le budget prévu.
            if minimum <= cap and minimum * loss_pct <= 2 * risk_budget:
                notional = minimum
                note = "taille relevée au minimum Kraken"
            else:
                return None, (
                    f"minimum Kraken {minimum:.2f}€ trop grand pour le capital/risque "
                    f"(max {cap:.2f}€, budget de risque {risk_budget:.2f}€)"
                )

        volume = pair.round_volume(notional / price)
        if volume < pair.ordermin:
            volume = pair.ordermin
        notional = volume * price
        if notional > cap * 1.0001 or notional < pair.costmin:
            return None, f"taille {notional:.2f}€ hors limites (max {cap:.2f}€, min {pair.costmin:.2f}€)"
        if volume <= 0:
            return None, "volume nul"
        return SizePlan(volume=volume, notional=notional, max_loss=notional * loss_pct, note=note), ""
