"""Configuration de l'agent, lue depuis les variables d'environnement (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _get(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() != "" else None


@dataclass
class Settings:
    # --- Kraken ---
    kraken_api_key: str = ""
    kraken_api_secret: str = ""
    mode: str = "paper"  # "paper" (simulation) ou "live" (argent réel)
    quote: str = "EUR"

    # --- Univers de trading ---
    universe: list[str] = field(default_factory=list)  # vide = sélection automatique
    universe_size: int = 15
    min_volume_24h: float = 1_000_000.0  # volume 24h minimum en EUR
    max_spread_pct: float = 0.30
    universe_refresh_hours: float = 6.0

    # --- Boucle ---
    candle_interval: int = 240  # minutes (60, 240, 1440...) ; 4h = moins de trades, donc moins de frais
    loop_seconds: int = 60

    # --- Stratégie (scanner de signaux) ---
    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend: int = 100
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 72.0
    atr_period: int = 14
    breakout_lookback: int = 20
    min_atr_pct: float = 0.4
    max_atr_pct: float = 6.0
    atr_stop_mult: float = 2.0
    min_stop_pct: float = 2.0
    max_stop_pct: float = 8.0
    reward_risk: float = 2.5
    atr_trail_mult: float = 3.0
    max_hold_bars: int = 72

    # --- Gestion du risque (règles de survie, non négociables) ---
    risk_per_trade_pct: float = 2.0  # perte max visée par trade, en % du capital
    max_position_pct: float = 90.0  # taille max d'une position, en % du capital
    max_open_positions: int = 1
    survival_floor_pct: float = 75.0  # sous ce % du capital initial : tout vendre et s'arrêter
    daily_loss_limit_pct: float = 6.0  # perte journalière max avant pause des achats
    max_consecutive_losses: int = 3
    cooldown_hours: float = 12.0
    taker_fee_pct: float = 0.40  # frais taker Kraken Pro (palier le plus bas)
    slippage_pct: float = 0.05

    # --- Simulation ---
    paper_start_eur: float = 10.0

    # --- Cerveau IA ---
    brain: str = "auto"  # "auto", "claude" ou "rules"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    max_llm_calls_per_day: int = 6

    data_dir: Path = Path("data")

    @property
    def interval_seconds(self) -> int:
        return self.candle_interval * 60

    @property
    def round_trip_cost_pct(self) -> float:
        """Coût d'un aller-retour (achat + vente), frais et glissement inclus."""
        return 2 * (self.taker_fee_pct + self.slippage_pct)

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        settings = cls()
        for f in fields(cls):
            raw = _get(f.name.upper())
            if raw is None:
                continue
            current = getattr(settings, f.name)
            if isinstance(current, bool):
                value: object = raw.lower() in {"1", "true", "yes", "oui", "on"}
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            elif isinstance(current, list):
                value = [item.strip().upper() for item in raw.split(",") if item.strip()]
            elif isinstance(current, Path):
                value = Path(raw)
            else:
                value = raw
            setattr(settings, f.name, value)
        for key, value in overrides.items():
            if value is not None:
                setattr(settings, key, value)
        settings.mode = settings.mode.lower()
        settings.brain = settings.brain.lower()
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode not in {"paper", "live"}:
            raise ValueError("MODE doit valoir 'paper' ou 'live'")
        if self.brain not in {"auto", "claude", "rules"}:
            raise ValueError("BRAIN doit valoir 'auto', 'claude' ou 'rules'")
        if not 0 < self.risk_per_trade_pct <= 10:
            raise ValueError("RISK_PER_TRADE_PCT doit être entre 0 et 10")
        if not 0 < self.max_position_pct <= 100:
            raise ValueError("MAX_POSITION_PCT doit être entre 0 et 100")
        if not 0 <= self.survival_floor_pct < 100:
            raise ValueError("SURVIVAL_FLOOR_PCT doit être entre 0 et 100")
        if self.max_open_positions < 1:
            raise ValueError("MAX_OPEN_POSITIONS doit être >= 1")
        if self.min_stop_pct <= 0 or self.max_stop_pct < self.min_stop_pct:
            raise ValueError("MIN_STOP_PCT / MAX_STOP_PCT incohérents")
