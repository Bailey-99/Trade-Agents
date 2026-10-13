"""État persistant de l'agent (JSON) et journaux (JSONL)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import Position


def _default_stats() -> dict[str, float]:
    return {"trades": 0, "wins": 0, "losses": 0, "realized_pnl": 0.0, "fees_paid": 0.0}


def _default_llm() -> dict[str, Any]:
    return {
        "day": "",
        "calls_today": 0,
        "total_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "est_cost_usd": 0.0,
    }


@dataclass
class AgentState:
    mode: str = "paper"
    status: str = "alive"  # "alive" ou "halted"
    halt_reason: str = ""
    initial_capital: float | None = None
    peak_equity: float = 0.0
    last_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    last_analysis_candle: int = -1
    paper_cash: float = 0.0
    paper_holdings: dict[str, float] = field(default_factory=dict)
    positions: list[Position] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=_default_stats)
    llm: dict[str, Any] = field(default_factory=_default_llm)
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def position(self, pair: str) -> Position | None:
        return next((p for p in self.positions if p.pair == pair), None)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["positions"] = [p.to_dict() for p in self.positions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in data.items() if k in known}
        values["positions"] = [Position.from_dict(p) for p in data.get("positions", [])]
        state = cls(**values)
        state.stats = {**_default_stats(), **state.stats}
        state.llm = {**_default_llm(), **state.llm}
        return state


class StateStore:
    """Sauvegarde atomique de l'état. `data_dir=None` : tout reste en mémoire (backtest, tests)."""

    def __init__(self, data_dir: Path | None, mode: str):
        self.mode = mode
        self.data_dir = data_dir
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_log: list[tuple[str, dict[str, Any]]] = []

    @property
    def state_path(self) -> Path | None:
        return self.data_dir / f"state_{self.mode}.json" if self.data_dir else None

    def load(self) -> AgentState:
        path = self.state_path
        if path is None or not path.exists():
            return AgentState(mode=self.mode)
        return AgentState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, state: AgentState) -> None:
        path = self.state_path
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def append(self, journal: str, record: dict[str, Any]) -> None:
        if self.data_dir is None:
            self.memory_log.append((journal, record))
            return
        path = self.data_dir / f"{journal}_{self.mode}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
