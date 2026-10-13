from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_agent.models import Position
from kraken_agent.risk import RiskManager
from kraken_agent.state import AgentState


def state(**kw) -> AgentState:
    s = AgentState(initial_capital=10.0, day_start_equity=10.0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_position_size_targets_risk_budget(settings, btc_pair):
    risk = RiskManager(settings)
    plan, why = risk.size_position(
        state=state(), equity=10.0, cash=10.0, price=60_000.0, stop_pct=3.0, pair=btc_pair
    )
    assert plan is not None, why
    expected_loss = 10.0 * settings.risk_per_trade_pct / 100
    assert plan.max_loss == pytest.approx(expected_loss, rel=0.02)
    assert plan.volume >= btc_pair.ordermin


def test_position_size_capped_by_cash(settings, btc_pair):
    risk = RiskManager(settings)
    plan, _ = risk.size_position(state=state(), equity=10.0, cash=4.0, price=60_000.0, stop_pct=2.0, pair=btc_pair)
    assert plan is not None
    assert plan.notional * (1 + settings.taker_fee_pct / 100) <= 4.0


def test_kraken_minimum_is_accepted_only_within_twice_the_risk(settings, btc_pair):
    risk = RiskManager(settings)
    big_min = replace(btc_pair, ordermin=0.0001)  # 6€ à 60 000€
    plan, _ = risk.size_position(state=state(), equity=10.0, cash=10.0, price=60_000.0, stop_pct=3.0, pair=big_min)
    assert plan is not None and "minimum" in plan.note
    # Stop large : 6€ au minimum ferait risquer plus de 2x le budget -> refusé.
    plan, why = risk.size_position(state=state(), equity=10.0, cash=10.0, price=60_000.0, stop_pct=8.0, pair=big_min)
    assert plan is None
    huge_min = replace(btc_pair, ordermin=0.001)  # 60€ : impossible avec 10€
    plan, why = risk.size_position(state=state(), equity=10.0, cash=10.0, price=60_000.0, stop_pct=3.0, pair=huge_min)
    assert plan is None and "minimum" in why


def test_stop_out_never_breaches_survival_floor(settings, btc_pair):
    risk = RiskManager(settings)
    st = state()
    plan, _ = risk.size_position(state=st, equity=7.7, cash=7.7, price=60_000.0, stop_pct=8.0, pair=btc_pair)
    if plan is not None:
        assert 7.7 - plan.max_loss > risk.survival_floor(st)


def test_can_open_rules(settings):
    risk = RiskManager(settings)
    assert risk.can_open(state(), 10.0, now=0)[0]
    assert not risk.can_open(state(status="halted", halt_reason="x"), 10.0, now=0)[0]
    pos = Position("XBTEUR", "XXBT", 1, 1, 1, 0, 0.9, 0.9, 1, 0)
    assert not risk.can_open(state(positions=[pos]), 10.0, now=0)[0]
    assert not risk.can_open(state(cooldown_until=100.0), 10.0, now=0)[0]
    ok, why = risk.can_open(state(), 9.0, now=0)  # -10% dans la journée
    assert not ok and "journalière" in why


def test_consecutive_losses_trigger_cooldown(settings):
    risk = RiskManager(settings)
    st = state()
    for _ in range(settings.max_consecutive_losses):
        risk.record_result(st, -0.1, now=1000.0)
    assert st.cooldown_until == pytest.approx(1000.0 + settings.cooldown_hours * 3600)
    risk.record_result(st, 0.2, now=2000.0)
    assert st.consecutive_losses == 0


def test_survival_floor(settings):
    risk = RiskManager(settings)
    st = state()
    assert risk.survival_floor(st) == pytest.approx(7.5)
    assert risk.breached_floor(st, 7.49)
    assert not risk.breached_floor(st, 8.0)
