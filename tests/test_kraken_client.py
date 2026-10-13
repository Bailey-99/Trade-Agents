from __future__ import annotations

import pytest
import requests

from kraken_agent.kraken_client import KrakenClient, KrakenError


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)


def client(responses, **kwargs):
    return KrakenClient(
        "key", "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg==",
        session=FakeSession(responses), min_interval=0, sleep=lambda s: None, **kwargs,
    )  # fmt: skip


def test_signature_matches_kraken_documentation_example():
    secret = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    data = {"nonce": "1616492376594", "ordertype": "limit", "pair": "XBTUSD", "price": 37500, "type": "buy", "volume": 1.25}
    assert (
        KrakenClient.sign("/0/private/AddOrder", data, secret)
        == "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="
    )


def test_private_call_is_signed_with_increasing_nonce():
    c = client([{"error": [], "result": {"ZEUR": {"balance": "10"}}}, {"error": [], "result": {}}])
    assert c.balance_ex() == {"ZEUR": {"balance": "10"}}
    c.balance_ex()
    first, second = c.session.calls
    assert first[0] == "POST" and first[1].endswith("/0/private/BalanceEx")
    assert first[2]["headers"]["API-Key"] == "key"
    nonce1 = int(first[2]["data"].split("nonce=")[1])
    nonce2 = int(second[2]["data"].split("nonce=")[1])
    assert nonce2 > nonce1


def test_private_call_without_credentials_fails():
    c = KrakenClient(session=FakeSession([]), min_interval=0)
    with pytest.raises(KrakenError):
        c.balance_ex()


def test_kraken_error_is_raised():
    c = client([{"error": ["EOrder:Insufficient funds"]}])
    with pytest.raises(KrakenError) as exc:
        c.add_order(pair="XBTEUR", type="buy", ordertype="market", volume="1")
    assert exc.value.has("Insufficient funds")
    assert not exc.value.retryable


def test_rate_limit_is_retried():
    c = client([{"error": ["EAPI:Rate limit exceeded"]}, {"error": [], "result": {"ok": 1}}])
    assert c.public("Time") == {"ok": 1}
    assert len(c.session.calls) == 2


def test_order_is_not_retried_after_network_error():
    c = client([requests.ConnectionError("boom"), {"error": [], "result": {"txid": ["X"]}}])
    with pytest.raises(requests.ConnectionError):
        c.add_order(pair="XBTEUR", type="buy", ordertype="market", volume="1")
    assert len(c.session.calls) == 1


def test_read_call_is_retried_after_network_error():
    c = client([requests.ConnectionError("boom"), {"error": [], "result": {"ZEUR": {}}}])
    assert c.balance_ex() == {"ZEUR": {}}


def test_client_has_no_withdrawal_method():
    assert not any("withdraw" in name.lower() for name in dir(KrakenClient))
