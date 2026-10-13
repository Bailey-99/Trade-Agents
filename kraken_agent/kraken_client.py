"""Client minimal pour l'API REST de Kraken (spot).

Ce client n'expose volontairement AUCUNE méthode de retrait : l'agent ne peut
pas sortir d'argent du compte. Créez quand même votre clé API sans la
permission « Withdraw Funds ».
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Callable, Iterable

import requests

log = logging.getLogger(__name__)

# Erreurs pour lesquelles Kraken garantit que la requête n'a pas été traitée :
# on peut réessayer sans risque, même pour un ordre.
_RETRYABLE_PREFIXES = (
    "EAPI:Rate limit exceeded",
    "EAPI:Invalid nonce",
    "EService:Unavailable",
    "EService:Busy",
    "EGeneral:Temporary lockout",
)


class KrakenError(RuntimeError):
    def __init__(self, errors: Iterable[str], method: str = ""):
        self.errors = list(errors)
        self.method = method
        super().__init__(f"{method}: {', '.join(self.errors)}" if method else ", ".join(self.errors))

    @property
    def retryable(self) -> bool:
        return any(e.startswith(_RETRYABLE_PREFIXES) for e in self.errors)

    def has(self, fragment: str) -> bool:
        return any(fragment in e for e in self.errors)


class KrakenClient:
    BASE_URL = "https://api.kraken.com"
    VERSION = "0"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        *,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
        timeout: float = 20.0,
        min_interval: float = 1.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_call = 0.0
        self._last_nonce = 0

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    # --- signature ---------------------------------------------------------

    @staticmethod
    def sign(urlpath: str, data: dict[str, Any], secret: str) -> str:
        """Signature API-Sign : HMAC-SHA512(uri + SHA256(nonce + postdata))."""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _next_nonce(self) -> int:
        nonce = max(time.time_ns() // 1000, self._last_nonce + 1)
        self._last_nonce = nonce
        return nonce

    # --- transport ---------------------------------------------------------

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def public(self, method: str, params: dict[str, Any] | None = None) -> Any:
        urlpath = f"/{self.VERSION}/public/{method}"

        def send() -> requests.Response:
            return self.session.get(self.base_url + urlpath, params=params or {}, timeout=self.timeout)

        return self._call(method, send, network_retry=True)

    def private(self, method: str, data: dict[str, Any] | None = None, *, idempotent: bool = True) -> Any:
        """Appel privé signé.

        `idempotent=False` (ex. AddOrder) : en cas d'erreur réseau on ne réessaie
        pas, car l'ordre a peut-être été pris en compte ; l'agent se resynchronise
        avec les soldes au tour suivant.
        """
        if not self.has_credentials:
            raise KrakenError(["clés API Kraken manquantes (KRAKEN_API_KEY / KRAKEN_API_SECRET)"], method)
        urlpath = f"/{self.VERSION}/private/{method}"

        def send() -> requests.Response:
            payload = dict(data or {})
            payload["nonce"] = self._next_nonce()
            headers = {
                "API-Key": self.api_key,
                "API-Sign": self.sign(urlpath, payload, self.api_secret),
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            }
            return self.session.post(
                self.base_url + urlpath,
                data=urllib.parse.urlencode(payload),
                headers=headers,
                timeout=self.timeout,
            )

        return self._call(method, send, network_retry=idempotent)

    def _call(self, method: str, send: Callable[[], requests.Response], *, network_retry: bool) -> Any:
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                response = send()
                response.raise_for_status()
                body = response.json()
            except (requests.RequestException, ValueError) as exc:
                if network_retry and attempt <= self.max_retries:
                    delay = 2**attempt
                    log.warning("Kraken %s : erreur réseau (%s), nouvel essai dans %ss", method, exc, delay)
                    self._sleep(delay)
                    continue
                raise
            errors = body.get("error") or []
            if errors:
                err = KrakenError(errors, method)
                if err.retryable and attempt <= self.max_retries:
                    delay = 2**attempt
                    log.warning("Kraken %s : %s, nouvel essai dans %ss", method, err, delay)
                    self._sleep(delay)
                    continue
                raise err
            return body.get("result", {})

    # --- raccourcis --------------------------------------------------------

    def asset_pairs(self) -> dict[str, Any]:
        return self.public("AssetPairs")

    def ticker(self, pairs: Iterable[str] | None = None) -> dict[str, Any]:
        pairs = list(pairs or [])
        return self.public("Ticker", {"pair": ",".join(pairs)} if pairs else None)

    def ohlc(self, pair: str, interval: int = 60, since: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair, "interval": interval}
        if since is not None:
            params["since"] = since
        return self.public("OHLC", params)

    def balance_ex(self) -> dict[str, Any]:
        return self.private("BalanceEx")

    def open_orders(self) -> dict[str, Any]:
        return self.private("OpenOrders")

    def query_orders(self, txids: Iterable[str]) -> dict[str, Any]:
        return self.private("QueryOrders", {"txid": ",".join(txids)})

    def add_order(self, **params: Any) -> dict[str, Any]:
        return self.private("AddOrder", params, idempotent=False)

    def cancel_order(self, txid: str) -> dict[str, Any]:
        return self.private("CancelOrder", {"txid": txid})
