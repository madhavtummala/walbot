"""Schwab streaming market data: real-time quotes, bars, books and trade prints.

The REST fetchers answer "what has this symbol done"; the streamer answers "what is it
doing right now". One websocket per user (opening a second is rejected with code 12), so
the client here is a process-wide singleton managed by :mod:`src.connectors.streaming`,
never constructed ad hoc by callers.

Protocol, per Schwab's Streamer guide:

1. ``GET /marketdata/v1/userPreference`` returns the ``streamerInfo`` block -- socket URL,
   customer id, correlation id, channel and function id.
2. Connect to the socket URL and send an ``ADMIN``/``LOGIN`` request carrying the OAuth
   access token (no ``Bearer`` prefix). A response with ``code == 0`` opens the session.
3. ``SUBS`` requests per service select symbols (``keys``) and fields. Content arrives as
   JSON frames keyed by stringified field ids; responses, data and heartbeats ride the same
   pipe under ``response``, ``data`` and ``notify``.

Access tokens last ~30 minutes and the socket dies with them, so every reconnect re-fetches
a fresh token rather than reusing one. Book services need level-2 entitlements; a rejected
subscription downgrades that service for the process instead of killing the session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

STREAMER_INFO_URL = "https://api.schwabapi.com/marketdata/v1/userPreference"

#: Field-id maps, straight from the Streamer guide's tables. Only fields worth shipping to
#: the algorithms are subscribed; the maps are also what turns incoming numeric keys into
#: named dicts.
LEVEL_ONE_EQUITY_FIELDS: dict[int, str] = {
    0: "symbol",
    1: "bid",
    2: "ask",
    3: "last",
    4: "bid_size",
    5: "ask_size",
    8: "total_volume",
    9: "last_size",
    34: "quote_time",
    35: "trade_time",
}
CHART_EQUITY_FIELDS: dict[int, str] = {
    0: "symbol",
    1: "open",
    2: "high",
    3: "low",
    4: "close",
    5: "volume",
    6: "sequence",
    7: "chart_time",
}
BOOK_FIELDS: dict[int, str] = {0: "symbol", 1: "book_time", 2: "bids", 3: "asks"}
TIMESALE_FIELDS: dict[int, str] = {0: "symbol", 1: "trade_time", 2: "price", 3: "size", 4: "sequence"}


def _fields_param(mapping: dict[int, str]) -> str:
    return ",".join(str(field) for field in sorted(mapping))


QUOTE_SERVICE = "LEVELONE_EQUITIES"
BAR_SERVICE = "CHART_EQUITY"
TRADE_SERVICE = "TIMESALE_EQUITY"
BOOK_SERVICES = ("NYSE_BOOK", "NASDAQ_BOOK")

#: The wire form of each subscribed service's field selection.
SERVICE_FIELDS: dict[str, str] = {
    QUOTE_SERVICE: _fields_param(LEVEL_ONE_EQUITY_FIELDS),
    BAR_SERVICE: _fields_param(CHART_EQUITY_FIELDS),
    TRADE_SERVICE: _fields_param(TIMESALE_FIELDS),
    **{service: _fields_param(BOOK_FIELDS) for service in BOOK_SERVICES},
}

#: No frame of any kind for this long means the socket is a zombie: reconnect.
SILENCE_RECONNECT_SECONDS = 90.0
RECV_TIMEOUT_SECONDS = 5.0


def _value(content: dict[str, Any], name: str, field_id: int) -> Any:
    """Read one field, tolerating named or numeric-string keys."""
    if name in content:
        return content[name]
    return content.get(str(field_id))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _book_side(entries: Any) -> list[dict[str, float]]:
    """Normalize one side of a book into ``[{price, size}]``, best level first.

    Level objects arrive either with numeric-string keys (``"0"`` price, ``"1"`` total
    volume) or legacy named ones; both fold into the same shape. Per-exchange breakdowns
    are kept out of the normalized view -- aggregate size behind each level is what the
    liquidity signals read.
    """
    levels: list[dict[str, float]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        price = _number(
            entry.get("price")
            if "price" in entry
            else entry.get("BidPrice") if "BidPrice" in entry else entry.get("AskPrice", entry.get("0"))
        )
        size = _number(
            entry.get("size")
            if "size" in entry
            else entry.get("BidSize") if "BidSize" in entry else entry.get("AskSize", entry.get("1"))
        )
        if price is None or size is None:
            continue
        levels.append({"price": price, "size": size})
    return levels


def _normalize_content(service: str, content: dict[str, Any]) -> dict[str, Any] | None:
    received_at = time.time()
    symbol = str(_value(content, "symbol", 0) or content.get("key") or "").upper()
    if not symbol:
        return None
    if service == QUOTE_SERVICE:
        quote = {"kind": "quote", "symbol": symbol, "received_at": received_at}
        for field_id, name in LEVEL_ONE_EQUITY_FIELDS.items():
            if field_id == 0:
                continue
            value = _value(content, name, field_id)
            if value is not None:
                quote[name] = value
        return quote
    if service == BAR_SERVICE:
        bar = {"kind": "bar", "symbol": symbol, "received_at": received_at}
        for field_id, name in CHART_EQUITY_FIELDS.items():
            if field_id == 0:
                continue
            value = _value(content, name, field_id)
            if value is not None:
                bar[name] = value
        return bar
    if service == TRADE_SERVICE:
        price = _number(_value(content, "price", 2))
        if price is None:
            return None
        return {
            "kind": "trade",
            "symbol": symbol,
            "price": price,
            "size": _number(_value(content, "size", 3)) or 0.0,
            "trade_time": _value(content, "trade_time", 1),
            "sequence": _value(content, "sequence", 4),
            "received_at": received_at,
        }
    if service in BOOK_SERVICES:
        bids_raw = _value(content, "bids", 2)
        asks_raw = _value(content, "asks", 3)
        if bids_raw is None and isinstance(content.get("BIDS"), list):
            bids_raw = content["BIDS"]
        if asks_raw is None and isinstance(content.get("ASKS"), list):
            asks_raw = content["ASKS"]
        bids = _book_side(bids_raw)
        asks = _book_side(asks_raw)
        bids.sort(key=lambda level: -level["price"])
        asks.sort(key=lambda level: level["price"])
        return {
            "kind": "book",
            "service": service,
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "book_time": _value(content, "book_time", 1),
            "received_at": received_at,
        }
    logger.debug("Ignoring streaming content from unhandled service %s", service)
    return None


def parse_stream_frame(raw: str | bytes) -> list[dict[str, Any]]:
    """One websocket frame -> normalized events.

    Responses become ``{"kind": "response", ...}`` events (login/subs acks and errors),
    heartbeats ``{"kind": "heartbeat"}``, and each data record its service's event kind.
    Parsing is pure so tests can script frames without a socket.
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    payload = json.loads(text)
    events: list[dict[str, Any]] = []
    for response in payload.get("response") or []:
        content = response.get("content") if isinstance(response.get("content"), dict) else {}
        events.append(
            {
                "kind": "response",
                "service": response.get("service"),
                "command": response.get("command"),
                "requestid": response.get("requestid"),
                "code": content.get("code"),
                "message": content.get("msg"),
            }
        )
    for note in payload.get("notify") or []:
        if "heartbeat" in note:
            events.append({"kind": "heartbeat"})
    for block in payload.get("data") or []:
        service = str(block.get("service") or "")
        for content in block.get("content") or []:
            if isinstance(content, dict):
                normalized = _normalize_content(service, content)
                if normalized:
                    events.append(normalized)
    return events


class StreamStore:
    """Thread-safe latest-state snapshots from the stream.

    The reader loop only ever appends; algorithm threads read between runs. Quotes, bars
    and books keep the newest value per symbol, trades a bounded ring per symbol, and every
    record carries a local ``received_at`` stamp so consumers can judge freshness themselves.
    """

    def __init__(self, max_trades_per_symbol: int = 1000) -> None:
        # Reentrant: the snapshot accessors delegate to the per-symbol ones, which take
        # the same lock.
        self._lock = threading.RLock()
        self._quotes: dict[str, dict[str, Any]] = {}
        self._bars: dict[str, dict[str, Any]] = {}
        self._books: dict[str, dict[str, Any]] = {}
        self._trades: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_trades_per_symbol))
        self.last_message_at: float = 0.0

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        with self._lock:
            if kind in ("heartbeat", "response"):
                self.last_message_at = time.time()
                return
            if kind == "quote":
                self._quotes[event["symbol"]] = event
            elif kind == "bar":
                self._bars[event["symbol"]] = event
            elif kind == "book":
                self._books[event["symbol"]] = event
            elif kind == "trade":
                self._trades[event["symbol"]].append(event)
            else:
                return
            self.last_message_at = time.time()

    def quote(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            quote = self._quotes.get(symbol.upper())
            return dict(quote) if quote else None

    def book(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            book = self._books.get(symbol.upper())
            return {**book, "bids": list(book["bids"]), "asks": list(book["asks"])} if book else None

    def trades(self, symbol: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(trade) for trade in self._trades.get(symbol.upper(), ())]

    def fresh_quotes(self, symbols: list[str], max_age_seconds: float) -> dict[str, dict[str, Any]]:
        cutoff = time.time() - max_age_seconds
        fresh: dict[str, dict[str, Any]] = {}
        with self._lock:
            for symbol in symbols:
                quote = self._quotes.get(symbol.upper())
                if quote and float(quote.get("received_at") or 0.0) >= cutoff:
                    fresh[symbol.upper()] = dict(quote)
        return fresh

    def snapshot_books(self, symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
        wanted = [symbol.upper() for symbol in symbols] if symbols is not None else None
        with self._lock:
            names = wanted or list(self._books)
            return {symbol: self.book(symbol) for symbol in names if self._books.get(symbol)}

    def snapshot_trades(self, symbols: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
        wanted = [symbol.upper() for symbol in symbols] if symbols is not None else None
        with self._lock:
            names = wanted or list(self._trades)
            return {symbol: self.trades(symbol) for symbol in names if self._trades.get(symbol)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "quote_symbols": sorted(self._quotes),
                "book_symbols": sorted(self._books),
                "last_message_at": self.last_message_at,
            }


def default_token_info(config) -> tuple[str, dict[str, Any]]:
    """Fresh OAuth token plus the ``streamerInfo`` block, via the shared Schwab session."""
    from src.brokerages.schwab.client import SchwabSession

    session = SchwabSession(config)
    token = session.access_token()
    preferences = session.get(STREAMER_INFO_URL).json()
    infos = preferences.get("streamerInfo") or []
    info = infos[0] if infos else {}
    missing = [
        key
        for key in (
            "streamerSocketUrl",
            "schwabClientCustomerId",
            "schwabClientCorrelId",
            "schwabClientChannel",
            "schwabClientFunctionId",
        )
        if not info.get(key)
    ]
    if missing:
        raise RuntimeError(f"Schwab userPreference is missing streamer fields: {', '.join(missing)}")
    return token, info


async def default_connect(url: str):
    try:
        from websockets.asyncio.client import connect
    except ImportError:  # websockets < 13
        from websockets.legacy.client import connect  # type: ignore[no-redef]

    return connect(url, open_timeout=10, ping_interval=20, ping_timeout=20, close_timeout=5)


class SchwabStreamClient:
    """Owns the websocket lifecycle: login, subscriptions, reconnect, snapshot store.

    ``start()`` returns immediately; a daemon thread runs the sessions. Every session
    re-fetches the token (they expire ~30 minutes) and re-subscribes everything, so a
    disconnect heals without caller involvement. Services whose subscription is refused --
    the books, without level-2 entitlements -- are disabled for the process rather than
    retried forever.
    """

    def __init__(
        self,
        config,
        symbols: list[str] | None = None,
        *,
        store: StreamStore | None = None,
        token_info_fn: Callable[[Any], tuple[str, dict[str, Any]]] | None = None,
        connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.store = store or StreamStore()
        self._token_info_fn = token_info_fn or default_token_info
        self._connect_fn = connect_fn or default_connect
        self._wanted: list[str] = []
        for symbol in symbols or []:
            self._add_wanted(symbol)
        self._pending_adds: list[str] = []
        self._pending_lock = threading.Lock()
        self._disabled_services: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {"connected": False, "error": "", "sessions": 0}

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="schwab-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def add_symbols(self, symbols: list[str]) -> None:
        added = False
        for symbol in symbols or []:
            if self._add_wanted(symbol):
                added = True
                with self._pending_lock:
                    self._pending_adds.append(symbol.upper())
        if added:
            logger.info("Streaming subscriptions now cover %s", ", ".join(self._wanted))

    def status(self) -> dict[str, Any]:
        state = dict(self._state)
        state.update(self.store.status())
        state["symbols"] = list(self._wanted)
        state["disabled_services"] = sorted(self._disabled_services)
        return state

    # -- internals ------------------------------------------------------------------

    def _add_wanted(self, symbol: str) -> bool:
        normalized = str(symbol or "").upper()
        if not normalized or normalized in self._wanted:
            return False
        self._wanted.append(normalized)
        return True

    def _run(self) -> None:
        backoff = 5.0
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
                backoff = 5.0
            except Exception as exc:  # noqa: BLE001 - any failure just schedules a retry
                if self._stop.is_set():
                    break
                self._state["connected"] = False
                self._state["error"] = str(exc)
                logger.warning("Schwab stream disconnected, retrying in %ss: %s", int(backoff), exc)
            if self._stop.is_set():
                break
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)

    def _request(self, request_id: int, service: str, command: str, info: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "service": service,
            "requestid": str(request_id),
            "command": command,
            "SchwabClientCustomerId": info["schwabClientCustomerId"],
            "SchwabClientCorrelId": info["schwabClientCorrelId"],
            "parameters": parameters,
        }

    async def _send(self, ws, request: dict[str, Any]) -> None:
        await ws.send(json.dumps({"requests": [request]}))

    async def _await_response(self, ws, request_id: str, service: str, command: str) -> dict[str, Any]:
        """Read frames until the ack for this request; data frames go to the store."""
        while True:
            raw = await ws.recv()
            for event in parse_stream_frame(raw):
                if event["kind"] == "response":
                    if str(event.get("requestid")) == str(request_id):
                        if event.get("code") != 0:
                            raise RuntimeError(f"{service} {command} refused: {event.get('message')}")
                        return event
                    self.store.apply(event)
                else:
                    self.store.apply(event)

    async def _session(self) -> None:
        token, info = await asyncio.to_thread(self._token_info_fn, self.config)
        connect = self._connect_fn
        async with await connect(info["streamerSocketUrl"]) as ws:
            self._state["sessions"] += 1
            await self._send(ws, self._request(0, "ADMIN", "LOGIN", info, {
                "Authorization": token,
                "SchwabClientChannel": info["schwabClientChannel"],
                "SchwabClientFunctionId": info["schwabClientFunctionId"],
            }))
            await self._await_response(ws, "0", "ADMIN", "LOGIN")
            self._state["connected"] = True
            self._state["error"] = ""
            logger.info("Schwab stream logged in (%s)", info["streamerSocketUrl"])

            next_request_id = 1
            for service, fields in SERVICE_FIELDS.items():
                if service in self._disabled_services or not self._wanted:
                    continue
                await self._send(ws, self._request(next_request_id, service, "SUBS", info, {
                    "keys": ",".join(self._wanted),
                    "fields": fields,
                }))
                next_request_id += 1
                try:
                    await self._await_response(ws, str(next_request_id - 1), service, "SUBS")
                except RuntimeError as exc:
                    self._disable_service(service, exc)

            last_frame_at = time.monotonic()
            while not self._stop.is_set():
                with self._pending_lock:
                    pending, self._pending_adds = self._pending_adds, []
                for service, fields in SERVICE_FIELDS.items():
                    if service in self._disabled_services or not pending:
                        continue
                    await self._send(ws, self._request(next_request_id, service, "ADD", info, {
                        "keys": ",".join(pending),
                        "fields": fields,
                    }))
                    next_request_id += 1
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_frame_at > SILENCE_RECONNECT_SECONDS:
                        raise RuntimeError(
                            f"stream silent for {SILENCE_RECONNECT_SECONDS:.0f}s; reconnecting"
                        ) from None
                    continue
                last_frame_at = time.monotonic()
                for event in parse_stream_frame(raw):
                    self.store.apply(event)
                    if event["kind"] == "response" and event.get("code") not in (None, 0):
                        self._disable_service(str(event.get("service")), event.get("message"))

    def _disable_service(self, service: str, reason: Any) -> None:
        if not service or service == "ADMIN":
            return
        if service in self._disabled_services:
            return
        self._disabled_services.add(service)
        logger.warning("Schwab stream dropping %s for this process: %s", service, reason)
