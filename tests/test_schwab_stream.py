"""Schwab streaming: frame parsing, snapshot store, session handshake, price overlay."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.connectors import service as connectors
from src.connectors.market import schwab_stream as stream
from src.connectors.streaming import reset_stream
from src.core.config import Config
from src.core.config.coercion import _normalize_data_sources
from src.data.signals.orderbook import iceberg_evidence, quote_spread


def _frame(payload: dict) -> str:
    return json.dumps(payload)


def _ack(service: str, command: str, request_id: str, code: int = 0) -> dict:
    return {
        "response": [
            {"service": service, "command": command, "requestid": request_id, "content": {"code": code, "msg": "OK" if code == 0 else "denied"}}
        ]
    }


# -- parsing ----------------------------------------------------------------------


def test_parse_login_ack_and_heartbeat() -> None:
    events = stream.parse_stream_frame(_frame(_ack("ADMIN", "LOGIN", "0")))
    assert events == [{"kind": "response", "service": "ADMIN", "command": "LOGIN", "requestid": "0", "code": 0, "message": "OK"}]

    events = stream.parse_stream_frame(_frame({"notify": [{"heartbeat": ["1730000000000"]}]}))
    assert events == [{"kind": "heartbeat"}]


def test_parse_level_one_quote() -> None:
    frame = {
        "data": [
            {
                "service": "LEVELONE_EQUITIES",
                "timestamp": 1730000000000,
                "content": [
                    {"key": "SPY", "1": 559.2, "2": 559.4, "3": 559.3, "4": 300, "5": 200, "8": 123456, "9": 100, "34": 1730000000000}
                ],
            }
        ]
    }
    events = stream.parse_stream_frame(_frame(frame))
    assert len(events) == 1
    quote = events[0]
    assert quote["kind"] == "quote"
    assert quote["symbol"] == "SPY"
    assert quote["bid"] == 559.2
    assert quote["ask"] == 559.4
    assert quote["last"] == 559.3
    assert quote["bid_size"] == 300
    assert quote["total_volume"] == 123456
    assert quote["quote_time"] == 1730000000000


def test_parse_chart_bar_uses_official_field_order() -> None:
    # Schwab's Streamer guide: 0 key, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 sequence, 7 chartTime.
    frame = {
        "data": [
            {
                "service": "CHART_EQUITY",
                "timestamp": 1730000000000,
                "content": [{"0": "XSD", "1": 180.1, "2": 180.9, "3": 180.0, "4": 180.7, "5": 42000, "6": 77, "7": 1729999940000}],
            }
        ]
    }
    bar = stream.parse_stream_frame(_frame(frame))[0]
    assert bar["kind"] == "bar"
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (180.1, 180.9, 180.0, 180.7)
    assert bar["volume"] == 42000
    assert bar["chart_time"] == 1729999940000


def test_parse_book_numeric_and_named_keys() -> None:
    numeric = {
        "data": [
            {
                "service": "NASDAQ_BOOK",
                "timestamp": 1730000000000,
                "content": [
                    {
                        "0": "SPY",
                        "1": 1730000000000,
                        "2": [{"0": 559.0, "1": 800}, {"0": 559.1, "1": 500}],
                        "3": [{"0": 559.6, "1": 900}, {"0": 559.5, "1": 250}],
                    }
                ],
            }
        ]
    }
    book = stream.parse_stream_frame(_frame(numeric))[0]
    assert book["kind"] == "book"
    assert book["symbol"] == "SPY"
    assert [level["price"] for level in book["bids"]] == [559.1, 559.0]
    assert [level["size"] for level in book["asks"]] == [250, 900]

    named = {
        "data": [
            {
                "service": "NYSE_BOOK",
                "timestamp": 1730000000000,
                "content": [{"key": "QQQ", "BIDS": [{"BidPrice": 480.0, "BidSize": 100}], "ASKS": [{"AskPrice": 480.2, "AskSize": 150}]}],
            }
        ]
    }
    book = stream.parse_stream_frame(_frame(named))[0]
    assert book["bids"] == [{"price": 480.0, "size": 100}]
    assert book["asks"] == [{"price": 480.2, "size": 150}]


def test_parse_timesale_trade() -> None:
    frame = {
        "data": [
            {"service": "TIMESALE_EQUITY", "timestamp": 1730000000000, "content": [{"0": "SPY", "1": 1730000000000, "2": 559.31, "3": 42, "4": 90815}]}
        ]
    }
    trade = stream.parse_stream_frame(_frame(frame))[0]
    assert trade["kind"] == "trade"
    assert trade["symbol"] == "SPY"
    assert trade["price"] == 559.31
    assert trade["size"] == 42


# -- store ------------------------------------------------------------------------


def test_store_freshness_and_snapshots() -> None:
    import time as time_module

    now = time_module.time()
    store = stream.StreamStore()
    store.apply({"kind": "quote", "symbol": "SPY", "bid": 1.0, "ask": 1.1, "received_at": now})
    store.apply({"kind": "quote", "symbol": "QQQ", "bid": 2.0, "ask": 2.2, "received_at": now - 120})
    store.apply({"kind": "trade", "symbol": "SPY", "price": 1.05, "size": 10, "received_at": now})
    store.apply({"kind": "book", "symbol": "SPY", "bids": [{"price": 1.0, "size": 5}], "asks": [], "received_at": now})

    fresh = store.fresh_quotes(["SPY", "QQQ"], max_age_seconds=30)
    assert set(fresh) == {"SPY"}
    assert store.book("SPY")["bids"] == [{"price": 1.0, "size": 5}]
    assert store.trades("SPY")[0]["price"] == 1.05
    assert store.snapshot_books(["SPY", "QQQ"])["SPY"]["symbol"] == "SPY"


# -- session handshake ------------------------------------------------------------


class _EOF(Exception):
    pass


class FakeWebSocket:
    """Plays back scripted frames; records everything sent."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        if not self.script:
            raise _EOF()
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)


def _token_info(config):
    return "token-1", {
        "streamerSocketUrl": "wss://streamer-api.schwab.com/ws",
        "schwabClientCustomerId": "CUST",
        "schwabClientCorrelId": "CORR",
        "schwabClientChannel": "N9",
        "schwabClientFunctionId": "APIAPP",
    }


def _run_session(client: stream.SchwabStreamClient, ws: FakeWebSocket) -> None:
    client._connect_fn = lambda url: _Immediate(ws)
    try:
        asyncio.run(client._session())
    except _EOF:
        pass


class _Immediate:
    """Awaitable async-context-manager wrapper so ``async with await connect(url)`` works."""

    def __init__(self, ws: FakeWebSocket) -> None:
        self._ws = ws

    def __await__(self):
        if False:
            yield
        return self._ws


def test_session_logs_in_subscribes_and_feeds_store() -> None:
    script: list = [_ack("ADMIN", "LOGIN", "0")]
    for index, service in enumerate(stream.SERVICE_FIELDS, start=1):
        script.append(_ack(service, "SUBS", str(index)))
    script.append(
        {
            "data": [
                {
                    "service": "LEVELONE_EQUITIES",
                    "timestamp": 0,
                    "content": [{"key": "SPY", "1": 10.0, "2": 10.2, "3": 10.1}],
                }
            ]
        }
    )
    ws = FakeWebSocket(script)
    client = stream.SchwabStreamClient(Config(), ["SPY"], token_info_fn=_token_info)

    _run_session(client, ws)

    requests = [message["requests"][0] for message in ws.sent]
    login = requests[0]
    assert login["service"] == "ADMIN" and login["command"] == "LOGIN"
    assert login["parameters"]["Authorization"] == "token-1"
    assert login["SchwabClientCustomerId"] == "CUST"

    subs = {request["service"]: request for request in requests[1:]}
    assert set(subs) == set(stream.SERVICE_FIELDS)
    assert subs["LEVELONE_EQUITIES"]["parameters"]["keys"] == "SPY"
    assert subs["LEVELONE_EQUITIES"]["parameters"]["fields"].startswith("0,1")
    assert subs["CHART_EQUITY"]["parameters"]["fields"] == "0,1,2,3,4,5,6,7"

    quote = client.store.quote("SPY")
    assert quote and quote["last"] == 10.1
    assert client.status()["connected"] is True


def test_session_downgrades_refused_book_service() -> None:
    script: list = [_ack("ADMIN", "LOGIN", "0")]
    for index, service in enumerate(stream.SERVICE_FIELDS, start=1):
        code = 3 if service == "NASDAQ_BOOK" else 0
        script.append(_ack(service, "SUBS", str(index), code=code))
    ws = FakeWebSocket(script)
    client = stream.SchwabStreamClient(Config(), ["SPY"], token_info_fn=_token_info)

    _run_session(client, ws)

    assert client.status()["disabled_services"] == ["NASDAQ_BOOK"]
    subs = [message["requests"][0]["service"] for message in ws.sent][1:]
    assert subs.count("NASDAQ_BOOK") == 1


# -- price overlay ----------------------------------------------------------------


class StubStreamClient:
    def __init__(self, store: stream.StreamStore) -> None:
        self.store = store

    def add_symbols(self, symbols) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def status(self) -> dict:
        return {}


@pytest.fixture
def stubbed_stream():
    reset_stream()

    def install(store: stream.StreamStore):
        import src.connectors.streaming as streaming

        streaming._client = StubStreamClient(store)
        return streaming

    yield install
    reset_stream()


def _quote_config() -> Config:
    return Config(
        market_data_provider_order=["finnhub"],
        data_source_configs={
            "market_data": {"providers": {"finnhub": {"enabled": True, "api_key": "key"}}}
        },
    )


def test_the_price_an_algorithm_sizes_with_never_comes_from_the_stream(stubbed_stream, monkeypatch) -> None:
    """A live socket must not change what a run trades.

    ``load_current_prices`` feeds ``context.latest_prices``, which prices every intent --
    ``resolve_target_shares`` turns a weight or a notional into share counts with it. The
    stream used to answer first for any symbol it had seen in the last fifteen seconds, which
    made that number three things it must not be: different from what a replay sees (there is
    no socket in a backtest), differently shaped (a stream quote carries bid/ask, a REST
    snapshot does not), and dependent on whether that window happened to be open.
    """
    import time as time_module

    from src.connectors import base
    from tests.test_connectors import use_provider

    store = stream.StreamStore()
    store.apply({
        "kind": "quote", "symbol": "SPY", "bid": 559.2, "ask": 559.4, "last": 559.3,
        "received_at": time_module.time(),
    })
    stubbed_stream(store)

    use_provider(monkeypatch, "finnhub", price=lambda symbols, _config, **_kw: {
        symbol: {"symbol": symbol, "price": 101.0, "provider": "finnhub", "raw": {}}
        for symbol in symbols
    })
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(base, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(base, "save_cached_payload", lambda *args, **_kwargs: None)

    quotes = connectors.load_current_prices(["SPY"], _quote_config())

    # The stream holds a fresher quote for SPY and is ignored regardless.
    assert store.fresh_quotes(["SPY"], 60.0), "the fixture only bites while the stream has one"
    assert quotes["SPY"]["provider"] == "finnhub"
    assert quotes["SPY"]["price"] == 101.0


def test_prices_come_from_the_provider_walk(stubbed_stream, monkeypatch) -> None:
    store = stream.StreamStore()
    store.apply({"kind": "quote", "symbol": "SPY", "bid": 559.2, "ask": 559.4, "received_at": 1.0})
    stubbed_stream(store)

    from src.connectors import base
    from tests.test_connectors import use_provider

    use_provider(monkeypatch, "finnhub", price=lambda symbols, _config, **_kw: {
        symbol: {"symbol": symbol, "price": 101.0, "timestamp": "2026-08-20T15:30:00+00:00", "provider": "finnhub", "raw": {}}
        for symbol in symbols
    })
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(base, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(base, "save_cached_payload", lambda *args, **_kwargs: None)

    quotes = connectors.load_current_prices(["SPY"], _quote_config())

    assert quotes["SPY"]["provider"] == "finnhub"
    assert quotes["SPY"]["price"] == 101.0


# -- microstructure signals -------------------------------------------------------


def test_quote_spread() -> None:
    assert quote_spread({"bid": 100.0, "ask": 100.1})["spread"] == pytest.approx(0.1)
    assert quote_spread({"bid": 100.0, "ask": 100.1})["spread_bps"] == pytest.approx(10.0)
    assert quote_spread({"last": 100.0}) is None
    assert quote_spread({"bid": 101.0, "ask": 100.0}) is None


def test_iceberg_evidence_flags_hidden_liquidity() -> None:
    book = {"bids": [{"price": 100.0, "size": 100}], "asks": []}
    trades = [{"price": 100.0, "size": size} for size in (150, 120, 130)]
    candidates = iceberg_evidence(book, trades)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["price"] == 100.0
    assert candidate["executed_volume"] == 400
    assert candidate["displayed_size"] == 100
    assert candidate["hidden_multiple"] == 4.0

    # A level that traded lightly relative to its displayed size is not evidence.
    quiet_book = {"bids": [{"price": 50.0, "size": 1000}], "asks": []}
    quiet_trades = [{"price": 50.0, "size": 100} for _ in range(5)]
    assert iceberg_evidence(quiet_book, quiet_trades) == []
    assert iceberg_evidence(None, trades) == []


# -- config wiring ----------------------------------------------------------------


def test_streaming_market_data_category_normalizes() -> None:
    sources = _normalize_data_sources({"data_sources": {"streaming_market_data": {"providers": {"schwab": {}}}}})
    assert sources["streaming_market_data"]["provider_order"] == ["schwab"]

    empty = _normalize_data_sources({"data_sources": {}})
    assert empty["streaming_market_data"]["provider_order"] == []


def test_config_defaults_keep_streaming_off() -> None:
    config = Config()
    assert config.streaming_market_data_provider_order == []
