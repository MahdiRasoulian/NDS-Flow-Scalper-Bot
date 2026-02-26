import logging
from types import SimpleNamespace

import MetaTrader5 as mt5

from src.trading_bot.mt5_client import MT5Client


def _client() -> MT5Client:
    client = MT5Client.__new__(MT5Client)
    client._logger = logging.getLogger("test")
    return client


def test_sanitize_mt5_comment_handles_none_and_invalid_tokens() -> None:
    client = _client()
    assert client._sanitize_mt5_comment(None) == "BOT"
    assert client._sanitize_mt5_comment("   ") == "BOT"
    assert client._sanitize_mt5_comment("None") == "BOT"
    assert client._sanitize_mt5_comment("null") == "BOT"


def test_sanitize_mt5_comment_ascii_and_length_capping() -> None:
    client = _client()
    # includes emoji + arabic + newline + extra spaces
    raw = "NDS Scalping 🚀 - آسيا\n  LONDON SESSION"
    cleaned = client._sanitize_mt5_comment(raw)
    assert cleaned == "NDSScalping-LONDONSESSION"
    assert len(cleaned) <= 31

    long_comment = "NDS Scalping - VERY LONG SESSION NAME WITH DETAILS"
    assert client._sanitize_mt5_comment(long_comment) == "NDSScalping-VERYLONGSESSIONNAME"


def test_sanitize_mt5_request_always_provides_safe_comment() -> None:
    client = _client()

    no_comment = client.sanitize_mt5_request({"symbol": "XAUUSD", "volume": 0.02, "comment": None})
    assert no_comment["comment"] == "BOT"

    missing_comment = client.sanitize_mt5_request({"symbol": "XAUUSD", "volume": 0.02})
    assert missing_comment["comment"] == "BOT"

    unicode_comment = client.sanitize_mt5_request({"comment": "جلسه لندن 🚀"})
    assert unicode_comment["comment"] == "BOT"


def test_order_send_retries_without_comment_when_broker_rejects_comment(monkeypatch) -> None:
    client = _client()
    client._logger = logging.getLogger("test")
    client._mt5_lock = None

    state = {"calls": []}

    def fake_order_send(req):
        state["calls"].append(dict(req))
        if "comment" in req:
            return None
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            comment="done",
            order=123,
            deal=456,
            price=1.0,
            volume=0.01,
        )

    monkeypatch.setattr(mt5, "order_send", fake_order_send)
    client._mt5_call = lambda func, *a, **k: func(*a, **k)
    client._mt5_last_error = lambda: (-2, 'Invalid "comment" argument')

    result = client._order_send_with_retry(
        request={"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 1.0, "comment": "BAD"},
        symbol="XAUUSD",
        context="market",
        retry_on_none=False,
    )

    assert result["success"] is True
    assert result["ticket"] == 123
    assert any("comment" not in call for call in state["calls"])


def test_sanitize_mt5_request_blocks_limit_order_types() -> None:
    client = _client()
    blocked_types = [
        getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2),
        getattr(mt5, "ORDER_TYPE_SELL_LIMIT", 3),
    ]
    for blocked in blocked_types:
        try:
            client.sanitize_mt5_request({"type": blocked, "symbol": "XAUUSD", "volume": 0.01})
            assert False, "Expected ValueError for blocked limit type"
        except ValueError as exc:
            assert "disabled" in str(exc).lower()


def test_order_send_wrapper_blocks_limit_order_requests(monkeypatch) -> None:
    client = _client()
    client._logger = logging.getLogger("test")
    client._mt5_lock = None
    client._mt5_call = lambda func, *a, **k: func(*a, **k)
    client._mt5_last_error = lambda: (0, "ok")

    monkeypatch.setattr(mt5, "symbol_info", lambda _symbol: None)
    monkeypatch.setattr(mt5, "order_send", lambda _req: None)

    blocked = getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2)
    try:
        client._order_send_with_retry(
            request={"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": blocked, "price": 1.0},
            symbol="XAUUSD",
            context="pending",
            retry_on_none=False,
        )
        assert False, "Expected ValueError for blocked limit request"
    except ValueError as exc:
        assert "blocked limit order request" in str(exc).lower()
