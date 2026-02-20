from src.trading_bot.mt5_client import MT5Client


def _client() -> MT5Client:
    return MT5Client.__new__(MT5Client)


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
