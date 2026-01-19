import pytest

from src.trading_bot.config_utils import resolve_mt5_credentials


def test_resolve_mt5_credentials_precedence(tmp_path, monkeypatch):
    creds_path = tmp_path / "mt5_credentials.json"
    creds_path.write_text(
        '{"login": 111, "password": "file_pass", "server": "file_server"}',
        encoding="utf-8",
    )

    config_payload = {
        "mt5_credentials": {
            "login": 222,
            "password": "config_pass",
            "server": "config_server",
        }
    }

    monkeypatch.setenv("MT5_LOGIN", "333")
    monkeypatch.setenv("MT5_PASSWORD", "env_pass")
    monkeypatch.setenv("MT5_SERVER", "env_server")

    resolved = resolve_mt5_credentials(config_payload, [creds_path])

    assert resolved["is_complete"] is True
    assert resolved["credentials"]["login"] == "333"
    assert resolved["credentials"]["password"] == "env_pass"
    assert resolved["credentials"]["server"] == "env_server"
    assert "file:mt5_credentials.json" in resolved["sources"]
    assert "central_config" in resolved["sources"]
    assert "env" in resolved["sources"]


def test_resolve_mt5_credentials_incomplete():
    resolved = resolve_mt5_credentials({})
    assert resolved["is_complete"] is False
