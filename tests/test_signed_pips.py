from src.trading_bot.contracts import compute_pips


def test_compute_pips_signed_buy():
    cfg = {"POINT_SIZE": 0.01}
    result = compute_pips("XAUUSD", 100.0, 101.0, side="BUY", config_payload=cfg)
    assert abs(result - 10.0) < 1e-6


def test_compute_pips_signed_sell():
    cfg = {"POINT_SIZE": 0.01}
    result = compute_pips("XAUUSD", 100.0, 101.0, side="SELL", config_payload=cfg)
    assert abs(result + 10.0) < 1e-6
