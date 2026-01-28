from src.trading_bot.risk_manager import create_scalping_risk_manager


def test_rr_epsilon_allows_min_rr_edge():
    risk_manager = create_scalping_risk_manager(
        overrides={
            "risk_settings": {"MIN_RISK_REWARD": 0.9},
        }
    )
    risk_manager.settings["MIN_RISK_REWARD"] = 0.9
    risk_manager.settings["RR_EPSILON"] = 1e-6

    entry = 2000.0
    stop_loss = 1995.5
    sl_distance = entry - stop_loss
    take_profit = entry + sl_distance * (0.9 - 5e-7)

    params = risk_manager.calculate_scalping_position_size(
        account_equity=10000.0,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_confidence=90.0,
        atr_value=3.0,
        market_volatility=1.0,
        session="LONDON",
        max_risk_usd=50.0,
    )

    assert params.validation_passed
