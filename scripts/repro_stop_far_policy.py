from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def main() -> None:
    risk_manager = create_scalping_risk_manager()
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.1,
            "STOP_MAX_DEVIATION_PIPS": 70.0,
            "STOP_HARD_REJECT_PIPS": 120.0,
            "STOP_CONVERT_TO_LIMIT_PIPS": 25.0,
            "TREND_STRENGTH_ADX_MIN": 25.0,
            "MEAN_REVERSION_ADX_MAX": 18.0,
            "MAX_ENTRY_CAP_PIPS": 40.0,
        }
    )
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)

    analysis_payload = {
        "signal": "BUY",
        "confidence": 78.0,
        "entry_level": 5084.94,
        "entry_model": "STOP",
        "entry_idea": {
            "entry_level": 5084.94,
            "entry_model": "STOP",
        },
        "market_metrics": {
            "atr": 5.0,
            "adx": 14.2,
            "volatility_state": "LOW",
        },
        "context": {
            "entry_idea": {
                "entry_level": 5084.94,
                "entry_model": "STOP",
            },
        },
    }

    live_snapshot = LivePriceSnapshot(bid=5073.21, ask=5073.31)
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    print("allowed:", finalized.is_trade_allowed)
    print("order_type:", finalized.order_type)
    print("entry:", finalized.entry_price)
    print("reject_reason:", finalized.reject_reason)
    print("decision_notes:")
    for note in finalized.decision_notes:
        print(" -", note)


if __name__ == "__main__":
    main()
