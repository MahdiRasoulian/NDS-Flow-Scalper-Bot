from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _build_payload():
    return {
        "signal": "BUY",
        "confidence": 78.0,
        "entry_level": 5097.718,
        "entry_model": "STOP",
        "entry_idea": {
            "entry_level": 5097.718,
            "entry_model": "STOP",
        },
        "market_metrics": {
            "atr": 6.5,
            "adx": 22.0,
            "volatility_state": "MODERATE",
        },
        "context": {
            "entry_idea": {
                "entry_level": 5097.718,
                "entry_model": "STOP",
            },
        },
    }


def _build_config(rr_repair_enabled: bool):
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.9,
            "RR_REPAIR_ENABLED": rr_repair_enabled,
            "RR_REPAIR_MAX_TP_PIPS": 120.0,
            "RR_REPAIR_MAX_TP_ATR_MULT": 2.0,
            "STOP_MAX_DEVIATION_PIPS": 70.0,
            "STOP_HARD_REJECT_PIPS": 120.0,
        }
    )
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    return cfg


def _run_case(label: str, rr_repair_enabled: bool) -> None:
    risk_manager = create_scalping_risk_manager()
    finalized = risk_manager.finalize_order(
        analysis=_build_payload(),
        live=LivePriceSnapshot(bid=5096.14, ask=5096.42),
        symbol="XAUUSD",
        config=_build_config(rr_repair_enabled),
    )

    print(f"\n=== {label} (RR_REPAIR_ENABLED={rr_repair_enabled}) ===")
    print("allowed:", finalized.is_trade_allowed)
    print("order_type:", finalized.order_type)
    print("rr:", finalized.rr_ratio)
    print("reject_reason:", finalized.reject_reason)
    print("decision_notes:")
    for note in finalized.decision_notes:
        print(" -", note)


def main() -> None:
    _run_case("Before (RR repair disabled)", False)
    _run_case("After (RR repair enabled)", True)


if __name__ == "__main__":
    main()
