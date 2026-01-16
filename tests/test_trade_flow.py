from config.settings import config
from src.trading_bot.bot import NDSBot
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


class DummyMT5:
    connected = False

    def __init__(self, *args, **kwargs):
        pass


def _build_analysis_payload(signal: str, entry_level: float) -> dict:
    return {
        "signal": signal,
        "confidence": 80.0,
        "entry_level": entry_level,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": entry_level,
            "entry_model": "MARKET",
            "entry_type": "FLOW",
        },
        "market_metrics": {
            "atr": 5.0,
        },
        "context": {
            "entry_idea": {
                "entry_level": entry_level,
                "entry_model": "MARKET",
            },
        },
    }


def _build_config_payload() -> dict:
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"]["MIN_RR_RATIO"] = 0.1
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    return cfg


def test_entry_idea_flows_without_sl_tp_and_finalizes():
    bot = NDSBot(DummyMT5)
    analysis_payload = _build_analysis_payload("BUY", 2000.0)

    ok, reason, entry_level, entry_model = bot._validate_entry_idea(analysis_payload)
    assert ok, reason
    assert entry_level == 2000.0
    assert entry_model == "MARKET"

    risk_manager = create_scalping_risk_manager()
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp="2026-01-15T01:00:00")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=_build_config_payload(),
    )

    assert finalized.is_trade_allowed
    assert finalized.stop_loss is not None
    assert finalized.take_profit is not None


def test_geometry_validation_after_finalize_buy_sell():
    risk_manager = create_scalping_risk_manager()
    cfg = _build_config_payload()

    buy_payload = _build_analysis_payload("BUY", 2100.0)
    buy_final = risk_manager.finalize_order(
        analysis=buy_payload,
        live=LivePriceSnapshot(bid=2100.0, ask=2100.01, timestamp="2026-01-15T01:05:00"),
        symbol="XAUUSD",
        config=cfg,
    )
    assert buy_final.is_trade_allowed
    assert buy_final.stop_loss < buy_final.entry_price < buy_final.take_profit

    sell_payload = _build_analysis_payload("SELL", 2100.0)
    sell_final = risk_manager.finalize_order(
        analysis=sell_payload,
        live=LivePriceSnapshot(bid=2100.0, ask=2100.01, timestamp="2026-01-15T01:10:00"),
        symbol="XAUUSD",
        config=cfg,
    )
    assert sell_final.is_trade_allowed
    assert sell_final.take_profit < sell_final.entry_price < sell_final.stop_loss
