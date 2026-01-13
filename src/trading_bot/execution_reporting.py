"""Execution reporting utilities for execution events."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.trading_bot.contracts import ExecutionEvent


def _serialize_event(event: ExecutionEvent) -> Dict[str, Any]:
    def _serialize(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_serialize(v) for v in value]
        return value

    return _serialize(event)


def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _update_summary(path: Path, event: ExecutionEvent) -> None:
    summary = {}
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    summary["last_event"] = _serialize_event(event)
    summary["status"] = "CLOSED" if event["event_type"] == "CLOSE" else summary.get("status", "OPEN")

    if event["event_type"] == "OPEN":
        summary["open_event"] = _serialize_event(event)
        summary["status"] = "OPEN"
    if event["event_type"] == "CLOSE":
        summary["close_event"] = _serialize_event(event)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_trade_report(event: ExecutionEvent, df=None) -> Optional[Dict[str, Any]]:
    if event["event_type"] != "OPEN":
        return None

    metadata = event.get("metadata", {})
    analysis_snapshot = metadata.get("analysis_snapshot", {})
    market_metrics = metadata.get("market_metrics", {})
    if not analysis_snapshot:
        return None

    return {
        'order_id': event.get("order_ticket"),
        'symbol': event.get("symbol"),
        'signal': event.get("side"),
        'order_type': metadata.get("order_type"),
        'entry_price_planned': analysis_snapshot.get('entry_price'),
        'entry_price_actual': event.get("entry_price"),
        'stop_loss_planned': analysis_snapshot.get('stop_loss'),
        'stop_loss_actual': event.get("sl"),
        'take_profit_planned': analysis_snapshot.get('take_profit'),
        'take_profit_actual': event.get("tp"),
        'lot_size': event.get("volume"),
        'confidence': metadata.get("confidence", 0),
        'execution_time': event.get("event_time").strftime('%Y-%m-%d %H:%M:%S'),
        'scalping_params': {},
        'risk_params': {
            'risk_amount': metadata.get("risk_amount"),
            'risk_percent': None,
            'actual_risk_percent': None,
            'sl_distance': abs((event.get("entry_price") or 0) - (event.get("sl") or 0)),
            'scalping_grade': metadata.get("scalping_grade"),
            'max_holding_minutes': 60,
            'session': metadata.get("session") or 'N/A',
            'session_multiplier': None,
        },
        'timeframe': metadata.get("timeframe"),
        'signal_quality': analysis_snapshot.get('quality', 'MEDIUM'),
        'scalping_mode': True,
        'market_metrics': market_metrics,
        'real_time_data': {
            'bid_at_analysis': analysis_snapshot.get('execution_bid'),
            'ask_at_analysis': analysis_snapshot.get('execution_ask'),
            'price_deviation_pips': metadata.get("price_deviation_pips"),
            'execution_source': analysis_snapshot.get('source', 'unknown')
        }
    }


def generate_execution_report(
    logger,
    event: Optional[ExecutionEvent] = None,
    df=None,
    **legacy_kwargs: Any,
) -> None:
    """Generate JSON execution report(s) for OPEN/UPDATE/CLOSE events."""
    try:
        if event is None and legacy_kwargs:
            event = legacy_kwargs.get("event")

        if event is None:
            logger.warning("⚠️ Execution report skipped: no event provided")
            return

        event_time = event.get("event_time") or datetime.now()
        event_date = event_time.strftime("%Y-%m-%d")
        ticket = event.get("position_ticket") or event.get("order_ticket") or "unknown"
        trade_dir = Path("reports") / event_date / "trades" / str(ticket)
        events_file = trade_dir / "events.jsonl"
        summary_file = trade_dir / "summary.json"

        payload = _serialize_event(event)
        _write_jsonl(events_file, payload)
        _update_summary(summary_file, event)

        legacy_report = _build_trade_report(event, df=df)
        if legacy_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            execution_file = f"trade_reports/scalping_executions/{event.get('symbol')}_scalping_{timestamp}.json"
            Path("trade_reports/scalping_executions").mkdir(parents=True, exist_ok=True)
            with open(execution_file, 'w', encoding='utf-8') as f:
                json.dump(legacy_report, f, indent=2, ensure_ascii=False)
            logger.info(f"📝 گزارش خام معامله اسکلپینگ Real-Time در {execution_file} ذخیره شد")

            try:
                from src.reporting.report_generator import ReportGenerator
                if df is not None:
                    report_gen = ReportGenerator(output_dir="trade_reports/scalping_reports")
                    report_result = report_gen.generate_full_report(
                        df=df,
                        signal_data=event.get("metadata", {}).get("analysis_snapshot", {}),
                        order_details={
                            "signal": event.get("side"),
                            "side": event.get("side"),
                            "confidence": event.get("metadata", {}).get("confidence", 0),
                            "entry_planned": legacy_report.get("entry_price_planned"),
                            "entry_actual": legacy_report.get("entry_price_actual"),
                            "sl_planned": legacy_report.get("stop_loss_planned"),
                            "sl_actual": legacy_report.get("stop_loss_actual"),
                            "tp_planned": legacy_report.get("take_profit_planned"),
                            "tp_actual": legacy_report.get("take_profit_actual"),
                            "rr_ratio": event.get("metadata", {}).get("rr_ratio"),
                            "symbol": event.get("symbol"),
                            "timeframe": event.get("metadata", {}).get("timeframe"),
                            "execution_time": legacy_report.get("execution_time"),
                            "lot": event.get("volume"),
                            "scalping_grade": event.get("metadata", {}).get("scalping_grade"),
                            "scalping_mode": True,
                            "session": event.get("metadata", {}).get("session"),
                            "price_deviation_pips": event.get("metadata", {}).get("price_deviation_pips"),
                            "order_type": event.get("metadata", {}).get("order_type"),
                        },
                    )

                    if report_result.get("success"):
                        logger.info("📊 گزارش کامل معامله اسکلپینگ Real-Time ذخیره شد")
            except ImportError:
                logger.debug("ماژول گزارش‌گیری یافت نشد، فقط JSON ذخیره شد.")

    except Exception as e:
        logger.error(f"⚠️ خطا در فرآیند گزارش‌گیری اسکلپینگ: {e}")
