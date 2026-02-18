from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.reporting.report_generator import ReportGenerator, TradeMetrics


def _sample_signal() -> dict:
    return {
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "signal": "BUY",
        "confidence": 75,
        "score": 68,
        "structure": {"trend": "BULLISH", "bos": "UP", "choch": "NONE"},
    }


def test_generate_trade_summary_creates_file(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))

    report_path = generator.generate_trade_summary(_sample_signal(), df=None)

    assert report_path
    path = Path(report_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "TRADE SUMMARY" in content
    assert "Symbol: XAUUSD" in content


def test_generate_full_report_uses_single_method_chain(tmp_path, monkeypatch):
    generator = ReportGenerator(output_dir=str(tmp_path))
    calls = []

    def _excel(df, signal_data, order_details=None, trades_history=None, filename=None):
        calls.append("excel")
        return str(tmp_path / "excel" / "report.xlsx")

    def _chart(df, signal_data, order_details=None, filename=None):
        calls.append("chart")
        return str(tmp_path / "charts" / "report.png")

    def _summary(signal_data, order_details=None, filename=None, df=None):
        calls.append("summary")
        return str(tmp_path / "summaries" / "report.txt")

    monkeypatch.setattr(generator, "save_excel_report", _excel)
    monkeypatch.setattr(generator, "plot_chart", _chart)
    monkeypatch.setattr(generator, "generate_trade_summary", _summary)

    result = generator.generate_full_report(df=None, signal_data=_sample_signal(), base_filename="my_report")

    assert calls == ["excel", "chart", "summary"]
    assert result["success"] is True
    combined = Path(result["file_paths"]["combined"])
    assert combined.exists()
    payload = json.loads(combined.read_text(encoding="utf-8"))
    assert payload["symbol"] == "XAUUSD"


def test_update_daily_summary_accepts_iso_string_event_time(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    generator._calculate_metrics = lambda _trades: TradeMetrics(total_trades=1)
    event_time = "2026-02-18T08:15:00"
    event = {
        "event_type": "CLOSE",
        "event_time": event_time,
        "symbol": "XAUUSD",
        "position_ticket": 12345,
        "side": "BUY",
        "profit": 42.5,
        "pips": 10.0,
        "pips_abs": 10.0,
        "entry_price": 2000.0,
        "exit_price": 2001.0,
        "reason": "TP",
        "metadata": {},
    }

    path = Path(generator.update_daily_summary(event))

    assert path.exists()
    assert path.name == "daily_2026-02-18.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trades"][0]["close_time"] == datetime.fromisoformat(event_time).isoformat()
    assert payload["metrics"]["total_trades"] == 1
