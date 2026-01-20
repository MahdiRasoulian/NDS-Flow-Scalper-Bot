"""Trade tracking utilities for NDS bot."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.trading_bot.contracts import ExecutionEvent, PositionContract, TradeIdentity


class TradeTracker:
    """ردیاب کامل معاملات از باز شدن تا بسته شدن"""

    def __init__(self):
        self.active_trades: Dict[int, Dict] = {}
        self.pending_trades_by_order: Dict[int, Dict] = {}
        self.pending_closes: Dict[int, Dict] = {}
        self.closed_trades: List[Dict] = []
        self.max_daily_profit = 0.0
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0
        }

    @property
    def active_trades_view(self) -> Dict[int, Dict]:
        return self.active_trades

    def add_trade_open(self, event: ExecutionEvent) -> None:
        """ثبت معامله جدید با رویداد OPEN"""
        identity = TradeIdentity(
            order_ticket=event.get("order_ticket"),
            position_ticket=event.get("position_ticket"),
            symbol=event.get("symbol") or "",
            magic=event.get("metadata", {}).get("magic"),
            comment=event.get("metadata", {}).get("comment"),
            opened_at=event.get("event_time") or datetime.now(),
            detected_by=event.get("metadata", {}).get("detected_by", "order_send"),
        )

        record = {
            "trade_identity": identity,
            "open_event": event,
            "last_update_event": event,
            "close_event": None,
            "status": "OPEN",
        }

        if identity["position_ticket"]:
            self.active_trades[int(identity["position_ticket"])] = record
        elif identity["order_ticket"]:
            self.pending_trades_by_order[int(identity["order_ticket"])] = record

        if identity.get("detected_by") != "recovery_scan":
            self.daily_stats['total_trades'] += 1

    def update_trade_event(self, event: ExecutionEvent) -> None:
        """به‌روزرسانی رویدادهای OPEN/UPDATE"""
        position_ticket = event.get("position_ticket")
        order_ticket = event.get("order_ticket")

        if position_ticket and position_ticket in self.active_trades:
            self.active_trades[position_ticket]["last_update_event"] = event
            return

        if order_ticket and order_ticket in self.pending_trades_by_order:
            record = self.pending_trades_by_order[order_ticket]
            if position_ticket:
                record["trade_identity"]["position_ticket"] = position_ticket
                self.active_trades[position_ticket] = record
                del self.pending_trades_by_order[order_ticket]
                self.active_trades[position_ticket]["last_update_event"] = event
            else:
                record["last_update_event"] = event

    def close_trade_event(self, event: ExecutionEvent) -> None:
        """ثبت بسته شدن معامله"""
        position_ticket = event.get("position_ticket")
        trade = None
        if position_ticket in self.active_trades:
            trade = self.active_trades[position_ticket]
        elif position_ticket in self.pending_closes:
            trade = self.pending_closes[position_ticket].get("record")
        if trade is None:
            return

        trade["close_event"] = event
        trade["status"] = "CLOSED"
        self.closed_trades.append(trade)

        final_profit = float(event.get("profit") or 0.0)
        self.daily_stats['total_profit'] += final_profit
        if final_profit > 0:
            self.daily_stats['winning_trades'] += 1
        if final_profit > self.max_daily_profit:
            self.max_daily_profit = final_profit

        if position_ticket in self.active_trades:
            del self.active_trades[position_ticket]
        if position_ticket in self.pending_closes:
            del self.pending_closes[position_ticket]

    def mark_trade_unknown(self, position_ticket: int, reason: str) -> None:
        """علامت‌گذاری معامله برای بررسی مجدد در سیکل بعدی."""
        if position_ticket in self.active_trades:
            self.active_trades[position_ticket]["status"] = "UNKNOWN"
            self.active_trades[position_ticket]["unknown_reason"] = reason

    def register_pending_close(self, position_ticket: int, record: Dict, detected_time: datetime) -> bool:
        """ثبت معامله بسته‌شده در صف pending برای تایید تاریخچه."""
        if position_ticket in self.pending_closes:
            return False

        if position_ticket in self.active_trades:
            self.active_trades[position_ticket]["status"] = "PENDING_CLOSE"

        self.pending_closes[position_ticket] = {
            "record": record,
            "first_seen": detected_time,
            "last_attempt": None,
            "retries": 0,
        }

        if position_ticket in self.active_trades:
            del self.active_trades[position_ticket]

        return True

    def get_pending_close_candidates(
        self,
        now: datetime,
        base_backoff_sec: float,
        max_backoff_sec: float,
        timeout_sec: float,
    ) -> Tuple[List[Tuple[int, Dict]], List[Tuple[int, Dict]]]:
        """دریافت لیست pending برای بررسی یا timeout."""
        ready: List[Tuple[int, Dict]] = []
        timed_out: List[Tuple[int, Dict]] = []

        for position_ticket, payload in list(self.pending_closes.items()):
            first_seen = payload.get("first_seen") or now
            last_attempt = payload.get("last_attempt")
            retries = int(payload.get("retries") or 0)
            elapsed = (now - first_seen).total_seconds()
            if elapsed >= timeout_sec:
                timed_out.append((position_ticket, payload))
                continue

            backoff = min(base_backoff_sec * (2 ** retries), max_backoff_sec)
            if last_attempt is None or (now - last_attempt).total_seconds() >= backoff:
                ready.append((position_ticket, payload))

        return ready, timed_out

    def mark_pending_attempt(self, position_ticket: int, attempt_time: datetime) -> None:
        """ثبت تلاش برای تایید بسته شدن."""
        if position_ticket not in self.pending_closes:
            return
        payload = self.pending_closes[position_ticket]
        payload["last_attempt"] = attempt_time
        payload["retries"] = int(payload.get("retries") or 0) + 1

    def finalize_unknown_close(self, position_ticket: int, event: ExecutionEvent) -> None:
        """ثبت وضعیت CLOSE_UNKNOWN و خارج کردن از pending."""
        record = None
        if position_ticket in self.pending_closes:
            record = self.pending_closes[position_ticket].get("record")
        if record is None and position_ticket in self.active_trades:
            record = self.active_trades[position_ticket]
        if record is None:
            return

        record["close_event"] = event
        record["status"] = "CLOSE_UNKNOWN"
        self.closed_trades.append(record)

        if position_ticket in self.pending_closes:
            del self.pending_closes[position_ticket]
        if position_ticket in self.active_trades:
            del self.active_trades[position_ticket]

    def reconcile_with_open_positions(
        self, open_positions: List[PositionContract]
    ) -> Tuple[int, int, List[Dict]]:
        """همگام‌سازی وضعیت معاملات با پوزیشن‌های باز MT5."""
        added_count = 0
        updated_count = 0

        open_map = {pos["position_ticket"]: pos for pos in open_positions}
        unmatched_positions = set(open_map.keys())

        # Resolve pending trades by matching metadata
        for order_ticket, record in list(self.pending_trades_by_order.items()):
            identity: TradeIdentity = record["trade_identity"]
            for pos_ticket, position in open_map.items():
                if pos_ticket not in unmatched_positions:
                    continue
                if position["symbol"] != identity["symbol"]:
                    continue
                if identity.get("magic") and position["magic"] != identity["magic"]:
                    continue
                if identity.get("comment") and position["comment"] != identity["comment"]:
                    continue
                opened_at = identity.get("opened_at", datetime.min)
                if position["open_time"] < opened_at - timedelta(minutes=5):
                    continue
                record["trade_identity"]["position_ticket"] = pos_ticket
                self.active_trades[pos_ticket] = record
                del self.pending_trades_by_order[order_ticket]
                unmatched_positions.discard(pos_ticket)
                updated_count += 1
                break

        # Update active or add recovered positions
        for pos_ticket, position in open_map.items():
            if pos_ticket in self.active_trades:
                update_event: ExecutionEvent = {
                    "event_type": "UPDATE",
                    "event_time": position["update_time"] or datetime.now(),
                    "symbol": position["symbol"],
                    "order_ticket": None,
                    "position_ticket": pos_ticket,
                    "side": position["side"],
                    "volume": position["volume"],
                    "entry_price": position["entry_price"],
                    "exit_price": None,
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "profit": position["profit"],
                    "pips": None,
                    "reason": None,
                    "metadata": {"current_price": position["current_price"]},
                }
                self.update_trade_event(update_event)
                updated_count += 1
            else:
                open_event: ExecutionEvent = {
                    "event_type": "OPEN",
                    "event_time": position["open_time"],
                    "symbol": position["symbol"],
                    "order_ticket": None,
                    "position_ticket": pos_ticket,
                    "side": position["side"],
                    "volume": position["volume"],
                    "entry_price": position["entry_price"],
                    "exit_price": None,
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "profit": position["profit"],
                    "pips": None,
                    "reason": None,
                    "metadata": {"detected_by": "recovery_scan", "current_price": position["current_price"]},
                }
                self.add_trade_open(open_event)
                added_count += 1

        closed_candidates = []
        for pos_ticket, record in self.active_trades.items():
            if pos_ticket not in open_map:
                closed_candidates.append(record)

        return added_count, updated_count, closed_candidates

    def get_active_trades_count(self) -> int:
        """تعداد معاملات فعال"""
        return len(self.active_trades)

    def get_daily_stats(self) -> dict:
        """آمار روزانه"""
        win_rate = 0
        if self.daily_stats['total_trades'] > 0:
            win_rate = (self.daily_stats['winning_trades'] / self.daily_stats['total_trades']) * 100

        return {
            **self.daily_stats,
            'win_rate': win_rate,
            'max_daily_profit': self.max_daily_profit,
            'active_trades': self.get_active_trades_count(),
            'closed_trades': len(self.closed_trades)
        }
