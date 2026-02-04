"""Trade tracking utilities for NDS bot."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.trading_bot.contracts import ExecutionEvent, PositionContract, TradeIdentity

logger = logging.getLogger(__name__)


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
        self.last_reconcile_at: Optional[datetime] = None

    @property
    def active_trades_view(self) -> Dict[int, Dict]:
        return self.active_trades

    def normalize_trade_record(self, record: Optional[Dict]) -> Dict:
        """Normalize trade record schema (defensive against None/missing fields)."""
        if not isinstance(record, dict):
            return {}

        record.setdefault("trade_identity", {})
        record.setdefault("open_event", {})
        record.setdefault("last_update_event", record.get("open_event") or {})
        record.setdefault("status", record.get("status") or "OPEN")

        close_event = record.get("close_event")
        if close_event is None or not isinstance(close_event, dict):
            if close_event is not None:
                logger.warning("⚠️ Normalizing non-dict close_event to empty dict.")
            record["close_event"] = {}

        return record

    def normalize_trade_records(self, records: Optional[List[Dict]]) -> List[Dict]:
        """Normalize list of trade records loaded from disk/DB."""
        if not records:
            return []
        return [self.normalize_trade_record(record) for record in records if isinstance(record, dict)]

    def add_trade_open(self, event: ExecutionEvent) -> None:
        """ثبت معامله جدید با رویداد OPEN"""
        identity = self._build_trade_identity(event)
        if identity.get("detected_by") != "recovery_scan":
            if identity.get("magic") is None or identity.get("comment") in (None, ""):
                logger.warning(
                    "[TRADE][META_MISSING] order=%s position=%s symbol=%s magic=%s comment=%s",
                    identity.get("order_ticket"),
                    identity.get("position_ticket"),
                    identity.get("symbol"),
                    identity.get("magic"),
                    identity.get("comment"),
                )

        record = {
            "trade_identity": identity,
            "open_event": event,
            "last_update_event": event,
            "close_event": {},
            "status": "OPEN",
        }

        if identity["position_ticket"]:
            self.active_trades[int(identity["position_ticket"])] = record
        elif identity["order_ticket"]:
            self.pending_trades_by_order[int(identity["order_ticket"])] = record

        if identity.get("detected_by") != "recovery_scan":
            self.daily_stats['total_trades'] += 1

    def _build_trade_identity(self, event: ExecutionEvent) -> TradeIdentity:
        metadata = event.get("metadata", {}) or {}
        order_ticket = event.get("order_ticket") or metadata.get("order_ticket") or metadata.get("deal_ticket")
        position_ticket = event.get("position_ticket") or metadata.get("position_ticket")
        comment = metadata.get("request_comment") or metadata.get("comment")

        return TradeIdentity(
            order_ticket=order_ticket,
            position_ticket=position_ticket,
            symbol=event.get("symbol") or "",
            magic=metadata.get("magic"),
            comment=comment,
            opened_at=event.get("event_time") or datetime.utcnow(),
            detected_by=metadata.get("detected_by", "order_send"),
        )

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
        record = self.normalize_trade_record(record)
        if not record:
            return False
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

    def get_pending_close_tickets_for_symbol(self, symbol: Optional[str]) -> List[int]:
        """Return pending close tickets for a specific symbol."""
        if not symbol:
            return []
        matches: List[int] = []
        for position_ticket, payload in self.pending_closes.items():
            record = payload.get("record", {})
            identity = record.get("trade_identity", {}) if isinstance(record, dict) else {}
            if identity.get("symbol") == symbol:
                matches.append(int(position_ticket))
        return matches

    def mark_pending_attempt(self, position_ticket: int, attempt_time: datetime) -> None:
        """ثبت تلاش برای تایید بسته شدن."""
        if position_ticket not in self.pending_closes:
            return
        payload = self.pending_closes[position_ticket]
        payload["last_attempt"] = attempt_time
        payload["retries"] = int(payload.get("retries") or 0) + 1

    def register_partial_close(
        self,
        *,
        position_ticket: int,
        volume_closed: float,
        remaining_volume: float,
        reason: str,
    ) -> None:
        """ثبت رخداد partial close برای معامله فعال."""
        record = self.active_trades.get(position_ticket)
        if not record:
            record = self.pending_closes.get(position_ticket, {}).get("record")
        if not record:
            return

        record = self.normalize_trade_record(record)
        metadata = record.get("open_event", {}).get("metadata", {})
        partials = metadata.get("partial_closes", [])
        partials.append(
            {
                "time": datetime.utcnow().isoformat(),
                "volume_closed": float(volume_closed),
                "remaining_volume": float(remaining_volume),
                "reason": reason,
            }
        )
        metadata["partial_closes"] = partials
        metadata["partial_close_count"] = len(partials)
        metadata["remaining_volume_after_tp1"] = float(remaining_volume)
        record.get("open_event", {})["metadata"] = metadata

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
        self, open_positions: List[PositionContract], reconcile_time: Optional[datetime] = None
    ) -> Tuple[int, int, List[Dict]]:
        """همگام‌سازی وضعیت معاملات با پوزیشن‌های باز MT5."""
        self.last_reconcile_at = reconcile_time or datetime.utcnow()
        added_count = 0
        updated_count = 0

        open_map = {pos["position_ticket"]: pos for pos in open_positions}
        unmatched_positions = set(open_map.keys())

        def _comments_match(expected: Optional[str], actual: Optional[str]) -> bool:
            if not expected or not actual:
                return True
            return expected.strip() == actual.strip()

        def _score_candidate(position: PositionContract, record: Dict) -> Optional[Tuple[int, Dict[str, bool]]]:
            identity: TradeIdentity = record["trade_identity"]
            if position["symbol"] != identity.get("symbol"):
                return None
            side = record.get("open_event", {}).get("side")
            if side and position["side"] != side:
                return None
            identity_magic = identity.get("magic")
            if identity_magic is not None and position["magic"] != identity_magic:
                return None
            comment_match = _comments_match(identity.get("comment"), position.get("comment"))
            if not comment_match:
                return None

            opened_at = identity.get("opened_at", datetime.min)
            if position["open_time"] < opened_at - timedelta(minutes=5):
                return None

            score = 0
            match_fields = {
                "magic": identity_magic is not None and position["magic"] == identity_magic,
                "comment": comment_match,
                "side": bool(side and position["side"] == side),
                "symbol": position["symbol"] == identity.get("symbol"),
                "time": False,
                "price": False,
                "volume": False,
            }
            if identity_magic is not None:
                score += 2
            if identity.get("comment") and position.get("comment"):
                score += 2

            time_delta = abs((position["open_time"] - opened_at).total_seconds())
            if time_delta <= 900:
                score += 1
                match_fields["time"] = True

            entry_price = record.get("open_event", {}).get("entry_price")
            if entry_price and position.get("entry_price"):
                if abs(float(position["entry_price"]) - float(entry_price)) <= 0.5:
                    score += 1
                    match_fields["price"] = True

            volume = record.get("open_event", {}).get("volume")
            if volume and position.get("volume"):
                if abs(float(position["volume"]) - float(volume)) <= 1e-6:
                    score += 1
                    match_fields["volume"] = True

            return score, match_fields

        if self.pending_trades_by_order and unmatched_positions:
            pending_items = list(self.pending_trades_by_order.items())
            for pos_ticket in list(unmatched_positions):
                position = open_map[pos_ticket]
                candidates: List[Tuple[int, int, Dict, Dict[str, bool]]] = []
                for order_ticket, record in pending_items:
                    if order_ticket not in self.pending_trades_by_order:
                        continue
                    scored = _score_candidate(position, record)
                    if scored is None:
                        continue
                    score, match_fields = scored
                    candidates.append((score, order_ticket, record, match_fields))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0], reverse=True)
                best_score, order_ticket, record, match_fields = candidates[0]
                if len(candidates) > 1 and candidates[1][0] == best_score:
                    logger.warning(
                        "[TRADE][RECONCILE_SKIP] position=%s reason=score_tie best_score=%s",
                        pos_ticket,
                        best_score,
                    )
                    continue
                if best_score < 2:
                    logger.debug(
                        "[TRADE][RECONCILE_SKIP] position=%s reason=score_low score=%s",
                        pos_ticket,
                        best_score,
                    )
                    continue
                identity: TradeIdentity = record["trade_identity"]
                record["trade_identity"]["position_ticket"] = pos_ticket
                self.active_trades[pos_ticket] = record
                del self.pending_trades_by_order[order_ticket]
                unmatched_positions.discard(pos_ticket)
                updated_count += 1
                logger.info(
                    "[TRADE][PENDING_TO_OPEN] order=%s position=%s symbol=%s side=%s magic=%s comment=%s score=%s match=%s",
                    order_ticket,
                    pos_ticket,
                    identity.get("symbol"),
                    record.get("open_event", {}).get("side"),
                    identity.get("magic"),
                    identity.get("comment"),
                    best_score,
                    match_fields,
                )

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

        if self.pending_trades_by_order and unmatched_positions:
            for order_ticket, record in list(self.pending_trades_by_order.items()):
                identity = record.get("trade_identity", {})
                logger.warning(
                    "[TRADE][PENDING_UNMATCHED] order=%s symbol=%s side=%s magic=%s comment=%s pending=%s",
                    order_ticket,
                    identity.get("symbol"),
                    record.get("open_event", {}).get("side"),
                    identity.get("magic"),
                    identity.get("comment"),
                    len(self.pending_trades_by_order),
                )

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
