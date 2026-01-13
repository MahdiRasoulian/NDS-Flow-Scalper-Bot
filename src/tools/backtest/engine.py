"""
NDS Backtest Engine (XAUUSD scalping)
------------------------------------
Design goals:
- Works with your existing NDS analyzer (analyze_gold_market) and ScalpingRiskManager.finalize_order
- Candle-by-candle simulation (walk-forward) with reproducible records for analysis
- Parameter override friendly (grid search / random search)
- Produces a detailed trades ledger + equity curve + per-cycle diagnostics

Expected OHLCV input (CSV):
time, open, high, low, close, tick_volume (or volume)
Time must be parseable by pandas.to_datetime.

Notes:
- Bid/Ask are synthesized using spread (in price units, e.g., 0.25$). You may pass a fixed spread
  or a spread series (e.g., from MT5 ticks aggregated).
- SL/TP are evaluated intra-bar using high/low:
    BUY: SL hits if low <= SL; TP hits if high >= TP
    SELL: SL hits if high >= SL; TP hits if low <= TP
  If both hit in same bar, we apply a conservative rule: SL first (worst-case).

Anti-lookahead / Anti-leakage:
- Walk-forward candle-by-candle.
- For each bar i, analyzer only receives window up to i (inclusive).
- No future bars are exposed to analyzer or risk sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

import pandas as pd
import numpy as np


# -----------------------------
# Import your project modules
# -----------------------------
def _import_analyzer():
    # Prefer your package layout
    try:
        from src.trading_bot.nds.analyzer import analyze_gold_market
        return analyze_gold_market
    except Exception:
        pass

    # Fallback: if running standalone next to analyzer.py
    try:
        from analyzer import analyze_gold_market  # type: ignore
        return analyze_gold_market
    except Exception as e:
        raise ImportError(
            "Could not import analyze_gold_market. Place this backtest tool inside your repo "
            "or ensure analyzer.py is importable."
        ) from e


def _import_risk_manager():
    try:
        from src.trading_bot.risk_manager import ScalpingRiskManager
        return ScalpingRiskManager
    except Exception:
        pass

    try:
        from risk_manager import ScalpingRiskManager  # type: ignore
        return ScalpingRiskManager
    except Exception as e:
        raise ImportError(
            "Could not import ScalpingRiskManager. Place this backtest tool inside your repo "
            "or ensure risk_manager.py is importable."
        ) from e


analyze_gold_market = _import_analyzer()
ScalpingRiskManager = _import_risk_manager()


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Position:
    id: str
    side: str  # BUY / SELL
    symbol: str
    open_time: pd.Timestamp
    open_bar_index: int
    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float
    order_type: str  # MARKET/LIMIT (simulated)
    confidence: float
    score: float
    rr: float
    session: str
    planned_entry: float
    deviation_pips: float
    notes: List[str]

    close_time: Optional[pd.Timestamp] = None
    close_bar_index: Optional[int] = None
    close_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP/SL/TIMEOUT/EOD/MANUAL
    pnl_usd: Optional[float] = None
    duration_bars: Optional[int] = None


@dataclass
class PendingOrder:
    id: str
    side: str  # BUY / SELL
    symbol: str
    created_time: pd.Timestamp
    created_bar_index: int
    planned_entry: float
    stop_loss: float
    take_profit: float
    lot: float
    order_type: str  # LIMIT
    confidence: float
    score: float
    rr: float
    session: str
    deviation_pips: float
    notes: List[str]

    # filled info
    filled_time: Optional[pd.Timestamp] = None
    filled_bar_index: Optional[int] = None
    filled_price: Optional[float] = None

    def to_position(self) -> Position:
        return Position(
            id=self.id,
            side=self.side,
            symbol=self.symbol,
            open_time=self.filled_time or self.created_time,
            open_bar_index=self.filled_bar_index if self.filled_bar_index is not None else self.created_bar_index,
            entry_price=float(self.filled_price if self.filled_price is not None else self.planned_entry),
            stop_loss=float(self.stop_loss),
            take_profit=float(self.take_profit),
            lot=float(self.lot),
            order_type=str(self.order_type),
            confidence=float(self.confidence),
            score=float(self.score),
            rr=float(self.rr),
            session=str(self.session),
            planned_entry=float(self.planned_entry),
            deviation_pips=float(self.deviation_pips),
            notes=list(self.notes or []),
        )


@dataclass
class BacktestConfig:
    symbol: str = "XAUUSD!"
    timeframe: str = "M15"
    bars_to_fetch: int = 3500

    # simulation
    warmup_bars: int = 300  # minimum bars before allowing trades
    spread: float = 0.25    # price units ($)
    slippage: float = 0.10  # price units ($) - applied on entry/exit
    commission_per_lot: float = 0.0  # USD per lot, round-turn handled as two sides

    # governance
    allow_multiple_positions: bool = True
    max_positions: int = 5
    min_candles_between_trades: int = 10
    min_time_between_trades_minutes: int = 150
    daily_max_trades: int = 40

    # pending order simulation
    enable_limit_orders: bool = True
    pending_expire_minutes: int = 60  # cancel pending if not filled within this time

    # risk
    starting_equity: float = 1000.0
    max_daily_risk_percent: float = 6.0

    # meta
    random_seed: int = 7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    config: Dict[str, Any]
    overrides: Dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    cycle_log: pd.DataFrame
    metrics: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _get_cfg(cfg: Dict[str, Any], dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _normalize_signal(sig: Any) -> str:
    s = str(sig or "NONE").upper().strip()
    if s in ("BUY", "SELL"):
        return s
    return "NONE"


def _pnl_usd_gold(side: str, entry: float, exit_: float, lot: float, contract_size: float = 100.0) -> float:
    """
    For spot gold CFD/FX, common approximation:
      PnL = (exit - entry) * contract_size * lot   for BUY
      PnL = (entry - exit) * contract_size * lot   for SELL
    Contract size in your config is 100.
    """
    if side == "BUY":
        return (exit_ - entry) * contract_size * lot
    return (entry - exit_) * contract_size * lot


def _apply_slippage(price: float, side: str, is_entry: bool, slippage: float) -> float:
    """
    Conservative slippage:
      Entry: BUY worse -> +slip ; SELL worse -> -slip
      Exit : BUY worse -> -slip (sell) ; SELL worse -> +slip (buy to cover)
    """
    if slippage <= 0:
        return price
    if is_entry:
        return price + slippage if side == "BUY" else price - slippage
    return price - slippage if side == "BUY" else price + slippage


def _bar_hit(side: str, o: float, h: float, l: float, c: float, sl: float, tp: float) -> Tuple[Optional[str], Optional[float]]:
    """
    Determine if TP/SL is hit inside a bar. Worst-case ordering when both hit in same bar.
    """
    if side == "BUY":
        sl_hit = l <= sl
        tp_hit = h >= tp
        if sl_hit and tp_hit:
            return "SL", sl  # worst-case
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
        return None, None
    else:
        sl_hit = h >= sl
        tp_hit = l <= tp
        if sl_hit and tp_hit:
            return "SL", sl
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
        return None, None


def _limit_fill(side: str, planned_entry: float, o: float, h: float, l: float) -> bool:
    """
    Simple limit fill model:
      BUY LIMIT fills if low <= entry
      SELL LIMIT fills if high >= entry
    """
    if side == "BUY":
        return l <= planned_entry
    return h >= planned_entry


# -----------------------------
# Main engine
# -----------------------------
class NDSBacktester:
    def __init__(
        self,
        bot_config: Dict[str, Any],
        bt_cfg: Optional[BacktestConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.base_bot_config = dict(bot_config or {})
        self.overrides = overrides or {}
        self.bot_config = deep_update(self.base_bot_config, self.overrides)

        self.bt = bt_cfg or BacktestConfig(
            symbol=_get_cfg(self.bot_config, "trading_settings.SYMBOL", "XAUUSD!"),
            timeframe=_get_cfg(self.bot_config, "trading_settings.TIMEFRAME", "M15"),
            bars_to_fetch=int(_get_cfg(self.bot_config, "trading_settings.BARS_TO_FETCH", 3500) or 3500),
            starting_equity=float(self.bot_config.get("ACCOUNT_BALANCE", 1000.0) or 1000.0),
            max_positions=int(_get_cfg(self.bot_config, "trading_rules.MAX_POSITIONS", 5) or 5),
            allow_multiple_positions=bool(_get_cfg(self.bot_config, "trading_rules.ALLOW_MULTIPLE_POSITIONS", True)),
            min_candles_between_trades=int(_get_cfg(self.bot_config, "trading_rules.MIN_CANDLES_BETWEEN_TRADES", 10) or 10),
            min_time_between_trades_minutes=int(_get_cfg(self.bot_config, "trading_rules.MIN_TIME_BETWEEN_TRADES_MINUTES", 150) or 150),
            daily_max_trades=int(_get_cfg(self.bot_config, "trading_rules.DAILY_MAX_TRADES", 40) or 40),
            max_daily_risk_percent=float(_get_cfg(self.bot_config, "risk_settings.MAX_DAILY_RISK_PERCENT", 6.0) or 6.0),
            spread=float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.TYPICAL_SPREAD", 0.25) or 0.25),
            slippage=float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.TYPICAL_SLIPPAGE", 0.10) or 0.10),
        )

        self.rng = np.random.default_rng(self.bt.random_seed)

        self.risk_manager = ScalpingRiskManager(overrides=None)  # uses config.settings internally
        # but finalize_order expects a config dict; we pass self.bot_config each call

        self.positions: List[Position] = []
        self.pending: List[PendingOrder] = []

        self.last_trade_time: Optional[pd.Timestamp] = None
        self.last_trade_bar_index: Optional[int] = None

        self.equity = float(self.bt.starting_equity)
        self.balance = float(self.bt.starting_equity)

        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_risk_used = 0.0  # approximate using planned risk_amount_usd
        self._current_day: Optional[pd.Timestamp] = None

        # caches for outputs
        self._equity_rows: List[Dict[str, Any]] = []
        self._cycle_rows: List[Dict[str, Any]] = []
        self._trade_rows: List[Dict[str, Any]] = []

        # gold specs
        self.contract_size = float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.CONTRACT_SIZE", 100) or 100)

    def _reset_daily(self, day: pd.Timestamp):
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_risk_used = 0.0
        self._current_day = day

    def _can_open_new(self, now: pd.Timestamp, bar_index: int) -> Tuple[bool, str]:
        if not self.bt.allow_multiple_positions and self.positions:
            return False, "WAIT_FOR_CLOSE_BEFORE_NEW_TRADE"

        if len(self.positions) >= self.bt.max_positions:
            return False, "MAX_POSITIONS"

        if self._daily_trades >= self.bt.daily_max_trades:
            return False, "DAILY_MAX_TRADES"

        if self.last_trade_bar_index is not None:
            if (bar_index - self.last_trade_bar_index) < self.bt.min_candles_between_trades:
                return False, "MIN_CANDLES_BETWEEN_TRADES"

        if self.last_trade_time is not None:
            delta_min = (now - self.last_trade_time).total_seconds() / 60.0
            if delta_min < self.bt.min_time_between_trades_minutes:
                return False, "MIN_TIME_BETWEEN_TRADES_MINUTES"

        # daily risk cap (approx using planned risk)
        if self.bt.max_daily_risk_percent is not None and self.bt.max_daily_risk_percent > 0:
            cap = self.balance * (self.bt.max_daily_risk_percent / 100.0)
            if self._daily_risk_used >= cap:
                return False, "MAX_DAILY_RISK_PERCENT"

        return True, "OK"

    def _synthesize_live(self, close_price: float) -> Dict[str, Any]:
        half = float(self.bt.spread) / 2.0
        bid = float(close_price) - half
        ask = float(close_price) + half
        return {
            "bid": bid,
            "ask": ask,
            "last": float(close_price),
            "spread": float(self.bt.spread),
            "source": "backtest_synth",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _close_position(self, pos: Position, now: pd.Timestamp, bar_index: int, exit_price: float, reason: str):
        exit_exec = _apply_slippage(exit_price, pos.side, is_entry=False, slippage=self.bt.slippage)
        pnl = _pnl_usd_gold(pos.side, pos.entry_price, exit_exec, pos.lot, contract_size=self.contract_size)

        # commissions (one side on open + one side on close)
        pnl -= float(self.bt.commission_per_lot) * float(pos.lot)

        pos.close_time = now
        pos.close_bar_index = bar_index
        pos.close_price = float(exit_exec)
        pos.exit_reason = reason
        pos.pnl_usd = float(pnl)
        pos.duration_bars = int(bar_index - pos.open_bar_index)

        self.balance += float(pnl)
        self.equity = self.balance

        self._daily_pnl += float(pnl)

        self._trade_rows.append(asdict(pos))

    def _expire_pending(self, now: pd.Timestamp):
        if not self.pending:
            return
        expire_minutes = int(self.bt.pending_expire_minutes or 0)
        if expire_minutes <= 0:
            return
        keep: List[PendingOrder] = []
        for po in self.pending:
            age_min = (now - po.created_time).total_seconds() / 60.0
            if age_min >= expire_minutes:
                # expired -> drop silently (you can add a cycle log field if you want)
                continue
            keep.append(po)
        self.pending = keep

    def _try_fill_pending(self, now: pd.Timestamp, bar_index: int, o: float, h: float, l: float):
        if not self.pending:
            return
        filled: List[PendingOrder] = []
        keep: List[PendingOrder] = []
        for po in self.pending:
            if _limit_fill(po.side, po.planned_entry, o, h, l):
                po.filled_time = now
                po.filled_bar_index = bar_index
                # conservative fill price: planned entry + slippage on entry
                fill_exec = _apply_slippage(float(po.planned_entry), po.side, is_entry=True, slippage=self.bt.slippage)
                po.filled_price = float(fill_exec)
                filled.append(po)
            else:
                keep.append(po)
        self.pending = keep
        for po in filled:
            self.positions.append(po.to_position())
            self.last_trade_time = now
            self.last_trade_bar_index = bar_index
            self._daily_trades += 1

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        df must be sorted ascending by time.
        """
        if df is None or df.empty:
            raise ValueError("OHLCV dataframe is empty.")

        df = df.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            df = df.set_index("time")
        else:
            # assume index is datetime-like
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

        # normalize volume column for analyzer compatibility
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            else:
                df["volume"] = 0

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        times = df.index.to_list()
        n = len(df)

        # rolling simulation
        for i in range(n):
            now = times[i]

            # daily reset
            day = pd.Timestamp(now.date())
            if self._current_day is None or day != self._current_day:
                self._reset_daily(day)

            # update open positions first (intra-bar)
            o = float(df["open"].iloc[i])
            h = float(df["high"].iloc[i])
            l = float(df["low"].iloc[i])
            c = float(df["close"].iloc[i])

            # pending maintenance
            self._expire_pending(now)
            self._try_fill_pending(now, i, o, h, l)

            still_open: List[Position] = []
            for pos in self.positions:
                hit, level = _bar_hit(pos.side, o, h, l, c, pos.stop_loss, pos.take_profit)
                if hit:
                    self._close_position(pos, now, i, float(level), hit)
                else:
                    # optional timeout based on risk_manager_config.POSITION_TIMEOUT_MINUTES
                    timeout_min = _get_cfg(self.bot_config, "risk_manager_config.POSITION_TIMEOUT_MINUTES", None)
                    if timeout_min:
                        bars_per_min = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}
                        tf = str(self.bt.timeframe).upper()
                        bar_minutes = bars_per_min.get(tf, 15)
                        max_bars = int(math.ceil(float(timeout_min) / float(bar_minutes)))
                        if (i - pos.open_bar_index) >= max_bars:
                            self._close_position(pos, now, i, c, "TIMEOUT")
                            continue
                    still_open.append(pos)
            self.positions = still_open

            # record equity each bar
            self._equity_rows.append(
                {
                    "time": now,
                    "equity": self.equity,
                    "balance": self.balance,
                    "open_positions": len(self.positions),
                    "pending_orders": len(self.pending),
                    "daily_trades": self._daily_trades,
                    "daily_pnl": self._daily_pnl,
                }
            )

            # warmup: require enough bars for indicators + score history
            if i < self.bt.warmup_bars:
                continue

            # run analysis on a sliding window up to current bar (inclusive)
            window = df.iloc[max(0, i - self.bt.bars_to_fetch + 1): i + 1].copy()
            entry_factor = float(_get_cfg(self.bot_config, "technical_settings.ENTRY_FACTOR", 0.25) or 0.25)

            try:
                # IMPORTANT FIX:
                # Your analyze_gold_market signature does NOT accept analysis_only=...
                # Passing that kwarg causes exception each cycle -> fallback score=50 forever.
                raw = analyze_gold_market(
                    dataframe=window.reset_index(),
                    timeframe=self.bt.timeframe,
                    entry_factor=entry_factor,
                    config=self.bot_config,
                    scalping_mode=True,
                )
                result = raw if isinstance(raw, dict) else (raw.__dict__ if hasattr(raw, "__dict__") else {})
                if not isinstance(result, dict):
                    result = {}
            except Exception as e:
                result = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "score": 50.0,
                    "error": True,
                    "analyzer_error": True,
                    "analyzer_error_msg": str(e),
                    "reasons": [str(e)],
                    "context": {},
                }

            analyzer_signal = _normalize_signal(result.get("signal"))
            score = float(result.get("score", 0.0) or 0.0)
            confidence = float(result.get("confidence", 0.0) or 0.0)

            min_conf = float(_get_cfg(self.bot_config, "technical_settings.SCALPING_MIN_CONFIDENCE", 38) or 38)

            live = self._synthesize_live(c)

            decision = {
                "cycle": i,
                "time": now,
                "analyzer_signal": analyzer_signal,
                "score": score,
                "confidence": confidence,
                "min_conf": min_conf,
                "close": c,
                "open_positions": len(self.positions),
                "pending_orders": len(self.pending),
                "analyzer_error": False,
                "analyzer_error_msg": "",
            }

            # propagate analyzer exceptions (for diagnostics)
            if bool(result.get("analyzer_error") or result.get("error")):
                decision["analyzer_error"] = True
                decision["analyzer_error_msg"] = str(
                    result.get("analyzer_error_msg") or (result.get("reasons", [""])[0] if isinstance(result.get("reasons"), list) else "") or ""
                )

            # apply bot-level gate similar to bot.py (signal first)
            if analyzer_signal not in ("BUY", "SELL"):
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = "ANALYZER_NONE"
                self._cycle_rows.append(decision)
                continue

            # confidence gate
            if confidence < min_conf:
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = "CONF_TOO_LOW"
                self._cycle_rows.append(decision)
                continue

            can_open, gate_reason = self._can_open_new(now, i)
            if not can_open:
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = gate_reason
                self._cycle_rows.append(decision)
                continue

            # finalize (market/limit, deviation, rr, lot sizing)
            order = self.risk_manager.finalize_order(
                analysis=result,
                live=live,
                symbol=self.bt.symbol,
                config=self.bot_config,
            )

            order_d = order if isinstance(order, dict) else (order.__dict__ if hasattr(order, "__dict__") else {})
            if not isinstance(order_d, dict):
                order_d = {}

            allowed = bool(order_d.get("is_trade_allowed", False))

            decision["final_signal"] = _normalize_signal(order_d.get("signal"))
            decision["order_type"] = order_d.get("order_type")
            decision["is_trade_allowed"] = allowed
            decision["reject_reason"] = order_d.get("reject_reason")
            decision["deviation_pips"] = float(order_d.get("deviation_pips", 0.0) or 0.0)
            decision["rr_ratio"] = float(order_d.get("rr_ratio", 0.0) or 0.0)
            self._cycle_rows.append(decision)

            if not allowed:
                continue

            side = _normalize_signal(order_d.get("signal"))
            entry = float(order_d["entry_price"])
            sl = float(order_d["stop_loss"])
            tp = float(order_d["take_profit"])
            lot = float(order_d["lot_size"])
            order_type = str(order_d.get("order_type", "MARKET")).upper()

            pos_id = f"BT_{now.strftime('%Y%m%d_%H%M%S')}_{i}"
            notes = list(order_d.get("decision_notes") or [])
            session = str(result.get("context", {}).get("session_analysis", {}).get("current_session", "UNKNOWN"))

            if order_type == "LIMIT" and self.bt.enable_limit_orders:
                po = PendingOrder(
                    id=pos_id,
                    side=side,
                    symbol=self.bt.symbol,
                    created_time=now,
                    created_bar_index=i,
                    planned_entry=float(entry),
                    stop_loss=float(sl),
                    take_profit=float(tp),
                    lot=float(lot),
                    order_type="LIMIT",
                    confidence=float(confidence),
                    score=float(score),
                    rr=float(order_d.get("rr_ratio", 0.0) or 0.0),
                    session=session,
                    deviation_pips=float(order_d.get("deviation_pips", 0.0) or 0.0),
                    notes=notes,
                )
                self.pending.append(po)

                # approximate daily risk used = risk_amount_usd
                risk_used = float(order_d.get("risk_amount_usd", 0.0) or 0.0)
                self._daily_risk_used += risk_used

                # do NOT set last_trade_time here (filled later)
                continue

            # MARKET (default)
            entry_exec = _apply_slippage(entry, side, is_entry=True, slippage=self.bt.slippage)

            pos = Position(
                id=pos_id,
                side=side,
                symbol=self.bt.symbol,
                open_time=now,
                open_bar_index=i,
                entry_price=entry_exec,
                stop_loss=sl,
                take_profit=tp,
                lot=lot,
                order_type=str(order_d.get("order_type", "MARKET")),
                confidence=confidence,
                score=score,
                rr=float(order_d.get("rr_ratio", 0.0) or 0.0),
                session=session,
                planned_entry=float(result.get("entry_price", entry) or entry),
                deviation_pips=float(order_d.get("deviation_pips", 0.0) or 0.0),
                notes=notes,
            )
            self.positions.append(pos)

            self.last_trade_time = now
            self.last_trade_bar_index = i
            self._daily_trades += 1

            # approximate daily risk used = risk_amount_usd
            risk_used = float(order_d.get("risk_amount_usd", 0.0) or 0.0)
            self._daily_risk_used += risk_used

        trades_df = pd.DataFrame(self._trade_rows)
        equity_df = pd.DataFrame(self._equity_rows).set_index("time")
        cycle_df = pd.DataFrame(self._cycle_rows).set_index("time") if self._cycle_rows else pd.DataFrame()

        metrics = compute_metrics(trades_df, equity_df, self.bt.starting_equity)

        return BacktestResult(
            config=self.bot_config,
            overrides=self.overrides,
            trades=trades_df,
            equity_curve=equity_df,
            cycle_log=cycle_df,
            metrics=metrics,
        )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, starting_equity: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "starting_equity": float(starting_equity),
        "ending_equity": float(equity["equity"].iloc[-1]) if equity is not None and not equity.empty else float(starting_equity),
        "net_pnl": 0.0,
        "total_trades": int(len(trades)) if trades is not None else 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "avg_trade_pnl": 0.0,
        "median_trade_pnl": 0.0,
        "expectancy": 0.0,
    }

    if equity is not None and not equity.empty:
        out["net_pnl"] = out["ending_equity"] - float(starting_equity)

        eq = equity["equity"].astype(float)
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max.replace(0, np.nan)
        out["max_drawdown"] = float(dd.min() * 100.0) if len(dd) else 0.0  # percent

    if trades is None or trades.empty:
        return out

    pnls = trades["pnl_usd"].astype(float)
    out["avg_trade_pnl"] = float(pnls.mean())
    out["median_trade_pnl"] = float(pnls.median())

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    out["win_rate"] = float(len(wins) / len(pnls) * 100.0) if len(pnls) else 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    out["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # expectancy per trade
    p_win = len(wins) / len(pnls) if len(pnls) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    out["expectancy"] = float(p_win * avg_win - (1 - p_win) * avg_loss)

    return out
