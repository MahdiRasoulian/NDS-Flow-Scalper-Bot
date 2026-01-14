"""
تحلیل ساختار بازار و الگوهای SMC
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd


import math


from .models import (
    SwingPoint, SwingType, FVG, FVGType,
    OrderBlock, LiquiditySweep, MarketStructure, MarketTrend
)

logger = logging.getLogger(__name__)


class SMCAnalyzer:
    """
    تحلیل‌گر ساختار بازار و الگوهای Smart Money Concepts
    """

    def __init__(self, df: pd.DataFrame, atr_value: float, settings: dict = None):
        if settings is None:
            raise ValueError("SMCAnalyzer requires settings from bot_config.json")
        self.df = df
        self.atr = atr_value
        self.GOLD_SETTINGS = settings
        self.settings = self.GOLD_SETTINGS
        self.debug_smc = bool(self.settings.get("DEBUG_SMC", False))
        self._last_trend = MarketTrend.RANGING
        self._last_trend_confidence = 0.0
        self._prepare_data()

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_smc:
            logger.debug(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        logger.info(message, *args)

    def _log_verbose(self, message: str, *args: Any) -> None:
        """لاگ مرحله‌ای با سطح INFO (فقط وقتی DEBUG_SMC فعال باشد).

        در نسخه‌های قبلی پروژه، لاگ‌های مفصل در خروجی INFO نمایش داده می‌شد.
        این متد همان رفتار را (به صورت کنترل‌شده) بازمی‌گرداند تا بتوانید
        قدم‌به‌قدم رفتار ربات را رصد کنید.
        """
        if self.debug_smc:
            logger.info(message, *args)

    def _normalize_volatility_state(self, volatility_state: Optional[str]) -> Optional[str]:
        if not volatility_state:
            return None
        state = str(volatility_state).upper()
        mapping = {
            "HIGH": "HIGH_VOLATILITY",
            "LOW": "LOW_VOLATILITY",
            "MODERATE": "MODERATE_VOLATILITY",
        }
        if state in mapping:
            return mapping[state]
        if state.endswith("_VOLATILITY"):
            return state
        return state

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _safe_rvol(self, volume_analysis: Optional[Dict]) -> float:
        if not volume_analysis:
            return 1.0
        return max(0.1, self._safe_float(volume_analysis.get("rvol", 1.0), 1.0))

    def _volume_factor(self, rvol: float) -> float:
        """Map RVOL to a soft confidence multiplier."""
        rvol = self._safe_float(rvol, 1.0)
        if rvol <= 0.5:
            return 0.6
        if rvol <= 0.8:
            return 0.75
        if rvol <= 1.2:
            return 1.0
        if rvol <= 1.6:
            return 1.1
        return 1.2

    def _compute_displacement_atr(self, price: float, level: float) -> Tuple[float, float]:
        displacement = abs(float(price) - float(level))
        atr_value = float(self.atr) if self.atr else 0.0
        disp_atr = displacement / atr_value if atr_value > 0 else 0.0
        return displacement, disp_atr

    def _displacement_score(
        self,
        candle: pd.Series,
        atr_value: float,
        rvol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Score displacement quality for a candle (0..1).

        Returns:
          {
            "disp_atr": float,
            "body_ratio": float,
            "wick_ratio": float,
            "rvol_factor": float,
            "score": float,
            "ok": bool
          }
        """
        try:
            open_ = float(candle["open"])
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])
        except Exception:
            return {
                "disp_atr": 0.0,
                "body_ratio": 0.0,
                "wick_ratio": 1.0,
                "rvol_factor": 1.0,
                "score": 0.0,
                "ok": False,
            }

        candle_range = max(1e-6, high - low)
        body = abs(close - open_)
        body_ratio = body / candle_range
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        wick_ratio = (upper_wick + lower_wick) / candle_range

        atr_value = float(atr_value) if atr_value else 0.0
        disp_atr = (body / atr_value) if atr_value > 0 else 0.0

        rvol_factor = self._volume_factor(rvol) if rvol is not None else 1.0
        disp_component = min(1.0, disp_atr / 1.2) if disp_atr > 0 else 0.0
        wick_component = max(0.0, 1.0 - wick_ratio)

        base_score = (0.45 * disp_component) + (0.35 * body_ratio) + (0.2 * wick_component)
        score = min(1.0, base_score * min(1.15, rvol_factor))

        min_score = float(self.settings.get("DISPLACEMENT_SCORE_MIN", 0.55))
        ok = score >= min_score

        return {
            "disp_atr": disp_atr,
            "body_ratio": body_ratio,
            "wick_ratio": wick_ratio,
            "rvol_factor": rvol_factor,
            "score": score,
            "ok": ok,
        }

    def _zone_penetration(self, atr_value: float) -> float:
        try:
            penetration_atr = float(self.settings.get("FLOW_TOUCH_PENETRATION_ATR", 0.05))
        except Exception:
            penetration_atr = 0.05
        return float(atr_value) * penetration_atr if atr_value else 0.0

    def _is_zone_touch(
        self,
        high: float,
        low: float,
        close: float,
        top: float,
        bottom: float,
        atr_value: float,
    ) -> bool:
        penetration = self._zone_penetration(atr_value)
        close_in_zone = bottom <= close <= top
        wick_from_above = high >= top and low <= top - penetration
        wick_from_below = low <= bottom and high >= bottom + penetration
        return close_in_zone or wick_from_above or wick_from_below

    def _make_zone_key(
        self,
        zone_type: str,
        top: float,
        bottom: float,
        origin_idx: int,
        break_idx: int,
    ) -> str:
        precision = int(self.settings.get("FLOW_ZONE_KEY_PRECISION", 5))
        top_key = round(float(top), precision)
        bottom_key = round(float(bottom), precision)
        return f"{zone_type}|origin={origin_idx}|break={break_idx}|top={top_key}|bottom={bottom_key}"

    def _select_zone_candidate(
        self,
        existing: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        if existing is None:
            return candidate
        existing_conf = float(existing.get("confidence", 0.0))
        candidate_conf = float(candidate.get("confidence", 0.0))
        existing_touches = int(existing.get("touch_count", 0))
        candidate_touches = int(candidate.get("touch_count", 0))
        if candidate_touches < existing_touches:
            return candidate
        if candidate_touches == existing_touches and candidate_conf > existing_conf:
            return candidate
        return existing

    def _count_zone_touches(
        self,
        df: pd.DataFrame,
        start_idx: int,
        top: float,
        bottom: float,
        atr_value: float,
    ) -> int:
        touches = 0
        for j in range(start_idx, len(df)):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            close = float(df["close"].iloc[j])
            if self._is_zone_touch(high, low, close, top, bottom, atr_value):
                touches += 1
        return touches

    def _scan_zone_touches(
        self,
        df: pd.DataFrame,
        start_idx: int,
        top: float,
        bottom: float,
        zone_type: str,
        atr_value: float,
        retest_policy: str,
        zone_id: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[int], int, bool, str]:
        touch_indices: List[int] = []
        confirmed_hits: List[Tuple[int, str, int]] = []
        first_touch_idx = None
        max_touches = int(self.settings.get("FLOW_MAX_TOUCHES", 2))
        in_touch = False
        current_touch_idx: Optional[int] = None

        for j in range(start_idx, len(df)):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            close = float(df["close"].iloc[j])
            touched = self._is_zone_touch(high, low, close, top, bottom, atr_value)
            if touched:
                if not in_touch:
                    in_touch = True
                    current_touch_idx = j
                    touch_indices.append(j)
                    first_touch_idx = first_touch_idx or j
                    if zone_id:
                        candle_time = None
                        if "time" in df.columns:
                            candle_time = df["time"].iloc[j]
                        self._log_debug(
                            "[NDS][FLOW_DEBUG][TOUCH_INCREMENT] zone_id=%s idx=%s time=%s touches=%s",
                            zone_id,
                            j,
                            candle_time,
                            len(touch_indices),
                        )
                confirmed, reason = self._confirm_retest(
                    df=df,
                    retest_idx=j,
                    zone_type=zone_type,
                    top=top,
                    bottom=bottom,
                    atr_value=atr_value,
                )
                if confirmed and current_touch_idx is not None:
                    confirmed_hits.append((j, reason, current_touch_idx))
            else:
                in_touch = False
                current_touch_idx = None

        touch_count = len(touch_indices)
        retest_idx = None
        retest_reason = "NO_TOUCH"
        eligible = False

        if touch_count == 0:
            return first_touch_idx, retest_idx, touch_count, eligible, retest_reason

        retest_policy = str(retest_policy or "").upper()
        if retest_policy == "FIRST_TOUCH":
            retest_idx = first_touch_idx
            confirmed_first = None
            if first_touch_idx is not None:
                for hit_idx, reason, visit_idx in confirmed_hits:
                    if visit_idx == first_touch_idx:
                        confirmed_first = (hit_idx, reason)
                        break
            if confirmed_first:
                retest_idx, retest_reason = confirmed_first
                eligible = True
            else:
                retest_reason = "FIRST_TOUCH_UNCONFIRMED"
        else:
            if confirmed_hits:
                last_idx, last_reason, _ = confirmed_hits[-1]
                retest_idx, retest_reason = last_idx, last_reason
                eligible = True
            else:
                retest_idx = touch_indices[-1]
                retest_reason = "NO_CONFIRMED_TOUCH"

        if touch_count > max_touches:
            eligible = False
            retest_reason = "TOO_MANY_TOUCHES"
            if retest_idx is None:
                retest_idx = touch_indices[-1]

        return first_touch_idx, retest_idx, touch_count, eligible, retest_reason

    def _confirm_retest(
        self,
        df: pd.DataFrame,
        retest_idx: int,
        zone_type: str,
        top: float,
        bottom: float,
        atr_value: float,
    ) -> Tuple[bool, str]:
        """Validate retest with directional confirmation rules."""
        try:
            candle = df.iloc[retest_idx]
            open_ = float(candle["open"])
            close = float(candle["close"])
            high = float(candle["high"])
            low = float(candle["low"])
        except Exception:
            return False, "INVALID_CANDLE"

        direction = "BULLISH" if "BULLISH" in zone_type else "BEARISH"
        mid = (top + bottom) / 2.0

        if not self._is_zone_touch(high, low, close, top, bottom, atr_value):
            return False, "NO_TOUCH"

        if direction == "BULLISH" and low <= top and close > top:
            return True, "CLOSE_RECLAIM"
        if direction == "BEARISH" and high >= bottom and close < bottom:
            return True, "CLOSE_REJECT"

        if direction == "BULLISH" and low <= bottom and close > open_:
            return True, "WICK_REJECTION"
        if direction == "BEARISH" and high >= top and close < open_:
            return True, "WICK_REJECTION"

        if low <= mid <= high and (retest_idx + 1) < len(df):
            next_candle = df.iloc[retest_idx + 1]
            disp_meta = self._displacement_score(
                next_candle,
                atr_value,
                rvol=next_candle.get("rvol") if "rvol" in df.columns else None,
            )
            next_close = float(next_candle["close"])
            if disp_meta["ok"]:
                if direction == "BULLISH" and next_close > top:
                    return True, "MID_TOUCH_DISPLACEMENT"
                if direction == "BEARISH" and next_close < bottom:
                    return True, "MID_TOUCH_DISPLACEMENT"

        return False, "NO_CONFIRM"

    def _find_breaker_blocks(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[Dict[str, Any]]:
        """Identify breaker blocks (OB -> displacement break -> inversion retest)."""
        breakers: List[Dict[str, Any]] = []
        if df is None or len(df) < 6:
            return breakers

        atr_value = float(self.atr) if self.atr else 0.0
        min_move_mult = float(self.settings.get("BRK_MIN_MOVE_ATR", 0.8))
        min_move = atr_value * min_move_mult if atr_value > 0 else 0.0
        lookback = int(self.settings.get("BRK_LOOKBACK_BARS", 180))
        start_idx = max(2, len(df) - lookback)
        last_idx = len(df) - 1

        raw_obs: List[Dict[str, Any]] = []
        for i in range(start_idx, len(df) - 2):
            candle_a = df.iloc[i]
            candle_b = df.iloc[i + 1]
            a_open = float(candle_a["open"])
            a_close = float(candle_a["close"])
            a_high = float(candle_a["high"])
            a_low = float(candle_a["low"])
            b_close = float(candle_b["close"])
            b_open = float(candle_b["open"])

            is_red = a_close < a_open
            is_green = a_close > a_open

            move_up = b_close - a_high
            move_down = a_low - b_close

            if is_red and b_close > a_high and (move_up >= min_move):
                raw_obs.append(
                    {"type": "BEARISH_OB", "high": a_high, "low": a_low, "index": i}
                )
            if is_green and b_close < a_low and (move_down >= min_move):
                raw_obs.append(
                    {"type": "BULLISH_OB", "high": a_high, "low": a_low, "index": i}
                )

        for ob in raw_obs:
            ob_high = float(ob["high"])
            ob_low = float(ob["low"])
            ob_idx = int(ob["index"])
            break_idx = None
            disp_meta = None

            for j in range(ob_idx + 1, len(df)):
                candle = df.iloc[j]
                close = float(candle["close"])
                if ob["type"] == "BEARISH_OB" and close > ob_high:
                    disp_meta = self._displacement_score(
                        candle,
                        atr_value,
                        rvol=candle.get("rvol") if "rvol" in df.columns else None,
                    )
                    if disp_meta["ok"]:
                        break_idx = j
                        break
                if ob["type"] == "BULLISH_OB" and close < ob_low:
                    disp_meta = self._displacement_score(
                        candle,
                        atr_value,
                        rvol=candle.get("rvol") if "rvol" in df.columns else None,
                    )
                    if disp_meta["ok"]:
                        break_idx = j
                        break

            if break_idx is None:
                continue

            retest_idx = None
            zone_type = "BULLISH_BREAKER" if ob["type"] == "BEARISH_OB" else "BEARISH_BREAKER"
            retest_policy = str(self.settings.get("FLOW_RETEST_POLICY", "FIRST_TOUCH"))
            zone_id = self._make_zone_key(zone_type, ob_high, ob_low, ob_idx, break_idx)
            first_touch_idx, retest_idx, touch_count, eligible, retest_reason = self._scan_zone_touches(
                df=df,
                start_idx=break_idx + 1,
                top=ob_high,
                bottom=ob_low,
                zone_type=zone_type,
                atr_value=atr_value,
                retest_policy=retest_policy,
                zone_id=zone_id,
            )

            max_touches = int(self.settings.get("FLOW_MAX_TOUCHES", 2))
            touch_penalty = float(self.settings.get("FLOW_TOUCH_PENALTY", 0.55))

            fresh = touch_count == 1
            age_anchor = retest_idx if retest_idx is not None else break_idx
            age_bars = max(0, last_idx - age_anchor)
            confidence = float(disp_meta.get("score", 0.0))
            confidence *= max(0.3, 1.0 - (age_bars / 200.0))
            if not fresh:
                confidence *= touch_penalty

            zone_payload = {
                    "type": zone_type,
                    "top": ob_high,
                    "bottom": ob_low,
                    "mid": (ob_high + ob_low) / 2.0,
                    "index": retest_idx,
                    "age_bars": age_bars,
                    "disp_atr": disp_meta.get("disp_atr", 0.0),
                    "confidence": confidence,
                    "source_ob_index": ob_idx,
                    "first_touch_index": first_touch_idx,
                    "retest_index": retest_idx,
                    "touch_count": touch_count,
                    "fresh": fresh,
                    "eligible": eligible,
                    "retest_reason": retest_reason,
                    "zone_id": zone_id,
                    "notes": f"break_idx={break_idx} retest_idx={retest_idx} policy={retest_policy}",
            }
            self._log_debug(
                "[NDS][FLOW_DEBUG][ZONE_ID] zone_id=%s type=%s top=%.5f bottom=%.5f origin_idx=%s break_idx=%s retest_idx=%s touches=%s eligible=%s",
                zone_id,
                zone_type,
                float(ob_high),
                float(ob_low),
                ob_idx,
                break_idx,
                retest_idx,
                touch_count,
                eligible,
            )
            zone_key = zone_payload.get("zone_id")
            if zone_key:
                existing = next((z for z in breakers if z.get("zone_id") == zone_key), None)
                if existing is None:
                    breakers.append(zone_payload)
                else:
                    replacement = self._select_zone_candidate(existing, zone_payload)
                    if replacement is not existing:
                        breakers = [z for z in breakers if z.get("zone_id") != zone_key]
                        breakers.append(replacement)
            else:
                breakers.append(zone_payload)
            if eligible:
                self._log_info(
                    "[NDS][FLOW][RETEST_OK] type=BREAKER reason=%s idx=%s touches=%s fresh=%s",
                    retest_reason,
                    retest_idx,
                    touch_count,
                    fresh,
                )
            elif retest_reason == "TOO_MANY_TOUCHES":
                self._log_info(
                    "[NDS][FLOW][ZONE_REJECT] reason=TOO_MANY_TOUCHES type=%s idx=%s touches=%s max=%s",
                    zone_type,
                    retest_idx,
                    touch_count,
                    max_touches,
                )
            else:
                self._log_info(
                    "[NDS][FLOW][RETEST_FAIL] type=%s idx=%s reason=%s",
                    zone_type,
                    retest_idx,
                    retest_reason,
                )

        self._log_info("[NDS][SMC][BREAKER] detected=%s", len(breakers))
        if self.debug_smc and breakers:
            for z in breakers[:6]:
                self._log_verbose(
                    "[NDS][SMC][BREAKER] %s idx=%s age=%s zone=[%.2f-%.2f] disp_atr=%.2f conf=%.2f",
                    z.get("type"),
                    z.get("index"),
                    z.get("age_bars"),
                    float(z.get("bottom", 0.0)),
                    float(z.get("top", 0.0)),
                    float(z.get("disp_atr", 0.0)),
                    float(z.get("confidence", 0.0)),
                )

        return breakers

    def _find_inversion_fvgs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify inversion FVGs (FVG broken then retested as inverse zone).

        Definitions:
        - Bullish IFVG: bearish FVG is broken upward (close > top) and later retested as support.
        - Bearish IFVG: bullish FVG is broken downward (close < bottom) and later retested as resistance.
        """
        ifvgs: List[Dict[str, Any]] = []
        if df is None or len(df) < 6:
            return ifvgs

        atr_value = float(self.atr) if self.atr else 0.0
        fvgs = self.detect_fvgs()
        last_idx = len(df) - 1

        for fvg in fvgs:
            fvg_idx = int(getattr(fvg, "index", 0))
            top = float(fvg.top)
            bottom = float(fvg.bottom)
            break_idx = None
            disp_meta = None
            zone_type = None

            if fvg.type == FVGType.BULLISH:
                for j in range(fvg_idx + 1, len(df)):
                    close = float(df["close"].iloc[j])
                    if close < bottom:
                        candle = df.iloc[j]
                        disp_meta = self._displacement_score(
                            candle,
                            atr_value,
                            rvol=candle.get("rvol") if "rvol" in df.columns else None,
                        )
                        if disp_meta["ok"]:
                            break_idx = j
                            zone_type = "BEARISH_IFVG"
                            break
            else:
                for j in range(fvg_idx + 1, len(df)):
                    close = float(df["close"].iloc[j])
                    if close > top:
                        candle = df.iloc[j]
                        disp_meta = self._displacement_score(
                            candle,
                            atr_value,
                            rvol=candle.get("rvol") if "rvol" in df.columns else None,
                        )
                        if disp_meta["ok"]:
                            break_idx = j
                            zone_type = "BULLISH_IFVG"
                            break

            if break_idx is None or zone_type is None:
                continue

            retest_policy = str(self.settings.get("FLOW_RETEST_POLICY", "FIRST_TOUCH"))
            zone_id = self._make_zone_key(zone_type, top, bottom, fvg_idx, break_idx)
            first_touch_idx, retest_idx, touch_count, eligible, retest_reason = self._scan_zone_touches(
                df=df,
                start_idx=break_idx + 1,
                top=top,
                bottom=bottom,
                zone_type=zone_type,
                atr_value=atr_value,
                retest_policy=retest_policy,
                zone_id=zone_id,
            )

            max_touches = int(self.settings.get("FLOW_MAX_TOUCHES", 2))
            touch_penalty = float(self.settings.get("FLOW_TOUCH_PENALTY", 0.55))

            fresh = touch_count == 1
            age_anchor = retest_idx if retest_idx is not None else break_idx
            age_bars = max(0, last_idx - age_anchor)
            confidence = float(disp_meta.get("score", 0.0))
            confidence *= max(0.3, 1.0 - (age_bars / 200.0))
            if not fresh:
                confidence *= touch_penalty
            zone_payload = {
                    "type": zone_type,
                    "top": top,
                    "bottom": bottom,
                    "mid": (top + bottom) / 2.0,
                    "index": retest_idx,
                    "age_bars": age_bars,
                    "disp_atr": disp_meta.get("disp_atr", 0.0),
                    "confidence": confidence,
                    "first_touch_index": first_touch_idx,
                    "retest_index": retest_idx,
                    "touch_count": touch_count,
                    "fresh": fresh,
                    "eligible": eligible,
                    "retest_reason": retest_reason,
                    "zone_id": zone_id,
                    "notes": f"break_idx={break_idx} retest_idx={retest_idx} policy={retest_policy}",
            }
            self._log_debug(
                "[NDS][FLOW_DEBUG][ZONE_ID] zone_id=%s type=%s top=%.5f bottom=%.5f origin_idx=%s break_idx=%s retest_idx=%s touches=%s eligible=%s",
                zone_id,
                zone_type,
                float(top),
                float(bottom),
                fvg_idx,
                break_idx,
                retest_idx,
                touch_count,
                eligible,
            )
            zone_key = zone_payload.get("zone_id")
            if zone_key:
                existing = next((z for z in ifvgs if z.get("zone_id") == zone_key), None)
                if existing is None:
                    ifvgs.append(zone_payload)
                else:
                    replacement = self._select_zone_candidate(existing, zone_payload)
                    if replacement is not existing:
                        ifvgs = [z for z in ifvgs if z.get("zone_id") != zone_key]
                        ifvgs.append(replacement)
            else:
                ifvgs.append(zone_payload)
            if eligible:
                self._log_info(
                    "[NDS][FLOW][RETEST_OK] type=IFVG reason=%s idx=%s touches=%s fresh=%s",
                    retest_reason,
                    retest_idx,
                    touch_count,
                    fresh,
                )
            elif retest_reason == "TOO_MANY_TOUCHES":
                self._log_info(
                    "[NDS][FLOW][ZONE_REJECT] reason=TOO_MANY_TOUCHES type=%s idx=%s touches=%s max=%s",
                    zone_type,
                    retest_idx,
                    touch_count,
                    max_touches,
                )
            else:
                self._log_info(
                    "[NDS][FLOW][RETEST_FAIL] type=%s idx=%s reason=%s",
                    zone_type,
                    retest_idx,
                    retest_reason,
                )

        self._log_info("[NDS][SMC][IFVG] detected=%s", len(ifvgs))
        if self.debug_smc and ifvgs:
            for z in ifvgs[:6]:
                self._log_verbose(
                    "[NDS][SMC][IFVG] %s idx=%s age=%s zone=[%.2f-%.2f] disp_atr=%.2f conf=%.2f",
                    z.get("type"),
                    z.get("index"),
                    z.get("age_bars"),
                    float(z.get("bottom", 0.0)),
                    float(z.get("top", 0.0)),
                    float(z.get("disp_atr", 0.0)),
                    float(z.get("confidence", 0.0)),
                )

        return ifvgs

    def _extract_candle_metrics(self) -> Dict[str, float]:
        try:
            candle = self.df.iloc[-1]
            open_ = float(candle['open'])
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            candle_range = max(0.0001, high - low)
            body = abs(close - open_)
            body_ratio = body / candle_range
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            return {
                "range": candle_range,
                "body": body,
                "body_ratio": body_ratio,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "close": close,
                "open": open_,
            }
        except Exception:
            return {
                "range": 0.0,
                "body": 0.0,
                "body_ratio": 0.0,
                "upper_wick": 0.0,
                "lower_wick": 0.0,
                "close": 0.0,
                "open": 0.0,
            }

    def _compute_fakeout_risk(self, break_level: float, direction: str) -> float:
        """Estimate fakeout risk using wick/body structure."""
        metrics = self._extract_candle_metrics()
        candle_range = metrics["range"] or 0.0001
        body_ratio = metrics["body_ratio"]
        upper_wick_ratio = metrics["upper_wick"] / candle_range
        lower_wick_ratio = metrics["lower_wick"] / candle_range
        close = metrics["close"]

        risk = 0.0
        if body_ratio < 0.25:
            risk += 0.35
        if direction == "BULLISH":
            if upper_wick_ratio > 0.45:
                risk += 0.35
            if close < break_level:
                risk += 0.3
        elif direction == "BEARISH":
            if lower_wick_ratio > 0.45:
                risk += 0.35
            if close > break_level:
                risk += 0.3

        try:
            if len(self.df) >= 2:
                prev_close = float(self.df['close'].iloc[-2])
                if direction == "BULLISH" and prev_close > break_level and close < break_level:
                    risk += 0.4
                elif direction == "BEARISH" and prev_close < break_level and close > break_level:
                    risk += 0.4
        except Exception:
            pass

        return min(1.0, risk)

    def _prepare_data(self) -> None:
        """آماده‌سازی داده‌های پایه"""
        self.df = self.df.copy()
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['body_ratio'] = self.df['body'] / self.df['range'].replace(0, 0.001)
        self.df['mid_price'] = (self.df['high'] + self.df['low']) / 2

    def _get_swing_period(self, timeframe: str) -> int:
        """تعیین دوره سوینگ بر اساس تایم‌فریم (با نرمال‌سازی و مقدار پیش‌فرض امن)"""
        tf_raw = str(timeframe or "")
        tf = tf_raw.strip().upper()

        # Alias mapping برای تحمل فرمت‌های مختلف تایم‌فریم
        alias_map = {
            "15": "M15",
            "15M": "M15",
            "M_15": "M15",
            "MIN15": "M15",
            "M15 ": "M15",

            "5": "M5",
            "5M": "M5",
            "M_5": "M5",
            "MIN5": "M5",

            "1": "M1",
            "1M": "M1",
            "M_1": "M1",
            "MIN1": "M1",

            "30": "M30",
            "30M": "M30",
            "M_30": "M30",
            "MIN30": "M30",

            "60": "H1",
            "1H": "H1",
            "H_1": "H1",

            "240": "H4",
            "4H": "H4",
            "H_4": "H4",
        }
        tf = alias_map.get(tf, tf)

        # تنظیمات کاربر (اگر وجود دارد)
        swing_period_map = self.settings.get("SWING_PERIOD_MAP", {}) or {}

        # کلیدها را نرمال کنیم تا اگر کاربر lower/space گذاشته بود هم کار کند
        norm_map = {str(k).strip().upper(): v for k, v in swing_period_map.items()}

        # اگر در تنظیمات موجود بود، همان اولویت دارد
        if tf in norm_map:
            return int(norm_map[tf])

        # مقدار پیش‌فرض امن (برای اینکه smoke test و اجرا fail نشود)
        default_map = {
            "M1": 5,
            "M5": 7,
            "M15": 9,   # حساسیت مناسب برای M15
            "M30": 11,
            "H1": 13,
            "H4": 17,
        }
        if tf in default_map:
            return int(default_map[tf])

        raise KeyError(f"Missing SWING_PERIOD_MAP for timeframe: {timeframe}")


    def detect_swings(self, timeframe: str = 'M15') -> List[SwingPoint]:
        """
        نسخه نهایی و بهینه‌شده شناسایی سوینگ برای انس جهانی طلا
        تمرکز بر دقت در تایید ساختار (BOS/CHOCH)
        """
        period = self._get_swing_period(timeframe)
        df = self.df.reset_index(drop=True)

        if len(df) < period * 2 + 1:
            self._log_debug(
                "[NDS][SMC][SWINGS] insufficient data (have=%s need=%s)",
                len(df),
                period * 2 + 1,
            )
            return []

        high_series = df['high']
        low_series = df['low']

        high_rolling_max = high_series.rolling(window=2 * period + 1, center=True).max()
        low_rolling_min = low_series.rolling(window=2 * period + 1, center=True).min()

        valid_range = range(period, len(df) - period)
        high_indices = [i for i in high_series[high_series == high_rolling_max].index if i in valid_range]
        low_indices = [i for i in low_series[low_series == low_rolling_min].index if i in valid_range]

        self._log_verbose(
            "[NDS][SMC][SWINGS] فرکتال‌های اولیه - High: %s, Low: %s",
            len(high_indices),
            len(low_indices),
        )

        if not high_indices and not low_indices:
            self._log_debug("[NDS][SMC][SWINGS] no initial fractals found")

        min_distance = self.atr * self.settings.get('MIN_ATR_DISTANCE_MULTIPLIER', 1.2)
        min_vol_mult = self.settings.get('MIN_VOLUME_MULTIPLIER', 0.6)
        has_volume = 'volume' in df.columns

        high_swings = []
        last_h_price = None
        for idx in high_indices:
            price = float(df['high'].iloc[idx])
            volume_ok = True
            avg_vol = 1.0
            current_vol = None
            if has_volume:
                recent_vol = df['volume'].iloc[max(0, idx - period):idx]
                avg_vol = float(recent_vol.mean()) if not recent_vol.empty else 1.0
                if pd.isna(avg_vol):
                    avg_vol = 1.0
                current_vol = float(df['volume'].iloc[idx])
                if pd.isna(current_vol):
                    current_vol = avg_vol
                volume_ok = current_vol >= avg_vol * min_vol_mult

            if volume_ok and (last_h_price is None or abs(price - last_h_price) >= min_distance):
                high_swings.append(SwingPoint(
                    index=idx,
                    price=price,
                    time=df['time'].iloc[idx],
                    type=SwingType.HIGH,
                    side='HIGH'
                ))
                last_h_price = price
            self._log_debug(
                "[NDS][SMC][SWINGS] high idx=%s price=%.2f volume_ok=%s avg_vol=%.2f current_vol=%s",
                idx,
                price,
                volume_ok,
                avg_vol,
                f"{current_vol:.2f}" if current_vol is not None else "N/A",
            )

        low_swings = []
        last_l_price = None
        for idx in low_indices:
            price = float(df['low'].iloc[idx])
            volume_ok = True
            avg_vol = 1.0
            current_vol = None
            if has_volume:
                recent_vol = df['volume'].iloc[max(0, idx - period):idx]
                avg_vol = float(recent_vol.mean()) if not recent_vol.empty else 1.0
                if pd.isna(avg_vol):
                    avg_vol = 1.0
                current_vol = float(df['volume'].iloc[idx])
                if pd.isna(current_vol):
                    current_vol = avg_vol
                volume_ok = current_vol >= avg_vol * min_vol_mult

            if volume_ok and (last_l_price is None or abs(price - last_l_price) >= min_distance):
                low_swings.append(SwingPoint(
                    index=idx,
                    price=price,
                    time=df['time'].iloc[idx],
                    type=SwingType.LOW,
                    side='LOW'
                ))
                last_l_price = price
            self._log_debug(
                "[NDS][SMC][SWINGS] low idx=%s price=%.2f volume_ok=%s avg_vol=%.2f current_vol=%s",
                idx,
                price,
                volume_ok,
                avg_vol,
                f"{current_vol:.2f}" if current_vol is not None else "N/A",
            )

        self._log_verbose(
            "[NDS][SMC][SWINGS] سوینگ‌های اولیه - High: %s, Low: %s",
            len(high_swings),
            len(low_swings),
        )

        all_swings = sorted(high_swings + low_swings, key=lambda x: x.index)
        if not all_swings:
            self._log_debug("[NDS][SMC][SWINGS] no swings after filters")
            return []

        cleaned = self._clean_consecutive_swings(all_swings)
        if cleaned:
            all_swings = cleaned
            self._log_debug("[NDS][SMC][SWINGS] cleaning result=%s", len(all_swings))
        else:
            self._log_debug("[NDS][SMC][SWINGS] cleaning removed all swings")
            return []

        meaningful = self._filter_meaningful_swings(all_swings)
        if meaningful:
            all_swings = meaningful
            self._log_debug("[NDS][SMC][SWINGS] meaningful swings=%s", len(all_swings))
        else:
            self._log_debug("[NDS][SMC][SWINGS] meaningful filter removed all swings")
            return []

        last_h, last_l = None, None
        for swing in all_swings:
            if swing.side == 'HIGH':
                if last_h:
                    swing.type = SwingType.HH if swing.price > last_h.price else SwingType.LH
                else:
                    swing.type = SwingType.HIGH
                last_h = swing
            else:
                if last_l:
                    swing.type = SwingType.LL if swing.price < last_l.price else SwingType.HL
                else:
                    swing.type = SwingType.LOW
                last_l = swing

        self._log_debug("[NDS][SMC][SWINGS] final swings=%s", len(all_swings))
        self._log_info("[NDS][SMC][SWINGS] detected swings=%s", len(all_swings))

        if self.debug_smc and all_swings:
            self._log_verbose(
                "[NDS][SMC][SWINGS] Swing Types: HH=%s, LH=%s, LL=%s, HL=%s",
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.HH),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.LH),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.LL),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.HL),
            )
            tail = all_swings[-3:]
            for idx, s in enumerate(tail, start=max(1, len(all_swings) - 2)):
                self._log_verbose(
                    "[NDS][SMC][SWINGS] Swing %s: %s@%.2f (%s) time=%s",
                    idx,
                    getattr(s, 'side', 'N/A'),
                    float(getattr(s, 'price', 0.0)),
                    getattr(getattr(s, 'type', None), 'value', str(getattr(s, 'type', 'N/A'))),
                    getattr(s, 'time', 'N/A'),
                )

        return all_swings

    def _clean_consecutive_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف سوینگ‌های تکراری در یک سمت برای به دست آوردن ساختار زیگزاگی تمیز"""
        if not swings:
            self._log_debug("[NDS][SMC][SWINGS] cleaning empty input")
            return []

        cleaned = []
        for s in swings:
            if not cleaned:
                cleaned.append(s)
                continue

            last = cleaned[-1]
            if last.side == s.side:
                if s.side == 'HIGH' and s.price > last.price:
                    cleaned[-1] = s
                    self._log_debug(
                        "[NDS][SMC][SWINGS] replace high %.2f -> %.2f",
                        last.price,
                        s.price,
                    )
                elif s.side == 'LOW' and s.price < last.price:
                    cleaned[-1] = s
                    self._log_debug(
                        "[NDS][SMC][SWINGS] replace low %.2f -> %.2f",
                        last.price,
                        s.price,
                    )
            else:
                cleaned.append(s)

        return cleaned

    def _filter_meaningful_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف نوسانات فرسایشی که حرکت قیمتی موثری ندارند"""
        if len(swings) < 3:
            self._log_debug("[NDS][SMC][SWINGS] meaningful short list (%s swings)", len(swings))
            return swings

        atr_threshold = self.atr * self.settings.get('MEANINGFUL_MOVE_MULT', 0.5)
        meaningful = []

        for i, s in enumerate(swings):
            if i == 0 or i == len(swings) - 1:
                meaningful.append(s)
                continue

            move_size = abs(s.price - swings[i - 1].price)
            if move_size >= atr_threshold:
                meaningful.append(s)
            elif i + 1 < len(swings):
                next_move_size = abs(swings[i + 1].price - s.price)
                if next_move_size >= atr_threshold:
                    meaningful.append(s)

        return meaningful



    def detect_fvgs(self) -> List[FVG]:
        """شناسایی FVGها با تشخیص صحیح «باز بودن» (Unfilled) و لاگ مرحله‌ای.

        مشکل‌های رایج:
        1) اگر منطق filled فقط تا lookahead محدود شود، FVG های قدیمی که بعداً میتیگیت شده‌اند
        ممکن است برای همیشه unfilled بمانند => ورودی Analyzer آلوده می‌شود و entry های مرگبار تولید می‌شود.
        2) اگر filled خیلی سهل‌گیر باشد (touch)، unfilled=0 می‌شود و عملاً FVG برای entry استفاده نمی‌شود.

        سیاست جدید (ضد lookahead و deterministic):
        - filled/mitigated فقط بر اساس داده‌ی موجود در همین لحظه بررسی می‌شود (df همین لحظه).
        - اگر برای performance cap گذاشته شود، زون‌هایی که خارج از cap هستند و هنوز تعیین تکلیف نشده‌اند
        stale علامت می‌خورند تا به عنوان "open معتبر" وارد انتخاب entry نشوند (Analyzer می‌تواند stale را ignore کند).

        تنظیمات مرتبط:
        - FVG_MIN_SIZE_MULTIPLIER (پیش‌فرض: 0.1)
        - FVG_LOOKAHEAD_BARS (پیش‌فرض: 80)  [فقط برای performance cap - نه منطق قطعی]
        - FVG_EVAL_FULL_HISTORY (پیش‌فرض: True)
            * True  : از تشکیل FVG تا آخر df بررسی می‌کند (پیشنهادی و دقیق)
            * False : تا lookahead بررسی می‌کند ولی stale را علامت می‌زند (برای جلوگیری از mislabel)
        - FVG_FILL_MODE: {"full", "touch"} (پیش‌فرض: "full")
        """
        df = self.df
        fvg_list: List[FVG] = []

        if df is None or len(df) < 3:
            return fvg_list

        # --- safety ATR ---
        atr_val = float(self.atr) if self.atr is not None else 0.0
        if not math.isfinite(atr_val) or atr_val <= 0:
            atr_val = 0.0

        min_mult = float(self.settings.get('FVG_MIN_SIZE_MULTIPLIER', 0.1))
        min_fvg_size = float(atr_val * min_mult) if atr_val > 0 else 0.0

        fill_mode = str(self.settings.get("FVG_FILL_MODE", "full")).strip().lower()
        if fill_mode not in ("full", "touch"):
            fill_mode = "full"

        eval_full_history = bool(self.settings.get("FVG_EVAL_FULL_HISTORY", True))

        self._log_verbose(
            "[NDS][SMC][FVG] start | candles=%s atr=%.4f min_size=%.4f fill_mode=%s eval_full_history=%s",
            len(df),
            atr_val,
            min_fvg_size,
            fill_mode,
            eval_full_history,
        )

        # --- 1) Detection (3-candle pattern) ---
        # NOTE: منطق شما حفظ شده؛ فقط robustness روی type/value ها افزوده شده.
        for i in range(2, len(df)):
            c2_high = float(df['high'].iloc[i - 1])
            c2_low = float(df['low'].iloc[i - 1])
            c2_close = float(df['close'].iloc[i - 1])
            c2_open = float(df['open'].iloc[i - 1])

            c2_body = abs(c2_close - c2_open)
            c2_range = max(1e-7, c2_high - c2_low)

            c1_high = float(df['high'].iloc[i - 2])
            c1_low = float(df['low'].iloc[i - 2])
            c3_high = float(df['high'].iloc[i])
            c3_low = float(df['low'].iloc[i])

            # ---- Bullish FVG: c3_low > c1_high ----
            if c3_low > c1_high:
                gap_top = c3_low
                gap_bottom = c1_high
                gap_size = gap_top - gap_bottom

                body_condition = c2_close > c2_open
                body_size_condition = c2_body > (c2_range * 0.3)
                size_condition = (gap_size >= min_fvg_size) if min_fvg_size > 0 else (gap_size > 0)

                volume_condition = True
                if 'rvol' in df.columns:
                    try:
                        rv = float(df['rvol'].iloc[i - 1])
                        if not math.isfinite(rv):
                            rv = 1.0
                        volume_condition = rv > 0.8
                    except (TypeError, ValueError):
                        volume_condition = True

                if body_condition and body_size_condition and size_condition and volume_condition:
                    strength = 1.0
                    if c2_body > c2_range * 0.7:
                        strength = 1.5
                    if 'rvol' in df.columns:
                        try:
                            rv = float(df['rvol'].iloc[i - 1])
                            if math.isfinite(rv) and rv > 1.5:
                                strength *= 1.2
                        except (TypeError, ValueError):
                            pass

                    fvg_list.append(FVG(
                        type=FVGType.BULLISH,
                        top=float(gap_top),
                        bottom=float(gap_bottom),
                        mid=float((gap_top + gap_bottom) / 2.0),
                        time=df['time'].iloc[i - 1],
                        index=i - 1,
                        size=float(gap_size),
                        strength=float(strength),
                    ))

            # ---- Bearish FVG: c1_low > c3_high ----
            if c1_low > c3_high:
                gap_top = c1_low
                gap_bottom = c3_high
                gap_size = gap_top - gap_bottom

                body_condition = c2_close < c2_open
                body_size_condition = c2_body > (c2_range * 0.3)
                size_condition = (gap_size >= min_fvg_size) if min_fvg_size > 0 else (gap_size > 0)

                volume_condition = True
                if 'rvol' in df.columns:
                    try:
                        rv = float(df['rvol'].iloc[i - 1])
                        if not math.isfinite(rv):
                            rv = 1.0
                        volume_condition = rv > 0.8
                    except (TypeError, ValueError):
                        volume_condition = True

                if body_condition and body_size_condition and size_condition and volume_condition:
                    strength = 1.0
                    if c2_body > c2_range * 0.7:
                        strength = 1.5
                    if 'rvol' in df.columns:
                        try:
                            rv = float(df['rvol'].iloc[i - 1])
                            if math.isfinite(rv) and rv > 1.5:
                                strength *= 1.2
                        except (TypeError, ValueError):
                            pass

                    fvg_list.append(FVG(
                        type=FVGType.BEARISH,
                        top=float(gap_top),
                        bottom=float(gap_bottom),
                        mid=float((gap_top + gap_bottom) / 2.0),
                        time=df['time'].iloc[i - 1],
                        index=i - 1,
                        size=float(gap_size),
                        strength=float(strength),
                    ))

        if not fvg_list:
            self._log_info("[NDS][SMC][FVG] detected=0 unfilled=0")
            return fvg_list

        # --- 2) Fill / Unfilled evaluation ---
        # NOTE:
        # - رفتار قبلی: فقط تا lookahead بررسی می‌شد => FVGهای قدیمی ممکن بود برای همیشه unfilled بمانند.
        # - رفتار جدید:
        #   * اگر eval_full_history=True: تا انتهای df بررسی می‌کنیم (پیشنهادی).
        #   * اگر False: تا lookahead بررسی می‌کنیم، ولی FVGهایی که خارج از پنجره هستند و هنوز filled نشده‌اند stale می‌شوند.
        lookahead = int(self.settings.get('FVG_LOOKAHEAD_BARS', 80))
        lookahead = max(3, min(lookahead, len(df)))

        touched_count = 0
        filled_count = 0
        stale_count = 0

        last_bar = len(df) - 1

        for fvg in fvg_list:
            fvg_idx = int(getattr(fvg, "index", 0))
            start_j = fvg_idx + 1
            if start_j > last_bar:
                fvg.filled = False
                setattr(fvg, "stale", False)
                continue

            # تعیین check_limit
            if eval_full_history:
                check_limit = last_bar
            else:
                check_limit = min(fvg_idx + lookahead, last_bar)

            touched = False
            filled = False

            top = float(fvg.top)
            bottom = float(fvg.bottom)

            for j in range(start_j, check_limit + 1):
                high_j = float(df['high'].iloc[j])
                low_j = float(df['low'].iloc[j])

                # touched: هرگونه overlap با ناحیه
                if (low_j <= top) and (high_j >= bottom):
                    touched = True

                if fvg.type == FVGType.BULLISH:
                    # full fill: عبور از مرز مقابل (bottom)
                    if low_j <= bottom:
                        filled = True
                        break
                else:
                    # full fill: عبور از مرز مقابل (top)
                    if high_j >= top:
                        filled = True
                        break

            if fill_mode == "touch":
                fvg.filled = bool(touched)
            else:
                fvg.filled = bool(filled)

            # stale marking (فقط وقتی eval_full_history=False)
            is_stale = False
            if (not eval_full_history) and (check_limit < last_bar) and (not fvg.filled):
                # یعنی پنجره تمام شده ولی هنوز تکلیف کامل مشخص نیست.
                is_stale = True
                stale_count += 1
            setattr(fvg, "stale", is_stale)

            if touched:
                touched_count += 1
            if fvg.filled:
                filled_count += 1

        # unfilled = بازِ معتبر (نه filled) و نه stale
        open_valid = [f for f in fvg_list if (not getattr(f, "filled", False)) and (not getattr(f, "stale", False))]
        unfilled_count = len(open_valid)

        # --- 3) Logging ---
        self._log_info(
            "[NDS][SMC][FVG] detected=%s unfilled=%s | touched=%s filled=%s stale=%s",
            len(fvg_list),
            unfilled_count,
            touched_count,
            filled_count,
            stale_count,
        )

        if self.debug_smc:
            current_price = float(df['close'].iloc[-1])
            open_fvgs = open_valid  # فقط open معتبر را نمایش بده

            def _dist(f: FVG) -> float:
                return min(abs(current_price - float(f.top)), abs(current_price - float(f.bottom)))

            open_fvgs_sorted = sorted(open_fvgs, key=_dist)[:10]
            self._log_verbose(
                "[NDS][SMC][FVG] debug | open_valid=%s (showing up to 10 nearest)",
                len(open_fvgs),
            )
            for k, f in enumerate(open_fvgs_sorted, start=1):
                self._log_verbose(
                    "[NDS][SMC][FVG] open #%s | %s | idx=%s time=%s zone=[%.2f-%.2f] size=%.2f strength=%.2f dist=%.2f stale=%s",
                    k,
                    f.type.value if hasattr(f.type, "value") else str(f.type),
                    getattr(f, "index", -1),
                    getattr(f, "time", "N/A"),
                    float(f.bottom),
                    float(f.top),
                    float(getattr(f, "size", 0.0)),
                    float(getattr(f, "strength", 1.0)),
                    _dist(f),
                    bool(getattr(f, "stale", False)),
                )

        return fvg_list


    def detect_order_blocks(self, lookback: int = 50) -> List[OrderBlock]:
        """شناسایی Order Block های معتبر به سبک SMC (با فیلتر Fresh/Mitigated) و لاگ مرحله‌ای.

        مشکل کلاسیک:
        - اگر mitigated فقط تا lookahead بررسی شود، OBهایی که بعداً میتیگیت شده‌اند ممکن است برای همیشه fresh بمانند.

        سیاست جدید:
        - به صورت پیش‌فرض تا انتهای دیتای موجود بررسی می‌شود (ضد lookahead، چون df همان لحظه است).
        - اگر برای performance cap خواستید، eval_full_history=False کنید و OBهای خارج از پنجره را stale علامت بزنید.
        """
        df = self.df
        order_blocks: List[OrderBlock] = []

        if df is None or len(df) < lookback + 5:
            return order_blocks

        atr = float(self.atr) if self.atr else 0.0
        if not math.isfinite(atr) or atr <= 0:
            atr = 0.0

        min_move_mult = float(self.settings.get("OB_MIN_MOVE_ATR", 1.0))
        min_move_size = atr * min_move_mult

        eval_full_history = bool(self.settings.get("OB_EVAL_FULL_HISTORY", True))

        self._log_verbose(
            "[NDS][SMC][OB] start | candles=%s atr=%.4f lookback=%s min_move=%.4f eval_full_history=%s",
            len(df),
            atr,
            lookback,
            min_move_size,
            eval_full_history,
        )

        # --- 1) Raw detection ---
        for i in range(lookback, len(df) - 3):
            candle_a = df.iloc[i]
            candle_b = df.iloc[i + 1]
            candle_c = df.iloc[i + 2]

            a_open = float(candle_a['open'])
            a_close = float(candle_a['close'])
            a_high = float(candle_a['high'])
            a_low = float(candle_a['low'])

            b_open = float(candle_b['open'])
            b_close = float(candle_b['close'])

            c_close = float(candle_c['close'])

            # Bullish OB: last red candle before strong up displacement
            is_red = a_close < a_open
            move_up = b_close - a_high
            is_strong_up = (
                (b_close > a_high)
                and (b_close > b_open)
                and (
                    (move_up > min_move_size)
                    or ((b_close - b_open) > (atr * 0.8 if atr > 0 else 0.0))
                )
            )

            if is_red and is_strong_up:
                strength = 1.0
                if c_close > float(candle_b['high']):
                    strength += 0.5
                if 'rvol' in df.columns:
                    try:
                        rv = float(df['rvol'].iloc[i + 1])
                        if math.isfinite(rv) and rv > 1.2:
                            strength += 0.5
                    except (TypeError, ValueError):
                        pass

                order_blocks.append(OrderBlock(
                    type='BULLISH_OB',
                    high=float(a_high),
                    low=float(a_low),
                    time=candle_a['time'],
                    index=i,
                    strength=float(strength),
                ))

            # Bearish OB: last green candle before strong down displacement
            is_green = a_close > a_open
            move_down = a_low - b_close  # مثبت وقتی b_close پایین‌تر از a_low است
            is_strong_down = (
                (b_close < a_low)
                and (b_close < b_open)
                and (
                    (move_down > min_move_size)
                    or ((b_open - b_close) > (atr * 0.8 if atr > 0 else 0.0))
                )
            )

            if is_green and is_strong_down:
                strength = 1.0
                if c_close < float(candle_b['low']):
                    strength += 0.5
                if 'rvol' in df.columns:
                    try:
                        rv = float(df['rvol'].iloc[i + 1])
                        if math.isfinite(rv) and rv > 1.2:
                            strength += 0.5
                    except (TypeError, ValueError):
                        pass

                order_blocks.append(OrderBlock(
                    type='BEARISH_OB',
                    high=float(a_high),
                    low=float(a_low),
                    time=candle_a['time'],
                    index=i,
                    strength=float(strength),
                ))

        self._log_info("[NDS][SMC][OB] detected raw=%s", len(order_blocks))
        if not order_blocks:
            return order_blocks

        # --- 2) Fresh/Mitigated filter ---
        lookahead = int(self.settings.get("OB_LOOKAHEAD_BARS", 120))
        lookahead = max(10, min(lookahead, len(df)))
        return_limit = int(self.settings.get("OB_RETURN_LIMIT", 5))
        return_limit = max(1, min(return_limit, 20))

        fresh_blocks: List[OrderBlock] = []
        mitigated_count = 0
        stale_count = 0

        last_bar = len(df) - 1

        for ob in order_blocks:
            ob_idx = int(getattr(ob, "index", 0))
            start = ob_idx + 1
            if start > last_bar:
                # بعد از تشکیل OB هیچ دیتایی نداریم
                fresh_blocks.append(ob)
                setattr(ob, "stale", False)
                continue

            if eval_full_history:
                end = last_bar
            else:
                end = min(ob_idx + lookahead, last_bar)

            ob_high = float(ob.high)
            ob_low = float(ob.low)

            mitigated = False
            for j in range(start, end + 1):
                high_j = float(df['high'].iloc[j])
                low_j = float(df['low'].iloc[j])

                # overlap با محدوده OB => mitigated
                if (low_j <= ob_high) and (high_j >= ob_low):
                    mitigated = True
                    break

            if mitigated:
                mitigated_count += 1
                setattr(ob, "stale", False)
                continue

            # اگر eval_full_history=False و پنجره تمام شده ولی هنوز میتیگیت نشده => stale
            is_stale = False
            if (not eval_full_history) and (end < last_bar):
                is_stale = True
                stale_count += 1
            setattr(ob, "stale", is_stale)

            # فقط fresh معتبر را نگه داریم (نه stale)
            if not is_stale:
                fresh_blocks.append(ob)

        fresh_blocks_sorted = sorted(
            fresh_blocks,
            key=lambda x: (-(float(getattr(x, "strength", 1.0))), -int(getattr(x, "index", 0))),
        )

        selected = fresh_blocks_sorted[:return_limit]
        self._log_verbose(
            "[NDS][SMC][OB] raw=%s mitigated=%s stale=%s fresh_valid=%s selected=%s",
            len(order_blocks),
            mitigated_count,
            stale_count,
            len(fresh_blocks),
            len(selected),
        )

        if self.debug_smc:
            for k, ob in enumerate(selected, start=1):
                self._log_verbose(
                    "[NDS][SMC][OB] fresh #%s | %s | idx=%s time=%s range=[%.2f-%.2f] strength=%.2f stale=%s",
                    k,
                    getattr(ob, "type", "OB"),
                    int(getattr(ob, "index", -1)),
                    getattr(ob, "time", "N/A"),
                    float(getattr(ob, "low", 0.0)),
                    float(getattr(ob, "high", 0.0)),
                    float(getattr(ob, "strength", 1.0)),
                    bool(getattr(ob, "stale", False)),
                )

        return selected


    def detect_liquidity_sweeps(self, swings: List[SwingPoint], lookback_swings: int = 5) -> List[LiquiditySweep]:
        """
        شناسایی نفوذهای فیک (Liquidity Sweeps) با استانداردهای SMC

        بهبودها:
        - محافظت در برابر ATR=0 و NaN (جلوگیری از division by zero)
        - robust کردن rvol
        - حفظ رفتار قبلی (recent_data.tail(40) و یکتاسازی خروجی)
        """
        if not swings:
            return []

        df = self.df
        if df is None or len(df) < 5:
            return []

        sweeps: List[LiquiditySweep] = []
        recent_data = df.tail(40)

        recent_highs = [s for s in swings if s.side == 'HIGH'][-lookback_swings:]
        recent_lows = [s for s in swings if s.side == 'LOW'][-lookback_swings:]

        atr_value = float(self.atr) if self.atr is not None else 0.0
        if not math.isfinite(atr_value) or atr_value <= 0:
            atr_value = 0.0

        min_pen_mult = float(self.settings.get('MIN_SWEEP_PENETRATION_MULTIPLIER', 0.2))
        min_penetration = (atr_value * min_pen_mult) if atr_value > 0 else 0.0
        max_penetration = (atr_value * 3.0) if atr_value > 0 else float("inf")

        for _, row in recent_data.iterrows():
            try:
                high = float(row['high'])
                low = float(row['low'])
                open_ = float(row['open'])
                close = float(row['close'])
                t = row['time']
            except Exception:
                continue

            candle_range = high - low
            if atr_value > 0 and candle_range < (atr_value * 0.5):
                continue

            rvol_value = 1.0
            if 'rvol' in df.columns:
                try:
                    rvol_value = float(row.get('rvol', 1.0))
                    if (not math.isfinite(rvol_value)) or pd.isna(rvol_value):
                        rvol_value = 1.0
                except Exception:
                    rvol_value = 1.0

            # --- Sweep highs => bearish sweep ---
            for swing in recent_highs:
                if t <= swing.time:
                    continue

                if high > swing.price and close < swing.price:
                    penetration = high - swing.price
                    if penetration < 0:
                        continue

                    if (penetration >= min_penetration) and (penetration <= max_penetration):
                        upper_wick = high - max(open_, close)
                        body_size = abs(close - open_)

                        is_valid_shape = (
                            (upper_wick > body_size)
                            or (close < open_)
                            or (upper_wick > candle_range * 0.4)
                        )

                        has_high_volume = rvol_value > 1.5

                        if is_valid_shape or has_high_volume:
                            # strength: نرمال نسبت به ATR اگر ATR>0
                            if atr_value > 0:
                                base = penetration / atr_value
                            else:
                                base = 1.0  # fallback deterministic
                            strength = min(3.0, base + (0.5 if has_high_volume else 0.0))

                            sweeps.append(LiquiditySweep(
                                time=t,
                                type='BEARISH_SWEEP',
                                level=swing.price,
                                penetration=penetration,
                                description=f"Bearish Sweep (RVOL: {rvol_value:.1f}x)",
                                strength=strength
                            ))

            # --- Sweep lows => bullish sweep ---
            for swing in recent_lows:
                if t <= swing.time:
                    continue

                if low < swing.price and close > swing.price:
                    penetration = swing.price - low
                    if penetration < 0:
                        continue

                    if (penetration >= min_penetration) and (penetration <= max_penetration):
                        lower_wick = min(open_, close) - low
                        body_size = abs(close - open_)

                        is_valid_shape = (
                            (lower_wick > body_size)
                            or (close > open_)
                            or (lower_wick > candle_range * 0.4)
                        )

                        has_high_volume = rvol_value > 1.5

                        if is_valid_shape or has_high_volume:
                            if atr_value > 0:
                                base = penetration / atr_value
                            else:
                                base = 1.0
                            strength = min(3.0, base + (0.5 if has_high_volume else 0.0))

                            sweeps.append(LiquiditySweep(
                                time=t,
                                type='BULLISH_SWEEP',
                                level=swing.price,
                                penetration=penetration,
                                description=f"Bullish Sweep (RVOL: {rvol_value:.1f}x)",
                                strength=strength
                            ))

        # --- Deduplicate (keep last occurrence per key) ---
        unique_sweeps: List[LiquiditySweep] = []
        seen = set()
        for sweep in reversed(sweeps):
            key = (sweep.time, sweep.type, round(sweep.level, 2))
            if key not in seen:
                seen.add(key)
                unique_sweeps.append(sweep)

        unique_sweeps.reverse()

        self._log_info("[NDS][SMC][SWEEPS] detected fresh=%s", len(unique_sweeps))
        return unique_sweeps


    def determine_market_structure(
        self,
        swings: List[SwingPoint],
        lookback_swings: int = 4,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
        adx_value: Optional[float] = None,
    ) -> MarketStructure:
        """
        تعیین ساختار بازار با منطق NDS (Nodal Displacement Sequencing)
        تمرکز بر جابجایی نودها (Displacement) و تقارن فرکتالی
        """
        normalized_volatility = self._normalize_volatility_state(volatility_state)

        if len(swings) < 3:
            current_price = float(self.df['close'].iloc[-1])
            return MarketStructure(
                trend=MarketTrend.RANGING,
                bos="NONE",
                choch="NONE",
                last_high=None,
                last_low=None,
                current_price=current_price,
                bos_choch_confidence=0.0,
                volume_analysis=volume_analysis,
                volatility_state=normalized_volatility,
                adx_value=adx_value,
                structure_score=8.0
            )

        # در اسکلپینگ XAUUSD، کاهش فیلتر ATR در نوسان بالا از حذف نودهای مفید جلوگیری می‌کند.
        if normalized_volatility == "HIGH_VOLATILITY":
            dynamic_multiplier = 0.75
        elif normalized_volatility == "LOW_VOLATILITY":
            dynamic_multiplier = 1.1
        else:
            dynamic_multiplier = 1.0
        min_swing_distance = self.atr * dynamic_multiplier

        major_swings = []
        last_high_p, last_low_p = None, None

        for swing in swings:
            if swing.side == 'HIGH':
                if last_high_p is None or abs(swing.price - last_high_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_high_p = swing.price
            else:
                if last_low_p is None or abs(swing.price - last_low_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_low_p = swing.price

        recent_swings = self._get_relevant_swings(major_swings, lookback_swings)
        last_high = next((s for s in reversed(recent_swings) if s.side == 'HIGH'), None)
        last_low = next((s for s in reversed(recent_swings) if s.side == 'LOW'), None)

        current_price = float(self.df['close'].iloc[-1])
        current_high = float(self.df['high'].iloc[-1])
        current_low = float(self.df['low'].iloc[-1])

        trend, trend_strength, trend_confidence = self._determine_trend_with_confidence(
            recent_swings,
            current_price,
            volume_analysis,
            normalized_volatility,
            adx_value,
        )

        nds_displacement = False
        if last_high and current_price > last_high.price:
            trend = MarketTrend.UPTREND
            nds_displacement = True
        elif last_low and current_price < last_low.price:
            trend = MarketTrend.DOWNTREND
            nds_displacement = True

        bos, choch, bos_choch_confidence, bos_choch_context = self._detect_bos_choch(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_price,
            trend=trend,
            trend_strength=trend_strength,
            volume_analysis=volume_analysis,
            volatility_state=normalized_volatility,
        )
        if bos_choch_context.get("evidence") is not None:
            bos_choch_context["evidence"]["adx"] = adx_value

        trend_switch_conf = float(self.GOLD_SETTINGS.get("SMC_TREND_SWITCH_CONF", 0.65))
        if bos == "BULLISH_BOS" and bos_choch_confidence >= trend_switch_conf:
            trend = MarketTrend.UPTREND
            trend_confidence = max(trend_confidence, bos_choch_confidence)
        elif bos == "BEARISH_BOS" and bos_choch_confidence >= trend_switch_conf:
            trend = MarketTrend.DOWNTREND
            trend_confidence = max(trend_confidence, bos_choch_confidence)
        elif choch == "BULLISH_CHOCH" and bos_choch_confidence >= trend_switch_conf:
            trend = MarketTrend.UPTREND
            trend_confidence = max(trend_confidence, bos_choch_confidence * 0.9)
        elif choch == "BEARISH_CHOCH" and bos_choch_confidence >= trend_switch_conf:
            trend = MarketTrend.DOWNTREND
            trend_confidence = max(trend_confidence, bos_choch_confidence * 0.9)

        self._last_trend = trend
        self._last_trend_confidence = trend_confidence

        range_width, range_mid = None, None
        if last_high and last_low:
            range_width = abs(last_high.price - last_low.price)
            range_mid = (last_high.price + last_low.price) / 2

            min_range = self.atr * 0.5
            if range_width < min_range:
                range_width = None

        structure_score, structure_score_breakdown = self._calculate_structure_score(
            bos=bos,
            choch=choch,
            confidence=bos_choch_confidence,
            trend_strength=trend_strength,
            trend_confidence=trend_confidence,
            volume_analysis=volume_analysis,
            volatility_state=normalized_volatility,
            range_width=range_width,
            last_high=last_high,
            last_low=last_low,
            adx_value=adx_value,
            bos_choch_context=bos_choch_context,
        )

        if nds_displacement:
            # بونوس محدود برای جابجایی معتبر نودها، بدون تغییر BOS برای جلوگیری از سیگنال‌های کاذب.
            structure_score = min(100.0, structure_score + 10.0)
            if isinstance(structure_score_breakdown, dict):
                structure_score_breakdown["nds_displacement_bonus"] = 10.0

        structure_score = max(0.0, min(100.0, structure_score))
        if isinstance(structure_score_breakdown, dict):
            structure_score_breakdown["final_score"] = structure_score
        volume_payload = dict(volume_analysis) if volume_analysis else None
        if volume_payload is not None:
            volume_payload.setdefault("nds_displacement", nds_displacement)

        structure = MarketStructure(
            trend=trend,
            bos=bos,
            choch=choch,
            last_high=last_high,
            last_low=last_low,
            current_price=current_price,
            range_width=range_width,
            range_mid=range_mid,
            bos_choch_confidence=bos_choch_confidence,
            volume_analysis=volume_payload,
            volatility_state=normalized_volatility,
            adx_value=adx_value,
            structure_score=structure_score,
        )

        try:
            structure.trend_confidence = round(trend_confidence * 100.0, 1)
            structure.bos_choch_context = bos_choch_context
            structure.structure_score_breakdown = structure_score_breakdown
        except Exception:
            pass

        try:
            breakers = self._find_breaker_blocks(self.df, swings)
            inversion_fvgs = self._find_inversion_fvgs(self.df)
            setattr(structure, "breakers", breakers)
            setattr(structure, "inversion_fvgs", inversion_fvgs)
        except Exception as e:
            self._log_debug("[NDS][SMC][FLOW_ZONES] detection failed: %s", e)

        self._log_info(
            "[NDS][SMC][STRUCTURE] Trend=%s BOS=%s CHOCH=%s Conf=%.1f%% TrendConf=%.1f%% Score=%.1f",
            trend.value,
            bos,
            choch,
            bos_choch_confidence * 100,
            trend_confidence * 100,
            structure_score,
        )

        return structure

    def _calculate_structure_score(
        self,
        bos: str,
        choch: str,
        confidence: float,
        trend_strength: float,
        trend_confidence: float,
        volume_analysis: Optional[Dict],
        volatility_state: Optional[str],
        range_width: Optional[float],
        last_high: Optional[SwingPoint] = None,
        last_low: Optional[SwingPoint] = None,
        adx_value: Optional[float] = None,
        sweeps: Optional[List[LiquiditySweep]] = None,
        bos_choch_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        محاسبه امتیاز کیفیت ساختار - نسخه بهینه شده برای اسکلپینگ چابک طلا
        """
        # NOTE: این تابع یکی از حساس‌ترین نقاط تولید Score است.
        # برای traceability، بدون تغییر منطق امتیازدهی، breakdown هر جزء را track می‌کنیم.
        score = 0.0
        _parts: Dict[str, float] = {}
        def _add_part(name: str, delta: float) -> None:
            """Internal helper for INFO-level score breakdown (must not affect scoring)."""
            try:
                _parts[name] = float(_parts.get(name, 0.0)) + float(delta)
            except Exception:
                pass
        current_price = float(self.df['close'].iloc[-1])

        adx_threshold = self.GOLD_SETTINGS.get('ADX_THRESHOLD_WEAK', 15)

        score += 5.0
        _add_part("base", 5.0)

        if bos != "NONE":
            bos_component = 40 * confidence
            score += bos_component
            _add_part("bos_component", bos_component)
        elif choch != "NONE":
            choch_component = 32 * confidence
            score += choch_component
            _add_part("choch_component", choch_component)

        penetration_bonus = 0.0
        disp_atr = 0.0
        fakeout_risk = 0.0
        if bos_choch_context and bos_choch_context.get("evidence"):
            evidence = bos_choch_context.get("evidence", {})
            bos_ev = evidence.get("bos_evidence") or {}
            choch_ev = evidence.get("choch_evidence") or {}
            disp_atr = max(
                self._safe_float(bos_ev.get("disp_atr", 0.0), 0.0),
                self._safe_float(choch_ev.get("disp_atr", 0.0), 0.0),
            )
            fakeout_risk = max(
                self._safe_float(bos_ev.get("fakeout_risk", 0.0), 0.0),
                self._safe_float(choch_ev.get("fakeout_risk", 0.0), 0.0),
            )

        if last_high and current_price > last_high.price:
            penetration_bonus = 10.0 * (1.0 if confidence > 0.5 else 0.6)
            score += penetration_bonus
        elif last_low and current_price < last_low.price:
            penetration_bonus = 10.0 * (1.0 if confidence > 0.5 else 0.6)
            score += penetration_bonus
        if penetration_bonus:
            _add_part("penetration_bonus", penetration_bonus)

        if adx_value is not None:
            adx_component = 0.0
            if adx_value > 25:
                adx_component = 10.0
            elif adx_value > adx_threshold:
                adx_component = 5.0
            else:
                adx_component = -8.0
            score += adx_component
            _add_part("adx_component", adx_component)

        trend_score = 12 * trend_strength
        score += trend_score
        _add_part("trend_component", trend_score)

        trend_conf_component = 10.0 * min(1.0, max(0.0, trend_confidence))
        score += trend_conf_component
        _add_part("trend_conf_component", trend_conf_component)

        if volume_analysis:
            rvol = self._safe_rvol(volume_analysis)
            volume_factor = self._volume_factor(rvol)
            volume_component = 10.0 * (volume_factor - 0.8)
            score += volume_component
            _add_part("volume_component", volume_component)
            _parts["volume_factor"] = volume_factor

        vol_component = 0.0
        if volatility_state == "MODERATE_VOLATILITY":
            vol_component = 8.0
        elif volatility_state == "HIGH_VOLATILITY":
            vol_component = -8.0
        elif volatility_state == "LOW_VOLATILITY":
            vol_component = -4.0
        score += vol_component
        _add_part("volatility_component", vol_component)

        if range_width and hasattr(self, 'atr') and self.atr > 0:
            atr_ratio = range_width / self.atr
            range_component = 0.0
            if atr_ratio < 1.0:
                range_component = -8.0
            elif atr_ratio > 1.5:
                range_component = 8.0
            score += range_component
            _add_part("range_component", range_component)

        if last_high is not None:
            score += 2.5
            _add_part("swing_high_component", 2.5)
        if last_low is not None:
            score += 2.5
            _add_part("swing_low_component", 2.5)

        if disp_atr:
            displacement_component = 10.0 * min(1.0, disp_atr / 1.2)
            score += displacement_component
            _add_part("disp_component", displacement_component)

        if fakeout_risk:
            fakeout_component = -12.0 * fakeout_risk
            score += fakeout_component
            _add_part("fakeout_component", fakeout_component)

        if sweeps:
            sweep_total = 0.0
            for sweep in sweeps:
                if sweep.type == 'BULLISH_SWEEP':
                    delta = 8.0 * sweep.strength
                    score += delta
                    sweep_total += delta
                elif sweep.type == 'BEARISH_SWEEP':
                    delta = -8.0 * sweep.strength
                    score += delta
                    sweep_total += delta
            _add_part("sweeps_component", sweep_total)

        final_score = max(0.0, min(100.0, score))
        # clamp delta helps identify if score is saturating
        _add_part("clamp_delta", final_score - score)
        breakdown = {
            "parts": _parts,
            "raw_score": score,
            "final_score": final_score,
            "bos": bos,
            "choch": choch,
            "confidence": confidence,
            "trend_strength": trend_strength,
            "trend_confidence": trend_confidence,
            "disp_atr": disp_atr,
            "fakeout_risk": fakeout_risk,
            "volume_factor": _parts.get("volume_factor", 1.0),
        }

        # INFO breakdown log (single-line, parse-friendly)
        try:
            self._log_info(
                "[NDS][SMC][STRUCTURE_SCORE] bos=%s choch=%s conf=%.3f trend_strength=%.3f vol_state=%s "
                "parts(base=%.2f bos=%.2f choch=%.2f pen=%.2f adx=%.2f trend=%.2f trend_conf=%.2f vol=%.2f "
                "vzone=%.2f range=%.2f swh=%.2f swl=%.2f disp=%.2f fakeout=%.2f sweeps=%.2f clamp=%.2f) raw=%.2f final=%.2f",
                str(bos),
                str(choch),
                float(confidence),
                float(trend_strength),
                str(volatility_state),
                float(_parts.get("base", 0.0)),
                float(_parts.get("bos_component", 0.0)),
                float(_parts.get("choch_component", 0.0)),
                float(penetration_bonus),
                float(_parts.get("adx_component", 0.0)),
                float(_parts.get("trend_component", 0.0)),
                float(_parts.get("trend_conf_component", 0.0)),
                float(_parts.get("volatility_component", 0.0)),
                float(_parts.get("volume_component", 0.0)),
                float(_parts.get("range_component", 0.0)),
                float(_parts.get("swing_high_component", 0.0)),
                float(_parts.get("swing_low_component", 0.0)),
                float(_parts.get("disp_component", 0.0)),
                float(_parts.get("fakeout_component", 0.0)),
                float(_parts.get("sweeps_component", 0.0)),
                float(_parts.get("clamp_delta", 0.0)),
                float(score),
                float(final_score),
            )
        except Exception:
            pass

        return round(final_score, 2), breakdown

    def _get_relevant_swings(self, major_swings: List[SwingPoint], lookback: int) -> List[SwingPoint]:
        """انتخاب سوینگ‌های مرتبط"""
        if len(major_swings) <= lookback:
            return major_swings

        recent_by_time = []
        last_time = self.df['time'].iloc[-1]

        for swing in reversed(major_swings):
            time_diff = (last_time - swing.time).total_seconds() / 3600
            if time_diff <= 24:
                recent_by_time.append(swing)
            if len(recent_by_time) >= lookback * 2:
                break

        if recent_by_time:
            recent_by_time.sort(key=lambda x: x.time, reverse=True)
            return recent_by_time[:lookback]

        return major_swings[-lookback:]

    def _determine_trend_with_confidence(
        self,
        swings: List[SwingPoint],
        current_price: float,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
        adx_value: Optional[float] = None,
    ) -> Tuple[MarketTrend, float, float]:
        """تشخیص روند با اطمینان بر اساس چندین فاکتور"""
        if len(swings) < 2:
            return MarketTrend.RANGING, 0.0, 0.0

        highs = [s for s in swings if s.side == 'HIGH']
        lows = [s for s in swings if s.side == 'LOW']

        if len(highs) < 2 or len(lows) < 2:
            return MarketTrend.RANGING, 0.0, 0.0

        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i - 1].price)
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i - 1].price)
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i - 1].price)
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i - 1].price)

        total_pairs = max(1, (len(highs) + len(lows) - 2))
        bullish_score = (higher_highs + higher_lows) / total_pairs
        bearish_score = (lower_highs + lower_lows) / total_pairs

        if adx_value is not None:
            adx_strength = min(1.0, max(0.0, adx_value / 50.0))
        else:
            adx_strength = 0.0

        rvol = self._safe_rvol(volume_analysis)
        vol_factor = self._volume_factor(rvol)

        if volatility_state == "HIGH_VOLATILITY":
            vol_state_factor = 0.9
        elif volatility_state == "LOW_VOLATILITY":
            vol_state_factor = 0.95
        else:
            vol_state_factor = 1.05

        bullish_conf = min(1.0, (0.55 * bullish_score + 0.35 * adx_strength) * vol_factor * vol_state_factor)
        bearish_conf = min(1.0, (0.55 * bearish_score + 0.35 * adx_strength) * vol_factor * vol_state_factor)

        if bullish_conf > bearish_conf:
            trend = MarketTrend.UPTREND
            confidence = bullish_conf
        elif bearish_conf > bullish_conf:
            trend = MarketTrend.DOWNTREND
            confidence = bearish_conf
        else:
            trend = MarketTrend.RANGING
            confidence = 0.3

        switch_threshold = float(self.GOLD_SETTINGS.get("SMC_TREND_SWITCH_CONF", 0.65))
        hold_threshold = float(self.GOLD_SETTINGS.get("SMC_TREND_HOLD_CONF", 0.5))

        if self._last_trend in {MarketTrend.UPTREND, MarketTrend.DOWNTREND}:
            if trend != self._last_trend and confidence < switch_threshold:
                trend = self._last_trend
                confidence = max(confidence, self._last_trend_confidence * 0.9)
            elif trend == MarketTrend.RANGING and self._last_trend_confidence >= hold_threshold:
                trend = self._last_trend
                confidence = max(confidence, self._last_trend_confidence * 0.85)

        trend_strength = confidence
        self._last_trend = trend
        self._last_trend_confidence = confidence

        self._log_info(
            "[NDS][SMC][TREND] trend=%s conf=%.3f bull=%.3f bear=%.3f adx=%.2f rvol=%.2f vol_state=%s",
            trend.value,
            confidence,
            bullish_conf,
            bearish_conf,
            adx_strength,
            rvol,
            str(volatility_state),
        )
        return trend, trend_strength, min(1.0, confidence)

    def _detect_bos_choch(
        self,
        last_high: Optional[SwingPoint],
        last_low: Optional[SwingPoint],
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        trend_strength: float,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        تشخیص BOS/CHOCH با تأیید چندمرحله‌ای
        """
        bos = "NONE"
        choch = "NONE"
        confidence = 0.0
        context: Dict[str, Any] = {"evidence": {}, "decision": {}}

        if not last_high or not last_low:
            self._log_debug("[NDS][SMC][BOS_CHOCH] insufficient swings")
            return bos, choch, confidence, context

        base_buffer = self._calculate_dynamic_buffer(
            atr=self.atr,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            volume_analysis=volume_analysis,
        )

        bos, bos_confidence, bos_evidence = self._detect_bos_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            volume_analysis=volume_analysis,
        )

        choch, choch_confidence, choch_evidence = self._detect_choch_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            bos_detected=(bos != "NONE"),
            volume_analysis=volume_analysis,
        )

        final_bos, final_choch, final_confidence = self._validate_with_price_action(
            bos=bos,
            choch=choch,
            bos_confidence=bos_confidence,
            choch_confidence=choch_confidence,
            bos_evidence=bos_evidence,
            choch_evidence=choch_evidence,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            last_high_price=last_high.price,
            last_low_price=last_low.price,
            df=self.df,
        )

        context["evidence"] = {
            "bos_confidence": bos_confidence,
            "choch_confidence": choch_confidence,
            "final_confidence": final_confidence,
            "rvol": self._safe_rvol(volume_analysis),
            "bos_evidence": bos_evidence,
            "choch_evidence": choch_evidence,
            "volatility_state": volatility_state,
        }
        context["decision"] = {
            "bos": final_bos,
            "choch": final_choch,
        }

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] result bos=%s choch=%s conf=%.2f",
            final_bos,
            final_choch,
            final_confidence,
        )
        return final_bos, final_choch, final_confidence, context

    def _calculate_dynamic_buffer(
        self,
        atr: float,
        trend_strength: float,
        volatility_state: Optional[str],
        volume_analysis: Optional[Dict],
    ) -> Dict[str, float]:
        """محاسبه بافر پویا بر اساس شرایط مختلف بازار"""
        buffers = {
            'bos': atr * 0.15,
            'choch': atr * 0.12,
            'aggressive': atr * 0.08,
            'conservative': atr * 0.2,
        }

        if trend_strength > 0.7:
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.7
        elif trend_strength < 0.3:
            buffers['bos'] *= 1.5
            buffers['choch'] *= 1.3

        if volatility_state == "HIGH_VOLATILITY":
            buffers['bos'] *= 1.2
            buffers['choch'] *= 1.1
        elif volatility_state == "LOW_VOLATILITY":
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.9

        if volume_analysis:
            volume_zone = volume_analysis.get('volume_zone') or volume_analysis.get('zone')
            if volume_zone == "HIGH":
                buffers['bos'] *= 0.9
                buffers['choch'] *= 0.85

        return buffers

    def _confirm_with_candle_pattern(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        last_high_price: float,
        last_low_price: float,
        trend: MarketTrend,
    ) -> bool:
        """تأیید شکست با الگوهای کندل استیک"""
        candle_size = abs(current_high - current_low)
        current_open = float(self.df['open'].iloc[-1])
        if pd.isna(current_open):
            return False
        body_size = abs(current_close - current_open)

        if trend == MarketTrend.UPTREND:
            if current_close > last_high_price and (current_close - last_high_price) > (candle_size * 0.3):
                if body_size > (candle_size * 0.4):
                    return True

        elif trend == MarketTrend.DOWNTREND:
            if current_close < last_low_price and (last_low_price - current_close) > (candle_size * 0.3):
                if body_size > (candle_size * 0.4):
                    return True

        return False

    def _check_reversal_patterns(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        pattern_type: str,
    ) -> bool:
        """بررسی الگوهای بازگشتی کندلی"""
        try:
            current_candle = self.df.iloc[-1]
            prev_candle = self.df.iloc[-2]

            current_open = current_candle['open']
            prev_open = prev_candle['open']
            prev_close = prev_candle['close']
            prev_high = prev_candle['high']
            prev_low = prev_candle['low']

            current_body = abs(current_close - current_open)
            prev_body = abs(prev_close - prev_open)
            current_range = current_high - current_low
            prev_range = prev_high - prev_low

            if pattern_type == "bullish":
                if current_low < prev_low and current_close > (current_open + (current_range * 0.6)):
                    return True

                if current_close > prev_open and current_open < prev_close and current_body > (prev_body * 1.5):
                    return True

            elif pattern_type == "bearish":
                if current_high > prev_high and current_close < (current_open - (current_range * 0.6)):
                    return True

                if current_close < prev_open and current_open > prev_close and current_body > (prev_body * 1.5):
                    return True

        except (IndexError, KeyError):
            pass

        return False

    def _calculate_bearish_pressure(self, recent_candles: pd.DataFrame) -> float:
        """محاسبه فشار فروش در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0

        bearish_count = 0
        total_candles = len(recent_candles)

        for _, candle in recent_candles.iterrows():
            if candle['close'] < candle['open']:
                bearish_count += 1

        return bearish_count / total_candles

    def _calculate_bullish_pressure(self, recent_candles: pd.DataFrame) -> float:
        """محاسبه فشار خرید در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0

        bullish_count = 0
        total_candles = len(recent_candles)

        for _, candle in recent_candles.iterrows():
            if candle['close'] > candle['open']:
                bullish_count += 1

        return bullish_count / total_candles

    def _detect_bos_advanced(
        self,
        last_high: SwingPoint,
        last_low: SwingPoint,
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        base_buffer: Dict[str, float],
        volume_analysis: Optional[Dict],
    ) -> Tuple[str, float, Dict[str, Any]]:
        """تشخیص پیشرفته BOS با امتیازدهی نرم و بدون گیت سخت حجم"""
        bos = "NONE"
        confidence = 0.0
        evidence: Dict[str, Any] = {"candidate": None}

        last_high_price = last_high.price
        last_low_price = last_low.price

        price_break = False
        price_signal = ""

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] BOS check trend=%s last_high=%.2f last_low=%.2f close=%.2f buffer=%.4f",
            trend.value,
            last_high_price,
            last_low_price,
            current_close,
            base_buffer.get('bos', 0.0),
        )

        if trend == MarketTrend.UPTREND:
            if current_close > (last_high_price + base_buffer['bos']):
                price_break = True
                price_signal = "BULLISH_BOS"
        elif trend == MarketTrend.DOWNTREND:
            if current_close < (last_low_price - base_buffer['bos']):
                price_break = True
                price_signal = "BEARISH_BOS"

        candle_confirmation = self._confirm_with_candle_pattern(
            current_high,
            current_low,
            current_close,
            last_high_price,
            last_low_price,
            trend,
        )

        if price_break:
            break_level = last_high_price if price_signal == "BULLISH_BOS" else last_low_price
            displacement, disp_atr = self._compute_displacement_atr(current_close, break_level)
            rvol = self._safe_rvol(volume_analysis)
            volume_factor = self._volume_factor(rvol)
            candle_metrics = self._extract_candle_metrics()
            fakeout_risk = self._compute_fakeout_risk(
                break_level,
                "BULLISH" if price_signal == "BULLISH_BOS" else "BEARISH",
            )

            displacement_score = min(0.25, 0.15 + 0.2 * min(1.0, disp_atr / 1.2))
            candle_score = 0.15 if candle_confirmation else 0.05
            body_score = min(0.15, max(0.0, (candle_metrics["body_ratio"] - 0.25) * 0.4))
            volume_score = min(0.2, 0.1 + (volume_factor - 0.8) * 0.25)

            confidence = 0.35 + displacement_score + candle_score + body_score + volume_score
            confidence *= max(0.6, 1.0 - fakeout_risk * 0.7)
            confidence = min(1.0, max(0.0, confidence))

            min_keep = float(self.GOLD_SETTINGS.get("SMC_BOS_MIN_KEEP", 0.18))
            min_confirm = float(self.GOLD_SETTINGS.get("SMC_MIN_CONFIRM_CONF", 0.25))
            low_conf_reject = confidence < min_confirm and fakeout_risk > 0.6

            if confidence >= min_keep or (confidence >= min_confirm and fakeout_risk <= 0.6):
                bos = price_signal
            elif low_conf_reject:
                bos = "NONE"

            evidence = {
                "candidate": price_signal,
                "break_level": break_level,
                "displacement": displacement,
                "disp_atr": disp_atr,
                "rvol": rvol,
                "volume_factor": volume_factor,
                "body_ratio": candle_metrics["body_ratio"],
                "upper_wick": candle_metrics["upper_wick"],
                "lower_wick": candle_metrics["lower_wick"],
                "fakeout_risk": fakeout_risk,
                "candle_confirmation": candle_confirmation,
                "confidence": confidence,
                "kept": bos != "NONE",
            }

            self._log_info(
                "[NDS][SMC][BOS] candidate=%s conf=%.3f disp_atr=%.3f rvol=%.2f vol_factor=%.2f "
                "body_ratio=%.2f fakeout=%.2f candle_ok=%s",
                price_signal,
                confidence,
                disp_atr,
                rvol,
                volume_factor,
                candle_metrics["body_ratio"],
                fakeout_risk,
                candle_confirmation,
            )
        else:
            evidence = {"candidate": None}

        return bos, confidence, evidence

    def _detect_choch_advanced(
        self,
        last_high: SwingPoint,
        last_low: SwingPoint,
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        base_buffer: Dict[str, float],
        bos_detected: bool,
        volume_analysis: Optional[Dict] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """تشخیص پیشرفته CHOCH - حساس به تغییر روند"""
        choch = "NONE"
        confidence = 0.0
        evidence: Dict[str, Any] = {"candidate": None}
        candle_ok = False

        if bos_detected:
            return choch, confidence, evidence

        last_high_price = last_high.price
        last_low_price = last_low.price

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] CHOCH check trend=%s last_high=%.2f last_low=%.2f close=%.2f buffer=%.4f",
            trend.value,
            last_high_price,
            last_low_price,
            current_close,
            base_buffer.get('choch', 0.0),
        )

        if trend == MarketTrend.UPTREND:
            if current_close < (last_low_price - base_buffer['choch']):
                candle_ok = self._check_reversal_patterns(current_high, current_low, current_close, "bearish")
                if candle_ok:
                    choch = "BEARISH_CHOCH"
                    confidence = 0.55

        elif trend == MarketTrend.DOWNTREND:
            if current_close > (last_high_price + base_buffer['choch']):
                candle_ok = self._check_reversal_patterns(current_high, current_low, current_close, "bullish")
                if candle_ok:
                    choch = "BULLISH_CHOCH"
                    confidence = 0.55

        elif trend == MarketTrend.RANGING:
            range_buffer = base_buffer['choch'] * 1.5

            if current_close > (last_high_price + range_buffer):
                choch = "BULLISH_CHOCH"
                confidence = 0.5
            elif current_close < (last_low_price - range_buffer):
                choch = "BEARISH_CHOCH"
                confidence = 0.5

        if choch != "NONE":
            candidate_name = choch
            break_level = last_high_price if choch == "BULLISH_CHOCH" else last_low_price
            displacement, disp_atr = self._compute_displacement_atr(current_close, break_level)
            rvol = self._safe_rvol(volume_analysis)
            volume_factor = self._volume_factor(rvol)
            candle_metrics = self._extract_candle_metrics()
            fakeout_risk = self._compute_fakeout_risk(
                break_level,
                "BULLISH" if choch == "BULLISH_CHOCH" else "BEARISH",
            )

            confidence += min(0.2, 0.1 + 0.2 * min(1.0, disp_atr / 1.1))
            confidence *= max(0.6, 1.0 - fakeout_risk * 0.6)
            confidence *= volume_factor
            confidence = min(1.0, max(0.0, confidence))

            min_keep = float(self.GOLD_SETTINGS.get("SMC_CHOCH_MIN_KEEP", 0.2))
            min_confirm = float(self.GOLD_SETTINGS.get("SMC_MIN_CONFIRM_CONF", 0.25))
            low_conf_reject = confidence < min_confirm and fakeout_risk > 0.6

            if confidence < min_keep and low_conf_reject:
                choch = "NONE"

            evidence = {
                "candidate": candidate_name,
                "break_level": break_level,
                "displacement": displacement,
                "disp_atr": disp_atr,
                "rvol": rvol,
                "volume_factor": volume_factor,
                "body_ratio": candle_metrics["body_ratio"],
                "upper_wick": candle_metrics["upper_wick"],
                "lower_wick": candle_metrics["lower_wick"],
                "fakeout_risk": fakeout_risk,
                "confidence": confidence,
                "candle_confirmation": candle_ok,
                "kept": choch != "NONE",
            }

            self._log_info(
                "[NDS][SMC][CHOCH] candidate=%s conf=%.3f disp_atr=%.3f rvol=%.2f vol_factor=%.2f "
                "body_ratio=%.2f fakeout=%.2f candle_ok=%s",
                choch,
                confidence,
                disp_atr,
                rvol,
                volume_factor,
                candle_metrics["body_ratio"],
                fakeout_risk,
                candle_ok,
            )

        return choch, confidence, evidence

    def _validate_with_price_action(
self,
        bos: str,
        choch: str,
        bos_confidence: float,
        choch_confidence: float,
        current_high: float,
        current_low: float,
        current_close: float,
        last_high_price: float,
        last_low_price: float,
        df: pd.DataFrame,
        bos_evidence: Optional[Dict[str, Any]] = None,
        choch_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, float]:
        """اعتبارسنجی نهایی با پرایس اکشن چندکندلی"""
        final_bos = bos
        final_choch = choch
        final_confidence = max(bos_confidence, choch_confidence)

        if bos != "NONE" or choch != "NONE":
            recent_candles = df.iloc[-4:-1]

            if bos == "BULLISH_BOS":
                bearish_pressure = self._calculate_bearish_pressure(recent_candles)
                if bearish_pressure > 0.7:
                    self._log_debug("[NDS][SMC][BOS_CHOCH] Bullish BOS high bearish pressure")
                    final_confidence *= 0.7

            elif bos == "BEARISH_BOS":
                bullish_pressure = self._calculate_bullish_pressure(recent_candles)
                if bullish_pressure > 0.7:
                    self._log_debug("[NDS][SMC][BOS_CHOCH] Bearish BOS high bullish pressure")
                    final_confidence *= 0.7

        min_reject = float(self.GOLD_SETTINGS.get("SMC_MIN_CONFIRM_CONF", 0.25))
        fakeout_risk = 0.0
        if bos_evidence:
            fakeout_risk = max(fakeout_risk, self._safe_float(bos_evidence.get("fakeout_risk", 0.0), 0.0))
        if choch_evidence:
            fakeout_risk = max(fakeout_risk, self._safe_float(choch_evidence.get("fakeout_risk", 0.0), 0.0))

        if final_confidence < min_reject and fakeout_risk > 0.6:
            final_bos = "NONE"
            final_choch = "NONE"
        elif final_confidence < min_reject:
            self._log_debug(
                "[NDS][SMC][BOS_CHOCH] weak confirmation conf=%.2f fakeout=%.2f (kept weak)",
                final_confidence,
                fakeout_risk,
            )

        return final_bos, final_choch, final_confidence

    def analyze_premium_discount(self, structure: MarketStructure) -> Tuple[str, float]:
        """تحلیل مناطق Premium/Discount"""
        if not structure.last_high or not structure.last_low:
            return "NEUTRAL", 0.0

        if structure.trend == MarketTrend.RANGING:
            range_high = structure.last_high.price
            range_low = structure.last_low.price

            if range_high <= range_low:
                return "NEUTRAL", 0.0

            range_mid = (range_high + range_low) / 2
            current_price = structure.current_price

            discount_zone = range_low + (range_high - range_low) * 0.3
            premium_zone = range_low + (range_high - range_low) * 0.7

            if current_price < discount_zone:
                return "DISCOUNT", range_mid
            if current_price > premium_zone:
                return "PREMIUM", range_mid
            return "EQUILIBRIUM", range_mid

        range_high = structure.last_high.price
        range_low = structure.last_low.price
        range_mid = (range_high + range_low) / 2

        discount_zone = range_low + (range_high - range_low) * 0.33
        premium_zone = range_low + (range_high - range_low) * 0.66

        current_price = structure.current_price

        if current_price < discount_zone:
            return "DISCOUNT", range_mid
        if current_price > premium_zone:
            return "PREMIUM", range_mid
        return "EQUILIBRIUM", range_mid

    def analyze_range_position_gold(self, structure: MarketStructure) -> float:
        """تحلیل موقعیت قیمت در رنج مخصوص بازار طلا"""
        if not structure.range_width or structure.range_width < self.atr:
            return 0.0

        current_price = structure.current_price
        range_low = structure.last_low.price
        range_high = structure.last_high.price

        position = (current_price - range_low) / structure.range_width
        last_candle = self.df.iloc[-1]
        candle_range = last_candle['high'] - last_candle['low']

        score = 0.0

        if position < 0.3:
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            if lower_wick > candle_range * 0.4:
                score += 25
            elif lower_wick > candle_range * 0.25:
                score += 15
            else:
                score += 8

        elif position > 0.7:
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            if upper_wick > candle_range * 0.4:
                score -= 25
            elif upper_wick > candle_range * 0.25:
                score -= 15
            else:
                score -= 8

        return score

    def get_market_trend(self, swings: List[SwingPoint]) -> MarketTrend:
        """
        نسخه ارتقا یافته برای تشخیص سریع‌تر تغییر روند در اسکلپینگ
        """
        if len(swings) < 4:
            return MarketTrend.RANGING

        last_price = self.df['close'].iloc[-1]
        high_swings = [s for s in swings if s.side == 'HIGH']
        low_swings = [s for s in swings if s.side == 'LOW']

        if not high_swings or not low_swings:
            return MarketTrend.RANGING

        last_high = high_swings[-1]
        last_low = low_swings[-1]
        prev_high = high_swings[-2] if len(high_swings) > 1 else last_high
        prev_low = low_swings[-2] if len(low_swings) > 1 else last_low

        if last_price > last_high.price:
            return MarketTrend.UPTREND

        if last_price < last_low.price:
            return MarketTrend.DOWNTREND

        is_hh = last_high.price > prev_high.price
        is_hl = last_low.price > prev_low.price
        is_lh = last_high.price < prev_high.price
        is_ll = last_low.price < prev_low.price

        if is_hh or (is_hl and last_price > last_low.price):
            return MarketTrend.UPTREND

        if is_ll or (is_lh and last_price < last_high.price):
            return MarketTrend.DOWNTREND

        return MarketTrend.RANGING
