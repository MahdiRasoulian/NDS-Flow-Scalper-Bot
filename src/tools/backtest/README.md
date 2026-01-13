
# NDS Backtest Toolkit (برای پروژه nds scalp trading)

این فولدر یک بک‌تستر «قابل تحلیل و قابل آپتیمایز» برای ربات اسکلپینگ XAUUSD است که **مستقیماً** از همان دو ماژول اصلی پروژه شما استفاده می‌کند:

- آنالایزر: `analyze_gold_market(...)` در `nds/analyzer.py` fileciteturn1file13  
- ریسک منیجر: `ScalpingRiskManager.finalize_order(...)` در `risk_manager.py` fileciteturn1file10  

همان منطق فیلتر/گیت ربات نیز در بک‌تست رعایت شده است (سیگنال BUY/SELL، حداقل confidence و …) که در `bot.py` انجام می‌دهید. fileciteturn1file9  

---

## 1) ساختار خروجی بک‌تست

بعد از اجرا، در فولدر `--out` این فایل‌ها تولید می‌شود:

- `trades.csv` : دفتر معاملات (ورود/خروج/SL/TP/lot/rr/conf/score/notes…)
- `equity_curve.csv` : منحنی اکوئیتی روی هر کندل
- `cycle_log.csv` : لاگ هر سیکل (score/conf/signal/reject_reason/open_positions…)
- `metrics.json` : متریک‌های کلیدی (NetPnL / WinRate / PF / MaxDD / Expectancy…)
- نمودارها:
  - `equity_curve.png`
  - `drawdown.png`
  - `trade_pnl_hist.png`
  - `score_conf.png`

---

## 2) دیتا ورودی (CSV)

یک CSV از کندل‌های MT5 با ستون‌های زیر لازم است:

- `time` (قابل parse با pandas)
- `open, high, low, close`
- `tick_volume` یا `volume`

نمونه نام فایل:
`data/XAUUSD_M15.csv`

---

## 3) نصب و اجرا (داخل ریپوی خودتان)

این فولدر را داخل روت پروژه بگذارید (جایی که `src/` دارید):

```
your_repo/
  src/...
  config/bot_config.json
  tools/backtest/...
```

اجرا:

```bash
python -m tools.backtest.run_backtest \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --out out_bt
```

---

## 4) Override پارامترها (برای آپتیمایز)

هر پارامتر دلخواه را با `--override dotted.path=value` تغییر دهید:

```bash
python -m tools.backtest.run_backtest \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --out out_bt_try1 \
  --override technical_settings.ENTRY_FACTOR=0.22 \
  --override technical_settings.ATR_WINDOW=14 \
  --override technical_settings.ATR_SL_MULTIPLIER=2.2 \
  --override technical_settings.SCALPING_MIN_CONFIDENCE=34
```

---

## 5) Grid Search (آپتیمایز سریع)

در فایل `tools/backtest/optimize.py` یک Grid Search آماده شده.
می‌توانید یک اسکریپت کوتاه بسازید:

```python
import json
import pandas as pd
from tools.backtest.optimize import run_grid_search

df = pd.read_csv("data/XAUUSD_M15.csv")
cfg = json.load(open("config/bot_config.json", "r", encoding="utf-8"))

grid = {
  "technical_settings.ENTRY_FACTOR": [0.15, 0.2, 0.25, 0.3],
  "technical_settings.ATR_SL_MULTIPLIER": [2.0, 2.4, 2.8],
  "technical_settings.SCALPING_MIN_CONFIDENCE": [32, 35, 38],
  "risk_settings.RISK_AMOUNT_USD": [10, 15, 20, 25],
}

results = run_grid_search(df, cfg, grid)
print(results.head(20))
results.to_csv("out_bt/grid_results.csv", index=False, encoding="utf-8-sig")
```

---

## 6) نکته مهم (چرا الان معامله صفر است؟)

در لاگ شما، آنالایزر سیگنال `NONE` داد چون:
- confidence حدود 27.4 بوده و از حداقل 38 کمتر است
- همچنین اگر Entry Zone معتبر (FVG/OB) پیدا نشود، آنالایزر عمداً سیگنال را NONE می‌کند fileciteturn1file4  

پس در بک‌تست هم طبیعی است که در بخشی از بازار معامله کم شود تا زمانی که پارامترها/فیلترها را برای رفتار صحیح تنظیم کنیم.

---

## 7) پیشنهاد مسیر آپتیمایز (پارامترهای اثرگذار)

برای شروع، این‌ها معمولاً بیشترین اثر را دارند:

1) `technical_settings.SCALPING_MIN_CONFIDENCE`  
2) `technical_settings.ENTRY_FACTOR`  
3) `technical_settings.ATR_SL_MULTIPLIER` / `technical_settings.MIN_SL_DISTANCE` / `technical_settings.MAX_SL_DISTANCE`  
4) `risk_settings.MAX_PRICE_DEVIATION_PIPS` و `risk_settings.LIMIT_ORDER_MIN_CONFIDENCE`  
5) وزن سشن‌ها و گیت RVOL (چون در لاگ RVOL=0.62 LOW بود و confidence را downscale می‌کند) fileciteturn1file0  

---

## 8) محدودیت‌های شبیه‌سازی

- Bid/Ask از روی spread ثابت تولید می‌شود (قابل تغییر با `--spread`)
- slippage ثابت است (قابل تغییر با `--slippage`)
- اگر در یک کندل هم‌زمان TP و SL لمس شوند، محافظه‌کارانه SL اول اعمال می‌شود.

این محافظه‌کاری برای جلوگیری از overfit و توهم سود ضروری است.

