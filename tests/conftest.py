from __future__ import annotations

import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "MetaTrader5" not in sys.modules:
    mt5_stub = types.ModuleType("MetaTrader5")
    mt5_stub.TRADE_RETCODE_DONE = 10009
    mt5_stub.TRADE_ACTION_DEAL = 1
    mt5_stub.TRADE_ACTION_PENDING = 5
    mt5_stub.ORDER_TIME_GTC = 0
    mt5_stub.ORDER_FILLING_IOC = 1
    mt5_stub.ORDER_FILLING_RETURN = 2
    mt5_stub.ORDER_TYPE_BUY = 0
    mt5_stub.ORDER_TYPE_SELL = 1
    mt5_stub.ORDER_TYPE_BUY_LIMIT = 2
    mt5_stub.ORDER_TYPE_SELL_LIMIT = 3
    mt5_stub.ORDER_TYPE_BUY_STOP = 4
    mt5_stub.ORDER_TYPE_SELL_STOP = 5
    mt5_stub.ORDER_TYPE_BUY_STOP_LIMIT = 6
    mt5_stub.ORDER_TYPE_SELL_STOP_LIMIT = 7
    mt5_stub.positions_get = lambda *args, **kwargs: []
    mt5_stub.order_send = lambda *args, **kwargs: None
    mt5_stub.order_check = lambda *args, **kwargs: None
    mt5_stub.symbol_info = lambda *args, **kwargs: None
    mt5_stub.symbol_info_tick = lambda *args, **kwargs: None
    sys.modules["MetaTrader5"] = mt5_stub

try:
    import requests  # noqa: F401
except Exception:
    requests_stub = types.ModuleType("requests")
    requests_stub.Session = lambda *args, **kwargs: types.SimpleNamespace()
    sys.modules["requests"] = requests_stub

try:
    import dotenv  # noqa: F401
except Exception:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    dotenv_stub.find_dotenv = lambda *args, **kwargs: ""
    sys.modules["dotenv"] = dotenv_stub

try:
    import pandas  # noqa: F401
except Exception:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = type("DataFrame", (), {})
    pandas_stub.Series = type("Series", (), {})
    sys.modules["pandas"] = pandas_stub

try:
    import numpy  # noqa: F401
except Exception:
    sys.modules["numpy"] = types.ModuleType("numpy")

try:
    import ta  # noqa: F401
except Exception:
    ta_stub = types.ModuleType("ta")
    volatility_stub = types.ModuleType("ta.volatility")
    trend_stub = types.ModuleType("ta.trend")
    volume_stub = types.ModuleType("ta.volume")
    volatility_stub.AverageTrueRange = type("AverageTrueRange", (), {})
    trend_stub.ADXIndicator = type("ADXIndicator", (), {})
    volume_stub.VolumeWeightedAveragePrice = type("VolumeWeightedAveragePrice", (), {})
    sys.modules["ta"] = ta_stub
    sys.modules["ta.volatility"] = volatility_stub
    sys.modules["ta.trend"] = trend_stub
    sys.modules["ta.volume"] = volume_stub
