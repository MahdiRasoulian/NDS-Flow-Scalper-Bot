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
