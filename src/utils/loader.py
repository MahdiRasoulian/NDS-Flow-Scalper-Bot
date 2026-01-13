# src/utils/loader.py

import sys
import os
import importlib.util
from pathlib import Path

def import_module_safely(module_name, from_paths, project_root):
    """ایمپورت امن ماژول از مسیرهای مختلف"""
    for base_path in from_paths:
        try:
            # ساختار src.trading_bot.module
            module_full_name = f"{base_path}.{module_name}"
            module = __import__(module_full_name, fromlist=[module_name])
            print(f"✅ {module_name} از {base_path} بارگیری شد")
            return module
        except ImportError:
            try:
                # ساختار ساده‌تر فایل فیزیکی
                module_path = str(project_root / base_path / f"{module_name}.py")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.modules[module_name] = module
                    print(f"✅ {module_name} از {module_path} بارگیری شد")
                    return module
            except Exception:
                continue
    return None