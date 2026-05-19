"""
P3-T3: XGBoost Builder
Factory that returns a configured XGBRegressor using XGB_PARAMS from config.
"""
import sys
from pathlib import Path

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_xgb() -> xgb.XGBRegressor:
    """Build a configured XGBRegressor from config parameters."""
    return xgb.XGBRegressor(**config.XGB_PARAMS)
