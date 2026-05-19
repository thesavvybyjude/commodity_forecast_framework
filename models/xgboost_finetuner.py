"""
P3-T5: XGBoost Fine-Tuner
Encapsulates Group D warm-start fine-tune logic as a reusable module.
"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def finetune(model, X_new: np.ndarray, y_new: np.ndarray):
    """Warm-start fine-tune an XGBoost model by adding new trees.

    Uses xgb_model=model.get_booster() to add XGB_FINETUNE_ROUNDS
    new trees on top of the existing booster.
    """
    logger.info(
        "Warm-start fine-tuning XGBoost with %d samples, adding %d trees",
        len(X_new), config.XGB_FINETUNE_ROUNDS,
    )

    model.set_params(n_estimators=model.n_estimators + config.XGB_FINETUNE_ROUNDS)

    model.fit(
        X_new, y_new,
        xgb_model=model.get_booster(),
        eval_set=[(X_new, y_new)],
        verbose=False,
    )

    return model
