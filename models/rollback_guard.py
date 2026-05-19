"""
P3-T6: RollbackGuard
Validates new models against baseline RMSE — auto-rollback if performance degrades.
"""
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.model_version_control_agent import ModelVersionControlAgent

logger = logging.getLogger(__name__)


class RollbackGuard:
    """Validates model updates and rolls back if performance degrades."""

    def __init__(self):
        self.mvc = ModelVersionControlAgent()

    def check_lstm(self, model, X: np.ndarray, y: np.ndarray,
                   group: str, commodity: str) -> bool:
        """Check LSTM model performance; rollback if degraded.

        Returns True if model is acceptable, False if rolled back.
        """
        if len(X) <= config.LSTM_SEQUENCE:
            logger.info("Not enough data to evaluate LSTM — skipping check")
            return True

        # Build sequences and predict
        Xs, ys = [], []
        for i in range(config.LSTM_SEQUENCE, len(X)):
            Xs.append(X[i - config.LSTM_SEQUENCE:i])
            ys.append(y[i])
        X_seq = np.array(Xs)
        y_seq = np.array(ys)

        predictions = model.predict(X_seq, verbose=0).flatten()
        current_rmse = float(np.sqrt(np.mean((y_seq - predictions) ** 2)))

        return self._evaluate_and_rollback(
            'lstm', group, commodity, current_rmse, model,
        )

    def check_xgb(self, model, X: np.ndarray, y: np.ndarray,
                  group: str, commodity: str) -> bool:
        """Check XGBoost model performance; rollback if degraded."""
        predictions = model.predict(X)
        current_rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))

        return self._evaluate_and_rollback(
            'xgb', group, commodity, current_rmse, model,
        )

    def _evaluate_and_rollback(self, model_type: str, group: str,
                                commodity: str, current_rmse: float,
                                model) -> bool:
        """Compare RMSE against baseline and rollback if needed."""
        baseline_rmse = self.mvc.get_baseline_rmse(model_type, group)

        if baseline_rmse is None:
            logger.info("No baseline RMSE for %s_%s — accepting current model", model_type, group)
            self.mvc.update_metrics(
                model_type, group,
                rmse=current_rmse, mae=0, mape=0, r2=0,
            )
            return True

        threshold = baseline_rmse * config.RMSE_DEGRADATION

        if current_rmse > threshold:
            logger.warning(
                "ROLLBACK triggered for %s_%s/%s: "
                "current_rmse=%.4f > threshold=%.4f (baseline=%.4f * %.1f)",
                model_type, group, commodity,
                current_rmse, threshold, baseline_rmse, config.RMSE_DEGRADATION,
            )
            self._rollback(model_type, group)
            return False

        logger.info(
            "Model %s_%s/%s passed check: RMSE=%.4f (threshold=%.4f)",
            model_type, group, commodity, current_rmse, threshold,
        )
        self.mvc.update_metrics(
            model_type, group,
            rmse=current_rmse, mae=0, mape=0, r2=0,
        )
        return True

    def _rollback(self, model_type: str, group: str):
        """Restore the previous version of a model."""
        current_path = self.mvc.get_latest(model_type, group)
        if current_path is None:
            logger.error("Cannot rollback — no current model found")
            return

        prev_path = self._get_previous_version(current_path)
        if prev_path is None or not prev_path.exists():
            logger.warning("No previous version available for rollback")
            return

        # Re-save previous model as current
        if model_type == 'lstm':
            import tensorflow as tf
            old_model = tf.keras.models.load_model(prev_path)
            self.mvc.save_lstm(old_model, group)
        else:
            old_model = joblib.load(prev_path)
            self.mvc.save_xgb(old_model, group)

        logger.info("Rolled back to %s", prev_path)

    def _get_previous_version(self, current_path: Path) -> Path:
        """Construct previous version path by decrementing version number."""
        name = current_path.name
        match = re.search(r'_v(\d+)_', name)
        if not match:
            return None

        version = int(match.group(1))
        if version <= 1:
            return None

        prev_name = name.replace(f'_v{version}_', f'_v{version - 1}_')
        prev_path = current_path.parent / prev_name
        return prev_path
