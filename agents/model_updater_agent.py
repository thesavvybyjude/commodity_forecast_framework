"""
P3-T1: ModelUpdaterAgent
The heart of continuous learning — routes each commodity's data to the
appropriate group update function daily.
"""
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.model_version_control_agent import ModelVersionControlAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.performance_monitor_agent import PerformanceMonitorAgent
from models.rollback_guard import RollbackGuard

logger = logging.getLogger(__name__)


class ModelUpdaterAgent:
    """Dispatches daily model updates across all 5 experimental groups."""

    def __init__(self):
        self.mvc = ModelVersionControlAgent()
        self.trainer = ModelTrainerAgent()
        self.monitor = PerformanceMonitorAgent()
        self.rollback = RollbackGuard()

    def update_all(self, features: pd.DataFrame, feat_cols: list):
        """Loop over each commodity and dispatch to 5 group methods."""
        logger.info("Starting model updates for all groups")

        numeric_feat_cols = [
            c for c in feat_cols
            if c in features.columns
            and features[c].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        for commodity in config.COMMODITIES:
            commodity_data = features[features['commodity'] == commodity] if 'commodity' in features.columns else features

            if len(commodity_data) == 0:
                logger.warning("No data for commodity: %s", commodity)
                continue

            X = commodity_data[numeric_feat_cols].values
            y = commodity_data['price_ngn'].values if 'price_ngn' in commodity_data.columns else X[:, 0]

            logger.info("Updating models for %s (%d samples)", commodity, len(X))

            self._update_group_A(X, y, commodity)
            self._update_group_B(X, y, commodity)
            self._update_group_C(X, y, commodity)
            self._update_group_D(X, y, commodity)
            self._update_group_E(X, y, commodity)

    def _update_group_A(self, X: np.ndarray, y: np.ndarray, commodity: str):
        """Group A: LSTM Periodic Retrain (every 7 days)."""
        days = self.mvc.days_since_last_train('lstm', 'A')
        if days < config.PERIODIC_RETRAIN_DAYS:
            logger.info("Group A: Skip — only %d days since last train", days)
            return

        logger.info("Group A: Full LSTM retrain (due after %d days)", days)
        model = self.trainer.train_lstm(X, y, 'A')
        if model is not None:
            self.rollback.check_lstm(model, X, y, 'A', commodity)

    def _update_group_B(self, X: np.ndarray, y: np.ndarray, commodity: str):
        """Group B: LSTM Fine-Tune (daily, if enough new samples)."""
        if len(X) < config.FINETUNE_MIN_SAMPLES:
            logger.info("Group B: Skip — insufficient samples (%d < %d)",
                       len(X), config.FINETUNE_MIN_SAMPLES)
            return

        model_path = self.mvc.get_latest('lstm', 'B')
        if model_path is None:
            logger.warning("Group B: No existing model — training from scratch")
            self.trainer.train_lstm(X, y, 'B')
            return

        import tensorflow as tf
        from models.lstm_finetuner import finetune

        logger.info("Group B: Fine-tuning LSTM")
        model = tf.keras.models.load_model(model_path)
        model = finetune(model, X, y)
        self.mvc.save_lstm(model, 'B')
        self.rollback.check_lstm(model, X, y, 'B', commodity)

    def _update_group_C(self, X: np.ndarray, y: np.ndarray, commodity: str):
        """Group C: XGBoost Periodic Retrain (every 7 days)."""
        days = self.mvc.days_since_last_train('xgb', 'C')
        if days < config.PERIODIC_RETRAIN_DAYS:
            logger.info("Group C: Skip — only %d days since last train", days)
            return

        logger.info("Group C: Full XGBoost retrain (due after %d days)", days)
        self.trainer.train_xgboost(X, y, 'C')

    def _update_group_D(self, X: np.ndarray, y: np.ndarray, commodity: str):
        """Group D: XGBoost Warm-Start Fine-Tune (daily)."""
        if len(X) < config.FINETUNE_MIN_SAMPLES:
            logger.info("Group D: Skip — insufficient samples (%d < %d)",
                       len(X), config.FINETUNE_MIN_SAMPLES)
            return

        model_path = self.mvc.get_latest('xgb', 'D')
        if model_path is None:
            logger.warning("Group D: No existing model — training from scratch")
            self.trainer.train_xgboost(X, y, 'D')
            return

        from models.xgboost_finetuner import finetune

        logger.info("Group D: Warm-start fine-tuning XGBoost")
        model = joblib.load(model_path)
        model = finetune(model, X, y)
        self.mvc.save_xgb(model, 'D')

    def _update_group_E(self, X: np.ndarray, y: np.ndarray, commodity: str):
        """Group E: Static Predict Only — never trains, only predicts and logs metrics."""
        model_path = self.mvc.get_latest('xgb', 'E')
        if model_path is None:
            logger.warning("Group E: No model available — cannot predict")
            return

        logger.info("Group E: Static prediction only")
        model = joblib.load(model_path)

        # Predict on last 30 observations
        n = min(30, len(X))
        X_recent = X[-n:]
        y_recent = y[-n:]

        predictions = model.predict(X_recent)
        self.monitor.evaluate_group('E_static', y_recent, predictions, commodity)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("ModelUpdaterAgent ready. Called by AgentOrchestrator.run_daily_pipeline().")
