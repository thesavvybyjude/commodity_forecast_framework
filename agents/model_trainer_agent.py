"""
P2-T4: ModelTrainerAgent
Trains baseline LSTM and XGBoost models on the full historical dataset.
"""
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.model_version_control_agent import ModelVersionControlAgent

logger = logging.getLogger(__name__)


class ModelTrainerAgent:
    """Trains initial baseline LSTM and XGBoost models."""

    def __init__(self):
        self.mvc = ModelVersionControlAgent()

    def prepare_lstm_sequences(self, X: np.ndarray, y: np.ndarray,
                               seq_length: int = None) -> tuple:
        """Reshape tabular data into (samples, timesteps, features) for LSTM."""
        if seq_length is None:
            seq_length = config.LSTM_SEQUENCE

        Xs, ys = [], []
        for i in range(seq_length, len(X)):
            Xs.append(X[i - seq_length:i])
            ys.append(y[i])

        return np.array(Xs), np.array(ys)

    def train_lstm(self, X: np.ndarray, y: np.ndarray, group: str) -> object:
        """Train an LSTM model for a given experimental group."""
        import tensorflow as tf
        from models.lstm_builder import build_lstm

        logger.info("Training LSTM for group %s on %d samples", group, len(X))

        # Split: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, shuffle=False,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False,
        )

        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_lstm_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_lstm_sequences(X_val, y_val)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            logger.error("Not enough data to create LSTM sequences")
            return None

        n_features = X_train_seq.shape[2]
        model = build_lstm(n_features)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.LSTM_PATIENCE,
            restore_best_weights=True,
        )

        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=config.LSTM_EPOCHS_INIT,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )

        # Evaluate on test set
        X_test_seq, y_test_seq = self.prepare_lstm_sequences(X_test, y_test)
        if len(X_test_seq) > 0:
            test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
            rmse = float(np.sqrt(test_loss[0]) if isinstance(test_loss, list) else np.sqrt(test_loss))
            logger.info("LSTM group %s test RMSE: %.4f", group, rmse)
            metrics = {'rmse': rmse}
        else:
            metrics = {}

        # Save model
        self.mvc.save_lstm(model, group, metrics)
        return model

    def train_xgboost(self, X: np.ndarray, y: np.ndarray, group: str) -> object:
        """Train an XGBoost model for a given experimental group."""
        from models.xgboost_builder import build_xgb

        logger.info("Training XGBoost for group %s on %d samples", group, len(X))

        # Split: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, shuffle=False,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False,
        )

        model = build_xgb()

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        logger.info("XGBoost group %s test RMSE: %.4f, MAE: %.4f", group, rmse, mae)

        metrics = {'rmse': rmse, 'mae': mae}

        # Save model
        self.mvc.save_xgb(model, group, metrics)
        return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("ModelTrainerAgent ready. Use AgentOrchestrator.run_initial_training() to train.")
