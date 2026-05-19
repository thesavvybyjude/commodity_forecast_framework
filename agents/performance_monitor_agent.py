"""
P2-T6: PerformanceMonitorAgent
Evaluates model predictions and appends results to metrics_log.csv.
"""
import logging
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class PerformanceMonitorAgent:
    """Computes and logs model performance metrics."""

    def __init__(self):
        self.metrics_path = config.LOG_DIR / "metrics_log.csv"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_group(self, group: str, y_true: np.ndarray,
                       y_pred: np.ndarray, commodity: str) -> dict:
        """Compute all 4 metrics, append to CSV, return row dict."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Compute metrics
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        # MAPE with zero-division guard
        y_safe = np.where(y_true == 0, 1, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_safe)) * 100)

        r2 = float(r2_score(y_true, y_pred))

        row = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'group': group,
            'commodity': commodity,
            'rmse': round(rmse, 6),
            'mae': round(mae, 6),
            'mape': round(mape, 4),
            'r2': round(r2, 6),
            'n_samples': len(y_true),
        }

        # Append to CSV
        self._append_row(row)
        logger.info(
            "Metrics for %s/%s: RMSE=%.4f MAE=%.4f MAPE=%.2f%% R2=%.4f",
            group, commodity, rmse, mae, mape, r2,
        )

        return row

    def _append_row(self, row: dict):
        """Append a single row to the metrics log CSV."""
        df_row = pd.DataFrame([row])

        if self.metrics_path.exists():
            df_row.to_csv(self.metrics_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(self.metrics_path, index=False)

    def load_metrics(self) -> pd.DataFrame:
        """Read full metrics_log.csv as DataFrame."""
        if not self.metrics_path.exists():
            logger.warning("No metrics log found at %s", self.metrics_path)
            return pd.DataFrame(columns=[
                'date', 'group', 'commodity', 'rmse', 'mae', 'mape', 'r2', 'n_samples',
            ])

        return pd.read_csv(self.metrics_path)

    def check_degradation(self, group: str, commodity: str,
                          current_rmse: float, multiplier: float = None) -> bool:
        """Return True if RMSE exceeds baseline * multiplier."""
        if multiplier is None:
            multiplier = config.RMSE_DEGRADATION

        df = self.load_metrics()
        if df.empty:
            return False

        subset = df[(df['group'] == group) & (df['commodity'] == commodity)]
        if subset.empty:
            return False

        baseline_rmse = subset['rmse'].iloc[0]
        threshold = baseline_rmse * multiplier

        if current_rmse > threshold:
            logger.warning(
                "Degradation detected for %s/%s: current=%.4f > threshold=%.4f",
                group, commodity, current_rmse, threshold,
            )
            return True

        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    monitor = PerformanceMonitorAgent()
    df = monitor.load_metrics()
    print(f"Metrics log: {len(df)} rows")
    if not df.empty:
        print(df.tail())
