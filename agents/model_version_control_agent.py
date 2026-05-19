"""
P2-T5: ModelVersionControlAgent
Manages model_registry.json — tracks version numbers, file paths, timestamps, and metrics.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class ModelVersionControlAgent:
    """Manages model versioning and registry."""

    def __init__(self):
        self.registry_path = config.REGISTRY_PATH
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or initialise model registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Persist registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _get_key(self, model_type: str, group: str) -> str:
        """Generate registry key like 'lstm_A' or 'xgb_D'."""
        return f"{model_type}_{group}"

    def save_lstm(self, model, group: str, metrics: dict = None) -> Path:
        """Save a Keras LSTM model, increment version, update registry."""
        key = self._get_key('lstm', group)
        version = self._next_version(key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lstm_{group}_v{version}_{timestamp}.keras"
        filepath = config.MODEL_LSTM_DIR / filename

        config.MODEL_LSTM_DIR.mkdir(parents=True, exist_ok=True)
        model.save(filepath)

        self.registry[key] = {
            'version': version,
            'path': str(filepath),
            'timestamp': timestamp,
            'metrics': metrics or {},
        }
        self._save_registry()
        logger.info("Saved LSTM model: %s (v%d)", filename, version)
        return filepath

    def save_xgb(self, model, group: str, metrics: dict = None) -> Path:
        """Save an XGBoost model, increment version, update registry."""
        key = self._get_key('xgb', group)
        version = self._next_version(key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"xgb_{group}_v{version}_{timestamp}.joblib"
        filepath = config.MODEL_XGB_DIR / filename

        config.MODEL_XGB_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)

        self.registry[key] = {
            'version': version,
            'path': str(filepath),
            'timestamp': timestamp,
            'metrics': metrics or {},
        }
        self._save_registry()
        logger.info("Saved XGBoost model: %s (v%d)", filename, version)
        return filepath

    def get_latest(self, model_type: str, group: str) -> Path:
        """Return Path to the most recent model for a type+group key."""
        key = self._get_key(model_type, group)
        entry = self.registry.get(key)
        if not entry:
            logger.warning("No model found in registry for key: %s", key)
            return None

        path = Path(entry['path'])
        if not path.exists():
            logger.warning("Registered model file not found: %s", path)
            return None

        return path

    def update_metrics(self, model_type: str, group: str,
                       rmse: float, mae: float, mape: float, r2: float):
        """Write metrics into registry entry."""
        key = self._get_key(model_type, group)
        if key not in self.registry:
            logger.warning("Cannot update metrics — key '%s' not in registry", key)
            return

        self.registry[key]['metrics'] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
        }
        self._save_registry()
        logger.info("Updated metrics for %s: RMSE=%.4f MAE=%.4f", key, rmse, mae)

    def days_since_last_train(self, model_type: str, group: str) -> int:
        """Return int days since last save (used by updater)."""
        key = self._get_key(model_type, group)
        entry = self.registry.get(key)
        if not entry or 'timestamp' not in entry:
            return 999  # force retrain if no history

        try:
            last_time = datetime.strptime(entry['timestamp'], '%Y%m%d_%H%M%S')
            delta = datetime.now() - last_time
            return delta.days
        except (ValueError, KeyError):
            return 999

    def get_baseline_rmse(self, model_type: str, group: str) -> float:
        """Get baseline RMSE from registry."""
        key = self._get_key(model_type, group)
        entry = self.registry.get(key, {})
        metrics = entry.get('metrics', {})
        return metrics.get('rmse', None)

    def _next_version(self, key: str) -> int:
        """Get next version number for a registry key."""
        if key in self.registry:
            return self.registry[key].get('version', 0) + 1
        return 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mvc = ModelVersionControlAgent()
    print("Current registry:")
    print(json.dumps(mvc.registry, indent=2))
