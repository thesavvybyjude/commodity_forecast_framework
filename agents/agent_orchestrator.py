"""
P2-T7: AgentOrchestrator
Single entry point for the entire pipeline — connects all agents in sequence.
"""
import json
import logging
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.data_collector_agent import DataCollectorAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.model_version_control_agent import ModelVersionControlAgent
from agents.performance_monitor_agent import PerformanceMonitorAgent

logger = logging.getLogger(__name__)

# Configure logging to file
log_path = config.LOG_DIR / "pipeline.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
)
logging.getLogger().addHandler(file_handler)


class AgentOrchestrator:
    """Connects all agents into a single pipeline."""

    def __init__(self):
        self.collector = DataCollectorAgent()
        self.preprocessor = PreprocessingAgent()
        self.feature_engineer = FeatureEngineeringAgent()
        self.trainer = ModelTrainerAgent()
        self.mvc = ModelVersionControlAgent()
        self.monitor = PerformanceMonitorAgent()

    def run_daily_pipeline(self) -> dict:
        """Run the daily data collection + model update pipeline."""
        logger.info("=" * 60)
        logger.info("DAILY PIPELINE START")
        logger.info("=" * 60)

        try:
            # Step 1: Collect new data
            logger.info("Step 1: Data collection")
            collection_status = self.collector.collect_all()
            logger.info("Collection result: %s", collection_status)

            # Step 2: Load and preprocess
            logger.info("Step 2: Preprocessing")
            master_path = config.DATA_PROCESSED / "master_dataset_raw.csv"
            if not master_path.exists():
                from data.merge_datasets import merge_all
                merge_all()

            df = pd.read_csv(master_path, parse_dates=['date'])
            cleaned = self.preprocessor.clean(df)

            # Step 3: Feature engineering
            logger.info("Step 3: Feature engineering")
            features = self.feature_engineer.engineer(cleaned)
            feat_cols = self.feature_engineer.get_feature_columns(features)

            # Save daily features
            today = date.today().isoformat()
            features_path = config.DATA_PROCESSED / f"features_{today}.csv"
            features.to_csv(features_path, index=False)
            logger.info("Saved features to %s", features_path)

            # Step 4: Model updates
            logger.info("Step 4: Model updates")
            from agents.model_updater_agent import ModelUpdaterAgent
            updater = ModelUpdaterAgent()
            updater.update_all(features, feat_cols)

            # Step 5: Notifications
            logger.info("Step 5: Notifications check")
            try:
                from notifications.notifier import Notifier
                notifier = Notifier()
                notifier.check_and_alert(features)
            except Exception as e:
                logger.warning("Notification check failed: %s", e)

            logger.info("DAILY PIPELINE COMPLETE")
            return {'status': 'ok'}

        except Exception as e:
            logger.error("PIPELINE FAILED: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def run_initial_training(self) -> dict:
        """One-time call: preprocess historical data and train all baseline models."""
        logger.info("=" * 60)
        logger.info("INITIAL TRAINING START")
        logger.info("=" * 60)

        try:
            # Load and preprocess
            master_path = config.DATA_PROCESSED / "master_dataset_raw.csv"
            if not master_path.exists():
                from data.merge_datasets import merge_all
                merge_all()

            if not master_path.exists():
                return {'status': 'error', 'message': 'No master dataset available'}

            df = pd.read_csv(master_path, parse_dates=['date'])
            cleaned = self.preprocessor.clean(df)
            features = self.feature_engineer.engineer(cleaned)

            feat_cols = self.feature_engineer.get_feature_columns(features)
            logger.info("Feature columns: %d", len(feat_cols))

            # Prepare arrays
            numeric_feat_cols = [c for c in feat_cols if features[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            X = features[numeric_feat_cols].values
            y = features['price_ngn'].values if 'price_ngn' in features.columns else X[:, 0]

            # Train LSTM for groups A and B
            logger.info("Training LSTM Group A (periodic retrain)")
            self.trainer.train_lstm(X, y, 'A')

            logger.info("Training LSTM Group B (fine-tune)")
            self.trainer.train_lstm(X, y, 'B')

            # Train XGBoost for groups C, D, and E
            logger.info("Training XGBoost Group C (periodic retrain)")
            self.trainer.train_xgboost(X, y, 'C')

            logger.info("Training XGBoost Group D (warm-start)")
            self.trainer.train_xgboost(X, y, 'D')

            logger.info("Training XGBoost Group E (static baseline)")
            self.trainer.train_xgboost(X, y, 'E')

            logger.info("INITIAL TRAINING COMPLETE")
            return {'status': 'ok'}

        except Exception as e:
            logger.error("INITIAL TRAINING FAILED: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    orchestrator = AgentOrchestrator()
    print("AgentOrchestrator ready.")
    print("  orchestrator.run_initial_training() — one-time baseline training")
    print("  orchestrator.run_daily_pipeline()   — daily update pipeline")
