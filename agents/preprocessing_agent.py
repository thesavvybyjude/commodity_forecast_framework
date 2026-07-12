"""
P2-T1: PreprocessingAgent
Cleans raw data before feature engineering — handles missing values, outliers, normalisation.
"""
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """Cleans and normalises raw commodity price data."""

    def __init__(self):
        self.scaler = None
        self.scaler_path = config.SCALER_DIR / "minmax_scaler.joblib"
        self.numeric_cols = []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline: missing → outliers → normalise."""
        logger.info("Starting preprocessing on %d rows", len(df))
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        df = self._handle_missing(df)
        df = self._handle_outliers(df, col='price_ngn')
        df = self._normalise(df)

        logger.info("Preprocessing complete: %d rows, %d cols", len(df), len(df.columns))
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Linear interpolation for gaps <= 3 days, then forward-fill, then back-fill."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            df[col] = df[col].interpolate(method='linear', limit=3)

        df = df.ffill()
        df = df.bfill()

        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning("Still %d nulls after filling", remaining_nulls)

        return df

    def _handle_outliers(self, df: pd.DataFrame, col: str = 'price_ngn') -> pd.DataFrame:
        """IQR method (1.5x) — replaces outliers with median."""
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping outlier handling", col)
            return df

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = df[col].median()

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            df.loc[outlier_mask, col] = median
            logger.info("Replaced %d outliers in '%s' with median (%.2f)", n_outliers, col, median)

        return df

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """MinMaxScaler on all numeric columns, saves scaler to disk."""
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not self.numeric_cols:
            logger.warning("No numeric columns to normalise")
            return df

        self.scaler = MinMaxScaler()
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])

        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info("Saved MinMaxScaler to %s", self.scaler_path)

        return df

    def inverse_transform_price(self, scaled_values: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to NGN price values."""
        if self.scaler is None:
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded scaler from %s", self.scaler_path)
            else:
                logger.error("No scaler available for inverse transform")
                return scaled_values

        if 'price_ngn' not in self.numeric_cols:
            # Try to find price_ngn index
            self.numeric_cols = list(range(self.scaler.n_features_in_))

        price_idx = None
        for i, col in enumerate(self.numeric_cols):
            if col == 'price_ngn':
                price_idx = i
                break

        if price_idx is None:
            price_idx = 0

        dummy = np.zeros((len(scaled_values), self.scaler.n_features_in_))
        dummy[:, price_idx] = scaled_values.flatten()
        inverse = self.scaler.inverse_transform(dummy)
        return inverse[:, price_idx]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = config.DATA_PROCESSED / "master_dataset_raw.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        agent = PreprocessingAgent()
        cleaned = agent.clean(df)
        print(f"Cleaned: {cleaned.shape}")
        print(cleaned.head())
    else:
        print(f"Run merge_datasets.py first — {path} not found")
