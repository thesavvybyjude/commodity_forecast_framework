"""
P2-T2: FeatureEngineeringAgent
Creates the full feature matrix from clean data — produces 40+ features for model input.
"""
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

META_COLUMNS = [
    'date', 'commodity', 'market', 'region',
    'price_ngn', 'price_type', 'unit', 'source',
]


class FeatureEngineeringAgent:
    """Generates feature matrix with temporal, lag, rolling, momentum, and macro features."""

    def __init__(self):
        self.feature_columns = []

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        logger.info("Starting feature engineering on %d rows", len(df))
        df = df.copy()

        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Sort by commodity, market, date
        sort_cols = [c for c in ['commodity', 'market', 'date'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        df = self._temporal(df)
        df = self._lags(df)
        df = self._rolling(df)
        df = self._momentum(df)
        df = self._macro(df)
        df = self._encode(df)

        # Drop rows where lag/rolling features can't be computed
        df = df.dropna().reset_index(drop=True)

        # Save feature column list
        self.feature_columns = self.get_feature_columns(df)
        self._save_feature_columns()

        logger.info(
            "Feature engineering complete: %d rows, %d features",
            len(df), len(self.feature_columns),
        )
        return df

    def _temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features: month, quarter, year, season, is_harvest."""
        if 'date' not in df.columns:
            return df

        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year_feat'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear

        # Nigerian agricultural seasons
        def get_season(month: int) -> int:
            if month in [11, 12, 1, 2]:
                return 0  # Harmattan (dry)
            elif month in [3, 4]:
                return 1  # Planting
            elif month in [5, 6, 7, 8]:
                return 2  # Growing
            else:
                return 3  # Harvest (Sep, Oct)

        df['season'] = df['month'].apply(get_season)
        df['is_harvest'] = (df['month'] == 10).astype(int)

        return df

    def _lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price lags at 1, 7, 30, 90, 365 days (grouped by commodity+market)."""
        if 'price_ngn' not in df.columns:
            return df

        group_cols = [c for c in ['commodity', 'market'] if c in df.columns]

        for lag in [1, 7, 30, 90, 365]:
            col_name = f'price_lag_{lag}'
            if group_cols:
                df[col_name] = df.groupby(group_cols)['price_ngn'].shift(lag)
            else:
                df[col_name] = df['price_ngn'].shift(lag)

        return df

    def _rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean/std at 7, 30, 90 days; rolling min/max at 7 days."""
        if 'price_ngn' not in df.columns:
            return df

        group_cols = [c for c in ['commodity', 'market'] if c in df.columns]

        for window in [7, 30, 90]:
            mean_col = f'rolling_mean_{window}'
            std_col = f'rolling_std_{window}'

            if group_cols:
                grouped = df.groupby(group_cols)['price_ngn']
                df[mean_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
                df[std_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std())
            else:
                df[mean_col] = df['price_ngn'].rolling(window, min_periods=1).mean()
                df[std_col] = df['price_ngn'].rolling(window, min_periods=1).std()

        # Rolling min/max at 7 days
        if group_cols:
            grouped = df.groupby(group_cols)['price_ngn']
            df['rolling_min_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).min())
            df['rolling_max_7'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).max())
        else:
            df['rolling_min_7'] = df['price_ngn'].rolling(7, min_periods=1).min()
            df['rolling_max_7'] = df['price_ngn'].rolling(7, min_periods=1).max()

        return df

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features: % change from day before, rate of change."""
        if 'price_ngn' not in df.columns:
            return df

        group_cols = [c for c in ['commodity', 'market'] if c in df.columns]

        if group_cols:
            grouped = df.groupby(group_cols)['price_ngn']
            df['momentum_1'] = grouped.pct_change()
            df['rate_of_change'] = grouped.transform(
                lambda x: (x - x.shift(7)) / x.shift(7).replace(0, np.nan)
            )
        else:
            df['momentum_1'] = df['price_ngn'].pct_change()
            prev_7 = df['price_ngn'].shift(7).replace(0, np.nan)
            df['rate_of_change'] = (df['price_ngn'] - df['price_ngn'].shift(7)) / prev_7

        return df

    def _macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Macro features: price_fx_ratio, inflation_adj_price, transport_cost_proxy."""
        if 'price_ngn' in df.columns and 'exchange_rate_usd_ngn' in df.columns:
            fx = df['exchange_rate_usd_ngn'].replace(0, np.nan)
            df['price_fx_ratio'] = df['price_ngn'] / fx

        if 'price_ngn' in df.columns and 'inflation_rate' in df.columns:
            infl = (1 + df['inflation_rate'] / 100).replace(0, np.nan)
            df['inflation_adj_price'] = df['price_ngn'] / infl

        if 'fuel_price_ngn_per_litre' in df.columns:
            df['transport_cost_proxy'] = df['fuel_price_ngn_per_litre'] * 1.2

        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Target encoding for market and region, one-hot for season."""
        if 'market' in df.columns and 'price_ngn' in df.columns:
            market_means = df.groupby('market')['price_ngn'].mean()
            df['market_encoded'] = df['market'].map(market_means)

        if 'region' in df.columns and 'price_ngn' in df.columns:
            region_means = df.groupby('region')['price_ngn'].mean()
            df['region_encoded'] = df['region'].map(region_means)

        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season', dtype=int)
            df = pd.concat([df, season_dummies], axis=1)

        return df

    def get_feature_columns(self, df: pd.DataFrame = None) -> list:
        """Get list of feature columns (excluding meta columns)."""
        if df is not None:
            return [c for c in df.columns if c not in META_COLUMNS]

        # Load from saved file
        feat_path = config.DATA_PROCESSED / "feature_columns.json"
        if feat_path.exists():
            with open(feat_path) as f:
                return json.load(f)
        return self.feature_columns

    def _save_feature_columns(self):
        """Save feature column list to JSON."""
        output_path = config.DATA_PROCESSED / "feature_columns.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info("Saved %d feature columns to %s", len(self.feature_columns), output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = config.DATA_PROCESSED / "master_dataset_raw.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        agent = FeatureEngineeringAgent()
        features = agent.engineer(df)
        print(f"Features: {features.shape}")
        print(f"Feature columns ({len(agent.feature_columns)}):")
        for col in agent.feature_columns:
            print(f"  {col}")
    else:
        print(f"Run merge_datasets.py first — {path} not found")
