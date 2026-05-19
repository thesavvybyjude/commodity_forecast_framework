"""
P1-T8: Dataset Merger
Combines all data sources into master_dataset_raw.csv, aligned by date.
"""
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def load_wfp() -> pd.DataFrame:
    """Load the filtered WFP baseline data."""
    path = config.DATA_HISTORICAL / "wfp_nigeria_filtered.csv"
    if not path.exists():
        logger.warning("WFP data not found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=['date'])
    logger.info("Loaded WFP data: %d rows", len(df))
    return df


def load_inflation() -> pd.DataFrame:
    """Load inflation data."""
    path = config.DATA_HISTORICAL / "inflation.csv"
    if not path.exists():
        logger.warning("Inflation data not found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info("Loaded inflation data: %d rows", len(df))
    return df


def load_fuel_prices() -> pd.DataFrame:
    """Load fuel price data."""
    path = config.DATA_HISTORICAL / "fuel_prices.csv"
    if not path.exists():
        logger.warning("Fuel price data not found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
    logger.info("Loaded fuel price data: %d rows", len(df))
    return df


def load_exchange_rates() -> pd.DataFrame:
    """Load the most recent exchange rate data from raw daily dumps."""
    raw_dirs = sorted(config.DATA_RAW.glob('*/exchange_rates.csv'))
    if not raw_dirs:
        logger.warning("No exchange rate data found")
        return pd.DataFrame()

    dfs = []
    for path in raw_dirs:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
        combined = combined.rename(columns={'rate': 'exchange_rate_usd_ngn'})
    logger.info("Loaded exchange rate data: %d rows", len(combined))
    return combined


def load_acled() -> pd.DataFrame:
    """Load ACLED conflict data."""
    path = config.DATA_HISTORICAL / "acled_nigeria.csv"
    if not path.exists():
        logger.warning("ACLED data not found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info("Loaded ACLED data: %d rows", len(df))
    return df


def merge_all() -> pd.DataFrame:
    """Merge all data sources into a single master dataset."""
    logger.info("Starting dataset merge")

    # Base table: WFP prices
    master = load_wfp()
    if master.empty:
        logger.error("No WFP base data — cannot proceed with merge")
        return master

    master['date'] = pd.to_datetime(master['date'], errors='coerce')
    master['year'] = master['date'].dt.year
    master['year_month'] = master['date'].dt.to_period('M').astype(str)

    # Merge inflation by year
    inflation = load_inflation()
    if not inflation.empty and 'year' in inflation.columns:
        master = master.merge(
            inflation[['year', 'inflation_rate']],
            on='year',
            how='left',
        )
        logger.info("Merged inflation data")

    # Merge fuel prices by year_month
    fuel = load_fuel_prices()
    if not fuel.empty and 'year_month' in fuel.columns:
        fuel_monthly = fuel.groupby('year_month')['fuel_price_ngn_per_litre'].mean().reset_index()
        master = master.merge(
            fuel_monthly,
            on='year_month',
            how='left',
        )
        logger.info("Merged fuel price data")

    # Merge exchange rates by date
    exchange = load_exchange_rates()
    if not exchange.empty and 'date' in exchange.columns:
        exchange_daily = exchange.groupby('date')['exchange_rate_usd_ngn'].mean().reset_index()
        master = master.merge(
            exchange_daily,
            on='date',
            how='left',
        )
        logger.info("Merged exchange rate data")

    # Merge ACLED by year_month + region
    acled = load_acled()
    if not acled.empty and 'year_month' in acled.columns:
        merge_keys = ['year_month']
        if 'admin1' in acled.columns and 'region' in master.columns:
            acled = acled.rename(columns={'admin1': 'region'})
            merge_keys.append('region')

        master = master.merge(
            acled,
            on=merge_keys,
            how='left',
        )
        logger.info("Merged ACLED data")

    # Drop helper columns
    master = master.drop(columns=['year', 'year_month'], errors='ignore')

    # Save
    output_path = config.DATA_PROCESSED / "master_dataset_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    logger.info("Saved master dataset: %d rows, %d columns to %s", len(master), len(master.columns), output_path)

    # Summary
    logger.info("Dataset info:")
    logger.info("  Shape: %s", master.shape)
    logger.info("  Null counts:\n%s", master.isnull().sum())

    return master


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = merge_all()
    print(f"\nMerge complete: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
