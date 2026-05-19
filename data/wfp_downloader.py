"""
P1-T2: WFP Historical Downloader
Downloads WFP Nigeria price data from Humanitarian Data Exchange (2002-2023 baseline).
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

WFP_URL = (
    "https://data.humdata.org/dataset/"
    "wfp-food-prices-for-nigeria/resource/"
    "wfp_food_prices_nga.csv"
)

FALLBACK_URLS = [
    "https://data.humdata.org/dataset/wfp-food-prices-for-nigeria",
]


def download_wfp(url: str = WFP_URL, timeout: int = 60) -> Path:
    """Download raw WFP CSV file and save to data/historical/."""
    output_path = config.DATA_HISTORICAL / "wfp_nigeria_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading WFP data from %s", url)
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Primary URL failed: %s. Trying fallback.", e)
        for fallback in FALLBACK_URLS:
            try:
                response = requests.get(fallback, timeout=timeout)
                response.raise_for_status()
                break
            except requests.RequestException:
                continue
        else:
            logger.error("All download URLs failed.")
            raise

    output_path.write_bytes(response.content)
    logger.info("Saved raw WFP data to %s (%d bytes)", output_path, len(response.content))
    return output_path


def filter_and_save(raw_path: Path = None) -> pd.DataFrame:
    """Filter WFP data to target commodities and standardise columns."""
    if raw_path is None:
        raw_path = config.DATA_HISTORICAL / "wfp_nigeria_raw.csv"

    logger.info("Reading raw WFP data from %s", raw_path)
    df = pd.read_csv(raw_path, low_memory=False)

    # Standardise column names
    column_map = {
        'date': 'date',
        'admin1': 'region',
        'market': 'market',
        'commodity': 'commodity',
        'price': 'price_ngn',
        'unit': 'unit',
        'pricetype': 'price_type',
    }

    # Try to find matching columns (case-insensitive)
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in column_map:
            rename_dict[col] = column_map[col_lower]

    df = df.rename(columns=rename_dict)

    # Filter to target commodities
    commodity_keywords = {
        'rice': ['rice'],
        'maize': ['maize', 'corn'],
        'tomatoes': ['tomato', 'tomatoes'],
    }

    def map_commodity(name: str) -> str:
        name_lower = str(name).lower()
        for target, keywords in commodity_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return target
        return ''

    if 'commodity' in df.columns:
        df['commodity_mapped'] = df['commodity'].apply(map_commodity)
        df = df[df['commodity_mapped'].isin(config.COMMODITIES)].copy()
        df['commodity'] = df['commodity_mapped']
        df.drop(columns=['commodity_mapped'], inplace=True)

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # Ensure required columns exist
    required_cols = ['date', 'market', 'commodity', 'price_ngn']
    for col in required_cols:
        if col not in df.columns:
            logger.warning("Missing column: %s — creating placeholder", col)
            df[col] = None

    # Select and order columns
    output_cols = ['date', 'market', 'commodity', 'price_ngn', 'unit', 'price_type', 'region']
    existing_cols = [c for c in output_cols if c in df.columns]
    df = df[existing_cols]

    output_path = config.DATA_HISTORICAL / "wfp_nigeria_filtered.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved filtered WFP data: %d rows to %s", len(df), output_path)

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        raw = download_wfp()
        df = filter_and_save(raw)
        print(f"WFP download complete: {len(df)} rows")
        print(df.head())
    except Exception as e:
        logger.error("WFP download failed: %s", e)
        print(f"Download failed: {e}")
        print("You can manually download from HDX and place at data/historical/wfp_nigeria_raw.csv")
