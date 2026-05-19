"""
P1-T7: ACLED Security Data Downloader
Downloads conflict/security incident data for Nigeria — used as a macro feature.
"""
import logging
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

ACLED_API_URL = "https://api.acleddata.com/acled/read"


class ACLEDDownloader:
    """Downloads and aggregates ACLED conflict data for Nigeria."""

    def __init__(self):
        self.api_key = config.ACLED_KEY

    def download(self) -> pd.DataFrame:
        """Download ACLED data via API or load from local file."""
        local_path = config.DATA_HISTORICAL / "acled_nigeria.csv"

        # Try API first if key is available
        if self.api_key:
            df = self._download_api()
            if not df.empty:
                df.to_csv(local_path, index=False)
                return df

        # Fall back to local file
        if local_path.exists():
            logger.info("Loading ACLED data from local file: %s", local_path)
            return pd.read_csv(local_path)

        logger.warning(
            "No ACLED data available. Either set ACLED_KEY in config "
            "or place CSV at %s",
            local_path,
        )
        return pd.DataFrame(columns=['year_month', 'admin1', 'conflict_events', 'fatalities'])

    def _download_api(self) -> pd.DataFrame:
        """Download from ACLED API."""
        params = {
            'key': self.api_key,
            'email': config.EMAIL_SENDER,
            'country': 'Nigeria',
            'limit': 0,  # no limit
        }

        try:
            logger.info("Fetching ACLED data via API")
            response = requests.get(ACLED_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error("ACLED API request failed: %s", e)
            return pd.DataFrame()

        if 'data' not in data:
            logger.warning("ACLED API returned unexpected format")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])
        return self._aggregate(df)

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to monthly conflict event counts and fatalities per region."""
        if df.empty:
            return df

        # Standardise column names
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'event_date' in cl or cl == 'event_date':
                col_map[col] = 'event_date'
            elif cl == 'admin1':
                col_map[col] = 'admin1'
            elif cl == 'fatalities':
                col_map[col] = 'fatalities'

        df = df.rename(columns=col_map)

        if 'event_date' not in df.columns:
            logger.warning("No event_date column found in ACLED data")
            return pd.DataFrame(columns=['year_month', 'admin1', 'conflict_events', 'fatalities'])

        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df['year_month'] = df['event_date'].dt.to_period('M').astype(str)

        if 'fatalities' in df.columns:
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
        else:
            df['fatalities'] = 0

        if 'admin1' not in df.columns:
            df['admin1'] = 'Unknown'

        grouped = df.groupby(['year_month', 'admin1']).agg(
            conflict_events=('event_date', 'count'),
            fatalities=('fatalities', 'sum'),
        ).reset_index()

        logger.info("Aggregated ACLED data: %d rows", len(grouped))
        return grouped

    def load_acled(self) -> pd.DataFrame:
        """Public method to load/download ACLED data."""
        return self.download()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    downloader = ACLEDDownloader()
    result = downloader.download()
    print(f"ACLED data: {len(result)} records")
    if not result.empty:
        print(result.head())
