"""
P1-T5: World Bank Inflation API Connector
Fetches annual CPI inflation rate for Nigeria from the World Bank API.
"""
import logging
from pathlib import Path
import sys

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

WB_BASE_URL = "https://api.worldbank.org/v2/country/NG/indicator"
INFLATION_INDICATOR = "FP.CPI.TOTL.ZG"


class APIConnector:
    """Fetches macroeconomic data from the World Bank API."""

    def __init__(self):
        self.session = requests.Session()

    def fetch_inflation(self, start_year: int = 2000) -> pd.DataFrame:
        """Fetch annual CPI inflation for Nigeria."""
        url = f"{WB_BASE_URL}/{INFLATION_INDICATOR}"
        params = {
            'format': 'json',
            'per_page': 500,
            'date': f'{start_year}:2025',
        }

        logger.info("Fetching inflation data from World Bank API")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error("World Bank API request failed: %s", e)
            return pd.DataFrame(columns=['year', 'inflation_rate', 'source'])
        except ValueError as e:
            logger.error("Failed to parse API response as JSON: %s", e)
            return pd.DataFrame(columns=['year', 'inflation_rate', 'source'])

        # World Bank returns [metadata, [records]]
        if not isinstance(data, list) or len(data) < 2:
            logger.warning("Unexpected API response format")
            return pd.DataFrame(columns=['year', 'inflation_rate', 'source'])

        records = []
        for entry in data[1] or []:
            if entry.get('value') is not None:
                records.append({
                    'year': int(entry['date']),
                    'inflation_rate': float(entry['value']),
                    'source': 'world_bank',
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('year').reset_index(drop=True)

        output_path = config.DATA_HISTORICAL / "inflation.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d inflation records to %s", len(df), output_path)

        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    connector = APIConnector()
    result = connector.fetch_inflation()
    print(f"Inflation data: {len(result)} records")
    if not result.empty:
        print(result.head())
