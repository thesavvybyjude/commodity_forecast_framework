"""
P1-T3: BeautifulSoup Exchange Rate Scraper
Scrapes daily USD/NGN exchange rates from AbokiFX (static HTML).
"""
import logging
import time
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

ABOKIFX_URL = "https://abokifx.com/ratetypes/"

FALLBACK_URLS = [
    "https://abokifx.com/",
]


class BeautifulSoupAgent:
    """Scrapes USD/NGN exchange rates from AbokiFX."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })

    def collect(self, retries: int = 3) -> pd.DataFrame:
        """Scrape exchange rates with exponential backoff."""
        html = None
        urls_to_try = [ABOKIFX_URL] + FALLBACK_URLS

        for url in urls_to_try:
            for attempt in range(retries):
                try:
                    logger.info("Attempt %d: fetching %s", attempt + 1, url)
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    html = response.text
                    break
                except requests.RequestException as e:
                    wait = 2 ** attempt
                    logger.warning("Attempt %d failed: %s. Retrying in %ds", attempt + 1, e, wait)
                    time.sleep(wait)
            if html:
                break

        if not html:
            logger.error("All attempts to fetch exchange rate data failed")
            return pd.DataFrame(columns=['date', 'source', 'currency_pair', 'rate'])

        records = self._parse(html)
        if not records:
            logger.warning("No records parsed from exchange rate page")

        df = self._save(records)
        return df

    def _parse(self, html: str) -> list:
        """Parse exchange rate table from HTML."""
        records = []
        soup = BeautifulSoup(html, 'html.parser')

        tables = soup.find_all('table', class_='table')
        if not tables:
            tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # skip header
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        date_text = cols[0].get_text(strip=True)
                        rate_text = cols[1].get_text(strip=True)
                        rate = float(rate_text.replace(',', '').replace('₦', ''))
                        records.append({
                            'date': date_text,
                            'source': 'abokifx',
                            'currency_pair': 'USD/NGN',
                            'rate': rate,
                        })
                    except (ValueError, IndexError):
                        continue

        logger.info("Parsed %d exchange rate records", len(records))
        return records

    def _save(self, records: list) -> pd.DataFrame:
        """Save records to daily raw directory."""
        today = date.today().isoformat()
        output_dir = config.DATA_RAW / today
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "exchange_rates.csv"

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info("Saved %d records to %s", len(df), output_path)
        else:
            logger.warning("No exchange rate records to save")

        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = BeautifulSoupAgent()
    result = agent.collect()
    print(f"Exchange rates collected: {len(result)} records")
    if not result.empty:
        print(result.head())
