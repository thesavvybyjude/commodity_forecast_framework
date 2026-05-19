"""
P1-T6: Fuel Price Scraper
Scrapes PMS (petrol) prices from GlobalPetrolPrices — used as transport cost proxy.
"""
import logging
import time
from pathlib import Path
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

FUEL_URL = "https://www.globalpetrolprices.com/Nigeria/gasoline_prices/"


class FuelScraper:
    """Scrapes Nigerian fuel (PMS) prices from GlobalPetrolPrices."""

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
        """Scrape fuel prices with retry logic."""
        html = None

        for attempt in range(retries):
            try:
                logger.info("Attempt %d: fetching fuel prices", attempt + 1)
                response = self.session.get(FUEL_URL, timeout=30)
                response.raise_for_status()
                html = response.text
                break
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning("Attempt %d failed: %s. Retrying in %ds", attempt + 1, e, wait)
                time.sleep(wait)

        if not html:
            logger.error("All attempts to fetch fuel prices failed")
            return pd.DataFrame(columns=['date', 'fuel_price_ngn_per_litre'])

        records = self._parse(html)
        df = pd.DataFrame(records)

        if not df.empty:
            output_path = config.DATA_HISTORICAL / "fuel_prices.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info("Saved %d fuel price records to %s", len(df), output_path)

        return df

    def _parse(self, html: str) -> list:
        """Parse fuel price table from HTML."""
        records = []
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table', id='graphPageTableId')
        if not table:
            table = soup.find('table')

        if not table:
            logger.warning("No fuel price table found on page")
            return records

        rows = table.find_all('tr')
        for row in rows[1:]:  # skip header
            cols = row.find_all('td')
            if len(cols) >= 2:
                try:
                    date_text = cols[0].get_text(strip=True)
                    price_text = cols[1].get_text(strip=True)
                    price = float(price_text.replace(',', '').replace('₦', ''))
                    records.append({
                        'date': date_text,
                        'fuel_price_ngn_per_litre': price,
                    })
                except (ValueError, IndexError):
                    continue

        logger.info("Parsed %d fuel price records", len(records))
        return records


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    scraper = FuelScraper()
    result = scraper.collect()
    print(f"Fuel prices: {len(result)} records")
    if not result.empty:
        print(result.head())
