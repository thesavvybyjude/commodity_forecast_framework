"""
P1-T4: Selenium Market Price Scraper
Scrapes live commodity prices from PricePally.com (JavaScript-rendered).
"""
import logging
import time
from datetime import date
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

BASE_URL = "https://www.pricepally.com/search?q="


class SeleniumAgent:
    """Scrapes market prices from PricePally using Selenium."""

    def __init__(self):
        self.driver = None

    def _init_driver(self):
        """Initialize headless Chrome WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)

    def collect(self) -> pd.DataFrame:
        """Scrape prices for all target commodities."""
        all_records = []

        try:
            self._init_driver()

            for commodity in config.COMMODITIES:
                logger.info("Scraping prices for: %s", commodity)
                records = self._scrape_commodity(commodity)
                all_records.extend(records)
                time.sleep(2)  # polite crawling delay

        except Exception as e:
            logger.error("Selenium scraping failed: %s", e)
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

        df = self._save(all_records)
        return df

    def _scrape_commodity(self, commodity: str) -> list:
        """Scrape prices for a single commodity."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        records = []
        url = f"{BASE_URL}{commodity}"

        try:
            self.driver.get(url)

            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "product-card"))
            )

            cards = self.driver.find_elements(By.CLASS_NAME, "product-card")[:10]

            for card in cards:
                try:
                    name_el = card.find_element(By.CLASS_NAME, "product-name")
                    price_el = card.find_element(By.CLASS_NAME, "product-price")

                    product_name = name_el.text.strip()
                    price_text = price_el.text.strip()
                    price = float(
                        price_text.replace('₦', '')
                        .replace(',', '')
                        .replace('NGN', '')
                        .strip()
                    )

                    records.append({
                        'date': date.today().isoformat(),
                        'source': 'pricepally',
                        'commodity': commodity,
                        'product_name': product_name,
                        'price_ngn': price,
                        'market': 'Lagos',  # PricePally default market
                    })
                except Exception as e:
                    logger.debug("Failed to parse card: %s", e)
                    continue

            logger.info("Scraped %d records for %s", len(records), commodity)

        except Exception as e:
            logger.warning("Failed to scrape %s: %s", commodity, e)

        return records

    def _save(self, records: list) -> pd.DataFrame:
        """Save records to daily raw directory."""
        today = date.today().isoformat()
        output_dir = config.DATA_RAW / today
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "market_prices.csv"

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info("Saved %d market price records to %s", len(df), output_path)
        else:
            logger.warning("No market price records to save")

        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = SeleniumAgent()
    result = agent.collect()
    print(f"Market prices collected: {len(result)} records")
    if not result.empty:
        print(result.head())
