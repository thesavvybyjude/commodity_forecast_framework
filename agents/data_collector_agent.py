"""
P2-T3: DataCollectorAgent
Unified wrapper that runs all 4 scrapers in sequence and reports status.
"""
import logging
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from scrapers.beautifulsoup_agent import BeautifulSoupAgent
from scrapers.selenium_agent import SeleniumAgent
from scrapers.api_connector import APIConnector
from scrapers.fuel_scraper import FuelScraper

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """Runs all scrapers in sequence and reports status."""

    def __init__(self):
        self.scrapers = {
            'exchange_rates': BeautifulSoupAgent(),
            'market_prices': SeleniumAgent(),
            'inflation': APIConnector(),
            'fuel_prices': FuelScraper(),
        }

    def collect_all(self) -> dict:
        """Run all scrapers, each in try/except — one failure does NOT stop others."""
        today = date.today().isoformat()
        result = {
            'date': today,
            'sources': {},
        }

        # Exchange rates
        try:
            logger.info("Collecting exchange rates...")
            df = self.scrapers['exchange_rates'].collect()
            result['sources']['exchange_rates'] = {
                'records': len(df),
                'status': 'ok',
            }
        except Exception as e:
            logger.error("Exchange rates failed: %s", e)
            result['sources']['exchange_rates'] = {
                'records': 0,
                'status': f'FAILED: {e}',
            }

        # Market prices
        try:
            logger.info("Collecting market prices...")
            df = self.scrapers['market_prices'].collect()
            result['sources']['market_prices'] = {
                'records': len(df),
                'status': 'ok',
            }
        except Exception as e:
            logger.error("Market prices failed: %s", e)
            result['sources']['market_prices'] = {
                'records': 0,
                'status': f'FAILED: {e}',
            }

        # Inflation
        try:
            logger.info("Collecting inflation data...")
            df = self.scrapers['inflation'].fetch_inflation()
            result['sources']['inflation'] = {
                'records': len(df),
                'status': 'ok',
            }
        except Exception as e:
            logger.error("Inflation fetch failed: %s", e)
            result['sources']['inflation'] = {
                'records': 0,
                'status': f'FAILED: {e}',
            }

        # Fuel prices
        try:
            logger.info("Collecting fuel prices...")
            df = self.scrapers['fuel_prices'].collect()
            result['sources']['fuel_prices'] = {
                'records': len(df),
                'status': 'ok',
            }
        except Exception as e:
            logger.error("Fuel prices failed: %s", e)
            result['sources']['fuel_prices'] = {
                'records': 0,
                'status': f'FAILED: {e}',
            }

        # Summary
        total = sum(s['records'] for s in result['sources'].values())
        failed = sum(1 for s in result['sources'].values() if 'FAILED' in str(s['status']))
        logger.info("Collection complete: %d total records, %d failures", total, failed)

        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = DataCollectorAgent()
    status = agent.collect_all()
    print(f"\nCollection date: {status['date']}")
    for source, info in status['sources'].items():
        print(f"  {source}: {info['records']} records — {info['status']}")
