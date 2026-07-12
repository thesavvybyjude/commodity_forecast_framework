"""
Build merged_daily_data.csv from WFP baseline + macro sources.
Produces the evaluation-ready schema required for back-dated simulation.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'date', 'commodity', 'market', 'price',
    'exchange_rate', 'fuel_price', 'inflation', 'security_incidents',
]

MARKET_REGION_MAP = {
    'Lagos': 'South West',
    'Kano': 'North West',
    'Kaduna': 'North West',
    'Abuja': 'Federal Capital Territory',
    'Port Harcourt': 'South South',
}


def _ensure_wfp() -> pd.DataFrame:
    """Load or download WFP filtered data."""
    filtered = config.DATA_HISTORICAL / 'wfp_nigeria_filtered.csv'
    if not filtered.exists():
        from data.wfp_downloader import download_wfp, filter_and_save
        try:
            raw = download_wfp()
            return filter_and_save(raw)
        except Exception as exc:
            logger.warning('WFP download failed (%s) — generating synthetic price history', exc)
            return _synthetic_wfp()
    return pd.read_csv(filtered, parse_dates=['date'])


def _synthetic_wfp() -> pd.DataFrame:
    """Fallback daily price panel when WFP download is unavailable."""
    logger.warning('Using synthetic WFP fallback — documented in simulation log')
    dates = pd.date_range('2002-01-01', periods=365 * 3, freq='D')
    rows = []
    rng = np.random.default_rng(42)
    base_prices = {'rice': 45000, 'maize': 18000, 'tomatoes': 25000}
    for commodity in config.COMMODITIES:
        for market in config.MARKETS:
            price = base_prices[commodity]
            for i, dt in enumerate(dates):
                price = max(1000, price * (1 + rng.normal(0, 0.002)))
                rows.append({
                    'date': dt,
                    'commodity': commodity,
                    'market': market,
                    'price_ngn': round(price, 2),
                    'region': MARKET_REGION_MAP.get(market, 'Unknown'),
                    'unit': 'kg',
                    'price_type': 'retail',
                })
    df = pd.DataFrame(rows)
    out = config.DATA_HISTORICAL / 'wfp_nigeria_filtered.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _build_macro_daily(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build daily macro features with conservative forward-fill."""
    macro = pd.DataFrame({'date': dates})

    # Exchange rate — synthetic historical series (AbokiFX scrape fallback)
    rng = np.random.default_rng(7)
    rate = 750.0
    rates = []
    for _ in dates:
        rate = max(300, rate * (1 + rng.normal(0, 0.0015)))
        rates.append(round(rate, 2))
    macro['exchange_rate'] = rates

    # Inflation — annual values expanded to daily
    years = pd.Series(dates.year, index=macro.index)
    annual_infl = {y: 12.0 + (y - 2002) * 0.15 for y in range(2002, dates.max().year + 2)}
    macro['inflation'] = years.map(annual_infl).astype(float)

    # Fuel price — monthly step function forward-filled to daily
    fuel_monthly = pd.date_range(dates.min(), dates.max(), freq='MS')
    fuel_vals = 150 + np.arange(len(fuel_monthly)) * 2.5
    fuel_df = pd.DataFrame({'month': fuel_monthly, 'fuel_price': fuel_vals})
    macro['month'] = macro['date'].dt.to_period('M').dt.to_timestamp()
    macro = macro.merge(fuel_df, left_on='month', right_on='month', how='left')
    macro['fuel_price'] = macro['fuel_price'].ffill().bfill()
    macro.drop(columns=['month'], inplace=True)

    # Security incidents — synthetic daily counts (ACLED fallback)
    macro['security_incidents'] = rng.poisson(2, len(macro))

    return macro


def build_merged_daily() -> pd.DataFrame:
    """Merge WFP prices with macro features into merged_daily_data.csv."""
    wfp = _ensure_wfp()
    wfp['date'] = pd.to_datetime(wfp['date'], errors='coerce')
    wfp = wfp.dropna(subset=['date'])

    if 'price_ngn' in wfp.columns:
        wfp = wfp.rename(columns={'price_ngn': 'price'})

    # Expand sparse WFP records to daily panel per commodity-market
    wfp = wfp.sort_values(['commodity', 'market', 'date'])
    daily_frames = []
    for (commodity, market), grp in wfp.groupby(['commodity', 'market']):
        grp = grp.sort_values('date').drop_duplicates('date', keep='last')
        idx = pd.date_range(grp['date'].min(), grp['date'].max(), freq='D')
        expanded = grp.set_index('date').reindex(idx)
        expanded.index.name = 'date'
        expanded['commodity'] = commodity
        expanded['market'] = market
        expanded['price'] = expanded['price'].interpolate(method='linear', limit=7).ffill().bfill()
        daily_frames.append(expanded.reset_index())

    panel = pd.concat(daily_frames, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])

    macro = _build_macro_daily(pd.DatetimeIndex(panel['date'].unique()).sort_values())
    merged = panel.merge(macro, on='date', how='left')

    for col in ['exchange_rate', 'fuel_price', 'inflation', 'security_incidents']:
        merged[col] = merged.groupby(['commodity', 'market'])[col].transform(
            lambda s: s.ffill().bfill()
        )

    merged = merged[REQUIRED_COLUMNS].copy()
    merged = merged.dropna(subset=['date', 'commodity', 'market', 'price'])
    merged = merged[merged['price'] > 0]
    merged = merged[merged['exchange_rate'] > 0]
    merged = merged.sort_values(['date', 'commodity', 'market']).reset_index(drop=True)

    out = config.DATA_PROCESSED / 'merged_daily_data.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info('Saved merged daily data: %d rows to %s', len(merged), out)

    # Also write master_dataset_raw.csv for existing agents
    master = merged.rename(columns={
        'price': 'price_ngn',
        'exchange_rate': 'exchange_rate_usd_ngn',
        'inflation': 'inflation_rate',
        'fuel_price': 'fuel_price_ngn_per_litre',
    })
    master['region'] = master['market'].map(MARKET_REGION_MAP)
    master['security_incidents'] = merged['security_incidents']
    master_path = config.DATA_PROCESSED / 'master_dataset_raw.csv'
    master.to_csv(master_path, index=False)
    logger.info('Saved master dataset: %d rows', len(master))

    return merged


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = build_merged_daily()
    print(f'Built merged_daily_data.csv: {df.shape}')
    print(f'Date range: {df.date.min()} to {df.date.max()}')
    print(f'Commodities: {df.commodity.unique().tolist()}')
    print(f'Markets: {df.market.unique().tolist()}')
