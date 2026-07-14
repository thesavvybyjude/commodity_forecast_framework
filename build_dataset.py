"""
Build a complete, realistic historical dataset for Nigerian commodity prices.
Covers 2010-01-01 through 2025-05-16 (21 days before 2025-06-06 sim end).

Sources simulated:
- WFP-style commodity prices (rice, maize, tomatoes)
- Exchange rates NGN/USD
- Fuel prices (NGN/litre)
- Inflation (annual %)
- Security incidents (ACLED-style)

All values calibrated to real Nigerian market ranges.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config

np.random.seed(42)

# ── Date range ──────────────────────────────────────────────────────────────
START = pd.Timestamp("2010-01-01")
END   = pd.Timestamp("2025-06-06")    # sim window ends here
dates = pd.date_range(START, END, freq="D")
n_days = len(dates)

# ── Markets & commodities ───────────────────────────────────────────────────
MARKETS     = ["Lagos", "Kano", "Kaduna", "Abuja", "Port Harcourt"]
COMMODITIES = ["rice", "maize", "tomatoes"]

# ── Exchange rate (NGN per USD) realistic trajectory ────────────────────────
# ~155 in 2010 → ~460 in 2022 → ~760 in 2023 → ~1580 in 2024 → ~1520 in 2025
def generate_exchange_rate(dates):
    n = len(dates)
    year = np.array([d.year + d.dayofyear/365 for d in dates])
    # Piecewise linear trend
    base = np.interp(year, [2010, 2015, 2020, 2022, 2023, 2024, 2025],
                           [155,  197,  360,  430,  760, 1480, 1520])
    noise = np.cumsum(np.random.normal(0, 2, n))
    noise -= noise[0]
    rate = base + noise * 0.5
    rate = np.maximum(rate, 100)
    return rate

exch = generate_exchange_rate(dates)

# ── Fuel price (NGN/litre) ──────────────────────────────────────────────────
# ~65 in 2010 → ~145 in 2016 → ~162 in 2020 → ~198 in 2022 → ~617 in 2023 → ~855 in 2024
def generate_fuel_price(dates):
    n = len(dates)
    year = np.array([d.year + d.dayofyear/365 for d in dates])
    base = np.interp(year, [2010, 2016, 2020, 2022, 2023, 2024, 2025],
                           [65,   145,  162,  198,  617,  855,  870])
    # Monthly steps (policy changes happen stepwise)
    monthly_idx = pd.DatetimeIndex(dates).to_period("M").astype(int)
    monthly_idx -= monthly_idx.min()
    noise = np.random.normal(0, 3, max(monthly_idx)+1)
    fuel = base + np.array([noise[i] for i in monthly_idx])
    return np.maximum(fuel, 50)

fuel = generate_fuel_price(dates)

# ── Inflation (annual %, monthly cadence interpolated to daily) ─────────────
def generate_inflation(dates):
    year = np.array([d.year + d.dayofyear/365 for d in dates])
    base = np.interp(year, [2010, 2012, 2016, 2020, 2022, 2023, 2024, 2025],
                           [13.7, 12.2, 15.7,  13.2,  18.8,  24.5,  33.2, 29.0])
    noise = np.random.normal(0, 0.3, len(dates))
    return np.maximum(base + noise, 5)

infl = generate_inflation(dates)

# ── Security incidents ──────────────────────────────────────────────────────
def generate_security(dates):
    n = len(dates)
    year = np.array([d.year + d.dayofyear/365 for d in dates])
    # Higher in NE Nigeria from 2012-2019 (Boko Haram), then bandits 2020+
    base = np.interp(year, [2010, 2012, 2014, 2018, 2020, 2023, 2025],
                           [2,    5,    12,    8,    10,   14,   11])
    return np.maximum(np.random.poisson(base), 0).astype(float)

security = generate_security(dates)

# ── Commodity price generation ───────────────────────────────────────────────
# Price in NGN; calibrated to real WFP data
# Rice (50kg bag): ~6000 in 2010 → ~18000 in 2020 → ~65000 in 2024
# Maize (100kg):   ~3500 in 2010 → ~12000 in 2020 → ~48000 in 2024  
# Tomatoes (1kg):  ~80 in 2010   → ~250 in 2020   → ~1200 in 2024

PRICE_TRAJ = {
    "rice":     [2010,2012,2015,2018,2020,2022,2023,2024,2025],
               # anchor prices
    "maize":    [2010,2012,2015,2018,2020,2022,2023,2024,2025],
    "tomatoes": [2010,2012,2015,2018,2020,2022,2023,2024,2025],
}
PRICE_VALS = {
    "rice":     [6000, 8000, 11000, 14000, 18000, 35000, 55000, 65000, 68000],
    "maize":    [3500, 4500,  6000,  9000, 12000, 22000, 38000, 48000, 50000],
    "tomatoes": [  80,  110,   160,   220,   250,   600,  1100,  1200,  1250],
}

# Market-level multipliers (Lagos is index, others differ)
MARKET_MUL = {
    "Lagos":       1.00,
    "Kano":        0.88,
    "Kaduna":      0.91,
    "Abuja":       1.08,
    "Port Harcourt": 1.04,
}
# Seasonal pattern (month → multiplier, 1=Jan)
def seasonal(month, commodity):
    """Tomatoes are highly seasonal; rice and maize less so."""
    if commodity == "tomatoes":
        # Peak Oct-Nov (harvest), trough Jun-Aug (scarce)
        base = [1.0, 1.1, 1.2, 1.15, 1.0, 1.3, 1.4, 1.35, 1.1, 0.7, 0.75, 0.85]
    elif commodity == "maize":
        # Harvest Aug-Sep
        base = [1.1, 1.05, 1.0, 0.95, 0.9, 1.0, 1.1, 0.8, 0.75, 0.9, 1.05, 1.1]
    else:  # rice
        base = [1.05, 1.02, 0.98, 0.97, 0.96, 1.0, 1.02, 1.01, 0.99, 0.97, 1.0, 1.03]
    return base[month - 1]

rows = []
year_arr  = np.array([d.year + d.dayofyear/365 for d in dates])

for commodity in COMMODITIES:
    base_price = np.interp(year_arr,
                           PRICE_TRAJ[commodity],
                           PRICE_VALS[commodity])

    for market in MARKETS:
        mkt_mul = MARKET_MUL[market]

        # AR(1) noise component for realism
        np.random.seed(hash((commodity, market)) % (2**32))
        noise = np.zeros(n_days)
        noise[0] = 0
        for i in range(1, n_days):
            noise[i] = 0.92 * noise[i-1] + np.random.normal(0, base_price[i] * 0.015)

        for i, d in enumerate(dates):
            sea = seasonal(d.month, commodity)
            price = base_price[i] * mkt_mul * sea + noise[i]
            price = max(price, 1)

            rows.append({
                "date":               d,
                "commodity":          commodity,
                "market":             market,
                "region":             market,
                "price_ngn":          round(price, 2),
                "exchange_rate_usd_ngn": round(exch[i], 2),
                "fuel_price_ngn":     round(fuel[i], 2),
                "inflation_rate":     round(infl[i], 2),
                "security_incidents": int(security[i]),
                "price_type":         "retail",
                "unit":               "kg" if commodity == "tomatoes" else "50kg",
                "source":             "WFP_synthetic",
            })

df = pd.DataFrame(rows)
df = df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)

logger.info("Generated dataset: %d rows", len(df))
logger.info("Date range: %s to %s", df["date"].min().date(), df["date"].max().date())
logger.info("Commodities: %s", df["commodity"].unique().tolist())
logger.info("Markets: %s", df["market"].unique().tolist())
logger.info("Columns: %s", df.columns.tolist())

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = config.DATA_PROCESSED / "master_dataset_raw.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
logger.info("Saved to %s", out_path)

print(f"\n✓ Dataset built: {df.shape}")
print(f"  Rows: {len(df):,}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Saved: {out_path}")
