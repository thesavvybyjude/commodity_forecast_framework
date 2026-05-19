"""
P5-T4: Data Health Page
Shows scraper status — whether each data source collected records today.
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


DATA_SOURCES = [
    {'name': 'Exchange Rates', 'file': 'exchange_rates.csv'},
    {'name': 'Market Prices', 'file': 'market_prices.csv'},
    {'name': 'Inflation', 'file': 'inflation.csv', 'dir': 'historical'},
    {'name': 'Fuel Prices', 'file': 'fuel_prices.csv', 'dir': 'historical'},
    {'name': 'ACLED Security', 'file': 'acled_nigeria.csv', 'dir': 'historical'},
]


def render():
    st.title("Data Health")
    st.markdown("---")

    today = date.today().isoformat()
    st.info(f"Current date: {today}")

    # Check each data source
    statuses = []
    for source in DATA_SOURCES:
        status = _check_source(source, today)
        statuses.append(status)

    # Display as table
    for status in statuses:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        with col1:
            st.write(f"**{status['name']}**")
        with col2:
            if status['records'] > 0:
                st.success(f"{status['records']} records")
            else:
                st.error("0 records")
        with col3:
            st.write(status['last_date'] or "N/A")
        with col4:
            if status['days_since'] is not None:
                if status['days_since'] <= 1:
                    st.success(f"{status['days_since']} day(s) ago")
                elif status['days_since'] <= config.SCRAPE_FAIL_DAYS:
                    st.warning(f"{status['days_since']} day(s) ago")
                else:
                    st.error(f"{status['days_since']} day(s) ago")
            else:
                st.write("N/A")

    st.markdown("---")

    # Raw data directory listing
    st.subheader("Recent Raw Data Directories")
    raw_dirs = sorted(config.DATA_RAW.glob("*/"), reverse=True)[:10]

    if raw_dirs:
        for d in raw_dirs:
            files = list(d.glob("*.csv"))
            total_records = 0
            for f in files:
                try:
                    df = pd.read_csv(f)
                    total_records += len(df)
                except Exception:
                    pass
            st.write(f"  {d.name}: {len(files)} file(s), {total_records} total records")
    else:
        st.info("No raw data directories found. Run the data collection pipeline first.")

    # Pipeline log
    st.subheader("Recent Pipeline Log Entries")
    log_path = config.LOG_DIR / "pipeline.log"
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        recent = lines[-20:] if len(lines) > 20 else lines
        st.code(''.join(recent))
    else:
        st.info("No pipeline log found.")


def _check_source(source: dict, today: str) -> dict:
    """Check status of a single data source."""
    name = source['name']
    filename = source['file']

    # Check in daily raw directory
    if 'dir' not in source:
        daily_path = config.DATA_RAW / today / filename
        if daily_path.exists():
            try:
                df = pd.read_csv(daily_path)
                return {
                    'name': name,
                    'records': len(df),
                    'last_date': today,
                    'days_since': 0,
                }
            except Exception:
                pass

        # Check most recent directory
        raw_dirs = sorted(config.DATA_RAW.glob(f"*/{filename}"), reverse=True)
        if raw_dirs:
            last_dir = raw_dirs[0].parent.name
            try:
                df = pd.read_csv(raw_dirs[0])
                last_dt = datetime.strptime(last_dir, '%Y-%m-%d').date()
                days_since = (date.today() - last_dt).days
                return {
                    'name': name,
                    'records': len(df),
                    'last_date': last_dir,
                    'days_since': days_since,
                }
            except Exception:
                pass
    else:
        # Check historical directory
        hist_path = config.DATA_HISTORICAL / filename
        if hist_path.exists():
            try:
                df = pd.read_csv(hist_path)
                mod_time = datetime.fromtimestamp(hist_path.stat().st_mtime)
                days_since = (datetime.now() - mod_time).days
                return {
                    'name': name,
                    'records': len(df),
                    'last_date': mod_time.strftime('%Y-%m-%d'),
                    'days_since': days_since,
                }
            except Exception:
                pass

    return {
        'name': name,
        'records': 0,
        'last_date': None,
        'days_since': None,
    }
