"""
P1-T9: Dataset Validator
Automated quality check before any modelling begins.
"""
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'date', 'commodity', 'market', 'region',
    'price_ngn', 'exchange_rate_usd_ngn',
    'inflation_rate', 'fuel_price_ngn_per_litre',
]


def validate(df: pd.DataFrame = None) -> dict:
    """Run validation checks on the master dataset."""
    if df is None:
        path = config.DATA_PROCESSED / "master_dataset_raw.csv"
        if not path.exists():
            logger.error("Master dataset not found at %s", path)
            return {'status': 'error', 'message': f'File not found: {path}'}
        df = pd.read_csv(path, parse_dates=['date'])

    report = {
        'status': 'ok',
        'checks': [],
        'warnings': [],
        'errors': [],
    }

    # Check 1: Missing columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report['errors'].append(f"Missing columns: {missing_cols}")
        report['checks'].append(f"FAIL — Missing columns: {missing_cols}")
    else:
        report['checks'].append("PASS — All required columns present")

    # Check 2: Date range coverage
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        min_date = df['date'].min()
        max_date = df['date'].max()
        report['checks'].append(f"Date range: {min_date} to {max_date}")

        if min_date and min_date.year > 2002:
            report['warnings'].append(
                f"Data starts at {min_date.year}, expected 2002 or earlier"
            )

    # Check 3: Null rates per column
    null_rates = (df.isnull().sum() / len(df) * 100).round(2)
    for col, rate in null_rates.items():
        if rate > 20:
            report['warnings'].append(f"High null rate: {col} = {rate}%")
        report['checks'].append(f"Null rate — {col}: {rate}%")

    # Check 4: Commodity coverage
    if 'commodity' in df.columns:
        commodities = df['commodity'].unique().tolist()
        missing_commodities = [c for c in config.COMMODITIES if c not in commodities]
        if missing_commodities:
            report['warnings'].append(f"Missing commodities: {missing_commodities}")
        report['checks'].append(f"Commodities found: {commodities}")
    else:
        report['errors'].append("No 'commodity' column")

    # Check 5: Market coverage
    if 'market' in df.columns:
        markets = df['market'].unique().tolist()
        missing_markets = [m for m in config.MARKETS if m not in markets]
        if missing_markets:
            report['warnings'].append(f"Missing markets: {missing_markets}")
        report['checks'].append(f"Markets found: {markets}")
    else:
        report['errors'].append("No 'market' column")

    # Check 6: Row count
    report['checks'].append(f"Total rows: {len(df)}")
    report['checks'].append(f"Total columns: {len(df.columns)}")

    # Set overall status
    if report['errors']:
        report['status'] = 'error'
    elif report['warnings']:
        report['status'] = 'warning'

    return report


def write_report(report: dict) -> Path:
    """Write validation report to file."""
    output_path = config.DATA_PROCESSED / "validation_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "DATASET VALIDATION REPORT",
        "=" * 60,
        f"Status: {report['status'].upper()}",
        "",
        "--- Checks ---",
    ]
    for check in report['checks']:
        lines.append(f"  {check}")

    if report['warnings']:
        lines.append("")
        lines.append("--- Warnings ---")
        for w in report['warnings']:
            lines.append(f"  [WARN] {w}")

    if report['errors']:
        lines.append("")
        lines.append("--- Errors ---")
        for e in report['errors']:
            lines.append(f"  [ERROR] {e}")

    lines.append("")
    lines.append("=" * 60)

    output_path.write_text('\n'.join(lines))
    logger.info("Validation report saved to %s", output_path)
    return output_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report = validate()
    path = write_report(report)

    print(f"\nValidation Status: {report['status'].upper()}")
    for check in report['checks']:
        print(f"  {check}")
    if report['warnings']:
        print("\nWarnings:")
        for w in report['warnings']:
            print(f"  [WARN] {w}")
    if report['errors']:
        print("\nErrors:")
        for e in report['errors']:
            print(f"  [ERROR] {e}")
    print(f"\nFull report: {path}")
