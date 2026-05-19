"""
P4-T1: Evaluation Script
Computes mean +/- std for each metric across all group/commodity combinations.
Also computes % RMSE improvement of each group vs Group E (static baseline).
"""
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def load_metrics() -> pd.DataFrame:
    """Load the accumulated metrics log."""
    path = config.LOG_DIR / "metrics_log.csv"
    if not path.exists():
        logger.error("metrics_log.csv not found at %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (group, commodity), aggregate metrics with mean and std."""
    if df.empty:
        return df

    summary = df.groupby(['group', 'commodity']).agg(
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        mape_mean=('mape', 'mean'),
        mape_std=('mape', 'std'),
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        n_obs=('rmse', 'count'),
    ).reset_index()

    output_path = config.LOG_DIR / "evaluation_summary.csv"
    summary.to_csv(output_path, index=False)
    logger.info("Saved evaluation summary to %s", output_path)
    return summary


def improvement_vs_static(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % RMSE improvement of each group vs Group E (static baseline)."""
    if df.empty:
        return df

    summary = df.groupby(['group', 'commodity'])['rmse'].mean().reset_index()
    static = summary[summary['group'] == 'E_static'][['commodity', 'rmse']].rename(
        columns={'rmse': 'static_rmse'}
    )

    non_static = summary[summary['group'] != 'E_static']
    merged = non_static.merge(static, on='commodity', how='left')

    merged['improvement_pct'] = (
        (merged['static_rmse'] - merged['rmse']) / merged['static_rmse'] * 100
    ).round(2)

    output_path = config.LOG_DIR / "improvement_vs_static.csv"
    merged.to_csv(output_path, index=False)
    logger.info("Saved improvement comparison to %s", output_path)
    return merged


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = load_metrics()

    if df.empty:
        print("No metrics data available. Run the pipeline first.")
    else:
        print("=== Evaluation Summary ===")
        summary = summary_table(df)
        print(summary.to_string(index=False))

        print("\n=== Improvement vs Static Baseline ===")
        improvement = improvement_vs_static(df)
        print(improvement.to_string(index=False))
