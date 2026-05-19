"""
P4-T2: Statistical Significance Tests
Wilcoxon signed-rank test to prove statistical significance of continuous
learning vs static baseline.
"""
import logging
from pathlib import Path
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def wilcoxon_test(df: pd.DataFrame, group_a: str, group_b: str,
                  commodity: str) -> dict:
    """Run Wilcoxon signed-rank test comparing two groups for a commodity.

    Tests whether group_a has significantly lower RMSE than group_b.
    Uses one-tailed test (alternative='less').
    """
    a_data = df[(df['group'] == group_a) & (df['commodity'] == commodity)]['rmse'].values
    b_data = df[(df['group'] == group_b) & (df['commodity'] == commodity)]['rmse'].values

    # Pair by min length
    n_pairs = min(len(a_data), len(b_data))

    if n_pairs < 5:
        return {
            'group_a': group_a,
            'group_b': group_b,
            'commodity': commodity,
            'n_pairs': n_pairs,
            'statistic': None,
            'p_value': None,
            'significant': None,
            'interpretation': 'insufficient data (need at least 5 pairs)',
        }

    a = a_data[:n_pairs]
    b = b_data[:n_pairs]

    try:
        stat, p_value = stats.wilcoxon(a, b, alternative='less')
    except ValueError as e:
        return {
            'group_a': group_a,
            'group_b': group_b,
            'commodity': commodity,
            'n_pairs': n_pairs,
            'statistic': None,
            'p_value': None,
            'significant': None,
            'interpretation': f'test failed: {e}',
        }

    significant = p_value < 0.05

    if significant:
        interpretation = 'YES — continuous learning significantly better'
    else:
        interpretation = 'NO — no significant difference at p=0.05'

    result = {
        'group_a': group_a,
        'group_b': group_b,
        'commodity': commodity,
        'n_pairs': n_pairs,
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': significant,
        'interpretation': interpretation,
    }

    logger.info(
        "Wilcoxon test %s vs %s (%s): stat=%.2f, p=%.4f, significant=%s",
        group_a, group_b, commodity,
        stat, p_value, significant,
    )
    return result


def run_all_tests(df: pd.DataFrame, best_group: str = 'D_xgb') -> pd.DataFrame:
    """Run Wilcoxon tests for all commodities against the static baseline."""
    results = []
    for commodity in config.COMMODITIES:
        result = wilcoxon_test(df, best_group, 'E_static', commodity)
        results.append(result)

    results_df = pd.DataFrame(results)
    output_path = config.LOG_DIR / "statistical_tests.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Saved statistical tests to %s", output_path)
    return results_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    metrics_path = config.LOG_DIR / "metrics_log.csv"

    if not metrics_path.exists():
        print("No metrics data available. Run the pipeline first.")
    else:
        df = pd.read_csv(metrics_path)

        # Identify best group (lowest mean RMSE across all commodities)
        group_rmse = df.groupby('group')['rmse'].mean()
        non_static = group_rmse[group_rmse.index != 'E_static']
        if not non_static.empty:
            best_group = non_static.idxmin()
        else:
            best_group = 'D_xgb'

        print(f"Best performing group: {best_group}")
        print(f"Mean RMSE by group:\n{group_rmse}\n")

        results = run_all_tests(df, best_group)
        print("=== Statistical Test Results ===")
        print(results.to_string(index=False))
