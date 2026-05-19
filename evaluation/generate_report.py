"""
P4-T4: Report Generator
Auto-formatted results summary for dissertation Chapter 4.
"""
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def generate_report() -> str:
    """Generate a formatted text report summarising all evaluation results."""
    lines = [
        "=" * 70,
        "EVALUATION REPORT — Commodity Price Forecasting Framework",
        "=" * 70,
        "",
    ]

    # Section 4.1: Dataset Summary
    lines.append("SECTION 4.1 — Dataset Summary")
    lines.append("-" * 40)

    validation_path = config.DATA_PROCESSED / "validation_report.txt"
    if validation_path.exists():
        lines.append(validation_path.read_text())
    else:
        lines.append("  [validation_report.txt not found]")

    lines.append("")

    # Section 4.2: Model Performance Table
    lines.append("SECTION 4.2 — Model Performance Table")
    lines.append("-" * 40)

    summary_path = config.LOG_DIR / "evaluation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        lines.append(summary.to_string(index=False))

        # Highlight best group per commodity
        lines.append("")
        lines.append("Best performing group per commodity:")
        for commodity in config.COMMODITIES:
            subset = summary[summary['commodity'] == commodity]
            if not subset.empty:
                best = subset.loc[subset['rmse_mean'].idxmin()]
                lines.append(
                    f"  {commodity}: {best['group']} "
                    f"(RMSE={best['rmse_mean']:.4f} +/- {best['rmse_std']:.4f})"
                )
    else:
        lines.append("  [evaluation_summary.csv not found — run evaluate.py first]")

    lines.append("")

    # Section 4.3: Improvement Over Static Baseline
    lines.append("SECTION 4.3 — Improvement Over Static Baseline")
    lines.append("-" * 40)

    improvement_path = config.LOG_DIR / "improvement_vs_static.csv"
    if improvement_path.exists():
        improvement = pd.read_csv(improvement_path)
        lines.append(improvement.to_string(index=False))
    else:
        lines.append("  [improvement_vs_static.csv not found]")

    lines.append("")

    # Section 4.4: Statistical Significance
    lines.append("SECTION 4.4 — Statistical Significance")
    lines.append("-" * 40)

    stats_path = config.LOG_DIR / "statistical_tests.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        for _, row in stats_df.iterrows():
            p_str = f"p={row['p_value']:.4f}" if pd.notna(row['p_value']) else "p=N/A"
            lines.append(
                f"  {row['commodity']}: {row['group_a']} vs {row['group_b']} — "
                f"{p_str} — {row['interpretation']}"
            )
    else:
        lines.append("  [statistical_tests.csv not found — run statistical_tests.py first]")

    lines.append("")

    # Section 4.5: Visualisations
    lines.append("SECTION 4.5 — Visualisations")
    lines.append("-" * 40)

    plots_dir = config.LOG_DIR / "plots"
    if plots_dir.exists():
        plots = list(plots_dir.glob("*.png"))
        lines.append(f"  {len(plots)} chart files generated in {plots_dir}/")
        for p in sorted(plots):
            lines.append(f"    - {p.name}")
    else:
        lines.append("  [No charts generated yet — run plot_results.py first]")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report = '\n'.join(lines)

    # Save report
    output_path = config.LOG_DIR / "evaluation_report.txt"
    output_path.write_text(report)
    logger.info("Report saved to %s", output_path)

    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report = generate_report()
    print(report)
