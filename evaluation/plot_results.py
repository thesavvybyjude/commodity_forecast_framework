"""
P4-T3: Results Visualisation
Generates the 4 chart types needed for Chapter 4 of the dissertation.
"""
import logging
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

PLOTS_DIR = config.LOG_DIR / "plots"


def ensure_plots_dir():
    """Create plots directory if needed."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_rmse_comparison(df: pd.DataFrame):
    """Chart 1: RMSE Bar Chart — grouped bar chart by group and commodity."""
    ensure_plots_dir()

    summary = df.groupby(['group', 'commodity'])['rmse'].mean().reset_index()

    fig = px.bar(
        summary,
        x='group',
        y='rmse',
        color='commodity',
        barmode='group',
        title='RMSE Comparison Across Experimental Groups',
        labels={'rmse': 'Mean RMSE', 'group': 'Model Group'},
        width=1200,
        height=600,
    )

    fig.write_html(str(PLOTS_DIR / "rmse_comparison.html"))
    try:
        fig.write_image(str(PLOTS_DIR / "rmse_comparison.png"))
    except Exception as e:
        logger.warning("PNG export failed (install kaleido): %s", e)

    logger.info("Saved RMSE comparison chart")


def plot_rmse_over_time(df: pd.DataFrame, commodity: str):
    """Chart 2: RMSE Over Time — line chart per commodity."""
    ensure_plots_dir()

    subset = df[df['commodity'] == commodity].copy()
    if subset.empty:
        logger.warning("No data for commodity: %s", commodity)
        return

    subset['date'] = pd.to_datetime(subset['date'])

    fig = px.line(
        subset,
        x='date',
        y='rmse',
        color='group',
        title=f'RMSE Over Time — {commodity.title()}',
        labels={'rmse': 'RMSE', 'date': 'Date'},
    )

    fig.write_html(str(PLOTS_DIR / f"rmse_over_time_{commodity}.html"))
    try:
        fig.write_image(str(PLOTS_DIR / f"rmse_over_time_{commodity}.png"))
    except Exception as e:
        logger.warning("PNG export failed: %s", e)

    logger.info("Saved RMSE over time chart for %s", commodity)


def plot_r2_heatmap(df: pd.DataFrame):
    """Chart 3: R-squared Heatmap — groups vs commodities."""
    ensure_plots_dir()

    pivot = df.groupby(['group', 'commodity'])['r2'].mean().reset_index()
    pivot_table = pivot.pivot(index='group', columns='commodity', values='r2')

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns.tolist(),
        y=pivot_table.index.tolist(),
        colorscale='Blues',
        text=[[f'{v:.3f}' for v in row] for row in pivot_table.values],
        texttemplate='%{text}',
        textfont={"size": 14},
    ))

    fig.update_layout(
        title='R-squared Heatmap by Group and Commodity',
        xaxis_title='Commodity',
        yaxis_title='Model Group',
        width=800,
        height=400,
    )

    fig.write_html(str(PLOTS_DIR / "r2_heatmap.html"))
    try:
        fig.write_image(str(PLOTS_DIR / "r2_heatmap.png"))
    except Exception as e:
        logger.warning("PNG export failed: %s", e)

    logger.info("Saved R-squared heatmap")


def plot_improvement_chart(improvement_df: pd.DataFrame):
    """Chart 4: Improvement vs Static Baseline bar chart."""
    ensure_plots_dir()

    if improvement_df.empty:
        logger.warning("No improvement data to plot")
        return

    fig = px.bar(
        improvement_df,
        x='group',
        y='improvement_pct',
        color='commodity',
        barmode='group',
        title='RMSE Improvement vs Static Baseline (Group E)',
        labels={'improvement_pct': 'Improvement (%)', 'group': 'Model Group'},
        width=1200,
        height=600,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig.write_html(str(PLOTS_DIR / "improvement_vs_static.html"))
    try:
        fig.write_image(str(PLOTS_DIR / "improvement_vs_static.png"))
    except Exception as e:
        logger.warning("PNG export failed: %s", e)

    logger.info("Saved improvement chart")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    metrics_path = config.LOG_DIR / "metrics_log.csv"

    if not metrics_path.exists():
        print("No metrics data available. Run the pipeline first.")
    else:
        df = pd.read_csv(metrics_path)

        print("Generating charts...")
        plot_rmse_comparison(df)

        for commodity in config.COMMODITIES:
            plot_rmse_over_time(df, commodity)

        plot_r2_heatmap(df)

        # Generate improvement chart
        from evaluation.evaluate import improvement_vs_static
        improvement = improvement_vs_static(df)
        plot_improvement_chart(improvement)

        print(f"Charts saved to {PLOTS_DIR}")
