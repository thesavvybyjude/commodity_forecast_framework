"""
Generate weekly simulation reports (weeks 1-3).
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
import config

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).parent / 'reports'


def generate_weekly_report(week: int) -> Path:
    """Build reports/week{N}_report.md from metrics logged so far."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.LOG_DIR / 'metrics_log.csv'
    if not metrics_path.exists():
        logger.warning('No metrics yet for week %d — skipping report', week)
        out = REPORTS_DIR / f'week{week}_report.md'
        out.write_text(f'# Week {week} Report\n\nNo metrics logged yet.\n', encoding='utf-8')
        return out

    df = pd.read_csv(metrics_path)
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())
    end_idx = min(week * 7, len(dates)) - 1
    if end_idx < 0:
        end_idx = 0
    cutoff = dates[end_idx]
    subset = df[df['date'] <= cutoff]

    avg_rmse = subset['rmse'].mean()
    avg_mae = subset['mae'].mean()
    avg_mape = subset['mape'].mean()

    by_group = subset.groupby('group').agg(
        rmse=('rmse', 'mean'), mae=('mae', 'mean'), mape=('mape', 'mean'),
    ).round(4)

    chart_dir = REPORTS_DIR / 'charts'
    chart_dir.mkdir(parents=True, exist_ok=True)
    trend = subset.groupby(['date', 'group'])['rmse'].mean().reset_index()
    fig = px.line(trend, x='date', y='rmse', color='group',
                  title=f'Week {week} — RMSE Trend')
    chart_path = chart_dir / f'week{week}_rmse_trend.png'
    try:
        fig.write_image(str(chart_path))
    except Exception:
        fig.write_html(str(chart_path.with_suffix('.html')))

    lines = [
        f'# Week {week} Simulation Report',
        '',
        f'**Period end:** {cutoff.date()}',
        '',
        '## Summary Metrics',
        f'- Average RMSE: **{avg_rmse:.4f}**',
        f'- Average MAE: **{avg_mae:.4f}**',
        f'- Average MAPE: **{avg_mape:.2f}%**',
        '',
        '## Performance by Group',
        '```',
        by_group.to_string(),
        '```',
        '',
        '## RMSE Trend',
        f'Chart saved to `{chart_path}`',
        '',
        '## Observations',
        f'- {len(subset)} forecast evaluations logged through week {week}.',
        f'- Best group (lowest RMSE): **{by_group["rmse"].idxmin()}**',
        f'- Highest RMSE group: **{by_group["rmse"].idxmax()}**',
        '',
        '## Commentary',
        'Continuous-learning groups (B, D) are compared against periodic retraining '
        '(A, C) and the static baseline (E). RMSE trends indicate whether daily '
        'adaptation outperforms weekly full retrains during this replay window.',
    ]

    out = REPORTS_DIR / f'week{week}_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote %s', out)
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for w in (1, 2, 3):
        generate_weekly_report(w)
