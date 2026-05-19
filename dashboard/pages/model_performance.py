"""
P5-T3: Model Performance Page
Displays metrics_log.csv as interactive comparison charts.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from agents.performance_monitor_agent import PerformanceMonitorAgent


def render():
    st.title("Model Performance")
    st.markdown("---")

    monitor = PerformanceMonitorAgent()
    df = monitor.load_metrics()

    if df.empty:
        st.warning("No metrics data available. Run the pipeline first to generate metrics.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        commodities = df['commodity'].unique().tolist()
        selected_commodity = st.selectbox("Filter by Commodity", ['All'] + commodities)
    with col2:
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date.date(), max_date.date()),
                )

    # Apply filters
    filtered = df.copy()
    if selected_commodity != 'All':
        filtered = filtered[filtered['commodity'] == selected_commodity]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    # RMSE Over Time Line Chart
    st.subheader("RMSE Over Time (All Groups)")
    fig = px.line(
        filtered, x='date', y='rmse', color='group',
        title='RMSE Over Time by Model Group',
        labels={'rmse': 'RMSE', 'date': 'Date'},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.subheader("Performance Summary")
    summary = filtered.groupby('group').agg(
        mean_rmse=('rmse', 'mean'),
        mean_mae=('mae', 'mean'),
        mean_mape=('mape', 'mean'),
        mean_r2=('r2', 'mean'),
        observations=('rmse', 'count'),
    ).reset_index().round(4)

    st.dataframe(summary, use_container_width=True)

    # Improvement vs Group E
    st.subheader("Improvement vs Static Baseline (Group E)")
    static_rmse = filtered[filtered['group'] == 'E_static']['rmse'].mean()

    if pd.notna(static_rmse) and static_rmse > 0:
        improvement_data = []
        for group in filtered['group'].unique():
            if group == 'E_static':
                continue
            group_rmse = filtered[filtered['group'] == group]['rmse'].mean()
            improvement = ((static_rmse - group_rmse) / static_rmse * 100)
            improvement_data.append({
                'group': group,
                'improvement_pct': round(improvement, 2),
            })

        if improvement_data:
            imp_df = pd.DataFrame(improvement_data)
            fig = px.bar(
                imp_df, x='group', y='improvement_pct',
                title='RMSE Improvement vs Static Baseline',
                labels={'improvement_pct': 'Improvement (%)'},
                color='improvement_pct',
                color_continuous_scale='RdYlGn',
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Group E (static baseline) data available for comparison.")
