"""
P5-T2: Forecasts Page
Shows historical prices and 7-day forward projection for selected
commodity, market, and model group.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from agents.model_version_control_agent import ModelVersionControlAgent


GROUP_OPTIONS = {
    'A (LSTM Periodic)': ('lstm', 'A'),
    'B (LSTM Fine-tune)': ('lstm', 'B'),
    'C (XGB Periodic)': ('xgb', 'C'),
    'D (XGB Fine-tune)': ('xgb', 'D'),
}


@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    """Load most recent features file."""
    feature_files = sorted(config.DATA_PROCESSED.glob("features_*.csv"), reverse=True)
    if feature_files:
        return pd.read_csv(feature_files[0], parse_dates=['date'])

    master = config.DATA_PROCESSED / "master_dataset_raw.csv"
    if master.exists():
        return pd.read_csv(master, parse_dates=['date'])

    return pd.DataFrame()


def render():
    st.title("Price Forecasts")
    st.markdown("---")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        commodity = st.selectbox("Commodity", config.COMMODITIES)
    with col2:
        market = st.selectbox("Market", config.MARKETS)
    with col3:
        group_label = st.selectbox("Model Group", list(GROUP_OPTIONS.keys()))

    model_type, group_key = GROUP_OPTIONS[group_label]

    # Load data
    df = load_features()
    if df.empty:
        st.warning("No data available. Run the pipeline first.")
        return

    # Filter
    mask = pd.Series(True, index=df.index)
    if 'commodity' in df.columns:
        mask &= df['commodity'] == commodity
    if 'market' in df.columns:
        mask &= df['market'] == market

    filtered = df[mask].copy()
    if filtered.empty:
        st.warning(f"No data for {commodity} in {market}.")
        return

    if 'date' in filtered.columns:
        filtered = filtered.sort_values('date')

    # Historical price chart
    fig = go.Figure()

    if 'price_ngn' in filtered.columns and 'date' in filtered.columns:
        fig.add_trace(go.Scatter(
            x=filtered['date'],
            y=filtered['price_ngn'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2E75B6'),
        ))

    # Generate forecast
    forecast_values = _generate_forecast(filtered, model_type, group_key)
    if forecast_values is not None and 'date' in filtered.columns:
        last_date = filtered['date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=8, freq='D')[1:]

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='7-Day Forecast',
            line=dict(color='#FF6B35', dash='dash'),
        ))

    fig.update_layout(
        title=f'{commodity.title()} Price Forecast — {market} ({group_label})',
        xaxis_title='Date',
        yaxis_title='Price (NGN)',
        hovermode='x unified',
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    if 'price_ngn' in filtered.columns:
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"NGN {filtered['price_ngn'].iloc[-1]:,.2f}")
        with col2:
            st.metric("30-Day Avg", f"NGN {filtered['price_ngn'].tail(30).mean():,.2f}")
        with col3:
            st.metric("Min (Period)", f"NGN {filtered['price_ngn'].min():,.2f}")
        with col4:
            st.metric("Max (Period)", f"NGN {filtered['price_ngn'].max():,.2f}")


def _generate_forecast(data: pd.DataFrame, model_type: str,
                       group_key: str) -> list:
    """Generate 7-day price forecast using the selected model."""
    mvc = ModelVersionControlAgent()
    model_path = mvc.get_latest(model_type, group_key)

    if model_path is None:
        st.info("No trained model available for this group. Run initial training first.")
        return None

    try:
        feat_cols_path = config.DATA_PROCESSED / "feature_columns.json"
        if feat_cols_path.exists():
            with open(feat_cols_path) as f:
                feat_cols = json.load(f)
        else:
            feat_cols = [c for c in data.columns if data[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        numeric_cols = [c for c in feat_cols if c in data.columns]
        if not numeric_cols:
            return None

        X = data[numeric_cols].values

        if model_type == 'lstm':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            if len(X) >= config.LSTM_SEQUENCE:
                X_recent = X[-config.LSTM_SEQUENCE:]
                X_input = X_recent.reshape(1, config.LSTM_SEQUENCE, X_recent.shape[1])
                pred = float(model.predict(X_input, verbose=0)[0][0])
            else:
                return None
        else:
            import joblib
            model = joblib.load(model_path)
            X_recent = X[-1:].reshape(1, -1)
            pred = float(model.predict(X_recent)[0])

        # 7-day linear extrapolation from single-step forecast
        forecasts = [pred * (1 + 0.002 * i) for i in range(7)]
        return forecasts

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        return None
