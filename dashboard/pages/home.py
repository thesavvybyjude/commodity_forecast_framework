"""
Dashboard Home Page
Overview of the Nigeria Commodity Price Forecasting system.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def render():
    st.title("Nigeria Commodity Price Forecaster")
    st.markdown("---")

    st.markdown("""
    ### Welcome
    This dashboard provides real-time commodity price forecasts for the Nigerian market,
    powered by an automated agent-based machine learning framework.

    **Commodities Tracked:**
    - Rice
    - Maize
    - Tomatoes

    **Markets Covered:**
    - Lagos, Kano, Kaduna, Abuja, Port Harcourt

    ### System Components
    - **Data Collection**: Automated daily scraping from multiple sources
    - **ML Models**: LSTM and XGBoost with continuous learning
    - **5 Experimental Groups**: Comparing periodic retraining vs fine-tuning strategies
    - **Alerts**: Email notifications for price spikes
    """)

    # Quick stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Commodities", len(config.COMMODITIES))
    with col2:
        st.metric("Markets", len(config.MARKETS))
    with col3:
        st.metric("Model Groups", 5)

    st.markdown("---")

    # Model groups explanation
    st.subheader("Experimental Groups")
    st.markdown("""
    | Group | Model | Update Strategy | Purpose |
    |-------|-------|----------------|---------|
    | A | LSTM | Full retrain every 7 days | Periodic retraining baseline |
    | B | LSTM | Fine-tune on new data daily | Lightweight continuous adaptation |
    | C | XGBoost | Full retrain every 7 days | Periodic retraining (tree model) |
    | D | XGBoost | Warm-start fine-tune daily | Lightweight continuous (tree model) |
    | E | XGBoost | Never updated | Static baseline (control) |
    """)
