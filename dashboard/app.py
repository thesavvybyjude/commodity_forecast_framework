"""
P5-T1: Streamlit App Entry Point
Main entry point — sets page config and routes navigation to sub-pages.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Nigeria Commodity Price Forecaster",
    page_icon="\U0001F33D",  # corn emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Price Forecasts", "Model Performance", "Data Health"],
)

if page == "Home":
    from dashboard.pages.home import render
    render()
elif page == "Price Forecasts":
    from dashboard.pages.forecasts import render
    render()
elif page == "Model Performance":
    from dashboard.pages.model_performance import render
    render()
elif page == "Data Health":
    from dashboard.pages.data_health import render
    render()
