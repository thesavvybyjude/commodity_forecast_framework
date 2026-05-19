# Automated Agent-Based ML Framework for Commodity Price Forecasting | Nigeria

**Student:** Jude Thompson Aligit — U22CS1074  
**Institution:** Air Force Institute of Technology (AFIT), Kaduna  
**Supervisor:** Dr. M.M Lawal

---

## Overview

An automated system that:
1. **Collects** daily commodity price data from Nigerian online markets via web scrapers
2. **Processes** and enriches data with macro-economic features (exchange rates, inflation, fuel prices, security data)
3. **Trains** LSTM and XGBoost models with continuous learning strategies
4. **Compares** 5 experimental groups: periodic retraining vs fine-tuning vs static baseline
5. **Visualises** forecasts via a Streamlit dashboard
6. **Alerts** subscribers via email when prices spike

### Commodities
- Rice, Maize, Tomatoes

### Markets
- Lagos, Kano, Kaduna, Abuja, Port Harcourt

---

## Quick Start

```bash
# 1. Clone and navigate
git clone <repo-url>
cd commodity_forecast_framework

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
# Edit config.py — set EMAIL_SENDER, EMAIL_PASSWORD
# Or use environment variables

# 5. Run initial data setup
python data/wfp_downloader.py
python data/merge_datasets.py
python data/validate_dataset.py

# 6. Train baseline models
python -c "from agents.agent_orchestrator import AgentOrchestrator; AgentOrchestrator().run_initial_training()"

# 7. Start the dashboard
streamlit run dashboard/app.py

# 8. Start the API (for n8n triggers)
python api/flask_trigger.py
```

---

## Project Architecture

```
commodity_forecast_framework/
├── config.py                          ← Central configuration
├── model_registry.json                ← Auto-generated model tracking
├── requirements.txt
│
├── data/
│   ├── wfp_downloader.py              ← WFP historical data download
│   ├── merge_datasets.py              ← Merges all data sources
│   ├── validate_dataset.py            ← Dataset quality checks
│   ├── raw/                           ← Daily scrape dumps
│   ├── processed/                     ← Cleaned CSVs, feature matrix
│   └── historical/                    ← WFP baseline, ACLED, inflation
│
├── scrapers/
│   ├── beautifulsoup_agent.py         ← Exchange rates (AbokiFX)
│   ├── selenium_agent.py              ← Market prices (PricePally)
│   ├── api_connector.py               ← Inflation (World Bank API)
│   ├── fuel_scraper.py                ← Fuel prices (GlobalPetrolPrices)
│   └── acled_downloader.py            ← Security data (ACLED)
│
├── agents/
│   ├── data_collector_agent.py        ← Unified collection wrapper
│   ├── preprocessing_agent.py         ← Missing values, outliers, normalisation
│   ├── feature_engineering_agent.py   ← 40+ feature generation
│   ├── model_trainer_agent.py         ← Baseline LSTM + XGBoost training
│   ├── model_version_control_agent.py ← Model registry management
│   ├── performance_monitor_agent.py   ← Metrics logging (RMSE, MAE, MAPE, R²)
│   ├── model_updater_agent.py         ← Continuous learning engine
│   └── agent_orchestrator.py          ← Pipeline entry point
│
├── models/
│   ├── lstm_builder.py                ← LSTM architecture factory
│   ├── xgboost_builder.py             ← XGBoost config factory
│   ├── lstm_finetuner.py              ← Group B fine-tune logic
│   ├── xgboost_finetuner.py           ← Group D warm-start logic
│   ├── rollback_guard.py              ← Auto-rollback on degradation
│   ├── lstm/                          ← Saved .keras model files
│   ├── xgboost/                       ← Saved .joblib model files
│   └── scalers/                       ← MinMaxScaler objects
│
├── evaluation/
│   ├── evaluate.py                    ← Metrics summary tables
│   ├── statistical_tests.py           ← Wilcoxon signed-rank tests
│   ├── plot_results.py                ← Chapter 4 charts (Plotly)
│   └── generate_report.py             ← Auto-formatted report
│
├── dashboard/
│   ├── app.py                         ← Streamlit entry point
│   └── pages/
│       ├── home.py                    ← System overview
│       ├── forecasts.py               ← Live forecast charts
│       ├── model_performance.py       ← Metrics comparison
│       └── data_health.py             ← Scraper status
│
├── notifications/
│   ├── notifier.py                    ← Email alert engine
│   ├── subscriber_manager.py          ← SQLite subscriber list
│   └── email_templates/               ← HTML email templates
│
├── api/
│   └── flask_trigger.py               ← HTTP endpoint for n8n
│
├── n8n/workflows/
│   └── daily_workflow.json            ← 8 AM daily scheduler
│
└── logs/
    ├── pipeline.log
    ├── metrics_log.csv
    └── plots/
```

---

## 5 Experimental Groups

| Group | Model | Update Strategy | Purpose |
|-------|-------|----------------|---------|
| A | LSTM | Full retrain every 7 days | Periodic retraining baseline |
| B | LSTM | Fine-tune daily (lr=0.0001) | Lightweight continuous adaptation |
| C | XGBoost | Full retrain every 7 days | Periodic retraining (tree model) |
| D | XGBoost | Warm-start (+100 trees daily) | Lightweight continuous (tree model) |
| E | XGBoost | Never updated | Static baseline (control) |

---

## Daily Operations

```bash
# Terminal 1 — Flask API
python api/flask_trigger.py

# Terminal 2 — n8n Scheduler
docker start n8n

# Terminal 3 — Dashboard
streamlit run dashboard/app.py

# Manual pipeline trigger
curl -X POST http://localhost:5000/run-pipeline
```

---

## Evaluation

```bash
# After 2-4 weeks of accumulated data:
python evaluation/evaluate.py           # Summary tables
python evaluation/statistical_tests.py  # Wilcoxon tests
python evaluation/plot_results.py       # Charts for Chapter 4
python evaluation/generate_report.py    # Formatted report
```

---

## Environment

- **Python:** 3.9+
- **Key Libraries:** TensorFlow, XGBoost, Pandas, Scikit-learn, Streamlit, Plotly, Flask, Selenium, BeautifulSoup4
- **Automation:** n8n (via Docker)
- **Email:** Gmail SMTP with App Password

---

## License

This project is part of a final year dissertation at AFIT Kaduna.
