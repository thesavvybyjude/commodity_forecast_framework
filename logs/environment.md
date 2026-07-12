# Environment Documentation
# Captured: 2026-06-19

## Operating System

| Property | Value |
|----------|-------|
| OS | Microsoft Windows 11 |
| Version | 10.0.26200 (win32) |
| Shell | PowerShell |
| Workspace | `c:\Users\thesa\Downloads\hi` |

## Python

| Property | Value |
|----------|-------|
| Version | Python 3.11.15 |
| Virtual env | `c:\Users\thesa\Downloads\hi\venv` |
| Package manager | pip 24.0 + uv 0.11.19 |

## GPU / CUDA

| Property | Value |
|----------|-------|
| NVIDIA GPU | Not detected (`nvidia-smi` unavailable) |
| CUDA | Not applicable — CPU-only execution |
| TensorFlow device | CPU (default) |

## Key Installed Packages

Full freeze saved to: `logs/pip_freeze.txt`

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.21.0 | LSTM models (Groups A, B) |
| keras | 3.14.1 | LSTM backend |
| xgboost | 3.2.0 | XGBoost models (Groups C, D, E) |
| pandas | 3.0.3 | Data processing |
| numpy | 2.4.6 | Numerical operations |
| scikit-learn | 1.9.0 | MinMaxScaler, metrics, train_test_split |
| scipy | 1.17.1 | Wilcoxon signed-rank tests |
| matplotlib | 4.x | Evaluation PNG charts |
| plotly | 6.8.0 | Weekly report charts |
| kaleido | 1.3.0 | Plotly PNG export |
| joblib | 1.5.3 | Model/scaler persistence |
| requests | 2.34.2 | WFP data download |
| streamlit | 1.58.0 | Dashboard (not run in evaluation) |
| flask | 3.1.3 | API (not run in evaluation) |
| selenium | 4.44.0 | Scrapers (not run — synthetic data used) |

## Environment Variables

| Variable | Value | Notes |
|----------|-------|-------|
| TF_CPP_MIN_LOG_LEVEL | 2 | Set in backdate_simulation.py to suppress TF info logs |
| EMAIL_SENDER | (default placeholder) | Not used in evaluation |
| EMAIL_PASSWORD | (default placeholder) | Not used in evaluation |
| ACLED_KEY | (empty) | Not used — synthetic security data |

## Repository

| Property | Value |
|----------|-------|
| Remote | https://github.com/thesavvybyjude/commodity_forecast_framework.git |
| Branch | devin/1779176834-initial-implementation |
| Latest commit | d4d9fad — feat: complete implementation of all 5 phases |
| Git status | Modified agents + new evaluation/simulation files (uncommitted) |

## Configuration Overrides (Simulation Only)

Applied in `backdate_simulation.py` at runtime — not persisted to `config.py`:

```python
LSTM_EPOCHS_INIT = 3        # default: 100
LSTM_EPOCHS_FINETUNE = 1    # default: 10
XGB_PARAMS['n_estimators'] = 40  # default: 500
```

Reason: Reduce replay runtime while preserving architecture and update strategy.
Documented in `logs/model_decisions.md`.

## Data Paths

| Path | Status |
|------|--------|
| data/processed/merged_daily_data.csv | 910,268 bytes, 16,425 rows |
| data/processed/master_dataset_raw.csv | 1,143,541 bytes |
| data/processed/feature_columns.json | 35 features |
| data/historical/wfp_nigeria_filtered.csv | Synthetic (fallback) |
| logs/metrics_log.csv | 148,686 bytes, 2,550 rows |
| logs/metrics.csv | Copy of metrics_log.csv |

## Reproduction Command Sequence

```powershell
cd c:\Users\thesa\Downloads\hi
python -m venv venv
uv pip install -r requirements.txt --python .\venv\Scripts\python.exe
uv pip install matplotlib --python .\venv\Scripts\python.exe
.\venv\Scripts\python.exe data\build_merged_daily.py
.\venv\Scripts\python.exe backdate_simulation.py
.\venv\Scripts\python.exe evaluation\run_full_evaluation.py
.\venv\Scripts\python.exe run_weekly_report.py
```
