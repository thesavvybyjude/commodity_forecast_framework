"""
Central configuration for the Commodity Price Forecasting Framework.
All paths, thresholds, and settings live here — never hardcode values inside modules.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_HISTORICAL = BASE_DIR / 'data' / 'historical'
MODEL_LSTM_DIR = BASE_DIR / 'models' / 'lstm'
MODEL_XGB_DIR = BASE_DIR / 'models' / 'xgboost'
SCALER_DIR = BASE_DIR / 'models' / 'scalers'
LOG_DIR = BASE_DIR / 'logs'
REGISTRY_PATH = BASE_DIR / 'model_registry.json'

# Scope
COMMODITIES = ['rice', 'maize', 'tomatoes']
MARKETS = ['Lagos', 'Kano', 'Kaduna', 'Abuja', 'Port Harcourt']

# LSTM hyperparameters
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
LSTM_SEQUENCE = 30       # look-back window in days
LSTM_LR_INIT = 0.001
LSTM_LR_FINETUNE = 0.0001
LSTM_EPOCHS_INIT = 100
LSTM_EPOCHS_FINETUNE = 10
LSTM_PATIENCE = 10

# XGBoost hyperparameters
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'random_state': 42,
}
XGB_FINETUNE_ROUNDS = 100

# Continuous learning triggers
PERIODIC_RETRAIN_DAYS = 7
FINETUNE_MIN_SAMPLES = 10

# Alert thresholds
PRICE_SPIKE_PCT = 15     # % price change over 7 days
RMSE_DEGRADATION = 1.5   # multiplier vs baseline RMSE before rollback
SCRAPE_FAIL_DAYS = 2     # consecutive days of zero records

# Credentials (fill these in — never commit to public repos)
EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'your-email@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your-app-password')
SENDGRID_KEY = os.environ.get('SENDGRID_KEY', '')
ACLED_KEY = os.environ.get('ACLED_KEY', '')

# Flask
FLASK_PORT = 5000
FLASK_HOST = '0.0.0.0'
