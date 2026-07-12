"""
21-day back-dated historical replay simulation.
Reuses existing agents with strict date filtering — no future data leakage.
"""
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
import config
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.model_updater_agent import ModelUpdaterAgent
from agents.model_version_control_agent import ModelVersionControlAgent
from agents.preprocessing_agent import PreprocessingAgent

SIMULATION_DAYS = 21
HORIZONS = [1, 7, 30]
GROUP_LABELS = {
    'A': 'A_lstm',
    'B': 'B_lstm',
    'C': 'C_xgb',
    'D': 'D_xgb',
    'E': 'E_static',
}
LSTM_GROUPS = {'A', 'B'}
XGB_GROUPS = {'C', 'D', 'E'}

LOG_PATH = config.LOG_DIR / 'simulation.log'
METRICS_PATH = config.LOG_DIR / 'metrics_log.csv'
SIM_STATE_PATH = config.LOG_DIR / 'simulation_state.json'


class SimulationClock:
    """Patches time-sensitive registry lookups for historical replay."""

    def __init__(self, reference: pd.Timestamp):
        self.reference = pd.Timestamp(reference)
        self.last_train: dict[str, pd.Timestamp] = {}

    def days_since(self, key: str) -> int:
        if key not in self.last_train:
            return 999
        delta = self.reference - self.last_train[key]
        return max(0, delta.days)

    def mark_trained(self, key: str):
        self.last_train[key] = self.reference


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, mode='w'),
            logging.StreamHandler(),
        ],
    )


def load_dataset() -> pd.DataFrame:
    path = config.DATA_PROCESSED / 'merged_daily_data.csv'
    if not path.exists():
        from data.build_merged_daily import build_merged_daily
        build_merged_daily()
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df


def to_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map evaluation schema to agent-expected column names."""
    master = df.rename(columns={
        'price': 'price_ngn',
        'exchange_rate': 'exchange_rate_usd_ngn',
        'inflation': 'inflation_rate',
        'fuel_price': 'fuel_price_ngn_per_litre',
    }).copy()
    market_region = {
        'Lagos': 'South West', 'Kano': 'North West', 'Kaduna': 'North West',
        'Abuja': 'Federal Capital Territory', 'Port Harcourt': 'South South',
    }
    master['region'] = master['market'].map(market_region).fillna('Unknown')
    return master


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(y_true == 0, 1, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def append_metric(sim_date: pd.Timestamp, commodity: str, group: str,
                  horizon: int, metrics: dict):
    row = {
        'date': sim_date.strftime('%Y-%m-%d'),
        'commodity': commodity,
        'group': group,
        'horizon': horizon,
        'rmse': round(metrics['rmse'], 6),
        'mae': round(metrics['mae'], 6),
        'mape': round(metrics['mape'], 4),
    }
    df_row = pd.DataFrame([row])
    if METRICS_PATH.exists():
        df_row.to_csv(METRICS_PATH, mode='a', header=False, index=False)
    else:
        df_row.to_csv(METRICS_PATH, index=False)


def patch_mvc_for_simulation(mvc: ModelVersionControlAgent, clock: SimulationClock):
    """Monkey-patch days_since_last_train to use simulation clock."""

    def _days_since(model_type: str, group: str) -> int:
        return clock.days_since(f'{model_type}_{group}')

    mvc.days_since_last_train = _days_since  # type: ignore[method-assign]

    orig_save_lstm = mvc.save_lstm
    orig_save_xgb = mvc.save_xgb

    def save_lstm(model, group, metrics=None):
        path = orig_save_lstm(model, group, metrics)
        clock.mark_trained(f'lstm_{group}')
        return path

    def save_xgb(model, group, metrics=None):
        path = orig_save_xgb(model, group, metrics)
        clock.mark_trained(f'xgb_{group}')
        return path

    mvc.save_lstm = save_lstm  # type: ignore[method-assign]
    mvc.save_xgb = save_xgb  # type: ignore[method-assign]


def patch_monitor_for_simulation(updater: ModelUpdaterAgent):
    """Prevent PerformanceMonitorAgent from writing incompatible metrics rows."""

    def _noop_append(row: dict):
        return

    updater.monitor._append_row = _noop_append  # type: ignore[method-assign]


def reset_artifacts():
    """Clear prior simulation outputs."""
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()
    for d in [config.MODEL_LSTM_DIR, config.MODEL_XGB_DIR, config.SCALER_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    if config.REGISTRY_PATH.exists():
        config.REGISTRY_PATH.unlink()


def train_initial_models(features: pd.DataFrame, feat_cols: list, trainer: ModelTrainerAgent):
    numeric = [c for c in feat_cols if c in features.columns
               and features[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    X = features[numeric].values
    y = features['price_ngn'].values
    for group in ['A', 'B']:
        trainer.train_lstm(X, y, group)
    for group in ['C', 'D', 'E']:
        trainer.train_xgboost(X, y, group)


def predict_xgb(model, row_features: np.ndarray) -> float:
    return float(model.predict(row_features.reshape(1, -1))[0])


def predict_lstm(model, seq: np.ndarray) -> float:
    pred = model.predict(seq.reshape(1, *seq.shape), verbose=0)
    return float(pred.flatten()[0])


def get_actual_price(full_df: pd.DataFrame, sim_date: pd.Timestamp,
                     commodity: str, market: str, horizon: int) -> float | None:
    target = sim_date + pd.Timedelta(days=horizon)
    subset = full_df[
        (full_df['date'] == target)
        & (full_df['commodity'] == commodity)
        & (full_df['market'] == market)
    ]
    if subset.empty:
        return None
    return float(subset.iloc[0]['price_ngn'])


def generate_forecasts(sim_date: pd.Timestamp, features: pd.DataFrame,
                       feat_cols: list, full_master: pd.DataFrame,
                       mvc: ModelVersionControlAgent, trainer: ModelTrainerAgent,
                       preprocessor: PreprocessingAgent):
    """Generate and evaluate 1/7/30-day forecasts for all groups."""
    import tensorflow as tf

    mvc.registry = mvc._load_registry()

    numeric = [c for c in feat_cols if c in features.columns
               and features[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    day_features = features[features['date'] <= sim_date]

    models = {}
    for g in LSTM_GROUPS:
        p = mvc.get_latest('lstm', g)
        models[g] = tf.keras.models.load_model(p) if p else None
    for g in XGB_GROUPS:
        p = mvc.get_latest('xgb', g)
        models[g] = joblib.load(p) if p else None

    forecast_count = 0
    for commodity in config.COMMODITIES:
        for market in config.MARKETS:
            cm_all = day_features[
                (day_features['commodity'] == commodity)
                & (day_features['market'] == market)
            ]
            if cm_all.empty:
                continue
            cm = cm_all[cm_all['date'] == cm_all['date'].max()]

            hist = features[
                (features['commodity'] == commodity)
                & (features['market'] == market)
                & (features['date'] <= sim_date)
            ].sort_values('date')

            if len(hist) < config.LSTM_SEQUENCE + 1:
                continue

            row = cm.iloc[0]
            x_row = row[numeric].values.astype(float)

            for group_key, label in GROUP_LABELS.items():
                model = models.get(group_key)
                if model is None:
                    continue

                for horizon in HORIZONS:
                    actual = get_actual_price(
                        full_master, sim_date, commodity, market, horizon,
                    )
                    if actual is None:
                        continue

                    if group_key in LSTM_GROUPS:
                        seq_data = hist[numeric].values[-config.LSTM_SEQUENCE:]
                        pred_scaled = predict_lstm(model, seq_data)
                    else:
                        pred_scaled = predict_xgb(model, x_row)

                    pred = float(
                        preprocessor.inverse_transform_price(
                            np.array([pred_scaled], dtype=float),
                        )[0]
                    )
                    metrics = compute_metrics([actual], [pred])
                    append_metric(sim_date, commodity, label, horizon, metrics)
                    forecast_count += 1

    return forecast_count


def fast_update_all(updater: ModelUpdaterAgent, features: pd.DataFrame, feat_cols: list):
    """Run model updates once on full feature matrix (simulation speed optimisation)."""
    numeric_feat_cols = [
        c for c in feat_cols
        if c in features.columns
        and features[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
    X = features[numeric_feat_cols].values
    y = features['price_ngn'].values
    updater._update_group_A(X, y, 'combined')
    updater._update_group_B(X, y, 'combined')
    updater._update_group_C(X, y, 'combined')
    updater._update_group_D(X, y, 'combined')
    updater._update_group_E(X, y, 'combined')


def run_simulation():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    setup_logging()
    logger = logging.getLogger('backdate_simulation')
    logger.info('=== 21-DAY BACK-DATED SIMULATION START ===')

    # Fast training for replay (preserves architecture, reduces epochs)
    config.LSTM_EPOCHS_INIT = 3
    config.LSTM_EPOCHS_FINETUNE = 1
    config.XGB_PARAMS = {**config.XGB_PARAMS, 'n_estimators': 40}

    reset_artifacts()

    raw = load_dataset()
    unique_dates = sorted(raw['date'].unique())
    if len(unique_dates) < SIMULATION_DAYS + 31:
        raise ValueError(
            f'Need at least {SIMULATION_DAYS + 31} unique dates; got {len(unique_dates)}'
        )

    day21 = pd.Timestamp(unique_dates[-1])
    day0 = day21 - pd.Timedelta(days=SIMULATION_DAYS)
    sim_dates = [day0 + pd.Timedelta(days=d) for d in range(1, SIMULATION_DAYS + 1)]

    logger.info('Simulation window: %s to %s (day0=%s)', sim_dates[0], sim_dates[-1], day0)

    full_master = to_master_schema(raw)
    train_raw = raw[raw['date'] <= day0].copy()
    logger.info('Initial training rows (through day0): %d', len(train_raw))

    preprocessor = PreprocessingAgent()
    feature_engineer = FeatureEngineeringAgent()
    trainer = ModelTrainerAgent()
    updater = ModelUpdaterAgent()
    patch_monitor_for_simulation(updater)
    mvc = ModelVersionControlAgent()

    clock = SimulationClock(day0)
    patch_mvc_for_simulation(mvc, clock)
    patch_mvc_for_simulation(updater.mvc, clock)
    patch_mvc_for_simulation(trainer.mvc, clock)
    patch_mvc_for_simulation(updater.trainer.mvc, clock)

    # Initial training — data through day0 only
    train_master = to_master_schema(train_raw)
    cleaned = preprocessor.clean(train_master)
    features = feature_engineer.engineer(cleaned)
    feat_cols = feature_engineer.get_feature_columns(features)
    train_initial_models(features, feat_cols, trainer)
    updater.mvc.registry = updater.mvc._load_registry()
    trainer.mvc.registry = trainer.mvc._load_registry()
    logger.info('Initial baseline training complete (data <= %s)', day0.date())

    total_forecasts = 0
    for day_num, sim_date in enumerate(sim_dates, start=1):
        clock.reference = sim_date
        logger.info('--- Simulation day %d: %s ---', day_num, sim_date.date())
        updater.mvc.registry = updater.mvc._load_registry()

        available = raw[raw['date'] <= sim_date].copy()
        master = to_master_schema(available)
        cleaned = preprocessor.clean(master)
        features = feature_engineer.engineer(cleaned)
        feat_cols = feature_engineer.get_feature_columns(features)

        fast_update_all(updater, features, feat_cols)

        n_fc = generate_forecasts(
            sim_date, features, feat_cols, full_master, updater.mvc, trainer,
            preprocessor,
        )
        total_forecasts += n_fc
        logger.info('Day %d: %d forecast evaluations logged', day_num, n_fc)

        if day_num in (7, 14, 21) and METRICS_PATH.exists():
            from run_weekly_report import generate_weekly_report
            week = day_num // 7
            generate_weekly_report(week)
            logger.info('Generated week %d report', week)

    state = {
        'day0': str(day0.date()),
        'day1': str(sim_dates[0].date()),
        'day21': str(sim_dates[-1].date()),
        'simulation_days': SIMULATION_DAYS,
        'total_forecasts': total_forecasts,
        'completed_at': datetime.now().isoformat(),
    }
    SIM_STATE_PATH.write_text(json.dumps(state, indent=2))
    logger.info('=== SIMULATION COMPLETE: %d forecasts ===', total_forecasts)
    return state


if __name__ == '__main__':
    run_simulation()
