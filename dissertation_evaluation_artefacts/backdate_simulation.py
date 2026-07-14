"""
backdate_simulation.py — 21-day back-dated evaluation simulation.

Groups A/B: MLP (approximates LSTM depth) trained on last 3-years window.
Groups C/D: XGBoost, 3-year window.
Group  E:   XGBoost, frozen at day0.

No future leakage: each day only uses data through sim_date.
"""
import sys, logging, warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config

for d in [config.LOG_DIR, config.LOG_DIR/"plots",
          ROOT/"reports", ROOT/"evaluation",
          config.MODEL_LSTM_DIR, config.MODEL_XGB_DIR, config.SCALER_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

log_path = config.LOG_DIR / "simulation.log"
_fh = logging.FileHandler(log_path, mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
root_log = logging.getLogger(); root_log.setLevel(logging.INFO)
root_log.handlers.clear(); root_log.addHandler(_fh); root_log.addHandler(_ch)
logger = logging.getLogger(__name__)

SIM_DAYS    = 21
HORIZONS    = [1, 7, 30]
COMMODITIES = config.COMMODITIES
MARKETS     = config.MARKETS
GROUPS      = ["A","B","C","D","E"]
RETRAIN_N   = config.PERIODIC_RETRAIN_DAYS
WINDOW_DAYS = 365 * 2    # 2-year rolling training window

FEATURE_COLS = [
    "month","quarter","day_of_week","day_of_year","year",
    "season","is_harvest","market_enc",
    "exchange_rate_usd_ngn","fuel_price_ngn",
    "inflation_rate","security_incidents",
    "price_lag_1","price_lag_7","price_lag_14","price_lag_30",
    "roll_mean_7","roll_mean_14","roll_mean_30",
    "roll_std_7","roll_std_14","roll_std_30",
    "momentum_7","momentum_30",
]

def _season(m):
    if m in [11,12,1,2]: return 0
    if m in [3,4]:       return 1
    if m in [5,6,7,8]:  return 2
    return 3

def build_features(df, commodity, window_days=None):
    sub = df[df["commodity"]==commodity].copy()
    if window_days:
        cutoff = sub["date"].max() - timedelta(days=window_days)
        sub = sub[sub["date"] >= cutoff]
    sub = sub.sort_values("date").reset_index(drop=True)
    if sub.empty: return sub

    sub["month"]       = sub["date"].dt.month
    sub["quarter"]     = sub["date"].dt.quarter
    sub["day_of_week"] = sub["date"].dt.dayofweek
    sub["day_of_year"] = sub["date"].dt.dayofyear
    sub["year"]        = sub["date"].dt.year
    sub["season"]      = sub["month"].apply(_season)
    sub["is_harvest"]  = (sub["month"]==10).astype(int)
    sub["market_enc"]  = pd.Categorical(sub["market"],categories=MARKETS).codes

    for col in ["exchange_rate_usd_ngn","fuel_price_ngn","inflation_rate","security_incidents"]:
        if col not in sub.columns: sub[col]=0.0

    for lag in [1,7,14,30]:
        sub[f"price_lag_{lag}"] = sub.groupby("market")["price_ngn"].shift(lag)

    for win in [7,14,30]:
        g = sub.groupby("market")["price_ngn"]
        sub[f"roll_mean_{win}"] = g.transform(lambda x: x.shift(1).rolling(win,min_periods=1).mean())
        sub[f"roll_std_{win}"]  = g.transform(lambda x: x.shift(1).rolling(win,min_periods=1).std().fillna(0))

    sub["momentum_7"]  = sub.groupby("market")["price_ngn"].transform(lambda x: x.pct_change(7).fillna(0))
    sub["momentum_30"] = sub.groupby("market")["price_ngn"].transform(lambda x: x.pct_change(30).fillna(0))
    return sub.dropna().reset_index(drop=True)

def get_XY(feat_df):
    cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    return feat_df[cols].values.astype(np.float32), feat_df["price_ngn"].values.astype(np.float32), cols

class SimulationEngine:
    def __init__(self):
        self.full_df = None; self.metrics = []
        self.models = {}; self.scalers = {}
        self.day0 = None; self.sim_start = None; self.sim_end = None
        self.last_retrain = {g: None for g in GROUPS}

    def load_data(self):
        path = config.DATA_PROCESSED / "master_dataset_raw.csv"
        self.full_df = pd.read_csv(path, parse_dates=["date"])
        logger.info("Loaded: %d rows, %s → %s", len(self.full_df),
                    self.full_df["date"].min().date(), self.full_df["date"].max().date())
        last = self.full_df["date"].max()
        self.sim_end   = last
        self.sim_start = last - timedelta(days=SIM_DAYS-1)
        self.day0      = self.sim_start - timedelta(days=1)
        logger.info("Window: %s → %s  (day0=%s)",
                    self.sim_start.date(), self.sim_end.date(), self.day0.date())

    def _train_mlp(self, X, y, key, lr=0.001, iters=100):
        sc = MinMaxScaler(); X_s = sc.fit_transform(X)
        rng = y.max()-y.min(); y_s = (y-y.min())/(rng+1e-8)
        mdl = MLPRegressor(hidden_layer_sizes=(64,32),activation="relu",
                           solver="adam",learning_rate_init=lr,max_iter=iters,
                           random_state=42,early_stopping=True,
                           validation_fraction=0.1,n_iter_no_change=15,
                           warm_start=False)
        mdl.fit(X_s,y_s)
        self.models[key] = ("mlp",mdl,sc,float(y.min()),float(rng))

    def _train_xgb(self, X, y, key, n=200):
        mdl = XGBRegressor(n_estimators=n,max_depth=5,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,
                           random_state=42,verbosity=0,n_jobs=-1)
        mdl.fit(X,y)
        self.models[key] = ("xgb",mdl)

    def train_initial_models(self):
        logger.info("="*60)
        logger.info("INITIAL TRAINING — day0 = %s", self.day0.date())
        train_df = self.full_df[self.full_df["date"]<=self.day0].copy()
        for comm in COMMODITIES:
            feat = build_features(train_df, comm, WINDOW_DAYS)
            if feat.empty: continue
            X,y,_ = get_XY(feat)
            logger.info("  %s: %d samples", comm, len(X))
            for g in ["C","D","E"]: self._train_xgb(X,y,(g,comm))
            for g in ["A","B"]:     self._train_mlp(X,y,(g,comm))
            logger.info("  All 5 groups trained for %s", comm)
        for g in GROUPS: self.last_retrain[g]=self.day0
        logger.info("Initial training complete.")

    def _predict_one(self, group, commodity, feat_df, horizon):
        key = (group,commodity)
        if key not in self.models or feat_df.empty: return np.nan
        last = feat_df.tail(1).copy()
        fd = pd.Timestamp(last["date"].values[0]) + timedelta(days=horizon)
        last=last.copy()
        last["month"]=fd.month; last["quarter"]=(fd.month-1)//3+1
        last["day_of_week"]=fd.dayofweek; last["day_of_year"]=fd.dayofyear; last["year"]=fd.year
        cols=[c for c in FEATURE_COLS if c in last.columns]
        Xp = last[cols].values.astype(np.float32)
        entry=self.models[key]
        if entry[0]=="mlp":
            _,mdl,sc,ymin,rng=entry
            ys=float(mdl.predict(sc.transform(Xp))[0])
            return float(ys*(rng+1e-8)+ymin)
        else:
            return float(entry[1].predict(Xp)[0])

    def update_models(self, sim_date, avail_df):
        for comm in COMMODITIES:
            feat = build_features(avail_df, comm, WINDOW_DAYS)
            if feat.empty or len(feat)<20: continue
            X,y,_ = get_XY(feat)
            # A: retrain every 7d
            if (sim_date-self.last_retrain["A"]).days>=RETRAIN_N:
                self._train_mlp(X,y,("A",comm))
                self.last_retrain["A"]=sim_date
                logger.info("  Group A retrained (%s)", comm)
            # B: fine-tune daily
            self._train_mlp(X,y,("B",comm),lr=0.0001,iters=30)
            # C: retrain every 7d
            if (sim_date-self.last_retrain["C"]).days>=RETRAIN_N:
                self._train_xgb(X,y,("C",comm))
                self.last_retrain["C"]=sim_date
                logger.info("  Group C retrained (%s)", comm)
            # D: warm-start daily
            key=("D",comm); prev_n=200
            if key in self.models and self.models[key][0]=="xgb":
                prev_n=self.models[key][1].n_estimators
            self._train_xgb(X,y,key,min(prev_n+30,600))
            # E: frozen

    def evaluate_forecast(self, sim_date, horizon, group, commodity, fcast):
        target = sim_date + timedelta(days=horizon)
        act = self.full_df[
            (self.full_df["date"].dt.normalize()==pd.Timestamp(target.date()))&
            (self.full_df["commodity"]==commodity)
        ]["price_ngn"].values
        if len(act)==0 or np.isnan(fcast): return None
        am=float(act.mean()); err=abs(fcast-am)
        mp=err/am*100 if am!=0 else 0.0
        return {"date":sim_date.date().isoformat(),
                "target_date":target.date().isoformat(),
                "commodity":commodity,"group":group,"horizon":horizon,
                "forecast":round(float(fcast),2),"actual":round(am,2),
                "rmse":round(err,4),"mae":round(err,4),"mape":round(mp,4),"r2":0.0}

    def run(self):
        self.load_data(); self.train_initial_models()
        sim_dates=pd.date_range(self.sim_start,self.sim_end,freq="D"); n=0
        for sim_date in sim_dates:
            day_num=(sim_date-self.sim_start).days+1
            logger.info("── Day %2d/21  (%s) ──", day_num, sim_date.date())
            avail=self.full_df[self.full_df["date"]<=sim_date].copy()
            self.update_models(sim_date,avail)
            for comm in COMMODITIES:
                feat=build_features(avail,comm)
                if feat.empty: continue
                for g in GROUPS:
                    for h in HORIZONS:
                        if sim_date+timedelta(days=h)>self.sim_end: continue
                        fcast=self._predict_one(g,comm,feat,h)
                        res=self.evaluate_forecast(sim_date,h,g,comm,fcast)
                        if res: self.metrics.append(res); n+=1
            logger.info("  Day %d done — %d forecasts total", day_num, n)
        logger.info("SIMULATION COMPLETE — %d forecasts", n)
        self._save_metrics(); return n

    def _save_metrics(self):
        if not self.metrics: return
        df=pd.DataFrame(self.metrics)
        out=config.LOG_DIR/"metrics_log.csv"
        df.to_csv(out,index=False)
        logger.info("Saved %d rows → %s", len(df), out)

if __name__=="__main__":
    e=SimulationEngine(); n=e.run()
    print(f"\n✓ {n} forecasts → logs/metrics_log.csv")
