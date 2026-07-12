# Workflow Diary — 21-Day Back-Dated Evaluation

Step-by-step narrative of what was attempted, why, results, and next actions.

---

## Day 0 — Project Intake (2026-06-07)

**Attempted:** Read task specification for 21-day back-dated evaluation of commodity
forecasting framework.

**Why:** Understand scope before any code changes. Task requires audit-first approach.

**Result:** Identified 12 phases. Repository URL provided. Workspace empty.

**Next action:** Clone repository and audit existing components.

---

## Step 1 — Repository Clone & Audit

**Attempted:** `git clone https://github.com/thesavvybyjude/commodity_forecast_framework.git`

**Why:** Access existing implementation; task forbids rewrite.

**Result:** 57 Python files across 5 phases. Key findings:
- `AgentOrchestrator` — daily pipeline + initial training entry points
- `ModelUpdaterAgent` — 5 experimental groups (A–E)
- No `backdate_simulation.py`, no `merged_daily_data.csv`, no `run_full_evaluation.py`
- Data files gitignored (must be generated locally)

**Next action:** Install dependencies and verify/build dataset.

---

## Step 2 — Dependency Installation (First Attempt)

**Attempted:** `pip install -r requirements.txt` inside new venv.

**Why:** All agents require TensorFlow, XGBoost, pandas, etc.

**Result:** FAILED — pip ReadTimeoutError on TensorFlow wheel (334 MB).

**Next action:** Retry with `uv pip` which has faster package resolution.

---

## Step 3 — Dependency Installation (Second Attempt)

**Attempted:** `uv pip install -r requirements.txt --python venv\Scripts\python.exe`

**Why:** uv succeeded in a parallel global environment; apply to project venv.

**Result:** SUCCESS — 95 packages installed.

**Next action:** Download/build historical dataset.

---

## Step 4 — WFP Data Download

**Attempted:** `python data/wfp_downloader.py`

**Why:** WFP provides Nigeria food price baseline from 2002.

**Result:** FAILED — Primary CSV URL returns 404. Fallback URL saves HTML page.
`filter_and_save()` cannot parse HTML as CSV.

**Next action:** Create `data/build_merged_daily.py` with synthetic fallback documented.

---

## Step 5 — Build Merged Daily Dataset

**Attempted:** `python data/build_merged_daily.py`

**Why:** Spec requires `data/processed/merged_daily_data.csv` with columns:
date, commodity, market, price, exchange_rate, fuel_price, inflation, security_incidents.

**Result:** SUCCESS with documented fallbacks:
- 16,425 rows (3 commodities × 5 markets × ~1,095 days)
- Date range: 2002-01-01 to 2004-12-30
- Synthetic prices (seed=42), synthetic macro features
- Also wrote `master_dataset_raw.csv` for existing agent compatibility

**Next action:** Create back-dated simulation engine.

---

## Step 6 — Create Simulation Engine

**Attempted:** Write `backdate_simulation.py`

**Why:** No existing replay script. Must simulate 21 days with strict date filtering.

**Design:**
1. day0 = T-21, day1..day21 = simulation window
2. Initial training on data ≤ day0 only
3. Each day: preprocess → feature engineer → update models → forecast → evaluate
4. Append metrics to `logs/metrics_log.csv`
5. Weekly reports on days 7, 14, 21

**Result:** Script created. Supporting scripts: `run_weekly_report.py`, `evaluation/run_full_evaluation.py`.

**Next action:** Run simulation.

---

## Step 7 — Simulation Attempt 1 (XGBoost Inf Error)

**Attempted:** `python backdate_simulation.py`

**Result:** FAILED during initial XGBoost training.
Error: `Input data contains inf or a value too large`.

**Root cause:** `momentum_1` and `rate_of_change` features produce inf when dividing by zero.

**Next action:** Patch `PreprocessingAgent` and `FeatureEngineeringAgent` to sanitise inf values.

---

## Step 8 — Simulation Attempt 2 (Registry / Metrics Issues)

**Attempted:** Re-run after inf fix.

**Result:** FAILED at day 7. Zero forecast evaluations logged.
- `ModelVersionControlAgent` registry not reloaded after initial training
- `PerformanceMonitorAgent` wrote incompatible schema rows for Group E
- Weekly report crashed because metrics_log.csv had wrong format

**Next action:** Fix registry reload, disable monitor writes, fix inverse transform.

---

## Step 9 — Simulation Attempt 3 (Inverse Transform Bug)

**Attempted:** Re-run with registry fixes.

**Result:** FAILED at day 1 during `generate_forecasts()`.
Error: `'list' object has no attribute 'flatten'` in `inverse_transform_price()`.

**Root cause:** Passed Python list instead of numpy array to inverse transform.

**Next action:** Wrap prediction in `np.array([pred_scaled], dtype=float)`.

---

## Step 10 — Simulation Attempt 4 (Successful Run)

**Attempted:** `python backdate_simulation.py` with all fixes + optimisations.

**Optimisations applied:**
- `fast_update_all()` — one update per group per day (not per commodity)
- Reduced LSTM/XGBoost epochs for replay speed
- `patch_monitor_for_simulation()` — prevent schema corruption
- `SimulationClock` — correct periodic retrain timing

**Result:** SUCCESS
- 21 simulation days: 2004-12-10 to 2004-12-30
- 2,550 forecast evaluations logged
- Weekly reports generated for weeks 1–3
- State saved to `logs/simulation_state.json`

**Next action:** Run full evaluation for Chapter 4 artefacts.

---

## Step 11 — Full Evaluation

**Attempted:** `python evaluation/run_full_evaluation.py`

**Result:** FAILED — matplotlib not installed.

**Next action:** `uv pip install matplotlib` and re-run.

---

## Step 12 — Evaluation Completion

**Attempted:** Install matplotlib; run evaluation + weekly reports.

**Result:** SUCCESS (with one known limitation)
- `summary_table_1day.csv` — 15 rows (5 groups × 3 commodities)
- `summary_table_7day.csv` — 15 rows
- `summary_table_30day.csv` — 0 rows (30-day targets exceed dataset end date)
- `wilcoxon_results.csv` — 3 comparisons
- 4 PNG charts in `evaluation/`
- 3 weekly markdown reports in `reports/`

**Wilcoxon findings:**
- Group D vs E: p ≈ 0, significant (D better)
- Group D vs C: p ≈ 0, significant (D better)
- Group B vs A: p = 1.0, not significant

**Next action:** User requested logging infrastructure before continuing.

---

## Step 13 — Logging Infrastructure (Current Step)

**Attempted:** Create comprehensive `logs/` documentation per reproducibility requirements.

**Why:** User instruction: "before we continue" — nothing should exist only in terminal history.

**Result:** Created session.log, decisions.md, workflow.md, environment.md, training.log,
model_decisions.md, final_summary.md, metrics.csv, pip_freeze.txt, artefact_inventory.txt.

**Next action:** Await user confirmation to continue (e.g., fix 30-day horizon, commit, push).

---

## Open Items (Not Yet Addressed)

1. **30-day horizon evaluations:** 0 rows because simulation ends 2004-12-30 and
   30-day-ahead targets fall outside dataset for days after ~2004-12-01.
2. **Git commit:** Changes uncommitted per user rules (commit only when requested).
3. **WFP real data:** Synthetic fallback used; real HDX download URL may need update.
4. **Production monitor schema:** `PerformanceMonitorAgent` still uses different schema
   than simulation metrics — patched only during replay.
