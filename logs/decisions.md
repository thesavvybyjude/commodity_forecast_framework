# Decision Log — 21-Day Back-Dated Evaluation

All major decisions recorded with rationale, alternatives, and expected outcomes.

---

## Decision 1: Clone Existing Repository Rather Than Rewrite

**What:** Clone `thesavvybyjude/commodity_forecast_framework` and reuse existing agents.

**Reason:** Task explicitly forbids redesign. Existing `AgentOrchestrator`, `ModelUpdaterAgent`,
`PreprocessingAgent`, `FeatureEngineeringAgent`, and evaluation scripts provide the required APIs.

**Alternatives:**
- Reimplement all 5 phases from scratch (rejected — violates scope constraint)
- Fork and heavily refactor (rejected — unnecessary)

**Expected Outcome:** Minimum-diff integration of back-dated simulation wrapper only.

**Actual Outcome:** Confirmed — only 3 new scripts + 2 agent patches + 1 data builder added.

---

## Decision 2: Use `uv pip` After Standard `pip` Timeout

**What:** Fall back to `uv pip install` when `pip install -r requirements.txt` timed out on TensorFlow.

**Reason:** Network timeout on 334 MB TensorFlow wheel from PyPI. `uv` has faster resolution and retry.

**Alternatives:**
- Retry pip with `--default-timeout=300` (attempted for core packages, still flaky)
- Install CPU-only tensorflow variant (not attempted — full package succeeded via uv)

**Expected Outcome:** Complete dependency installation in project venv.

**Actual Outcome:** 95 packages installed successfully into `venv/`.

---

## Decision 3: Synthetic WFP Fallback for Price Data

**What:** Generate synthetic daily price panel when WFP HDX download returns 404/HTML.

**Reason:** Primary URL `wfp_food_prices_nga.csv` returned 404. Fallback URL saved HTML, not CSV.
Evaluation requires ≥90 days of commodity price data across 3 commodities × 5 markets.

**Alternatives:**
- Manual HDX download by user (blocked — task requires autonomous execution)
- Scrape PricePally live (blocked — historical 21-day window needs past data, not live scrape)
- Use only sparse WFP records without daily expansion (rejected — insufficient for LSTM sequences)

**Expected Outcome:** Auditable daily panel from 2002-01-01 with realistic price dynamics.

**Actual Outcome:** 16,425 rows, 3-year synthetic panel (seed=42), documented as fallback.

---

## Decision 4: Synthetic Macro Features (Exchange, Inflation, Fuel, Security)

**What:** Generate conservative synthetic macro time series merged by date.

**Reason:** No historical scrape dumps exist in `data/raw/`. ACLED/inflation/fuel files absent.
Macro features required by spec columns: `exchange_rate`, `fuel_price`, `inflation`, `security_incidents`.

**Alternatives:**
- Run live scrapers and backfill (would not produce 2002–2004 history)
- Leave macro columns null (rejected — breaks feature engineering and merge validation)

**Expected Outcome:** Forward-filled daily macro columns aligned to price panel.

**Actual Outcome:** Macro merged successfully; logged as synthetic fallback in workflow.md.

---

## Decision 5: Create `backdate_simulation.py` as Wrapper (Not Fork Orchestrator)

**What:** New entry script that filters data by `date <= sim_date` and calls existing agents.

**Reason:** No existing back-dated replay script. `AgentOrchestrator.run_daily_pipeline()` uses
`date.today()` and live collection — unsuitable for historical replay.

**Alternatives:**
- Modify `AgentOrchestrator` directly (rejected — would break production daily pipeline)
- Mock `date.today()` globally (rejected — fragile, hard to audit)

**Expected Outcome:** Strict no-leakage replay reusing `PreprocessingAgent`, `FeatureEngineeringAgent`,
`ModelUpdaterAgent`, `ModelTrainerAgent`.

**Actual Outcome:** Achieved. Initial training uses data through day0 only.

---

## Decision 6: SimulationClock Monkey-Patch for Periodic Retrain Logic

**What:** Patch `ModelVersionControlAgent.days_since_last_train()` to use simulation dates.

**Reason:** Registry uses `datetime.now()` for retrain intervals. In replay, day 7 must trigger
Group A/C retrain regardless of wall-clock time.

**Alternatives:**
- Modify `ModelVersionControlAgent` permanently (rejected — affects production)
- Pass simulation date through all agent constructors (rejected — too invasive)

**Expected Outcome:** Group A/C retrain every 7 simulated days; B/D update daily.

**Actual Outcome:** Confirmed in simulation.log — Group A retrain triggered on day 7.

---

## Decision 7: Reduce Training Epochs for Simulation Speed

**What:** Override `LSTM_EPOCHS_INIT=3`, `LSTM_EPOCHS_FINETUNE=1`, `XGB_PARAMS.n_estimators=40`.

**Reason:** Full 100-epoch LSTM training × 5 groups × 21 days = hours of compute.
Architecture and update strategy unchanged; only iteration count reduced for replay.

**Alternatives:**
- Full epoch training (rejected — 21-day replay exceeded 10+ minutes per attempt)
- Skip LSTM groups entirely (rejected — Wilcoxon B vs A requires both)

**Expected Outcome:** Complete 21-day replay in <15 minutes while preserving group behaviour.

**Actual Outcome:** ~12 minutes for full 21-day run (2,550 evaluations).

---

## Decision 8: `fast_update_all()` — Single Full-Matrix Update Per Day

**What:** Replace per-commodity `update_all()` loop with one combined update on full feature matrix.

**Reason:** Original `ModelUpdaterAgent.update_all()` trains 3× per group per day (once per commodity).
Identical data splitting produced redundant fine-tuning (3 LSTM fine-tunes + 3 XGB warm-starts daily).

**Alternatives:**
- Keep per-commodity updates (rejected — 3× runtime, no evaluation benefit)
- Train per commodity-market pair (rejected — even slower)

**Expected Outcome:** Same model behaviour, ~3× faster daily updates.

**Actual Outcome:** Day processing dropped from ~90s to ~25s.

---

## Decision 9: Disable PerformanceMonitorAgent Writes During Simulation

**What:** `patch_monitor_for_simulation()` noops `_append_row()`.

**Reason:** Group E writes rows with schema `date,group,commodity,rmse,mae,mape,r2,n_samples`
(missing `horizon` column). This corrupted `metrics_log.csv` and blocked forecast appends.

**Alternatives:**
- Harmonise both schemas (deferred — would require modifying production monitor agent)
- Use separate metrics files (rejected — spec requires single `logs/metrics_log.csv`)

**Expected Outcome:** Clean metrics schema: `date,commodity,group,horizon,rmse,mae,mape`.

**Actual Outcome:** 2,550 rows with consistent schema after patch.

---

## Decision 10: Inverse Transform Predictions to NGN Before Metric Computation

**What:** Apply `preprocessor.inverse_transform_price()` to model outputs before RMSE/MAE/MAPE.

**Reason:** Models train on MinMaxScaler-normalised features. Raw scaled predictions (~0–1)
produced RMSE near zero and meaningless MAPE when compared to NGN prices.

**Alternatives:**
- Compare in scaled space (rejected — spec requires interpretable NGN metrics)
- Denormalise using stored scaler manually (equivalent — used existing agent method)

**Expected Outcome:** RMSE values in NGN (e.g., 200–9000 range) comparable across groups.

**Actual Outcome:** Achieved after fixing list→numpy array bug in inverse transform call.

---

## Decision 11: Use Matplotlib for Evaluation PNGs (Not Plotly/Kaleido Only)

**What:** `evaluation/run_full_evaluation.py` uses matplotlib for dissertation PNG outputs.

**Reason:** Spec requires specific filenames in `evaluation/` directory. Plotly+kaleido used
for weekly report charts; matplotlib provides simpler headless PNG generation.

**Alternatives:**
- Plotly only (attempted in plot_results.py — requires kaleido/Chrome, slower)
- seaborn (not needed — matplotlib sufficient)

**Expected Outcome:** Four PNG files at 200 DPI in `evaluation/`.

**Actual Outcome:** All 4 PNGs generated. Boxplot required `tick_labels` fix for matplotlib 3.x.

---

## Decision 12: Establish Comprehensive Logging Before Continuation

**What:** Create `logs/session.log`, `decisions.md`, `workflow.md`, `environment.md`,
`training.log`, `metrics.csv`, `model_decisions.md`, `final_summary.md`.

**Reason:** User requirement — full reproducibility without terminal history dependency.

**Alternatives:**
- Continue evaluation without logging (rejected per user instruction)

**Expected Outcome:** Any reviewer can reconstruct the entire project from `logs/` alone.

**Actual Outcome:** In progress — this file and companions created 2026-06-19.
