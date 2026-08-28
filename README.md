# Revenue Forecasting Dashboard

Interactive Streamlit dashboard reproducing a Prophet-based branch revenue forecasting analysis —
a mock working-capital credit-facility engagement for a 30-branch Ghanaian retail chain.

## Overview

The source analysis (`Prophet_Branch_Revenue_Forecasting.ipynb`) starts from a messy monthly Excel
export with a serious, invisible defect — overlapping daily/weekly/monthly reporting rows for the
same branch-month, which silently double-counts revenue under naive aggregation — and works through
auditing it, cleaning it, and forecasting on the corrected series.

This dashboard is a faithful, **case-study reproduction**: it bundles the real dataset used in the
notebook (`data/Branch_Revenue_Forecasting_Raw_Dataset.xlsx`, synthetic/practice data, not a real
client's) and runs the same cleaning pipeline and the same Prophet configurations live, so every
number shown is computed from the data, not hardcoded. See `utils/data_processing.py` and
`utils/modeling.py` for the pipeline logic, ported directly from the notebook's own cells.

**Pages:**
- 🏠 **Overview** — engagement framing and headline numbers.
- 🧹 **Data Quality** — raw-export audit, the overlapping-granularity defect and its fix, threshold
  sensitivity testing, and a naive-vs-correct aggregation comparison.
- 🛠️ **Model Development** — STL trend/seasonality decomposition, a 48-configuration cross-validated
  grid search, and a baseline-vs-tuned benchmark comparison on a fully held-out year.
- 🔮 **Forecast & Scenarios** — the final model refit on the complete history, a 12-month 2026
  forecast with worst/expected/best-case scenarios, components, and residual diagnostics.
- 📋 **Findings & Limitations** — what the analysis supports, what it doesn't, and how the result
  should (and shouldn't) be used for a credit decision.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads `data/Branch_Revenue_Forecasting_Raw_Dataset.xlsx` automatically — no upload
step. Prophet fits are cached per session (`st.cache_resource`), so the first visit to each page
takes a few seconds; the 48-combination cross-validation grid search itself is *not* run live (it's
too slow to redo per session) — its results are precomputed and checked in at
`case_study/cv_grid_results.csv`. Regenerate it with:

```bash
python scripts/build_cv_results.py
```

## Project structure

```
revenue_forecast_dashboard/
├── app.py                        # Streamlit entry point (st.navigation)
├── pages_src/                    # one module per page
├── utils/
│   ├── data_processing.py        # raw-data audit + the 8-step cleaning pipeline
│   ├── modeling.py                # Prophet configs, metrics, benchmarks, scenarios
│   ├── visualizations.py         # Plotly chart builders
│   └── loaders.py                # shared cached data/pipeline loader
├── case_study/
│   └── cv_grid_results.csv       # precomputed 48-row CV grid search
├── scripts/
│   └── build_cv_results.py       # regenerates the CV grid artifact
├── data/
│   └── Branch_Revenue_Forecasting_Raw_Dataset.xlsx
└── docs/                         # separate static portfolio page (GitHub Pages)
```

## Model configuration

**Baseline** (Prophet defaults): additive seasonality, `changepoint_prior_scale=0.05`,
`interval_width=0.90`.

**Tuned (`FINAL_PARAMS`)**: `seasonality_mode='multiplicative'`,
`changepoint_prior_scale=0.10`, `seasonality_prior_scale=10.0`, `yearly_seasonality=10`. Chosen
from a 48-configuration rolling-origin CV grid search — but deliberately *not* the single
lowest-CV-RMSE row (see the Model Development page for why: the top row's more flexible
changepoint prior fits the training window best but risks over-fitting when extrapolating 12
months ahead).

Tuning improves MAPE from ~9.9% to ~5.5% on the held-out test year versus the untuned baseline.

## Key findings

1. **Data quality crisis.** The raw export mixes daily/weekly/monthly rows for the same
   branch-month; naively summing them overstates revenue substantially and distorts the trend's
   shape, not just its level. A statistical rule (retain rows within 0.4×–2.5× of the branch-month
   median) resolves it, and is stable under threshold sensitivity testing.
2. **Business growth.** Chain revenue grew steadily across the sample, with visible acceleration
   during a branch-expansion period that Prophet locates as a trend changepoint without being told
   where it was.
3. **Seasonality.** Stable and multiplicative — the December peak and August/September trough scale
   with the size of the business.
4. **Model performance.** The tuned model achieves single-digit MAPE on a fully held-out year,
   beating naive benchmarks by a wide margin.
5. **Limitations.** 90% prediction intervals are narrower than their nominal coverage on the
   held-out year; the forecast has no causal factors and is chain-level only. See the Findings page
   for the full list.

## References

- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.
- Prophet documentation: https://facebook.github.io/prophet/
- Source notebook: `Prophet_Branch_Revenue_Forecasting.ipynb`
