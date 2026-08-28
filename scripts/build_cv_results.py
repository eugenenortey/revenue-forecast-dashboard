"""Regenerates case_study/cv_grid_results.csv.

Reproduces the notebook's Section 14 cross-validation grid search exactly:
48 Prophet configurations x rolling-origin CV (CV_SETTINGS), ranked by
cv_rmse. This is checked in rather than recomputed per Streamlit session
because a 48-combination CV grid search (each combo refit at every
rolling-origin cutoff) is too slow to redo on every app cold start.

Run from the repo root:
    python scripts/build_cv_results.py
"""
import itertools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from utils import data_processing as dp
from utils import modeling as md

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def main():
    raw, _, _ = dp.load_raw.__wrapped__()
    result = dp.run_pipeline.__wrapped__(raw)
    prophet_df = md.to_prophet_df(result["chain_monthly"])
    train, _ = md.split_train_test(prophet_df)

    keys = list(md.CV_PARAM_GRID.keys())
    combos = list(itertools.product(*md.CV_PARAM_GRID.values()))
    print(f"Running CV grid search: {len(combos)} combinations x {md.CV_SETTINGS}")

    rows = []
    t0 = time.time()
    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        model = Prophet(growth="linear", weekly_seasonality=False, daily_seasonality=False,
                         interval_width=0.90, **params)
        model.fit(train)
        cv = cross_validation(model, **md.CV_SETTINGS, disable_tqdm=True)
        perf = performance_metrics(cv, rolling_window=1)
        row = dict(params)
        row["cv_rmse"] = perf["rmse"].iloc[0]
        row["cv_mae"] = perf["mae"].iloc[0]
        row["cv_mape"] = perf["mape"].iloc[0] * 100 if "mape" in perf else None
        rows.append(row)
        elapsed = time.time() - t0
        print(f"  [{i}/{len(combos)}] {params} -> RMSE={row['cv_rmse']:,.0f}  ({elapsed:.0f}s elapsed)")

    cv_table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    cv_table.insert(0, "rank", range(1, len(cv_table) + 1))

    out_path = Path(__file__).resolve().parent.parent / "case_study" / "cv_grid_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    cv_table.to_csv(out_path, index=False)
    print(f"\nWrote {len(cv_table)} rows to {out_path}")

    best = cv_table.iloc[0]
    print(f"\nBest CV RMSE: {best['cv_rmse']:,.0f}  "
          f"(changepoint={best['changepoint_prior_scale']}, "
          f"seasonality_prior={best['seasonality_prior_scale']}, "
          f"mode={best['seasonality_mode']}, yearly={best['yearly_seasonality']})")

    shortlist = md.cv_shortlist(cv_table)
    print(f"Shortlist (within 5% of best RMSE): {len(shortlist)} of {len(cv_table)} configs")

    final_match = cv_table[
        (cv_table["changepoint_prior_scale"] == md.FINAL_PARAMS["changepoint_prior_scale"]) &
        (cv_table["seasonality_prior_scale"] == md.FINAL_PARAMS["seasonality_prior_scale"]) &
        (cv_table["seasonality_mode"] == md.FINAL_PARAMS["seasonality_mode"]) &
        (cv_table["yearly_seasonality"] == md.FINAL_PARAMS["yearly_seasonality"])
    ]
    if not final_match.empty:
        r = final_match.iloc[0]
        pct_vs_best = (r["cv_rmse"] / best["cv_rmse"] - 1) * 100
        print(f"FINAL_PARAMS rank: {int(r['rank'])}/{len(cv_table)}  "
              f"(CV RMSE {r['cv_rmse']:,.0f}, +{pct_vs_best:.1f}% vs. best)")


if __name__ == "__main__":
    main()
