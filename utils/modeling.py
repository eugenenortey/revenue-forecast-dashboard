"""Prophet model configuration, evaluation, and forecasting.

Ports Sections 10-17 of the source notebook: baseline vs. cross-validated
tuned Prophet configs, the evaluation-metric formulas, naive benchmarks,
and the final 12-month forecast with scenario bands.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

CV_RESULTS_PATH = Path(__file__).resolve().parent.parent / "case_study" / "cv_grid_results.csv"

SPLIT_DATE = "2025-01-01"
FORECAST_HORIZON = 12

BASELINE_PARAMS = dict(
    growth="linear",
    seasonality_mode="additive",       # Prophet default — deliberately not yet tuned
    changepoint_prior_scale=0.05,      # Prophet default
    seasonality_prior_scale=10.0,      # Prophet default
    yearly_seasonality=True,
    weekly_seasonality=False,          # impossible to identify from monthly data
    daily_seasonality=False,           # impossible to identify from monthly data
    interval_width=0.90,               # 90% band for credit purposes
)

# Chosen from a 48-combination rolling-origin CV grid search (case_study/cv_grid_results.csv),
# NOT the single lowest-RMSE row (rank 10 of 48, +3.6% RMSE vs. the best config). The top row
# (changepoint_prior_scale=0.50) fits the training window best but is too flexible for a
# 12-month-ahead extrapolation; 0.10 trades a little in-sample CV RMSE for a more stable trend.
FINAL_PARAMS = dict(
    growth="linear",
    changepoint_prior_scale=0.10,
    seasonality_prior_scale=10.0,
    seasonality_mode="multiplicative",
    yearly_seasonality=10,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.90,
)

CV_SETTINGS = dict(initial="1278 days", period="183 days", horizon="365 days")
CV_PARAM_GRID = {
    "changepoint_prior_scale": [0.05, 0.10, 0.50],
    "seasonality_prior_scale": [1.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
    "yearly_seasonality": [3, 4, 6, 10],
}


def to_prophet_df(chain_monthly: pd.DataFrame) -> pd.DataFrame:
    df = (chain_monthly[["ds", "revenue"]]
          .rename(columns={"revenue": "y"})
          .sort_values("ds")
          .reset_index(drop=True))
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def split_train_test(prophet_df: pd.DataFrame, split_date: str = SPLIT_DATE):
    train = prophet_df[prophet_df["ds"] < split_date].copy()
    test = prophet_df[prophet_df["ds"] >= split_date].copy()
    return train, test


# --- Evaluation metrics (exact formulas from the notebook) ----------------------------------

def evaluate(y_true, y_pred, lower=None, upper=None, label=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    out = {
        "model": label,
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE_%": np.mean(np.abs(err / y_true)) * 100,
        "sMAPE_%": np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100,
        "Bias_%": np.mean(err / y_true) * 100,
    }
    if lower is not None:
        out["Coverage_%"] = np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))) * 100
    return out


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    # This month next year = same month last year.
    lookup = train.set_index("ds")["y"]
    preds = []
    for ds in test["ds"]:
        ref = ds - pd.DateOffset(years=1)
        preds.append(lookup.get(ref, np.nan))
    return np.array(preds)


def drift_naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    last_value = train["y"].iloc[-1]
    mean_diff = train["y"].diff().mean()
    return np.array([last_value + mean_diff * (i + 1) for i in range(len(test))])


def compute_benchmarks(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sn = seasonal_naive_forecast(train, test)
    rows.append(evaluate(test["y"], sn, label="Seasonal naive"))
    dn = drift_naive_forecast(train, test)
    rows.append(evaluate(test["y"], dn, label="Drift naive"))
    return pd.DataFrame(rows)


# --- Prophet fitting --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def fit_and_evaluate(params: dict, train: pd.DataFrame, test: pd.DataFrame, label: str) -> dict:
    model = Prophet(**params)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq="MS")
    forecast = model.predict(future)
    test_forecast = forecast[forecast["ds"].isin(test["ds"])].reset_index(drop=True)
    metrics = evaluate(test["y"].values, test_forecast["yhat"].values,
                        test_forecast["yhat_lower"].values, test_forecast["yhat_upper"].values,
                        label=label)
    return {"model": model, "forecast": forecast, "test_forecast": test_forecast, "metrics": metrics}


@st.cache_resource(show_spinner=False)
def fit_final_forecast(params: dict, prophet_df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> dict:
    model = Prophet(**params)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    future_only = forecast[forecast["ds"] > prophet_df["ds"].max()].reset_index(drop=True)
    return {"model": model, "forecast": forecast, "future": future_only}


def scenario_summary(future_forecast: pd.DataFrame, actual_prior_year: float | None = None) -> dict:
    expected = future_forecast["yhat"].sum()
    worst = future_forecast["yhat_lower"].sum()
    best = future_forecast["yhat_upper"].sum()
    out = {"worst": worst, "expected": expected, "best": best}
    if actual_prior_year:
        out["growth_worst_%"] = (worst / actual_prior_year - 1) * 100
        out["growth_expected_%"] = (expected / actual_prior_year - 1) * 100
        out["growth_best_%"] = (best / actual_prior_year - 1) * 100
    peak_row = future_forecast.loc[future_forecast["yhat"].idxmax()]
    trough_row = future_forecast.loc[future_forecast["yhat"].idxmin()]
    out["peak_month"] = peak_row["ds"]
    out["peak_value"] = peak_row["yhat"]
    out["trough_month"] = trough_row["ds"]
    out["trough_value"] = trough_row["yhat"]
    return out


# --- CV grid (precomputed — see scripts/build_cv_results.py) --------------------------------

@st.cache_data(show_spinner=False)
def load_cv_results() -> pd.DataFrame:
    if not CV_RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CV_RESULTS_PATH)


def cv_shortlist(cv_table: pd.DataFrame, tolerance: float = 1.05) -> pd.DataFrame:
    if cv_table.empty:
        return cv_table
    best_rmse = cv_table["cv_rmse"].min()
    return cv_table[cv_table["cv_rmse"] <= best_rmse * tolerance].sort_values("cv_rmse").reset_index(drop=True)
