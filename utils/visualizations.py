"""Plotly chart builders for the dashboard pages."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "danger": "#c0392b",
    "warning": "#e67e22",
    "success": "#16a085",
    "muted": "#95a5a6",
    "band": "rgba(41, 128, 185, 0.18)",
}

TEMPLATE = "plotly_white"


def fmt_millions(series):
    return series / 1e6


def plot_missingness(true_missing: pd.Series, n_rows: int) -> go.Figure:
    pct = (true_missing / n_rows * 100).sort_values()
    pct = pct[pct > 0]
    fig = go.Figure(go.Bar(
        x=pct.values, y=pct.index, orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.1f}%" for v in pct.values], textposition="outside",
    ))
    fig.update_layout(template=TEMPLATE, title="Missing values by column (including disguised placeholders)",
                       xaxis_title="% missing", yaxis_title="", height=max(320, 24 * len(pct)))
    return fig


def plot_naive_vs_correct(naive: pd.Series, correct: pd.Series) -> go.Figure:
    naive_aligned = naive.reindex(correct.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=correct.index, y=fmt_millions(naive_aligned), mode="lines+markers",
                              name="Naive (sum every row) — WRONG", line=dict(color=COLORS["danger"], width=2)))
    fig.add_trace(go.Scatter(x=correct.index, y=fmt_millions(correct), mode="lines+markers",
                              name="Correct (granularity-resolved)", line=dict(color=COLORS["success"], width=2.5)))
    fig.update_layout(template=TEMPLATE, title="Naive vs. correct chain-level monthly revenue",
                       xaxis_title="", yaxis_title="GHS millions", height=440,
                       legend=dict(orientation="h", y=1.08))
    return fig


def plot_audit_trail(audit_log: pd.DataFrame, original_rows: int) -> go.Figure:
    steps = audit_log["step"].tolist()
    removed = audit_log["rows_removed"].tolist()
    running = original_rows
    bases, values = [], []
    for r in removed:
        bases.append(running - r)
        values.append(r)
        running -= r
    fig = go.Figure()
    fig.add_trace(go.Bar(x=steps, y=values, base=bases, marker_color=COLORS["danger"],
                          text=[f"-{v:,}" for v in values], textposition="outside", name="Rows removed"))
    fig.update_layout(template=TEMPLATE, title="Preprocessing audit trail — rows removed at each step",
                       xaxis_title="", yaxis_title="Rows remaining", height=440, showlegend=False)
    fig.update_xaxes(tickangle=-20)
    return fig


def plot_cv_grid(cv_table: pd.DataFrame, final_params: dict) -> go.Figure:
    df = cv_table.sort_values("cv_rmse").reset_index(drop=True)
    is_final = (
        (df["changepoint_prior_scale"] == final_params["changepoint_prior_scale"]) &
        (df["seasonality_prior_scale"] == final_params["seasonality_prior_scale"]) &
        (df["seasonality_mode"] == final_params["seasonality_mode"]) &
        (df["yearly_seasonality"] == final_params["yearly_seasonality"])
    )
    colors = [COLORS["success"] if f else COLORS["muted"] for f in is_final]
    fig = go.Figure(go.Bar(x=df.index + 1, y=df["cv_rmse"], marker_color=colors))
    fig.update_layout(template=TEMPLATE, title="CV grid search — 48 configurations ranked by CV RMSE "
                                                 "(green = chosen FINAL_PARAMS)",
                       xaxis_title="Rank", yaxis_title="Cross-validation RMSE (GHS)", height=420, showlegend=False)
    return fig


def plot_benchmark_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAPE %", "Bias %"))
    palette = [COLORS["muted"], COLORS["warning"], COLORS["accent"], COLORS["success"]]
    fig.add_trace(go.Bar(x=metrics_df["model"], y=metrics_df["MAPE_%"], marker_color=palette,
                          showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics_df["model"], y=metrics_df["Bias_%"], marker_color=palette,
                          showlegend=False), row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=420, title="Model comparison on held-out 2025")
    return fig


def plot_historical_with_forecast(prophet_df: pd.DataFrame, future_forecast: pd.DataFrame,
                                   split_date: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df["ds"], y=fmt_millions(prophet_df["y"]), mode="lines",
                              name="Actual", line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast["ds"], future_forecast["ds"][::-1]]),
        y=pd.concat([fmt_millions(future_forecast["yhat_upper"]), fmt_millions(future_forecast["yhat_lower"])[::-1]]),
        fill="toself", fillcolor=COLORS["band"], line=dict(color="rgba(0,0,0,0)"),
        name="90% interval", showlegend=True))
    fig.add_trace(go.Scatter(x=future_forecast["ds"], y=fmt_millions(future_forecast["yhat"]), mode="lines+markers",
                              name="Forecast", line=dict(color=COLORS["accent"], width=2.5, dash="dash")))
    if split_date:
        fig.add_vline(x=pd.Timestamp(split_date), line_dash="dot", line_color=COLORS["muted"])
    fig.update_layout(template=TEMPLATE, title="Historical revenue and 12-month forecast",
                       xaxis_title="", yaxis_title="GHS millions", height=460,
                       legend=dict(orientation="h", y=1.08))
    return fig


def plot_stl_decomposition(ds: pd.Series, trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=ds, y=trend, mode="lines", line=dict(color=COLORS["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=ds, y=seasonal, mode="lines", line=dict(color=COLORS["accent"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=ds, y=resid, mode="markers", marker=dict(color=COLORS["muted"], size=4)), row=3, col=1)
    fig.update_layout(template=TEMPLATE, height=560, showlegend=False, title="STL decomposition")
    return fig


def plot_prophet_components(model, forecast: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend (with changepoints)", "Yearly seasonality"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines",
                              line=dict(color=COLORS["primary"], width=2)), row=1, col=1)
    changepoints = model.changepoints
    cp_effects = model.params["delta"].mean(axis=0)
    sig = np.abs(cp_effects) > 0.01 * np.abs(cp_effects).max() if len(cp_effects) else []
    sig_cps = changepoints[sig] if len(sig) else []
    for cp in sig_cps:
        fig.add_vline(x=cp, line_dash="dot", line_color=COLORS["danger"], opacity=0.5, row=1, col=1)

    yearly = forecast[["ds", "yearly"]].copy() if "yearly" in forecast.columns else None
    if yearly is not None:
        yearly["month"] = pd.to_datetime(yearly["ds"]).dt.month
        monthly_avg = yearly.groupby("month")["yearly"].mean().reindex(range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig.add_trace(go.Scatter(x=month_names, y=monthly_avg.values, mode="lines+markers",
                                  line=dict(color=COLORS["accent"], width=2)), row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=560, showlegend=False, title="Forecast components")
    return fig


def plot_residuals(comparison: pd.DataFrame) -> go.Figure:
    residuals = comparison["y"] - comparison["yhat"]
    pct_err = residuals / comparison["y"] * 100
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Actual vs. forecast (with 90% interval)", "Residual (GHS)", "% error"))
    fig.add_trace(go.Scatter(x=comparison["ds"], y=fmt_millions(comparison["y"]), mode="lines+markers",
                              name="Actual", line=dict(color=COLORS["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=comparison["ds"], y=fmt_millions(comparison["yhat"]), mode="lines+markers",
                              name="Forecast", line=dict(color=COLORS["accent"], dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=comparison["ds"], y=fmt_millions(residuals), marker_color=COLORS["muted"],
                          showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=comparison["ds"], y=pct_err, marker_color=COLORS["warning"],
                          showlegend=False), row=3, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color=COLORS["danger"], row=3, col=1)
    fig.add_hline(y=-10, line_dash="dot", line_color=COLORS["danger"], row=3, col=1)
    fig.update_layout(template=TEMPLATE, height=700, title="Residual diagnostics — held-out 2025",
                       legend=dict(orientation="h", y=1.05))
    return fig
