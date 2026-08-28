import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import STL

from utils.loaders import get_data
from utils import modeling as md
from utils import visualizations as viz

st.title("🛠️ Model Development")
st.caption("Baseline Prophet vs. a rolling-origin cross-validated tuned configuration, "
           "benchmarked against naive forecasts on a fully held-out year.")

with st.spinner("Loading data and fitting models..."):
    raw, data_dict, issues, result = get_data()
    prophet_df = md.to_prophet_df(result["chain_monthly"])
    train, test = md.split_train_test(prophet_df)

st.subheader("Trend & seasonality")
st.markdown("Before fitting Prophet, an independent STL decomposition of the cleaned monthly "
            "series confirms what the model configuration should assume: a real upward trend, and "
            "a seasonal component whose amplitude grows with the level of the series — which is "
            "why the tuned model uses `seasonality_mode='multiplicative'` rather than the additive "
            "default.")
stl_series = prophet_df.set_index("ds")["y"]
stl_result = STL(stl_series, period=12, robust=True).fit()
st.plotly_chart(
    viz.plot_stl_decomposition(stl_series.index, stl_result.trend, stl_result.seasonal, stl_result.resid),
    width='stretch',
)

st.divider()
st.subheader("Train / test split")
c1, c2, c3 = st.columns(3)
c1.metric("Train period", f"{train['ds'].min():%b %Y} – {train['ds'].max():%b %Y}", f"{len(train)} months")
c2.metric("Test period", f"{test['ds'].min():%b %Y} – {test['ds'].max():%b %Y}", f"{len(test)} months")
c3.metric("Split date", md.SPLIT_DATE)
st.markdown("A chronological split — the test year is never touched during model selection. All "
            "tuning below happens via cross-validation **within the training period only**.")

st.divider()
st.subheader("1. Cross-validation grid search")

cv_table = md.load_cv_results()
if cv_table.empty:
    st.warning("CV grid results not found. Run `python scripts/build_cv_results.py` to generate "
               "`case_study/cv_grid_results.csv`.")
else:
    st.markdown(f"""
    A 48-combination grid over `changepoint_prior_scale`, `seasonality_prior_scale`,
    `seasonality_mode`, and `yearly_seasonality` (Fourier order), evaluated with rolling-origin
    cross-validation (`initial={md.CV_SETTINGS['initial']}`, `period={md.CV_SETTINGS['period']}`,
    `horizon={md.CV_SETTINGS['horizon']}`), ranked by CV RMSE.
    """)
    st.plotly_chart(viz.plot_cv_grid(cv_table, md.FINAL_PARAMS), width='stretch')

    shortlist = md.cv_shortlist(cv_table)
    best = cv_table.sort_values("cv_rmse").iloc[0]
    final_row = shortlist[
        (shortlist["changepoint_prior_scale"] == md.FINAL_PARAMS["changepoint_prior_scale"]) &
        (shortlist["seasonality_prior_scale"] == md.FINAL_PARAMS["seasonality_prior_scale"]) &
        (shortlist["seasonality_mode"] == md.FINAL_PARAMS["seasonality_mode"]) &
        (shortlist["yearly_seasonality"] == md.FINAL_PARAMS["yearly_seasonality"])
    ]
    final_rank = int(final_row["rank"].iloc[0]) if not final_row.empty else None
    pct_vs_best = (final_row["cv_rmse"].iloc[0] / best["cv_rmse"] - 1) * 100 if not final_row.empty else None

    st.markdown(f"""
    **Why the chosen configuration isn't the top row.** The single lowest-CV-RMSE configuration
    (`changepoint_prior_scale=0.50`, `seasonality_mode=multiplicative`, CV RMSE
    {best['cv_rmse']:,.0f}) fits the training window best, but a changepoint prior that flexible
    risks over-fitting recent wiggles when extrapolating 12 months ahead. **{len(shortlist)} of
    {len(cv_table)}** configurations sit within 5% of the best CV RMSE — statistically
    indistinguishable — so the final choice is made from that shortlist on structural grounds
    (multiplicative seasonality, since the December peak scales with the size of the business; a
    more conservative `changepoint_prior_scale=0.10` for extrapolation stability).

    The chosen `FINAL_PARAMS` ranks **{final_rank} of {len(cv_table)}** by CV RMSE
    (+{pct_vs_best:.1f}% vs. the best row) — a deliberate trade of a little in-sample CV accuracy
    for a more stable trend.
    """)

    with st.expander("Full CV grid (48 configurations)"):
        st.dataframe(cv_table, width='stretch', hide_index=True)

st.divider()
st.subheader("2. Model configurations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Baseline** (Prophet defaults)")
    st.json(md.BASELINE_PARAMS)
with col2:
    st.markdown("**Tuned (FINAL_PARAMS)**")
    st.json(md.FINAL_PARAMS)

st.divider()
st.subheader("3. Benchmark comparison on held-out 2025")

with st.spinner("Fitting baseline and tuned models..."):
    benchmarks = md.compute_benchmarks(train, test)
    baseline_result = md.fit_and_evaluate(md.BASELINE_PARAMS, train, test, "Prophet baseline")
    final_result = md.fit_and_evaluate(md.FINAL_PARAMS, train, test, "Prophet tuned (FINAL)")

metrics_df = pd.concat([benchmarks, pd.DataFrame([baseline_result["metrics"], final_result["metrics"]])],
                        ignore_index=True)
display_cols = ["model", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "Bias_%", "Coverage_%"]
st.dataframe(
    metrics_df[display_cols].style.format({
        "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE_%": "{:.2f}", "sMAPE_%": "{:.2f}",
        "Bias_%": "{:+.2f}", "Coverage_%": "{:.2f}",
    }),
    width='stretch', hide_index=True,
)
st.plotly_chart(viz.plot_benchmark_comparison(metrics_df), width='stretch')

mape_improvement = (baseline_result["metrics"]["MAPE_%"] - final_result["metrics"]["MAPE_%"]) / baseline_result["metrics"]["MAPE_%"] * 100
rmse_improvement = (baseline_result["metrics"]["RMSE"] - final_result["metrics"]["RMSE"]) / baseline_result["metrics"]["RMSE"] * 100
st.success(f"Tuning improved MAPE by {mape_improvement:.1f}% and RMSE by {rmse_improvement:.1f}% "
           f"versus the baseline. Bias fell from {baseline_result['metrics']['Bias_%']:+.1f}% to "
           f"{final_result['metrics']['Bias_%']:+.1f}%.")
st.caption(f"Note: baseline 90% interval coverage on the test year is "
           f"{baseline_result['metrics']['Coverage_%']:.1f}% — below the nominal 90%, and the "
           f"tuned model's {final_result['metrics']['Coverage_%']:.1f}% coverage is closer but "
           f"still short. See Findings & Limitations for what this means for facility sizing.")
