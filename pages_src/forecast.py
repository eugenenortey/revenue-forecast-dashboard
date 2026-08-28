import streamlit as st

from utils.loaders import get_data
from utils import modeling as md
from utils import visualizations as viz

st.title("🔮 Forecast & Scenarios")
st.caption("Final model, refit on the complete cleaned history, projecting 12 months of 2026 revenue.")

with st.spinner("Loading data and fitting the final model..."):
    raw, data_dict, issues, result = get_data()
    prophet_df = md.to_prophet_df(result["chain_monthly"])
    final = md.fit_final_forecast(md.FINAL_PARAMS, prophet_df, horizon=md.FORECAST_HORIZON)

future = final["future"]
actual_2025 = prophet_df[prophet_df["ds"].dt.year == prophet_df["ds"].dt.year.max()]["y"].sum()
scenario = md.scenario_summary(future, actual_2025)

st.subheader("2026 scenarios")
c1, c2, c3 = st.columns(3)
c1.metric("Worst case (90% lower)", f"GHS {scenario['worst']/1e6:,.1f}m",
          f"{scenario['growth_worst_%']:+.1f}% vs. {prophet_df['ds'].dt.year.max()}")
c2.metric("Expected case", f"GHS {scenario['expected']/1e6:,.1f}m",
          f"{scenario['growth_expected_%']:+.1f}% vs. {prophet_df['ds'].dt.year.max()}")
c3.metric("Best case (90% upper)", f"GHS {scenario['best']/1e6:,.1f}m",
          f"{scenario['growth_best_%']:+.1f}% vs. {prophet_df['ds'].dt.year.max()}")

c4, c5 = st.columns(2)
c4.metric("Peak month", f"{scenario['peak_month']:%b %Y}", f"GHS {scenario['peak_value']/1e6:.1f}m")
c5.metric("Trough month", f"{scenario['trough_month']:%b %Y}", f"GHS {scenario['trough_value']/1e6:.1f}m")

st.plotly_chart(
    viz.plot_historical_with_forecast(prophet_df, future, split_date=md.SPLIT_DATE),
    width='stretch',
)

st.subheader("Monthly forecast detail")
table = future[["ds", "yhat_lower", "yhat", "yhat_upper"]].copy()
table["interval_width_%"] = ((table["yhat_upper"] - table["yhat_lower"]) / table["yhat"] * 100).round(0)
table.columns = ["Month", "Lower (90%)", "Forecast", "Upper (90%)", "Interval width %"]
st.dataframe(
    table.style.format({"Lower (90%)": "{:,.0f}", "Forecast": "{:,.0f}", "Upper (90%)": "{:,.0f}",
                         "Interval width %": "{:.0f}%"}),
    width='stretch', hide_index=True,
)
st.download_button("Download forecast as CSV", table.to_csv(index=False).encode("utf-8"),
                    file_name="2026_revenue_forecast.csv", mime="text/csv")

st.divider()
st.subheader("Forecast components")
st.plotly_chart(viz.plot_prophet_components(final["model"], final["forecast"]), width='stretch')
st.caption("Trend changepoints mark where Prophet detected a shift in growth rate, found from the "
           "data alone — no branch-opening dates were given to the model.")

st.divider()
st.subheader("Residual diagnostics (held-out 2025)")

with st.spinner("Fitting train/test model for diagnostics..."):
    train, test = md.split_train_test(prophet_df)
    final_test = md.fit_and_evaluate(md.FINAL_PARAMS, train, test, "Prophet tuned (FINAL)")

comparison = test.merge(
    final_test["test_forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left"
)
st.plotly_chart(viz.plot_residuals(comparison), width='stretch')
