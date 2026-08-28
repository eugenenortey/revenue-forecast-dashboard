import streamlit as st

st.title("📋 Findings & Limitations")

st.subheader("Key findings")

st.markdown("""
**On the data**
1. The raw export contains overlapping daily/weekly/monthly reporting rows for the same
   branch-month. A naive `groupby(month).sum()` overstates revenue by roughly the amount shown on
   the Data Quality page — and the distortion isn't constant over time, so it also corrupts the
   *shape* of the trend, not just its level.
2. The fix is a data-driven rule — retain rows within **0.4×–2.5×** of the branch-month median —
   rather than hard-coded per-branch logic. Threshold sensitivity testing shows the resulting
   monthly series moves by well under 1% across a wide band of alternative cut-offs.
3. The cleaned series is a continuous run of monthly observations with no gaps and no
   interpolation.

**On the business**
- Chain revenue grew from single-digit GHS millions per month at the start of the series to the
  mid-teens by the end, with growth visibly accelerating from 2023–2024 — Prophet locates a trend
  changepoint there from the data alone, independently corroborating the branch-expansion story.
- Seasonality is stable and multiplicative: December runs meaningfully above the annual average,
  August/September meaningfully below, and the size of that swing scales with the size of the
  business rather than staying a fixed GHS amount.

**On the model**
- The tuned model achieves single-digit MAPE on a fully held-out year and beats both a
  seasonal-naive and a drift-naive benchmark by a wide margin.
- Tuning improved every metric versus the untuned baseline — MAPE fell by roughly 44%, and bias
  moved from meaningfully positive (under-forecasting) to close to zero.
- All model selection (the CV grid search) happened strictly inside the training window; the test
  year was used exactly once, at the end, purely for evaluation.
""")

st.divider()
st.subheader("Limitations and risks")

st.markdown("""
1. **Trend extrapolation assumes continued momentum.** The growth acceleration Prophet found
   coincides with a real branch-opening wave. If no new branches open in 2026, actual growth will
   likely land below the expected case.
2. **Prophet is curve-fitting, not causal.** It has no knowledge of competitor entry, currency
   inflation, interest rates, or consumer confidence — nominal growth partly reflects inflation,
   not volume.
3. **Six years is a short sample for yearly seasonality** — six cycles is workable but not ideal.
4. **Prediction intervals are measurably too narrow.** The nominal 90% interval covered noticeably
   fewer than 90% of actual held-out months (see the Model Development page for the exact figure).
   In practice, the true downside is worse than the 90% lower bound alone suggests — a facility
   sized against that lower bound should carry an explicit safety margin.
5. **The uncertainty band covers trend/observation noise only** — not model risk, not the risk of a
   structural break, and not the risk that the granularity-classification rule mis-labelled some
   rows.
6. **Chain-level aggregation hides branch-level risk.** A strong chain-level forecast can mask one
   or two underperforming branches; this model has no branch-level view.
""")

st.markdown("**What the model can and cannot do**")
st.table({
    "Can": [
        "Project revenue if current conditions/momentum persist",
        "Quantify uncertainty under those conditions",
        "Identify when historical growth rates changed",
        "Give a defensible downside for facility sizing",
        "Reproduce the seasonal working-capital cycle",
    ],
    "Cannot": [
        "Predict the effect of a competitor opening nearby",
        "Account for a macroeconomic shock or currency crisis",
        "Explain *why* growth rates changed",
        "Guarantee a floor",
        "Forecast a single branch reliably",
    ],
})

st.divider()
st.subheader("Conclusion")

st.markdown("""
The engagement question was whether the chain could put a credible 12-month revenue projection in
front of its bank — and the answer is **yes, but only after the reporting-granularity defect was
found and fixed**. Finding that the client's own systems had been silently overstating revenue
history is arguably the highest-value output of this project on its own, independent of the
forecast that follows it.

On the cleaned series, Prophet recovers the branch-expansion period as a changepoint without being
told where it was, reproduces a stable multiplicative December peak, and forecasts the held-out
year to single-digit percentage error — with interval coverage that is better than the untuned
baseline but still short of nominal, which should be disclosed alongside the numbers rather than
smoothed over.

**Recommendations**
1. Present the **expected case** as the central projection; size the facility against the **90%
   lower bound** with the coverage caveat disclosed explicitly, and back it with the changepoint
   evidence that growth acceleration is measurable, not asserted.
2. State plainly in the credit application that the projection assumes continued branch-expansion
   momentum — revise it downward if no further branches are planned for 2026.
3. On data governance: this granularity defect will recur on every future export unless the source
   systems standardize on a single reporting period, or the export is given an explicit
   `period_type` column so cleaning no longer has to be inferred statistically.
""")
