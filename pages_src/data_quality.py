import numpy as np
import pandas as pd
import streamlit as st

from utils.loaders import get_data
from utils import data_processing as dp
from utils import visualizations as viz

st.title("🧹 Data Quality")
st.caption("Auditing the raw export, then resolving its most serious defect: overlapping "
           "daily/weekly/monthly reporting rows for the same branch-month.")

with st.spinner("Loading and cleaning the dataset..."):
    raw, data_dict, issues, result = get_data()

rq = dp.raw_quality_report(raw)

st.subheader("1. Raw export audit")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows × columns", f"{rq['n_rows']:,} × {rq['n_cols']}")
c2.metric("Fully blank rows", rq["blank_rows"])
c3.metric("Exact duplicate rows", rq["exact_dupes"])
c4.metric("Disguised missing values", f"{int(rq['placeholder_counts'].sum()):,}",
          help="Text placeholders like 'N/A', 'unknown', 'TBD' that pandas' isna() doesn't catch.")

st.plotly_chart(viz.plot_missingness(rq["true_missing"], rq["n_rows"]), width='stretch')

with st.expander("Currency column conversion check"):
    st.markdown("Every currency column is loaded as text (`GHS` prefix, `₵` symbol, comma "
                "separators) and converted with a custom parser. `unexplained_loss` counts values "
                "that were present as text but failed to convert — it should be zero everywhere.")
    st.dataframe(rq["conversion_check"], width='stretch', hide_index=True)

with st.expander("Impossible values"):
    st.markdown(f"- Revenue ≤ 0: **{rq['revenue_le_zero']}** rows (a gross revenue figure cannot be "
                f"zero or negative)\n"
                f"- Headcount sentinels/negatives (-1, 999, 9999): **{rq['headcount_negative_or_sentinel']}**\n"
                f"- Negative footfall: **{rq['footfall_negative']}**")

st.divider()
st.subheader("2. The trap: overlapping reporting granularities")

st.markdown("""
The export's `Report_Month` + `Branch_Code` combination is **not unique**. For a branch reporting
daily, a single month can contain 28-31 rows *plus* a coarser weekly or monthly summary row for the
same period — all carrying the label `Monthly_Revenue_GHS`. Summed naively, this counts the same
revenue two or three times over.
""")

work_pre = result["work_pre_granularity"]
example = work_pre[(work_pre["branch"] == "BR-001") & (work_pre["date"].dt.to_period("M") == "2021-01")].copy()
if example.empty:
    # fall back to whichever branch-month actually has the most overlapping rows
    counts = work_pre.groupby(["branch", work_pre["date"].dt.to_period("M")]).size()
    top_branch, top_month = counts.idxmax()
    example = work_pre[(work_pre["branch"] == top_branch) &
                        (work_pre["date"].dt.to_period("M") == top_month)].copy()
else:
    top_branch, top_month = "BR-001", "2021-01"

median_rev = example["revenue"].median()
example["size_ratio"] = example["revenue"] / median_rev
example["row_class"] = np.select(
    [example["size_ratio"] > dp.GRANULARITY_UPPER, example["size_ratio"] < dp.GRANULARITY_LOWER],
    ["coarser summary row", "implausibly small"], default="base period observation")

st.markdown(f"**Example: {top_branch}, {top_month}** — {len(example)} raw rows for a single branch-month.")
st.dataframe(
    example[["date", "revenue", "size_ratio", "row_class"]].sort_values("date")
           .style.format({"revenue": "{:,.2f}", "size_ratio": "{:.2f}"}),
    width='stretch', hide_index=True,
)
st.caption("Rows near ratio ≈ 1 are the base (finest) reporting period for that branch-month; rows "
           "many times larger are coarser summaries of the same period stacked on top.")

st.subheader("Row classification across the whole dataset")
rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**Row counts by class**")
    st.dataframe(result["row_class_counts"].rename("rows"), width='stretch')
with rc2:
    st.markdown("**Share of total raw revenue by class**")
    st.dataframe(result["row_class_revenue_share"].rename("% of revenue"), width='stretch')

st.markdown(f"""
**The rule.** Within each branch-month, the *median* row size represents the finest period that
branch was reporting at — a day, a week or a month. Rows within
**[{dp.GRANULARITY_LOWER}×, {dp.GRANULARITY_UPPER}×]** of that median are kept as base-period
observations; rows above that band are coarser duplicate summaries, and rows below it are
implausibly small (e.g. a stray value entered in the wrong currency or off by a factor of 100).
""")

st.subheader("Threshold sensitivity")
st.markdown("Does the exact `[0.4×, 2.5×]` cutoff matter? Re-running the rule with several "
            "alternative threshold pairs shows the resulting monthly series barely moves.")
sens = dp.granularity_sensitivity(result["work_pre_granularity"])
st.dataframe(sens, width='stretch', hide_index=True)

st.divider()
st.subheader("3. Cleaning pipeline — audit trail")

st.plotly_chart(viz.plot_audit_trail(result["audit_log"], result["original_rows"]), width='stretch')
st.dataframe(result["audit_log"], width='stretch', hide_index=True)

st.metric("Final retained rows", f"{result['final_rows']:,} / {result['original_rows']:,}",
          f"{result['final_rows'] / result['original_rows'] * 100:.1f}% retained")

st.divider()
st.subheader("4. Naive vs. correct aggregation")

st.markdown("""
The chart below is the strongest evidence for why this cleaning step matters. The **naive** series
sums every raw row per month with no granularity resolution — exactly what a spreadsheet `SUMIFS`
or an un-audited `groupby().sum()` would produce. The **correct** series is the fully cleaned pipeline
output. The naive series doesn't just run high — it's *shaped wrong*, with step-changes that look
like real growth but are actually artifacts of when branches switched reporting frequency.
""")

naive = dp.get_naive_aggregation(raw)
correct = result["chain_monthly"].set_index("ds")["revenue"]
st.plotly_chart(viz.plot_naive_vs_correct(naive, correct), width='stretch')
