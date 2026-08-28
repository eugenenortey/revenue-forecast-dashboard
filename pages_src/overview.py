import streamlit as st

from utils.loaders import get_data
from utils import modeling as md

st.title("📈 Branch Revenue Forecasting")
st.caption("Prophet-based revenue forecasting for a 30-branch Ghanaian retail chain — "
           "built for a working-capital credit facility application.")

with st.spinner("Loading and cleaning the dataset..."):
    raw, data_dict, issues, result = get_data()

chain_monthly = result["chain_monthly"]
prophet_df = md.to_prophet_df(chain_monthly)

st.markdown("""
**The brief.** A retail chain needs a defensible 12-month revenue projection to support a
working-capital facility application. The finance team's monthly Excel exports are the only
available history — before any forecasting can happen, those exports need to be audited and
cleaned, because they turn out to contain a serious, invisible defect.
""")

st.subheader("Headline numbers")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw rows retained", f"{result['final_rows']:,} / {result['original_rows']:,}",
            f"{result['final_rows'] / result['original_rows'] * 100:.1f}%")
col2.metric("Clean monthly history", f"{len(chain_monthly)} months",
            f"{prophet_df['ds'].min():%b %Y} – {prophet_df['ds'].max():%b %Y}")
first_year_avg = prophet_df[prophet_df["ds"].dt.year == prophet_df["ds"].dt.year.min()]["y"].mean()
last_year_avg = prophet_df[prophet_df["ds"].dt.year == prophet_df["ds"].dt.year.max()]["y"].mean()
col3.metric("Avg. monthly revenue growth", f"GHS {last_year_avg/1e6:.1f}m",
            f"from GHS {first_year_avg/1e6:.1f}m in {prophet_df['ds'].dt.year.min()}")
overstatement = result["row_class_revenue_share"].get("coarser summary row", 0)
col4.metric("Naive aggregation overstatement", f"~{overstatement:.0f}% of revenue",
            "would be double-counted", delta_color="inverse")

st.divider()

st.subheader("What this dashboard shows")

st.markdown("**1. Data Quality** — the raw export's data-quality problems, and the row-by-row "
            "cleaning pipeline that resolves them. The critical finding: the export mixes daily, "
            "weekly and monthly reporting rows for the same branch-month, so a naive `sum()` "
            "double-counts revenue.")
st.page_link("pages_src/data_quality.py", label="Open Data Quality", icon="🧹")

st.markdown("**2. Model Development** — a Prophet baseline vs. a rolling-origin cross-validated "
            "tuned configuration, benchmarked against naive forecasts.")
st.page_link("pages_src/model_development.py", label="Open Model Development", icon="🛠️")

st.markdown("**3. Forecast & Scenarios** — the final model, refit on the full cleaned history, "
            "producing a 12-month 2026 forecast with worst/expected/best-case scenarios.")
st.page_link("pages_src/forecast.py", label="Open Forecast & Scenarios", icon="🔮")

st.markdown("**4. Findings & Limitations** — what the analysis supports, what it doesn't, and how "
            "the result should (and shouldn't) be used for a credit decision.")
st.page_link("pages_src/findings.py", label="Open Findings & Limitations", icon="📋")

st.caption("Every number on this dashboard is computed live from the bundled dataset using the same "
           "cleaning rules and model configuration as the source notebook — nothing here is hardcoded.")
