import streamlit as st

st.set_page_config(page_title="Branch Revenue Forecasting", page_icon="📈", layout="wide")

overview = st.Page("pages_src/overview.py", title="Overview", icon="🏠", default=True)
data_quality = st.Page("pages_src/data_quality.py", title="Data Quality", icon="🧹")
model_development = st.Page("pages_src/model_development.py", title="Model Development", icon="🛠️")
forecast = st.Page("pages_src/forecast.py", title="Forecast & Scenarios", icon="🔮")
findings = st.Page("pages_src/findings.py", title="Findings & Limitations", icon="📋")

pg = st.navigation([overview, data_quality, model_development, forecast, findings])
pg.run()
