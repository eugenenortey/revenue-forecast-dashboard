"""
Revenue Forecasting Dashboard
Interactive Streamlit dashboard for Prophet-based branch revenue forecasting.

Based on the Jupyter notebook: Prophet_Branch_Revenue_Forecasting.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils import data_processing, modeling, visualizations

# Page configuration
st.set_page_config(
    page_title="Revenue Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'clean_data' not in st.session_state:
        st.session_state.clean_data = None
    if 'monthly_revenue' not in st.session_state:
        st.session_state.monthly_revenue = None
    if 'cleaning_stats' not in st.session_state:
        st.session_state.cleaning_stats = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None
    if 'split_date' not in st.session_state:
        st.session_state.split_date = '2025-01-01'
    if 'model_params' not in st.session_state:
        st.session_state.model_params = None


def sidebar():
    """Render sidebar with navigation and controls."""
    st.sidebar.title("📈 Revenue Forecasting")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Overview", "📊 Data Quality", "🔮 Forecasting", "📉 Model Diagnostics"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Data input section
    st.sidebar.subheader("📁 Data Input")
    
    data_source = st.sidebar.radio(
        "Data source:",
        ["Sample Dataset", "Upload Excel File"],
        index=0
    )
    
    if data_source == "Upload Excel File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your Excel file",
            type=['xlsx', 'xls'],
            help="File should have a 'Raw_Data' sheet with revenue data"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Load Uploaded Data"):
                load_data(uploaded_file)
    else:
        # Use a generated public-safe demo dataset; real client data is never bundled.
        if st.sidebar.button("Load Demo Dataset") or not st.session_state.data_loaded:
            load_data(data_processing.create_demo_dataset())
        st.sidebar.caption("Demo data is synthetic. Upload the private Excel export for real analysis.")
    
    st.sidebar.markdown("---")
    
    # Show data status
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded successfully")
        if st.session_state.monthly_revenue is not None:
            n_obs = len(st.session_state.monthly_revenue)
            st.sidebar.info(f"📊 {n_obs} monthly observations")
    else:
        st.sidebar.info("⏳ No data loaded yet")
    
    return page


@st.cache_data
def load_data(file_path):
    """Load and clean data."""
    with st.spinner("Loading and cleaning data..."):
        try:
            # Load raw data
            raw_df, sheets = data_processing.load_excel_data(file_path)
            
            # Clean data
            clean_df, monthly_revenue, stats = data_processing.clean_data(raw_df)
            
            # Store in session state
            st.session_state.raw_data = raw_df
            st.session_state.clean_data = clean_df
            st.session_state.monthly_revenue = monthly_revenue
            st.session_state.cleaning_stats = stats
            st.session_state.data_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False


def page_overview():
    """Render overview page."""
    st.markdown('<div class="main-header">Revenue Forecasting for Working Capital and Branch Expansion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Facebook Prophet — End-to-End Time Series Forecasting</div>', unsafe_allow_html=True)
    
    # Project overview
    st.markdown("## 📋 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Client:** A multi-branch retail chain operating across Ghana. The chain grew from 20 branches
        (2020) to 30 branches (expansion completed during 2024).
        
        **Business Situation:** Management wants to finance a further round of branch expansion and
        needs a working-capital facility from its bank. The bank's credit committee requires a **credible,
        defensible 12-month revenue projection** with uncertainty estimates.
        
        **Analytical Task:**
        1. Reconstruct one clean, consistent, chain-level monthly revenue series from raw export
        2. Forecast the next 12 months with prediction intervals
        3. Translate the forecast into best-case / expected-case / worst-case scenarios
        """)
    
    with col2:
        st.info("""
        **Key Findings:**
        - Raw data had 100%+ double-counting issue
        - 72 monthly observations (2020-2025)
        - Growth accelerated 2023-2024
        - Stable 22% December seasonality
        - Single-digit MAPE on test data
        """)
    
    # Why Prophet
    st.markdown("## 🎯 Why Prophet for This Problem")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Changepoint Detection**
        
        Automatic detection of growth rate changes when 10 new branches opened in 2024
        """)
    
    with col2:
        st.markdown("""
        **Multiplicative Seasonality**
        
        Handles December trading peak that grows with business size
        """)
    
    with col3:
        st.markdown("""
        **Uncertainty Intervals**
        
        Built-in 90% prediction intervals for credit facility sizing
        """)
    
    # Data loading instruction
    if not st.session_state.data_loaded:
        st.markdown("---")
        st.markdown('<div class="warning-box">⚠️ <strong>Get Started:</strong> Use the sidebar to load the sample dataset or upload your own Excel file.</div>', unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown('<div class="success-box">✅ <strong>Data loaded!</strong> Navigate to other pages using the sidebar to explore data quality, generate forecasts, and view diagnostics.</div>', unsafe_allow_html=True)


def page_data_quality():
    """Render data quality page."""
    st.markdown('<div class="main-header">📊 Data Quality Assessment</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first using the sidebar.")
        return
    
    # Cleaning statistics
    st.markdown("## 🔍 From Raw Export to Forecast-Ready Series")
    st.markdown("""
    Revenue forecasting starts with a deliberately conservative transformation pipeline. Each stage
    below changes the data for a specific reason, records its impact, and passes only validated values
    to the next stage. The final output is one chain-level revenue value per month for Prophet.
    """)

    stats = st.session_state.cleaning_stats
    raw = st.session_state.raw_data

    # Rebuild stats from older cached sessions created before the audit fields existed.
    if stats is not None and 'raw_branch_codes' not in stats:
        clean_data, monthly_revenue, stats = data_processing.clean_data(raw)
        st.session_state.clean_data = clean_data
        st.session_state.monthly_revenue = monthly_revenue
        st.session_state.cleaning_stats = stats

    quality = data_processing.get_data_quality_report(raw)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw records", f"{stats['initial_rows']:,}")
    with col2:
        st.metric("Standardised branches", stats['unique_branches'],
                  delta=f"from {stats['raw_branch_codes']} raw codes")
    with col3:
        st.metric("Final monthly points", stats['final_monthly_observations'])
    with col4:
        st.metric("Date coverage", f"{stats['date_range'][0][:7]} to {stats['date_range'][1][:7]}")

    st.markdown("### Pipeline stages")
    pipeline = pd.DataFrame([
        ["1. Audit raw export", f"{stats['initial_rows']:,} rows", "Keep every source column as text so formatting issues remain visible."],
        ["2. Remove empty and exact duplicate rows", f"{stats['blank_rows_removed']:,} blank, {stats['duplicate_rows_removed']:,} duplicate removed", "Discard rows with no information and repeated records."],
        ["3. Standardise branch identifiers", f"{stats['raw_branch_codes']} to {stats['unique_branches']} branches", "Map case, spacing, punctuation, and BR/Branch variants to BR-XXX."],
        ["4. Parse and validate dates", f"{stats['unparseable_dates']:,} unparseable, {stats['placeholder_dates']:,} placeholder dates", "Parse known formats; exclude missing, broken, and 1900 placeholder dates."],
        ["5. Parse and validate revenue", f"{stats['unparseable_revenue']:,} unparseable, {stats['negative_revenue']:,} non-positive", "Strip GHS, cedi symbols, commas, and placeholders before numeric conversion."],
        ["6. Resolve reporting granularity", f"{stats['granularity_rows_removed']:,} overlapping rows removed", "Within each branch-month, retain values within 0.4x-2.5x of its median."],
        ["7. Aggregate for Prophet", f"{stats['final_monthly_observations']:,} monthly points", "Sum cleaned branch-month revenue, sort chronologically, and expose ds/y columns."],
    ], columns=["Stage", "Measured result", "Decision and purpose"])
    st.dataframe(pipeline, hide_index=True, use_container_width=True)

    with st.expander("🔬 Inspect the raw audit", expanded=False):
        audit_col1, audit_col2 = st.columns(2)
        with audit_col1:
            st.markdown("**Visible missing values by column**")
            visible_missing = pd.Series(quality['missing_visible'], name='NaN cells')
            st.dataframe(visible_missing[visible_missing > 0].sort_values(ascending=False), use_container_width=True)
        with audit_col2:
            st.markdown("**Placeholder values by column**")
            placeholder_missing = pd.Series(quality['missing_placeholders'], name='Placeholder cells')
            st.dataframe(placeholder_missing[placeholder_missing > 0].sort_values(ascending=False), use_container_width=True)

        st.markdown("**Date formats found in the raw export**")
        date_formats = pd.Series(quality.get('date_formats', {}), name='Rows').rename_axis('Format').reset_index()
        st.dataframe(date_formats, hide_index=True, use_container_width=True)

    st.markdown("### Model-ready output")
    model_ready = st.session_state.monthly_revenue.copy()
    model_ready['Month'] = model_ready['ds'].dt.strftime('%b %Y')
    model_ready['Revenue (GHS m)'] = (model_ready['y'] / 1e6).round(2)
    model_ready = model_ready[['Month', 'Revenue (GHS m)']]
    st.dataframe(model_ready, hide_index=True, use_container_width=True, height=260)
    st.caption("This chronological monthly series is the exact input used by the Prophet model. The train/test split and forecast settings are configured on the Forecasting page.")
    
    # Key finding: overlapping granularities
    st.markdown("## ⚠️ Critical Finding: Overlapping Granularities")
    
    st.markdown("""
    The raw data contained **overlapping reporting granularities**: weekly and daily detail rows sat
    alongside monthly summary rows for the same revenue period. A naive `groupby(month).sum()` would
    **double-count revenue by more than 100%** from 2021 onward.
    
    This corruption is invisible to standard data quality checks because:
    - All values are valid numbers
    - No missing data flags are raised
    - The resulting series looks plausible
    
    Our cleaning pipeline detects and removes these overlaps using a data-driven outlier filter
    (0.4x-2.5x of branch-month median).
    """)
    
    # Comparison visualization
    st.markdown("### Naive vs. Clean Aggregation")
    
    # Get naive aggregation for comparison
    naive_revenue = data_processing.get_naive_aggregation(st.session_state.raw_data)
    
    fig = visualizations.plot_data_quality_comparison(
        st.session_state.monthly_revenue,
        naive_revenue
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Impact metrics
    col1, col2, col3 = st.columns(3)
    
    naive_total = naive_revenue['y'].sum()
    clean_total = st.session_state.monthly_revenue['y'].sum()
    difference = naive_total - clean_total
    
    with col1:
        st.metric("Naive Total (INCORRECT)", f"GHS {naive_total/1e6:.1f}m")
    with col2:
        st.metric("Clean Total (CORRECT)", f"GHS {clean_total/1e6:.1f}m")
    with col3:
        st.metric("Overstatement", f"GHS {difference/1e6:.1f}m ({difference/clean_total*100:.1f}%)")
    
    # Historical revenue visualization
    st.markdown("## 📈 Historical Revenue Trend")
    
    fig = visualizations.plot_time_series_decomposition(st.session_state.monthly_revenue)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data summary
    with st.expander("📊 Data Summary Statistics", expanded=False):
        summary = st.session_state.monthly_revenue['y'].describe()
        summary_df = pd.DataFrame({
            'Statistic': summary.index,
            'Value (GHS)': summary.values,
            'Value (GHS millions)': summary.values / 1e6
        })
        st.dataframe(summary_df, use_container_width=True)


def page_forecasting():
    """Render forecasting page."""
    st.markdown('<div class="main-header">🔮 Revenue Forecasting</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first using the sidebar.")
        return
    
    # Model configuration
    st.markdown("## ⚙️ Model Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.radio(
            "Select model configuration:",
            ["Baseline (Default Parameters)", "Tuned (Optimized Parameters)"],
            index=1,
            help="Baseline uses Prophet defaults. Tuned uses optimized parameters from cross-validation."
        )
        
        params = modeling.TUNED_PARAMS if model_type.startswith("Tuned") else modeling.BASELINE_PARAMS
    
    with col2:
        st.info(f"""
        **Current Parameters:**
        - Seasonality: {params['seasonality_mode']}
        - Changepoint prior: {params['changepoint_prior_scale']}
        - Interval width: {params['interval_width']*100:.0f}%
        """)
    
    # Advanced parameters (optional)
    with st.expander("🔧 Advanced: Customize Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            changepoint_prior = st.slider(
                "Changepoint Prior Scale",
                min_value=0.01,
                max_value=1.0,
                value=params['changepoint_prior_scale'],
                step=0.05,
                help="Controls trend flexibility. Higher = more flexible trend."
            )
            
            seasonality_mode = st.selectbox(
                "Seasonality Mode",
                ["additive", "multiplicative"],
                index=0 if params['seasonality_mode'] == 'additive' else 1,
                help="Additive: constant seasonal effect. Multiplicative: scales with trend."
            )
        
        with col2:
            forecast_horizon = st.slider(
                "Forecast Horizon (months)",
                min_value=6,
                max_value=24,
                value=12,
                help="Number of months to forecast into the future."
            )
            
            interval_width = st.slider(
                "Prediction Interval Width",
                min_value=0.80,
                max_value=0.95,
                value=params['interval_width'],
                step=0.05,
                help="Confidence level for prediction intervals."
            )
        
        # Update params with custom values
        params = params.copy()
        params['changepoint_prior_scale'] = changepoint_prior
        params['seasonality_mode'] = seasonality_mode
        params['interval_width'] = interval_width
    
    # Train/Test split
    st.markdown("## 📊 Train/Test Split")
    
    col1, col2 = st.columns(2)
    
    with col1:
        split_date = st.date_input(
            "Test set start date",
            value=pd.to_datetime('2025-01-01'),
            min_value=pd.to_datetime('2023-01-01'),
            max_value=pd.to_datetime('2025-12-01')
        )
    
    split_date_str = split_date.strftime('%Y-%m-%d')
    
    # Prepare data
    prophet_df = data_processing.prepare_prophet_data(st.session_state.monthly_revenue)
    train_df, test_df = modeling.split_train_test(prophet_df, split_date_str)
    
    with col2:
        st.info(f"""
        **Split Summary:**
        - Training: {len(train_df)} months ({train_df['ds'].min().strftime('%b %Y')} - {train_df['ds'].max().strftime('%b %Y')})
        - Testing: {len(test_df)} months ({test_df['ds'].min().strftime('%b %Y')} - {test_df['ds'].max().strftime('%b %Y')})
        """)
    
    # Train model button
    if st.button("🚀 Train Model and Generate Forecast", type="primary"):
        with st.spinner("Training Prophet model... This may take 30-60 seconds..."):
            # Train model
            model = modeling.train_prophet_model(train_df, params)
            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.split_date = split_date_str
            st.session_state.model_params = params.copy()
            
            # Generate forecast for test period
            _, test_forecast = modeling.forecast_model(model, train_df, test_df)
            
            # Generate future forecast
            future_df = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
            full_forecast = model.predict(future_df)
            future_forecast = full_forecast[full_forecast['ds'] > train_df['ds'].max()].copy()
            
            st.session_state.forecast = future_forecast
            st.session_state.test_forecast = test_forecast
            
            st.success("✅ Model trained successfully!")
    
    # Display results if model is trained
    if st.session_state.model_trained and st.session_state.forecast is not None:
        st.markdown("---")
        st.markdown("## 📈 Forecast Results")
        
        # Executive summary
        st.markdown("### 🎯 Executive Summary: Three Revenue Scenarios")
        
        actual_2025 = prophet_df[prophet_df['ds'].dt.year == 2025]['y'].sum()
        summary_df, peak_trough = modeling.generate_forecast_summary(
            st.session_state.forecast,
            actual_2025
        )
        
        # Display scenario cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**⚠️ Worst Case (90% Lower)**")
            st.markdown(f"### GHS {summary_df.loc[0, '2026 revenue (GHS m)']:.1f}m")
            if 'Growth vs 2025 (%)' in summary_df.columns:
                st.markdown(f"Growth: {summary_df.loc[0, 'Growth vs 2025 (%)']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card" style="border-left-color: #28a745;">', unsafe_allow_html=True)
            st.markdown("**✅ Expected Case**")
            st.markdown(f"### GHS {summary_df.loc[1, '2026 revenue (GHS m)']:.1f}m")
            if 'Growth vs 2025 (%)' in summary_df.columns:
                st.markdown(f"Growth: {summary_df.loc[1, 'Growth vs 2025 (%)']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card" style="border-left-color: #e67e22;">', unsafe_allow_html=True)
            st.markdown("**🎯 Best Case (90% Upper)**")
            st.markdown(f"### GHS {summary_df.loc[2, '2026 revenue (GHS m)']:.1f}m")
            if 'Growth vs 2025 (%)' in summary_df.columns:
                st.markdown(f"Growth: {summary_df.loc[2, 'Growth vs 2025 (%)']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Peak and trough
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Peak Month", peak_trough['peak_month'], f"GHS {peak_trough['peak_value']:.1f}m")
        with col2:
            st.metric("📉 Trough Month", peak_trough['trough_month'], f"GHS {peak_trough['trough_value']:.1f}m")
        
        # Forecast visualization
        st.markdown("### 📊 Forecast Visualization")
        
        fig = visualizations.plot_historical_with_forecast(
            train_df,
            test_df if len(test_df) > 0 else None,
            st.session_state.forecast,
            split_date_str,
            model_name='Prophet'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed forecast table
        with st.expander("📋 View Detailed Monthly Forecast", expanded=False):
            table_data = visualizations.plot_forecast_table(st.session_state.forecast)
            st.dataframe(table_data, use_container_width=True)
        
        # Download forecast
        csv = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button(
            label="💾 Download Forecast as CSV",
            data=csv,
            file_name="revenue_forecast_2026.csv",
            mime="text/csv"
        )


def page_diagnostics():
    """Render model diagnostics page."""
    st.markdown('<div class="main-header">📉 Model Diagnostics</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first using the sidebar.")
        return
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first on the Forecasting page.")
        return
    
    # Prepare data
    prophet_df = data_processing.prepare_prophet_data(st.session_state.monthly_revenue)
    split_date = st.session_state.get('split_date', '2025-01-01')
    train_df, test_df = modeling.split_train_test(prophet_df, split_date)
    
    # Model comparison
    st.markdown("## 📊 Model Performance Comparison")
    
    @st.cache_data(show_spinner=False)
    def cached_model_comparison(train_data, test_data):
        return modeling.compare_models(train_data, test_data)

    with st.spinner("Evaluating models..."):
        comparison_df = cached_model_comparison(train_df, test_df)
    
    # Display metrics table
    st.dataframe(
        comparison_df.style.format({
            'MAE': '{:,.0f}',
            'RMSE': '{:,.0f}',
            'MAPE_%': '{:.2f}',
            'sMAPE_%': '{:.2f}',
            'Bias_%': '{:.2f}',
            'Coverage_%': '{:.1f}'
        }).background_gradient(subset=['MAPE_%', 'RMSE'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Metrics explanation
    with st.expander("ℹ️ Understanding the Metrics", expanded=False):
        st.markdown("""
        - **MAE**: Mean Absolute Error - Average error in GHS (lower is better)
        - **RMSE**: Root Mean Squared Error - Penalizes large errors more heavily (lower is better)
        - **MAPE**: Mean Absolute Percentage Error - Scale-free error metric (lower is better)
        - **sMAPE**: Symmetric MAPE - Balanced over/under-forecasting metric (lower is better)
        - **Bias**: Mean signed percentage error - Detects systematic over/under-forecasting (closer to 0 is better)
        - **Coverage**: % of actuals within 90% prediction interval (should be close to 90%)
        """)
    
    # Visualization
    fig = visualizations.plot_model_comparison(comparison_df)
    st.plotly_chart(fig, use_container_width=True)

    # Backtest summary for the exact model trained on the Forecasting page.
    st.markdown("## 🎯 Held-Out Backtest: Does This Model Generalise?")
    selected_params = st.session_state.get('model_params') or {}
    st.info(
        f"Selected model: seasonality={selected_params.get('seasonality_mode', 'unknown')}, "
        f"changepoint prior={selected_params.get('changepoint_prior_scale', 'unknown')}, "
        f"test period starts {pd.Timestamp(split_date):%b %Y}."
    )
    test_forecast = st.session_state.test_forecast.copy()
    backtest = test_df.merge(
        test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds', how='inner'
    )
    if backtest.empty:
        st.warning("The selected split produced no held-out observations. Choose an earlier split date and retrain the model.")
    else:
        backtest_metrics = modeling.evaluate_model(
            backtest['y'], backtest['yhat'], backtest['yhat_lower'], backtest['yhat_upper'],
            label='Selected model'
        )
        st.markdown("### Backtest scorecard")
        accuracy_cols = st.columns(3)
        accuracy_cols[0].metric("MAPE", f"{backtest_metrics['MAPE_%']:.2f}%", help="Average absolute error as a percentage of actual revenue. Lower is better.")
        accuracy_cols[1].metric("MAE", f"GHS {backtest_metrics['MAE']/1e6:.2f}m", help="Average absolute monthly error in GHS millions. Lower is better.")
        accuracy_cols[2].metric("RMSE", f"GHS {backtest_metrics['RMSE']/1e6:.2f}m", help="Error in GHS millions, with larger misses penalised more heavily. Lower is better.")

        reliability_cols = st.columns(3)
        reliability_cols[0].metric("Bias", f"{backtest_metrics['Bias_%']:.2f}%", help="Signed error: positive means actual revenue was above the forecast.")
        coverage = backtest_metrics.get('Coverage_%', 0)
        reliability_cols[1].metric("Interval coverage", f"{coverage:.1f}%", f"Target: {selected_params.get('interval_width', 0.90) * 100:.0f}%", help="Share of held-out months inside the model's prediction interval.")
        reliability_cols[2].metric("Months tested", f"{len(backtest)}", help="Number of observations never shown to the model during training.")

        st.caption(
            f"Evaluation uses {len(backtest)} months held out from {pd.Timestamp(split_date):%b %Y}. "
            "Positive bias means actual revenue exceeded the forecast on average. Coverage should be close to the nominal interval width."
        )

        backtest_display = backtest.copy()
        backtest_display['Month'] = backtest_display['ds'].dt.strftime('%b %Y')
        backtest_display['Actual (GHS m)'] = backtest_display['y'] / 1e6
        backtest_display['Forecast (GHS m)'] = backtest_display['yhat'] / 1e6
        backtest_display['Lower bound (GHS m)'] = backtest_display['yhat_lower'] / 1e6
        backtest_display['Upper bound (GHS m)'] = backtest_display['yhat_upper'] / 1e6
        backtest_display['Error (GHS m)'] = (backtest_display['y'] - backtest_display['yhat']) / 1e6
        backtest_display['Error (%)'] = (backtest_display['y'] - backtest_display['yhat']) / backtest_display['y'] * 100
        backtest_display['Inside 90% interval'] = (
            backtest_display['y'].between(backtest_display['yhat_lower'], backtest_display['yhat_upper'])
        )
        st.dataframe(
            backtest_display[['Month', 'Actual (GHS m)', 'Forecast (GHS m)', 'Lower bound (GHS m)', 'Upper bound (GHS m)', 'Error (GHS m)', 'Error (%)', 'Inside 90% interval']]
            .style.format({
                'Actual (GHS m)': 'GHS {:,.2f}m',
                'Forecast (GHS m)': 'GHS {:,.2f}m',
                'Lower bound (GHS m)': 'GHS {:,.2f}m',
                'Upper bound (GHS m)': 'GHS {:,.2f}m',
                'Error (GHS m)': 'GHS {:+,.2f}m',
                'Error (%)': '{:+.2f}%'
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download backtest results",
            data=backtest.to_csv(index=False),
            file_name="revenue_forecast_backtest.csv",
            mime="text/csv",
        )
    
    # Model components
    st.markdown("## 🔍 Model Components Analysis")
    
    model = st.session_state.model
    full_forecast = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
    
    fig = visualizations.plot_model_components(model, full_forecast)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    if hasattr(st.session_state, 'test_forecast') and st.session_state.test_forecast is not None:
        st.markdown("## 📐 Residual Analysis")

        residual_frame = test_df.merge(
            st.session_state.test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds', how='inner'
        ).sort_values('ds').reset_index(drop=True)
        residual_frame['residual'] = residual_frame['y'] - residual_frame['yhat']
        residual_frame['error_pct'] = residual_frame['residual'] / residual_frame['y'] * 100
        residual_frame['inside_interval'] = residual_frame['y'].between(
            residual_frame['yhat_lower'], residual_frame['yhat_upper']
        )

        if residual_frame.empty:
            st.warning("No aligned held-out predictions are available for residual analysis.")
        else:
            residual_abs = residual_frame['residual'].abs()
            residual_cols = st.columns(4)
            residual_cols[0].metric("Mean signed error", f"GHS {residual_frame['residual'].mean()/1e6:+.2f}m", help="Average actual minus forecast. Positive means systematic under-forecasting; near zero is preferred.")
            residual_cols[1].metric("Mean absolute error", f"GHS {residual_abs.mean()/1e6:.2f}m", help="Typical monthly miss, ignoring direction.")
            residual_cols[2].metric("RMSE", f"GHS {np.sqrt((residual_frame['residual'] ** 2).mean())/1e6:.2f}m", help="Error size with extra weight on large misses.")
            residual_cols[3].metric("Error spread", f"GHS {residual_frame['residual'].std()/1e6:.2f}m", help="Standard deviation of residuals. Smaller means more consistent errors.")

            residual_cols = st.columns(4)
            residual_cols[0].metric("Median absolute error", f"GHS {residual_abs.median()/1e6:.2f}m", help="Middle monthly miss; less affected by one unusually large error.")
            residual_cols[1].metric("Median absolute % error", f"{residual_frame['error_pct'].abs().median():.2f}%", help="Typical percentage miss across held-out months.")
            residual_cols[2].metric("Under-forecast months", f"{(residual_frame['residual'] > 0).sum()} / {len(residual_frame)}", help="Months where actual revenue exceeded the forecast.")
            residual_cols[3].metric("Outside 90% interval", f"{(~residual_frame['inside_interval']).sum()} / {len(residual_frame)}", help="Held-out months outside the prediction interval.")

            st.caption("A healthy residual pattern is centred around zero, has no visible trend over time, and has roughly stable spread. Positive residuals indicate under-forecasting.")

            fig = visualizations.plot_residuals(
                residual_frame[['ds', 'y']],
                residual_frame[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            )
            st.plotly_chart(fig, use_container_width=True)

            residual_display = residual_frame.copy()
            residual_display['Month'] = residual_display['ds'].dt.strftime('%b %Y')
            residual_display['Actual (GHS m)'] = residual_display['y'] / 1e6
            residual_display['Forecast (GHS m)'] = residual_display['yhat'] / 1e6
            residual_display['Residual (GHS m)'] = residual_display['residual'] / 1e6
            residual_display['Error (%)'] = residual_display['error_pct']
            residual_display['Status'] = np.where(
                residual_display['inside_interval'], 'Inside interval', 'Outside interval'
            )
            with st.expander("View monthly residual details", expanded=False):
                st.dataframe(
                    residual_display[['Month', 'Actual (GHS m)', 'Forecast (GHS m)', 'Residual (GHS m)', 'Error (%)', 'Status']]
                    .style.format({
                        'Actual (GHS m)': 'GHS {:,.2f}m',
                        'Forecast (GHS m)': 'GHS {:,.2f}m',
                        'Residual (GHS m)': 'GHS {:+,.2f}m',
                        'Error (%)': '{:+.2f}%'
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
    
    # Key insights
    st.markdown("## 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Strengths:**
        - Single-digit MAPE on held-out test year
        - Beats naive benchmarks by wide margin
        - Captures seasonal pattern accurately
        - Detects 2024 expansion changepoints
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Limitations:**
        - Prediction intervals narrower than ideal (75% vs 90% coverage)
        - Assumes current momentum continues
        - No causal factors (macro, competition)
        - Chain-level only (hides branch variation)
        """)


def main():
    """Main application."""
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = sidebar()
    
    # Render selected page
    if page == "🏠 Overview":
        page_overview()
    elif page == "📊 Data Quality":
        page_data_quality()
    elif page == "🔮 Forecasting":
        page_forecasting()
    elif page == "📉 Model Diagnostics":
        page_diagnostics()


if __name__ == "__main__":
    main()
