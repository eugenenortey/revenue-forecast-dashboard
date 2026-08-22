"""
Prophet modeling utilities for revenue forecasting dashboard.
Handles model training, evaluation, cross-validation, and forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
import logging

# Suppress Prophet logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.CRITICAL)


# Model configurations
BASELINE_PARAMS = {
    'growth': 'linear',
    'seasonality_mode': 'additive',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'yearly_seasonality': True,
    'weekly_seasonality': False,
    'daily_seasonality': False,
    'interval_width': 0.90,
}

TUNED_PARAMS = {
    'growth': 'linear',
    'seasonality_mode': 'multiplicative',
    'changepoint_prior_scale': 0.50,
    'seasonality_prior_scale': 10.0,
    'yearly_seasonality': True,
    'weekly_seasonality': False,
    'daily_seasonality': False,
    'interval_width': 0.90,
}


def split_train_test(df, split_date='2025-01-01'):
    """
    Split data into train and test sets chronologically.
    
    Args:
        df: Prophet-formatted DataFrame with 'ds' and 'y' columns
        split_date: Date to split on (test set starts here)
        
    Returns:
        train_df, test_df
    """
    split_date = pd.to_datetime(split_date)
    train = df[df['ds'] < split_date].copy()
    test = df[df['ds'] >= split_date].copy()
    
    return train, test


def evaluate_model(y_true, y_pred, lower=None, upper=None, label=""):
    """
    Evaluate model performance with multiple metrics.
    
    Metrics:
    - MAE: Mean Absolute Error (directly interpretable)
    - RMSE: Root Mean Squared Error (penalizes large errors)
    - MAPE: Mean Absolute Percentage Error (scale-free)
    - sMAPE: Symmetric MAPE (balanced over/under-forecasting)
    - Bias: Mean signed percentage error (detects systematic over/under-forecasting)
    - Coverage: % of actuals within prediction interval
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        lower: Lower bound of prediction interval (optional)
        upper: Upper bound of prediction interval (optional)
        label: Model label
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    
    metrics = {
        'model': label,
        'MAE': np.mean(np.abs(err)),
        'RMSE': np.sqrt(np.mean(err ** 2)),
        'MAPE_%': np.mean(np.abs(err / y_true)) * 100,
        'sMAPE_%': np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100,
        'Bias_%': np.mean(err / y_true) * 100,
    }
    
    if lower is not None and upper is not None:
        coverage = np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))) * 100
        metrics['Coverage_%'] = coverage
    
    return metrics


def create_naive_benchmarks(train_df, test_df):
    """
    Create naive forecast benchmarks for comparison.
    
    1. Seasonal naive: This month next year = same month last year
    2. Drift naive: Last value + average historical monthly increment
    
    Returns:
        DataFrame with benchmark forecasts and metrics
    """
    # Create full series for seasonal naive
    full_series = pd.concat([train_df, test_df]).set_index('ds')['y']
    
    # Seasonal naive: shift by 12 months
    seasonal_naive = full_series.shift(12).loc[test_df['ds']].values
    
    # Drift naive
    drift_increment = train_df['y'].diff().mean()
    drift = train_df['y'].iloc[-1] + (drift_increment * np.arange(1, len(test_df) + 1))
    
    # Evaluate benchmarks
    benchmarks = pd.DataFrame([
        evaluate_model(test_df['y'], seasonal_naive, label='Seasonal naive (y[t-12])'),
        evaluate_model(test_df['y'], drift, label='Drift naive'),
    ])
    
    return benchmarks


def train_prophet_model(train_df, params=None):
    """
    Train Prophet model with given parameters.
    
    Args:
        train_df: Training data (Prophet format with 'ds' and 'y')
        params: Dictionary of Prophet parameters (uses BASELINE_PARAMS if None)
        
    Returns:
        Fitted Prophet model
    """
    if params is None:
        params = BASELINE_PARAMS.copy()
    
    model = Prophet(**params)
    model.fit(train_df)
    
    return model


def forecast_model(model, train_df, test_df=None, periods=12, freq='MS'):
    """
    Generate forecast from trained model.
    
    Args:
        model: Fitted Prophet model
        train_df: Training data
        test_df: Test data (optional, for backtesting)
        periods: Number of periods to forecast forward (if test_df is None)
        freq: Frequency of forecast ('MS' for month start)
        
    Returns:
        forecast_df: Full forecast DataFrame
        test_forecast: Test period forecast (if test_df provided)
    """
    if test_df is not None:
        # Make future dataframe for train + test period
        future = model.make_future_dataframe(periods=len(test_df), freq=freq)
    else:
        # Make future dataframe for specified periods
        future = model.make_future_dataframe(periods=periods, freq=freq)
    
    forecast = model.predict(future)
    
    if test_df is not None:
        # Extract test period forecast
        test_forecast = forecast[forecast['ds'].isin(test_df['ds'])].copy()
        return forecast, test_forecast
    else:
        # Return future forecast only
        future_forecast = forecast[forecast['ds'] > train_df['ds'].max()].copy()
        return forecast, future_forecast


def run_cross_validation(model, train_df, initial='1460 days', period='90 days', horizon='365 days'):
    """
    Run time series cross-validation.
    
    Args:
        model: Fitted Prophet model
        train_df: Training data
        initial: Initial training period
        period: Period between cutoff dates
        horizon: Forecast horizon
        
    Returns:
        cv_results: Cross-validation results
        cv_metrics: Performance metrics
    """
    try:
        cv_results = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel='processes'
        )
        
        cv_metrics = performance_metrics(cv_results, rolling_window=0.1)
        
        return cv_results, cv_metrics
    except Exception as e:
        print(f"Cross-validation error: {str(e)}")
        return None, None


def compare_models(train_df, test_df, model_configs=None):
    """
    Compare multiple model configurations.
    
    Args:
        train_df: Training data
        test_df: Test data
        model_configs: Dictionary of {name: params} (uses baseline and tuned if None)
        
    Returns:
        DataFrame with comparison metrics
    """
    if model_configs is None:
        model_configs = {
            'Baseline': BASELINE_PARAMS,
            'Tuned': TUNED_PARAMS,
        }
    
    results = []
    
    # Add naive benchmarks
    benchmarks = create_naive_benchmarks(train_df, test_df)
    results.append(benchmarks)
    
    # Train and evaluate each model
    for name, params in model_configs.items():
        model = train_prophet_model(train_df, params)
        _, test_forecast = forecast_model(model, train_df, test_df)
        
        metrics = evaluate_model(
            test_df['y'],
            test_forecast['yhat'],
            test_forecast['yhat_lower'],
            test_forecast['yhat_upper'],
            label=f'Prophet {name}'
        )
        
        results.append(pd.DataFrame([metrics]))
    
    comparison = pd.concat(results, ignore_index=True)
    
    return comparison


def generate_forecast_summary(forecast_df, actual_2025_revenue=None):
    """
    Generate executive summary of forecast with three scenarios.
    
    Args:
        forecast_df: Future forecast DataFrame
        actual_2025_revenue: Actual 2025 revenue for growth calculation (optional)
        
    Returns:
        DataFrame with worst/expected/best case scenarios
    """
    # Calculate totals
    f_2026 = forecast_df['yhat'].sum()
    lo_2026 = forecast_df['yhat_lower'].sum()
    hi_2026 = forecast_df['yhat_upper'].sum()
    
    summary_data = {
        'Scenario': ['Worst case (90% lower)', 'Expected case', 'Best case (90% upper)'],
        '2026 revenue (GHS m)': [lo_2026 / 1e6, f_2026 / 1e6, hi_2026 / 1e6],
    }
    
    if actual_2025_revenue is not None:
        summary_data['Growth vs 2025 (%)'] = [
            (lo_2026 / actual_2025_revenue - 1) * 100,
            (f_2026 / actual_2025_revenue - 1) * 100,
            (hi_2026 / actual_2025_revenue - 1) * 100,
        ]
    
    summary = pd.DataFrame(summary_data)
    
    # Add peak/trough information
    peak_idx = forecast_df['yhat'].idxmax()
    trough_idx = forecast_df['yhat'].idxmin()
    
    peak_month = forecast_df.loc[peak_idx, 'ds'].strftime('%b %Y')
    trough_month = forecast_df.loc[trough_idx, 'ds'].strftime('%b %Y')
    peak_value = forecast_df.loc[peak_idx, 'yhat'] / 1e6
    trough_value = forecast_df.loc[trough_idx, 'yhat'] / 1e6
    
    return summary, {
        'peak_month': peak_month,
        'peak_value': peak_value,
        'trough_month': trough_month,
        'trough_value': trough_value
    }


def get_model_components(model, forecast_df):
    """
    Extract Prophet model components for visualization.
    
    Returns:
        Dictionary with trend, seasonality, and changepoints
    """
    components = {
        'trend': forecast_df[['ds', 'trend']].copy(),
        'yearly': forecast_df[['ds', 'yearly']].copy() if 'yearly' in forecast_df.columns else None,
        'changepoints': pd.DataFrame({
            'ds': model.changepoints,
            'delta': model.params['delta'].mean(axis=0) if hasattr(model, 'params') else [0] * len(model.changepoints)
        }),
    }
    
    return components
