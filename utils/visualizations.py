"""
Visualization utilities for revenue forecasting dashboard.
Creates interactive plots using Plotly for better user experience.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Color scheme
COLORS = {
    'train': '#7f8c8d',
    'test': '#2c3e50',
    'forecast': '#e67e22',
    'baseline': '#c0392b',
    'interval': '#e67e22',
    'naive': '#95a5a6',
    'trend': '#3498db',
    'seasonality': '#9b59b6',
    'changepoint': '#e74c3c',
}


def plot_data_quality_comparison(clean_revenue, naive_revenue):
    """
    Plot comparison between clean and naive aggregation.
    Shows the impact of handling overlapping granularities.
    """
    fig = go.Figure()
    
    # Naive aggregation (incorrect - double counts)
    fig.add_trace(go.Scatter(
        x=naive_revenue['ds'],
        y=naive_revenue['y'] / 1e6,
        mode='lines+markers',
        name='Naive aggregation (INCORRECT)',
        line=dict(color=COLORS['naive'], width=2, dash='dash'),
        marker=dict(size=4),
        hovertemplate='%{x|%b %Y}<br>GHS %{y:.2f}m<extra></extra>'
    ))
    
    # Clean aggregation (correct)
    fig.add_trace(go.Scatter(
        x=clean_revenue['ds'],
        y=clean_revenue['y'] / 1e6,
        mode='lines+markers',
        name='Clean aggregation (CORRECT)',
        line=dict(color=COLORS['train'], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%b %Y}<br>GHS %{y:.2f}m<extra></extra>'
    ))
    
    fig.update_layout(
        title='Impact of Data Cleaning: Naive vs. Clean Aggregation',
        xaxis_title='Date',
        yaxis_title='Monthly Revenue (GHS millions)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_historical_with_forecast(train_df, test_df, forecast_df, split_date, model_name='Prophet'):
    """
    Plot historical data with forecast and prediction intervals.
    """
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_df['ds'],
        y=train_df['y'] / 1e6,
        mode='lines',
        name='Historical (training)',
        line=dict(color=COLORS['train'], width=2),
        hovertemplate='%{x|%b %Y}<br>Actual: GHS %{y:.2f}m<extra></extra>'
    ))
    
    # Test data (actual)
    if test_df is not None and len(test_df) > 0:
        fig.add_trace(go.Scatter(
            x=test_df['ds'],
            y=test_df['y'] / 1e6,
            mode='lines+markers',
            name='Actual (test)',
            line=dict(color=COLORS['test'], width=3),
            marker=dict(size=8),
            hovertemplate='%{x|%b %Y}<br>Actual: GHS %{y:.2f}m<extra></extra>'
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'] / 1e6,
        mode='lines+markers',
        name=f'{model_name} forecast',
        line=dict(color=COLORS['forecast'], width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='%{x|%b %Y}<br>Forecast: GHS %{y:.2f}m<extra></extra>'
    ))
    
    # Prediction interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'] / 1e6,
        mode='lines',
        name='90% interval (upper)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'] / 1e6,
        mode='lines',
        name='90% prediction interval',
        line=dict(width=0),
        fillcolor=f'rgba({int(COLORS["interval"][1:3], 16)}, {int(COLORS["interval"][3:5], 16)}, {int(COLORS["interval"][5:7], 16)}, 0.2)',
        fill='tonexty',
        hovertemplate='%{x|%b %Y}<br>Lower: %{y:.2f}m<extra></extra>'
    ))
    
    # Add vertical line at split date
    if split_date:
        split_timestamp = pd.to_datetime(split_date)
        fig.add_shape(
            type="line",
            x0=split_timestamp,
            x1=split_timestamp,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="black", dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=split_timestamp,
            y=1,
            xref="x",
            yref="paper",
            text="train | test",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=6,
        )
    
    fig.update_layout(
        title=f'{model_name} Forecast with 90% Prediction Intervals',
        xaxis_title='Date',
        yaxis_title='Monthly Revenue (GHS millions)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_model_components(model, forecast_df):
    """
    Plot Prophet model components (trend and seasonality).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Trend with Changepoints', 'Yearly Seasonality'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Trend
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['trend'] / 1e6,
        mode='lines',
        name='Trend',
        line=dict(color=COLORS['trend'], width=3),
        hovertemplate='%{x|%b %Y}<br>Trend: GHS %{y:.2f}m<extra></extra>'
    ), row=1, col=1)
    
    # Changepoints
    if hasattr(model, 'changepoints') and len(model.changepoints) > 0:
        changepoint_dates = pd.to_datetime(model.changepoints)
        changepoint_trends = []
        for cp in changepoint_dates:
            if cp in forecast_df['ds'].values:
                idx = forecast_df[forecast_df['ds'] == cp].index[0]
                changepoint_trends.append(forecast_df.loc[idx, 'trend'])
            else:
                # Find nearest date
                idx = (forecast_df['ds'] - cp).abs().idxmin()
                changepoint_trends.append(forecast_df.loc[idx, 'trend'])
        
        fig.add_trace(go.Scatter(
            x=changepoint_dates,
            y=np.array(changepoint_trends) / 1e6,
            mode='markers',
            name='Changepoints',
            marker=dict(color=COLORS['changepoint'], size=10, symbol='x', line=dict(width=2)),
            hovertemplate='%{x|%b %Y}<br>Changepoint<extra></extra>'
        ), row=1, col=1)
    
    # Yearly seasonality
    if 'yearly' in forecast_df.columns:
        # Create a year's worth of data for clean visualization
        yearly_data = forecast_df[['ds', 'yearly']].copy()
        yearly_data['month'] = yearly_data['ds'].dt.month
        yearly_seasonal = yearly_data.groupby('month')['yearly'].mean().reset_index()
        yearly_seasonal['month_name'] = pd.to_datetime(yearly_seasonal['month'], format='%m').dt.strftime('%b')
        
        fig.add_trace(go.Scatter(
            x=yearly_seasonal['month_name'],
            y=yearly_seasonal['yearly'],
            mode='lines+markers',
            name='Yearly seasonality',
            line=dict(color=COLORS['seasonality'], width=3),
            marker=dict(size=8),
            hovertemplate='%{x}<br>Effect: %{y:.0f} GHS<extra></extra>'
        ), row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (GHS millions)", row=1, col=1)
    fig.update_yaxes(title_text="Seasonal Effect (GHS)", row=2, col=1)
    
    fig.update_layout(
        title='Model Components Decomposition',
        template='plotly_white',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_model_comparison(comparison_df):
    """
    Plot comparison of model performance metrics.
    """
    # Select key metrics for visualization
    metrics = ['MAPE_%', 'RMSE', 'Bias_%', 'Coverage_%']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, metric in enumerate(available_metrics):
        row, col = positions[i]
        
        fig.add_trace(go.Bar(
            x=comparison_df['model'],
            y=comparison_df[metric],
            name=metric,
            marker_color=[COLORS['naive'], COLORS['naive'], COLORS['baseline'], COLORS['forecast']],
            text=comparison_df[metric].round(2),
            textposition='outside',
            showlegend=False,
            hovertemplate='%{x}<br>' + metric + ': %{y:.2f}<extra></extra>'
        ), row=row, col=col)
        
        fig.update_yaxes(title_text=metric, row=row, col=col)
    
    fig.update_layout(
        title='Model Performance Comparison',
        template='plotly_white',
        height=700,
        showlegend=False
    )
    
    return fig


def plot_forecast_table(forecast_df):
    """
    Create formatted table of forecast values.
    """
    table_data = forecast_df[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].copy()
    table_data.columns = ['Month', 'Lower (90%)', 'Forecast', 'Upper (90%)']
    table_data['Month'] = table_data['Month'].dt.strftime('%b %Y')
    
    # Calculate interval width
    table_data['Interval width %'] = (
        (forecast_df['yhat_upper'] - forecast_df['yhat_lower']) / forecast_df['yhat'] * 100
    )
    
    # Format to millions
    for col in ['Lower (90%)', 'Forecast', 'Upper (90%)']:
        table_data[col] = (table_data[col] / 1e6).round(2)
    
    table_data['Interval width %'] = table_data['Interval width %'].round(1)
    
    return table_data


def plot_residuals(test_df, test_forecast):
    """
    Plot readable, full-width residual diagnostics.
    """
    aligned = test_df[['ds', 'y']].merge(
        test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner'
    ).sort_values('ds')

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Actual revenue versus forecast', 'Signed residuals by month', 'Percentage errors by month'),
        vertical_spacing=0.12,
    )

    if aligned.empty:
        fig.update_layout(title='Residual Analysis: no aligned observations')
        return fig

    residuals = aligned['y'] - aligned['yhat']
    residuals_pct = residuals / aligned['y'] * 100
    months = aligned['ds'].dt.strftime('%b %Y')
    residual_colors = np.where(residuals >= 0, '#2e8b57', '#c0392b')

    # Actuals, forecast, and prediction interval.
    fig.add_trace(go.Scatter(
        x=months, y=aligned['y'] / 1e6, mode='lines+markers', name='Actual',
        line=dict(color=COLORS['test'], width=3), marker=dict(size=7),
        hovertemplate='%{x}<br>Actual: GHS %{y:.2f}m<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=months, y=aligned['yhat'] / 1e6, mode='lines+markers', name='Forecast',
        line=dict(color=COLORS['forecast'], width=3, dash='dash'), marker=dict(size=7),
        hovertemplate='%{x}<br>Forecast: GHS %{y:.2f}m<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=months, y=aligned['yhat_upper'] / 1e6, mode='lines', name='90% upper bound',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=months, y=aligned['yhat_lower'] / 1e6, mode='lines', name='90% interval',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(230,126,34,0.16)',
        hovertemplate='%{x}<br>Lower bound: GHS %{y:.2f}m<extra></extra>'
    ), row=1, col=1)

    # Signed residuals: positive means under-forecasting.
    fig.add_trace(go.Bar(
        x=months, y=residuals / 1e6, name='Residual', marker_color=residual_colors,
        hovertemplate='%{x}<br>Residual: GHS %{y:+.2f}m<extra></extra>'
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

    # Percentage errors with a practical +/-10% guide.
    fig.add_trace(go.Bar(
        x=months, y=residuals_pct, name='Percentage error', marker_color=residual_colors,
        hovertemplate='%{x}<br>Error: %{y:+.2f}%<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)
    fig.add_hline(y=10, line_dash='dot', line_color='#b0b0b0', row=3, col=1)
    fig.add_hline(y=-10, line_dash='dot', line_color='#b0b0b0', row=3, col=1)

    fig.update_xaxes(title_text='Held-out month', row=1, col=1)
    fig.update_xaxes(title_text='Held-out month', row=2, col=1)
    fig.update_xaxes(title_text='Held-out month', row=3, col=1, tickangle=-45)
    fig.update_yaxes(title_text='Revenue (GHS m)', row=1, col=1)
    fig.update_yaxes(title_text='Residual (GHS m)', row=2, col=1)
    fig.update_yaxes(title_text='Error (%)', row=3, col=1)

    fig.update_layout(
        title='Residual Analysis: actuals, errors, and prediction intervals',
        template='plotly_white', height=950, showlegend=True,
        hovermode='x unified', margin=dict(l=60, r=30, t=100, b=100),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig


def plot_time_series_decomposition(monthly_revenue):
    """
    Simple time series plot showing overall trend.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_revenue['ds'],
        y=monthly_revenue['y'] / 1e6,
        mode='lines+markers',
        name='Monthly Revenue',
        line=dict(color=COLORS['train'], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%b %Y}<br>Revenue: GHS %{y:.2f}m<extra></extra>'
    ))
    
    fig.update_layout(
        title='Historical Monthly Chain Revenue',
        xaxis_title='Date',
        yaxis_title='Monthly Revenue (GHS millions)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig
