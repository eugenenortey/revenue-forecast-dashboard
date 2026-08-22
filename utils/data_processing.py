"""
Data processing utilities for revenue forecasting dashboard.
Handles data loading, cleaning, and aggregation following the notebook's methodology.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# Constants
PLACEHOLDERS = {"n/a", "na", "nan", "none", "null", "unknown", "-", "?",
                "missing", "999999", "not recorded", "tbd", ""}

CURRENCY_COLS = ["Monthly_Revenue_GHS", "COGS_GHS", "Operating_Expenses_GHS",
                 "Accounts_Receivable_GHS", "Accounts_Payable_GHS",
                 "Inventory_GHS", "Working_Capital_GHS", "Avg_Transaction_Value_GHS"]


def load_excel_data(file_path):
    """
    Load Excel file with all columns as strings initially.
    
    Args:
        file_path: Path to Excel file or file-like object
        
    Returns:
        DataFrame with raw data, all columns as strings
    """
    try:
        if isinstance(file_path, str):
            sheets = pd.read_excel(file_path, sheet_name=None)
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)
        
        # Load the raw data sheet with all columns as string
        raw = pd.read_excel(file_path, sheet_name="Raw_Data", dtype=str)
        
        return raw, sheets
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {str(e)}")


def count_placeholders(series):
    """Count placeholder text that represents missing values."""
    return series.dropna().str.strip().str.lower().isin(PLACEHOLDERS).sum()


def to_number(value):
    """
    Convert messy currency/number string to float.
    Handles GHS prefix, cedi symbol, thousands separators, and placeholders.
    
    Returns NaN if not a real number.
    """
    if pd.isna(value):
        return np.nan
    t = str(value).replace("GHS", "").replace("₵", "").replace(",", "").strip()
    if t.lower() in PLACEHOLDERS:
        return np.nan
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else np.nan


def parse_date(value):
    """
    Format-aware date parser based on pattern analysis:
    - slash format (xx/xx/YYYY) -> DD/MM/YYYY (UK convention)
    - dash format (xx-xx-YYYY) -> MM-DD-YYYY (US convention)
    - everything else parses unambiguously
    """
    if pd.isna(value):
        return pd.NaT
    t = str(value).strip()
    
    # Handle ambiguous formats based on empirical evidence
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", t):
        return pd.to_datetime(t, format="%d/%m/%Y", errors="coerce")
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", t):
        return pd.to_datetime(t, format="%m-%d-%Y", errors="coerce")
    
    # Unambiguous formats
    return pd.to_datetime(t, errors="coerce", format="mixed")


def standardise_branch_code(value):
    """
    Standardise branch code to BR-XXX format.
    Handles: 'br-004', 'BR004', ' BR-004', 'Branch 004' -> 'BR-004'
    """
    if pd.isna(value):
        return np.nan
    t = str(value).strip().upper().replace("_", " ")
    t = re.sub(r"^BRANCH\s*", "BR-", t)
    t = re.sub(r"^BR[\s-]*", "BR-", t)
    m = re.fullmatch(r"BR-(\d+)", t)
    return f"BR-{int(m.group(1)):03d}" if m else t


def get_data_quality_report(df):
    """
    Generate comprehensive data quality report.
    
    Returns:
        Dictionary with quality metrics
    """
    report = {}
    
    # Basic shape
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    
    # Missing values
    report['missing_visible'] = df.isna().sum().to_dict()
    report['missing_placeholders'] = {c: count_placeholders(df[c]) for c in df.columns}
    
    # Blank and duplicate rows
    report['blank_rows'] = df.isna().all(axis=1).sum()
    report['exact_duplicates'] = df.duplicated(keep='first').sum()
    
    # Date format distribution
    if 'Report_Month' in df.columns:
        report['date_formats'] = classify_date_formats(df['Report_Month'])
    
    # Branch code count
    if 'Branch_Code' in df.columns:
        report['unique_branch_codes_raw'] = df['Branch_Code'].nunique()
    
    return report


def classify_date_formats(series):
    """Classify date format for each value in the series."""
    def classify_one(value):
        if pd.isna(value):
            return "MISSING"
        v = str(value).strip()
        patterns = {
            "ISO  YYYY-MM-DD":   r"\d{4}-\d{2}-\d{2}",
            "ISO  YYYY/MM/DD":   r"\d{4}/\d{2}/\d{2}",
            "xx/xx/YYYY":        r"\d{2}/\d{2}/\d{4}",
            "xx-xx-YYYY":        r"\d{2}-\d{2}-\d{4}",
            "DD-Mon-YYYY":       r"\d{2}-[A-Za-z]{3}-\d{4}",
            "Month DD, YYYY":    r"[A-Za-z]+ \d{1,2}, \d{4}",
        }
        for label, pat in patterns.items():
            if re.fullmatch(pat, v):
                return label
        return "UNPARSEABLE"
    
    return series.map(classify_one).value_counts().to_dict()


def clean_data(raw_df):
    """
    Main cleaning pipeline following notebook methodology.
    
    Steps:
    1. Remove blank rows and exact duplicates
    2. Standardise branch codes
    3. Parse dates
    4. Convert currency columns to numeric
    5. Filter invalid records (negative revenue, unparseable dates)
    6. Detect and handle overlapping granularities
    7. Aggregate to monthly chain-level revenue
    
    Returns:
        clean_df: Cleaned DataFrame
        monthly_revenue: Final monthly chain revenue series
        cleaning_stats: Dictionary with cleaning statistics
    """
    stats = {}
    stats['initial_rows'] = len(raw_df)
    
    # Step 1: Remove blank rows and exact duplicates
    df = raw_df.copy()
    df = df[~df.isna().all(axis=1)]
    stats['after_blank_removal'] = len(df)
    stats['blank_rows_removed'] = stats['initial_rows'] - stats['after_blank_removal']
    
    df = df[~df.duplicated(keep='first')]
    stats['after_duplicate_removal'] = len(df)
    stats['duplicate_rows_removed'] = stats['after_blank_removal'] - stats['after_duplicate_removal']
    
    # Step 2: Standardise branch codes
    raw_branch_count = df['Branch_Code'].nunique(dropna=True)
    df['branch'] = df['Branch_Code'].map(standardise_branch_code)
    stats['unique_branches'] = df['branch'].nunique()
    stats['raw_branch_codes'] = raw_branch_count
    stats['branch_codes_standardised'] = int((
        df['Branch_Code'].fillna('').astype(str).str.strip().str.upper() !=
        df['branch'].fillna('').astype(str)
    ).sum())
    
    # Step 3: Parse dates
    df['date'] = df['Report_Month'].map(parse_date)
    stats['unparseable_dates'] = df['date'].isna().sum()
    stats['placeholder_dates'] = int((df['date'].notna() & (df['date'].dt.year <= 1990)).sum())
    
    # Step 4: Convert revenue to numeric
    df['revenue'] = df['Monthly_Revenue_GHS'].map(to_number)
    stats['unparseable_revenue'] = df['revenue'].isna().sum()
    stats['negative_revenue'] = int((df['revenue'] <= 0).sum())
    
    # Step 5: Filter invalid records
    # Remove rows with invalid dates (including 1900 placeholder dates)
    valid = df[df['date'].notna() & (df['date'].dt.year > 1990)].copy()
    stats['after_date_filter'] = len(valid)
    
    # Remove negative or zero revenue
    valid = valid[valid['revenue'] > 0].copy()
    stats['after_revenue_filter'] = len(valid)
    
    # Step 6: Handle overlapping granularities
    # Calculate median revenue per branch-month to detect outliers
    valid['year_month'] = valid['date'].dt.to_period('M')
    
    # For each branch-month, calculate median and filter outliers
    def filter_branch_month_outliers(group):
        """
        Filter outliers within a branch-month using 0.4x-2.5x threshold.
        This removes overlapping granularity issues (weekly/daily records mixed with monthly summaries).
        """
        if len(group) == 1:
            return group
        
        median_rev = group['revenue'].median()
        # Keep records within 0.4x to 2.5x of median
        mask = (group['revenue'] >= median_rev * 0.4) & (group['revenue'] <= median_rev * 2.5)
        return group[mask]
    
    cleaned = valid.groupby(['branch', 'year_month'], group_keys=False).apply(filter_branch_month_outliers)
    stats['after_granularity_filter'] = len(cleaned)
    stats['granularity_rows_removed'] = stats['after_revenue_filter'] - stats['after_granularity_filter']
    stats['branch_months_before_filter'] = valid.groupby(['branch', 'year_month']).ngroups
    stats['branch_months_after_filter'] = cleaned.groupby(['branch', 'year_month']).ngroups
    
    # Step 7: Aggregate to monthly chain-level revenue
    monthly_revenue = (cleaned.groupby('year_month')['revenue']
                       .sum()
                       .reset_index()
                       .rename(columns={'revenue': 'y'}))
    
    # Convert Period to datetime (first day of month)
    monthly_revenue['ds'] = monthly_revenue['year_month'].apply(lambda x: x.to_timestamp())
    monthly_revenue = monthly_revenue[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    stats['final_monthly_observations'] = len(monthly_revenue)
    stats['date_range'] = (monthly_revenue['ds'].min().strftime('%Y-%m-%d'),
                          monthly_revenue['ds'].max().strftime('%Y-%m-%d'))
    stats['expected_monthly_observations'] = (
        (monthly_revenue['ds'].max().year - monthly_revenue['ds'].min().year) * 12
        + monthly_revenue['ds'].max().month - monthly_revenue['ds'].min().month + 1
    )
    stats['missing_months'] = stats['expected_monthly_observations'] - stats['final_monthly_observations']
    
    return cleaned, monthly_revenue, stats


def get_naive_aggregation(raw_df):
    """
    Create naive aggregation (incorrect - for comparison).
    This demonstrates what happens if you don't handle overlapping granularities.
    
    Returns:
        DataFrame with naive monthly aggregation
    """
    df = raw_df.copy()
    df['branch'] = df['Branch_Code'].map(standardise_branch_code)
    df['date'] = df['Report_Month'].map(parse_date)
    df['revenue'] = df['Monthly_Revenue_GHS'].map(to_number)
    
    # Filter to valid records only
    valid = df[(df['date'].notna()) & 
               (df['date'].dt.year > 1990) & 
               (df['revenue'] > 0)].copy()
    
    # Naive groupby - this DOUBLE COUNTS due to overlapping granularities
    valid['year_month'] = valid['date'].dt.to_period('M')
    naive = (valid.groupby('year_month')['revenue']
             .sum()
             .reset_index()
             .rename(columns={'revenue': 'y'}))
    
    naive['ds'] = naive['year_month'].apply(lambda x: x.to_timestamp())
    naive = naive[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    return naive


def prepare_prophet_data(monthly_revenue):
    """
    Prepare data for Prophet modeling.
    Prophet expects columns named 'ds' (date) and 'y' (target).
    
    Args:
        monthly_revenue: DataFrame with 'ds' and 'y' columns
        
    Returns:
        DataFrame ready for Prophet
    """
    prophet_df = monthly_revenue.copy()
    
    # Ensure ds is datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    return prophet_df[['ds', 'y']]
