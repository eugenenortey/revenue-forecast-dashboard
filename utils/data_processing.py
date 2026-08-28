"""Data loading and cleaning pipeline.

This is a direct port of Sections 6-9 of the source notebook
(Prophet_Branch_Revenue_Forecasting.ipynb) — the raw-data audit and the
row-by-row cleaning pipeline that resolves overlapping daily/weekly/monthly
reporting granularity. Every function here mirrors the notebook's own
cell source, not a re-derivation, so the row counts it produces on the
bundled dataset should match the notebook's printed audit trail exactly.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Branch_Revenue_Forecasting_Raw_Dataset.xlsx"

PLACEHOLDERS = {"n/a", "na", "nan", "none", "null", "unknown", "-", "?",
                "missing", "999999", "not recorded", "tbd", ""}

CURRENCY_COLS = ["Monthly_Revenue_GHS", "COGS_GHS", "Operating_Expenses_GHS",
                 "Accounts_Receivable_GHS", "Accounts_Payable_GHS",
                 "Inventory_GHS", "Working_Capital_GHS", "Avg_Transaction_Value_GHS"]

TYPO_FIXES = {
    "Branch_Type":      {"Standrad": "Standard", "Flag Ship": "Flagship", "Kios": "Kiosk"},
    "Customer_Segment": {"Retial": "Retail", "S.M.E": "SME", "Sme": "SME", "Corperate": "Corporate"},
    "Region":           {"Gt. Accra": "Greater Accra", "Accara": "Accra",
                          "Ashanti Region": "Ashanti", "Greater Accra Region": "Greater Accra"},
}

BOOL_MAP = {"yes": True, "y": True, "true": True, "1": True,
            "no": False, "n": False, "false": False, "0": False}

GRANULARITY_LOWER, GRANULARITY_UPPER = 0.4, 2.5
SENSITIVITY_CONFIGS = [(0.30, 2.00), (0.40, 2.50), (0.50, 3.00), (0.35, 2.20), (0.45, 2.80)]


# --- Section 6/8 core conversion functions (verbatim from the notebook) ------------------

def count_placeholders(series):
    return series.dropna().astype("string").str.strip().str.lower().isin(PLACEHOLDERS).sum()


def to_number(value):
    # Convert one messy currency/number string to float. Returns NaN if not a real number.
    if pd.isna(value):
        return np.nan
    t = str(value).replace("GHS", "").replace("₵", "").replace(",", "").strip()
    if t.lower() in PLACEHOLDERS:
        return np.nan
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else np.nan


def parse_date(value):
    # Format-aware date parser: slash format -> DD/MM/YYYY (UK), dash format -> MM-DD-YYYY (US).
    if pd.isna(value):
        return pd.NaT
    t = str(value).strip()
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", t):
        return pd.to_datetime(t, format="%d/%m/%Y", errors="coerce")
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", t):
        return pd.to_datetime(t, format="%m-%d-%Y", errors="coerce")
    return pd.to_datetime(t, errors="coerce", format="mixed")


def standardise_branch_code(value):
    # 'br-004', 'BR004', ' BR-004', 'Branch 004'  ->  'BR-004'
    if pd.isna(value):
        return np.nan
    t = str(value).strip().upper().replace("_", " ")
    t = re.sub(r"^BRANCH\s*", "BR-", t)
    t = re.sub(r"^BR[\s-]*", "BR-", t)
    m = re.fullmatch(r"BR-(\d+)", t)
    return f"BR-{int(m.group(1)):03d}" if m else t


def clean_text(series):
    # Collapse whitespace and tabs, strip, title-case.
    return (series.astype("string")
                  .str.replace(r"[\t_]+", " ", regex=True)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip()
                  .str.title())


# --- Loading -------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw():
    raw = pd.read_excel(DATA_PATH, sheet_name="Raw_Data", dtype=str)
    data_dict = pd.read_excel(DATA_PATH, sheet_name="Data_Dictionary", header=3).dropna(how="all")
    issues = pd.read_excel(DATA_PATH, sheet_name="Known_Data_Issues", header=2).dropna(how="all")
    return raw, data_dict, issues


# --- Section 6: raw data quality audit ------------------------------------------------------

def raw_quality_report(raw: pd.DataFrame) -> dict:
    isna_counts = raw.isna().sum()
    placeholder_counts = pd.Series(
        {col: count_placeholders(raw[col]) for col in raw.columns}, dtype="int64"
    )
    true_missing = isna_counts + placeholder_counts

    blank_rows = int(raw.isna().all(axis=1).sum())
    exact_dupes = int(raw.duplicated(keep="first").sum())

    parsed_dates = raw["Report_Month"].map(parse_date)
    unparseable_mask = parsed_dates.isna() & raw["Report_Month"].notna()
    date_fail_examples = raw.loc[unparseable_mask, "Report_Month"].value_counts()
    placeholder_1900 = int((parsed_dates.dt.year <= 1990).sum())

    revenue_num = raw["Monthly_Revenue_GHS"].map(to_number)
    headcount_num = raw["Headcount"].map(to_number)
    footfall_num = raw["Customer_Footfall"].map(to_number)

    categorical_before = {
        col: raw[col].nunique()
        for col in ["Branch_Code", "Region", "Branch_Type", "Customer_Segment", "New_Branch_Last_Quarter"]
    }

    conversion_check = []
    for col in CURRENCY_COLS:
        text_missing = int(raw[col].isna().sum() + count_placeholders(raw[col]))
        numeric_missing = int(raw[col].map(to_number).isna().sum())
        conversion_check.append({
            "column": col,
            "missing_as_text": text_missing,
            "missing_after_conversion": numeric_missing,
            "unexplained_loss": numeric_missing - text_missing,
        })

    return {
        "n_rows": len(raw),
        "n_cols": raw.shape[1],
        "isna_counts": isna_counts,
        "placeholder_counts": placeholder_counts,
        "true_missing": true_missing,
        "blank_rows": blank_rows,
        "exact_dupes": exact_dupes,
        "unparseable_dates": int(unparseable_mask.sum()),
        "placeholder_dates_1900": placeholder_1900,
        "date_fail_examples": date_fail_examples,
        "revenue_le_zero": int((revenue_num <= 0).sum()),
        "headcount_negative_or_sentinel": int((headcount_num.isin([-1, 999, 9999]) | (headcount_num < 0)).sum()),
        "footfall_negative": int((footfall_num < 0).sum()),
        "categorical_before": categorical_before,
        "conversion_check": pd.DataFrame(conversion_check),
    }


# --- Section 7.5: naive (wrong) aggregation, for the "before" comparison --------------------

def get_naive_aggregation(raw: pd.DataFrame) -> pd.Series:
    eda = raw.copy()
    eda["date"] = eda["Report_Month"].map(parse_date)
    eda["revenue"] = eda["Monthly_Revenue_GHS"].map(to_number)
    valid = eda.dropna(subset=["date"])
    valid = valid[valid["date"].dt.year > 1990]
    tmp = valid[valid["revenue"] > 0].copy()
    tmp["ym"] = tmp["date"].dt.to_period("M")
    naive_monthly = tmp.groupby("ym")["revenue"].sum()
    naive_monthly.index = naive_monthly.index.to_timestamp()
    return naive_monthly


# --- Section 8: the cleaning pipeline --------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_pipeline(raw: pd.DataFrame) -> dict:
    audit_log = []

    def log_step(name, df_before, df_after, note=""):
        removed = len(df_before) - len(df_after)
        audit_log.append({"step": name, "rows_before": len(df_before),
                           "rows_after": len(df_after), "rows_removed": removed, "note": note})

    work = raw.copy()

    # 8.1 Remove fully blank rows
    before = work
    work = work.dropna(how="all").copy()
    log_step("8.1 Drop fully blank rows", before, work, "corrupted export batch")

    # 8.2 Remove exact duplicate rows
    before = work
    work = work.drop_duplicates().copy()
    log_step("8.2 Drop exact duplicate rows", before, work, "re-run export appended twice")

    # 8.3 Standardise identifiers and categorical columns
    work["branch"] = work["Branch_Code"].map(standardise_branch_code)
    for col in ["Region", "Branch_Type", "Customer_Segment", "Branch_Name"]:
        work[col] = clean_text(work[col]).replace(TYPO_FIXES.get(col, {}))
        work.loc[work[col].str.lower().isin(PLACEHOLDERS), col] = pd.NA
    work["is_new_branch"] = (work["New_Branch_Last_Quarter"].astype("string")
                                 .str.strip().str.lower().map(BOOL_MAP))

    # 8.4 Parse dates, drop unrecoverable ones
    work["date"] = work["Report_Month"].map(parse_date)
    work["branch_open_date"] = work["Branch_Open_Date"].map(parse_date)
    before = work
    work = work[work["date"].notna() & (work["date"].dt.year > 1990)].copy()
    log_step("8.4 Drop broken / placeholder dates", before, work, "'TBD', 31/13/2023, 1900-01-01")

    # 8.5 Convert currency columns to numeric, then verify
    conversion_check = []
    for col in CURRENCY_COLS:
        text_missing = work[col].isna().sum() + count_placeholders(work[col])
        work[col + "_num"] = work[col].map(to_number)
        numeric_missing = work[col + "_num"].isna().sum()
        conversion_check.append({
            "column": col,
            "missing_as_text": text_missing,
            "missing_after_conversion": numeric_missing,
            "unexplained_loss": numeric_missing - text_missing,
        })
    work["revenue"] = work["Monthly_Revenue_GHS_num"]

    # 8.6 Remove impossible values
    before = work
    work = work[work["revenue"] > 0].copy()
    log_step("8.6 Drop non-positive revenue", before, work, "gross revenue cannot be <= 0")

    for col, sentinels in [("Headcount", [-1, 999, 9999]), ("Customer_Footfall", [])]:
        v = work[col].map(to_number)
        bad = v.isin(sentinels) | (v < 0)
        work[col + "_num"] = v.mask(bad)

    work_pre_granularity = work.copy()  # snapshot used for the sensitivity check and the "trap" illustration

    # 8.7 Resolve mixed reporting granularity — the critical step
    work["year_month"] = work["date"].dt.to_period("M")
    work["bm_median"] = work.groupby(["branch", "year_month"])["revenue"].transform("median")
    work["size_ratio"] = work["revenue"] / work["bm_median"]
    work["row_class"] = np.select(
        [work["size_ratio"] > GRANULARITY_UPPER, work["size_ratio"] < GRANULARITY_LOWER],
        ["coarser summary row", "implausibly small"],
        default="base period observation")

    row_class_counts = work["row_class"].value_counts()
    row_class_revenue_share = (work.groupby("row_class")["revenue"].sum() / work["revenue"].sum() * 100).round(2)

    before = work
    work = work[work["row_class"] == "base period observation"].copy()
    log_step("8.7 Keep base-granularity rows only", before, work, "removes double-counting")

    # 8.8 Remove near-duplicate records
    before = work
    work = (work.sort_values(["branch", "date", "revenue"])
                .drop_duplicates(subset=["branch", "date"], keep="first")
                .copy())
    log_step("8.8 One record per branch per date", before, work, "catches near-duplicates")

    # 8.9 Aggregate to branch-month, then to chain-month
    branch_monthly = (work.groupby(["branch", "year_month"], as_index=False)
                          .agg(revenue=("revenue", "sum"),
                               n_records=("revenue", "size")))
    chain_monthly = (branch_monthly.groupby("year_month", as_index=False)
                                   .agg(revenue=("revenue", "sum"),
                                        active_branches=("branch", "nunique")))
    chain_monthly["ds"] = chain_monthly["year_month"].dt.to_timestamp()

    log_df = pd.DataFrame(audit_log)
    log_df["pct_of_original"] = (log_df["rows_removed"] / len(raw) * 100).round(2)

    return {
        "chain_monthly": chain_monthly,
        "branch_monthly": branch_monthly,
        "audit_log": log_df,
        "conversion_check": pd.DataFrame(conversion_check),
        "row_class_counts": row_class_counts,
        "row_class_revenue_share": row_class_revenue_share,
        "work_pre_granularity": work_pre_granularity,
        "final_rows": len(work),
        "original_rows": len(raw),
    }


# --- Section 8.11: threshold sensitivity -----------------------------------------------------

def granularity_sensitivity(work_pre_granularity: pd.DataFrame) -> pd.DataFrame:
    pre = work_pre_granularity

    # Recompute size_ratio explicitly (pre-8.7 snapshot doesn't carry it).
    ym = pre["date"].dt.to_period("M")
    bm_median = pre.groupby(["branch", ym])["revenue"].transform("median")
    size_ratio = pre["revenue"] / bm_median

    def series_for_ratio(lower, upper):
        keep = pre[(size_ratio >= lower) & (size_ratio <= upper)].copy()
        keep["year_month"] = ym[keep.index]
        keep = keep.sort_values(["branch", "date", "revenue"]).drop_duplicates(["branch", "date"])
        return keep.groupby("year_month")["revenue"].sum()

    baseline_series = series_for_ratio(GRANULARITY_LOWER, GRANULARITY_UPPER)
    rows = []
    for lo, hi in SENSITIVITY_CONFIGS:
        s = series_for_ratio(lo, hi)
        diff = ((s - baseline_series) / baseline_series * 100).abs()
        rows.append({"lower": lo, "upper": hi, "total_GHS_m": s.sum() / 1e6,
                     "mean_abs_diff_%": diff.mean(), "max_abs_diff_%": diff.max()})
    return pd.DataFrame(rows).round(2)
