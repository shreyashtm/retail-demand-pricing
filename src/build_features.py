"""
build_features.py
-----------------
Feature engineering for the demand forecasting pipeline.

Features produced
-----------------
Calendar   : day_of_week, month, year, week_of_year, is_weekend,
             day_of_month, quarter
Lag        : lag_1, lag_7, lag_14, lag_28  (per store × item)
Rolling    : rolling_mean_7, rolling_mean_28, rolling_std_7
Promo      : onpromotion (current), promo_lag_7 (was on promo last 7 days)
Contextual : transactions (if available), oil_price (if available)
Encoded    : family, city, state, type, cluster  → integer codes

All operations return a NEW DataFrame — input is never mutated.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.dayofweek.astype("int8")   # 0=Mon
    df["month"]        = df["date"].dt.month.astype("int8")
    df["year"]         = df["date"].dt.year.astype("int16")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["quarter"]      = df["date"].dt.quarter.astype("int8")
    return df


# ---------------------------------------------------------------------------
# Lag & rolling features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(store_nbr, item_nbr) lag and rolling statistics.

    IMPORTANT: df must be sorted by date BEFORE calling this function.
    Lags respect temporal ordering — no future leakage.
    """
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()
    grp = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]

    # Lag features — shift(n) looks n days back
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            grp.shift(lag).fillna(0).astype("float32")
        )

    # Rolling statistics on the already-shifted series (shift(1) = no leakage)
    shifted = grp.shift(1)

    df["rolling_mean_7"] = (
        shifted.transform(lambda s: s.rolling(7,  min_periods=1).mean())
        .fillna(0).astype("float32")
    )
    df["rolling_mean_28"] = (
        shifted.transform(lambda s: s.rolling(28, min_periods=1).mean())
        .fillna(0).astype("float32")
    )
    df["rolling_std_7"] = (
        shifted.transform(lambda s: s.rolling(7,  min_periods=2).std())
        .fillna(0).astype("float32")
    )

    # Promo lag: was item on promo any day in the last 7 days?
    promo_grp = df.groupby(["store_nbr", "item_nbr"])["onpromotion"]
    df["promo_lag_7"] = (
        promo_grp.shift(1)
        .transform(lambda s: s.rolling(7, min_periods=1).max())
        .fillna(0).astype("int8")
    )

    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode low-cardinality string columns to integer codes."""
    df = df.copy()
    for col in ["family", "city", "state", "type", "cluster"]:
        if col in df.columns and (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object):
            df[col] = df[col].astype("category").cat.codes.astype("int16")
    return df


# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

# Core features always present after build_features()
FEATURE_COLS = [
    # Calendar
    "day_of_week", "month", "year", "week_of_year",
    "is_weekend", "day_of_month", "quarter",
    # Lag
    "lag_1", "lag_7", "lag_14", "lag_28",
    # Rolling
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
    # Promo
    "onpromotion", "promo_lag_7",
    # Holiday
    "is_national_holiday", "is_local_holiday",
    # Store / item
    "store_nbr", "cluster", "family",
    # Optional (included only if present in df)
    # "transactions", "oil_price", "perishable"
]

# Optional columns — added to FEATURE_COLS if present in the DataFrame
OPTIONAL_COLS = ["transactions", "oil_price", "perishable"]

TARGET_COL = "unit_sales"


def get_active_feature_cols(df: pd.DataFrame) -> list:
    """Return FEATURE_COLS + any OPTIONAL_COLS present in df."""
    optional_present = [c for c in OPTIONAL_COLS if c in df.columns]
    return FEATURE_COLS + optional_present


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete feature engineering and return enriched DataFrame."""
    print("[build_features] Engineering features ...")
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = encode_categoricals(df)
    print(f"  Output shape: {df.shape}")
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Return (X, y, feature_names) ready for sklearn.

    Rows with any NaN in feature columns are dropped (first rows per
    group where lags cannot be computed).
    """
    cols     = get_active_feature_cols(df)
    present  = [c for c in cols if c in df.columns]
    df_clean = df[present + [TARGET_COL]].dropna()

    X = df_clean[present].values.astype("float32")
    y = df_clean[TARGET_COL].values.astype("float32")
    return X, y, present
