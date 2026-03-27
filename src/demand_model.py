"""
demand_model.py
---------------
Demand forecasting models with proper temporal train/validation split.

Models
------
1. MeanBaseline      — item × store historical mean  (benchmark to beat)
2. Ridge             — regularised linear regression on engineered features
3. RandomForest      — ensemble tree model for non-linear interactions

Split strategy
--------------
We split by DATE, not randomly.  A random split would leak future sales
patterns into the training window, producing falsely optimistic MAE scores.
Temporal split = train on earlier dates, validate on later dates.

Evaluation metric: Mean Absolute Error (MAE)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Temporal train / validation split
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame, val_weeks: int = 8):
    """
    Split df into train and validation by date.

    The last `val_weeks` weeks form the validation set.
    Everything before that is training data.

    Args:
        df        : DataFrame with a 'date' column (datetime).
        val_weeks : Weeks to hold out for validation.

    Returns:
        (train_df, val_df)
    """
    cutoff   = df["date"].max() - pd.Timedelta(weeks=val_weeks)
    train_df = df[df["date"] <= cutoff].copy()
    val_df   = df[df["date"] >  cutoff].copy()

    print(f"  Train : {train_df['date'].min().date()} → "
          f"{train_df['date'].max().date()}  ({len(train_df):,} rows)")
    print(f"  Val   : {val_df['date'].min().date()} → "
          f"{val_df['date'].max().date()}  ({len(val_df):,} rows)")

    return train_df, val_df


# ---------------------------------------------------------------------------
# Baseline model — item × store historical mean
# ---------------------------------------------------------------------------

class MeanBaseline:
    """
    Predicts the training-set mean unit_sales for each (store_nbr, item_nbr).

    This is the benchmark.  Any ML model that cannot beat this on MAE
    is not adding value over a simple average.
    """

    def __init__(self):
        self.means_       = None
        self.global_mean_ = None

    def fit(self, train_df: pd.DataFrame) -> "MeanBaseline":
        self.means_ = (
            train_df
            .groupby(["store_nbr", "item_nbr"])["unit_sales"]
            .mean()
            .rename("pred")
        )
        self.global_mean_ = float(train_df["unit_sales"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = (
            df[["store_nbr", "item_nbr"]]
            .merge(self.means_.reset_index(), # type: ignore
                   on=["store_nbr", "item_nbr"], how="left")["pred"]
            .fillna(self.global_mean_)
            .values
        )
        return preds # type: ignore

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds   = self.predict(df)
        actuals = df["unit_sales"].values
        return {
            "model" : "MeanBaseline",
            "MAE"   : round(float(mean_absolute_error(actuals, preds)), 4), # type: ignore
            "RMSE"  : round(float(np.sqrt(mean_squared_error(actuals, preds))), 4), # type: ignore
        }


# ---------------------------------------------------------------------------
# ML forecaster wrapper
# ---------------------------------------------------------------------------

class DemandForecaster:
    """
    Thin sklearn wrapper for Ridge and RandomForest demand models.

    Handles:
      - feature column selection (only columns that exist in the DataFrame)
      - StandardScaler for linear models
      - clipping predictions to >= 0 (demand cannot be negative)
      - consistent evaluate() and feature_importances() interface
    """

    SUPPORTED = {
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        ),
    }

    def __init__(self, model_name: str, feature_cols: list):
        if model_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(self.SUPPORTED)}"
            )
        self.model_name   = model_name
        self.feature_cols = feature_cols
        self.model_       = self.SUPPORTED[model_name]()
        self.scaler_      = None
        self.used_cols_   = None

    def _get_X(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        cols            = [c for c in self.feature_cols if c in df.columns]
        self.used_cols_ = cols
        subset = df[cols].fillna(0).copy()
        # Encode any remaining string / categorical columns to int codes
        for col in subset.select_dtypes(include=["object", "category"]).columns:
            subset[col] = subset[col].astype("category").cat.codes
        X = subset.values.astype("float32")

        if self.model_name == "ridge":
            if fit_scaler:
                self.scaler_ = StandardScaler()
                X = self.scaler_.fit_transform(X)
            elif self.scaler_ is not None:
                X = self.scaler_.transform(X)

        return X

    def fit(self, train_df: pd.DataFrame) -> "DemandForecaster":
        print(f"  Training {self.model_name} ...")
        X = self._get_X(train_df, fit_scaler=True)
        y = train_df["unit_sales"].values.astype("float32")
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X     = self._get_X(df)
        preds = self.model_.predict(X)
        return np.clip(preds, 0, None)

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds   = self.predict(df)
        actuals = df["unit_sales"].values
        return {
            "model" : self.model_name,
            "MAE"   : round(float(mean_absolute_error(actuals, preds)), 4), # type: ignore
            "RMSE"  : round(float(np.sqrt(mean_squared_error(actuals, preds))), 4), # type: ignore
        }

    def feature_importances(self):
        """Return sorted feature importances (tree models only)."""
        if not hasattr(self.model_, "feature_importances_"):
            return None
        return (
            pd.DataFrame({
                "feature"   : self.used_cols_,
                "importance": self.model_.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Model comparison runner
# ---------------------------------------------------------------------------

def run_model_comparison(
    train_df     : pd.DataFrame,
    val_df       : pd.DataFrame,
    feature_cols : list,
) -> tuple:
    """
    Train all three models on train_df, evaluate on val_df.

    Random Forest is trained on a sample (max 500k rows) to keep
    runtime reasonable on the full Favorita dataset.

    Returns:
        (results_df, baseline, ridge, rf)
    """
    results = []

    # 1. Mean baseline
    print("\n  --- MeanBaseline ---")
    baseline = MeanBaseline().fit(train_df)
    results.append(baseline.evaluate(val_df))

    # 2. Ridge
    print("\n  --- Ridge Regression ---")
    ridge = DemandForecaster("ridge", feature_cols).fit(train_df)
    results.append(ridge.evaluate(val_df))

    # 3. Random Forest (sample for speed)
    print("\n  --- Random Forest ---")
    sample_size  = min(500_000, len(train_df))
    train_sample = (
        train_df.sample(sample_size, random_state=42)
        if len(train_df) > sample_size else train_df
    )
    rf = DemandForecaster("random_forest", feature_cols).fit(train_sample)
    results.append(rf.evaluate(val_df))

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 45)
    print("  Model Comparison — Validation Set")
    print("=" * 45)
    print(results_df.to_string(index=False))
    print("=" * 45)

    return results_df, baseline, ridge, rf
