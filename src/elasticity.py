"""
elasticity.py
-------------
Promotion elasticity estimation using a log-linear OLS model.

Dataset context
---------------
The Favorita dataset has NO price column.  We use the binary
`onpromotion` flag as a price-change instrument.  A promotion
typically represents a 15–30% price reduction.

Method
------
For each (family × cluster) segment:
  1. Aggregate to daily total_demand and mean promo_rate
  2. Compute log(total_demand)
  3. Fit OLS: log(demand) ~ promo_rate
  4. The coefficient is the log-linear elasticity

Interpretation
--------------
  elasticity = 0.4 →  a 10pp increase in promo_rate
               is associated with ~4% demand lift
  elasticity ≤ 0 →  promotions do not lift demand in this segment

Outputs saved
-------------
  data/processed/elasticity.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PROCESSED_DIR = Path("data/processed")

MIN_OBS         = 30    # minimum observations per segment
MIN_PROMO_DAYS  = 5     # minimum days with promo_rate > 0


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def build_daily_promo_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales to daily (family × cluster) level.

    Returns columns: date, family, cluster, total_demand, promo_rate,
                     log_demand
    """
    daily = (
        df
        .groupby(["date", "family", "cluster"])
        .agg(
            total_demand=("unit_sales",   "sum"),
            promo_rate  =("onpromotion",  "mean"),
        )
        .reset_index()
    )

    # Drop zero-demand days (log undefined; often data gaps)
    daily = daily[daily["total_demand"] > 0].copy()
    daily["log_demand"] = np.log(daily["total_demand"]).astype("float32")

    return daily


# ---------------------------------------------------------------------------
# Per-segment elasticity estimation
# ---------------------------------------------------------------------------

def _estimate_segment(group: pd.DataFrame) -> dict:
    """OLS elasticity for one (family, cluster) segment."""
    n          = len(group)
    promo_days = (group["promo_rate"] > 0).sum()

    if n < MIN_OBS or promo_days < MIN_PROMO_DAYS:
        return {"elasticity": np.nan, "n_obs": n, "r_squared": np.nan}

    X = group[["promo_rate"]].values
    y = group["log_demand"].values

    model  = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    return {
        "elasticity": round(float(model.coef_[0]), 4),
        "n_obs"     : n,
        "r_squared" : round(float(r2_score(y, y_pred)), 4),
    }


def compute_elasticity(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate promotion elasticity for every (family × cluster) pair.

    Returns a DataFrame sorted by elasticity descending.
    """
    records = []

    for (family, cluster), group in daily_df.groupby(["family", "cluster"]):
        rec = _estimate_segment(group)
        records.append({"family": family, "cluster": cluster, **rec})

    out = (
        pd.DataFrame(records)
        .dropna(subset=["elasticity"])
        .sort_values("elasticity", ascending=False)
        .reset_index(drop=True)
    )

    return out


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_demand_lift(elasticity: float, promo_change: float) -> float:
    """
    Multiplicative demand lift from the log-linear model.

      demand_new / demand_base = exp(elasticity × Δpromo_rate)

    Args:
        elasticity   : Segment elasticity coefficient.
        promo_change : Change in promo rate (e.g., 0.5 = 50pp increase).

    Returns:
        Demand multiplier (> 1 = lift, < 1 = decline).
    """
    return float(np.exp(elasticity * promo_change))


def simulate_revenue_impact(
    elasticity   : float,
    promo_change : float,
    discount_pct : float,
    base_price   : float = 100.0,
    base_demand  : float = 1_000.0,
) -> dict:
    """
    Project revenue change under a given promotion scenario.

    Args:
        elasticity   : Segment elasticity.
        promo_change : Change in promo rate (0–1).
        discount_pct : Price markdown applied (e.g., 0.20 = 20% off).
        base_price   : Reference price (default 100 for relative analysis).
        base_demand  : Reference daily demand units.

    Returns dict:
        base_revenue, promo_revenue, demand_lift_x,
        revenue_delta_pct, is_profitable
    """
    demand_lift   = simulate_demand_lift(elasticity, promo_change)
    base_revenue  = base_price * base_demand
    promo_price   = base_price * (1 - discount_pct)
    promo_demand  = base_demand * demand_lift
    promo_revenue = promo_price * promo_demand
    delta_pct     = (promo_revenue - base_revenue) / base_revenue * 100

    return {
        "base_revenue"      : round(base_revenue,  2),
        "promo_revenue"     : round(promo_revenue, 2),
        "demand_lift_x"     : round(demand_lift,   4),
        "revenue_delta_pct" : round(delta_pct,     2),
        "is_profitable"     : promo_revenue >= base_revenue,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_elasticity_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end: build daily dataset → estimate elasticity → save parquet.

    Args:
        df : Enriched DataFrame (output of load_and_merge).

    Returns:
        elasticity_df
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[elasticity] Building daily promo dataset ...")
    daily_df = build_daily_promo_dataset(df)
    print(f"  Daily rows : {len(daily_df):,}")

    print("[elasticity] Estimating per-segment elasticity ...")
    elasticity_df = compute_elasticity(daily_df)
    print(f"  Valid segments : {len(elasticity_df)}")

    out_path = PROCESSED_DIR / "elasticity.parquet"
    elasticity_df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")

    # Summary stats
    e = elasticity_df["elasticity"]
    print(f"\n  Elasticity stats")
    print(f"    mean   : {e.mean():.3f}")
    print(f"    median : {e.median():.3f}")
    print(f"    min    : {e.min():.3f}")
    print(f"    max    : {e.max():.3f}")
    n_pos = (e > 0).sum()
    print(f"\n  Segments where promotions lift demand : "
          f"{n_pos} / {len(elasticity_df)}")

    return elasticity_df
