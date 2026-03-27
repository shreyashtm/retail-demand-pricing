"""
pricing.py
----------
Dynamic pricing recommendation engine.

Logic
-----
For each product segment (family × cluster) with a valid elasticity:

  For each candidate discount rate d in {5%, 10%, 15%, 20%, 25%, 30%}:
    - Simulate demand lift  = exp(elasticity × promo_change)
    - Simulate revenue      = (1 − d) × demand_lift × base_revenue

  Recommendation:
    - PROMOTE  if at least one discount rate yields revenue >= baseline
               AND demand lift >= MIN_DEMAND_LIFT (risk buffer)
    - Optimal discount = highest profitable discount found
    - DO NOT PROMOTE otherwise

Output columns
--------------
  family, cluster, elasticity,
  promote (bool),
  recommended_discount_pct,
  demand_lift_x,
  revenue_delta_pct,
  reasoning
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.elasticity import simulate_revenue_impact

PROCESSED_DIR      = Path("data/processed")
DISCOUNT_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MIN_DEMAND_LIFT     = 1.05   # require at least 5% demand lift before recommending


# ---------------------------------------------------------------------------
# Per-segment recommendation
# ---------------------------------------------------------------------------

def recommend_discount(
    elasticity          : float,
    promo_change        : float = 0.5,
    discount_candidates : list  = DISCOUNT_CANDIDATES,
) -> dict:
    """
    Find the optimal discount rate for one segment.

    Evaluates candidates from highest to lowest; returns the first
    (highest) discount that is both revenue-positive and exceeds the
    minimum demand lift threshold.

    Args:
        elasticity          : Segment elasticity coefficient.
        promo_change        : Assumed promo rate increase when promotion runs.
        discount_candidates : Discount rates to evaluate (ascending order).

    Returns:
        dict with promote, recommended_discount_pct, demand_lift_x,
             revenue_delta_pct, reasoning
    """
    for discount in sorted(discount_candidates, reverse=True):
        result = simulate_revenue_impact(
            elasticity   = elasticity,
            promo_change = promo_change,
            discount_pct = discount,
        )
        if result["is_profitable"] and result["demand_lift_x"] >= MIN_DEMAND_LIFT:
            return {
                "promote"                  : True,
                "recommended_discount_pct" : round(discount * 100, 1),
                "demand_lift_x"            : result["demand_lift_x"],
                "revenue_delta_pct"        : result["revenue_delta_pct"],
                "reasoning": (
                    f"A {discount*100:.0f}% discount is projected to lift demand "
                    f"by {(result['demand_lift_x']-1)*100:.1f}% and grow revenue "
                    f"by {result['revenue_delta_pct']:.1f}%."
                ),
            }

    # No profitable discount found
    lift = float(np.exp(elasticity * promo_change))
    return {
        "promote"                  : False,
        "recommended_discount_pct" : 0.0,
        "demand_lift_x"            : round(lift, 4),
        "revenue_delta_pct"        : 0.0,
        "reasoning": (
            "No tested discount rate produces a revenue gain for this segment. "
            "Promotion not recommended."
        ),
    }


# ---------------------------------------------------------------------------
# Full recommendations table
# ---------------------------------------------------------------------------

def generate_pricing_recommendations(elasticity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a recommendation row for every segment in elasticity_df.

    Returns a DataFrame sorted by revenue_delta_pct descending.
    """
    records = []

    for _, row in elasticity_df.iterrows():
        rec = recommend_discount(float(row["elasticity"]))
        records.append({
            "family"    : row["family"],
            "cluster"   : row["cluster"],
            "elasticity": row["elasticity"],
            **rec,
        })

    recs_df = (
        pd.DataFrame(records)
        .sort_values("revenue_delta_pct", ascending=False)
        .reset_index(drop=True)
    )

    return recs_df


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summarize_recommendations(recs_df: pd.DataFrame) -> None:
    """Print a formatted summary of pricing recommendations."""
    total     = len(recs_df)
    n_promote = int(recs_df["promote"].sum())
    n_skip    = total - n_promote

    print("\n" + "=" * 55)
    print("  Pricing Recommendation Summary")
    print("=" * 55)
    print(f"  Total segments evaluated : {total}")
    print(f"  Recommend promotion      : {n_promote}  "
          f"({n_promote/total*100:.1f}%)")
    print(f"  Do not promote           : {n_skip}  "
          f"({n_skip/total*100:.1f}%)")

    promo_df = recs_df[recs_df["promote"]]
    if len(promo_df) > 0:
        print(f"\n  Among promotable segments:")
        print(f"    Avg demand lift          : "
              f"{promo_df['demand_lift_x'].mean():.3f}x")
        print(f"    Avg revenue uplift       : "
              f"+{promo_df['revenue_delta_pct'].mean():.1f}%")
        print(f"    Avg recommended discount : "
              f"{promo_df['recommended_discount_pct'].mean():.1f}%")

        print("\n  Top 10 Segments by Revenue Upside")
        print("  " + "-" * 55)
        top = promo_df.head(10)[
            ["family", "cluster", "elasticity",
             "recommended_discount_pct", "revenue_delta_pct"]
        ]
        print(top.to_string(index=False))

    print("=" * 55)
