"""
evaluate.py
-----------
Standalone evaluation and diagnostic utilities.

Run this script after main.py to print a full results report:
    python -m src.evaluate

Functions:
  - print_model_report()      : load and display model comparison results
  - print_elasticity_report() : summarise elasticity estimates
  - print_pricing_report()    : top recommendations with reasoning
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")


def print_model_report():
    path = PROCESSED / "model_results.parquet"
    if not path.exists():
        print("[!] model_results.parquet not found. Run main.py first.")
        return

    results = pd.read_parquet(path)
    baseline_mae = results.loc[results["model"] == "MeanBaseline", "MAE"].values[0]

    print("\n" + "=" * 55)
    print("  DEMAND FORECASTING — MODEL COMPARISON")
    print("=" * 55)
    print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8}  {'vs Baseline':>12}")
    print("  " + "-" * 51)

    for _, row in results.iterrows():
        delta = (row["MAE"] - baseline_mae) / baseline_mae * 100
        sign  = "+" if delta > 0 else ""
        marker = "  ← baseline" if row["model"] == "MeanBaseline" else f"  {sign}{delta:.1f}%"
        print(f"  {row['model']:<20} {row['MAE']:>8.4f} {row['RMSE']:>8.4f}{marker}")

    print()

    fi_path = PROCESSED / "feature_importances.parquet"
    if fi_path.exists():
        fi = pd.read_parquet(fi_path).head(10)
        print("  TOP 10 FEATURE IMPORTANCES (Random Forest)")
        print("  " + "-" * 40)
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<22} {row['importance']:.4f}  {bar}")
        print()


def print_elasticity_report():
    path = PROCESSED / "elasticity.parquet"
    if not path.exists():
        print("[!] elasticity.parquet not found. Run main.py first.")
        return

    el = pd.read_parquet(path)

    print("=" * 55)
    print("  PROMOTION ELASTICITY SUMMARY")
    print("=" * 55)
    print(f"  Segments estimated       : {len(el)}")
    print(f"  Positive elasticity (lift): {(el['elasticity'] > 0).sum()}")
    print(f"  Negative elasticity (drag): {(el['elasticity'] < 0).sum()}")
    print(f"  Mean elasticity           : {el['elasticity'].mean():.4f}")
    print(f"  Median elasticity         : {el['elasticity'].median():.4f}")
    print(f"  Std dev                   : {el['elasticity'].std():.4f}")
    print()

    print("  TOP 10 MOST ELASTIC SEGMENTS")
    print("  " + "-" * 45)
    top = el.head(10)[["family", "cluster", "elasticity", "r_squared", "n_obs"]]
    print(top.to_string(index=False))
    print()

    print("  LEAST ELASTIC SEGMENTS (bottom 5)")
    print("  " + "-" * 45)
    bot = el.tail(5)[["family", "cluster", "elasticity", "r_squared", "n_obs"]]
    print(bot.to_string(index=False))
    print()


def print_pricing_report():
    path = PROCESSED / "pricing_recommendations.parquet"
    if not path.exists():
        print("[!] pricing_recommendations.parquet not found. Run main.py first.")
        return

    recs = pd.read_parquet(path)
    promote = recs[recs["promote"]]
    skip    = recs[~recs["promote"]]

    print("=" * 55)
    print("  DYNAMIC PRICING RECOMMENDATIONS")
    print("=" * 55)
    print(f"  Total segments            : {len(recs)}")
    print(f"  Promote recommended       : {len(promote)} ({len(promote)/len(recs)*100:.1f}%)")
    print(f"  Do not promote            : {len(skip)}  ({len(skip)/len(recs)*100:.1f}%)")

    if len(promote) > 0:
        print(f"\n  Avg demand lift           : {promote['demand_lift_x'].mean():.3f}x")
        print(f"  Avg revenue delta         : +{promote['revenue_delta_pct'].mean():.1f}%")
        print(f"  Avg recommended discount  : {promote['recommended_discount_pct'].mean():.1f}%")

        print("\n  TOP 15 SEGMENTS BY REVENUE UPSIDE")
        print("  " + "-" * 55)
        cols = ["family", "cluster", "elasticity", "recommended_discount_pct", "revenue_delta_pct"]
        print(promote[cols].head(15).to_string(index=False))

    print()


def full_report():
    print_model_report()
    print_elasticity_report()
    print_pricing_report()


if __name__ == "__main__":
    full_report()
