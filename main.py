"""
main.py
-------
End-to-end pipeline for the Retail Demand Forecasting &
Dynamic Pricing Engine.

Usage:
  python main.py

Stages:
  1. Load & merge raw Favorita CSVs
  2. Feature engineering
  3. Temporal train / validation split
  4. Train & compare forecasting models (Baseline, Ridge, RandomForest)
  5. Estimate promotion elasticity by product segment
  6. Generate dynamic pricing recommendations
  7. Save all outputs to data/processed/
"""

import pandas as pd
from pathlib import Path

from src.load_data      import load_and_merge
from src.build_features import build_features, get_active_feature_cols
from src.demand_model   import time_based_split, run_model_comparison
from src.elasticity     import run_elasticity_pipeline
from src.pricing        import generate_pricing_recommendations, summarize_recommendations

PROCESSED_DIR = Path("data/processed")

def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def main():
    banner("Retail Demand Forecasting & Dynamic Pricing Engine")

    # ------------------------------------------------------------------
    # Stage 1 — Load & merge
    # ------------------------------------------------------------------
    from src.load_data import SAMPLE_ROWS
    print(f"\n[1/6] Loading and merging data ...")
    print(f"  Sample size: {SAMPLE_ROWS:,} rows" if SAMPLE_ROWS else "  Sample size: full dataset")
    print(f"  To change: edit SAMPLE_ROWS in src/load_data.py")
    enriched_path = PROCESSED_DIR / "favorita_enriched.parquet"

    if enriched_path.exists():
        print(f"  Cached enriched dataset found at {enriched_path}")
        print("  Loading from cache (delete data/processed/favorita_enriched.parquet to re-merge) ...")
        df          = pd.read_parquet(enriched_path)
        df["date"]  = pd.to_datetime(df["date"])
    else:
        df = load_and_merge()

    print(f"  Dataset shape  : {df.shape}")
    print(f"  Date range     : {df['date'].min().date()} → "
          f"{df['date'].max().date()}")
    print(f"  Unique stores  : {df['store_nbr'].nunique()}")
    print(f"  Unique items   : {df['item_nbr'].nunique()}")

    # ------------------------------------------------------------------
    # Stage 2 — Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/6] Building features ...")
    df = build_features(df)

    # ------------------------------------------------------------------
    # Stage 3 — Temporal split
    # ------------------------------------------------------------------
    print("\n[3/6] Splitting data (temporal, last 8 weeks = validation) ...")
    train_df, val_df = time_based_split(df, val_weeks=8)

    # ------------------------------------------------------------------
    # Stage 4 — Model comparison
    # ------------------------------------------------------------------
    print("\n[4/6] Training and evaluating forecasting models ...")
    feature_cols = get_active_feature_cols(df)

    results_df, baseline, ridge, rf = run_model_comparison(
        train_df, val_df, feature_cols
    )

    # Save model results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results_path = PROCESSED_DIR / "model_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"\n  Model results saved → {results_path}")

    # Save feature importances
    fi = rf.feature_importances()
    if fi is not None:
        fi_path = PROCESSED_DIR / "feature_importances.parquet"
        fi.to_parquet(fi_path, index=False)
        print(f"  Feature importances saved → {fi_path}")
        print("\n  Top 10 Features (Random Forest):")
        print("  " + "-" * 40)
        print(fi.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Stage 5 — Elasticity estimation
    # ------------------------------------------------------------------
    print("\n[5/6] Estimating promotion elasticity ...")
    elasticity_df = run_elasticity_pipeline(df)

    # ------------------------------------------------------------------
    # Stage 6 — Pricing recommendations
    # ------------------------------------------------------------------
    print("\n[6/6] Generating pricing recommendations ...")
    recs_df = generate_pricing_recommendations(elasticity_df)
    summarize_recommendations(recs_df)

    recs_path = PROCESSED_DIR / "pricing_recommendations.parquet"
    recs_df.to_parquet(recs_path, index=False)
    print(f"\n  Recommendations saved → {recs_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    banner("Pipeline complete")
    print("  Output files:")
    for p in sorted(PROCESSED_DIR.glob("*.parquet")):
        size_kb = p.stat().st_size // 1024
        print(f"    {p.name:<45} {size_kb:>6} KB")
    print()


if __name__ == "__main__":
    main()
