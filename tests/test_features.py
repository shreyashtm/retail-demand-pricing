"""
tests/test_features.py
-----------------------
Unit tests for build_features.py

Run:  python -m pytest tests/ -v          (if pytest installed)
  or: python -m unittest tests.test_features -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from src.build_features import (
    add_calendar_features, add_lag_features,
    encode_categoricals, build_features,
    get_active_feature_cols, TARGET_COL,
)


def make_df(n_stores=2, n_items=2, n_days=40):
    rng   = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")
    rows  = []
    for s in range(1, n_stores + 1):
        for i in range(1, n_items + 1):
            for d in dates:
                rows.append({
                    "date": d, "store_nbr": s, "item_nbr": i,
                    "unit_sales": float(rng.integers(0, 50)),
                    "onpromotion": int(rng.integers(0, 2)),
                    "family": "GROCERY", "cluster": 1,
                    "city": "Quito", "state": "Pichincha", "type": "A",
                    "is_national_holiday": 0, "is_local_holiday": 0,
                })
    return pd.DataFrame(rows)


class TestCalendarFeatures(unittest.TestCase):
    def setUp(self): self.df = make_df(n_days=14)

    def test_expected_columns_created(self):
        out = add_calendar_features(self.df)
        for col in ["day_of_week","month","year","week_of_year","is_weekend","day_of_month","quarter"]:
            self.assertIn(col, out.columns)

    def test_day_of_week_range(self):
        out = add_calendar_features(self.df)
        self.assertTrue(out["day_of_week"].between(0, 6).all())

    def test_is_weekend_binary(self):
        out = add_calendar_features(self.df)
        self.assertTrue(set(out["is_weekend"].unique()).issubset({0, 1}))

    def test_month_range(self):
        out = add_calendar_features(make_df(n_days=365))
        self.assertTrue(out["month"].between(1, 12).all())

    def test_does_not_mutate_input(self):
        df = make_df(n_days=10); orig = set(df.columns)
        add_calendar_features(df)
        self.assertEqual(set(df.columns), orig)

    def test_row_count_unchanged(self):
        df = make_df(n_days=10)
        self.assertEqual(len(add_calendar_features(df)), len(df))


class TestLagFeatures(unittest.TestCase):
    def setUp(self): self.df = make_df(n_stores=2, n_items=2, n_days=35)

    def test_lag_columns_created(self):
        out = add_lag_features(self.df)
        for lag in [1, 7, 14, 28]:
            self.assertIn(f"lag_{lag}", out.columns)

    def test_rolling_columns_created(self):
        out = add_lag_features(self.df)
        for col in ["rolling_mean_7","rolling_mean_28","rolling_std_7"]:
            self.assertIn(col, out.columns)

    def test_promo_lag_created(self):
        self.assertIn("promo_lag_7", add_lag_features(self.df).columns)

    def test_no_negative_lags(self):
        out = add_lag_features(self.df)
        for lag in [1, 7, 14, 28]:
            self.assertTrue((out[f"lag_{lag}"] >= 0).all())

    def test_row_count_preserved(self):
        self.assertEqual(len(add_lag_features(self.df)), len(self.df))

    def test_lag1_first_row_is_zero(self):
        df  = make_df(n_stores=1, n_items=1, n_days=10)
        out = add_lag_features(df).sort_values("date")
        self.assertEqual(out.iloc[0]["lag_1"], 0.0)

    def test_does_not_mutate_input(self):
        df = make_df(n_days=35); orig = set(df.columns)
        add_lag_features(df)
        self.assertEqual(set(df.columns), orig)


class TestEncoding(unittest.TestCase):
    def setUp(self): self.df = make_df(n_days=5)

    def test_family_becomes_numeric(self):
        out = encode_categoricals(self.df)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["family"]))

    def test_no_nan_after_encoding(self):
        out = encode_categoricals(self.df)
        for col in ["family","city","state","type"]:
            self.assertEqual(out[col].isna().sum(), 0)

    def test_cluster_stays_numeric(self):
        out = encode_categoricals(self.df)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["cluster"]))


class TestBuildFeatures(unittest.TestCase):
    def test_full_pipeline_runs(self):
        df = make_df(n_stores=2, n_items=2, n_days=40)
        out = build_features(df)
        self.assertGreater(len(out.columns), len(df.columns))

    def test_active_cols_present_in_df(self):
        df  = make_df(n_days=40)
        out = build_features(df)
        for c in get_active_feature_cols(out):
            self.assertIn(c, out.columns)

    def test_target_col_present(self):
        self.assertIn(TARGET_COL, build_features(make_df(n_days=40)).columns)


if __name__ == "__main__":
    unittest.main()
