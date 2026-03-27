"""
tests/test_elasticity.py
------------------------
Unit tests for elasticity.py and pricing.py

Run:  python -m pytest tests/ -v
  or: python -m unittest tests.test_elasticity -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from src.elasticity import (
    build_daily_promo_dataset, compute_elasticity,
    simulate_demand_lift, simulate_revenue_impact,
)
from src.pricing import recommend_discount, generate_pricing_recommendations


def make_sales_df(n_days=90):
    rng   = np.random.default_rng(7)
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    rows  = []
    for store in [1, 2]:
        for item in [100, 200, 300]:
            promo = rng.integers(0, 2, size=n_days)
            sales = (rng.integers(50, 150, size=n_days).astype(float)
                     + promo * rng.uniform(10, 30, size=n_days))
            for j, d in enumerate(dates):
                rows.append({
                    "date": d, "store_nbr": store, "item_nbr": item,
                    "unit_sales": sales[j], "onpromotion": int(promo[j]),
                    "family": "GROCERY" if item <= 200 else "BEVERAGES",
                    "cluster": 1 if store == 1 else 2,
                })
    return pd.DataFrame(rows)


class TestDailyPromoDataset(unittest.TestCase):
    def setUp(self):
        self.daily = build_daily_promo_dataset(make_sales_df())

    def test_output_columns(self):
        for col in ["date","family","cluster","total_demand","promo_rate","log_demand"]:
            self.assertIn(col, self.daily.columns)

    def test_no_zero_demand_rows(self):
        self.assertTrue((self.daily["total_demand"] > 0).all())

    def test_log_demand_finite(self):
        self.assertTrue(np.isfinite(self.daily["log_demand"]).all())

    def test_promo_rate_in_range(self):
        self.assertTrue(self.daily["promo_rate"].between(0, 1).all())


class TestElasticity(unittest.TestCase):
    def setUp(self):
        daily        = build_daily_promo_dataset(make_sales_df())
        self.el_df   = compute_elasticity(daily)

    def test_output_columns(self):
        for col in ["family","cluster","elasticity","n_obs","r_squared"]:
            self.assertIn(col, self.el_df.columns)

    def test_no_nan_elasticity(self):
        self.assertEqual(self.el_df["elasticity"].isna().sum(), 0)

    def test_n_obs_positive(self):
        self.assertTrue((self.el_df["n_obs"] > 0).all())

    def test_r_squared_valid(self):
        self.assertTrue(self.el_df["r_squared"].between(0, 1).all())


class TestSimulation(unittest.TestCase):

    def test_zero_change_gives_multiplier_1(self):
        self.assertAlmostEqual(simulate_demand_lift(0.5, 0.0), 1.0, places=6)

    def test_positive_elasticity_lifts_demand(self):
        self.assertGreater(simulate_demand_lift(0.5, 0.5), 1.0)

    def test_negative_elasticity_reduces_demand(self):
        self.assertLess(simulate_demand_lift(-0.5, 0.5), 1.0)

    def test_revenue_impact_keys(self):
        result = simulate_revenue_impact(0.8, 0.5, 0.20)
        for key in ["base_revenue","promo_revenue","demand_lift_x",
                    "revenue_delta_pct","is_profitable"]:
            self.assertIn(key, result)

    def test_high_elasticity_is_profitable(self):
        result = simulate_revenue_impact(2.0, 0.5, 0.20)
        self.assertTrue(result["is_profitable"])

    def test_zero_elasticity_not_profitable(self):
        result = simulate_revenue_impact(0.0, 0.5, 0.10)
        self.assertFalse(result["is_profitable"])

    def test_base_revenue_calculation(self):
        result = simulate_revenue_impact(1.0, 0.5, 0.20, base_price=50.0, base_demand=200.0)
        self.assertAlmostEqual(result["base_revenue"], 10_000.0, places=1)


class TestPricing(unittest.TestCase):

    def test_high_elasticity_gets_recommendation(self):
        rec = recommend_discount(2.0, promo_change=0.5)
        self.assertTrue(rec["promote"])
        self.assertGreater(rec["recommended_discount_pct"], 0)

    def test_low_elasticity_no_recommendation(self):
        rec = recommend_discount(0.01, promo_change=0.5)
        self.assertFalse(rec["promote"])

    def test_reasoning_is_string(self):
        rec = recommend_discount(1.5)
        self.assertIsInstance(rec["reasoning"], str)
        self.assertGreater(len(rec["reasoning"]), 10)

    def test_generate_recommendations_shape(self):
        daily = build_daily_promo_dataset(make_sales_df())
        el_df = compute_elasticity(daily)
        recs  = generate_pricing_recommendations(el_df)
        self.assertEqual(len(recs), len(el_df))
        for col in ["family","cluster","promote","recommended_discount_pct","revenue_delta_pct"]:
            self.assertIn(col, recs.columns)


if __name__ == "__main__":
    unittest.main()
