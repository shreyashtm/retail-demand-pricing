"""
tests/test_models.py
---------------------
Unit tests for demand_model.py

Run:  python -m pytest tests/ -v
  or: python -m unittest tests.test_models -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from src.build_features import build_features, get_active_feature_cols
from src.demand_model   import time_based_split, MeanBaseline, DemandForecaster


def make_featured_df(n_days=80):
    rng   = np.random.default_rng(1)
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    rows  = []
    for s in [1, 2]:
        for i in [10, 20]:
            for d in dates:
                rows.append({
                    "date": d, "store_nbr": s, "item_nbr": i,
                    "unit_sales": float(rng.integers(1, 40)),
                    "onpromotion": int(rng.integers(0, 2)),
                    "family": "GROCERY", "cluster": 1,
                    "is_national_holiday": 0, "is_local_holiday": 0,
                })
    return build_features(pd.DataFrame(rows))


class TestTimeSplit(unittest.TestCase):
    def setUp(self):
        self.df = make_featured_df(n_days=60)

    def test_no_overlap(self):
        train, val = time_based_split(self.df, val_weeks=4)
        self.assertLess(train["date"].max(), val["date"].min())

    def test_no_data_loss(self):
        train, val = time_based_split(self.df, val_weeks=4)
        self.assertEqual(len(train) + len(val), len(self.df))

    def test_val_is_later(self):
        train, val = time_based_split(self.df, val_weeks=4)
        self.assertGreater(val["date"].min(), train["date"].min())


class TestMeanBaseline(unittest.TestCase):
    def setUp(self):
        df = make_featured_df()
        self.train, self.val = time_based_split(df, val_weeks=4)
        self.model = MeanBaseline().fit(self.train)

    def test_predict_shape(self):
        preds = self.model.predict(self.val)
        self.assertEqual(len(preds), len(self.val))

    def test_predictions_non_negative(self):
        preds = self.model.predict(self.val)
        self.assertTrue((preds >= 0).all())

    def test_evaluate_returns_mae(self):
        result = self.model.evaluate(self.val)
        self.assertIn("MAE", result)
        self.assertGreater(result["MAE"], 0)

    def test_unseen_items_no_nan(self):
        fake = self.val.copy()
        fake["item_nbr"] = 99999
        preds = self.model.predict(fake)
        self.assertFalse(np.any(np.isnan(preds)))


class TestDemandForecaster(unittest.TestCase):
    def setUp(self):
        df = make_featured_df(n_days=80)
        self.train, self.val = time_based_split(df, val_weeks=4)
        self.feature_cols    = get_active_feature_cols(df)

    def test_ridge_fit_predict(self):
        model = DemandForecaster("ridge", self.feature_cols).fit(self.train)
        preds = model.predict(self.val)
        self.assertEqual(len(preds), len(self.val))

    def test_rf_fit_predict(self):
        model = DemandForecaster("random_forest", self.feature_cols).fit(self.train)
        preds = model.predict(self.val)
        self.assertEqual(len(preds), len(self.val))

    def test_predictions_non_negative(self):
        for name in ["ridge", "random_forest"]:
            preds = DemandForecaster(name, self.feature_cols).fit(self.train).predict(self.val)
            self.assertTrue((preds >= 0).all(), f"{name} produced negative predictions")

    def test_evaluate_keys(self):
        result = DemandForecaster("ridge", self.feature_cols).fit(self.train).evaluate(self.val)
        for key in ["model","MAE","RMSE"]:
            self.assertIn(key, result)

    def test_rf_feature_importances(self):
        model = DemandForecaster("random_forest", self.feature_cols).fit(self.train)
        fi    = model.feature_importances()
        self.assertIsNotNone(fi)
        self.assertIn("feature",    fi.columns)
        self.assertIn("importance", fi.columns)

    def test_ridge_no_feature_importances(self):
        model = DemandForecaster("ridge", self.feature_cols).fit(self.train)
        self.assertIsNone(model.feature_importances())

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            DemandForecaster("xgboost", self.feature_cols)


if __name__ == "__main__":
    unittest.main()
