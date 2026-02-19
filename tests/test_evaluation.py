"""Tests for BusinessEvaluator."""

import numpy as np
import pandas as pd

from fraud_detection.evaluation import BusinessEvaluator, EvaluationMetrics


def _make_test_data(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-06-01", periods=n, freq="h")
    n_fraud = max(1, n // 10)
    labels = [0] * (n - n_fraud) + [1] * n_fraud
    return pd.DataFrame({
        "transactionTime": dates,
        "transactionAmount": rng.uniform(10, 500, n),
        "isFraud": labels,
    })


class TestBusinessEvaluator:
    def test_evaluate_returns_metrics(self):
        df = _make_test_data()
        scores = np.random.default_rng(1).random(len(df))
        ev = BusinessEvaluator(review_capacity=50)
        m = ev.evaluate(df, scores)
        assert isinstance(m, EvaluationMetrics)
        assert m.captured_value_gbp >= 0
        assert m.uplift_vs_random_x >= 0

    def test_higher_capacity_captures_more(self):
        df = _make_test_data()
        scores = np.random.default_rng(1).random(len(df))
        m_low = BusinessEvaluator(review_capacity=10).evaluate(df, scores)
        m_high = BusinessEvaluator(review_capacity=50).evaluate(df, scores)
        assert m_high.captured_value_gbp >= m_low.captured_value_gbp

    def test_ev_curve(self):
        df = _make_test_data()
        scores = np.random.default_rng(1).random(len(df))
        ev = BusinessEvaluator(review_capacity=50)
        curve = ev.ev_curve(df, scores, k_list=[10, 25, 50])
        assert len(curve) == 3
        assert list(curve.columns) == ["K", "captured", "baseline", "uplift_x"]

    def test_to_dict(self):
        m = EvaluationMetrics(roc_auc=0.95, captured_value_gbp=1000.0)
        d = m.to_dict()
        assert d["roc_auc"] == 0.95
        assert d["captured_value_gbp"] == 1000.0

    def test_random_baseline_is_monotone(self):
        df = _make_test_data(200)
        scores = np.random.default_rng(1).random(len(df))
        baselines = []
        for k in [10, 50, 100]:
            ev = BusinessEvaluator(review_capacity=k)
            m = ev.evaluate(df, scores)
            baselines.append(m.baseline_random_value_gbp)
        assert baselines == sorted(baselines)
