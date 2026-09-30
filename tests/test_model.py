"""Tests for FraudModel that do not need CatBoost."""

import numpy as np
import pandas as pd

from fraud_detection.evaluation import BusinessEvaluator
from fraud_detection.model import FraudModel


class _StubClassifier:
    def __init__(self, p: np.ndarray) -> None:
        self.p = p

    def predict_proba(self, X: object) -> np.ndarray:
        return np.column_stack([1 - self.p, self.p])


class _StubRegressor:
    def __init__(self, ev: np.ndarray) -> None:
        self.ev = ev

    def predict(self, X: object) -> np.ndarray:
        return self.ev


def _val_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transactionTime": pd.date_range("2024-01-01", periods=6, freq="D"),
            "transactionAmount": [10.0, 500.0, 20.0, 30.0, 40.0, 50.0],
            "isFraud": [1, 1, 0, 0, 0, 0],
        }
    )


def test_value_weights():
    y = np.array([0, 0, 0, 1])
    amounts = np.array([1.0, 1.0, 1.0, np.e - 1])
    w = FraudModel._value_weights(y, amounts, exp=1.0)
    assert w[:3].tolist() == [1.0, 1.0, 1.0]
    # (1 - 0.25) / 0.25 = 3, times (1 + log(e)) = 2
    assert np.isclose(w[3], 6.0)


def test_expected_value_uses_calibrated_probability():
    model = FraudModel()
    model.classifier = _StubClassifier(np.array([0.2, 0.8]))
    model.fit_calibrator(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    ev = model.predict_expected_value(None, np.array([100.0, 100.0]))
    assert ev.tolist() == [0.0, 100.0]


def test_select_by_validation_prefers_higher_captured_value():
    val = _val_frame()
    model = FraudModel()
    # The classifier ranks the small fraud first; the regressor finds the 500.
    model.classifier = _StubClassifier(np.array([0.9, 0.01, 0.1, 0.1, 0.1, 0.1]))
    model.regressor = _StubRegressor(np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0]))
    captured = model.select_by_validation(val, None, BusinessEvaluator(1))
    assert captured == {"classifier": 10.0, "regressor": 500.0}
    assert model.use_regressor


def test_without_regressor_classifier_is_kept():
    model = FraudModel()
    model.classifier = _StubClassifier(np.full(6, 0.5))
    captured = model.select_by_validation(_val_frame(), None, BusinessEvaluator(2))
    assert set(captured) == {"classifier"}
    assert not model.use_regressor
