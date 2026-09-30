"""End-to-end run on synthetic data (trains small CatBoost models)."""

import pandas as pd
import pytest

from fraud_detection.model import TrainConfig
from fraud_detection.pipeline import _known_by, run_pipeline, temporal_split
from fraud_detection.synthetic import make_transactions


def test_synthetic_data_is_reproducible():
    a = make_transactions(n=500, seed=3)
    b = make_transactions(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a["transactionTime"].is_monotonic_increasing


def test_temporal_split_has_no_overlap():
    df = make_transactions(n=2000, seed=1)
    train, val, test = temporal_split(df, "2024-06-01", "2024-07-15")
    assert len(train) + len(val) + len(test) == len(df)
    assert train["transactionTime"].max() < val["transactionTime"].min()
    assert val["transactionTime"].max() < test["transactionTime"].min()


def test_late_fraud_reports_are_hidden_from_earlier_periods():
    df = pd.DataFrame(
        {
            "isFraud": [1, 1, 0],
            "reportedTime": pd.to_datetime(["2024-01-05", "2024-03-01", None]),
        }
    )
    known = _known_by(df, pd.Timestamp("2024-02-01"))
    assert known["isFraud"].tolist() == [1, 0, 0]


def test_pipeline_end_to_end():
    df = make_transactions(n=12_000, seed=5)
    result = run_pipeline(
        df, "2024-06-01", "2024-07-15", capacity=30, config=TrainConfig(iterations=60)
    )
    assert sum(result.rows.values()) == len(df)
    assert result.selected_model in {"classifier", "regressor"}
    assert "eventId" not in result.feature_names
    assert "isFraud" not in result.feature_names
    m = result.test_metrics
    assert 0.0 <= m.roc_auc <= 1.0
    # The synthetic data has a strong planted signal, so ranking by expected
    # value should beat random review.
    assert m.uplift_vs_random_x > 1.0


def test_pipeline_rejects_period_without_fraud():
    df = make_transactions(n=3000, seed=2)
    df["isFraud"] = 0
    with pytest.raises(ValueError, match="needs both fraud and non-fraud"):
        run_pipeline(df, "2024-06-01", "2024-07-15")
