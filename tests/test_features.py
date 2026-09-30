"""Tests for FeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import FeatureEngineer


def _make_transactions(n: int = 200) -> pd.DataFrame:
    """Generate synthetic transaction data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "eventId": range(n),
            "transactionTime": dates,
            "transactionAmount": rng.exponential(50, n),
            "availableCash": rng.uniform(0, 5000, n),
            "accountNumber": rng.choice(["A001", "A002", "A003"], n),
            "merchantId": rng.choice(["M01", "M02", "M03"], n),
            "mcc": rng.choice(["5411", "5812", "7011"], n),
            "merchantCountry": rng.choice(["GB", "US", "DE"], n),
            "merchantZip": rng.choice(["CB1", "CB2", "SW1"], n),
            "posEntryMode": rng.choice(["chip", "swipe", "online"], n),
            "isFraud": rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], n),
        }
    )
    return df


class TestBaseFeatures:
    def test_temporal_features(self):
        df = _make_transactions(50)
        result, cat_cols, num_cols = FeatureEngineer.create_base_features(df)
        assert "hour" in result.columns
        assert "dayOfWeek" in result.columns
        assert "isWeekend" in result.columns
        assert "isNight" in result.columns
        assert "hour_sin" in result.columns
        assert "logAmount" in result.columns

    def test_cat_features(self):
        df = _make_transactions(50)
        _, cat_cols, _ = FeatureEngineer.create_base_features(df)
        assert "accountNumber" in cat_cols
        assert "merchantId" in cat_cols

    def test_num_features(self):
        df = _make_transactions(50)
        _, _, num_cols = FeatureEngineer.create_base_features(df)
        assert "transactionAmount" in num_cols

    def test_missing_cats_filled(self):
        df = _make_transactions(20)
        df.loc[0, "mcc"] = None
        result, _, _ = FeatureEngineer.create_base_features(df)
        assert result.loc[0, "mcc"] == "UNK"


class TestAggregates:
    def test_fit_creates_stores(self):
        df = _make_transactions(100)
        df, _, _ = FeatureEngineer.create_base_features(df)
        fe = FeatureEngineer()
        fe.fit(df)
        assert fe.stores is not None
        assert fe.stores.global_fraud_rate > 0

    def test_transform_adds_columns(self):
        df = _make_transactions(100)
        df, _, _ = FeatureEngineer.create_base_features(df)
        fe = FeatureEngineer().fit(df)
        result = fe.transform(df)
        assert "acc_mean" in result.columns
        assert "mch_mean_amount" in result.columns
        assert "merchantId_risk" in result.columns

    def test_transform_before_fit_raises(self):
        fe = FeatureEngineer()
        df = _make_transactions(10)
        with pytest.raises(RuntimeError):
            fe.transform(df)


class TestVelocity:
    def test_velocity_columns(self):
        df = _make_transactions(50)
        df, _, _ = FeatureEngineer.create_base_features(df)
        result = FeatureEngineer.transform_velocity(df, df.iloc[0:0])
        assert "time_since_last_acc" in result.columns
        assert "time_since_last_mch" in result.columns
        assert "transactionTime" not in result.columns


def test_velocity_uses_history_for_first_current_row():
    hist = pd.DataFrame(
        {
            "accountNumber": ["A", "A"],
            "merchantId": ["M", "M"],
            "mcc": ["1", "1"],
            "transactionTime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
        }
    )
    cur = pd.DataFrame(
        {
            "accountNumber": ["A", "B"],
            "merchantId": ["M", "M"],
            "mcc": ["1", "1"],
            "transactionTime": pd.to_datetime(["2024-01-01 03:00", "2024-01-01 04:00"]),
            "transactionAmount": [5.0, 6.0],
        },
        index=[100, 101],
    )
    out = FeatureEngineer.transform_velocity(cur, hist)
    assert list(out.index) == [100, 101]
    assert out.loc[100, "time_since_last_acc"] == 7200.0
    assert out.loc[100, "time_since_last_mch"] == 7200.0
    assert np.isnan(out.loc[101, "time_since_last_acc"])


def test_categoricals_become_strings():
    df = _make_transactions(10)
    df["mcc"] = [5411] * 9 + [None]  # pandas stores this as float64
    result, _, _ = FeatureEngineer.create_base_features(df)
    assert result["mcc"].tolist() == ["5411"] * 9 + ["UNK"]


def test_numeric_gaps_use_training_median():
    train, _, _ = FeatureEngineer.create_base_features(_make_transactions(50))
    fe = FeatureEngineer().fit(train)
    test = train.head(3).copy()
    test.loc[test.index[0], "availableCash"] = np.nan
    out = fe.transform(test)
    assert out.iloc[0]["availableCash"] == train["availableCash"].median()


def test_training_risk_encoding_is_out_of_fold():
    df = _make_transactions(200)
    df.loc[0, "merchantId"] = "ONLY_ONCE"
    df.loc[0, "isFraud"] = 1
    df, _, _ = FeatureEngineer.create_base_features(df)
    fe = FeatureEngineer(smooth_m=1.0).fit(df)
    in_sample = fe.transform(df).loc[0, "merchantId_risk"]
    oof = fe.transform_train(df, n_folds=5).loc[0, "merchantId_risk"]
    # In-sample, the row's own fraud label raises its merchant's rate. Out of
    # fold the merchant is unseen, so it falls back to the global rate.
    assert in_sample > 0.5
    assert oof < 0.5
