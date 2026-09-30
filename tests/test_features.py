"""Tests for FeatureEngineer."""

import numpy as np
import pandas as pd

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
        try:
            fe.transform(df)
            assert False, "Should have raised"
        except RuntimeError:
            pass


class TestVelocity:
    def test_velocity_columns(self):
        df = _make_transactions(50)
        df, _, _ = FeatureEngineer.create_base_features(df)
        result = FeatureEngineer.transform_velocity(df, df.iloc[0:0])
        assert "time_since_last_acc" in result.columns
        assert "time_since_last_mch" in result.columns
        assert "transactionTime" not in result.columns
