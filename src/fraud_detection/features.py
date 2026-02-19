"""Feature engineering module.

Temporal, categorical, aggregate, and velocity features
for transaction fraud detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SMOOTH_M = 50.0


@dataclass
class FeatureStores:
    """Pre-computed aggregate stores fitted on training data."""

    account_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    merchant_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mcc_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_encodings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    global_fraud_rate: float = 0.0
    numeric_medians: Dict[str, float] = field(default_factory=dict)


class FeatureEngineer:
    """Create features for fraud detection.

    Follows a strict fit/transform pattern to prevent data leakage.
    Aggregates and risk encodings are fitted on training data only.

    Parameters
    ----------
    smooth_m : float
        Smoothing factor for m-estimate risk encodings.
    """

    def __init__(self, smooth_m: float = SMOOTH_M) -> None:
        self.smooth_m = smooth_m
        self.stores: Optional[FeatureStores] = None
        self.cat_features = [
            "accountNumber", "merchantId", "mcc",
            "merchantCountry", "merchantZip", "posEntryMode",
        ]
        self.num_features: List[str] = []

    # ------------------------------------------------------------------
    # Base features
    # ------------------------------------------------------------------
    @staticmethod
    def create_base_features(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Create temporal and categorical base features."""
        df = df.copy()
        df["hour"] = df["transactionTime"].dt.hour
        df["dayOfWeek"] = df["transactionTime"].dt.dayofweek
        df["isWeekend"] = (df["dayOfWeek"] >= 5).astype(int)
        df["isNight"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["logAmount"] = np.log1p(df["transactionAmount"])

        cat_features = [
            "accountNumber", "merchantId", "mcc",
            "merchantCountry", "merchantZip", "posEntryMode",
        ]
        df["merchantCountry"] = df["merchantCountry"].astype(str)

        num_features: List[str] = []
        for c in ["transactionAmount", "availableCash"]:
            if c in df.columns:
                num_features.append(c)

        for c in cat_features:
            if c in df.columns:
                df[c] = df[c].fillna("UNK")

        return df, cat_features, num_features

    # ------------------------------------------------------------------
    # Aggregates (fit / transform)
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """Fit aggregate stores on training data."""
        stores = FeatureStores()

        # Account-level statistics
        acc = train_df.groupby("accountNumber").agg(
            acc_mean=("transactionAmount", "mean"),
            acc_std=("transactionAmount", "std"),
            acc_max=("transactionAmount", "max"),
            acc_count=("transactionAmount", "count"),
            acc_merchants=("merchantId", "nunique"),
            acc_mccs=("mcc", "nunique"),
        ).fillna(0)
        stores.account_stats = acc.to_dict(orient="index")

        # Merchant statistics
        mch = train_df.groupby("merchantId").agg(
            mean_amount=("transactionAmount", "mean"),
            std_amount=("transactionAmount", "std"),
            tx_count=("transactionAmount", "count"),
        ).fillna(0)
        stores.merchant_stats = mch.to_dict(orient="index")

        # MCC statistics
        mcc = train_df.groupby("mcc").agg(
            mean_amount=("transactionAmount", "mean"),
            std_amount=("transactionAmount", "std"),
            tx_count=("transactionAmount", "count"),
        ).fillna(0)
        stores.mcc_stats = mcc.to_dict(orient="index")

        # Smoothed risk encodings
        gfr = max(1e-6, train_df["isFraud"].mean())
        stores.global_fraud_rate = gfr

        for col in ["merchantId", "mcc", "merchantCountry", "posEntryMode"]:
            stores.risk_encodings[col] = self._smoothed_rate(
                train_df, col, self.smooth_m, gfr
            )

        stores.numeric_medians = train_df.median(numeric_only=True).to_dict()
        self.stores = stores
        return self

    @staticmethod
    def _smoothed_rate(
        df: pd.DataFrame, col: str, m: float, global_rate: float
    ) -> Dict[str, float]:
        grp = (
            df.groupby(col)["isFraud"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "fraud", "count": "n"})
        )
        grp["rate"] = (grp["fraud"] + m * global_rate) / (grp["n"] + m)
        return grp["rate"].to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted aggregates and risk encodings."""
        if self.stores is None:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        stores = self.stores

        # Account stats
        for col in ["acc_mean", "acc_std", "acc_max", "acc_count", "acc_merchants", "acc_mccs"]:
            df[col] = df["accountNumber"].map(
                {k: v.get(col, 0) for k, v in stores.account_stats.items()}
            ).fillna(0)

        # Merchant stats
        for col in ["mean_amount", "std_amount", "tx_count"]:
            df[f"mch_{col}"] = df["merchantId"].map(
                {k: v.get(col, 0) for k, v in stores.merchant_stats.items()}
            ).fillna(0)

        # MCC stats
        for col in ["mean_amount", "std_amount", "tx_count"]:
            df[f"mcc_{col}"] = df["mcc"].map(
                {k: v.get(col, 0) for k, v in stores.mcc_stats.items()}
            ).fillna(0)

        # Risk encodings
        for col in ["merchantId", "mcc", "merchantCountry", "posEntryMode"]:
            risk_map = stores.risk_encodings.get(col, {})
            df[f"{col}_risk"] = df[col].map(risk_map).fillna(stores.global_fraud_rate)

        return df

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------
    @staticmethod
    def transform_velocity(
        df: pd.DataFrame, historical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute velocity features using only historical transactions."""
        hist_cols = ["accountNumber", "merchantId", "mcc", "transactionTime"]
        hist_subset = historical_df[hist_cols].copy()
        df_subset = df[hist_cols].copy()
        df_index = df.index

        combined = pd.concat([hist_subset, df_subset], axis=0, ignore_index=True)
        combined = combined.sort_values(["accountNumber", "transactionTime"])
        combined["time_since_last_acc"] = (
            combined.groupby("accountNumber")["transactionTime"]
            .diff()
            .dt.total_seconds()
        )
        combined = combined.sort_values(["accountNumber", "merchantId", "transactionTime"])
        combined["time_since_last_mch"] = (
            combined.groupby(["accountNumber", "merchantId"])["transactionTime"]
            .diff()
            .dt.total_seconds()
        )
        combined = combined.sort_values(["accountNumber", "mcc", "transactionTime"])
        combined["time_since_last_mcc"] = (
            combined.groupby(["accountNumber", "mcc"])["transactionTime"]
            .diff()
            .dt.total_seconds()
        )
        vel = combined.loc[
            df_index,
            ["time_since_last_acc", "time_since_last_mch", "time_since_last_mcc"],
        ]
        return pd.concat([df.drop(columns=["transactionTime"]), vel], axis=1)
