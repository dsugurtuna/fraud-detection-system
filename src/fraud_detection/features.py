"""Feature engineering for transaction fraud detection.

Temporal, categorical, aggregate, risk-encoding and velocity features. All
statistics are fitted on the training period only. For the training rows
themselves, :meth:`FeatureEngineer.transform_train` computes risk encodings
out of fold, so no row's own label feeds into its features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH_M = 50.0
RISK_COLS = ["merchantId", "mcc", "merchantCountry", "posEntryMode"]
NUMERIC_COLS = ["transactionAmount", "availableCash"]


def _as_category(values: pd.Series) -> pd.Series:
    """Categorical codes as strings, with ``UNK`` for missing values.

    An integer code column with gaps is read by pandas as float (5411.0);
    it is converted back to whole numbers so the code stays "5411".
    """
    if pd.api.types.is_float_dtype(values) and values.dropna().mod(1).eq(0).all():
        values = values.astype("Int64")
    return values.astype(object).where(values.notna(), "UNK").astype(str)


@dataclass
class FeatureStores:
    """Pre-computed aggregate stores fitted on training data."""

    account_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    merchant_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    mcc_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    risk_encodings: dict[str, dict[str, float]] = field(default_factory=dict)
    global_fraud_rate: float = 0.0
    numeric_medians: dict[str, float] = field(default_factory=dict)


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
        self.stores: FeatureStores | None = None
        self.cat_features = [
            "accountNumber",
            "merchantId",
            "mcc",
            "merchantCountry",
            "merchantZip",
            "posEntryMode",
        ]
        self.num_features: list[str] = []

    # ------------------------------------------------------------------
    # Base features
    # ------------------------------------------------------------------
    @staticmethod
    def create_base_features(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
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
            "accountNumber",
            "merchantId",
            "mcc",
            "merchantCountry",
            "merchantZip",
            "posEntryMode",
        ]
        num_features = [c for c in NUMERIC_COLS if c in df.columns]

        # CatBoost needs categorical values as strings (or ints), never floats,
        # so fill missing values first and then cast.
        for c in cat_features:
            if c in df.columns:
                df[c] = _as_category(df[c])

        return df, cat_features, num_features

    # ------------------------------------------------------------------
    # Aggregates (fit / transform)
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> FeatureEngineer:
        """Fit aggregate stores on training data."""
        stores = FeatureStores()

        # Account-level statistics
        acc = (
            train_df.groupby("accountNumber")
            .agg(
                acc_mean=("transactionAmount", "mean"),
                acc_std=("transactionAmount", "std"),
                acc_max=("transactionAmount", "max"),
                acc_count=("transactionAmount", "count"),
                acc_merchants=("merchantId", "nunique"),
                acc_mccs=("mcc", "nunique"),
            )
            .fillna(0)
        )
        stores.account_stats = acc.to_dict(orient="index")

        # Merchant statistics
        mch = (
            train_df.groupby("merchantId")
            .agg(
                mean_amount=("transactionAmount", "mean"),
                std_amount=("transactionAmount", "std"),
                tx_count=("transactionAmount", "count"),
            )
            .fillna(0)
        )
        stores.merchant_stats = mch.to_dict(orient="index")

        # MCC statistics
        mcc = (
            train_df.groupby("mcc")
            .agg(
                mean_amount=("transactionAmount", "mean"),
                std_amount=("transactionAmount", "std"),
                tx_count=("transactionAmount", "count"),
            )
            .fillna(0)
        )
        stores.mcc_stats = mcc.to_dict(orient="index")

        # Smoothed risk encodings
        gfr = max(1e-6, train_df["isFraud"].mean())
        stores.global_fraud_rate = gfr

        for col in RISK_COLS:
            stores.risk_encodings[col] = self._smoothed_rate(
                train_df, col, self.smooth_m, gfr
            )

        present = [c for c in NUMERIC_COLS if c in train_df.columns]
        stores.numeric_medians = train_df[present].median().to_dict()
        self.stores = stores
        return self

    @staticmethod
    def _smoothed_rate(
        df: pd.DataFrame, col: str, m: float, global_rate: float
    ) -> dict[str, float]:
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

        # Impute numeric gaps with training medians only.
        for col, median in stores.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # Account stats
        for col in [
            "acc_mean",
            "acc_std",
            "acc_max",
            "acc_count",
            "acc_merchants",
            "acc_mccs",
        ]:
            df[col] = (
                df["accountNumber"]
                .map({k: v.get(col, 0) for k, v in stores.account_stats.items()})
                .fillna(0)
            )

        # Merchant stats
        for col in ["mean_amount", "std_amount", "tx_count"]:
            df[f"mch_{col}"] = (
                df["merchantId"]
                .map({k: v.get(col, 0) for k, v in stores.merchant_stats.items()})
                .fillna(0)
            )

        # MCC stats
        for col in ["mean_amount", "std_amount", "tx_count"]:
            df[f"mcc_{col}"] = (
                df["mcc"]
                .map({k: v.get(col, 0) for k, v in stores.mcc_stats.items()})
                .fillna(0)
            )

        # Risk encodings
        for col in RISK_COLS:
            risk_map = stores.risk_encodings.get(col, {})
            df[f"{col}_risk"] = df[col].map(risk_map).fillna(stores.global_fraud_rate)

        return df

    def transform_train(
        self, train_df: pd.DataFrame, n_folds: int = 5, seed: int = 0
    ) -> pd.DataFrame:
        """Transform the training rows with out-of-fold risk encodings.

        Plain :meth:`transform` on the training set would encode each
        merchant's fraud rate using the very rows being labelled, which leaks
        the target into the features and makes the model over-trust them.
        Here each fold's risk encodings come from the other folds only. The
        other aggregates carry no labels and are unchanged.
        """
        out = self.transform(train_df)
        if n_folds < 2:
            return out
        folds = np.random.default_rng(seed).integers(0, n_folds, len(train_df))
        for k in range(n_folds):
            rest = train_df[folds != k]
            gfr = max(1e-6, float(rest["isFraud"].mean()))
            for col in RISK_COLS:
                rates = self._smoothed_rate(rest, col, self.smooth_m, gfr)
                in_fold = folds == k
                out.loc[in_fold, f"{col}_risk"] = (
                    train_df.loc[in_fold, col].map(rates).fillna(gfr).to_numpy()
                )
        return out

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------
    @staticmethod
    def transform_velocity(
        df: pd.DataFrame, historical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Seconds since the previous transaction by account, merchant and MCC.

        ``historical_df`` supplies earlier transactions (for example the
        training period) so the first transactions in ``df`` are not all
        missing. Rows of ``df`` are matched back by position, so any index
        works, and a historical row at the same timestamp sorts first.
        """
        cols = ["accountNumber", "merchantId", "mcc", "transactionTime"]
        hist = historical_df[cols].assign(_current=0, _pos=-1)
        cur = df[cols].assign(_current=1, _pos=np.arange(len(df)))
        combined = pd.concat([hist, cur], axis=0, ignore_index=True)

        groups = {
            "time_since_last_acc": ["accountNumber"],
            "time_since_last_mch": ["accountNumber", "merchantId"],
            "time_since_last_mcc": ["accountNumber", "mcc"],
        }
        for name, keys in groups.items():
            combined = combined.sort_values(
                [*keys, "transactionTime", "_current"], kind="mergesort"
            )
            combined[name] = (
                combined.groupby(keys)["transactionTime"].diff().dt.total_seconds()
            )

        current = combined[combined["_current"] == 1].sort_values("_pos")
        vel = pd.DataFrame(current[list(groups)].to_numpy(), columns=list(groups))
        vel.index = df.index
        return pd.concat([df.drop(columns=["transactionTime"]), vel], axis=1)
