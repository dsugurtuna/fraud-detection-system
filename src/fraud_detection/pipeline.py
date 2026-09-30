"""End-to-end run: temporal split, features, training, selection, test metrics.

Mirrors the original script's flow as a function that returns results
instead of printing them, so it can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import BusinessEvaluator, EvaluationMetrics
from .features import FeatureEngineer
from .model import FraudModel, TrainConfig

NON_FEATURES = {"eventId", "isFraud", "reportedTime"}


def _fraud_value(df: pd.DataFrame) -> np.ndarray:
    """Regression target: the amount for fraud, 0 otherwise."""
    return (df["isFraud"] * df["transactionAmount"]).to_numpy()


@dataclass
class PipelineResult:
    """What a run produced."""

    test_metrics: EvaluationMetrics
    selected_model: str
    validation_captured: dict[str, float]
    rows: dict[str, int] = field(default_factory=dict)
    frauds: dict[str, int] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


def load_transactions(
    transactions_csv: str | Path, labels_csv: str | Path
) -> pd.DataFrame:
    """Load transactions plus a labels file of reported frauds.

    ``labels_csv`` lists ``eventId`` (and optionally ``reportedTime``) for
    transactions reported as fraud; every other transaction is labelled 0.
    """
    tx = pd.read_csv(transactions_csv)
    labels = pd.read_csv(labels_csv)
    keep = [c for c in ("eventId", "reportedTime") if c in labels.columns]
    df = tx.merge(labels[keep].assign(isFraud=1), on="eventId", how="left")
    df["isFraud"] = df["isFraud"].fillna(0).astype(int)
    df["transactionTime"] = pd.to_datetime(df["transactionTime"])
    if "reportedTime" in df.columns:
        df["reportedTime"] = pd.to_datetime(df["reportedTime"])
    return df.sort_values("transactionTime").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, val_start: str | pd.Timestamp, test_start: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by transaction time into train, validation and test periods."""
    t = df["transactionTime"]
    v, s = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if t.dt.tz is not None:
        v, s = v.tz_localize(t.dt.tz), s.tz_localize(t.dt.tz)
    return (
        df[t < v].copy(),
        df[(t >= v) & (t < s)].copy(),
        df[t >= s].copy(),
    )


def _known_by(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Labels as they were known at ``cutoff``: later fraud reports become 0."""
    if "reportedTime" not in df.columns:
        return df
    df = df.copy()
    reported = df["reportedTime"]
    if reported.dt.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(reported.dt.tz)
    late = (df["isFraud"] == 1) & reported.notna() & (reported >= cutoff)
    df.loc[late, "isFraud"] = 0
    return df


def run_pipeline(
    df: pd.DataFrame,
    val_start: str,
    test_start: str,
    capacity: int = 400,
    config: TrainConfig | None = None,
    respect_label_delay: bool = True,
) -> PipelineResult:
    """Train on the first period, choose a model on the second, test on the third.

    With ``respect_label_delay`` and a ``reportedTime`` column, training and
    validation only see frauds reported before the next period starts, as
    would be true in operation. Test metrics use every label.
    """
    fe = FeatureEngineer()
    df, cat_cols, _ = fe.create_base_features(df)
    train, val, test = temporal_split(df, val_start, test_start)
    if respect_label_delay:
        train = _known_by(train, pd.Timestamp(val_start))
        val = _known_by(val, pd.Timestamp(test_start))
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty or part["isFraud"].nunique() < 2:
            raise ValueError(f"{name} period needs both fraud and non-fraud rows")

    fe.fit(train)
    train_x = fe.transform_velocity(fe.transform_train(train), train.iloc[0:0])
    val_x = fe.transform_velocity(fe.transform(val), train)
    test_x = fe.transform_velocity(fe.transform(test), pd.concat([train, val]))

    features = [c for c in train_x.columns if c not in NON_FEATURES]
    cat_idx = [features.index(c) for c in cat_cols if c in features]
    y_tr = train["isFraud"].to_numpy()
    y_val = val["isFraud"].to_numpy()

    model = FraudModel(config)
    model.train_classifier(
        train_x[features],
        y_tr,
        train["transactionAmount"].to_numpy(),
        val_x[features],
        y_val,
        cat_idx,
    )
    model.fit_calibrator(model.raw_scores(val_x[features]), y_val)
    model.train_regressor(
        train_x[features],
        _fraud_value(train),
        val_x[features],
        _fraud_value(val),
        cat_idx,
    )

    evaluator = BusinessEvaluator(capacity)
    captured = model.select_by_validation(val, val_x[features], evaluator)

    amounts = test["transactionAmount"].to_numpy()
    metrics = evaluator.evaluate(
        test,
        model.predict_expected_value(test_x[features], amounts),
        is_ev_regressor=True,
    )
    # Ranking metrics for the calibrated classifier, whichever model was kept.
    from sklearn.metrics import average_precision_score, roc_auc_score

    proba = model.predict_proba(test_x[features])
    metrics.roc_auc = float(roc_auc_score(test["isFraud"], proba))
    metrics.pr_auc = float(average_precision_score(test["isFraud"], proba))

    parts = {"train": train, "validation": val, "test": test}
    return PipelineResult(
        test_metrics=metrics,
        selected_model="regressor" if model.use_regressor else "classifier",
        validation_captured=captured,
        rows={k: len(v) for k, v in parts.items()},
        frauds={k: int(np.sum(v["isFraud"])) for k, v in parts.items()},
        feature_names=features,
    )
