"""Model training and calibration module.

CatBoost classifier with value-weighted loss, isotonic calibration,
and direct expected-value regressor for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

AMOUNT_WEIGHT_EXP = 1.0


@dataclass
class TrainConfig:
    """Training configuration."""

    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_state: int = 42
    amount_weight_exp: float = AMOUNT_WEIGHT_EXP
    calibrate: bool = True


class FraudModel:
    """Value-weighted fraud detection model.

    Wraps CatBoost classifier with value-weighted loss and
    optional isotonic calibration.

    Parameters
    ----------
    config : TrainConfig
        Training parameters.
    """

    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()
        self.classifier = None
        self.regressor = None
        self.calibrator = None
        self.use_regressor: bool = False

    def _build_catboost_params(self, *, loss: str, metric: str) -> dict:
        cfg = self.config
        return {
            "loss_function": loss,
            "eval_metric": metric,
            "iterations": cfg.iterations,
            "learning_rate": cfg.learning_rate,
            "depth": cfg.depth,
            "l2_leaf_reg": cfg.l2_leaf_reg,
            "random_seed": cfg.random_state,
            "verbose": False,
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": 50,
            "task_type": "CPU",
        }

    @staticmethod
    def _value_weights(
        y: np.ndarray, amounts: np.ndarray, exp: float
    ) -> np.ndarray:
        """Compute value-weighted sample weights."""
        fraud_rate = max(1e-6, float(np.mean(y)))
        base_pos = (1.0 - fraud_rate) / fraud_rate
        w = np.ones_like(y, dtype=float)
        pos = y == 1
        w[pos] = base_pos * (1.0 + np.log1p(amounts[pos])) ** exp
        return w

    def train_classifier(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        amounts_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cat_idx: List[int],
    ) -> None:
        """Train value-weighted CatBoost classifier."""
        from catboost import CatBoostClassifier

        weights = self._value_weights(y_tr, amounts_tr, self.config.amount_weight_exp)
        params = self._build_catboost_params(loss="Logloss", metric="AUC")
        self.classifier = CatBoostClassifier(**params)
        self.classifier.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_idx,
            sample_weight=weights,
            verbose=False,
        )

    def train_regressor(
        self,
        X_tr: np.ndarray,
        y_ev_tr: np.ndarray,
        X_val: np.ndarray,
        y_ev_val: np.ndarray,
        cat_idx: List[int],
    ) -> None:
        """Train direct expected-value regressor."""
        from catboost import CatBoostRegressor

        params = self._build_catboost_params(loss="RMSE", metric="RMSE")
        self.regressor = CatBoostRegressor(**params)
        self.regressor.fit(
            X_tr, y_ev_tr,
            eval_set=(X_val, y_ev_val),
            cat_features=cat_idx,
            verbose=False,
        )

    def fit_calibrator(
        self, val_scores: np.ndarray, y_val: np.ndarray
    ) -> None:
        """Fit isotonic calibration on validation scores."""
        from sklearn.isotonic import IsotonicRegression

        order = np.argsort(val_scores)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_scores[order], y_val[order])
        self.calibrator = iso

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate scores using the selected model."""
        if self.use_regressor and self.regressor is not None:
            return np.clip(self.regressor.predict(X), 0.0, None)
        if self.classifier is None:
            raise RuntimeError("No classifier trained.")
        scores = self.classifier.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            scores = np.clip(self.calibrator.predict(scores), 0.0, 1.0)
        return scores
