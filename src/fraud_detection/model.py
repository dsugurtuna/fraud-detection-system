"""Model training, calibration and selection.

Two ways to rank transactions for review:

* a CatBoost classifier trained with value-weighted sample weights, whose
  scores are calibrated with isotonic regression and multiplied by the
  transaction amount to give an expected fraud value, and
* a CatBoost regressor that predicts ``isFraud * amount`` directly.

``select_by_validation`` keeps whichever captures more fraud value on the
validation period at the given review capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from .evaluation import BusinessEvaluator

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
    early_stopping_rounds: int = 50


class FraudModel:
    """Value-weighted fraud model with optional EV regressor."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        # CatBoost and scikit-learn estimators; typed loosely because the
        # libraries do not ship complete type information.
        self.classifier: Any = None
        self.regressor: Any = None
        self.calibrator: Any = None
        self.use_regressor: bool = False

    def _build_catboost_params(self, *, loss: str, metric: str) -> dict[str, Any]:
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
            "od_wait": cfg.early_stopping_rounds,
            "task_type": "CPU",
        }

    @staticmethod
    def _value_weights(y: np.ndarray, amounts: np.ndarray, exp: float) -> np.ndarray:
        """Sample weights: class balance times a log-amount factor for fraud.

        Non-fraud rows weigh 1. Fraud rows weigh ``(1 - p) / p`` (to offset
        class imbalance, with ``p`` the fraud rate) times
        ``(1 + log(1 + amount)) ** exp``, so missing a large fraud costs more
        than missing a small one. The weights distort predicted
        probabilities, which is why the classifier is calibrated afterwards.
        """
        fraud_rate = max(1e-6, float(np.mean(y)))
        base_pos = (1.0 - fraud_rate) / fraud_rate
        w = np.ones_like(y, dtype=float)
        pos = y == 1
        w[pos] = base_pos * (1.0 + np.log1p(amounts[pos])) ** exp
        return w

    def train_classifier(
        self,
        X_tr: Any,
        y_tr: np.ndarray,
        amounts_tr: np.ndarray,
        X_val: Any,
        y_val: np.ndarray,
        cat_idx: list[int],
    ) -> None:
        """Train the value-weighted classifier (early stopping on validation)."""
        from catboost import CatBoostClassifier

        weights = self._value_weights(y_tr, amounts_tr, self.config.amount_weight_exp)
        params = self._build_catboost_params(loss="Logloss", metric="AUC")
        self.classifier = CatBoostClassifier(**params)
        self.classifier.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_idx,
            sample_weight=weights,
            verbose=False,
        )

    def train_regressor(
        self,
        X_tr: Any,
        y_ev_tr: np.ndarray,
        X_val: Any,
        y_ev_val: np.ndarray,
        cat_idx: list[int],
    ) -> None:
        """Train a regressor on the expected value ``isFraud * amount``."""
        from catboost import CatBoostRegressor

        params = self._build_catboost_params(loss="RMSE", metric="RMSE")
        self.regressor = CatBoostRegressor(**params)
        self.regressor.fit(
            X_tr,
            y_ev_tr,
            eval_set=(X_val, y_ev_val),
            cat_features=cat_idx,
            verbose=False,
        )

    def raw_scores(self, X: Any) -> np.ndarray:
        """Uncalibrated classifier probabilities for the fraud class."""
        if self.classifier is None:
            raise RuntimeError("No classifier trained.")
        return np.asarray(self.classifier.predict_proba(X)[:, 1], dtype=float)

    def fit_calibrator(self, val_scores: np.ndarray, y_val: np.ndarray) -> None:
        """Fit isotonic calibration from raw validation scores to outcomes."""
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_scores, y_val)
        self.calibrator = iso

    def predict_proba(self, X: Any) -> np.ndarray:
        """Calibrated fraud probability (raw probability if not calibrated)."""
        scores = self.raw_scores(X)
        if self.calibrator is not None:
            scores = np.clip(self.calibrator.predict(scores), 0.0, 1.0)
        return scores

    def predict_expected_value(
        self, X: Any, amounts: np.ndarray, *, use_regressor: bool | None = None
    ) -> np.ndarray:
        """Expected fraud value per transaction, from either model."""
        regressor = self.use_regressor if use_regressor is None else use_regressor
        if regressor:
            if self.regressor is None:
                raise RuntimeError("No regressor trained.")
            return np.clip(np.asarray(self.regressor.predict(X), dtype=float), 0, None)
        return self.predict_proba(X) * np.asarray(amounts, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        """Scores from the selected model.

        Calibrated probabilities for the classifier, expected values for the
        regressor. Prefer :meth:`predict_expected_value` when ranking.
        """
        if self.use_regressor and self.regressor is not None:
            return np.clip(np.asarray(self.regressor.predict(X), dtype=float), 0, None)
        return self.predict_proba(X)

    def select_by_validation(
        self, val_df: pd.DataFrame, X_val: Any, evaluator: BusinessEvaluator
    ) -> dict[str, float]:
        """Keep the model that captures more fraud value on validation.

        ``val_df`` needs transactionTime, transactionAmount and isFraud.
        Returns the captured value for each candidate and sets
        ``use_regressor``. Validation data was also used for early stopping
        and calibration, so these numbers are optimistic; judge the chosen
        model on a later test period.
        """
        amounts = val_df["transactionAmount"].to_numpy()
        captured = {
            "classifier": evaluator.evaluate(
                val_df,
                self.predict_expected_value(X_val, amounts, use_regressor=False),
                is_ev_regressor=True,
            ).captured_value_gbp
        }
        if self.regressor is not None:
            captured["regressor"] = evaluator.evaluate(
                val_df,
                self.predict_expected_value(X_val, amounts, use_regressor=True),
                is_ev_regressor=True,
            ).captured_value_gbp
        self.use_regressor = captured.get("regressor", -1.0) > captured["classifier"]
        return captured
