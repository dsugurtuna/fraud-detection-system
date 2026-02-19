"""Evaluation module.

Business-aligned metrics: fraud value captured, uplift versus random
baseline, expected-value curve generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Business and ML evaluation metrics."""

    roc_auc: float = -1.0
    pr_auc: float = -1.0
    months_in_test: int = 0
    n_reviews_per_month: int = 0
    captured_value_gbp: float = 0.0
    baseline_random_value_gbp: float = 0.0
    improvement_gbp: float = 0.0
    uplift_vs_random_x: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "months_in_test": self.months_in_test,
            "n_reviews_per_month": self.n_reviews_per_month,
            "captured_value_gbp": self.captured_value_gbp,
            "baseline_random_value_gbp": self.baseline_random_value_gbp,
            "improvement_gbp": self.improvement_gbp,
            "uplift_vs_random_x": self.uplift_vs_random_x,
        }


class BusinessEvaluator:
    """Compute business-aligned fraud detection metrics.

    Uses monthly top-N expected-value ranking to simulate
    operational review capacity constraints.

    Parameters
    ----------
    review_capacity : int
        Maximum transactions reviewed per month.
    """

    def __init__(self, review_capacity: int = 400) -> None:
        self.review_capacity = review_capacity

    @staticmethod
    def _monthly_top_n_value(
        df: pd.DataFrame, ev_col: str, n_per_month: int
    ) -> float:
        """Total fraud value captured by reviewing top-N per month."""
        df = df.copy()
        df["_ym"] = df["transactionTime"].dt.to_period("M")
        captured = 0.0
        for _, grp in df.groupby("_ym"):
            top = grp.sort_values(ev_col, ascending=False).head(n_per_month)
            captured += top.loc[top["isFraud"] == 1, "transactionAmount"].sum()
        return float(captured)

    @staticmethod
    def _random_baseline(df: pd.DataFrame, n_per_month: int) -> float:
        """Expected fraud value captured by random selection."""
        df = df.copy()
        df["_ym"] = df["transactionTime"].dt.to_period("M")
        baseline = 0.0
        for _, grp in df.groupby("_ym"):
            n_total = len(grp)
            if n_total == 0:
                continue
            month_fraud = grp.loc[grp["isFraud"] == 1, "transactionAmount"].sum()
            frac = min(n_per_month, n_total) / max(1.0, n_total)
            baseline += frac * month_fraud
        return float(baseline)

    def evaluate(
        self,
        df_test: pd.DataFrame,
        scores: np.ndarray,
        *,
        is_ev_regressor: bool = False,
    ) -> EvaluationMetrics:
        """Compute business metrics on the test set.

        Parameters
        ----------
        df_test : DataFrame
            Must contain transactionTime, transactionAmount, isFraud.
        scores : array
            Model output — probabilities or expected values.
        is_ev_regressor : bool
            If True, scores are treated as direct expected values.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        tmp = df_test.copy()
        tmp["score"] = scores

        if is_ev_regressor:
            tmp["expected_value"] = tmp["score"]
            roc = -1.0
            prauc = -1.0
        else:
            tmp["expected_value"] = tmp["score"] * tmp["transactionAmount"]
            roc = float(roc_auc_score(tmp["isFraud"], tmp["score"]))
            prauc = float(average_precision_score(tmp["isFraud"], tmp["score"]))

        months = tmp["transactionTime"].dt.to_period("M").nunique()
        captured = self._monthly_top_n_value(tmp, "expected_value", self.review_capacity)
        baseline = self._random_baseline(tmp, self.review_capacity)
        uplift = captured / max(1e-9, baseline)

        return EvaluationMetrics(
            roc_auc=roc,
            pr_auc=prauc,
            months_in_test=int(months),
            n_reviews_per_month=self.review_capacity,
            captured_value_gbp=captured,
            baseline_random_value_gbp=baseline,
            improvement_gbp=captured - baseline,
            uplift_vs_random_x=uplift,
        )

    def ev_curve(
        self,
        df_test: pd.DataFrame,
        scores: np.ndarray,
        k_list: List[int],
        *,
        is_ev_regressor: bool = False,
    ) -> pd.DataFrame:
        """Generate EV curve across review capacity levels."""
        rows = []
        for k in k_list:
            saved = self.review_capacity
            self.review_capacity = k
            m = self.evaluate(df_test, scores, is_ev_regressor=is_ev_regressor)
            self.review_capacity = saved
            rows.append({
                "K": k,
                "captured": m.captured_value_gbp,
                "baseline": m.baseline_random_value_gbp,
                "uplift_x": m.uplift_vs_random_x,
            })
        return pd.DataFrame(rows)
