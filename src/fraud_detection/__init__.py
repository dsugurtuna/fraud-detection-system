"""Fraud Detection System — value-weighted transaction fraud detection."""

__version__ = "2.0.0"

from .features import FeatureEngineer
from .model import FraudModel
from .evaluation import BusinessEvaluator, EvaluationMetrics

__all__ = [
    "FeatureEngineer",
    "FraudModel",
    "BusinessEvaluator",
    "EvaluationMetrics",
]
