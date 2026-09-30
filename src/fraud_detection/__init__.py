"""Fraud Detection System: value-weighted transaction fraud detection."""

__version__ = "2.0.0"

from .evaluation import BusinessEvaluator, EvaluationMetrics
from .features import FeatureEngineer
from .model import FraudModel, TrainConfig
from .pipeline import PipelineResult, load_transactions, run_pipeline
from .synthetic import make_transactions

__all__ = [
    "FeatureEngineer",
    "FraudModel",
    "TrainConfig",
    "PipelineResult",
    "load_transactions",
    "run_pipeline",
    "make_transactions",
    "BusinessEvaluator",
    "EvaluationMetrics",
]
