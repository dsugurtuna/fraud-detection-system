# Fraud Detection System

[![CI](https://github.com/dsugurtuna/fraud-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/dsugurtuna/fraud-detection-system/actions/workflows/ci.yml)

**Value-weighted transaction fraud detection with CatBoost, isotonic calibration, and expected-value ranking.**

Traditional fraud detection models optimise for ROC-AUC or precision–recall, treating all fraud cases equally. In practice, missing a high-value fraud case is far more costly than missing a low-value one. This system implements **value-weighted learning** combined with **expected-value ranking** to align predictions directly with business impact (money saved).

> **Portfolio project.** Uses obfuscated synthetic transaction data. No real financial records are included.

---

## Architecture

```
src/fraud_detection/
    __init__.py          # Public API exports
    features.py          # Temporal, aggregate, risk, and velocity features
    model.py             # CatBoost classifier + regressor, isotonic calibration
    evaluation.py        # Business metrics, EV curve, uplift calculation
tests/
    test_features.py     # Feature engineering tests
    test_evaluation.py   # Business evaluator tests
legacy/
    Fraud_detection.py   # Original monolithic script (1,200+ lines)
data/
    transactions_obf.csv # Obfuscated transaction data
    labels_obf.csv       # Fraud labels
```

---

## Quick start

```bash
pip install -e ".[dev]"
pytest -v
```

### Python API

```python
from fraud_detection import FeatureEngineer, FraudModel, BusinessEvaluator

# Feature engineering with strict fit/transform to prevent leakage
fe = FeatureEngineer(smooth_m=50.0)
train_df, cat_cols, num_cols = fe.create_base_features(train_df)
fe.fit(train_df)
train_df = fe.transform(train_df)

# Value-weighted classifier training
model = FraudModel()
model.train_classifier(X_train, y_train, amounts_train, X_val, y_val, cat_idx)
model.fit_calibrator(val_scores, y_val)

# Business-aligned evaluation
evaluator = BusinessEvaluator(review_capacity=400)
metrics = evaluator.evaluate(test_df, model.predict(X_test))
print(f"Uplift vs random: {metrics.uplift_vs_random_x:.1f}x")
print(f"Fraud value captured: £{metrics.captured_value_gbp:,.2f}")
```

### CLI (legacy script)

```bash
python legacy/Fraud_detection.py --data-dir ./data --capacity 400
```

---

## Key features

| Feature | Detail |
| :--- | :--- |
| **Value-weighted loss** | Higher penalty for missing high-value fraud via log-amount scaling |
| **Temporal validation** | Strict time-based train/val/test splits preventing data leakage |
| **Risk encodings** | m-estimate smoothed fraud rates for high-cardinality categoricals |
| **Velocity features** | Time-since-last transaction per account, merchant, and MCC |
| **Isotonic calibration** | Maps raw scores to calibrated probabilities |
| **EV regressor** | Direct expected-value regression as alternative to classifier |
| **Model selection** | Automatic selection between classifier and regressor on validation |
| **Executive reporting** | PowerPoint deck, EV curve plot, feature importance chart |

## Development

```bash
make dev        # install with dev dependencies
make test       # run pytest
make lint       # run ruff
make clean      # remove build artefacts
```

## Outputs

The legacy CLI generates a `model_outputs/` directory:
- **Fraud_detection.pptx** — executive summary presentation
- **ev_curve.png** — value captured versus review capacity
- **feature_importance.png** — top predictive features
- **metrics.json** — AUC, uplift, savings
- **predictions.csv** — scored test set with monthly rankings

---

*Created by [dsugurtuna](https://github.com/dsugurtuna)*
