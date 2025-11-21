# Fraud Detection Model for Transaction Monitoring

This repository contains a production-ready fraud detection system designed to maximize fraud value captured within operational capacity constraints.

## Overview

Traditional fraud detection models often optimize for ROC-AUC or Precision-Recall, which treats all fraud cases equally. In reality, missing a high-value fraud case is far more costly than missing a low-value one.

This system implements a **Value-Weighted Learning** approach combined with **Expected Value Ranking** to align the model's predictions with business impact (money saved).

## Key Features

- **Value-Weighted Learning**: The model is trained with a custom loss function that penalizes missing high-value fraud more than low-value fraud.
- **Temporal Validation**: Strict time-based splitting (Train/Validation/Test) ensures no data leakage and realistic performance estimation.
- **Risk-Based Feature Engineering**: Includes smoothed risk encodings for high-cardinality categorical features (Merchant, MCC, Country).
- **Isotonic Calibration**: Calibrates raw model scores to true probabilities for accurate expected value calculations.
- **Automated Reporting**: Generates a PowerPoint presentation and visualization plots summarizing business impact.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-model.git
   cd fraud-detection-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The model uses the included dataset:
- `transactions_obf.csv`: Transaction details (amount, time, merchant info, etc.)
- `labels_obf.csv`: Fraud labels (eventId, isFraud)

## Usage

### Basic Run
Run the model with default settings (assumes CSVs are in the current directory):
```bash
python Fraud_detection.py
```

### Custom Configuration
You can customize the data path, date splits, and review capacity:

```bash
python Fraud_detection.py \
  --data-dir ./data \
  --val-start 2017-09-01 \
  --test-start 2017-11-01 \
  --capacity 500
```

### Arguments
- `--data-dir`: Directory containing input CSVs (default: current dir)
- `--output-dir`: Directory for results (default: `model_outputs`)
- `--capacity`: Monthly review capacity for business metrics (default: 400)
- `--calibrate` / `--no-calibrate`: Enable/Disable probability calibration (default: Enabled)

## Outputs

The script generates a `model_outputs/` directory containing:
- **Fraud_detection.pptx**: Executive summary presentation.
- **ev_curve.png**: Plot showing value captured vs. review capacity.
- **feature_importance.png**: Visualization of top predictive features.
- **metrics.json**: Detailed performance metrics (AUC, Uplift, Savings).
- **predictions.csv**: Model scores and rankings for the test set.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
