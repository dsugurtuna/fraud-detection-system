#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Detection Model for Transaction Monitoring
================================================

This script implements a production-ready fraud detection system designed to maximize 
fraud value captured within operational capacity constraints.

Technical Approach:
- Temporal validation splits to prevent data leakage
- Value-weighted learning to prioritize high-value fraud
- Risk-based feature engineering with proper smoothing
- Isotonic calibration for improved probability estimates
- Expected value ranking for business-aligned predictions

Author: Ugur
Date: 2025

Usage:
    # Basic usage (CSVs in current directory):
    python fraud_detection_model.py
    
    # Specify data directory:
    python fraud_detection_model.py --data-dir /path/to/data
    
    # Custom validation/test dates and review capacity:
    python fraud_detection_model.py --val-start 2017-09-01 --test-start 2017-11-01 --capacity 500
    
    # Disable calibration:
    python fraud_detection_model.py --no-calibrate
    
    The script expects two CSV files:
    - transactions_obf.csv
    - labels_obf.csv
"""

import os
import sys
import json
import argparse
import warnings
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Dependency management
def _ensure_deps():
    """Ensure required packages are installed."""
    required_packages = ["catboost", "python-pptx", "matplotlib", "scikit-learn", "pandas", "numpy"]
   
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing_packages)

_ensure_deps()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_PARAGRAPH_ALIGNMENT
from pptx.enum.shapes import MSO_SHAPE


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fraud Detection Model for Transaction Monitoring',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get("DATA_DIR", "."),
        help='Path to directory containing transactions_obf.csv and labels_obf.csv'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_outputs',
        help='Directory to save outputs (presentation, plots, predictions)'
    )
    
    parser.add_argument(
        '--val-start',
        type=str,
        default='2017-09-01T00:00:00Z',
        help='Validation set start date (ISO format)'
    )
    
    parser.add_argument(
        '--test-start',
        type=str,
        default='2017-11-01T00:00:00Z',
        help='Test set start date (ISO format)'
    )
    
    parser.add_argument(
        '--capacity',
        type=int,
        default=400,
        help='Monthly review capacity (number of transactions)'
    )
    
    parser.add_argument(
        '--calibrate',
        dest='calibrate',
        action='store_true',
        help='Enable isotonic calibration (default)'
    )
    
    parser.add_argument(
        '--no-calibrate',
        dest='calibrate',
        action='store_false',
        help='Disable isotonic calibration'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.set_defaults(calibrate=True)
    
    return parser.parse_args()


# Model hyperparameters (not exposed as CLI args for simplicity)
AMOUNT_WEIGHT_EXP = 1.0
SMOOTH_M = 50.0
DPI_EXPORT = 300
TOPN_FEATURES = 12
EV_K_LIST = [100, 200, 400, 600, 800]


def detect_gpu():
    """Detect available GPU for training."""
    try:
        import subprocess
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if output:
            return True, output.split('\n')[0]
    except Exception:
        pass
    return False, None


def gbp_fmt(x, pos=None):
    """Format currency values with thousand separators."""
    return f"£{x:,.0f}"


def load_data(data_zip: Optional[str], data_dir: Optional[str]) -> pd.DataFrame:
    """
    Load transaction and fraud label data.
    
    Args:
        data_zip: Path to zipped data archive (optional)
        data_dir: Path to data directory containing CSVs
    
    Returns:
        Merged DataFrame with transactions and fraud labels
    """
    if data_zip:
        import zipfile
        with zipfile.ZipFile(data_zip, 'r') as z:
            try:
                with z.open('data-new/transactions_obf.csv') as f:
                    df_trans = pd.read_csv(f)
                with z.open('data-new/labels_obf.csv') as f:
                    df_labels = pd.read_csv(f)
            except KeyError:
                 with z.open('transactions_obf.csv') as f:
                    df_trans = pd.read_csv(f)
                 with z.open('labels_obf.csv') as f:
                    df_labels = pd.read_csv(f)
    else:
        t_path = Path(data_dir) / 'transactions_obf.csv'
        l_path = Path(data_dir) / 'labels_obf.csv'
        if not t_path.exists() or not l_path.exists():
            abs_path = Path(data_dir).resolve()
            raise FileNotFoundError(f"Could not find CSVs in {abs_path}. "
                                    f"Expected transactions_obf.csv and labels_obf.csv.")
        df_trans = pd.read_csv(t_path)
        df_labels = pd.read_csv(l_path)

    df_labels = df_labels.copy()
    df_labels['isFraud'] = 1
    df = pd.merge(df_trans, df_labels[['eventId', 'isFraud']], on='eventId', how='left')
    df['isFraud'] = df['isFraud'].fillna(0).astype(int)
    df['transactionTime'] = pd.to_datetime(df['transactionTime'])
    df = df.sort_values('transactionTime').reset_index(drop=True)
 
    return df


def create_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Create temporal and categorical features.
    Note: Numeric imputation is performed later using training data only
    to prevent data leakage.
    """
    df = df.copy()
    
    # Temporal features
    df['hour'] = df['transactionTime'].dt.hour
    df['dayOfWeek'] = df['transactionTime'].dt.dayofweek
    df['isWeekend'] = (df['dayOfWeek'] >= 5).astype(int)
    df['isNight'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['logAmount'] = np.log1p(df['transactionAmount'])

    cat_features = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry',
                    'merchantZip', 'posEntryMode']
    df['merchantCountry'] = df['merchantCountry'].astype(str)

    num_features = []
    for c in ['transactionAmount', 'availableCash']:
        if c in df.columns:
            num_features.append(c)

    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].fillna('UNK')

    return df, cat_features, num_features


def fit_aggregates(train_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute historical aggregates and smoothed risk encodings from training data.
    Uses m-estimate smoothing to balance empirical fraud rates with global priors,
    preventing overfitting on rare categories.
    """
    stores: Dict[str, Dict] = {}

    # Account-level statistics
    acc_stats = train_df.groupby('accountNumber').agg(
        acc_mean=('transactionAmount', 'mean'),
        acc_std=('transactionAmount', 'std'),
        acc_max=('transactionAmount', 'max'),
        acc_count=('transactionAmount', 'count'),
        acc_merchants=('merchantId', 'nunique'),
        acc_mccs=('mcc', 'nunique'),
        acc_age_days=('transactionTime', lambda s: (s.max() - s.min()).days + 1),
    ).fillna(0)
    stores['accountNumber'] = acc_stats.to_dict(orient='index')

 
    # Merchant statistics
    mch_stats = train_df.groupby('merchantId').agg(
        mean_amount=('transactionAmount', 'mean'),
        std_amount=('transactionAmount', 'std'),
        tx_count=('transactionAmount', 'count'),
    ).fillna(0)
    stores['merchantId'] = mch_stats.to_dict(orient='index')

    # MCC statistics
    mcc_stats = train_df.groupby('mcc').agg(
        mean_amount=('transactionAmount', 'mean'),
        std_amount=('transactionAmount', 'std'),
        tx_count=('transactionAmount', 'count'),
    ).fillna(0)
    stores['mcc'] = mcc_stats.to_dict(orient='index')

   
    # Smoothed risk encodings
    global_fraud_rate = max(1e-6, train_df['isFraud'].mean())
    stores['global_fraud_rate'] = global_fraud_rate

    def smoothed_rate(df: pd.DataFrame, col: str, m: float, global_rate: float) -> Dict:
        """Apply m-estimate smoothing to fraud rates."""
        grp = df.groupby(col)['isFraud'].agg(['sum', 'count']).rename(
            columns={'sum': 'fraud', 'count': 'n'}
        )
        grp['rate'] = (grp['fraud'] + m * global_rate) / (grp['n'] + m)
    
        return grp['rate'].to_dict()

    for col in ['merchantId', 'mcc', 'merchantCountry', 'posEntryMode']:
        stores[col + '_risk'] = smoothed_rate(train_df, col, m=SMOOTH_M, global_rate=global_fraud_rate)

    stores['numeric_medians'] = train_df.median(numeric_only=True).to_dict()
    return stores


def transform_aggregates(df: pd.DataFrame, stores: Dict[str, Dict]) -> pd.DataFrame:
    """Apply pre-computed aggregates and risk encodings to dataset."""
    df = df.copy()

    acc_map = stores.get('accountNumber', {})
    for col in ['acc_mean', 'acc_std', 'acc_max', 'acc_count', 'acc_merchants', 'acc_mccs', 'acc_age_days']:
        df[col] = df['accountNumber'].map({k: v.get(col, 0) for k, v in acc_map.items()}).fillna(0)

    mch_map = stores.get('merchantId', {})
    for col in ['mean_amount', 'std_amount', 'tx_count']:
        df[f'mch_{col}'] = df['merchantId'].map({k: v.get(col, 0) for k, v in mch_map.items()}).fillna(0)

    mcc_map = stores.get('mcc', {})
    for col in ['mean_amount', 'std_amount', 'tx_count']:
        df[f'mcc_{col}'] = df['mcc'].map({k: v.get(col, 0) for k, v in mcc_map.items()}).fillna(0)

    global_fraud_rate = stores.get('global_fraud_rate', 0.01)
    for col in ['merchantId', 'mcc', 'merchantCountry', 'posEntryMode']:
        risk_map = stores.get(col + '_risk', {})
  
        df[f'{col}_risk'] = df[col].map(risk_map).fillna(global_fraud_rate)
    
    return df


def transform_velocity(df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity features using only historical information.
    This ensures no data leakage by computing time-since-last features
    using only transactions that occurred before the current window.
    """
    hist_cols = ['accountNumber', 'merchantId', 'mcc', 'transactionTime']
    hist_subset = historical_df[hist_cols].copy()
    df_subset = df[hist_cols].copy()
    df_index = df.index

    combined = pd.concat([hist_subset, df_subset], axis=0, ignore_index=True)
    combined = combined.sort_values(['accountNumber', 'transactionTime'])
    combined['time_since_last_acc'] = combined.groupby('accountNumber')['transactionTime'].diff().dt.total_seconds()

    combined = combined.sort_values(['accountNumber', 'merchantId', 'transactionTime'])
    combined['time_since_last_mch'] = combined.groupby(['accountNumber', 'merchantId'])['transactionTime'].diff().dt.total_seconds()

    combined = combined.sort_values(['accountNumber', 'mcc', 'transactionTime'])
    combined['time_since_last_mcc'] = combined.groupby(['accountNumber', 'mcc'])['transactionTime'].diff().dt.total_seconds()

    vel = combined.loc[df_index, ['time_since_last_acc', 'time_since_last_mch', 'time_since_last_mcc']]
    return pd.concat([df.drop(columns=['transactionTime']), vel], axis=1)


def monthly_top_n_value(df_scored: pd.DataFrame, score_col: str, n_per_month: int) -> float:
    """Calculate total fraud value captured using monthly top-N ranking."""
    tmp = df_scored.copy()
    if 'expected_value' not in tmp.columns:
         tmp['expected_value'] = tmp[score_col] * tmp['transactionAmount']
         
    tmp['year_month'] = tmp['transactionTime'].dt.to_period('M')
    captured_value = 0.0
    for ym, group in tmp.groupby('year_month'):
        g = group.sort_values('expected_value', ascending=False).head(n_per_month)
        captured_value += g.loc[g['isFraud'] == 1, 'transactionAmount'].sum()
    return float(captured_value)


def business_metrics(df_test: pd.DataFrame, scores: np.ndarray, review_cap: int, 
                    is_ev_regressor: bool = False) -> Dict[str, float]:
    """
    Compute model performance metrics aligned with business objectives.
    Returns both traditional ML metrics (AUC) and business metrics
    (fraud value captured, uplift vs random baseline).
    """
    df_scored = df_test.copy()
    df_scored['score'] = scores
    
    if is_ev_regressor:
        df_scored['expected_value'] = df_scored['score']
        roc = -1.0
        prauc = -1.0
    else:
        df_scored['expected_value'] = df_scored['score'] * df_scored['transactionAmount']
        roc = roc_auc_score(df_scored['isFraud'], df_scored['score'])
        prauc = average_precision_score(df_scored['isFraud'], df_scored['score'])

    months_dt = df_scored['transactionTime'].dt.to_period('M')
    months = months_dt.nunique()
    
    captured = monthly_top_n_value(df_scored, 'expected_value', review_cap)

    # Calculate random baseline
    df_scored['year_month'] = months_dt
    baseline = 0.0
    for ym, group in df_scored.groupby('year_month'):
        n_total = len(group)
        if n_total == 0:
            continue
        month_fraud_value = group.loc[group['isFraud'] == 1, 'transactionAmount'].sum()
        frac_selected = min(review_cap, n_total) / max(1.0, n_total)
        baseline += float(frac_selected * month_fraud_value)
    df_scored = df_scored.drop(columns=['year_month'])

    uplift = captured / max(1e-9, baseline)
    improvement = captured - baseline

    return {
        'roc_auc': float(roc),
        'pr_auc': float(prauc),
        'months_in_test': int(months),
        'n_reviews_per_month': int(review_cap),
        'captured_value_gbp': float(captured),
        'baseline_random_value_gbp': float(baseline),
        'improvement_gbp': float(improvement),
        'uplift_vs_random_x': float(uplift),
    }


def ev_curve(df_test: pd.DataFrame, scores: np.ndarray, k_list: List[int], 
            is_ev_regressor: bool = False) -> pd.DataFrame:
    """Generate expected value curve across different review capacity levels."""
    results = []
    for k in k_list:
        metrics = business_metrics(df_test, scores, review_cap=k, is_ev_regressor=is_ev_regressor)
        results.append({
            'K': k,
            'captured': metrics['captured_value_gbp'],
            'baseline': metrics['baseline_random_value_gbp'],
            'uplift_x': metrics['uplift_vs_random_x']
        })
    return pd.DataFrame(results)


def train_value_weighted_classifier(X_tr, y_tr, amounts_tr, X_val, y_val, cat_idx: List[int], random_state: int):
    """
    Train CatBoost classifier with value-weighted loss.
    Applies higher weights to high-value fraud cases, aligning the
    learning objective with the business goal of maximizing fraud value captured.
    """
    fraud_rate = max(1e-6, np.mean(y_tr))
    base_pos_weight = (1.0 - fraud_rate) / fraud_rate

    sample_weights = np.ones_like(y_tr, dtype=float)
    pos_mask = (y_tr == 1)
    sample_weights[pos_mask] = base_pos_weight * (1.0 + np.log1p(amounts_tr[pos_mask])) ** AMOUNT_WEIGHT_EXP

    params = dict(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        od_type='Iter',
        od_wait=50,
        task_type='CPU'
    )
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
              cat_features=cat_idx, sample_weight=sample_weights, verbose=False)
    return model


def fit_isotonic(val_scores: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression for probability calibration."""
    order = np.argsort(val_scores)
    x = val_scores[order]
    y = y_val[order]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(x, y)
    return iso


def train_ev_regressor(X_tr, y_ev_tr, X_val, y_ev_val, cat_idx: List[int], random_state: int):
    """Train CatBoost regressor to directly predict expected fraud value."""
    params = dict(
        loss_function='RMSE',
        eval_metric='RMSE',
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        od_type='Iter',
        od_wait=50,
        task_type='CPU'
    )
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_ev_tr, eval_set=(X_val, y_ev_val), cat_features=cat_idx, verbose=False)
    return model


def plot_ev_curve_pretty(df_curve: pd.DataFrame, months_in_test: int, out_path: Path):
    """Create professional expected value curve visualization."""
    plt.style.use('seaborn-v0_8-darkgrid')
   
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    color_model = '#1f77b4'
    color_baseline = '#ff7f0e'
    
    ax.plot(df_curve['K'], df_curve['captured'], 
            marker='o', markersize=8, linewidth=3, 
            color=color_model, label='ML Model', 
            markeredgewidth=1.5, markeredgecolor='white', zorder=3)
    
    ax.plot(df_curve['K'], df_curve['baseline'], 
            marker='s', markersize=7, linewidth=2.5, 
            linestyle='--', color=color_baseline, 
            label='Random Baseline',
            markeredgewidth=1.5, markeredgecolor='white', zorder=3)
    
    ax.fill_between(df_curve['K'], df_curve['baseline'], df_curve['captured'], 
                    alpha=0.15, color=color_model, zorder=1)
    
    for x, y in zip(df_curve['K'], df_curve['captured']):
        ax.annotate(f'£{y:,.0f}', xy=(x, y), xytext=(0, 12), 
                   textcoords='offset points', ha='center', fontsize=10, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color_model, 
                         edgecolor='none', alpha=0.8),
                   color='white', zorder=4)
    
    ax.yaxis.set_major_formatter(FuncFormatter(gbp_fmt))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    ax.set_xlabel('Monthly Review Capacity', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Fraud Value Captured', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_title('Expected Value Captured vs Review Capacity', 
                 fontsize=16, fontweight='bold', pad=20, color='#1f3a5f')
    
    legend = ax.legend(loc='lower right', frameon=True, shadow=True, 
                      fontsize=11, edgecolor='#cccccc', fancybox=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#d0d0d0')
        spine.set_linewidth(1.5)
    
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50', length=6, width=1.5)
    
    plt.tight_layout()
   
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_feature_importance_pretty(model, feature_names: List[str], out_path: Path, top_n: int = 12):
    """Create professional feature importance visualization."""
    try:
        importances = model.get_feature_importance(type='PredictionValuesChange')
    except Exception:
        importances = model.get_feature_importance()
    
    imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp = imp.sort_values('importance', ascending=False).head(top_n)
    
    def clean_feature_name(name):
        """Format feature names for readability."""
    
        name = name.replace('_', ' ').title()
        name = name.replace('Mch', 'Merchant').replace('Acc', 'Account')
        name = name.replace('Tx', 'Transaction').replace('Mcc', 'MCC')
        return name
    
    imp['feature_clean'] = imp['feature'].apply(clean_feature_name)
    imp = imp.iloc[::-1]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(imp)))
    bars = ax.barh(range(len(imp)), 
                   imp['importance'], 
                   color=colors, edgecolor='#2c3e50', linewidth=1.5, height=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, imp['importance'])):
        width = bar.get_width()
        ax.text(width + max(imp['importance']) * 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', ha='left', fontsize=10, 
               fontweight='bold', color='#2c3e50')
   
    
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['feature_clean'], fontsize=11, color='#2c3e50')
    ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold', 
                 color='#2c3e50', labelpad=10)
    ax.set_title('Top Model Features Driving Predictions', 
                fontsize=16, fontweight='bold', pad=20, color='#1f3a5f')
    
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('#d0d0d0')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_edgecolor('#d0d0d0')
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50', length=6, width=1.5)
    ax.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return imp


def build_pptx(output_dir: Path, metrics: Dict[str, float], ev_curve_path: Path,
               feat_imp_path: Path) -> Path:
    """Generate executive presentation with model results."""
    
    CLR_NAVY = RGBColor(31, 56, 100)
    CLR_BLUE = RGBColor(41, 98, 255)
    CLR_GREEN = RGBColor(16, 185, 129)
    CLR_ORANGE = RGBColor(251, 146, 60)
    CLR_GRAY_DARK = RGBColor(55, 65, 81)
    CLR_GRAY_MED = RGBColor(107, 114, 128)
    CLR_GRAY_LIGHT = RGBColor(229, 231, 235)
    CLR_BG = RGBColor(249, 250, 251)
    CLR_WHITE = RGBColor(255, 255, 255)
    
    MARGIN = Inches(0.5)
    CONTENT_WIDTH = Inches(12.333)
    
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1: Overview
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    
    header = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.5))
    header.fill.solid()
    header.fill.fore_color.rgb = CLR_NAVY
    header.line.fill.background()
    
    title = slide1.shapes.add_textbox(MARGIN, Inches(0.3), CONTENT_WIDTH, Inches(0.6))
    p = title.text_frame.paragraphs[0]
    p.text = "ADAPTIVE FRAUD DETECTION SYSTEM"
    p.font.name = 'Arial'
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = CLR_WHITE
   
    
    subtitle = slide1.shapes.add_textbox(MARGIN, Inches(0.95), CONTENT_WIDTH, Inches(0.35))
    p = subtitle.text_frame.paragraphs[0]
    p.text = "Machine Learning for Value-Optimized Financial Crime Prevention"
    p.font.name = 'Arial'
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(203, 213, 225)
    
    metric_x = Inches(9.5)
    metric_w = Inches(3.333)
    metric_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, metric_x, Inches(1.8), metric_w, Inches(1.1))
    metric_box.fill.solid()
    metric_box.fill.fore_color.rgb = CLR_BLUE
    metric_box.line.fill.background()
    
    metric_tb = slide1.shapes.add_textbox(metric_x + Inches(0.15), Inches(1.9), 
                                          metric_w - Inches(0.3), Inches(0.9))
    tf = metric_tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p = tf.paragraphs[0]
    p.text = f"{metrics['uplift_vs_random_x']:.1f}×"
    p.font.name = 'Arial'
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = CLR_WHITE
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
    p.space_after = Pt(0)
    
    p = tf.add_paragraph()
    p.text = "Better than Random"
    p.font.name = 'Arial'
    p.font.size = Pt(13)
    p.font.color.rgb = CLR_WHITE
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
    
    prob_y = Inches(1.8)
    prob_w = Inches(9.2)
    prob_h = Inches(1.5)
    
    prob_box = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, prob_y, prob_w, prob_h)
    prob_box.fill.solid()
    prob_box.fill.fore_color.rgb = CLR_WHITE
    prob_box.line.color.rgb = CLR_GRAY_LIGHT
    prob_box.line.width = Pt(1)
    
    prob_tb = slide1.shapes.add_textbox(MARGIN + Inches(0.3), prob_y + Inches(0.2), 
                                        prob_w - Inches(0.6), prob_h - Inches(0.4))
    tf = prob_tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "THE CHALLENGE"
    p.font.name = 'Arial'
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_NAVY
    p.space_after = Pt(8)
    
    p = tf.add_paragraph()
    p.text = "Limited analyst capacity to review transactions means high-value fraud is missed, costing millions. Traditional systems flag cases at random, wasting resources and revenue."
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.color.rgb = CLR_GRAY_DARK
    p.line_spacing = 1.3
    
    sol_y = Inches(3.5)
    sol_h = Inches(3.0)
    
    sol_bg = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, sol_y, CONTENT_WIDTH, sol_h)
    sol_bg.fill.solid()
    sol_bg.fill.fore_color.rgb = CLR_BG
    sol_bg.line.fill.background()
    
    sol_title = slide1.shapes.add_textbox(MARGIN + Inches(0.3), sol_y + Inches(0.25), 
                                          CONTENT_WIDTH - Inches(0.6), Inches(0.4))
    p = sol_title.text_frame.paragraphs[0]
    p.text = "THE SOLUTION"
    p.font.name = 'Arial'
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_NAVY
    
    col_w = Inches(3.85)
    col_spacing = Inches(0.15)
    col_y = sol_y + Inches(0.8)
    col_h = Inches(2.0)
    
    solutions = [
        ("Temporal Integrity", 
         "Time-aware validation with strict splits prevents data leakage, ensuring production-ready performance"),
        ("Value-Weighted Learning", 
         "Training prioritizes high-value fraud through amount-weighted loss, aligning ML with business impact"),
        ("Expected Value Ranking", 
         "Rank by Probability × Amount to maximize fraud capture within capacity constraints")
    ]
    
    for i, (title, desc) in enumerate(solutions):
        col_x = MARGIN + i * (col_w + col_spacing)
        
        col_box = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, col_x, col_y, col_w, col_h)
        col_box.fill.solid()
        col_box.fill.fore_color.rgb = CLR_WHITE
        col_box.line.color.rgb = CLR_GRAY_LIGHT
        col_box.line.width = Pt(1)
        
        badge_size = Inches(0.35)
        badge = slide1.shapes.add_shape(MSO_SHAPE.OVAL, col_x + Inches(0.25), col_y + Inches(0.25), 
                                        badge_size, badge_size)
        badge.fill.solid()
        badge.fill.fore_color.rgb = CLR_BLUE
        badge.line.fill.background()
        
        badge_tb = slide1.shapes.add_textbox(col_x + Inches(0.25), col_y + Inches(0.25), 
                                             badge_size, badge_size)
        p = badge_tb.text_frame.paragraphs[0]
        p.text = str(i+1)
        p.font.name = 'Arial'
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = CLR_WHITE
        p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
        badge_tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        title_tb = slide1.shapes.add_textbox(col_x + Inches(0.25), col_y + Inches(0.7), 
                                              col_w - Inches(0.5), Inches(0.35))
        p = title_tb.text_frame.paragraphs[0]
        p.text = title
        p.font.name = 'Arial'
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = CLR_GRAY_DARK
        
       
        desc_tb = slide1.shapes.add_textbox(col_x + Inches(0.25), col_y + Inches(1.1), 
                                           col_w - Inches(0.5), Inches(0.75))
        tf = desc_tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.name = 'Arial'
        p.font.size = Pt(12)
        p.font.color.rgb = CLR_GRAY_MED
        p.line_spacing = 1.25
    
    footer = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(6.8), Inches(13.333), Inches(0.7))
    footer.fill.solid()
    footer.fill.fore_color.rgb = CLR_NAVY
    footer.line.fill.background()
    
    footer_tb = slide1.shapes.add_textbox(MARGIN, Inches(6.95), CONTENT_WIDTH, Inches(0.4))
    p = footer_tb.text_frame.paragraphs[0]
    p.text = f"CatBoost + Isotonic Calibration  •  {metrics['months_in_test']}-month Test Period  •  {metrics['n_reviews_per_month']} Reviews/Month Capacity"
    p.font.name = 'Arial'
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER

    # Slide 2: Results
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    
    header2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.15))
    header2.fill.solid()
    header2.fill.fore_color.rgb = CLR_NAVY
    header2.line.fill.background()
    
    title2 = slide2.shapes.add_textbox(MARGIN, Inches(0.25), CONTENT_WIDTH, Inches(0.5))
    p = title2.text_frame.paragraphs[0]
    p.text = "BUSINESS IMPACT & MODEL PERFORMANCE"
    p.font.name = 'Arial'
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = CLR_WHITE
    
    annual_savings = metrics['captured_value_gbp'] / metrics['months_in_test'] * 12
    subtitle2 = slide2.shapes.add_textbox(MARGIN, Inches(0.75), CONTENT_WIDTH, Inches(0.3))
    p = subtitle2.text_frame.paragraphs[0]
    p.text = f"Projected Annual Value Protection: £{annual_savings:,.0f}"
    p.font.name = 'Arial'
    p.font.size = Pt(16)
    p.font.color.rgb = CLR_ORANGE
    
    card_y = Inches(1.4)
    card_h = Inches(1.0)
    
    card_w = Inches(4.0)
    card_spacing = Inches(0.15)
    
    cards = [
        ("Model Captured", f"£{metrics['captured_value_gbp']:,.0f}", CLR_GREEN),
        ("Random Baseline", f"£{metrics['baseline_random_value_gbp']:,.0f}", CLR_GRAY_MED),
        ("Incremental Value", f"+£{metrics['improvement_gbp']:,.0f}", CLR_ORANGE)
    ]
    
    for i, (label, value, color) in enumerate(cards):
        card_x = MARGIN + i * (card_w + card_spacing)
        
       
        card = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, card_x, card_y, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CLR_WHITE
        card.line.color.rgb = color
        card.line.width = Pt(2.5)
        
        label_tb = slide2.shapes.add_textbox(card_x + Inches(0.2), card_y + Inches(0.15), 
                                              card_w - Inches(0.4), Inches(0.25))
        p = label_tb.text_frame.paragraphs[0]
        p.text = label.upper()
        p.font.name = 'Arial'
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CLR_GRAY_MED
        p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
        
     
        value_tb = slide2.shapes.add_textbox(card_x + Inches(0.2), card_y + Inches(0.45), 
                                             card_w - Inches(0.4), Inches(0.4))
        p = value_tb.text_frame.paragraphs[0]
        p.text = value
        p.font.name = 'Arial'
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = color
        p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
    
    perf_y = Inches(2.55)
    perf_h = Inches(0.5)
    
    perf_bar = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, perf_y, CONTENT_WIDTH, perf_h)
    perf_bar.fill.solid()
    perf_bar.fill.fore_color.rgb = CLR_BG
    perf_bar.line.fill.background()
    
    perf_tb = slide2.shapes.add_textbox(MARGIN + Inches(0.3), perf_y + Inches(0.1), 
                                        CONTENT_WIDTH - Inches(0.6), Inches(0.3))
    p = perf_tb.text_frame.paragraphs[0]
    
    if metrics['roc_auc'] != -1.0:
        perf_text = f"Model Performance: ROC-AUC {metrics['roc_auc']:.3f}  •  PR-AUC {metrics['pr_auc']:.3f}  •  {metrics['uplift_vs_random_x']:.1f}× Uplift"
    else:
        perf_text = f"Model Performance: Expected Value Regressor  •  {metrics['uplift_vs_random_x']:.1f}× Uplift"
    
 
    p.text = perf_text
    p.font.name = 'Arial'
    p.font.size = Pt(13)
    p.font.color.rgb = CLR_GRAY_DARK
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
    
    chart_y = Inches(3.2)
    chart_w = Inches(6.05)
    chart_h = Inches(3.8)
    chart_spacing = Inches(0.2)
    
    chart1_x = MARGIN
    chart1_box = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, chart1_x, chart_y, chart_w, chart_h)
    chart1_box.fill.solid()
    chart1_box.fill.fore_color.rgb = CLR_WHITE
    chart1_box.line.color.rgb = CLR_GRAY_LIGHT
    chart1_box.line.width = Pt(1)
 
    
    chart1_title = slide2.shapes.add_textbox(chart1_x + Inches(0.2), chart_y + Inches(0.15), 
                                             chart_w - Inches(0.4), Inches(0.3))
    p = chart1_title.text_frame.paragraphs[0]
    p.text = "Value Captured vs Review Capacity"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CLR_NAVY
    
    slide2.shapes.add_picture(str(ev_curve_path), chart1_x + Inches(0.1), 
                             chart_y + Inches(0.5), height=Inches(3.2))
    
    chart2_x = MARGIN + chart_w + chart_spacing
    chart2_box = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, chart2_x, chart_y, chart_w, chart_h)
    chart2_box.fill.solid()
    chart2_box.fill.fore_color.rgb = CLR_WHITE
    chart2_box.line.color.rgb = CLR_GRAY_LIGHT
    chart2_box.line.width = Pt(1)
    
   
    chart2_title = slide2.shapes.add_textbox(chart2_x + Inches(0.2), chart_y + Inches(0.15), 
                                             chart_w - Inches(0.4), Inches(0.3))
    p = chart2_title.text_frame.paragraphs[0]
    p.text = "Key Model Drivers (Feature Importance)"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CLR_NAVY
   
  
    slide2.shapes.add_picture(str(feat_imp_path), chart2_x + Inches(0.1), 
                             chart_y + Inches(0.5), height=Inches(3.2))
    
    footer2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.15), Inches(13.333), Inches(0.35))
    footer2.fill.solid()
    footer2.fill.fore_color.rgb = CLR_NAVY
    footer2.line.fill.background()
    
    footer_tb2 = slide2.shapes.add_textbox(MARGIN, Inches(7.23), CONTENT_WIDTH, Inches(0.2))
    p = footer_tb2.text_frame.paragraphs[0]
    p.text = "End-to-End ML Pipeline: Feature Engineering  →  Model Training  →  Calibration  →  Production Deployment"
    p.font.name = 'Arial'
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
    
    out_path = output_dir / "Fraud_detection.pptx"
    prs.save(str(out_path))
    return out_path


def main():
    """Main execution pipeline."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_state)
    
    # Detect GPU
    HAS_GPU, GPU_NAME = detect_gpu()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 76)
    print(" FRAUD DETECTION MODEL - TRAINING PIPELINE")
    print("=" * 76)
    print("Approach: Value-weighted learning with temporal validation")
    print("Model: CatBoost with isotonic calibration")
    if HAS_GPU:
        print(f"Compute: CPU (GPU available: {GPU_NAME})")
    else:
        print("Compute: CPU")
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Validation start: {args.val_start}")
    print(f"  Test start: {args.test_start}")
    print(f"  Review capacity: {args.capacity}/month")
    print(f"  Calibration: {'Enabled' if args.calibrate else 'Disabled'}")
    print(f"  Random state: {args.random_state}")
    print()

    # Data loading
    print("=" * 76)
    print(" DATA LOADING")
    print("=" * 76)
    df = load_data(None, args.data_dir)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")
    print(f"Date range: {df['transactionTime'].min().date()} to {df['transactionTime'].max().date()}")

    # Feature engineering
    df, cat_cols, num_cols = create_base_features(df)
    print("\nBase features created")

    # Temporal splits
    val_start = pd.to_datetime(args.val_start)
    test_start = pd.to_datetime(args.test_start)
    train_df = df[df['transactionTime'] < val_start].copy()
    val_df   = df[(df['transactionTime'] >= val_start) & (df['transactionTime'] < test_start)].copy()
    test_df  = df[df['transactionTime'] >= test_start].copy()

    print(f"\nTrain set: {len(train_df):>7,} rows ({train_df['isFraud'].sum():>3} frauds)")
    print(f"Val set:   {len(val_df):>7,} rows ({val_df['isFraud'].sum():>3} frauds)")
    print(f"Test set:  {len(test_df):>7,} rows ({test_df['isFraud'].sum():>3} frauds)")

 
    # Train-only imputation
    med = train_df[num_cols].median(numeric_only=True)
    for df_ in (train_df, val_df, test_df):
        for c in num_cols:
            if c in df_.columns:
                df_[c] = df_[c].fillna(med[c])

    # Fit aggregates on training data
    print("\nFitting aggregates and risk encodings...")
    stores = fit_aggregates(train_df)
    train_df = transform_aggregates(train_df, stores)
    val_df   = transform_aggregates(val_df, stores)
    test_df  = transform_aggregates(test_df, stores)

    # Velocity features
    print("Computing velocity features...")
    train_df_v = transform_velocity(train_df, historical_df=train_df.iloc[0:0])
    val_df_v   = transform_velocity(val_df,   historical_df=train_df)
    test_df_v  = transform_velocity(test_df,  historical_df=pd.concat([train_df, val_df], axis=0))

    # Prepare feature matrices
    target = 'isFraud'
    drop_cols = ['eventId']
    feature_cols = [c for c in train_df_v.columns if c not in drop_cols + [target]]
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    X_tr, y_tr = train_df_v[feature_cols], train_df[target].values
    X_val, y_val = val_df_v[feature_cols], val_df[target].values
    X_te, y_te = test_df_v[feature_cols], test_df[target].values

    amounts_tr = train_df['transactionAmount'].values
    amounts_val = val_df['transactionAmount'].values
    amounts_te  = test_df['transactionAmount'].values

    print(f"Feature count: {len(feature_cols)} ({len(cat_idx)} categorical)")

    # Model training
    print("\n" + "=" * 76)
    print(" MODEL TRAINING")
    print("=" * 76)
    print(f"Class imbalance: {np.mean(y_tr)*100:.3f}% fraud rate")
    clf = train_value_weighted_classifier(X_tr, y_tr, amounts_tr, X_val, y_val, cat_idx, args.random_state)

  
    # Calibration
    val_scores_clf = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_scores_clf)
    print(f"Validation AUC: {val_auc:.4f}")

    iso = None
    if args.calibrate:
        print("Applying isotonic calibration...")
        iso = fit_isotonic(val_scores_clf, y_val)
        val_scores_clf_cal = np.clip(iso.predict(val_scores_clf), 0.0, 1.0)
    else:
        val_scores_clf_cal = val_scores_clf

    # Alternative: Direct EV regression
    print("\nEvaluating expected value regressor...")
    
    y_ev_tr  = (train_df['isFraud'] * train_df['transactionAmount']).values
    y_ev_val = (val_df['isFraud']   * val_df['transactionAmount']).values
    evr = train_ev_regressor(X_tr, y_ev_tr, X_val, y_ev_val, cat_idx, args.random_state)
    val_scores_evr = evr.predict(X_val)
    val_scores_evr = np.clip(val_scores_evr, 0.0, None)

    # Model selection
    val_df_for_eval = val_df.copy()
    ev_val_clf = business_metrics(val_df_for_eval, val_scores_clf_cal, review_cap=args.capacity, is_ev_regressor=False)
    ev_val_evr = business_metrics(val_df_for_eval, val_scores_evr, review_cap=args.capacity, is_ev_regressor=True)

    use_evr = ev_val_evr['captured_value_gbp'] > ev_val_clf['captured_value_gbp']
    final_model = evr if use_evr else clf
    final_kind  = "EVRegressor" if use_evr else "Classifier+Isotonic"
  
    print(f"Selected model: {final_kind}")

    # Test evaluation
    print("\nScoring test set...")
    if use_evr:
        scores_te = final_model.predict(X_te)
        scores_te = np.clip(scores_te, 0.0, None)
    else:
        scores_te = clf.predict_proba(X_te)[:, 1]
        if iso is not None:
            scores_te = np.clip(iso.predict(scores_te), 0.0, 1.0)

    print("\n" + "=" * 76)
    print(" EVALUATION RESULTS")
    print("=" * 76)
    test_eval_df = test_df.copy()
    metrics = business_metrics(test_eval_df, scores_te, review_cap=args.capacity, is_ev_regressor=use_evr)

    total_fraud = test_eval_df.loc[test_eval_df['isFraud'] == 1, 'transactionAmount'].sum()
    model_prevented = metrics['captured_value_gbp']
    baseline = metrics['baseline_random_value_gbp']
    uplift = metrics['uplift_vs_random_x']

    # Predictions export
    out_pred = test_df[['eventId', 'transactionTime', 'transactionAmount', 'isFraud']].copy()
    out_pred['score'] = scores_te
    out_pred['year_month'] = out_pred['transactionTime'].dt.to_period('M')
    if use_evr:
        out_pred['expected_value'] = out_pred['score']
    else:
        out_pred['expected_value'] = out_pred['score'] * out_pred['transactionAmount']
    out_pred['rank_monthly'] = out_pred.groupby('year_month')['expected_value'].rank(method='first', ascending=False)
    out_pred['selected_for_review'] = (out_pred['rank_monthly'] <= args.capacity).astype(int)

    selected = out_pred[out_pred['selected_for_review'] == 1]
    frauds_caught = (selected['isFraud'] == 1).sum()
    total_frauds = (test_eval_df['isFraud'] == 1).sum()
    recall = frauds_caught / total_frauds if total_frauds > 0 else 0.0
    precision = frauds_caught / len(selected) if len(selected) > 0 else 0.0

    # Results summary
    print(f"\nModel type: {final_kind}")
    if not use_evr:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"\nBusiness Impact (@ {args.capacity} reviews/month):")
    print(f"Total fraud value in test:    £{total_fraud:>10,.2f}")
    print(f"Random baseline captures:     £{baseline:>10,.2f}")
    print(f"Model captures (EV-ranked):   £{model_prevented:>10,.2f}")
    print(f"Incremental improvement:      £{metrics['improvement_gbp']:>10,.2f}")
    print(f"\nUplift vs baseline: {uplift:.1f}x")
    print(f"\nOperational Metrics:")
    print(f"Recall:    {recall*100:>5.1f}% ({frauds_caught}/{total_frauds} frauds)")
    print(f"Precision: {precision*100:>5.1f}%")
    print(f"\nProjected annual savings: £{model_prevented/metrics['months_in_test']*12:,.2f}")

    # Generate visualizations
    curve_df = ev_curve(test_eval_df, scores_te, EV_K_LIST, is_ev_regressor=use_evr)
    ev_curve_path = Path(args.output_dir) / "ev_curve.png"
    plot_ev_curve_pretty(curve_df, metrics['months_in_test'], ev_curve_path)

    feat_imp_path = Path(args.output_dir) / "feature_importance.png"
    try:
        imp_df = plot_feature_importance_pretty(final_model, feature_cols, feat_imp_path, top_n=TOPN_FEATURES)
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot. {e}")
        imp_df = pd.DataFrame(columns=['feature', 'importance'])

    # Save outputs
    pred_path = Path(args.output_dir) / "predictions.csv"
    out_pred.drop(columns=['year_month']).to_csv(pred_path, index=False)

    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            **metrics,
            'model_kind': final_kind,
            'recall_at_capacity': float(recall),
            'precision_at_capacity': float(precision)
        }, f, indent=2)

    if not imp_df.empty:
        imp_df.to_csv(Path(args.output_dir) / "feature_importance.csv", index=False)

    slides_path = build_pptx(Path(args.output_dir), metrics, ev_curve_path, feat_imp_path)

    print("\n" + "=" * 76)
    print(" PIPELINE COMPLETE")
    print("=" * 76)
    print(f"Final uplift vs baseline: {uplift:.1f}x")
    print(f"Projected annual savings: £{model_prevented/metrics['months_in_test']*12:,.2f}")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  - EV curve:           {ev_curve_path.name}")
    print(f"  - Feature importance: {feat_imp_path.name}")
    print(f"  - Predictions:       {pred_path.name}")
    print(f"  - Metrics:           {metrics_path.name}")
    print(f"  - Presentation:      {slides_path.name}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()