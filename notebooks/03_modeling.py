"""
03_modeling.py  —  Model Training, Evaluation & Explainability
===============================================================
Covers:
  1. Model training with cross-validation
  2. ROC & PR curve analysis
  3. Threshold optimization (F1 vs cost matrix)
  4. Confusion matrix at different thresholds
  5. SHAP feature importance (global + local)
  6. Business metric sensitivity to threshold
  7. Model comparison (if multiple algorithms available)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve

from src import load_config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer
from src.evaluator import ChurnEvaluator

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
os.makedirs("plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL TRAINING, EVALUATION & EXPLAINABILITY")
print("=" * 60)

cfg = load_config("../config/config.yaml")

df = DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)
fe = FeatureEngineer(cfg)
df = fe.transform(df)

preprocessor = ChurnPreprocessor(cfg)
splits = preprocessor.fit_transform(df)

trainer = ChurnTrainer(cfg)
meta = trainer.train(splits)

y_test = splits["y_test"]
y_scores = trainer.predict_proba(splits["X_test"])
feature_names = splits.get("feature_names", [])

print(f"\nAlgorithm   : {meta['algorithm']}")
print(f"CV ROC-AUC  : {meta['cv_roc_auc_mean']:.4f} ± {meta['cv_roc_auc_std']:.4f}")
print(f"Test samples: {len(y_test):,}")

# ══════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE & PR CURVE
# ══════════════════════════════════════════════════════════════════════════
print("\n── 1. ROC & PR Curves ──────────────────────────────────")

fpr, tpr, roc_thresh = roc_curve(y_test, y_scores)
precision_vals, recall_vals, pr_thresh = precision_recall_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
pr_auc = average_precision_score(y_test, y_scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance Curves", fontsize=14, fontweight="bold")

# ROC Curve
axes[0].plot(fpr, tpr, color="#4C9BE8", linewidth=2.5, label=f"ROC-AUC = {roc_auc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
axes[0].fill_between(fpr, tpr, alpha=0.08, color="#4C9BE8")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.02])

# PR Curve
baseline = y_test.mean()
axes[1].plot(recall_vals, precision_vals, color="#E85D5D", linewidth=2.5, label=f"PR-AUC = {pr_auc:.4f}")
axes[1].axhline(baseline, color="#888", linestyle="--", linewidth=1, label=f"Baseline = {baseline:.2f}")
axes[1].fill_between(recall_vals, precision_vals, alpha=0.08, color="#E85D5D")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right")
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("plots/model_01_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"ROC-AUC = {roc_auc:.4f} | PR-AUC = {pr_auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 2. THRESHOLD OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════
print("\n── 2. Threshold Optimization ───────────────────────────")

thresholds = np.linspace(0.01, 0.99, 200)
f1_scores, precisions, recalls = [], [], []

for thresh in thresholds:
    y_pred_t = (y_scores >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))

f1_scores = np.array(f1_scores)
precisions = np.array(precisions)
recalls = np.array(recalls)

best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Threshold Optimization Analysis", fontsize=14, fontweight="bold")

# Metric vs threshold
axes[0].plot(thresholds, f1_scores, label="F1 Score", color="#4C9BE8", linewidth=2)
axes[0].plot(thresholds, precisions, label="Precision", color="#E85D5D", linewidth=2)
axes[0].plot(thresholds, recalls, label="Recall", color="#55AA55", linewidth=2)
axes[0].axvline(best_threshold, color="orange", linestyle="--", linewidth=1.5,
                label=f"Optimal F1 threshold = {best_threshold:.3f}")
axes[0].axvline(0.5, color="#888", linestyle=":", linewidth=1, label="Default threshold = 0.5")
axes[0].set_xlabel("Classification Threshold")
axes[0].set_ylabel("Metric Value")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].legend(loc="center right", fontsize=9)
axes[0].set_xlim([0, 1])

# Business cost vs threshold
fn_cost, fp_cost = 5, 1   # Asymmetric costs: missing a churner costs 5×
business_costs = []
for thresh in thresholds:
    y_pred_t = (y_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    cost = fn * fn_cost + fp * fp_cost
    business_costs.append(cost)

business_costs = np.array(business_costs)
best_cost_idx = np.argmin(business_costs)
best_cost_threshold = thresholds[best_cost_idx]

axes[1].plot(thresholds, business_costs, color="#9B59B6", linewidth=2, label="Total cost")
axes[1].axvline(best_cost_threshold, color="orange", linestyle="--", linewidth=1.5,
                label=f"Min cost threshold = {best_cost_threshold:.3f}")
axes[1].axvline(0.5, color="#888", linestyle=":", linewidth=1, label="Default = 0.5")
axes[1].set_xlabel("Classification Threshold")
axes[1].set_ylabel(f"Business Cost (FN×{fn_cost} + FP×{fp_cost})")
axes[1].set_title(f"Business Cost vs Threshold\n(FN cost={fn_cost}x, FP cost={fp_cost}x)")
axes[1].legend()
axes[1].set_xlim([0, 1])

plt.tight_layout()
plt.savefig("plots/model_02_threshold_optimization.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Default threshold (0.50)  → F1={f1_score(y_test, (y_scores>=0.5).astype(int)):.4f}")
print(f"F1-optimal threshold      → thresh={best_threshold:.3f}, F1={f1_scores[best_f1_idx]:.4f}")
print(f"Cost-optimal threshold    → thresh={best_cost_threshold:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRIX COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n── 3. Confusion Matrix at Different Thresholds ─────────")

test_thresholds = [0.5, best_threshold, best_cost_threshold]
thresh_labels = ["Default (0.50)", f"F1-Optimal ({best_threshold:.2f})", f"Cost-Optimal ({best_cost_threshold:.2f})"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrix Comparison Across Thresholds", fontsize=13, fontweight="bold")

for ax, thresh, lbl in zip(axes, test_thresholds, thresh_labels):
    y_pred_t = (y_scores >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    f1_t = f1_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        ax=ax, linewidths=0.5, linecolor="gray",
        xticklabels=["Pred: No Churn", "Pred: Churn"],
        yticklabels=["True: No Churn", "True: Churn"],
        annot_kws={"size": 13},
    )
    ax.set_title(f"{lbl}\nF1={f1_t:.3f} | P={prec_t:.3f} | R={rec_t:.3f}", fontsize=10)

plt.tight_layout()
plt.savefig("plots/model_03_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION CURVE
# ══════════════════════════════════════════════════════════════════════════
print("\n── 4. Calibration Curve ──────────────────────────────────")

fig, ax = plt.subplots(figsize=(8, 6))

prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10)

ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
ax.plot(prob_pred, prob_true, marker="o", color="#4C9BE8", linewidth=2,
        label="Model calibration", markersize=7)
ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="#E85D5D", label="Calibration error")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (True Rate)")
ax.set_title("Calibration Curve\n(Closer to diagonal = better calibrated)")
ax.legend()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("plots/model_04_calibration.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 5. SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
print("\n── 5. Prediction Score Distribution ──────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Churn Score Distributions", fontsize=13, fontweight="bold")

# KDE by true class
for label, color, name in [(0, "#4C9BE8", "No Churn"), (1, "#E85D5D", "Churn")]:
    mask = y_test == label
    pd.Series(y_scores[mask]).plot.kde(ax=axes[0], label=name, color=color, linewidth=2)
axes[0].axvline(best_threshold, color="orange", linestyle="--",
                label=f"Threshold={best_threshold:.3f}")
axes[0].set_title("Score KDE by True Class\n(Good separation = good model)")
axes[0].set_xlabel("Predicted Churn Probability")
axes[0].legend()
axes[0].set_xlim([0, 1])

# Histogram stacked
bins = np.linspace(0, 1, 25)
for label, color, name in [(0, "#4C9BE8", "No Churn"), (1, "#E85D5D", "Churn")]:
    mask = y_test == label
    axes[1].hist(y_scores[mask], bins=bins, alpha=0.6, color=color, label=name, density=True)
axes[1].axvline(best_threshold, color="orange", linestyle="--", label=f"Threshold={best_threshold:.3f}")
axes[1].set_title("Score Histogram (Density) by True Class")
axes[1].set_xlabel("Predicted Churn Probability")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/model_05_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 6. BUSINESS SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n── 6. Business Metric Sensitivity ────────────────────────")

avg_revenue = 65.0
retention_cost = 15.0
retention_rate = 0.40

revenues, costs, net_benefits = [], [], []
for thresh in thresholds:
    y_pred_t = (y_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    rev = tp * avg_revenue * retention_rate
    cost = (tp + fp) * retention_cost
    revenues.append(rev)
    costs.append(cost)
    net_benefits.append(rev - cost)

net_benefits = np.array(net_benefits)
best_roi_idx = np.argmax(net_benefits)
best_roi_threshold = thresholds[best_roi_idx]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(thresholds, net_benefits, color="#27AE60", linewidth=2.5, label="Net Benefit ($)")
ax.plot(thresholds, revenues, color="#4C9BE8", linewidth=1.5, linestyle="--", label="Revenue Saved ($)")
ax.plot(thresholds, costs, color="#E85D5D", linewidth=1.5, linestyle="--", label="Retention Cost ($)")
ax.axvline(best_roi_threshold, color="orange", linestyle="-", linewidth=2,
           label=f"Max ROI threshold = {best_roi_threshold:.3f}")
ax.axhline(0, color="#888", linestyle=":", linewidth=1)
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Amount ($) — Test Set")
ax.set_title(f"Business Value vs Threshold\n(avg revenue=${avg_revenue}, retention cost=${retention_cost}, success rate={retention_rate:.0%})")
ax.legend()

plt.tight_layout()
plt.savefig("plots/model_06_business_value.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nROI-optimal threshold : {best_roi_threshold:.3f}")
print(f"Max net benefit       : ${net_benefits[best_roi_idx]:,.2f} (test set)")

# ══════════════════════════════════════════════════════════════════════════
# 7. FINAL EVALUATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════
evaluator = ChurnEvaluator(cfg)
results = evaluator.evaluate(y_test, y_scores, feature_names=feature_names,
                             model=trainer.model, X_test=splits["X_test"])

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)
print(f"Algorithm        : {meta['algorithm']}")
print(f"CV ROC-AUC       : {meta['cv_roc_auc_mean']:.4f} ± {meta['cv_roc_auc_std']:.4f}")
print(f"Test ROC-AUC     : {results['roc_auc']:.4f}")
print(f"Test PR-AUC      : {results['pr_auc']:.4f}")
print(f"Optimal threshold: {results['threshold']:.4f}")
print(f"F1 Score         : {results['f1']:.4f}")
print(f"Precision        : {results['precision']:.4f}")
print(f"Recall           : {results['recall']:.4f}")

biz = results["business_metrics"]
print(f"\nBusiness Impact  :")
print(f"  Churn coverage : {biz['churn_coverage_pct']}%")
print(f"  Targeting eff. : {biz['targeting_efficiency_pct']}%")
print(f"  Net benefit    : ${biz['net_business_benefit_usd']:,.2f}")
print("\nAll plots saved to: plots/")
print("=" * 60)
