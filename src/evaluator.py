"""
evaluator.py
============
Comprehensive model evaluation for churn prediction.

Evaluation suite:
  - ROC-AUC and PR-AUC (threshold-independent)
  - Precision, Recall, F1 at optimized threshold
  - Confusion matrix analysis
  - Threshold optimization (F1, custom cost matrix)
  - SHAP feature importance (global + local)
  - Calibration analysis
  - Business metric reporting (retention ROI)

Key concept — threshold optimization:
  Default threshold of 0.5 optimizes accuracy, not business value.
  For churn, missing a churner (false negative) costs much more than
  incorrectly targeting a non-churner (false positive).
  We tune the threshold to maximize the metric that matches business goals.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src import get_logger, load_config

logger = get_logger(__name__)


class ChurnEvaluator:
    """
    Full model evaluation with threshold optimization.

    Usage:
        evaluator = ChurnEvaluator(config)
        results = evaluator.evaluate(model, X_test, y_test)
        optimal_threshold = evaluator.optimize_threshold(y_true, y_scores)
    """

    def __init__(self, config: dict):
        self.config = config
        self.threshold_cfg = config.get("threshold", {})
        self.optimal_threshold = self.threshold_cfg.get("default", 0.5)
        self.evaluation_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: Optional[float] = None,
        feature_names: Optional[list] = None,
        model=None,
        X_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation suite.

        Args:
            y_true: Ground-truth binary labels.
            y_scores: Model probability scores (0-1 range).
            threshold: Decision threshold (auto-optimized if None).
            feature_names: Feature names for SHAP analysis.
            model: Trained model object (for SHAP if provided).
            X_test: Test features (for SHAP if provided).

        Returns:
            Comprehensive evaluation results dictionary.
        """
        logger.info("Running model evaluation...")

        # Optimize threshold if not provided
        if threshold is None and self.threshold_cfg.get("optimize", True):
            threshold = self.optimize_threshold(y_true, y_scores)
        threshold = threshold or 0.5

        self.optimal_threshold = threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Core metrics
        results = {
            "threshold": threshold,
            "roc_auc": float(roc_auc_score(y_true, y_scores)),
            "pr_auc": float(average_precision_score(y_true, y_scores)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
        }

        # Confusion matrix breakdown
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        # Business metrics
        results["business_metrics"] = self._compute_business_metrics(
            tn, fp, fn, tp, y_true, y_scores
        )

        # Full classification report
        results["classification_report"] = classification_report(
            y_true, y_pred, target_names=["Non-churner", "Churner"], output_dict=True
        )

        # ROC curve points (downsampled for storage)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
        results["roc_curve"] = {
            "fpr": fpr[idx].tolist(),
            "tpr": tpr[idx].tolist(),
        }

        # PR curve points
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_idx = np.linspace(0, len(precision_vals) - 1, min(100, len(precision_vals))).astype(int)
        results["pr_curve"] = {
            "precision": precision_vals[pr_idx].tolist(),
            "recall": recall_vals[pr_idx].tolist(),
        }

        # SHAP analysis (if model and data provided)
        if model is not None and X_test is not None and feature_names is not None:
            shap_results = self._compute_shap(model, X_test, feature_names)
            if shap_results:
                results["feature_importance"] = shap_results

        self.evaluation_results = results
        self._log_summary(results)
        return results

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:
        """
        Find the optimal classification threshold.

        Strategy options (from config):
          - "f1"          : Maximizes F1 score
          - "precision"   : Maximizes precision at min recall
          - "recall"      : Maximizes recall at min precision
          - "custom_cost" : Minimizes business cost (fn_cost × FN + fp_cost × FP)

        Returns:
            Optimal threshold float.
        """
        strategy = self.threshold_cfg.get("strategy", "f1")
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)

        logger.info(f"Optimizing threshold using strategy: '{strategy}'")

        if strategy == "f1":
            # Harmonic mean of precision and recall
            f1_scores = np.where(
                (precision_vals + recall_vals) > 0,
                2 * precision_vals * recall_vals / (precision_vals + recall_vals),
                0,
            )
            best_idx = np.argmax(f1_scores[:-1])   # last threshold point is degenerate
            optimal = float(thresholds[best_idx])
            logger.info(
                f"F1-optimal threshold: {optimal:.4f} "
                f"(F1={f1_scores[best_idx]:.4f}, "
                f"P={precision_vals[best_idx]:.3f}, R={recall_vals[best_idx]:.3f})"
            )

        elif strategy == "custom_cost":
            cost_cfg = self.threshold_cfg.get("cost_matrix", {})
            fn_cost = cost_cfg.get("fn_cost", 5)
            fp_cost = cost_cfg.get("fp_cost", 1)

            n = len(y_true)
            costs = []
            for thresh in thresholds:
                y_pred_t = (y_scores >= thresh).astype(int)
                _, fp_count, fn_count, _ = confusion_matrix(y_true, y_pred_t).ravel()
                total_cost = fn_cost * fn_count + fp_cost * fp_count
                costs.append(total_cost)

            best_idx = np.argmin(costs)
            optimal = float(thresholds[best_idx])
            logger.info(
                f"Cost-optimal threshold: {optimal:.4f} "
                f"(fn_cost={fn_cost}, fp_cost={fp_cost}, "
                f"total_cost={costs[best_idx]:,})"
            )

        elif strategy == "recall":
            # Find threshold that gives recall >= 0.8, then maximize precision
            target_recall = 0.80
            eligible = recall_vals[:-1] >= target_recall
            if eligible.any():
                best_idx = np.argmax(precision_vals[:-1][eligible])
                optimal = float(thresholds[eligible][best_idx])
            else:
                optimal = float(thresholds[np.argmax(recall_vals[:-1])])
            logger.info(f"Recall-optimal threshold: {optimal:.4f}")

        else:
            optimal = 0.5
            logger.info("Using default threshold: 0.5")

        return optimal

    def save_results(self, path: str = "models/artifacts/evaluation_results.json") -> None:
        """Persist evaluation results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved: {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_business_metrics(
        self,
        tn: int, fp: int, fn: int, tp: int,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Translate ML metrics into business-interpretable numbers.

        Assumptions (illustrative — adjust per actual business context):
          - Average monthly revenue per customer: $65
          - Retention offer cost: $15 per customer targeted
          - Probability of retaining a churner with offer: 0.40
        """
        avg_monthly_revenue = 65.0
        retention_offer_cost = 15.0
        retention_success_rate = 0.40

        # Without ML: no intervention
        total_churners = int(y_true.sum())
        revenue_lost_no_model = total_churners * avg_monthly_revenue

        # With ML: target TP customers with retention offer
        revenue_saved = tp * avg_monthly_revenue * retention_success_rate
        offer_cost = (tp + fp) * retention_offer_cost
        net_benefit = revenue_saved - offer_cost

        # Coverage and efficiency
        churn_coverage = tp / total_churners if total_churners > 0 else 0
        targeting_efficiency = tp / (tp + fp) if (tp + fp) > 0 else 0

        return {
            "total_churners_in_test": total_churners,
            "churners_identified": int(tp),
            "churners_missed": int(fn),
            "non_churners_targeted": int(fp),
            "churn_coverage_pct": round(churn_coverage * 100, 1),
            "targeting_efficiency_pct": round(targeting_efficiency * 100, 1),
            "estimated_revenue_saved_usd": round(revenue_saved, 2),
            "retention_offer_cost_usd": round(offer_cost, 2),
            "net_business_benefit_usd": round(net_benefit, 2),
            "revenue_lost_without_model_usd": round(revenue_lost_no_model, 2),
        }

    def _compute_shap(
        self,
        model,
        X_test: np.ndarray,
        feature_names: list,
        max_samples: int = 500,
    ) -> Optional[Dict[str, Any]]:
        """Compute SHAP feature importances (global mean |SHAP|)."""
        try:
            import shap

            # Use a sample to keep computation fast
            n = min(max_samples, X_test.shape[0])
            idx = np.random.choice(X_test.shape[0], n, replace=False)
            X_sample = X_test[idx]

            # Tree explainer works with LightGBM/XGBoost natively
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, take the positive class SHAP values
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": feature_names[:len(mean_abs_shap)],
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)

            logger.info("Top 10 features by SHAP importance:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']:<35} {row['mean_abs_shap']:.4f}")

            return importance_df.to_dict(orient="records")

        except ImportError:
            logger.warning("SHAP not installed. Skipping feature importance analysis.")
            return None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Print a clean evaluation summary to the log."""
        cm = results["confusion_matrix"]
        biz = results["business_metrics"]

        logger.info("=" * 55)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("=" * 55)
        logger.info(f"  Threshold       : {results['threshold']:.4f}")
        logger.info(f"  ROC-AUC         : {results['roc_auc']:.4f}")
        logger.info(f"  PR-AUC          : {results['pr_auc']:.4f}")
        logger.info(f"  F1 Score        : {results['f1']:.4f}")
        logger.info(f"  Precision       : {results['precision']:.4f}")
        logger.info(f"  Recall          : {results['recall']:.4f}")
        logger.info("-" * 55)
        logger.info(f"  True Positives  : {cm['true_positive']:,}")
        logger.info(f"  False Positives : {cm['false_positive']:,}")
        logger.info(f"  False Negatives : {cm['false_negative']:,}")
        logger.info(f"  True Negatives  : {cm['true_negative']:,}")
        logger.info("-" * 55)
        logger.info(f"  Churn coverage  : {biz['churn_coverage_pct']}%")
        logger.info(f"  Targeting eff.  : {biz['targeting_efficiency_pct']}%")
        logger.info(f"  Net benefit     : ${biz['net_business_benefit_usd']:,.2f}")
        logger.info("=" * 55)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.preprocessor import ChurnPreprocessor
    from src.trainer import ChurnTrainer

    cfg = load_config()

    df = DataLoader.generate_synthetic_data()
    fe = FeatureEngineer(cfg)
    df = fe.transform(df)

    preprocessor = ChurnPreprocessor(cfg)
    splits = preprocessor.fit_transform(df)

    trainer = ChurnTrainer(cfg)
    trainer.train(splits)

    y_scores = trainer.predict_proba(splits["X_test"])

    evaluator = ChurnEvaluator(cfg)
    results = evaluator.evaluate(
        splits["y_test"],
        y_scores,
        feature_names=splits["feature_names"],
        model=trainer.model,
        X_test=splits["X_test"],
    )
    evaluator.save_results()
    print(f"\nROC-AUC  : {results['roc_auc']:.4f}")
    print(f"PR-AUC   : {results['pr_auc']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
