"""
train_pipeline.py
=================
End-to-end training pipeline orchestrator.

Run this single script to reproduce the entire pipeline from scratch:
  1. Load raw data (or generate synthetic data for demo)
  2. Feature engineering
  3. Preprocessing (split, encode, scale)
  4. Model training with cross-validation
  5. Threshold optimization
  6. Comprehensive evaluation
  7. Save all artifacts (model, pipeline, evaluation results)

Usage:
  python train_pipeline.py                        # Use data from config path
  python train_pipeline.py --synthetic            # Generate and use synthetic data
  python train_pipeline.py --algorithm xgboost    # Use XGBoost instead of LightGBM
  python train_pipeline.py --optimize-hpo         # Enable Optuna HPO (slow)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from src import load_config, get_logger
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer
from src.evaluator import ChurnEvaluator

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction — Training Pipeline"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and use synthetic data (for demo/testing)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["lightgbm", "xgboost"],
        default=None,
        help="Override model algorithm from config",
    )
    parser.add_argument(
        "--optimize-hpo",
        action="store_true",
        help="Enable Optuna hyperparameter optimization (slower)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=7043,
        help="Number of synthetic samples to generate (default: 7043)",
    )
    return parser.parse_args()


def run_pipeline(args=None):
    """
    Execute the full training pipeline.

    Returns:
        Dict with evaluation results and artifact paths.
    """
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Step 0: Configuration
    # ----------------------------------------------------------------
    if args is None:
        args = parse_args()

    config_path = getattr(args, "config", "config/config.yaml")
    config = load_config(config_path)

    # Override algorithm if specified
    if getattr(args, "algorithm", None):
        config["model"]["algorithm"] = args.algorithm
        logger.info(f"Algorithm override: {args.algorithm}")

    # Enable HPO if requested
    if getattr(args, "optimize_hpo", False):
        config["optuna"]["enabled"] = True
        logger.info("Optuna HPO enabled")

    # ----------------------------------------------------------------
    # Step 1: Data Loading
    # ----------------------------------------------------------------
    logger.info("\n[1/5] DATA LOADING")
    loader = DataLoader(config)

    if getattr(args, "synthetic", False):
        n = getattr(args, "n_samples", 7043)
        logger.info(f"Generating synthetic dataset ({n:,} samples)...")
        df_raw = DataLoader.generate_synthetic_data(n_samples=n)
        os.makedirs("data/raw", exist_ok=True)
        df_raw.to_csv("data/raw/telco_churn.csv", index=False)
        logger.info("Synthetic data saved to data/raw/telco_churn.csv")
    else:
        df_raw = loader.load()

    # ----------------------------------------------------------------
    # Step 2: Feature Engineering
    # ----------------------------------------------------------------
    logger.info("\n[2/5] FEATURE ENGINEERING")
    fe = FeatureEngineer(config)
    df_enriched = fe.transform(df_raw)
    logger.info(f"Dataset shape after feature engineering: {df_enriched.shape}")

    # Save enriched features for analysis
    os.makedirs("data/features", exist_ok=True)
    df_enriched.to_csv("data/features/engineered_features.csv", index=False)
    logger.info("Engineered features saved to data/features/engineered_features.csv")

    # ----------------------------------------------------------------
    # Step 3: Preprocessing
    # ----------------------------------------------------------------
    logger.info("\n[3/5] PREPROCESSING")
    preprocessor = ChurnPreprocessor(config)
    splits = preprocessor.fit_transform(df_enriched)
    pipeline_path = preprocessor.save()
    logger.info(f"Preprocessing pipeline saved: {pipeline_path}")

    # ----------------------------------------------------------------
    # Step 4: Model Training
    # ----------------------------------------------------------------
    logger.info("\n[4/5] MODEL TRAINING")
    trainer = ChurnTrainer(config)
    training_meta = trainer.train(splits)
    model_path = trainer.save()

    logger.info(
        f"Model saved: {model_path}\n"
        f"CV ROC-AUC: {training_meta['cv_roc_auc_mean']:.4f} ± "
        f"{training_meta['cv_roc_auc_std']:.4f}"
    )

    # ----------------------------------------------------------------
    # Step 5: Evaluation
    # ----------------------------------------------------------------
    logger.info("\n[5/5] EVALUATION")
    evaluator = ChurnEvaluator(config)

    y_scores = trainer.predict_proba(splits["X_test"])
    eval_results = evaluator.evaluate(
        splits["y_test"],
        y_scores,
        feature_names=splits.get("feature_names"),
        model=trainer.model,
        X_test=splits["X_test"],
    )
    evaluator.save_results()

    # ----------------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------------
    total_time = time.time() - pipeline_start

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time       : {total_time:.1f}s")
    logger.info(f"Algorithm        : {config['model']['algorithm'].upper()}")
    logger.info(f"CV ROC-AUC       : {training_meta['cv_roc_auc_mean']:.4f}")
    logger.info(f"Test ROC-AUC     : {eval_results['roc_auc']:.4f}")
    logger.info(f"Test PR-AUC      : {eval_results['pr_auc']:.4f}")
    logger.info(f"Optimal threshold: {eval_results['threshold']:.4f}")
    logger.info(f"Test F1          : {eval_results['f1']:.4f}")
    logger.info(f"Test Precision   : {eval_results['precision']:.4f}")
    logger.info(f"Test Recall      : {eval_results['recall']:.4f}")
    logger.info(f"Model path       : {model_path}")
    logger.info("-" * 60)

    biz = eval_results["business_metrics"]
    logger.info("Business Impact (test set estimate):")
    logger.info(f"  Churn coverage   : {biz['churn_coverage_pct']}%")
    logger.info(f"  Targeting eff.   : {biz['targeting_efficiency_pct']}%")
    logger.info(f"  Net benefit      : ${biz['net_business_benefit_usd']:,.2f}")
    logger.info("=" * 60)

    logger.info("\nTo start the API:")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")

    return {
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "eval_results": eval_results,
        "training_metadata": training_meta,
    }


if __name__ == "__main__":
    try:
        result = run_pipeline()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
