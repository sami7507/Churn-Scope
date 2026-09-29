"""
train_pipeline.py — End-to-end training orchestrator for ChurnScope.

Runs the full pipeline in one command:
  1. Load or generate data
  2. Feature engineering (22 domain features)
  3. Preprocessing (split, impute, encode, scale)
  4. Model training with 5-fold CV
  5. Threshold optimisation
  6. Comprehensive evaluation (ROC, PR, SHAP, business ROI)
  7. Save all artifacts (model, pipeline, evaluation JSON)

Usage:
  python train_pipeline.py                        # use real data
  python train_pipeline.py --synthetic            # generate synthetic data
  python train_pipeline.py --algorithm xgboost    # use XGBoost
  python train_pipeline.py --optimize-hpo         # enable Optuna HPO
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime
from src import load_config, get_logger
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer
from src.evaluator import ChurnEvaluator
logger = get_logger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="ChurnScope — Training Pipeline")
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--algorithm",    choices=["lightgbm","xgboost"], default=None)
    p.add_argument("--optimize-hpo", action="store_true")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--n-samples",    type=int, default=7043)
    return p.parse_args()

def run_pipeline(args=None):
    start = time.time()
    logger.info("="*60)
    logger.info("CHURNSCOPE — TRAINING PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    if args is None: args = parse_args()
    config = load_config(getattr(args,"config","config/config.yaml"))
    if getattr(args,"algorithm",None): config["model"]["algorithm"] = args.algorithm
    if getattr(args,"optimize_hpo",False): config["optuna"]["enabled"] = True

    logger.info("\n[1/5] DATA LOADING")
    loader = DataLoader(config)
    if getattr(args,"synthetic",False):
        n = getattr(args,"n_samples",7043)
        logger.info(f"Generating synthetic data ({n:,} rows)...")
        df_raw = DataLoader.generate_synthetic_data(n_samples=n)
        os.makedirs("data/raw",exist_ok=True)
        df_raw.to_csv("data/raw/telco_churn.csv",index=False)
    else:
        df_raw = loader.load()

    logger.info("\n[2/5] FEATURE ENGINEERING")
    df_enriched = FeatureEngineer(config).transform(df_raw)
    os.makedirs("data/features",exist_ok=True)
    df_enriched.to_csv("data/features/engineered_features.csv",index=False)

    logger.info("\n[3/5] PREPROCESSING")
    preprocessor = ChurnPreprocessor(config)
    splits       = preprocessor.fit_transform(df_enriched)
    pipeline_path = preprocessor.save()

    logger.info("\n[4/5] MODEL TRAINING")
    trainer  = ChurnTrainer(config)
    meta     = trainer.train(splits)
    model_path = trainer.save()

    logger.info("\n[5/5] EVALUATION")
    evaluator  = ChurnEvaluator(config)
    y_scores   = trainer.predict_proba(splits["X_test"])
    eval_results = evaluator.evaluate(splits["y_test"],y_scores,
                                      feature_names=splits.get("feature_names"),
                                      model=trainer.model,X_test=splits["X_test"])
    evaluator.save_results()

    elapsed = time.time()-start
    logger.info("\n"+"="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time        : {elapsed:.1f}s")
    logger.info(f"Algorithm         : {config['model']['algorithm'].upper()}")
    logger.info(f"CV ROC-AUC        : {meta['cv_roc_auc_mean']:.4f}")
    logger.info(f"Test ROC-AUC      : {eval_results['roc_auc']:.4f}")
    logger.info(f"Optimal threshold : {eval_results['threshold']:.4f}")
    logger.info(f"Test F1           : {eval_results['f1']:.4f}")
    logger.info(f"Model path        : {model_path}")
    biz = eval_results["business_metrics"]
    logger.info(f"Churn coverage    : {biz['churn_coverage_pct']}%")
    logger.info(f"Net benefit       : ${biz['net_business_benefit_usd']:,.2f}")
    logger.info("\nTo start the API:")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("To launch the dashboard:")
    logger.info("  streamlit run streamlit_app.py")
    return {"model_path":model_path,"pipeline_path":pipeline_path,"eval_results":eval_results,"training_metadata":meta}

if __name__ == "__main__":
    try:
        run_pipeline(); sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Interrupted."); sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True); sys.exit(1)
