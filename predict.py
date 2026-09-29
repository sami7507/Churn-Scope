"""
predict.py — CLI batch inference for ChurnScope.

Loads a CSV, scores all rows, outputs sorted predictions CSV.

Usage:
  python predict.py --input data/raw/customers.csv
  python predict.py --input data/raw/customers.csv --output results/scores.csv
  python predict.py --input data/raw/customers.csv --threshold 0.45
"""
from __future__ import annotations
import argparse, os, sys, time
import pandas as pd
from src import load_config, get_logger
from src.predictor import ChurnPredictor
logger = get_logger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="ChurnScope — Batch Inference CLI")
    p.add_argument("--input",     required=True)
    p.add_argument("--output",    default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--config",    default="config/config.yaml")
    return p.parse_args()

def run_batch_inference(args=None):
    if args is None: args = parse_args()
    config = load_config(args.config)
    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}"); sys.exit(1)
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} customer records")
    predictor = ChurnPredictor(config)
    predictor.load()
    if args.threshold is not None:
        predictor.optimal_threshold = args.threshold
        logger.info(f"Custom threshold: {args.threshold}")
    start   = time.time()
    results = predictor.predict_batch(df)
    elapsed = time.time()-start
    logger.info(f"Predicted {len(results):,} customers in {elapsed:.2f}s ({len(results)/elapsed:.0f}/s)")
    df["churn_probability"] = [r["churn_probability"] for r in results]
    df["churn_prediction"]  = [r["churn_prediction"]  for r in results]
    df["churn_label"]       = [r["churn_label"]        for r in results]
    df["risk_tier"]         = [r["risk_tier"]          for r in results]
    df["confidence"]        = [r["confidence"]         for r in results]
    df = df.sort_values("churn_probability",ascending=False)
    if args.output is None:
        from datetime import datetime
        os.makedirs("results",exist_ok=True)
        args.output = f"results/churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    df.to_csv(args.output,index=False)
    churn_n = df["churn_prediction"].sum()
    logger.info(f"Results → {args.output}")
    logger.info(f"Churners: {churn_n:,} ({churn_n/len(df):.1%})")
    logger.info("Risk tier distribution:")
    for tier,cnt in df["risk_tier"].value_counts().items():
        logger.info(f"  {tier:<10}: {cnt:>5,} ({cnt/len(df)*100:.1f}%)")
    return df

if __name__ == "__main__":
    run_batch_inference()
