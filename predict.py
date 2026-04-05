"""
predict.py
==========
CLI batch inference script.

Loads a CSV of customer records, runs churn prediction on all rows,
and outputs results to a CSV file with churn scores and risk tiers.

Usage:
  python predict.py --input data/raw/new_customers.csv
  python predict.py --input data/raw/customers.csv --output results/predictions.csv
  python predict.py --input data/raw/customers.csv --threshold 0.45
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from src import load_config, get_logger
from src.predictor import ChurnPredictor

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction — Batch Inference CLI"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output CSV (default: results/churn_predictions_<timestamp>.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override prediction threshold (default: use model's optimal threshold)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML",
    )
    return parser.parse_args()


def run_batch_inference(args=None):
    if args is None:
        args = parse_args()

    config = load_config(args.config)

    # Load input data
    logger.info(f"Loading input data: {args.input}")
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} customer records")

    # Load predictor
    predictor = ChurnPredictor(config)
    predictor.load()

    # Override threshold if provided
    if args.threshold is not None:
        predictor.optimal_threshold = args.threshold
        logger.info(f"Using custom threshold: {args.threshold}")

    # Run batch prediction
    start = time.time()
    results = predictor.predict_batch(df)
    elapsed = time.time() - start

    logger.info(
        f"Predicted {len(results):,} customers in {elapsed:.2f}s "
        f"({len(results)/elapsed:.0f} predictions/sec)"
    )

    # Build output DataFrame
    result_df = df.copy()
    result_df["churn_probability"] = [r["churn_probability"] for r in results]
    result_df["churn_prediction"] = [r["churn_prediction"] for r in results]
    result_df["churn_label"] = [r["churn_label"] for r in results]
    result_df["risk_tier"] = [r["risk_tier"] for r in results]
    result_df["confidence"] = [r["confidence"] for r in results]

    # Sort by churn probability descending (highest risk first)
    result_df = result_df.sort_values("churn_probability", ascending=False)

    # Save output
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/churn_predictions_{ts}.csv"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # Summary
    churn_count = result_df["churn_prediction"].sum()
    logger.info(f"\nResults saved to: {args.output}")
    logger.info(f"Total customers  : {len(result_df):,}")
    logger.info(f"Predicted churners: {churn_count:,} ({churn_count/len(result_df):.1%})")
    logger.info("\nRisk tier distribution:")
    for tier, count in result_df["risk_tier"].value_counts().items():
        pct = count / len(result_df) * 100
        logger.info(f"  {tier:<10}: {count:>5,} ({pct:.1f}%)")

    logger.info(f"\nTop 5 highest-risk customers:")
    top5_cols = ["customerID", "churn_probability", "risk_tier", "Contract", "tenure", "MonthlyCharges"]
    available = [c for c in top5_cols if c in result_df.columns]
    print(result_df[available].head().to_string(index=False))

    return result_df


if __name__ == "__main__":
    run_batch_inference()
