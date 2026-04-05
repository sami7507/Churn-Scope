"""
data_loader.py
==============
Handles all data ingestion for the churn prediction pipeline.

Responsibilities:
  - Load raw data from CSV, Parquet, or SQL sources
  - Validate schema (required columns, expected dtypes)
  - Log data quality metrics (shape, null counts, target distribution)
  - Provide a clean DataFrame to downstream stages

Design principle:
  All I/O lives here. The rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src import get_logger, load_config

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and validates raw churn data from multiple source types.

    Usage:
        loader = DataLoader(config)
        df = loader.load()
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.paths = config["paths"]
        self.target_col = self.data_cfg["target_column"]
        self.id_col = self.data_cfg["customer_id_column"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from the configured source.

        Args:
            source: Override path/connection string. Uses config default if None.

        Returns:
            Validated raw DataFrame.
        """
        source = source or self.paths["raw_data"]
        logger.info(f"Loading data from: {source}")

        df = self._read_source(source)
        df = self._coerce_types(df)
        self._validate_schema(df)
        self._log_data_quality(df)

        logger.info(f"Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_source(self, source: str) -> pd.DataFrame:
        """Dispatch to the correct reader based on file extension."""
        path = Path(source)
        ext = path.suffix.lower()

        if ext == ".csv":
            return pd.read_csv(source)
        elif ext in (".parquet", ".pq"):
            return pd.read_parquet(source)
        elif ext == ".json":
            return pd.read_json(source)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(source)
        else:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                "Supported: .csv, .parquet, .json, .xlsx"
            )

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply known type coercions for the Telco Churn dataset.
        Extend this method for other datasets.
        """
        df = df.copy()

        # TotalCharges arrives as object in Telco dataset — coerce to float
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            n_coerced = df["TotalCharges"].isna().sum()
            if n_coerced > 0:
                logger.warning(
                    f"TotalCharges: {n_coerced} values could not be parsed → set to NaN"
                )

        # SeniorCitizen is 0/1 integer — treat as categorical string for encoding
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        # Target: "Yes"/"No" → 1/0
        if self.target_col in df.columns:
            if df[self.target_col].dtype == object:
                df[self.target_col] = df[self.target_col].map({"Yes": 1, "No": 0})
                logger.info("Target column mapped: Yes→1, No→0")

        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Check that all required columns are present."""
        required_cols = (
            self.data_cfg.get("numeric_features", [])
            + self.data_cfg.get("categorical_features", [])
        )
        if self.target_col in df.columns:
            required_cols.append(self.target_col)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Schema validation passed")

    def _log_data_quality(self, df: pd.DataFrame) -> None:
        """Log a concise data quality summary."""
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        logger.info(f"Dataset shape: {df.shape}")

        if not cols_with_nulls.empty:
            for col, cnt in cols_with_nulls.items():
                pct = cnt / len(df) * 100
                logger.warning(f"  Nulls in '{col}': {cnt} ({pct:.1f}%)")
        else:
            logger.info("No missing values detected")

        if self.target_col in df.columns:
            churn_rate = df[self.target_col].mean() * 100
            logger.info(
                f"Target distribution — Churn rate: {churn_rate:.1f}% "
                f"({df[self.target_col].sum():,} churners / {len(df):,} total)"
            )

    # ------------------------------------------------------------------
    # Synthetic data generator (for demo / testing when no real data)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
        """
        Generate a realistic synthetic Telco-style churn dataset.
        Useful for running the full pipeline without real data.

        Args:
            n_samples: Number of customer rows to generate.
            random_state: Reproducibility seed.

        Returns:
            Synthetic DataFrame matching the Telco schema.
        """
        rng = np.random.default_rng(random_state)

        # Core numeric features
        tenure = rng.integers(0, 72, n_samples)
        monthly_charges = rng.uniform(18, 120, n_samples).round(2)
        total_charges = (tenure * monthly_charges + rng.normal(0, 50, n_samples)).clip(0).round(2)

        # Categorical features
        binary_yes_no = lambda p: rng.choice(["Yes", "No"], n_samples, p=[p, 1 - p])
        internet_svc = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22])

        # Build DataFrame
        df = pd.DataFrame({
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": rng.choice(["Male", "Female"], n_samples),
            "SeniorCitizen": rng.choice(["0", "1"], n_samples, p=[0.84, 0.16]),
            "Partner": binary_yes_no(0.48),
            "Dependents": binary_yes_no(0.30),
            "tenure": tenure,
            "PhoneService": binary_yes_no(0.90),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.49, 0.09]),
            "InternetService": internet_svc,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.29, 0.50, 0.21]),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.34, 0.44, 0.22]),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.34, 0.44, 0.22]),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.29, 0.49, 0.22]),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.38, 0.40, 0.22]),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.39, 0.39, 0.22]),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]),
            "PaperlessBilling": binary_yes_no(0.59),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n_samples, p=[0.34, 0.23, 0.22, 0.21]
            ),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        })

        # Churn probability based on business logic:
        # High monthly charges + month-to-month + short tenure = higher churn risk
        churn_prob = (
            0.05
            + 0.15 * (df["Contract"] == "Month-to-month").astype(float)
            + 0.10 * (df["InternetService"] == "Fiber optic").astype(float)
            + 0.10 * (df["tenure"] < 12).astype(float)
            + 0.05 * (df["MonthlyCharges"] > 80).astype(float)
            - 0.08 * (df["tenure"] > 36).astype(float)
            - 0.05 * (df["Contract"] == "Two year").astype(float)
        ).clip(0.03, 0.80)

        df["Churn"] = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

        logger.info(f"Synthetic dataset generated: {n_samples:,} rows, churn rate: {df['Churn'].mean():.1%}")
        return df


# ------------------------------------------------------------------
# CLI / quick-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    loader = DataLoader(cfg)

    # Generate and save synthetic data for demo
    df = DataLoader.generate_synthetic_data(n_samples=7043)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/telco_churn.csv", index=False)
    print(f"Saved synthetic dataset: data/raw/telco_churn.csv ({len(df):,} rows)")
    print(df.head())
    print(f"\nChurn rate: {df['Churn'].mean():.1%}")
