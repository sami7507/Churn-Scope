"""
preprocessor.py
===============
Data preprocessing pipeline for churn prediction.

Responsibilities:
  - Impute missing values (median for numerics, mode for categoricals)
  - Encode categorical features (OrdinalEncoder)
  - Scale numeric features (StandardScaler)
  - Stratified train/validation/test split
  - Persist the fitted pipeline as a joblib artifact (critical for leak-free inference)

Key design decision:
  All transformations are wrapped in a single sklearn Pipeline. At inference time,
  the SAME fitted pipeline is loaded and applied — no risk of train/test leakage
  from recomputing statistics on new data.
"""

from __future__ import annotations

import os
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src import get_logger, load_config

logger = get_logger(__name__)


class ChurnPreprocessor:
    """
    Fit-transform preprocessing pipeline.

    Fit once on training data → transform train/val/test splits consistently.

    Usage:
        preprocessor = ChurnPreprocessor(config)
        splits = preprocessor.fit_transform(df)
        preprocessor.save()
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.paths = config["paths"]

        self.target_col = self.data_cfg["target_column"]
        self.id_col = self.data_cfg["customer_id_column"]
        self.numeric_features = self.data_cfg["numeric_features"]
        self.categorical_features = self.data_cfg["categorical_features"]

        self.test_size = self.data_cfg.get("test_size", 0.2)
        self.val_size = self.data_cfg.get("val_size", 0.1)
        self.random_state = self.data_cfg.get("random_state", 42)

        self.pipeline: Pipeline = None
        self.feature_names: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Full preprocessing: split → fit on train → transform all splits.

        Args:
            df: Raw DataFrame with target column present.

        Returns:
            Dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                            feature_names, class_weights
        """
        logger.info("Starting preprocessing pipeline...")

        # Drop ID column
        df = df.drop(columns=[self.id_col], errors="ignore")

        # Separate features / target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(int)

        logger.info(f"Features: {X.shape[1]} columns | Target balance: {y.mean():.1%} positive")

        # Train / (val+test) split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Val / test split from the remainder
        val_fraction = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - val_fraction,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(
            f"Split sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        # Build and fit the pipeline on training data only
        self.pipeline = self._build_pipeline()
        X_train_proc = self.pipeline.fit_transform(X_train)
        X_val_proc = self.pipeline.transform(X_val)
        X_test_proc = self.pipeline.transform(X_test)

        # Retrieve feature names after transformation
        self.feature_names = self._get_feature_names()

        # Class weights for imbalanced target
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        class_weights = {0: 1.0, 1: n_neg / n_pos}
        logger.info(f"Class weights: {class_weights}")

        splits = {
            "X_train": X_train_proc,
            "X_val": X_val_proc,
            "X_test": X_test_proc,
            "y_train": y_train.values,
            "y_val": y_val.values,
            "y_test": y_test.values,
            "feature_names": self.feature_names,
            "class_weights": class_weights,
        }

        logger.info(
            f"Preprocessing complete. "
            f"Feature matrix shape: {X_train_proc.shape[1]} features"
        )
        return splits

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted pipeline to new data (inference).

        Args:
            df: Raw input DataFrame (without target column).

        Returns:
            Processed feature array.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() or load() first.")
        df = df.drop(columns=[self.id_col], errors="ignore")
        df = df.drop(columns=[self.target_col], errors="ignore")
        return self.pipeline.transform(df)

    def save(self, path: str = None) -> str:
        """
        Persist the fitted pipeline to disk.

        Returns the saved file path.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted yet.")

        os.makedirs(self.paths["artifact_dir"], exist_ok=True)
        save_path = path or os.path.join(
            self.paths["artifact_dir"], "preprocessing_pipeline.joblib"
        )
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names},
            save_path,
        )
        logger.info(f"Preprocessing pipeline saved: {save_path}")
        return save_path

    def load(self, path: str = None) -> "ChurnPreprocessor":
        """Load a previously saved pipeline."""
        load_path = path or os.path.join(
            self.paths["artifact_dir"], "preprocessing_pipeline.joblib"
        )
        artifact = joblib.load(load_path)
        self.pipeline = artifact["pipeline"]
        self.feature_names = artifact["feature_names"]
        logger.info(f"Preprocessing pipeline loaded: {load_path}")
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        """
        Construct the ColumnTransformer + Pipeline.

        Numeric path:  SimpleImputer(median) → StandardScaler
        Categorical path: SimpleImputer(most_frequent) → OrdinalEncoder
        """
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        # Only include columns that actually exist in the data
        existing_numeric = [c for c in self.numeric_features if c in self._known_columns()]
        existing_categorical = [c for c in self.categorical_features if c in self._known_columns()]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, existing_numeric),
                ("cat", categorical_transformer, existing_categorical),
            ],
            remainder="drop",   # Drop unknown / ID columns
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline([("preprocessor", preprocessor)])
        logger.info(
            f"Pipeline built: {len(existing_numeric)} numeric features, "
            f"{len(existing_categorical)} categorical features"
        )
        return pipeline

    def _known_columns(self) -> list:
        """Return all expected feature columns (used before we have the actual df)."""
        return self.numeric_features + self.categorical_features

    def _get_feature_names(self) -> list:
        """Extract output feature names from the fitted ColumnTransformer."""
        try:
            ct = self.pipeline.named_steps["preprocessor"]
            return list(ct.get_feature_names_out())
        except Exception:
            # Fallback if sklearn version doesn't support get_feature_names_out
            return self.numeric_features + self.categorical_features


# ------------------------------------------------------------------
# CLI / quick-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader

    cfg = load_config()
    loader = DataLoader(cfg)
    df = DataLoader.generate_synthetic_data()

    preprocessor = ChurnPreprocessor(cfg)
    splits = preprocessor.fit_transform(df)
    preprocessor.save()

    print("\n=== Preprocessing Summary ===")
    print(f"X_train shape : {splits['X_train'].shape}")
    print(f"X_val shape   : {splits['X_val'].shape}")
    print(f"X_test shape  : {splits['X_test'].shape}")
    print(f"Feature names : {splits['feature_names'][:5]}...")
    print(f"Class weights : {splits['class_weights']}")
