"""
trainer.py
==========
Model training module for the churn prediction system.

Supports:
  - LightGBM (default, recommended for tabular churn data)
  - XGBoost (alternative)
  - Stratified K-Fold cross-validation for robust evaluation
  - Early stopping to prevent overfitting
  - Optional Optuna hyperparameter optimization
  - Model versioning and registry tracking

MLOps practices:
  - Every training run is logged (params, metrics, artifacts)
  - Model is saved with a version stamp
  - model_registry.json tracks all versions for rollback capability
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import get_logger, load_config

logger = get_logger(__name__)


class ChurnTrainer:
    """
    Trains LightGBM or XGBoost churn prediction models.

    Usage:
        trainer = ChurnTrainer(config)
        result = trainer.train(splits)
        trainer.save()
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config["model"]
        self.paths = config["paths"]
        self.algorithm = self.model_cfg.get("algorithm", "lightgbm")
        self.model = None
        self.training_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model with cross-validation.

        Args:
            splits: Output from ChurnPreprocessor.fit_transform() containing
                    X_train, X_val, y_train, y_val, feature_names.

        Returns:
            Training result metadata dict.
        """
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_val = splits["X_val"]
        y_val = splits["y_val"]
        feature_names = splits.get("feature_names", [])

        logger.info(f"Training {self.algorithm.upper()} model...")
        logger.info(f"Training samples: {len(X_train):,} | Validation samples: {len(X_val):,}")

        start_time = time.time()
        run_id = str(uuid.uuid4())[:8]

        # Optional Optuna HPO
        if self.config.get("optuna", {}).get("enabled", False):
            params = self._run_optuna(X_train, y_train)
            logger.info("Optuna HPO complete. Using tuned parameters.")
        else:
            params = self._get_default_params()

        # Cross-validation on training set
        cv_scores = self._cross_validate(X_train, y_train, params)

        # Final model trained on full training split
        self.model = self._build_model(params)
        self._fit_model(X_train, y_train, X_val, y_val)

        elapsed = time.time() - start_time

        # Collect training metadata
        self.training_metadata = {
            "run_id": run_id,
            "algorithm": self.algorithm,
            "params": params,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "cv_roc_auc_scores": cv_scores.tolist(),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "n_features": int(X_train.shape[1]),
            "feature_names": feature_names,
            "training_time_seconds": round(elapsed, 2),
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Training complete in {elapsed:.1f}s | "
            f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        return self.training_metadata

    def save(self, version: str = None) -> str:
        """
        Save the trained model and update the model registry.

        Returns:
            Path to the saved model file.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        version = version or self.model_cfg.get("version", "v1")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"churn_model_{self.algorithm}_{version}_{timestamp}.joblib"

        os.makedirs(self.paths["model_dir"], exist_ok=True)
        save_path = os.path.join(self.paths["model_dir"], filename)

        artifact = {
            "model": self.model,
            "algorithm": self.algorithm,
            "version": version,
            "metadata": self.training_metadata,
        }
        joblib.dump(artifact, save_path)
        logger.info(f"Model saved: {save_path}")

        # Update registry
        self._update_registry(save_path, version)

        return save_path

    def load(self, path: str) -> "ChurnTrainer":
        """Load a previously saved model artifact."""
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.algorithm = artifact["algorithm"]
        self.training_metadata = artifact.get("metadata", {})
        logger.info(f"Model loaded from: {path}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return churn probability scores."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_default_params(self) -> dict:
        """Get default hyperparameters from config."""
        key = f"{self.algorithm}_params"
        params = self.model_cfg.get(key, {}).copy()
        # Remove sklearn-style params not used in direct API
        params.pop("class_weight", None)
        params.pop("use_label_encoder", None)
        params.pop("eval_metric", None)
        return params

    def _build_model(self, params: dict):
        """Instantiate the model with given parameters."""
        if self.algorithm == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**params, verbose=-1)
            except ImportError:
                logger.warning("LightGBM not installed. Falling back to XGBoost.")
                self.algorithm = "xgboost"

        if self.algorithm == "xgboost":
            try:
                from xgboost import XGBClassifier
                xgb_params = {k: v for k, v in params.items()
                              if k not in ("class_weight",)}
                return XGBClassifier(**xgb_params, verbosity=0)
            except ImportError:
                logger.warning("XGBoost not installed. Falling back to GradientBoosting.")

        # Last resort fallback
        from sklearn.ensemble import GradientBoostingClassifier
        logger.warning("Using sklearn GradientBoostingClassifier as fallback.")
        safe_params = {k: v for k, v in params.items()
                       if k in ("n_estimators", "learning_rate", "max_depth", "random_state")}
        return GradientBoostingClassifier(**safe_params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit the model with early stopping using validation set."""
        early_stopping_cfg = self.config["model"].get("early_stopping", {})
        use_early_stopping = early_stopping_cfg.get("enabled", True)
        rounds = early_stopping_cfg.get("rounds", 50)

        try:
            if self.algorithm == "lightgbm" and use_early_stopping:
                import lightgbm as lgb
                callbacks = [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(-1)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                )
                logger.info(f"Best iteration: {self.model.best_iteration_}")

            elif self.algorithm == "xgboost" and use_early_stopping:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=rounds,
                    verbose=False,
                )
            else:
                self.model.fit(X_train, y_train)

        except Exception as e:
            logger.warning(f"Early stopping failed ({e}), fitting without it.")
            self.model.fit(X_train, y_train)

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Run stratified k-fold CV and return ROC-AUC scores."""
        cv_cfg = self.config["model"]["cross_validation"]
        n_splits = cv_cfg.get("n_splits", 5)

        logger.info(f"Running {n_splits}-fold stratified cross-validation...")

        cv_model = self._build_model(params)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = cross_val_score(
            cv_model, X, y,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
        )

        for i, s in enumerate(scores, 1):
            logger.info(f"  Fold {i}: ROC-AUC = {s:.4f}")

        return scores

    def _run_optuna(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Run Optuna hyperparameter optimization.
        Returns the best parameters found.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return self._get_default_params()

        optuna_cfg = self.config["optuna"]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective(trial):
            if self.algorithm == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "random_state": 42,
                    "verbose": -1,
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "random_state": 42,
                    "verbosity": 0,
                }

            model = self._build_model(params)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=optuna_cfg.get("n_trials", 50),
            timeout=optuna_cfg.get("timeout", 600),
        )

        logger.info(f"Optuna best ROC-AUC: {study.best_value:.4f}")
        return study.best_params

    def _update_registry(self, model_path: str, version: str) -> None:
        """Update the model registry JSON with this training run."""
        os.makedirs(self.paths["model_dir"].replace("trained", "registry"), exist_ok=True)
        registry_path = self.paths["registry"]

        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)

        entry = {
            "model_path": model_path,
            "version": version,
            "algorithm": self.algorithm,
            "cv_roc_auc": self.training_metadata.get("cv_roc_auc_mean"),
            "trained_at": self.training_metadata.get("trained_at"),
            "n_features": self.training_metadata.get("n_features"),
            "status": "active",
        }

        registry.setdefault("models", []).append(entry)
        registry["latest"] = model_path

        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Model registry updated: {registry_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.preprocessor import ChurnPreprocessor

    cfg = load_config()

    df = DataLoader.generate_synthetic_data()
    fe = FeatureEngineer(cfg)
    df = fe.transform(df)

    preprocessor = ChurnPreprocessor(cfg)
    splits = preprocessor.fit_transform(df)

    trainer = ChurnTrainer(cfg)
    metadata = trainer.train(splits)
    model_path = trainer.save()

    print(f"\nModel saved: {model_path}")
    print(f"CV ROC-AUC : {metadata['cv_roc_auc_mean']:.4f} ± {metadata['cv_roc_auc_std']:.4f}")
