"""
predictor.py
============
Production inference engine for the churn prediction system.

Key design decisions:
  1. Singleton pattern — model loads ONCE at startup, not per request.
     This keeps API latency at ~1-5ms per prediction instead of ~500ms.

  2. Loads the preprocessing pipeline alongside the model, ensuring
     IDENTICAL transformations as training — no feature drift.

  3. Validates inputs before prediction and returns structured outputs
     with both probability scores and binary labels.

  4. Thread-safe: multiple API workers can share the same Predictor instance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src import get_logger, load_config
from src.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ChurnPredictor:
    """
    Singleton inference class.

    Loads model + preprocessing artifacts once.
    Call predict() for every inference request.

    Usage:
        predictor = ChurnPredictor(config)
        predictor.load()
        result = predictor.predict(customer_dict)
    """

    _instance: Optional["ChurnPredictor"] = None

    def __init__(self, config: dict):
        self.config = config
        self.paths = config["paths"]
        self.threshold_cfg = config.get("threshold", {})

        self.model = None
        self.preprocessing_pipeline = None
        self.feature_names: list = []
        self.optimal_threshold: float = self.threshold_cfg.get("default", 0.5)
        self.model_version: str = "unknown"
        self.model_metadata: dict = {}
        self._is_loaded: bool = False

        # Feature engineer (stateless, no fitting needed)
        self.feature_engineer = FeatureEngineer(config)

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config: dict) -> "ChurnPredictor":
        """
        Return the shared Predictor instance.
        Creates and loads it on first call (lazy initialization).
        """
        if cls._instance is None:
            cls._instance = cls(config)
            cls._instance.load()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: Optional[str] = None,
        pipeline_path: Optional[str] = None,
    ) -> "ChurnPredictor":
        """
        Load the model and preprocessing pipeline from disk.

        Args:
            model_path: Override model file path. Uses registry latest if None.
            pipeline_path: Override pipeline file path.

        Returns:
            self (for method chaining)
        """
        # Load model
        model_path = model_path or self._get_latest_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_artifact = joblib.load(model_path)
        self.model = model_artifact["model"]
        self.model_version = model_artifact.get("version", "v1")
        self.model_metadata = model_artifact.get("metadata", {})

        logger.info(f"Model loaded: {model_path} (version={self.model_version})")

        # Load preprocessing pipeline
        pipeline_path = pipeline_path or os.path.join(
            self.paths["artifact_dir"], "preprocessing_pipeline.joblib"
        )
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found: {pipeline_path}")

        pipeline_artifact = joblib.load(pipeline_path)
        self.preprocessing_pipeline = pipeline_artifact["pipeline"]
        self.feature_names = pipeline_artifact.get("feature_names", [])

        logger.info(f"Preprocessing pipeline loaded: {pipeline_path}")

        # Load optimal threshold from evaluation results if available
        eval_path = os.path.join(self.paths["artifact_dir"], "evaluation_results.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_results = json.load(f)
            saved_threshold = eval_results.get("threshold")
            if saved_threshold is not None:
                self.optimal_threshold = float(saved_threshold)
                logger.info(f"Optimal threshold loaded: {self.optimal_threshold:.4f}")

        self._is_loaded = True
        logger.info("ChurnPredictor ready for inference.")
        return self

    def predict(self, customer_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Predict churn probability for a single customer.

        Args:
            customer_data: Dict of raw customer features OR single-row DataFrame.

        Returns:
            Dict with:
              - churn_probability: float (0-1)
              - churn_prediction: int (0 or 1)
              - churn_label: str ("Churn" or "No Churn")
              - risk_tier: str ("Low" / "Medium" / "High" / "Critical")
              - confidence: float
              - threshold_used: float
        """
        self._check_loaded()

        # Convert dict to DataFrame
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()

        # Feature engineering
        df = self.feature_engineer.transform(df)

        # Preprocessing (same pipeline as training)
        X = self.preprocessing_pipeline.transform(df)

        # Predict
        proba = float(self.model.predict_proba(X)[0, 1])
        prediction = int(proba >= self.optimal_threshold)

        return self._format_output(proba, prediction)

    def predict_batch(
        self,
        customers: Union[List[Dict], pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Predict churn for multiple customers efficiently.

        Args:
            customers: List of customer dicts OR DataFrame with multiple rows.

        Returns:
            List of prediction result dicts.
        """
        self._check_loaded()

        if isinstance(customers, list):
            df = pd.DataFrame(customers)
        else:
            df = customers.copy()

        max_batch = self.config.get("api", {}).get("max_batch_size", 1000)
        if len(df) > max_batch:
            raise ValueError(f"Batch size {len(df)} exceeds maximum of {max_batch}")

        logger.info(f"Batch prediction: {len(df)} customers")

        # Feature engineering (vectorized)
        df = self.feature_engineer.transform(df)

        # Preprocessing
        X = self.preprocessing_pipeline.transform(df)

        # Batch predict
        probas = self.model.predict_proba(X)[:, 1]
        predictions = (probas >= self.optimal_threshold).astype(int)

        results = [
            self._format_output(float(p), int(pred))
            for p, pred in zip(probas, predictions)
        ]

        churn_count = sum(r["churn_prediction"] for r in results)
        logger.info(
            f"Batch complete: {churn_count}/{len(results)} predicted as churners "
            f"({churn_count/len(results):.1%})"
        )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for the /health endpoint."""
        return {
            "model_version": self.model_version,
            "algorithm": self.model_metadata.get("algorithm", "unknown"),
            "cv_roc_auc": self.model_metadata.get("cv_roc_auc_mean"),
            "n_features": self.model_metadata.get("n_features"),
            "trained_at": self.model_metadata.get("trained_at"),
            "threshold": self.optimal_threshold,
            "is_loaded": self._is_loaded,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_latest_model_path(self) -> str:
        """Look up the latest model from the registry."""
        registry_path = self.paths["registry"]
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
            latest = registry.get("latest")
            if latest and os.path.exists(latest):
                logger.info(f"Using latest model from registry: {latest}")
                return latest

        # Fallback: scan model directory
        model_dir = self.paths["model_dir"]
        if os.path.exists(model_dir):
            model_files = sorted(Path(model_dir).glob("*.joblib"), reverse=True)
            if model_files:
                path = str(model_files[0])
                logger.info(f"No registry found. Using most recent model file: {path}")
                return path

        raise FileNotFoundError(
            f"No model found in registry or directory '{model_dir}'. "
            "Run train_pipeline.py first."
        )

    def _format_output(self, proba: float, prediction: int) -> Dict[str, Any]:
        """Format raw prediction into a structured response."""
        risk_tier = self._get_risk_tier(proba)
        confidence = abs(proba - 0.5) * 2   # 0 = uncertain, 1 = highly confident

        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "risk_tier": risk_tier,
            "confidence": round(confidence, 4),
            "threshold_used": self.optimal_threshold,
        }

    @staticmethod
    def _get_risk_tier(proba: float) -> str:
        """
        Map probability to a business-friendly risk tier.
        Thresholds should be calibrated for your business context.
        """
        if proba >= 0.75:
            return "Critical"
        elif proba >= 0.50:
            return "High"
        elif proba >= 0.30:
            return "Medium"
        else:
            return "Low"

    def _check_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                "Predictor not loaded. Call predictor.load() before predict()."
            )


# ------------------------------------------------------------------
# CLI / quick-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader

    cfg = load_config()
    predictor = ChurnPredictor(cfg)
    predictor.load()

    # Single prediction with a high-risk profile
    sample_customer = {
        "customerID": "TEST-00001",
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    }

    result = predictor.predict(sample_customer)
    print("\n=== Single Prediction ===")
    for k, v in result.items():
        print(f"  {k:<25} {v}")

    print("\n=== Model Info ===")
    for k, v in predictor.get_model_info().items():
        print(f"  {k:<25} {v}")
