"""
predictor.py — Singleton inference engine for ChurnScope.

Model loads ONCE at startup → shared across all API workers.
Latency: ~1-5ms per prediction (vs ~500ms if reloaded per-request).
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from src import get_logger, load_config
from src.feature_engineering import FeatureEngineer
logger = get_logger(__name__)

class ChurnPredictor:
    _instance: Optional["ChurnPredictor"] = None

    def __init__(self, config: dict):
        self.config          = config
        self.paths           = config["paths"]
        self.threshold_cfg   = config.get("threshold",{})
        self.model           = None
        self.preprocessing_pipeline = None
        self.feature_names:  list  = []
        self.optimal_threshold: float = self.threshold_cfg.get("default",0.5)
        self.model_version:  str   = "unknown"
        self.model_metadata: dict  = {}
        self._is_loaded:     bool  = False
        self.feature_engineer = FeatureEngineer(config)

    @classmethod
    def get_instance(cls, config: dict) -> "ChurnPredictor":
        if cls._instance is None:
            cls._instance = cls(config)
            cls._instance.load()
        return cls._instance

    def load(self, model_path=None, pipeline_path=None) -> "ChurnPredictor":
        model_path = model_path or self._get_latest_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        artifact = joblib.load(model_path)
        self.model          = artifact["model"]
        self.model_version  = artifact.get("version","v1")
        self.model_metadata = artifact.get("metadata",{})
        logger.info(f"Model loaded: {model_path} (version={self.model_version})")

        pipeline_path = pipeline_path or os.path.join(self.paths["artifact_dir"],"preprocessing_pipeline.joblib")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        pa = joblib.load(pipeline_path)
        self.preprocessing_pipeline = pa["pipeline"]
        self.feature_names          = pa.get("feature_names",[])
        logger.info(f"Pipeline loaded: {pipeline_path}")

        eval_path = os.path.join(self.paths["artifact_dir"],"evaluation_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f: ev = json.load(f)
            saved = ev.get("threshold")
            if saved is not None:
                self.optimal_threshold = float(saved)
                logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")

        self._is_loaded = True
        logger.info("ChurnPredictor ready.")
        return self

    def predict(self, customer_data: Union[Dict,pd.DataFrame]) -> Dict[str,Any]:
        self._check_loaded()
        df = pd.DataFrame([customer_data]) if isinstance(customer_data,dict) else customer_data.copy()
        df = self.feature_engineer.transform(df)
        X  = self.preprocessing_pipeline.transform(df)
        proba = float(self.model.predict_proba(X)[0,1])
        pred  = int(proba >= self.optimal_threshold)
        return self._format_output(proba, pred)

    def predict_batch(self, customers: Union[List[Dict],pd.DataFrame]) -> List[Dict[str,Any]]:
        self._check_loaded()
        df  = pd.DataFrame(customers) if isinstance(customers,list) else customers.copy()
        max_b = self.config.get("api",{}).get("max_batch_size",1000)
        if len(df) > max_b: raise ValueError(f"Batch size {len(df)} > max {max_b}")
        df    = self.feature_engineer.transform(df)
        X     = self.preprocessing_pipeline.transform(df)
        probas = self.model.predict_proba(X)[:,1]
        preds  = (probas >= self.optimal_threshold).astype(int)
        results = [self._format_output(float(p),int(pr)) for p,pr in zip(probas,preds)]
        churn_n = sum(r["churn_prediction"] for r in results)
        logger.info(f"Batch: {churn_n}/{len(results)} predicted churners ({churn_n/len(results):.1%})")
        return results

    def get_model_info(self) -> Dict[str,Any]:
        return {"model_version":self.model_version,"algorithm":self.model_metadata.get("algorithm","unknown"),
                "cv_roc_auc":self.model_metadata.get("cv_roc_auc_mean"),
                "n_features":self.model_metadata.get("n_features"),
                "trained_at":self.model_metadata.get("trained_at"),
                "threshold":self.optimal_threshold,"is_loaded":self._is_loaded}

    def _get_latest_model_path(self) -> str:
        rp = self.paths["registry"]
        if os.path.exists(rp):
            with open(rp) as f: reg = json.load(f)
            latest = reg.get("latest")
            if latest and os.path.exists(latest): return latest
        md = self.paths["model_dir"]
        if os.path.exists(md):
            files = sorted(Path(md).glob("*.joblib"),reverse=True)
            if files: return str(files[0])
        raise FileNotFoundError(f"No model in registry or {md}. Run train_pipeline.py first.")

    def _format_output(self, proba: float, prediction: int) -> Dict[str,Any]:
        return {"churn_probability":round(proba,4),"churn_prediction":prediction,
                "churn_label":"Churn" if prediction==1 else "No Churn",
                "risk_tier":self._get_risk_tier(proba),
                "confidence":round(abs(proba-0.5)*2,4),
                "threshold_used":self.optimal_threshold}

    @staticmethod
    def _get_risk_tier(proba: float) -> str:
        if proba >= 0.75: return "Critical"
        elif proba >= 0.50: return "High"
        elif proba >= 0.30: return "Medium"
        return "Low"

    def _check_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")
