"""
preprocessor.py — Fit/transform preprocessing pipeline for ChurnScope.

Key design: All transformations wrapped in one sklearn Pipeline.
Fit on train only → transform all splits → save as joblib → load at inference.
No train/test leakage possible.
"""
from __future__ import annotations
import os
import joblib
from typing import Dict, Any
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
    def __init__(self, config: dict):
        self.config       = config
        self.data_cfg     = config["data"]
        self.paths        = config["paths"]
        self.target_col   = self.data_cfg["target_column"]
        self.id_col       = self.data_cfg["customer_id_column"]
        self.numeric_feats    = self.data_cfg["numeric_features"]
        self.categorical_feats = self.data_cfg["categorical_features"]
        self.test_size    = self.data_cfg.get("test_size", 0.2)
        self.val_size     = self.data_cfg.get("val_size", 0.1)
        self.random_state = self.data_cfg.get("random_state", 42)
        self.pipeline     = None
        self.feature_names: list = []

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting preprocessing pipeline...")
        df = df.drop(columns=[self.id_col], errors="ignore")
        X  = df.drop(columns=[self.target_col])
        y  = df[self.target_col].astype(int)
        logger.info(f"Features: {X.shape[1]} | Churn rate: {y.mean():.1%}")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size+self.val_size,
            random_state=self.random_state, stratify=y)
        val_frac = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_frac,
            random_state=self.random_state, stratify=y_temp)

        logger.info(f"Train:{len(X_train):,} Val:{len(X_val):,} Test:{len(X_test):,}")
        self.pipeline = self._build_pipeline()
        X_train_p = self.pipeline.fit_transform(X_train)
        X_val_p   = self.pipeline.transform(X_val)
        X_test_p  = self.pipeline.transform(X_test)
        self.feature_names = self._get_feature_names()

        n_neg = (y_train==0).sum(); n_pos = (y_train==1).sum()
        class_weights = {0:1.0, 1:n_neg/n_pos}
        logger.info(f"Class weights: {class_weights}")
        return {"X_train":X_train_p,"X_val":X_val_p,"X_test":X_test_p,
                "y_train":y_train.values,"y_val":y_val.values,"y_test":y_test.values,
                "feature_names":self.feature_names,"class_weights":class_weights}

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() or load() first.")
        df = df.drop(columns=[self.id_col,self.target_col], errors="ignore")
        return self.pipeline.transform(df)

    def save(self, path: str = None) -> str:
        if self.pipeline is None: raise RuntimeError("Pipeline not fitted.")
        os.makedirs(self.paths["artifact_dir"], exist_ok=True)
        save_path = path or os.path.join(self.paths["artifact_dir"],"preprocessing_pipeline.joblib")
        joblib.dump({"pipeline":self.pipeline,"feature_names":self.feature_names}, save_path)
        logger.info(f"Pipeline saved: {save_path}")
        return save_path

    def load(self, path: str = None) -> "ChurnPreprocessor":
        load_path = path or os.path.join(self.paths["artifact_dir"],"preprocessing_pipeline.joblib")
        artifact  = joblib.load(load_path)
        self.pipeline      = artifact["pipeline"]
        self.feature_names = artifact["feature_names"]
        logger.info(f"Pipeline loaded: {load_path}")
        return self

    def _build_pipeline(self) -> Pipeline:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        known = self.numeric_feats + self.categorical_feats
        ex_num = [c for c in self.numeric_feats if c in known]
        ex_cat = [c for c in self.categorical_feats if c in known]
        preprocessor = ColumnTransformer(
            transformers=[("num",num_transformer,ex_num),("cat",cat_transformer,ex_cat)],
            remainder="drop", verbose_feature_names_out=False)
        logger.info(f"Pipeline: {len(ex_num)} numeric, {len(ex_cat)} categorical features")
        return Pipeline([("preprocessor",preprocessor)])

    def _get_feature_names(self) -> list:
        try:
            return list(self.pipeline.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            return self.numeric_feats + self.categorical_feats
