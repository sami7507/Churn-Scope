"""
trainer.py — Model training for ChurnScope.

Supports LightGBM (default) and XGBoost with:
  - Stratified K-Fold cross-validation
  - Early stopping on validation set
  - Optional Optuna hyperparameter optimisation
  - Model versioning + registry tracking
"""
from __future__ import annotations
import json, os, time, uuid
from datetime import datetime
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src import get_logger
logger = get_logger(__name__)

class ChurnTrainer:
    def __init__(self, config: dict):
        self.config    = config
        self.model_cfg = config["model"]
        self.paths     = config["paths"]
        self.algorithm = self.model_cfg.get("algorithm","lightgbm")
        self.model     = None
        self.training_metadata: Dict[str,Any] = {}

    def train(self, splits: Dict[str,Any]) -> Dict[str,Any]:
        X_train,y_train = splits["X_train"],splits["y_train"]
        X_val,  y_val   = splits["X_val"],  splits["y_val"]
        feature_names   = splits.get("feature_names",[])
        logger.info(f"Training {self.algorithm.upper()}...")
        logger.info(f"Train:{len(X_train):,} Val:{len(X_val):,}")
        start  = time.time()
        run_id = str(uuid.uuid4())[:8]
        params = self._run_optuna(X_train,y_train) if self.config.get("optuna",{}).get("enabled") else self._get_default_params()
        cv_scores = self._cross_validate(X_train,y_train,params)
        self.model = self._build_model(params)
        self._fit_model(X_train,y_train,X_val,y_val)
        elapsed = time.time()-start
        self.training_metadata = {
            "run_id":run_id,"algorithm":self.algorithm,"params":params,
            "cv_roc_auc_mean":float(cv_scores.mean()),"cv_roc_auc_std":float(cv_scores.std()),
            "cv_roc_auc_scores":cv_scores.tolist(),"train_samples":int(len(X_train)),
            "val_samples":int(len(X_val)),"n_features":int(X_train.shape[1]),
            "feature_names":feature_names,"training_time_seconds":round(elapsed,2),
            "trained_at":datetime.now().isoformat(),
        }
        logger.info(f"Training done in {elapsed:.1f}s | CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.training_metadata

    def save(self, version: str = None) -> str:
        if self.model is None: raise RuntimeError("No model to save. Call train() first.")
        version   = version or self.model_cfg.get("version","v1")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"churn_model_{self.algorithm}_{version}_{timestamp}.joblib"
        os.makedirs(self.paths["model_dir"], exist_ok=True)
        save_path = os.path.join(self.paths["model_dir"], filename)
        joblib.dump({"model":self.model,"algorithm":self.algorithm,"version":version,"metadata":self.training_metadata}, save_path)
        logger.info(f"Model saved: {save_path}")
        self._update_registry(save_path, version)
        return save_path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model not loaded.")
        return self.model.predict_proba(X)[:,1]

    def _get_default_params(self) -> dict:
        key = f"{self.algorithm}_params"
        p   = self.model_cfg.get(key,{}).copy()
        for k in ("class_weight","use_label_encoder","eval_metric"): p.pop(k,None)
        return p

    def _build_model(self, params: dict):
        if self.algorithm == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**params, verbose=-1)
            except ImportError: self.algorithm = "xgboost"
        if self.algorithm == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**{k:v for k,v in params.items() if k!="class_weight"},verbosity=0)
            except ImportError: pass
        from sklearn.ensemble import GradientBoostingClassifier
        safe = {k:v for k,v in params.items() if k in ("n_estimators","learning_rate","max_depth","random_state")}
        return GradientBoostingClassifier(**safe)

    def _fit_model(self, X_train,y_train,X_val,y_val) -> None:
        es_cfg = self.config["model"].get("early_stopping",{})
        use_es = es_cfg.get("enabled",True)
        rounds = es_cfg.get("rounds",50)
        try:
            if self.algorithm=="lightgbm" and use_es:
                import lightgbm as lgb
                self.model.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                               callbacks=[lgb.early_stopping(rounds,verbose=False),lgb.log_evaluation(-1)])
            elif self.algorithm=="xgboost" and use_es:
                self.model.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                               early_stopping_rounds=rounds,verbose=False)
            else:
                self.model.fit(X_train,y_train)
        except Exception as e:
            logger.warning(f"Early stopping failed ({e}), fitting without.")
            self.model.fit(X_train,y_train)

    def _cross_validate(self, X,y,params) -> np.ndarray:
        n = self.config["model"]["cross_validation"].get("n_splits",5)
        logger.info(f"Running {n}-fold stratified CV...")
        scores = cross_val_score(self._build_model(params),X,y,
                                 cv=StratifiedKFold(n_splits=n,shuffle=True,random_state=42),
                                 scoring="roc_auc",n_jobs=-1)
        for i,s in enumerate(scores,1): logger.info(f"  Fold {i}: {s:.4f}")
        return scores

    def _run_optuna(self, X_train,y_train) -> dict:
        try: import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError: return self._get_default_params()
        cfg = self.config["optuna"]
        skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
        def objective(trial):
            if self.algorithm=="lightgbm":
                p={"n_estimators":trial.suggest_int("n_estimators",100,1000),
                   "learning_rate":trial.suggest_float("learning_rate",.01,.3,log=True),
                   "max_depth":trial.suggest_int("max_depth",3,10),
                   "num_leaves":trial.suggest_int("num_leaves",15,127),
                   "subsample":trial.suggest_float("subsample",.5,1.),
                   "colsample_bytree":trial.suggest_float("colsample_bytree",.5,1.),
                   "random_state":42}
            else:
                p={"n_estimators":trial.suggest_int("n_estimators",100,1000),
                   "learning_rate":trial.suggest_float("learning_rate",.01,.3,log=True),
                   "max_depth":trial.suggest_int("max_depth",3,8),"random_state":42}
            return cross_val_score(self._build_model(p),X_train,y_train,cv=skf,scoring="roc_auc",n_jobs=-1).mean()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective,n_trials=cfg.get("n_trials",50),timeout=cfg.get("timeout",600))
        logger.info(f"Optuna best ROC-AUC: {study.best_value:.4f}")
        return study.best_params

    def _update_registry(self, model_path,version):
        registry_path = self.paths["registry"]
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path) as f: registry = json.load(f)
        registry.setdefault("models",[]).append({
            "model_path":model_path,"version":version,"algorithm":self.algorithm,
            "cv_roc_auc":self.training_metadata.get("cv_roc_auc_mean"),
            "trained_at":self.training_metadata.get("trained_at"),
            "n_features":self.training_metadata.get("n_features"),"status":"active",
        })
        registry["latest"] = model_path
        with open(registry_path,"w") as f: json.dump(registry,f,indent=2)
        logger.info(f"Registry updated: {registry_path}")
