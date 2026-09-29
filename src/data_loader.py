"""
data_loader.py — Multi-source data ingestion and schema validation for ChurnScope.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src import get_logger, load_config
logger = get_logger(__name__)

class DataLoader:
    def __init__(self, config: dict):
        self.config    = config
        self.data_cfg  = config["data"]
        self.paths     = config["paths"]
        self.target_col = self.data_cfg["target_column"]
        self.id_col    = self.data_cfg["customer_id_column"]

    def load(self, source: Optional[str] = None) -> pd.DataFrame:
        source = source or self.paths["raw_data"]
        logger.info(f"Loading data from: {source}")
        df = self._read_source(source)
        df = self._coerce_types(df)
        self._validate_schema(df)
        self._log_data_quality(df)
        logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df

    def _read_source(self, source: str) -> pd.DataFrame:
        ext = Path(source).suffix.lower()
        if ext == ".csv":     return pd.read_csv(source)
        elif ext in (".parquet",".pq"): return pd.read_parquet(source)
        elif ext == ".json":  return pd.read_json(source)
        elif ext in (".xlsx",".xls"):   return pd.read_excel(source)
        raise ValueError(f"Unsupported format: {ext}")

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        if self.target_col in df.columns and df[self.target_col].dtype == object:
            df[self.target_col] = df[self.target_col].map({"Yes":1,"No":0})
            logger.info("Target mapped: Yes→1, No→0")
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = self.data_cfg.get("numeric_features",[]) + self.data_cfg.get("categorical_features",[])
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        logger.info("Schema validation passed")

    def _log_data_quality(self, df: pd.DataFrame) -> None:
        nulls = df.isnull().sum()
        for col, n in nulls[nulls > 0].items():
            logger.warning(f"  Nulls in {col!r}: {n} ({n/len(df)*100:.1f}%)")
        if self.target_col in df.columns:
            logger.info(f"Churn rate: {df[self.target_col].mean():.1%}")

    @staticmethod
    def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        tenure          = rng.integers(0, 72, n_samples)
        monthly_charges = rng.uniform(18, 120, n_samples).round(2)
        total_charges   = (tenure * monthly_charges + rng.normal(0, 50, n_samples)).clip(0).round(2)
        yn = lambda p: rng.choice(["Yes","No"], n_samples, p=[p, 1-p])
        internet_svc = rng.choice(["DSL","Fiber optic","No"], n_samples, p=[0.34,0.44,0.22])
        df = pd.DataFrame({
            "customerID":       [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender":           rng.choice(["Male","Female"], n_samples),
            "SeniorCitizen":    rng.choice(["0","1"], n_samples, p=[0.84,0.16]),
            "Partner":          yn(0.48),
            "Dependents":       yn(0.30),
            "tenure":           tenure,
            "PhoneService":     yn(0.90),
            "MultipleLines":    rng.choice(["Yes","No","No phone service"], n_samples, p=[0.42,0.49,0.09]),
            "InternetService":  internet_svc,
            "OnlineSecurity":   rng.choice(["Yes","No","No internet service"], n_samples, p=[0.29,0.50,0.21]),
            "OnlineBackup":     rng.choice(["Yes","No","No internet service"], n_samples, p=[0.34,0.44,0.22]),
            "DeviceProtection": rng.choice(["Yes","No","No internet service"], n_samples, p=[0.34,0.44,0.22]),
            "TechSupport":      rng.choice(["Yes","No","No internet service"], n_samples, p=[0.29,0.49,0.22]),
            "StreamingTV":      rng.choice(["Yes","No","No internet service"], n_samples, p=[0.38,0.40,0.22]),
            "StreamingMovies":  rng.choice(["Yes","No","No internet service"], n_samples, p=[0.39,0.39,0.22]),
            "Contract":         rng.choice(["Month-to-month","One year","Two year"], n_samples, p=[0.55,0.21,0.24]),
            "PaperlessBilling": yn(0.59),
            "PaymentMethod":    rng.choice(
                ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
                n_samples, p=[0.34,0.23,0.22,0.21]),
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        })
        churn_prob = (
            0.05
            + 0.15*(df["Contract"]=="Month-to-month").astype(float)
            + 0.10*(df["InternetService"]=="Fiber optic").astype(float)
            + 0.10*(df["tenure"]<12).astype(float)
            + 0.05*(df["MonthlyCharges"]>80).astype(float)
            - 0.08*(df["tenure"]>36).astype(float)
            - 0.05*(df["Contract"]=="Two year").astype(float)
        ).clip(0.03, 0.80)
        df["Churn"] = (rng.uniform(0,1,n_samples) < churn_prob).astype(int)
        logger.info(f"Synthetic data: {n_samples:,} rows, churn rate {df['Churn'].mean():.1%}")
        return df

if __name__ == "__main__":
    cfg = load_config()
    df  = DataLoader.generate_synthetic_data(n_samples=7043)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/telco_churn.csv", index=False)
    print(f"Saved synthetic dataset: {len(df):,} rows | Churn rate: {df['Churn'].mean():.1%}")
