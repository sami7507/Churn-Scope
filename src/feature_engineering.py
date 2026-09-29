"""
feature_engineering.py — Domain-specific feature creation for ChurnScope.

Creates 22 engineered features across 5 categories:
  1. Tenure bands        — lifecycle stage segmentation
  2. Charge ratios       — value perception normalisation
  3. Service count       — stickiness measurement
  4. Interaction features — non-linear signal amplification
  5. Contract risk flags  — known high-churn pattern encoding
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import get_logger
logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.fe_cfg = config.get("feature_engineering", {})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        initial = df.shape[1]
        logger.info("Starting feature engineering...")
        if self.fe_cfg.get("create_tenure_bands", True):
            df = self._tenure_features(df)
        if self.fe_cfg.get("create_charge_ratios", True):
            df = self._charge_ratio_features(df)
        if self.fe_cfg.get("create_service_count", True):
            df = self._service_count_features(df)
        if self.fe_cfg.get("create_interaction_features", True):
            df = self._interaction_features(df)
        df = self._contract_risk_features(df)
        logger.info(f"Feature engineering: {df.shape[1]-initial} new features ({initial}→{df.shape[1]})")
        return df

    def _tenure_features(self, df):
        if "tenure" not in df.columns: return df
        df["tenure_band"]       = np.select([df["tenure"]<=6,df["tenure"]<=12,df["tenure"]<=24,df["tenure"]<=48],[0,1,2,3],default=4)
        df["is_new_customer"]   = (df["tenure"] <= 6).astype(int)
        df["is_loyal_customer"] = (df["tenure"] >= 24).astype(int)
        df["log_tenure"]        = np.log1p(df["tenure"])
        return df

    def _charge_ratio_features(self, df):
        if "MonthlyCharges" not in df.columns: return df
        df["monthly_charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        if "TotalCharges" in df.columns:
            expected = df["MonthlyCharges"] * df["tenure"]
            df["charge_discrepancy"] = (df["TotalCharges"] - expected) / (expected + 1)
        df["is_high_value"]      = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
        df["log_monthly_charges"]= np.log1p(df["MonthlyCharges"])
        return df

    def _service_count_features(self, df):
        service_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        avail = [c for c in service_cols if c in df.columns]
        if not avail: return df
        svc_flags = pd.DataFrame({f"has_{c.lower()}":(df[c]=="Yes").astype(int) for c in avail})
        df["total_services"]      = svc_flags.sum(axis=1)
        df["service_penetration"] = df["total_services"] / len(avail)
        if "InternetService" in df.columns:
            df["has_internet"] = (df["InternetService"] != "No").astype(int)
            df["has_fiber"]    = (df["InternetService"] == "Fiber optic").astype(int)
        return df

    def _interaction_features(self, df):
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["charge_tenure_risk"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        if "total_services" in df.columns and "tenure" in df.columns:
            df["stickiness_score"] = df["total_services"] * np.log1p(df["tenure"])
        if "is_new_customer" in df.columns and "is_high_value" in df.columns:
            df["high_risk_flag"] = (df["is_new_customer"] & df["is_high_value"]).astype(int)
        return df

    def _contract_risk_features(self, df):
        if "Contract" in df.columns:
            df["is_month_to_month"]      = (df["Contract"] == "Month-to-month").astype(int)
            df["is_long_term_contract"]  = (df["Contract"] == "Two year").astype(int)
        if "PaymentMethod" in df.columns:
            df["is_electronic_check"]    = (df["PaymentMethod"] == "Electronic check").astype(int)
            df["is_auto_payment"]        = df["PaymentMethod"].str.contains("automatic",case=False,na=False).astype(int)
        if "PaperlessBilling" in df.columns:
            df["is_paperless"]           = (df["PaperlessBilling"] == "Yes").astype(int)
        if all(c in df.columns for c in ["is_month_to_month","is_electronic_check"]):
            df["highest_churn_risk_profile"] = (df["is_month_to_month"] & df["is_electronic_check"]).astype(int)
        return df

    def get_new_feature_names(self) -> list:
        return ["tenure_band","is_new_customer","is_loyal_customer","log_tenure",
                "monthly_charge_per_tenure","charge_discrepancy","is_high_value","log_monthly_charges",
                "total_services","service_penetration","has_internet","has_fiber",
                "charge_tenure_risk","stickiness_score","high_risk_flag",
                "is_month_to_month","is_long_term_contract","is_electronic_check",
                "is_auto_payment","is_paperless","highest_churn_risk_profile"]
