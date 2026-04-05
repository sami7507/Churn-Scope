"""
feature_engineering.py
=======================
Creates domain-specific features that raw data doesn't contain.

Why feature engineering matters:
  - Raw columns like "tenure" and "MonthlyCharges" contain information,
    but derived features like "charges_per_month_of_tenure" or
    "is_new_customer" can expose non-linear patterns more directly.
  - Good features often matter more than model choice.
  - All features here are business-interpretable, which aids in explaining
    model decisions to stakeholders.

Feature categories created:
  1. Tenure bands — segment customers by lifecycle stage
  2. Charge ratios — normalize charges by tenure
  3. Service count — total number of add-on services
  4. Interaction features — multiply correlated signals
  5. Contract risk flags — encode high-churn indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import get_logger, load_config

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Creates engineered features from raw churn data.

    Designed to run BEFORE the preprocessing pipeline so engineered
    features are included in the scaling/encoding step.

    Usage:
        fe = FeatureEngineer(config)
        df_enriched = fe.transform(df_raw)
    """

    def __init__(self, config: dict):
        self.config = config
        self.fe_cfg = config.get("feature_engineering", {})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Args:
            df: Raw DataFrame (post type-coercion, pre preprocessing).

        Returns:
            DataFrame with additional engineered feature columns.
        """
        df = df.copy()
        initial_cols = df.shape[1]

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

        new_cols = df.shape[1] - initial_cols
        logger.info(
            f"Feature engineering complete: {new_cols} new features created "
            f"({initial_cols} → {df.shape[1]} total columns)"
        )
        return df

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers by lifecycle stage.

        Business insight:
          New customers (<6 months) churn at 3-4× the rate of long-tenured
          customers. Creating explicit bands lets the model learn
          threshold effects directly.
        """
        if "tenure" not in df.columns:
            return df

        # Tenure bands (categorical signal)
        conditions = [
            df["tenure"] <= 6,
            df["tenure"] <= 12,
            df["tenure"] <= 24,
            df["tenure"] <= 48,
        ]
        choices = [0, 1, 2, 3]   # 0=new, 1=developing, 2=established, 3=loyal, 4=champion
        df["tenure_band"] = np.select(conditions, choices, default=4)

        # Is new customer (strong churn signal)
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

        # Is loyal customer (low churn signal)
        df["is_loyal_customer"] = (df["tenure"] >= 24).astype(int)

        # Log tenure (reduces skew)
        df["log_tenure"] = np.log1p(df["tenure"])

        logger.debug("Tenure features created: tenure_band, is_new_customer, is_loyal_customer, log_tenure")
        return df

    def _charge_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize charges by tenure to capture value perception.

        Business insight:
          A customer paying $100/month but only 1 month in has a very
          different risk profile than one paying $100/month for 5 years.
          The ratio captures this.
        """
        if "MonthlyCharges" not in df.columns:
            return df

        # Charges per tenure month (avoid div/0 for tenure=0)
        df["monthly_charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

        # Total charges vs expected (monthly × tenure) — gap signals promotions / downgrades
        if "TotalCharges" in df.columns:
            expected_total = df["MonthlyCharges"] * df["tenure"]
            df["charge_discrepancy"] = (df["TotalCharges"] - expected_total) / (expected_total + 1)

        # High-value customer flag
        df["is_high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

        # Log monthly charges (reduce right skew)
        df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

        logger.debug("Charge ratio features created")
        return df

    def _service_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count total subscribed add-on services.

        Business insight:
          Customers with more services are stickier — each service is a
          switching cost. A customer with 6 services is less likely to
          churn than one with 1.
        """
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        available_service_cols = [c for c in service_cols if c in df.columns]

        if not available_service_cols:
            return df

        # Count services = "Yes"
        def is_active(series: pd.Series) -> pd.Series:
            return (series == "Yes").astype(int)

        service_flags = pd.DataFrame({
            f"has_{col.lower()}": is_active(df[col])
            for col in available_service_cols
            if col in df.columns
        })
        df["total_services"] = service_flags.sum(axis=1)
        df["service_penetration"] = df["total_services"] / len(available_service_cols)

        # Has internet service (internet customers churn more from Fiber optic)
        if "InternetService" in df.columns:
            df["has_internet"] = (df["InternetService"] != "No").astype(int)
            df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

        logger.debug(f"Service count features created from {len(available_service_cols)} service columns")
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multiply correlated signals to create non-linear features.

        Business insight:
          High charges + short tenure is the canonical churn profile.
          An interaction term captures this pattern more precisely than
          either feature alone.
        """
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            # High charges + new customer = very high risk
            df["charge_tenure_risk"] = df["MonthlyCharges"] * (1 / (df["tenure"] + 1))

        if "total_services" in df.columns and "tenure" in df.columns:
            # Service stickiness: more services + longer tenure = very sticky
            df["stickiness_score"] = df["total_services"] * np.log1p(df["tenure"])

        if "is_new_customer" in df.columns and "is_high_value" in df.columns:
            # New + high value = highest priority retention target
            df["high_risk_flag"] = (df["is_new_customer"] & df["is_high_value"]).astype(int)

        logger.debug("Interaction features created")
        return df

    def _contract_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode known high-churn contract and payment patterns.

        Business insight:
          Month-to-month contracts + electronic check payment is the
          highest-churn combination in Telco data — it signals customers
          who haven't committed and are most price-sensitive.
        """
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
            df["is_long_term_contract"] = (df["Contract"] == "Two year").astype(int)

        if "PaymentMethod" in df.columns:
            df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
            df["is_auto_payment"] = df["PaymentMethod"].str.contains(
                "automatic", case=False, na=False
            ).astype(int)

        if "PaperlessBilling" in df.columns:
            df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

        # Combined highest-risk profile
        if all(c in df.columns for c in ["is_month_to_month", "is_electronic_check"]):
            df["highest_churn_risk_profile"] = (
                df["is_month_to_month"] & df["is_electronic_check"]
            ).astype(int)

        logger.debug("Contract risk features created")
        return df

    def get_new_feature_names(self) -> list:
        """Return names of all features this class creates."""
        return [
            # Tenure
            "tenure_band", "is_new_customer", "is_loyal_customer", "log_tenure",
            # Charges
            "monthly_charge_per_tenure", "charge_discrepancy",
            "is_high_value", "log_monthly_charges",
            # Services
            "total_services", "service_penetration", "has_internet", "has_fiber",
            # Interactions
            "charge_tenure_risk", "stickiness_score", "high_risk_flag",
            # Contract risk
            "is_month_to_month", "is_long_term_contract",
            "is_electronic_check", "is_auto_payment", "is_paperless",
            "highest_churn_risk_profile",
        ]


# ------------------------------------------------------------------
# CLI / quick-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader

    cfg = load_config()
    df_raw = DataLoader.generate_synthetic_data(n_samples=1000)

    fe = FeatureEngineer(cfg)
    df_enriched = fe.transform(df_raw)

    print(f"\nOriginal columns : {df_raw.shape[1]}")
    print(f"Enriched columns : {df_enriched.shape[1]}")
    print(f"New features     : {df_enriched.shape[1] - df_raw.shape[1]}")
    print("\nSample enriched features:")
    new_feature_cols = [
        "tenure_band", "is_new_customer", "total_services",
        "charge_tenure_risk", "is_month_to_month", "stickiness_score"
    ]
    print(df_enriched[[c for c in new_feature_cols if c in df_enriched.columns]].head())
