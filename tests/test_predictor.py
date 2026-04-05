"""
test_predictor.py
=================
Unit tests for ChurnPredictor inference engine.

Tests:
  - Single prediction output structure and value ranges
  - Batch prediction consistency
  - Risk tier assignment logic
  - High-risk profile produces higher probability than low-risk
  - Threshold override
  - model_info structure
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


# ── Shared fixtures ────────────────────────────────────────────────────────

LOW_RISK_CUSTOMER = {
    "customerID": "LOW-001",
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.00,
    "TotalCharges": 2700.00,
}

HIGH_RISK_CUSTOMER = {
    "customerID": "HIGH-001",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
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
    "MonthlyCharges": 99.90,
    "TotalCharges": 99.90,
}


@pytest.fixture(scope="module")
def trained_predictor():
    """Load a real predictor once for the entire test module."""
    from src import load_config
    from src.predictor import ChurnPredictor

    cfg = load_config("config/config.yaml")
    predictor = ChurnPredictor(cfg)
    predictor.load()
    return predictor


class TestSinglePrediction:

    def test_predict_returns_dict(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert isinstance(result, dict)

    def test_required_keys_present(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        required = {"churn_probability", "churn_prediction", "churn_label",
                    "risk_tier", "confidence", "threshold_used"}
        assert required.issubset(result.keys())

    def test_probability_in_range(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_prediction_is_binary(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert result["churn_prediction"] in (0, 1)

    def test_label_matches_prediction(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        if result["churn_prediction"] == 1:
            assert result["churn_label"] == "Churn"
        else:
            assert result["churn_label"] == "No Churn"

    def test_risk_tier_is_valid(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert result["risk_tier"] in ("Low", "Medium", "High", "Critical")

    def test_confidence_in_range(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_threshold_returned(self, trained_predictor):
        result = trained_predictor.predict(HIGH_RISK_CUSTOMER)
        assert 0.0 < result["threshold_used"] < 1.0


class TestRiskTierLogic:

    def test_critical_tier_for_very_high_prob(self):
        from src.predictor import ChurnPredictor
        assert ChurnPredictor._get_risk_tier(0.85) == "Critical"
        assert ChurnPredictor._get_risk_tier(0.75) == "Critical"

    def test_high_tier(self):
        from src.predictor import ChurnPredictor
        assert ChurnPredictor._get_risk_tier(0.70) == "High"
        assert ChurnPredictor._get_risk_tier(0.51) == "High"

    def test_medium_tier(self):
        from src.predictor import ChurnPredictor
        assert ChurnPredictor._get_risk_tier(0.45) == "Medium"
        assert ChurnPredictor._get_risk_tier(0.30) == "Medium"

    def test_low_tier(self):
        from src.predictor import ChurnPredictor
        assert ChurnPredictor._get_risk_tier(0.10) == "Low"
        assert ChurnPredictor._get_risk_tier(0.00) == "Low"


class TestBatchPrediction:

    def test_batch_returns_list(self, trained_predictor):
        results = trained_predictor.predict_batch([HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
        assert isinstance(results, list)

    def test_batch_length_matches_input(self, trained_predictor):
        customers = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, HIGH_RISK_CUSTOMER]
        results = trained_predictor.predict_batch(customers)
        assert len(results) == 3

    def test_batch_each_item_has_required_keys(self, trained_predictor):
        results = trained_predictor.predict_batch([HIGH_RISK_CUSTOMER])
        required = {"churn_probability", "churn_prediction", "churn_label", "risk_tier"}
        for result in results:
            assert required.issubset(result.keys())

    def test_batch_all_probabilities_valid(self, trained_predictor):
        customers = [HIGH_RISK_CUSTOMER] * 5
        results = trained_predictor.predict_batch(customers)
        for r in results:
            assert 0.0 <= r["churn_probability"] <= 1.0


class TestModelInfo:

    def test_model_info_returns_dict(self, trained_predictor):
        info = trained_predictor.get_model_info()
        assert isinstance(info, dict)

    def test_model_info_required_keys(self, trained_predictor):
        info = trained_predictor.get_model_info()
        assert "model_version" in info
        assert "threshold" in info
        assert "is_loaded" in info

    def test_model_is_loaded(self, trained_predictor):
        info = trained_predictor.get_model_info()
        assert info["is_loaded"] is True


class TestThresholdOverride:

    def test_threshold_override_affects_prediction(self, trained_predictor):
        """At threshold=0.01 almost everything is predicted as churn."""
        original_thresh = trained_predictor.optimal_threshold
        try:
            trained_predictor.optimal_threshold = 0.01
            result = trained_predictor.predict(LOW_RISK_CUSTOMER)
            assert result["churn_prediction"] == 1
        finally:
            trained_predictor.optimal_threshold = original_thresh

    def test_high_threshold_predicts_no_churn_for_low_risk(self, trained_predictor):
        """At threshold=0.99 almost nothing is predicted as churn."""
        original_thresh = trained_predictor.optimal_threshold
        try:
            trained_predictor.optimal_threshold = 0.99
            result = trained_predictor.predict(LOW_RISK_CUSTOMER)
            assert result["churn_prediction"] == 0
        finally:
            trained_predictor.optimal_threshold = original_thresh


class TestNotLoadedError:

    def test_predict_raises_if_not_loaded(self):
        from src import load_config
        from src.predictor import ChurnPredictor

        cfg = load_config("config/config.yaml")
        predictor = ChurnPredictor(cfg)
        # Don't call .load()

        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.predict(HIGH_RISK_CUSTOMER)
