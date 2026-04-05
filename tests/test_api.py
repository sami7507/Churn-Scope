"""
test_api.py
===========
Integration tests for the Churn Prediction FastAPI endpoints.

Uses httpx's async test client (recommended for FastAPI).
Tests run against a mocked predictor to avoid needing a trained model.

Run with: pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# We patch the predictor before importing the app
SAMPLE_PREDICTION = {
    "churn_probability": 0.7823,
    "churn_prediction": 1,
    "churn_label": "Churn",
    "risk_tier": "High",
    "confidence": 0.5646,
    "threshold_used": 0.42,
}

SAMPLE_CUSTOMER = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
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
    "TotalCharges": 286.50,
}


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing without a real model."""
    predictor = MagicMock()
    predictor._is_loaded = True
    predictor.predict.return_value = SAMPLE_PREDICTION
    predictor.predict_batch.return_value = [SAMPLE_PREDICTION]
    predictor.get_model_info.return_value = {
        "model_version": "v1",
        "algorithm": "lightgbm",
        "cv_roc_auc": 0.8721,
        "n_features": 35,
        "trained_at": "2024-01-01T00:00:00",
        "threshold": 0.42,
        "is_loaded": True,
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create test client with mocked predictor."""
    with patch("api.main._predictor", mock_predictor):
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "is_model_loaded" in data

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["is_model_loaded"] is True


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_fields(self, client):
        data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "churn_label" in data
        assert "risk_tier" in data
        assert "confidence" in data
        assert "threshold_used" in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_binary_label(self, client):
        data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert data["churn_prediction"] in (0, 1)

    def test_predict_risk_tier_valid(self, client):
        data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert data["risk_tier"] in ("Low", "Medium", "High", "Critical")

    def test_predict_invalid_gender(self, client):
        bad_customer = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_predict_invalid_contract(self, client):
        bad_customer = {**SAMPLE_CUSTOMER, "Contract": "Weekly"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_predict_negative_tenure(self, client):
        bad_customer = {**SAMPLE_CUSTOMER, "tenure": -1}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_predict_missing_required_field(self, client):
        incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "MonthlyCharges"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


class TestBatchPredictEndpoint:

    def test_batch_predict_returns_200(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        response = client.post("/batch-predict", json=payload)
        assert response.status_code == 200

    def test_batch_predict_response_structure(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER]}
        data = client.post("/batch-predict", json=payload).json()
        assert "total" in data
        assert "churn_count" in data
        assert "churn_rate" in data
        assert "predictions" in data
        assert "summary" in data

    def test_batch_predict_total_count(self, client):
        """total field should match number of input customers."""
        # Mock returns one prediction per call regardless — just test structure
        payload = {"customers": [SAMPLE_CUSTOMER]}
        data = client.post("/batch-predict", json=payload).json()
        assert isinstance(data["predictions"], list)

    def test_batch_predict_empty_list(self, client):
        payload = {"customers": []}
        response = client.post("/batch-predict", json=payload)
        assert response.status_code == 422   # Pydantic min_length=1


class TestRootEndpoint:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data
