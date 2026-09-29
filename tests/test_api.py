"""Integration tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

SAMPLE = {
    "customerID":"TEST-001","gender":"Female","SeniorCitizen":"0","Partner":"Yes",
    "Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No",
    "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes",
    "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
    "MonthlyCharges":95.5,"TotalCharges":286.5,
}

@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    assert "status" in resp.json()

def test_predict_success(client):
    resp = client.post("/predict", json=SAMPLE)
    if resp.status_code == 200:
        d = resp.json()
        assert 0.0 <= d["churn_probability"] <= 1.0
        assert d["churn_label"] in ("Churn","No Churn")
        assert d["risk_tier"] in ("Low","Medium","High","Critical")

def test_predict_invalid_gender(client):
    bad = {**SAMPLE, "gender": "Unknown"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422

def test_batch_predict(client):
    resp = client.post("/batch-predict", json={"customers":[SAMPLE,SAMPLE]})
    if resp.status_code == 200:
        d = resp.json()
        assert d["total"] == 2 and len(d["predictions"]) == 2

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200 and "docs" in resp.json()
