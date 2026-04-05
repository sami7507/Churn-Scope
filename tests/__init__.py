"""
tests/__init__.py
=================
Test Package — Customer Churn Prediction System

Test modules:
    test_preprocessor.py — Unit tests for ChurnPreprocessor
        - Split size correctness (train + val + test = total)
        - Feature count consistency across splits
        - No NaN values in processed output
        - Target is strictly binary (0/1)
        - Feature names returned correctly
        - Class weights are positive
        - Transform on unseen data (no refit)
        - Save/load pipeline round-trip produces identical output

    test_predictor.py    — Unit tests for ChurnPredictor inference engine
        - Output dict has all required keys
        - Probability is in [0.0, 1.0]
        - Prediction is binary (0 or 1)
        - Label matches prediction
        - Risk tier is one of Low/Medium/High/Critical
        - Confidence is in [0.0, 1.0]
        - Risk tier thresholds (all 4 tiers)
        - Batch prediction returns correct count
        - Model info dict has required keys
        - Threshold override changes prediction
        - Unloaded predictor raises RuntimeError

    test_api.py          — Integration tests for FastAPI endpoints
        - GET  /health   — 200, schema, status=healthy
        - POST /predict  — 200, schema, value ranges, validation errors
        - POST /batch-predict — 200, structure, empty list rejection
        - GET  /         — 200, docs link present

Run all tests:
    pytest tests/ -v

Run with coverage:
    pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

Run a single module:
    pytest tests/test_predictor.py -v
"""

# Shared test constants used across test modules

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

__all__ = ["LOW_RISK_CUSTOMER", "HIGH_RISK_CUSTOMER"]
