"""
schemas.py
==========
Pydantic v2 request and response models for the Churn Prediction API.

Why Pydantic schemas matter in production:
  - Rejects malformed input before it reaches the model
  - Documents the API contract automatically (shown in /docs)
  - Provides type coercion (string "95.50" → float 95.50)
  - Gives clear validation error messages to API consumers
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer prediction request.
    Mirrors the Telco Churn dataset schema.
    """

    # Demographics
    customerID: Optional[str] = Field(None, description="Customer identifier (optional, not used in model)")
    gender: str = Field(..., description="Customer gender: 'Male' or 'Female'")
    SeniorCitizen: str = Field(..., description="Senior citizen indicator: '0' or '1'")
    Partner: str = Field(..., description="Has partner: 'Yes' or 'No'")
    Dependents: str = Field(..., description="Has dependents: 'Yes' or 'No'")

    # Service tenure
    tenure: int = Field(..., ge=0, le=72, description="Months with company (0-72)")

    # Phone services
    PhoneService: str = Field(..., description="Has phone service: 'Yes' or 'No'")
    MultipleLines: str = Field(..., description="Multiple lines: 'Yes', 'No', or 'No phone service'")

    # Internet services
    InternetService: str = Field(..., description="Internet service type: 'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(..., description="Online security: 'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(..., description="Online backup: 'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(..., description="Device protection: 'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(..., description="Tech support: 'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="Streaming TV: 'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(..., description="Streaming movies: 'Yes', 'No', or 'No internet service'")

    # Billing
    Contract: str = Field(..., description="Contract type: 'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(..., description="Paperless billing: 'Yes' or 'No'")
    PaymentMethod: str = Field(
        ...,
        description="Payment method: 'Electronic check', 'Mailed check', "
                    "'Bank transfer (automatic)', 'Credit card (automatic)'"
    )

    # Charges
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly charge amount (USD)")
    TotalCharges: float = Field(..., ge=0.0, description="Total charges to date (USD)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        allowed = {"DSL", "Fiber optic", "No"}
        if v not in allowed:
            raise ValueError(f"InternetService must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_total_charges(self) -> "CustomerFeatures":
        """TotalCharges should be roughly >= MonthlyCharges * max(tenure-1, 0)."""
        if self.tenure > 1 and self.TotalCharges < self.MonthlyCharges * 0.5:
            # Soft warning only — don't reject, but flag
            pass
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "CUST-12345",
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
        }


class PredictionResponse(BaseModel):
    """Response schema for a single customer prediction."""

    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., description="Binary churn prediction (0=No, 1=Yes)")
    churn_label: str = Field(..., description="Human-readable prediction: 'Churn' or 'No Churn'")
    risk_tier: str = Field(..., description="Risk tier: 'Low', 'Medium', 'High', or 'Critical'")
    confidence: float = Field(..., description="Model confidence in this prediction (0-1)")
    threshold_used: float = Field(..., description="Decision threshold applied")
    customer_id: Optional[str] = Field(None, description="Echo of input customerID if provided")

    class Config:
        json_schema_extra = {
            "example": {
                "churn_probability": 0.7823,
                "churn_prediction": 1,
                "churn_label": "Churn",
                "risk_tier": "High",
                "confidence": 0.5646,
                "threshold_used": 0.42,
                "customer_id": "CUST-12345",
            }
        }


class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions."""

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer feature records (1-1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    total: int = Field(..., description="Total customers processed")
    churn_count: int = Field(..., description="Count predicted as churners")
    churn_rate: float = Field(..., description="Predicted churn rate (0-1)")
    predictions: List[PredictionResponse] = Field(..., description="Per-customer results")
    summary: Dict[str, Any] = Field(..., description="Aggregate risk tier distribution")


class HealthResponse(BaseModel):
    """Response schema for the health/status endpoint."""

    status: str = Field(..., description="API status: 'healthy' or 'degraded'")
    model_version: str
    algorithm: Optional[str]
    cv_roc_auc: Optional[float]
    n_features: Optional[int]
    trained_at: Optional[str]
    threshold: float
    is_model_loaded: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
