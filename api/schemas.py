"""
schemas.py — Pydantic v2 request/response models for the ChurnScope API.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

class CustomerFeatures(BaseModel):
    customerID:       Optional[str]  = Field(None)
    gender:           str            = Field(..., description="Male or Female")
    SeniorCitizen:    str            = Field(..., description="0 or 1")
    Partner:          str            = Field(..., description="Yes or No")
    Dependents:       str            = Field(..., description="Yes or No")
    tenure:           int            = Field(..., ge=0, le=120)
    PhoneService:     str            = Field(...)
    MultipleLines:    str            = Field(...)
    InternetService:  str            = Field(...)
    OnlineSecurity:   str            = Field(...)
    OnlineBackup:     str            = Field(...)
    DeviceProtection: str            = Field(...)
    TechSupport:      str            = Field(...)
    StreamingTV:      str            = Field(...)
    StreamingMovies:  str            = Field(...)
    Contract:         str            = Field(...)
    PaperlessBilling: str            = Field(...)
    PaymentMethod:    str            = Field(...)
    MonthlyCharges:   float          = Field(..., ge=0.0, le=500.0)
    TotalCharges:     float          = Field(..., ge=0.0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"Male","Female"}:
            raise ValueError("gender must be Male or Female")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in {"Month-to-month","One year","Two year"}:
            raise ValueError("Invalid Contract value")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        if v not in {"DSL","Fiber optic","No"}:
            raise ValueError("Invalid InternetService value")
        return v

    class Config:
        json_schema_extra = {"example": {
            "customerID":"CUST-12345","gender":"Female","SeniorCitizen":"0",
            "Partner":"Yes","Dependents":"No","tenure":3,"PhoneService":"Yes",
            "MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No",
            "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
            "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
            "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
            "MonthlyCharges":95.5,"TotalCharges":286.5,
        }}

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int
    churn_label:       str
    risk_tier:         str
    confidence:        float
    threshold_used:    float
    customer_id:       Optional[str] = None

class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)

class BatchPredictionResponse(BaseModel):
    total:       int
    churn_count: int
    churn_rate:  float
    predictions: List[PredictionResponse]
    summary:     Dict[str, Any]

class HealthResponse(BaseModel):
    status:         str
    model_version:  str
    algorithm:      Optional[str]
    cv_roc_auc:     Optional[float]
    n_features:     Optional[int]
    trained_at:     Optional[str]
    threshold:      float
    is_model_loaded: bool

class ErrorResponse(BaseModel):
    error:       str
    detail:      Optional[str] = None
    status_code: int
