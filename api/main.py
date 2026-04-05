"""
main.py
=======
FastAPI application for the Customer Churn Prediction System.

Endpoints:
  GET  /health           — Model status and metadata
  POST /predict          — Single customer churn prediction
  POST /batch-predict    — Batch customer churn prediction (up to 1000)
  GET  /docs             — Auto-generated Swagger UI (OpenAPI)

Production patterns used:
  - Lifespan context manager for model loading (FastAPI recommended approach)
  - Dependency injection for the predictor (testable, swappable)
  - Pydantic schemas for request/response validation
  - Request logging middleware with request IDs
  - Structured error handling with appropriate HTTP status codes
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections import Counter
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import load_config, get_logger
from src.predictor import ChurnPredictor
from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from api.middleware import RequestLoggingMiddleware

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------
_config: dict = {}
_predictor: ChurnPredictor = None


# ------------------------------------------------------------------
# Lifespan — runs at startup and shutdown
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; clean up on shutdown."""
    global _config, _predictor

    logger.info("Starting Churn Prediction API...")

    # Load config
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    _config = load_config(config_path)

    # Load model (singleton — loaded once, shared across all requests)
    try:
        _predictor = ChurnPredictor(_config)
        _predictor.load()
        logger.info("Model loaded successfully. API is ready.")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Run 'python train_pipeline.py' first to train and save a model.")
        # API starts in degraded mode — /predict returns 503
        _predictor = None

    yield   # App runs here

    # Shutdown
    logger.info("Shutting down Churn Prediction API.")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Production-grade REST API for customer churn prediction. "
        "Powered by LightGBM trained on Telco customer data. "
        "See /docs for interactive API documentation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
app.add_middleware(RequestLoggingMiddleware)


# ------------------------------------------------------------------
# Dependency injection
# ------------------------------------------------------------------

def get_predictor() -> ChurnPredictor:
    """Dependency that returns the loaded predictor or raises 503."""
    if _predictor is None or not _predictor._is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. The API is starting up or the model "
                "has not been trained yet. Run train_pipeline.py first."
            ),
        )
    return _predictor


PredictorDep = Annotated[ChurnPredictor, Depends(get_predictor)]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and model status.

    Returns model version, algorithm, training metrics, and readiness status.
    """
    if _predictor is None or not _predictor._is_loaded:
        return HealthResponse(
            status="degraded",
            model_version="none",
            algorithm=None,
            cv_roc_auc=None,
            n_features=None,
            trained_at=None,
            threshold=0.5,
            is_model_loaded=False,
        )

    info = _predictor.get_model_info()
    return HealthResponse(
        status="healthy",
        model_version=info.get("model_version", "unknown"),
        algorithm=info.get("algorithm"),
        cv_roc_auc=info.get("cv_roc_auc"),
        n_features=info.get("n_features"),
        trained_at=info.get("trained_at"),
        threshold=info.get("threshold", 0.5),
        is_model_loaded=True,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict churn for a single customer",
)
async def predict_single(
    customer: CustomerFeatures,
    predictor: PredictorDep,
):
    """
    Predict churn probability and risk tier for a single customer.

    **Input**: Customer feature record (all fields required except customerID)

    **Output**:
    - `churn_probability`: Score from 0 (safe) to 1 (very likely to churn)
    - `churn_prediction`: Binary label (0 = Stay, 1 = Churn)
    - `risk_tier`: Low / Medium / High / Critical
    - `confidence`: How certain the model is

    **Business use**: Flag `risk_tier=Critical` customers for immediate retention outreach.
    """
    try:
        customer_dict = customer.model_dump()
        result = predictor.predict(customer_dict)
        result["customer_id"] = customer.customerID
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction input error: {e}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs for details.",
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict churn for multiple customers",
)
async def predict_batch(
    request: BatchPredictRequest,
    predictor: PredictorDep,
):
    """
    Predict churn for a batch of up to 1000 customers in a single request.

    **Input**: List of customer feature records

    **Output**: Per-customer predictions + aggregate summary statistics

    **Business use**: Run nightly against your full customer base to generate
    a prioritized retention list.
    """
    try:
        customers_data = [c.model_dump() for c in request.customers]
        df = pd.DataFrame(customers_data)

        results = predictor.predict_batch(df)

        # Aggregate summary
        total = len(results)
        churn_count = sum(r["churn_prediction"] for r in results)
        tier_counts = Counter(r["risk_tier"] for r in results)

        response_items = [
            PredictionResponse(
                **r,
                customer_id=customers_data[i].get("customerID"),
            )
            for i, r in enumerate(results)
        ]

        return BatchPredictionResponse(
            total=total,
            churn_count=churn_count,
            churn_rate=round(churn_count / total, 4) if total > 0 else 0.0,
            predictions=response_items,
            summary={
                "risk_tiers": dict(tier_counts),
                "avg_churn_probability": round(
                    sum(r["churn_probability"] for r in results) / total, 4
                ),
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        )


@app.get("/", tags=["Root"])
async def root():
    """API root — redirects to documentation."""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


# ------------------------------------------------------------------
# Run directly (development)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    api_cfg = cfg.get("api", {})

    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=True,   # Set to False in production
        log_level="info",
    )
