"""
main.py — FastAPI application for ChurnScope.

Endpoints:
  GET  /health         — Model status and metadata
  POST /predict        — Single customer churn prediction
  POST /batch-predict  — Batch prediction (up to 1000)
  GET  /docs           — Swagger UI
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from collections import Counter
from typing import Annotated
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src import load_config, get_logger
from src.predictor import ChurnPredictor
from api.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchPredictRequest, BatchPredictionResponse, HealthResponse,
)
from api.middleware import RequestLoggingMiddleware

logger = get_logger(__name__)
_config: dict = {}
_predictor: ChurnPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _predictor
    logger.info("Starting ChurnScope API...")
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    _config = load_config(config_path)
    try:
        _predictor = ChurnPredictor(_config)
        _predictor.load()
        logger.info("Model loaded. API ready.")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Run: python train_pipeline.py --synthetic")
        _predictor = None
    yield
    logger.info("Shutting down ChurnScope API.")

app = FastAPI(
    title="ChurnScope Prediction API",
    description="Production ML API for customer churn prediction. Powered by LightGBM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)

def get_predictor() -> ChurnPredictor:
    if _predictor is None or not _predictor._is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train_pipeline.py first.",
        )
    return _predictor

PredictorDep = Annotated[ChurnPredictor, Depends(get_predictor)]

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if _predictor is None or not _predictor._is_loaded:
        return HealthResponse(status="degraded", model_version="none",
                              algorithm=None, cv_roc_auc=None, n_features=None,
                              trained_at=None, threshold=0.5, is_model_loaded=False)
    info = _predictor.get_model_info()
    return HealthResponse(status="healthy",
                          model_version=info.get("model_version","unknown"),
                          algorithm=info.get("algorithm"),
                          cv_roc_auc=info.get("cv_roc_auc"),
                          n_features=info.get("n_features"),
                          trained_at=info.get("trained_at"),
                          threshold=info.get("threshold", 0.5),
                          is_model_loaded=True)

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"],
          summary="Predict churn for a single customer")
async def predict_single(customer: CustomerFeatures, predictor: PredictorDep):
    try:
        result = predictor.predict(customer.model_dump())
        result["customer_id"] = customer.customerID
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Input error: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"],
          summary="Predict churn for multiple customers (up to 1000)")
async def predict_batch(request: BatchPredictRequest, predictor: PredictorDep):
    try:
        customers_data = [c.model_dump() for c in request.customers]
        results = predictor.predict_batch(pd.DataFrame(customers_data))
        total = len(results)
        churn_count = sum(r["churn_prediction"] for r in results)
        tier_counts = Counter(r["risk_tier"] for r in results)
        return BatchPredictionResponse(
            total=total,
            churn_count=churn_count,
            churn_rate=round(churn_count / total, 4) if total > 0 else 0.0,
            predictions=[PredictionResponse(**r, customer_id=customers_data[i].get("customerID"))
                         for i, r in enumerate(results)],
            summary={"risk_tiers": dict(tier_counts),
                     "avg_churn_probability": round(sum(r["churn_probability"] for r in results)/total, 4)},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

@app.get("/", tags=["Root"])
async def root():
    return {"message": "ChurnScope Prediction API", "docs": "/docs",
            "health": "/health", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    uvicorn.run("api.main:app", host=api_cfg.get("host","0.0.0.0"),
                port=api_cfg.get("port",8000), reload=True, log_level="info")
