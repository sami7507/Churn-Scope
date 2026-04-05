"""
api/__init__.py
===============
API Package — Customer Churn Prediction System

Exports the FastAPI application instance so it can be referenced as:
    uvicorn api.main:app

Package contents:
    main.py       — FastAPI app with /predict, /batch-predict, /health routes
    schemas.py    — Pydantic v2 request/response models
    middleware.py — Request logging + request ID injection middleware
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("churn-prediction")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]
