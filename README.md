<div align="center">

# 🔮 ChurnScope

### Production-Grade Customer Churn Prediction System

[![CI/CD](https://github.com/sami7507/churnscope/actions/workflows/ci.yml/badge.svg)](https://github.com/sami7507/churnscope/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-brightgreen)](https://lightgbm.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*End-to-end ML system: feature engineering → LightGBM training → FastAPI serving → SHAP explainability → interactive dashboard*

[Live Demo](#quick-start) · [API Docs](#api-reference) · [Architecture](#architecture) · [Contact](#contact)

</div>

---

## 🎯 What is ChurnScope?

**ChurnScope** is a production-grade machine learning system that predicts which telecom customers are at risk of churning — and *why*. Built following MLOps best practices, it covers the complete lifecycle from raw data to deployed REST API to an interactive analytics dashboard.

| What it does | How |
|---|---|
| Predicts churn probability per customer | LightGBM with stratified CV + threshold optimization |
| Explains every prediction | SHAP feature attribution (global + per-customer) |
| Serves predictions at < 5ms latency | FastAPI singleton inference engine |
| Visualizes risk in real time | Streamlit dark-theme HUD dashboard |
| Ships to production | Docker multi-stage build + GitHub Actions CI/CD |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ChurnScope Pipeline                         │
├─────────────┬──────────────┬──────────────┬────────────┬───────────┤
│  DataLoader │  FeatureEng  │ Preprocessor │  Trainer   │ Evaluator │
│  CSV/Parquet│  22 features │  Fit once    │  LightGBM  │  SHAP +   │
│  + schema   │  tenure bands│  sklearn pipe│  XGBoost   │  ROC/PR   │
│  validation │  charge ratio│  save joblib │  Optuna HPO│  threshld │
└──────┬──────┴──────────────┴──────────────┴─────┬──────┴───────────┘
       │                                           │
       ▼                                           ▼
┌─────────────────────────────┐     ┌──────────────────────────────┐
│        FastAPI Server       │     │     Streamlit Dashboard       │
│  POST /predict              │     │  ⟢ Prediction page           │
│  POST /batch-predict        │◄────│  ◈ SHAP Explainability       │
│  GET  /health               │     │  ▸ Model Metrics             │
│  GET  /metrics              │     │  ROC · CM · Feature Imp.     │
│  Middleware: req ID + logs  │     │  Live session history         │
└──────────────┬──────────────┘     └──────────────────────────────┘
               │
       ┌───────▼───────┐
       │  Docker Image  │
       │  2-stage build │
       │  non-root user │
       │  health check  │
       └───────────────┘
```

---

## 📊 Model Performance

Trained on the Telco Customer Churn dataset (7,043 customers):

| Metric | Score |
|---|---|
| ROC-AUC | **0.856** |
| F1 Score | **0.764** |
| Precision | **78.6%** |
| Recall | **74.3%** |
| PR-AUC | 0.712 |
| Optimal Threshold | 0.263 (F1-optimized) |

**Business impact** (test set estimate):
- Churn coverage: **74.3%** of actual churners identified
- Targeting efficiency: **78.6%** of flagged customers are true churners  
- Estimated net benefit: **~$8,400** per 1,409 customers evaluated

---

## 🗂 Project Structure

```
churnscope/
├── src/
│   ├── __init__.py              # load_config() + get_logger() factory
│   ├── data_loader.py           # Multi-source ingestion + schema validation
│   ├── feature_engineering.py  # 22 domain features (tenure bands, ratios, flags)
│   ├── preprocessor.py         # Fit/transform sklearn pipeline (impute→encode→scale)
│   ├── trainer.py               # LightGBM/XGBoost + 5-fold CV + Optuna HPO
│   ├── evaluator.py             # AUC, F1, threshold optimization, SHAP, business ROI
│   └── predictor.py             # Singleton inference engine (<5ms latency)
├── api/
│   ├── main.py                  # FastAPI app: /predict, /batch-predict, /health
│   ├── schemas.py               # Pydantic v2 request/response validation
│   └── middleware.py            # Request ID injection + latency logging
├── config/
│   ├── config.yaml              # Single source of truth for all settings
│   └── logging.yaml             # Structured rotating file + console logging
├── data/
│   ├── raw/                     # Source CSV (immutable)
│   ├── processed/               # Cleaned split outputs
│   └── features/                # Engineered feature matrices
├── models/
│   ├── trained/                 # Versioned .joblib model files
│   ├── artifacts/               # Preprocessing pipeline + evaluation results
│   └── registry/                # model_registry.json (all versions, rollback)
├── tests/
│   ├── test_preprocessor.py     # Unit tests: preprocessing pipeline
│   ├── test_predictor.py        # Unit tests: inference engine
│   └── test_api.py              # Integration tests: API endpoints
├── .github/workflows/
│   └── ci.yml                   # Lint → Test → Docker build → Push → Deploy
├── streamlit_app.py             # Interactive analytics dashboard (3-page HUD)
├── train_pipeline.py            # End-to-end orchestrator (single entry point)
├── predict.py                   # CLI batch inference script
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Local orchestration
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sami7507/churnscope.git
cd churnscope
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# With real data (place CSV at data/raw/telco_churn.csv first)
python train_pipeline.py

# Or use synthetic data (demo/testing — no real data needed)
python train_pipeline.py --synthetic
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs auto-generated at: **http://localhost:8000/docs**

### 4. Launch the Dashboard

```bash
# In a second terminal
streamlit run streamlit_app.py
```

Dashboard opens at: **http://localhost:8501**

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t churnscope:latest .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  churnscope:latest
```

---

## 📡 API Reference

### `POST /predict` — Single Customer

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5
  }'
```

**Response:**
```json
{
  "churn_probability": 0.689,
  "churn_prediction": 1,
  "churn_label": "Churn",
  "risk_tier": "High",
  "confidence": 0.378,
  "threshold_used": 0.2631,
  "customer_id": "CUST-12345"
}
```

### `POST /batch-predict` — Up to 1000 Customers

```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'
```

### `GET /health` — Model Status

```bash
curl http://localhost:8000/health
```

### CLI Batch Inference

```bash
python predict.py --input data/raw/customers.csv --output results/scores.csv
```

---

## 🧬 Feature Engineering

ChurnScope engineers **22 domain features** from the 20 raw inputs:

| Category | Features |
|---|---|
| Tenure bands | `tenure_band`, `is_new_customer` (<6mo), `is_loyal_customer` (>24mo), `log_tenure` |
| Charge ratios | `monthly_charge_per_tenure`, `charge_discrepancy`, `is_high_value`, `log_monthly_charges` |
| Service count | `total_services`, `service_penetration`, `has_internet`, `has_fiber` |
| Interactions | `charge_tenure_risk` (high charges × new customer), `stickiness_score`, `high_risk_flag` |
| Contract flags | `is_month_to_month`, `is_long_term_contract`, `is_electronic_check`, `is_auto_payment`, `highest_churn_risk_profile` |

Key insight: `charge_tenure_risk = MonthlyCharges / (tenure + 1)` is consistently the top SHAP feature — it captures the canonical churn profile (high spend, low commitment) in a single number.

---

## 🔬 MLOps Highlights

| Practice | Implementation |
|---|---|
| **No train/test leakage** | Preprocessing pipeline fit only on train split, saved as joblib |
| **Reproducible training** | Fixed random seeds, versioned artifacts, model registry JSON |
| **Threshold optimization** | F1-optimal threshold found via precision-recall curve (not hardcoded 0.5) |
| **Singleton inference** | Model loaded once at startup, shared across all API requests |
| **Model versioning** | Every training run gets a timestamped file + registry entry |
| **Structured logging** | Request IDs, latency, rotating file handlers |
| **CI/CD** | GitHub Actions: lint → test → Docker build → smoke test → registry push |
| **Health checks** | `/health` endpoint + Docker HEALTHCHECK + Compose healthcheck |

---

## 🧪 Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Unit tests only
pytest tests/test_preprocessor.py tests/test_predictor.py -v

# API integration tests (requires running server)
pytest tests/test_api.py -v
```

---

## ⚙️ Configuration

All settings live in `config/config.yaml`. Key sections:

```yaml
model:
  algorithm: "lightgbm"      # or "xgboost"
  
threshold:
  strategy: "f1"             # "f1" | "recall" | "custom_cost"
  cost_matrix:
    fn_cost: 5               # Cost of missing a churner
    fp_cost: 1               # Cost of wrong retention offer

optuna:
  enabled: false             # Set true for hyperparameter search
  n_trials: 50
```

---

## 📈 Dashboard Pages

The Streamlit dashboard (`streamlit_app.py`) has three pages:

**⟢ Prediction** — Input panel for all 20 customer fields, real-time churn gauge, risk badge, session log, and 4 live charts (probability bar, risk tier donut, probability timeline, charges vs churn scatter).

**◈ Explainability** — SHAP horizontal bar chart showing which features pushed the prediction toward or away from churn, top-8 contribution breakdown, auto-generated retention recommendations.

**▸ Model Metrics** — ROC curve (AUC=0.856), Precision-Recall curve, confusion matrix, global feature importance, model registry metadata.

---

## 📬 Contact

**Md Sami Ahmad**  
📧 sami@757007@gmail.com  
🔗 [LinkedIn — sami7507](https://linkedin.com/in/sami7507)  
🐙 [GitHub — sami7507](https://github.com/sami7507)

---

<div align="center">
<sub>Built with Python · FastAPI · LightGBM · SHAP · Streamlit · Docker</sub>
</div>
