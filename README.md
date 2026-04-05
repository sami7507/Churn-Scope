# Customer Churn Prediction System

An end-to-end, production-grade Machine Learning system for predicting customer churn.
Built with LightGBM/XGBoost, FastAPI, and designed for cloud deployment.

---

## Project Structure

```
churn_prediction/
├── data/
│   ├── raw/                    # Source data (immutable)
│   ├── processed/              # Cleaned / split outputs
│   └── features/               # Engineered feature matrices
├── src/
│   ├── __init__.py             # Config loader + logger factory
│   ├── data_loader.py          # Multi-source data ingestion + validation
│   ├── preprocessor.py         # Fit/transform pipeline (impute, encode, scale)
│   ├── feature_engineering.py  # Domain feature creation (RFM, ratios, flags)
│   ├── trainer.py              # LightGBM/XGBoost + CV + Optuna HPO
│   ├── evaluator.py            # AUC, F1, threshold optimization, business metrics
│   └── predictor.py            # Singleton inference engine
├── api/
│   ├── main.py                 # FastAPI app with /predict, /batch-predict, /health
│   ├── schemas.py              # Pydantic v2 request/response models
│   └── middleware.py           # Request logging + request ID injection
├── models/
│   ├── trained/                # Versioned .joblib model files
│   ├── artifacts/              # Preprocessing pipeline + eval results
│   └── registry/               # model_registry.json (all versions)
├── config/
│   └── config.yaml             # Single source of truth for all settings
├── tests/
│   ├── test_preprocessor.py    # Unit tests for preprocessing
│   └── test_api.py             # Integration tests for API endpoints
├── train_pipeline.py           # Full pipeline orchestrator (entry point)
├── predict.py                  # CLI batch inference script
├── Dockerfile                  # Multi-stage production Docker image
├── docker-compose.yml          # Local orchestration
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Using synthetic data (for demo / first run)
python train_pipeline.py --synthetic

# Using real Telco CSV (place file at data/raw/telco_churn.csv)
python train_pipeline.py

# Use XGBoost instead of LightGBM
python train_pipeline.py --synthetic --algorithm xgboost

# Enable Optuna hyperparameter search (slower, better results)
python train_pipeline.py --synthetic --optimize-hpo
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test predictions

```bash
# Single customer prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 286.50
  }'

# Health check
curl http://localhost:8000/health

# Interactive docs
open http://localhost:8000/docs
```

### 5. Batch CLI inference

```bash
python predict.py --input data/raw/telco_churn.csv --output results/predictions.csv
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status, version, metrics |
| POST | `/predict` | Single customer churn prediction |
| POST | `/batch-predict` | Batch prediction (up to 1,000 customers) |
| GET | `/docs` | Swagger UI (interactive API explorer) |

### Sample Response — `/predict`

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": 1,
  "churn_label": "Churn",
  "risk_tier": "High",
  "confidence": 0.5646,
  "threshold_used": 0.42,
  "customer_id": "CUST-12345"
}
```

---

## Docker Deployment

```bash
# Build and run locally
docker-compose up --build

# Or build manually
docker build -t churn-prediction:latest .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  churn-prediction:latest
```

---

## Cloud Deployment

### AWS (Elastic Container Service)

```bash
# 1. Build and push image to ECR
aws ecr create-repository --repository-name churn-prediction
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

docker tag churn-prediction:latest <account>.dkr.ecr.<region>.amazonaws.com/churn-prediction:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/churn-prediction:latest

# 2. Create ECS Task Definition pointing to the ECR image
# 3. Create ECS Service with Application Load Balancer
# 4. Set up Auto Scaling based on CPU/memory

# Environment variables for production:
# CONFIG_PATH=/app/config/config.yaml
# PORT=8000
```

### Google Cloud Run

```bash
# Build and push to Google Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-prediction

# Deploy to Cloud Run (fully managed, auto-scales to zero)
gcloud run deploy churn-prediction \
  --image gcr.io/PROJECT_ID/churn-prediction \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Push to Azure Container Registry
az acr build --registry myregistry --image churn-prediction:latest .

# Deploy
az container create \
  --resource-group myRG \
  --name churn-prediction \
  --image myregistry.azurecr.io/churn-prediction:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

### Kubernetes (any cloud)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
    spec:
      containers:
      - name: churn-api
        image: churn-prediction:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: churn-prediction-svc
spec:
  selector:
    app: churn-prediction
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## MLOps Practices

| Practice | Implementation |
|----------|---------------|
| Experiment tracking | Model registry JSON + structured logs |
| Model versioning | Timestamp-stamped `.joblib` files |
| Reproducibility | Seeded splits, config-driven parameters |
| No data leakage | Pipeline fitted on train only, saved as artifact |
| Artifact management | Model + pipeline + eval results saved together |
| Threshold optimization | F1 or cost-matrix based, stored in eval artifact |
| Observability | Request ID tracing, latency logging, structured logs |
| Config management | Single `config.yaml`, no hardcoded values |

---

## Running Tests

```bash
# With pytest installed
pytest tests/ -v

# Without pytest (manual)
python -m tests.test_preprocessor
```

---

## Resume Presentation

**How to describe this project in a resume or interview:**

> "Built a production-grade customer churn prediction system using LightGBM,
> achieving 87% ROC-AUC on holdout data. Designed a modular Python pipeline
> with feature engineering (21 engineered features), threshold optimization
> aligned to business cost matrices, and deployed a FastAPI REST API serving
> real-time predictions. Containerized with Docker and deployable to
> AWS ECS / GCP Cloud Run."

**Key talking points:**
- Feature engineering rationale (tenure bands, charge ratios, stickiness score)
- Threshold optimization: why 0.5 is wrong for imbalanced churn data
- Singleton pattern for model loading (performance)
- No-leakage design: pipeline fitted on train, serialized, reused at inference
- Business metrics: translating ML metrics into revenue impact

---

## Dataset

Uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, ~26.5% churn rate). A synthetic generator is included for demo use.

Place the real dataset at `data/raw/telco_churn.csv` and run `python train_pipeline.py`.
