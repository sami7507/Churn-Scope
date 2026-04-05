# ============================================================
# Makefile — Customer Churn Prediction System
# Developer convenience commands
# ============================================================

.PHONY: help setup train api test lint format docker-build docker-run clean predict

PYTHON := python3
CONFIG := config/config.yaml
PORT := 8000

# ── Help ──────────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo ""
	@echo "Customer Churn Prediction System — Available Commands"
	@echo "======================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
setup: ## Install all dependencies and create required directories
	@echo "Setting up project..."
	pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/features
	mkdir -p models/trained models/artifacts models/registry
	mkdir -p logs results notebooks/plots
	@echo "Setup complete. Run 'make train' to train the model."

# ── Training ──────────────────────────────────────────────────────────────
train: ## Train model using data from config (data/raw/telco_churn.csv)
	$(PYTHON) train_pipeline.py

train-synthetic: ## Train model using generated synthetic data (for demo)
	$(PYTHON) train_pipeline.py --synthetic --n-samples 7043

train-hpo: ## Train model with Optuna hyperparameter optimization (slow, better results)
	$(PYTHON) train_pipeline.py --synthetic --optimize-hpo

train-xgboost: ## Train using XGBoost instead of LightGBM
	$(PYTHON) train_pipeline.py --synthetic --algorithm xgboost

# ── API ───────────────────────────────────────────────────────────────────
api: ## Start the FastAPI server in development mode (hot reload)
	uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --reload

api-prod: ## Start the API in production mode (no reload, 4 workers)
	uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

# ── Inference ─────────────────────────────────────────────────────────────
predict: ## Run batch predictions on data/raw/telco_churn.csv
	$(PYTHON) predict.py --input data/raw/telco_churn.csv

predict-file: ## Run batch predictions on INPUT file: make predict-file INPUT=path/to/file.csv
	$(PYTHON) predict.py --input $(INPUT)

# ── Tests ─────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage report
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

test-unit: ## Run only unit tests (fast)
	pytest tests/test_preprocessor.py tests/test_predictor.py -v

test-api: ## Run only API integration tests
	pytest tests/test_api.py -v

test-fast: ## Run tests without coverage (fastest)
	pytest tests/ -v -x

# ── Code Quality ──────────────────────────────────────────────────────────
lint: ## Run Ruff linter
	ruff check src/ api/ tests/ train_pipeline.py predict.py

format: ## Auto-format code with Ruff
	ruff format src/ api/ tests/ train_pipeline.py predict.py

lint-fix: ## Auto-fix lint issues and format
	ruff check --fix src/ api/ tests/
	ruff format src/ api/ tests/

# ── Notebooks ─────────────────────────────────────────────────────────────
eda: ## Run EDA notebook/script
	cd notebooks && $(PYTHON) 01_eda.py

features: ## Run feature engineering analysis
	cd notebooks && $(PYTHON) 02_feature_engineering.py

modeling: ## Run modeling and evaluation analysis
	cd notebooks && $(PYTHON) 03_modeling.py

# ── Docker ────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build --target production -t churn-prediction:latest .

docker-run: ## Run Docker container (requires trained model in models/)
	docker run -p $(PORT):$(PORT) \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/logs:/app/logs \
		churn-prediction:latest

docker-up: ## Start full stack with docker-compose
	docker-compose up --build -d
	@echo "API running at http://localhost:$(PORT)"
	@echo "Docs at http://localhost:$(PORT)/docs"

docker-down: ## Stop docker-compose stack
	docker-compose down

docker-logs: ## Show API container logs
	docker-compose logs -f churn-api

# ── Utilities ─────────────────────────────────────────────────────────────
health: ## Check API health (must be running)
	curl -s http://localhost:$(PORT)/health | python3 -m json.tool

sample-predict: ## Send a sample prediction request to the running API
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"gender":"Female","SeniorCitizen":"0","Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":286.5}' \
		| python3 -m json.tool

clean: ## Remove generated artifacts (keep source code)
	rm -rf models/trained/*.joblib
	rm -rf models/artifacts/*.joblib
	rm -rf models/artifacts/evaluation_results.json
	rm -rf models/registry/model_registry.json
	rm -rf data/processed/* data/features/*
	rm -rf logs/*.log
	rm -rf results/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "Cleaned generated artifacts."

clean-all: clean ## Remove ALL generated files including raw data
	rm -rf data/raw/*
	@echo "Cleaned everything including data."
