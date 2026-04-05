# ============================================================
# Customer Churn Prediction System — Dockerfile
# Multi-stage build for minimal production image size
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Production Image ----
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY train_pipeline.py .
COPY predict.py .

# Create required directories with correct ownership
RUN mkdir -p data/raw data/processed data/features \
             models/trained models/artifacts models/registry \
             logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config/config.yaml \
    PORT=8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
