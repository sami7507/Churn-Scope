# ============================================================
# ChurnScope — Dockerfile
# Multi-stage build for minimal production image
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Production ----
FROM python:3.11-slim AS production
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY train_pipeline.py predict.py ./
RUN mkdir -p data/raw data/processed data/features models/trained models/artifacts models/registry logs && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CONFIG_PATH=/app/config/config.yaml PORT=8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
