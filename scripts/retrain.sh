#!/usr/bin/env bash
# ============================================================
# retrain.sh — Automated model retraining script
# Designed to be run by cron, CI/CD, or a scheduler.
#
# Usage:
#   bash scripts/retrain.sh                   # Use real data
#   bash scripts/retrain.sh --synthetic       # Use synthetic data
#   NOTIFY=1 bash scripts/retrain.sh          # + Slack notification
#
# Cron example (retrain every Sunday at 2am):
#   0 2 * * 0 /path/to/project/scripts/retrain.sh >> /var/log/churn_retrain.log 2>&1
# ============================================================

set -euo pipefail

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/retrain_$(date +%Y%m%d).log"
SYNTHETIC="${1:-}"

echo "=============================================="
echo "CHURN MODEL RETRAINING — $TIMESTAMP"
echo "=============================================="

cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create log directory
mkdir -p logs

# Run the training pipeline
echo "[$(date '+%H:%M:%S')] Starting training pipeline..."

if [ "$SYNTHETIC" = "--synthetic" ]; then
    python train_pipeline.py --synthetic --n-samples 7043 2>&1 | tee -a "$LOG_FILE"
else
    python train_pipeline.py 2>&1 | tee -a "$LOG_FILE"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] Training succeeded ✓"

    # Optional: Send Slack notification on success
    if [ "${NOTIFY:-0}" = "1" ] && [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"✅ Churn model retrained successfully at $TIMESTAMP\"}" \
            > /dev/null
    fi

    # Optional: Restart the API to load the new model
    # docker-compose restart churn-api
    # OR: kill -HUP $(pgrep -f "uvicorn api.main")

else
    echo "[$(date '+%H:%M:%S')] Training FAILED with exit code $EXIT_CODE"

    # Notify on failure
    if [ "${NOTIFY:-0}" = "1" ] && [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"🚨 Churn model retraining FAILED at $TIMESTAMP. Check logs: $LOG_FILE\"}" \
            > /dev/null
    fi

    exit 1
fi

echo "Log saved: $LOG_FILE"
