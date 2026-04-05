#!/usr/bin/env bash
# ============================================================
# download_data.sh — Download Telco Churn dataset from Kaggle
# Usage: bash scripts/download_data.sh
#
# Prerequisites:
#   - Kaggle account: https://www.kaggle.com
#   - Kaggle API credentials at ~/.kaggle/kaggle.json
#   - kaggle CLI: pip install kaggle
# ============================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

DATASET="blastchar/telco-customer-churn"
OUTPUT_DIR="data/raw"
OUTPUT_FILE="$OUTPUT_DIR/telco_churn.csv"

info "Downloading Telco Customer Churn dataset from Kaggle..."

# Check kaggle CLI
if ! command -v kaggle &>/dev/null; then
    error "kaggle CLI not found. Install with: pip install kaggle"
fi

# Check credentials
KAGGLE_JSON="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    error "Kaggle credentials not found at $KAGGLE_JSON\n  Go to https://www.kaggle.com/settings → API → Create New Token"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download
info "Downloading dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR" --unzip

# The downloaded file is named WA_Fn-UseC_-Telco-Customer-Churn.csv
RAW_FILE="$OUTPUT_DIR/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if [ -f "$RAW_FILE" ]; then
    mv "$RAW_FILE" "$OUTPUT_FILE"
    info "Renamed to: $OUTPUT_FILE"
fi

# Verify
if [ -f "$OUTPUT_FILE" ]; then
    ROWS=$(wc -l < "$OUTPUT_FILE")
    info "Dataset downloaded: $OUTPUT_FILE ($ROWS rows including header)"
    info ""
    info "Now run: python train_pipeline.py"
    info "Or     : make train"
else
    error "Download failed. File not found at $OUTPUT_FILE"
fi
