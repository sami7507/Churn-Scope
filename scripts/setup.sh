#!/usr/bin/env bash
# ============================================================
# setup.sh — One-command project setup
# Usage: bash scripts/setup.sh
# ============================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

info "=============================================="
info "  Churn Prediction System — Project Setup"
info "=============================================="

# ── Check Python version ──────────────────────────────────────
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt "$REQUIRED_MAJOR" ] || ([ "$MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$MINOR" -lt "$REQUIRED_MINOR" ]); then
    error "Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required. Found: $PYTHON_VERSION"
fi
info "Python $PYTHON_VERSION ✓"

# ── Create virtual environment ────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv .venv
    info "Virtual environment created at .venv/"
else
    warn "Virtual environment already exists. Skipping creation."
fi

# ── Activate venv ─────────────────────────────────────────────
info "Activating virtual environment..."
source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────
info "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed ✓"

# ── Create directory structure ────────────────────────────────
info "Creating project directories..."
mkdir -p data/raw data/processed data/features
mkdir -p models/trained models/artifacts models/registry
mkdir -p logs results notebooks/plots

# Create .gitkeep placeholders so empty dirs are tracked
touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep
touch logs/.gitkeep results/.gitkeep

info "Directories created ✓"

# ── Copy .env if it doesn't exist ────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    info "Created .env from .env.example (review and update values)"
else
    warn ".env already exists. Skipping."
fi

# ── Train model with synthetic data ──────────────────────────
read -p "$(echo -e "${YELLOW}Train model now with synthetic data? [y/N]:${NC} ")" TRAIN_NOW
if [[ "$TRAIN_NOW" =~ ^[Yy]$ ]]; then
    info "Training model with synthetic data..."
    python train_pipeline.py --synthetic --n-samples 7043
    info "Model trained successfully ✓"
else
    warn "Skipping training. Run 'make train-synthetic' or 'python train_pipeline.py --synthetic' later."
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
info "=============================================="
info "  Setup Complete!"
info "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv   :  source .venv/bin/activate"
echo "  2. Start API       :  make api"
echo "  3. Run tests       :  make test"
echo "  4. View all cmds   :  make help"
echo "  5. API docs        :  http://localhost:8000/docs"
echo ""
