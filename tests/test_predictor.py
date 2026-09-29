"""Unit tests for inference engine."""
import pytest
import numpy as np

def test_shap_risk_tier():
    """Test that risk tier boundaries are correct."""
    from src.predictor import ChurnPredictor
    assert ChurnPredictor._get_risk_tier(0.80) == "Critical"
    assert ChurnPredictor._get_risk_tier(0.60) == "High"
    assert ChurnPredictor._get_risk_tier(0.40) == "Medium"
    assert ChurnPredictor._get_risk_tier(0.20) == "Low"

def test_format_output():
    """Test prediction output structure."""
    from src.predictor import ChurnPredictor
    from src import load_config
    cfg = load_config()
    p = ChurnPredictor(cfg)
    p.optimal_threshold = 0.5
    result = p._format_output(0.75, 1)
    assert result["churn_probability"] == 0.75
    assert result["churn_label"] == "Churn"
    assert "risk_tier" in result
    assert "confidence" in result
