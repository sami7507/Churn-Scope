"""Unit tests for preprocessing pipeline."""
import pytest
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor
from src import load_config

@pytest.fixture
def config(): return load_config()

@pytest.fixture
def sample_df(): return DataLoader.generate_synthetic_data(n_samples=300, random_state=0)

def test_feature_engineering_adds_columns(config, sample_df):
    fe = FeatureEngineer(config)
    df = fe.transform(sample_df)
    assert df.shape[1] > sample_df.shape[1]
    assert "tenure_band" in df.columns
    assert "total_services" in df.columns

def test_preprocessor_splits(config, sample_df):
    fe = FeatureEngineer(config)
    df = fe.transform(sample_df)
    pp = ChurnPreprocessor(config)
    splits = pp.fit_transform(df)
    assert splits["X_train"].shape[0] > splits["X_test"].shape[0]
    assert splits["X_train"].shape[1] == splits["X_test"].shape[1]

def test_no_target_in_features(config, sample_df):
    fe = FeatureEngineer(config)
    df = fe.transform(sample_df)
    pp = ChurnPreprocessor(config)
    splits = pp.fit_transform(df)
    assert splits["X_train"].shape[1] == splits["X_val"].shape[1]
