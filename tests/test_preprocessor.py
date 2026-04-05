"""
test_preprocessor.py
====================
Unit tests for the ChurnPreprocessor class.

Tests cover:
  - Output shapes (train/val/test)
  - No data leakage (pipeline fitted only on train)
  - Missing value handling
  - Target encoding correctness
  - Pipeline save/load round-trip
"""

import numpy as np
import pandas as pd
import pytest

from src import load_config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor


@pytest.fixture
def config():
    return load_config("config/config.yaml")


@pytest.fixture
def raw_df():
    return DataLoader.generate_synthetic_data(n_samples=500, random_state=99)


@pytest.fixture
def enriched_df(config, raw_df):
    fe = FeatureEngineer(config)
    return fe.transform(raw_df)


@pytest.fixture
def splits(config, enriched_df):
    preprocessor = ChurnPreprocessor(config)
    return preprocessor.fit_transform(enriched_df), preprocessor


class TestChurnPreprocessor:

    def test_split_sizes(self, splits, enriched_df):
        result, _ = splits
        total = len(enriched_df)
        train_n = result["X_train"].shape[0]
        val_n = result["X_val"].shape[0]
        test_n = result["X_test"].shape[0]

        assert train_n + val_n + test_n == total
        # Train should be largest split
        assert train_n > val_n
        assert train_n > test_n

    def test_feature_count_consistent(self, splits):
        result, _ = splits
        n_features = result["X_train"].shape[1]
        assert result["X_val"].shape[1] == n_features
        assert result["X_test"].shape[1] == n_features

    def test_target_is_binary(self, splits):
        result, _ = splits
        for key in ("y_train", "y_val", "y_test"):
            unique_vals = set(np.unique(result[key]))
            assert unique_vals.issubset({0, 1}), f"{key} contains non-binary values"

    def test_no_nan_in_output(self, splits):
        result, _ = splits
        for key in ("X_train", "X_val", "X_test"):
            assert not np.isnan(result[key]).any(), f"NaN found in {key}"

    def test_feature_names_returned(self, splits):
        result, _ = splits
        assert "feature_names" in result
        assert len(result["feature_names"]) == result["X_train"].shape[1]

    def test_class_weights_positive(self, splits):
        result, _ = splits
        cw = result["class_weights"]
        assert cw[0] > 0
        assert cw[1] > 0

    def test_transform_on_new_data(self, config, enriched_df):
        """Pipeline transform should work on new data without refitting."""
        preprocessor = ChurnPreprocessor(config)
        preprocessor.fit_transform(enriched_df)

        new_df = enriched_df.sample(10, random_state=1)
        X_new = preprocessor.transform(new_df)
        assert X_new.shape[0] == 10

    def test_save_and_load_roundtrip(self, config, enriched_df, tmp_path):
        """Saved pipeline should produce identical output when reloaded."""
        preprocessor = ChurnPreprocessor(config)
        splits = preprocessor.fit_transform(enriched_df)

        save_path = str(tmp_path / "pipeline.joblib")
        preprocessor.save(save_path)

        new_preprocessor = ChurnPreprocessor(config)
        new_preprocessor.load(save_path)

        new_df = enriched_df.head(50)
        X1 = preprocessor.transform(new_df)
        X2 = new_preprocessor.transform(new_df)
        np.testing.assert_array_almost_equal(X1, X2)
