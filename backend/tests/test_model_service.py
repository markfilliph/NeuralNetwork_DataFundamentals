"""Tests for machine learning model service."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.model_service import ModelService, ModelTrainingConfig, ModelInfo
from backend.core.exceptions import ValidationError


class TestModelService:
    """Test suite for ModelService."""

    def setup_method(self):
        """Set up test environment."""
        self.model_service = ModelService()
        
        # Create simple test dataset
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        self.y = pd.Series(2 * self.X['feature1'] + 3 * self.X['feature2'] + np.random.normal(0, 0.1, 100))

    def test_train_linear_regression(self):
        """Test linear regression training."""
        config = ModelTrainingConfig(
            model_type="linear_regression",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42
        )
        
        model_info = self.model_service.train_model(
            self.X, self.y, config, "test-user", "test-model"
        )
        
        assert isinstance(model_info, ModelInfo)
        assert model_info.model_type == "linear_regression"
        assert model_info.status == "trained"
        assert model_info.feature_columns == ["feature1", "feature2"]

    def test_model_prediction(self):
        """Test model prediction."""
        config = ModelTrainingConfig(
            model_type="linear_regression",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        model_info = self.model_service.train_model(
            self.X, self.y, config, "test-user", "test-model"
        )
        
        # Test prediction
        predictions = self.model_service.predict(
            model_info.model_id, self.X.head(5)
        )
        
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics."""
        config = ModelTrainingConfig(
            model_type="linear_regression",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        model_info = self.model_service.train_model(
            self.X, self.y, config, "test-user", "test-model"
        )
        
        # Check metrics exist
        assert 'r2_score' in model_info.metrics
        assert 'mse' in model_info.metrics
        assert 'mae' in model_info.metrics
        
        # R² should be high for our synthetic data
        assert model_info.metrics['r2_score'] > 0.9

    def test_invalid_model_type(self):
        """Test invalid model type handling."""
        config = ModelTrainingConfig(
            model_type="invalid_model",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        try:
            self.model_service.train_model(
                self.X, self.y, config, "test-user", "test-model"
            )
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        config = ModelTrainingConfig(
            model_type="linear_regression",
            target_column="target",
            feature_columns=[]
        )
        
        try:
            self.model_service.train_model(
                empty_X, empty_y, config, "test-user", "test-model"
            )
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected

    def test_missing_features(self):
        """Test handling of missing feature columns."""
        config = ModelTrainingConfig(
            model_type="linear_regression",
            target_column="target",
            feature_columns=["missing_feature"]
        )
        
        try:
            self.model_service.train_model(
                self.X, self.y, config, "test-user", "test-model"
            )
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected