"""Model training service with scikit-learn integration for linear regression."""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE

from backend.core.config import settings
from backend.core.logging import audit_logger, EventType
from backend.core.exceptions import ValidationError, SecurityError
from backend.services.cache_service import cache_service
from backend.services.data_service import data_service
from backend.models.database import db_manager


@dataclass
class ModelTrainingConfig:
    """Model training configuration."""
    model_type: str = "linear_regression"
    test_size: float = 0.2
    random_state: int = 42
    scaling_method: Optional[str] = None  # 'standard', 'minmax', 'robust'
    feature_selection: Optional[str] = None  # 'kbest', 'rfe'
    feature_selection_k: int = 10
    cross_validation: bool = True
    cv_folds: int = 5
    hyperparameter_tuning: bool = False
    regularization_alpha: Optional[float] = None


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    r2_score: float
    adjusted_r2: float
    mse: float
    mae: float
    rmse: float
    mape: float
    explained_variance: float
    cv_scores: Optional[List[float]]
    cv_mean: Optional[float]
    cv_std: Optional[float]


@dataclass
class ModelInfo:
    """Complete model information."""
    model_id: str
    name: str
    model_type: str
    dataset_id: str
    owner_id: str
    target_column: str
    feature_columns: List[str]
    training_config: ModelTrainingConfig
    performance: ModelPerformance
    feature_importance: Dict[str, float]
    model_size: int
    training_time: float
    created_at: str
    status: str
    metadata: Dict[str, Any]


class ModelTrainingService:
    """Service for training and managing machine learning models."""
    
    SUPPORTED_MODELS = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet
    }
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    MAX_FEATURES = 1000  # Maximum number of features
    MIN_SAMPLES = 10  # Minimum samples for training
    CACHE_TTL = 7200  # 2 hours cache
    
    def __init__(self):
        """Initialize model training service."""
        self.models_storage = Path(settings.PROCESSED_PATH) / "models"
        self.models_storage.mkdir(parents=True, exist_ok=True)
    
    async def train_model(self, dataset_id: str, target_column: str,
                         feature_columns: Optional[List[str]],
                         config: ModelTrainingConfig,
                         user_id: str) -> ModelInfo:
        """Train a machine learning model.
        
        Args:
            dataset_id: Dataset to train on
            target_column: Target variable column name
            feature_columns: Feature columns (None for auto-selection)
            config: Training configuration
            user_id: User training the model
            
        Returns:
            ModelInfo with training results
        """
        start_time = datetime.utcnow()
        
        # Load dataset
        df = await data_service.load_dataset(dataset_id, user_id)
        
        # Validate training data
        self._validate_training_data(df, target_column, feature_columns)
        
        # Prepare features and target
        X, y, feature_names = await self._prepare_features_target(
            df, target_column, feature_columns, config
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Apply scaling if specified
        scaler = None
        if config.scaling_method:
            scaler = self.SCALERS[config.scaling_method]()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        model = await self._train_model_with_config(X_train, y_train, config)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate performance metrics
        performance = self._calculate_performance(
            y_train, y_pred_train, y_test, y_pred_test, X_train, config
        )
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(model, feature_names)
        
        # Generate model ID and save
        model_id = f"model_{int(datetime.utcnow().timestamp())}_{hash(dataset_id)}"
        model_path = await self._save_model(
            model_id, model, scaler, feature_names, target_column
        )
        
        # Calculate training time
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            name=f"{config.model_type}_{target_column}_{dataset_id[:8]}",
            model_type=config.model_type,
            dataset_id=dataset_id,
            owner_id=user_id,
            target_column=target_column,
            feature_columns=feature_names,
            training_config=config,
            performance=performance,
            feature_importance=feature_importance,
            model_size=model_path.stat().st_size,
            training_time=training_time,
            created_at=datetime.utcnow().isoformat(),
            status="trained",
            metadata={
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'original_features': df.shape[1] - 1,
                'selected_features': len(feature_names)
            }
        )
        
        # Save model info to database
        await self._save_model_info(model_info)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="model_trained",
            details={
                "model_id": model_id,
                "model_type": config.model_type,
                "target_column": target_column,
                "features": len(feature_names),
                "r2_score": performance.r2_score,
                "training_time": training_time
            }
        )
        
        return model_info
    
    def _validate_training_data(self, df: pd.DataFrame, target_column: str,
                              feature_columns: Optional[List[str]]) -> None:
        """Validate training data and parameters.
        
        Args:
            df: Training dataset
            target_column: Target column name
            feature_columns: Feature column names
        """
        if target_column not in df.columns:
            raise ValidationError(f"Target column '{target_column}' not found in dataset")
        
        if df[target_column].isnull().all():
            raise ValidationError(f"Target column '{target_column}' contains only null values")
        
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValidationError(f"Target column '{target_column}' must be numeric for regression")
        
        if feature_columns:
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValidationError(f"Feature columns not found: {list(missing_cols)}")
        
        # Check minimum samples
        valid_rows = df[target_column].notna().sum()
        if valid_rows < self.MIN_SAMPLES:
            raise ValidationError(f"Need at least {self.MIN_SAMPLES} valid samples, got {valid_rows}")
    
    async def _prepare_features_target(self, df: pd.DataFrame, target_column: str,
                                     feature_columns: Optional[List[str]],
                                     config: ModelTrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training.
        
        Args:
            df: Dataset
            target_column: Target column name
            feature_columns: Feature column names
            config: Training configuration
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Get target
        y = df[target_column].values
        
        # Get features
        if feature_columns:
            feature_df = df[feature_columns]
        else:
            # Auto-select numeric columns (excluding target)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            feature_df = df[numeric_cols]
        
        # Remove rows with missing target values
        valid_mask = ~pd.isna(y)
        y = y[valid_mask]
        feature_df = feature_df.loc[valid_mask]
        
        # Handle missing values in features (simple imputation with mean)
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Limit number of features
        if len(feature_df.columns) > self.MAX_FEATURES:
            feature_df = feature_df.iloc[:, :self.MAX_FEATURES]
        
        # Feature selection
        if config.feature_selection:
            feature_df, selected_features = self._apply_feature_selection(
                feature_df, y, config
            )
        else:
            selected_features = feature_df.columns.tolist()
        
        X = feature_df.values
        
        return X, y, selected_features
    
    def _apply_feature_selection(self, X_df: pd.DataFrame, y: np.ndarray,
                               config: ModelTrainingConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Apply feature selection to reduce dimensionality.
        
        Args:
            X_df: Feature DataFrame
            y: Target array
            config: Training configuration
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        if config.feature_selection == 'kbest':
            # Select k best features using f_regression
            k = min(config.feature_selection_k, len(X_df.columns))
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X_df, y)
            
            selected_mask = selector.get_support()
            selected_features = X_df.columns[selected_mask].tolist()
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
        
        elif config.feature_selection == 'rfe':
            # Recursive feature elimination
            k = min(config.feature_selection_k, len(X_df.columns))
            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X_df, y)
            
            selected_mask = selector.get_support()
            selected_features = X_df.columns[selected_mask].tolist()
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
        
        return X_df, X_df.columns.tolist()
    
    async def _train_model_with_config(self, X_train: np.ndarray, y_train: np.ndarray,
                                     config: ModelTrainingConfig):
        """Train model based on configuration.
        
        Args:
            X_train: Training features
            y_train: Training target
            config: Training configuration
            
        Returns:
            Trained model
        """
        model_class = self.SUPPORTED_MODELS[config.model_type]
        
        # Set up model parameters
        model_params = {}
        if config.model_type in ['ridge', 'lasso', 'elastic_net'] and config.regularization_alpha:
            model_params['alpha'] = config.regularization_alpha
        
        if config.hyperparameter_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = self._get_param_grid(config.model_type)
            model = GridSearchCV(
                model_class(**model_params),
                param_grid,
                cv=config.cv_folds,
                scoring='r2',
                n_jobs=-1
            )
        else:
            model = model_class(**model_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    
    def _get_param_grid(self, model_type: str) -> Dict[str, List]:
        """Get hyperparameter grid for tuning.
        
        Args:
            model_type: Model type
            
        Returns:
            Parameter grid dictionary
        """
        if model_type == 'ridge':
            return {'alpha': [0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'lasso':
            return {'alpha': [0.01, 0.1, 1.0, 10.0]}
        elif model_type == 'elastic_net':
            return {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        else:
            return {}
    
    def _calculate_performance(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                             y_test: np.ndarray, y_pred_test: np.ndarray,
                             X_train: np.ndarray, config: ModelTrainingConfig) -> ModelPerformance:
        """Calculate comprehensive model performance metrics.
        
        Args:
            y_train: Training target values
            y_pred_train: Training predictions
            y_test: Test target values
            y_pred_test: Test predictions
            X_train: Training features (for adjusted R2)
            config: Training configuration
            
        Returns:
            ModelPerformance object
        """
        # Primary metrics on test set
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (handle division by zero)
        mask = y_test != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(y_test[mask], y_pred_test[mask])
        else:
            mape = float('inf')
        
        explained_var = explained_variance_score(y_test, y_pred_test)
        
        # Adjusted R²
        n_samples, n_features = X_train.shape
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        # Cross-validation scores
        cv_scores = None
        cv_mean = None
        cv_std = None
        
        if config.cross_validation:
            # Note: This is simplified - in production you'd want to use the same preprocessing
            try:
                from sklearn.linear_model import LinearRegression
                temp_model = LinearRegression()
                cv_scores = cross_val_score(
                    temp_model, X_train, y_train, 
                    cv=config.cv_folds, scoring='r2'
                ).tolist()
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            except Exception:
                pass  # Skip CV if it fails
        
        return ModelPerformance(
            r2_score=round(r2, 4),
            adjusted_r2=round(adjusted_r2, 4),
            mse=round(mse, 4),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mape=round(mape, 4) if mape != float('inf') else None,
            explained_variance=round(explained_var, 4),
            cv_scores=cv_scores,
            cv_mean=round(cv_mean, 4) if cv_mean else None,
            cv_std=round(cv_std, 4) if cv_std else None
        )
    
    def _calculate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from model coefficients.
        
        Args:
            model: Trained model
            feature_names: Feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Handle GridSearchCV models
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        if hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coefficients = np.abs(model.coef_)
            
            # Normalize to sum to 1
            if coefficients.sum() > 0:
                importance_scores = coefficients / coefficients.sum()
            else:
                importance_scores = coefficients
            
            return {
                name: round(float(score), 4) 
                for name, score in zip(feature_names, importance_scores)
            }
        
        return {}
    
    async def _save_model(self, model_id: str, model, scaler, feature_names: List[str],
                         target_column: str) -> Path:
        """Save trained model to disk.
        
        Args:
            model_id: Model identifier
            model: Trained model
            scaler: Fitted scaler (if used)
            feature_names: Feature names
            target_column: Target column name
            
        Returns:
            Path to saved model file
        """
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'target_column': target_column,
            'created_at': datetime.utcnow().isoformat()
        }
        
        model_path = self.models_storage / f"{model_id}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return model_path
    
    async def _save_model_info(self, model_info: ModelInfo) -> bool:
        """Save model information to database.
        
        Args:
            model_info: Model information
            
        Returns:
            True if saved successfully
        """
        query = '''
            INSERT INTO models 
            (model_id, name, model_type, dataset_id, owner_id, model_data, metrics, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        # Serialize complex objects
        model_data_json = json.dumps({
            'target_column': model_info.target_column,
            'feature_columns': model_info.feature_columns,
            'training_config': asdict(model_info.training_config),
            'feature_importance': model_info.feature_importance,
            'model_size': model_info.model_size,
            'training_time': model_info.training_time,
            'metadata': model_info.metadata
        })
        
        metrics_json = json.dumps(asdict(model_info.performance))
        
        affected = db_manager.execute_update(
            query,
            (
                model_info.model_id,
                model_info.name,
                model_info.model_type,
                model_info.dataset_id,
                model_info.owner_id,
                model_data_json,
                metrics_json,
                model_info.status,
                model_info.created_at
            )
        )
        
        return affected > 0
    
    async def load_model(self, model_id: str, user_id: str):
        """Load a trained model from storage.
        
        Args:
            model_id: Model identifier
            user_id: User requesting the model
            
        Returns:
            Loaded model data
        """
        # Check permissions
        model_info = await self.get_model_info(model_id, user_id)
        if not model_info:
            raise SecurityError("Model not found or access denied")
        
        # Check cache first
        cache_key = f"model:{model_id}"
        cached_model = await cache_service.get(cache_key)
        
        if cached_model:
            return cached_model
        
        # Load from disk
        model_path = self.models_storage / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Cache the loaded model
        await cache_service.set(cache_key, model_data, ttl=self.CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=model_info.dataset_id,
            operation="model_loaded",
            details={"model_id": model_id, "model_type": model_info.model_type}
        )
        
        return model_data
    
    async def predict(self, model_id: str, input_data: Union[Dict, List[Dict]], 
                     user_id: str) -> Dict[str, Any]:
        """Make predictions using a trained model.
        
        Args:
            model_id: Model identifier
            input_data: Input data for prediction
            user_id: User making predictions
            
        Returns:
            Prediction results
        """
        # Load model
        model_data = await self.load_model(model_id, user_id)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Handle GridSearchCV models
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Validate features
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            raise ValidationError(f"Missing required features: {list(missing_features)}")
        
        # Select and order features
        X = input_df[feature_names].values
        
        # Handle missing values (simple imputation with 0)
        X = np.nan_to_num(X)
        
        # Apply scaling if used during training
        if scaler:
            X = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate prediction confidence (if possible)
        confidence = None
        if hasattr(model, 'predict_proba'):
            # For classification models
            confidence = np.max(model.predict_proba(X), axis=1).tolist()
        
        results = {
            'model_id': model_id,
            'predictions': predictions.tolist(),
            'confidence': confidence,
            'feature_names': feature_names,
            'n_samples': len(predictions),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id="prediction_request",
            operation="model_prediction",
            details={
                "model_id": model_id,
                "n_predictions": len(predictions)
            }
        )
        
        return results
    
    async def get_model_info(self, model_id: str, user_id: str) -> Optional[ModelInfo]:
        """Get model information from database.
        
        Args:
            model_id: Model identifier
            user_id: User requesting information
            
        Returns:
            ModelInfo object or None if not found
        """
        query = "SELECT * FROM models WHERE model_id = ?"
        results = db_manager.execute_query(query, (model_id,))
        
        if not results:
            return None
        
        row = results[0]
        
        # Check permissions
        if row['owner_id'] != user_id:
            # TODO: Check if user has read permissions
            return None
        
        # Parse JSON data
        model_data = json.loads(row['model_data'])
        metrics_data = json.loads(row['metrics'])
        
        return ModelInfo(
            model_id=row['model_id'],
            name=row['name'],
            model_type=row['model_type'],
            dataset_id=row['dataset_id'],
            owner_id=row['owner_id'],
            target_column=model_data['target_column'],
            feature_columns=model_data['feature_columns'],
            training_config=ModelTrainingConfig(**model_data['training_config']),
            performance=ModelPerformance(**metrics_data),
            feature_importance=model_data['feature_importance'],
            model_size=model_data['model_size'],
            training_time=model_data['training_time'],
            created_at=row['created_at'],
            status=row['status'],
            metadata=model_data['metadata']
        )
    
    async def list_models(self, user_id: str, limit: int = 50, 
                         offset: int = 0) -> List[ModelInfo]:
        """List models owned by user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            List of ModelInfo objects
        """
        query = '''
            SELECT * FROM models 
            WHERE owner_id = ? 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        '''
        
        results = db_manager.execute_query(query, (user_id, limit, offset))
        
        models = []
        for row in results:
            model_data = json.loads(row['model_data'])
            metrics_data = json.loads(row['metrics'])
            
            model_info = ModelInfo(
                model_id=row['model_id'],
                name=row['name'],
                model_type=row['model_type'],
                dataset_id=row['dataset_id'],
                owner_id=row['owner_id'],
                target_column=model_data['target_column'],
                feature_columns=model_data['feature_columns'],
                training_config=ModelTrainingConfig(**model_data['training_config']),
                performance=ModelPerformance(**metrics_data),
                feature_importance=model_data['feature_importance'],
                model_size=model_data['model_size'],
                training_time=model_data['training_time'],
                created_at=row['created_at'],
                status=row['status'],
                metadata=model_data['metadata']
            )
            
            models.append(model_info)
        
        return models
    
    async def delete_model(self, model_id: str, user_id: str) -> bool:
        """Delete a model and its files.
        
        Args:
            model_id: Model identifier
            user_id: User requesting deletion
            
        Returns:
            True if deleted successfully
        """
        # Check permissions
        model_info = await self.get_model_info(model_id, user_id)
        if not model_info:
            raise SecurityError("Model not found or access denied")
        
        # Delete from database
        query = "DELETE FROM models WHERE model_id = ? AND owner_id = ?"
        affected = db_manager.execute_update(query, (model_id, user_id))
        
        if affected > 0:
            # Delete model file
            model_path = self.models_storage / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()
            
            # Clear cache
            cache_key = f"model:{model_id}"
            await cache_service.delete(cache_key)
            
            audit_logger.log_data_access(
                user_id=user_id,
                dataset_id=model_info.dataset_id,
                operation="model_deleted",
                details={"model_id": model_id, "model_type": model_info.model_type}
            )
            
            return True
        
        return False


# Global model service instance
model_service = ModelTrainingService()