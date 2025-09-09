"""
Neural Network models implementation using TensorFlow
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from pathlib import Path
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
import uuid
from datetime import datetime

from backend.core.config import settings, model_config
from backend.core.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

class DataPreprocessor:
    """Data preprocessing for neural networks"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        self.is_classification = False
        self.num_classes = 0
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = None, task_type: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data"""
        with PerformanceLogger("Data preprocessing"):
            try:
                # Make a copy to avoid modifying original
                data = df.copy()
                
                # Determine target column
                if target_column is None:
                    # Use last column as target
                    target_column = data.columns[-1]
                
                self.target_column = target_column
                
                # Separate features and target
                if target_column in data.columns:
                    X = data.drop(columns=[target_column])
                    y = data[target_column]
                else:
                    # No target column - unsupervised or prediction mode
                    X = data
                    y = None
                
                # Store feature columns
                self.feature_columns = X.columns.tolist()
                
                # Process features
                X_processed = self._process_features(X, fit=True)
                
                # Process target
                if y is not None:
                    y_processed = self._process_target(y, task_type)
                else:
                    y_processed = None
                
                logger.info(f"Preprocessing completed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
                
                return X_processed, y_processed
                
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                raise
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        try:
            # Select only the feature columns used during training
            X = df[self.feature_columns]
            return self._process_features(X, fit=False)
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            raise
    
    def _process_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Process feature columns"""
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean(numeric_only=True))
        X_processed = X_processed.fillna("unknown")  # For categorical
        
        # Encode categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    unique_values = set(le.classes_)
                    X_processed[col] = X_processed[col].apply(
                        lambda x: x if x in unique_values else "unknown"
                    )
                    X_processed[col] = le.transform(X_processed[col].astype(str))
                else:
                    # If encoder not found, set to 0
                    X_processed[col] = 0
        
        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def _process_target(self, y: pd.Series, task_type: str) -> np.ndarray:
        """Process target variable"""
        # Determine task type automatically if not specified
        if task_type == "auto":
            if y.dtype == 'object' or len(y.unique()) < 20:
                task_type = "classification"
            else:
                task_type = "regression"
        
        if task_type == "classification":
            self.is_classification = True
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                self.label_encoders['target'] = le
                self.num_classes = len(le.classes_)
            else:
                y_encoded = y.values
                self.num_classes = len(y.unique())
            
            # One-hot encode for multi-class
            if self.num_classes > 2:
                y_processed = tf.keras.utils.to_categorical(y_encoded, num_classes=self.num_classes)
            else:
                y_processed = y_encoded
                
        else:
            self.is_classification = False
            y_processed = y.values
        
        return y_processed
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_classification': self.is_classification,
            'num_classes': self.num_classes
        }
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor from file"""
        preprocessor_data = joblib.load(filepath)
        preprocessor = cls()
        
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.feature_columns = preprocessor_data['feature_columns']
        preprocessor.target_column = preprocessor_data['target_column']
        preprocessor.is_classification = preprocessor_data['is_classification']
        preprocessor.num_classes = preprocessor_data['num_classes']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

class NeuralNetworkModel:
    """Neural Network model wrapper"""
    
    def __init__(self, architecture: str = "simple", **kwargs):
        self.architecture = architecture
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.history = None
        self.model_id = str(uuid.uuid4())
        self.is_trained = False
        
        # Model configuration
        self.config = model_config.ARCHITECTURES.get(architecture, model_config.ARCHITECTURES["simple"])
        self.config.update(kwargs)
        
        # Training parameters
        self.epochs = settings.EPOCHS
        self.batch_size = settings.BATCH_SIZE
        self.learning_rate = settings.LEARNING_RATE
        
    def build_model(self, input_shape: int, output_shape: int = 1) -> keras.Model:
        """Build neural network architecture"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.config["layers"][0],
            activation=self.config["activation"],
            input_shape=(input_shape,),
            name="input_layer"
        ))
        
        # Hidden layers
        for i, units in enumerate(self.config["layers"][1:], 1):
            model.add(layers.Dense(
                units,
                activation=self.config["activation"],
                name=f"hidden_layer_{i}"
            ))
            
            # Add dropout
            if self.config.get("dropout", 0) > 0:
                model.add(layers.Dropout(self.config["dropout"], name=f"dropout_{i}"))
        
        # Output layer
        if self.preprocessor.is_classification:
            if self.preprocessor.num_classes > 2:
                # Multi-class classification
                model.add(layers.Dense(
                    self.preprocessor.num_classes,
                    activation="softmax",
                    name="output_layer"
                ))
            else:
                # Binary classification
                model.add(layers.Dense(
                    1,
                    activation="sigmoid",
                    name="output_layer"
                ))
        else:
            # Regression
            model.add(layers.Dense(
                output_shape,
                activation="linear",
                name="output_layer"
            ))
        
        return model
    
    def compile_model(self, model: keras.Model):
        """Compile the model with appropriate loss and metrics"""
        if self.preprocessor.is_classification:
            if self.preprocessor.num_classes > 2:
                loss = "categorical_crossentropy"
                metrics = ["accuracy"]
            else:
                loss = "binary_crossentropy"
                metrics = ["accuracy"]
        else:
            loss = "mse"
            metrics = ["mae"]
        
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with loss: {loss}, optimizer: Adam(lr={self.learning_rate})")
    
    def train(self, df: pd.DataFrame, target_column: str = None, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the neural network"""
        with PerformanceLogger(f"Neural network training - {self.architecture}"):
            try:
                # Preprocess data
                X, y = self.preprocessor.fit_transform(df, target_column)
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
                
                # Build and compile model
                self.model = self.build_model(
                    input_shape=X.shape[1],
                    output_shape=y.shape[1] if len(y.shape) > 1 else 1
                )
                self.compile_model(self.model)
                
                # Callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(
                        patience=model_config.TENSORFLOW_SETTINGS["early_stopping_patience"],
                        restore_best_weights=True,
                        verbose=1
                    ),
                    callbacks.ReduceLROnPlateau(
                        patience=model_config.TENSORFLOW_SETTINGS["reduce_lr_patience"],
                        factor=0.5,
                        verbose=1
                    )
                ]
                
                # Train model
                self.history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                self.is_trained = True
                
                # Evaluate model
                train_metrics = self.evaluate(X_train, y_train)
                val_metrics = self.evaluate(X_val, y_val)
                
                training_results = {
                    "model_id": self.model_id,
                    "architecture": self.architecture,
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "epochs_completed": len(self.history.history["loss"]),
                    "final_training_loss": float(self.history.history["loss"][-1]),
                    "final_validation_loss": float(self.history.history["val_loss"][-1]),
                    "training_metrics": train_metrics,
                    "validation_metrics": val_metrics,
                    "is_classification": self.preprocessor.is_classification,
                    "num_classes": self.preprocessor.num_classes if self.preprocessor.is_classification else None
                }
                
                logger.info(f"Training completed successfully: {training_results}")
                return training_results
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Preprocess data
            X = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Post-process predictions
            if self.preprocessor.is_classification:
                if self.preprocessor.num_classes > 2:
                    # Multi-class: get class probabilities and predicted classes
                    predicted_classes = np.argmax(predictions, axis=1)
                    if 'target' in self.preprocessor.label_encoders:
                        le = self.preprocessor.label_encoders['target']
                        predicted_labels = le.inverse_transform(predicted_classes)
                    else:
                        predicted_labels = predicted_classes
                    
                    results = {
                        "predictions": predicted_labels.tolist(),
                        "probabilities": predictions.tolist(),
                        "predicted_classes": predicted_classes.tolist()
                    }
                else:
                    # Binary classification
                    predicted_probs = predictions.flatten()
                    predicted_classes = (predicted_probs > 0.5).astype(int)
                    
                    results = {
                        "predictions": predicted_classes.tolist(),
                        "probabilities": predicted_probs.tolist()
                    }
            else:
                # Regression
                results = {
                    "predictions": predictions.flatten().tolist()
                }
            
            results["prediction_count"] = len(predictions)
            results["model_id"] = self.model_id
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.model.predict(X)
        
        if self.preprocessor.is_classification:
            if self.preprocessor.num_classes > 2:
                y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_true = y
                y_pred = (predictions.flatten() > 0.5).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred)
            
            return {
                "accuracy": float(accuracy),
                "loss": float(self.model.evaluate(X, y, verbose=0)[0])
            }
        else:
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2_score": float(r2),
                "rmse": float(np.sqrt(mse))
            }
    
    def save(self, filepath: str):
        """Save model and preprocessor"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = f"{filepath}_model.h5"
            self.model.save(model_path)
            
            # Save preprocessor
            preprocessor_path = f"{filepath}_preprocessor.pkl"
            self.preprocessor.save(preprocessor_path)
            
            # Save metadata
            metadata = {
                "model_id": self.model_id,
                "architecture": self.architecture,
                "config": self.config,
                "is_trained": self.is_trained,
                "is_classification": self.preprocessor.is_classification,
                "num_classes": self.preprocessor.num_classes,
                "feature_columns": self.preprocessor.feature_columns,
                "target_column": self.preprocessor.target_column,
                "created_at": datetime.now().isoformat()
            }
            
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved successfully to {filepath}")
            
            return {
                "model_path": model_path,
                "preprocessor_path": preprocessor_path,
                "metadata_path": metadata_path
            }
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str):
        """Load model and preprocessor"""
        try:
            # Load metadata
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model instance
            model_instance = cls(architecture=metadata["architecture"])
            model_instance.model_id = metadata["model_id"]
            model_instance.config = metadata["config"]
            model_instance.is_trained = metadata["is_trained"]
            
            # Load preprocessor
            preprocessor_path = f"{filepath}_preprocessor.pkl"
            model_instance.preprocessor = DataPreprocessor.load(preprocessor_path)
            
            # Load model
            model_path = f"{filepath}_model.h5"
            model_instance.model = keras.models.load_model(model_path)
            
            logger.info(f"Model loaded successfully from {filepath}")
            return model_instance
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

# Model factory for different architectures
class ModelFactory:
    """Factory for creating different neural network models"""
    
    @staticmethod
    def create_model(architecture: str = "simple", **kwargs) -> NeuralNetworkModel:
        """Create a neural network model with specified architecture"""
        if architecture not in model_config.ARCHITECTURES:
            logger.warning(f"Unknown architecture '{architecture}', using 'simple'")
            architecture = "simple"
        
        return NeuralNetworkModel(architecture=architecture, **kwargs)
    
    @staticmethod
    def get_available_architectures() -> List[str]:
        """Get list of available model architectures"""
        return list(model_config.ARCHITECTURES.keys())

# Export main components
__all__ = [
    "NeuralNetworkModel",
    "DataPreprocessor",
    "ModelFactory"
]