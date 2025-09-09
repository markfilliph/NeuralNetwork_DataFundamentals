"""
Model service for training neural networks and making predictions
"""
import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.core.config import settings, model_config
from backend.core.database import UploadedFile, PredictionJob, ModelInfo, get_db_session
from backend.core.logging_config import get_logger, log_prediction_request, PerformanceLogger
from backend.models.neural_networks import NeuralNetworkModel, ModelFactory
from backend.services.data_service import data_service

logger = get_logger(__name__)

class ModelService:
    """Service for machine learning model operations"""
    
    def __init__(self):
        self.model_dir = Path(settings.MODEL_SAVE_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded models
        self._model_cache = {}
    
    async def train_model(
        self,
        file_id: str,
        model_config: Dict[str, Any] = None,
        user_ip: str = None
    ) -> Dict[str, Any]:
        """Train a neural network model on uploaded data"""
        with PerformanceLogger(f"Model training for file: {file_id}"):
            try:
                # Create prediction job
                job_id = str(uuid.uuid4())
                job_record = self._create_prediction_job(job_id, file_id, model_config)
                
                # Log security event
                log_prediction_request(
                    file_id,
                    model_config.get("model_type", "neural_network"),
                    user_ip
                )
                
                # Get file data
                file_data = data_service.get_file_data(file_id)
                df = pd.DataFrame(file_data["data"])
                
                # Update job status
                self._update_job_status(job_id, "running", started_at=datetime.now())
                
                # Configure model
                architecture = model_config.get("architecture", "simple")
                target_column = model_config.get("target_column")
                
                # Create and train model
                model = ModelFactory.create_model(architecture=architecture)
                
                training_results = model.train(df, target_column=target_column)
                
                # Save trained model
                model_save_path = self.model_dir / f"model_{job_id}"
                saved_paths = model.save(str(model_save_path))
                
                # Save model info to database
                model_info = self._save_model_info(model, job_id, file_id, saved_paths, training_results)
                
                # Update job with results
                self._update_job_results(job_id, training_results, str(model_save_path))
                
                logger.info(f"Model training completed successfully: {job_id}")
                
                return {
                    "job_id": job_id,
                    "model_id": model.model_id,
                    "status": "completed",
                    "training_results": training_results,
                    "model_path": str(model_save_path),
                    "message": "Model trained successfully"
                }
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                # Update job status to failed
                try:
                    self._update_job_status(job_id, "failed", error_message=str(e))
                except:
                    pass
                raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    async def predict(
        self,
        file_id: str,
        model_id: str = None,
        prediction_config: Dict[str, Any] = None,
        user_ip: str = None
    ) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        with PerformanceLogger(f"Prediction for file: {file_id}"):
            try:
                # Log security event
                log_prediction_request(
                    file_id,
                    prediction_config.get("model_type", "neural_network") if prediction_config else "neural_network",
                    user_ip
                )
                
                # Get file data
                file_data = data_service.get_file_data(file_id)
                df = pd.DataFrame(file_data["data"])
                
                # Find or train model
                if model_id:
                    model = self._load_model(model_id)
                else:
                    # Train a new model if none specified
                    training_result = await self.train_model(file_id, prediction_config or {}, user_ip)
                    model_id = training_result["model_id"]
                    model = self._load_model(model_id)
                
                # Make predictions
                predictions = model.predict(df)
                
                # Save predictions
                predictions_path = self._save_predictions(file_id, model_id, predictions, df)
                
                # Create response
                result = {
                    "file_id": file_id,
                    "model_id": model_id,
                    "predictions_count": predictions["prediction_count"],
                    "predictions": predictions["predictions"][:100],  # Limit response size
                    "model_type": "neural_network",
                    "architecture": model.architecture,
                    "predictions_file": str(predictions_path),
                    "download_url": f"/api/data/download/{Path(predictions_path).name}",
                    "message": "Predictions generated successfully"
                }
                
                # Add classification-specific information
                if model.preprocessor.is_classification:
                    result["task_type"] = "classification"
                    result["num_classes"] = model.preprocessor.num_classes
                    if "probabilities" in predictions:
                        result["probabilities"] = predictions["probabilities"][:100]
                else:
                    result["task_type"] = "regression"
                
                logger.info(f"Predictions generated successfully for file: {file_id}")
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a prediction job"""
        try:
            db = get_db_session()
            try:
                job_record = db.query(PredictionJob).filter(PredictionJob.job_id == job_id).first()
                
                if not job_record:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                # Parse model metrics if available
                model_metrics = {}
                if job_record.model_metrics:
                    try:
                        model_metrics = json.loads(job_record.model_metrics)
                    except json.JSONDecodeError:
                        pass
                
                return {
                    "job_id": job_record.job_id,
                    "file_id": job_record.file_id,
                    "model_type": job_record.model_type,
                    "architecture": job_record.model_architecture,
                    "status": job_record.status,
                    "progress": job_record.progress,
                    "error_message": job_record.error_message,
                    "predictions_count": job_record.predictions_count,
                    "accuracy_score": job_record.accuracy_score,
                    "loss_value": job_record.loss_value,
                    "model_metrics": model_metrics,
                    "results_file": job_record.results_file_path,
                    "created_at": job_record.created_at.isoformat(),
                    "started_at": job_record.started_at.isoformat() if job_record.started_at else None,
                    "completed_at": job_record.completed_at.isoformat() if job_record.completed_at else None
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get job status")
    
    def list_models(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List available trained models"""
        try:
            db = get_db_session()
            try:
                query = db.query(ModelInfo).filter(ModelInfo.is_active == True).order_by(ModelInfo.created_at.desc())
                
                total_count = query.count()
                models = query.offset(offset).limit(limit).all()
                
                model_list = []
                for model_record in models:
                    # Parse feature info
                    input_features = []
                    if model_record.input_features:
                        try:
                            input_features = json.loads(model_record.input_features)
                        except json.JSONDecodeError:
                            pass
                    
                    model_list.append({
                        "model_id": model_record.model_id,
                        "model_name": model_record.model_name,
                        "model_type": model_record.model_type,
                        "architecture": model_record.architecture,
                        "training_file_id": model_record.training_file_id,
                        "epochs_trained": model_record.epochs_trained,
                        "final_loss": model_record.final_loss,
                        "final_accuracy": model_record.final_accuracy,
                        "input_features": input_features,
                        "is_active": model_record.is_active,
                        "created_at": model_record.created_at.isoformat()
                    })
                
                return {
                    "models": model_list,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_next": offset + limit < total_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise HTTPException(status_code=500, detail="Failed to list models")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            db = get_db_session()
            try:
                model_record = db.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
                
                if not model_record:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Parse JSON fields
                input_features = []
                output_features = []
                preprocessing_config = {}
                
                if model_record.input_features:
                    try:
                        input_features = json.loads(model_record.input_features)
                    except json.JSONDecodeError:
                        pass
                
                if model_record.output_features:
                    try:
                        output_features = json.loads(model_record.output_features)
                    except json.JSONDecodeError:
                        pass
                
                if model_record.preprocessing_config:
                    try:
                        preprocessing_config = json.loads(model_record.preprocessing_config)
                    except json.JSONDecodeError:
                        pass
                
                return {
                    "model_id": model_record.model_id,
                    "model_name": model_record.model_name,
                    "model_type": model_record.model_type,
                    "architecture": model_record.architecture,
                    "model_file_path": model_record.model_file_path,
                    "weights_file_path": model_record.weights_file_path,
                    "scaler_file_path": model_record.scaler_file_path,
                    "training_file_id": model_record.training_file_id,
                    "epochs_trained": model_record.epochs_trained,
                    "final_loss": model_record.final_loss,
                    "final_accuracy": model_record.final_accuracy,
                    "input_features": input_features,
                    "output_features": output_features,
                    "preprocessing_config": preprocessing_config,
                    "is_active": model_record.is_active,
                    "created_at": model_record.created_at.isoformat(),
                    "updated_at": model_record.updated_at.isoformat()
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise HTTPException(status_code=500, detail="Failed to get model information")
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a trained model"""
        try:
            db = get_db_session()
            try:
                model_record = db.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
                
                if not model_record:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Delete physical files
                try:
                    if model_record.model_file_path and os.path.exists(model_record.model_file_path):
                        os.remove(model_record.model_file_path)
                    
                    if model_record.weights_file_path and os.path.exists(model_record.weights_file_path):
                        os.remove(model_record.weights_file_path)
                    
                    if model_record.scaler_file_path and os.path.exists(model_record.scaler_file_path):
                        os.remove(model_record.scaler_file_path)
                    
                    # Remove from cache
                    if model_id in self._model_cache:
                        del self._model_cache[model_id]
                        
                except OSError as e:
                    logger.warning(f"Failed to delete model files: {e}")
                
                # Mark as inactive instead of deleting (for audit trail)
                model_record.is_active = False
                db.commit()
                
                logger.info(f"Model deleted successfully: {model_id}")
                
                return {
                    "model_id": model_id,
                    "message": "Model deleted successfully"
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete model")
    
    def get_available_architectures(self) -> Dict[str, Any]:
        """Get available model architectures"""
        return {
            "architectures": ModelFactory.get_available_architectures(),
            "configurations": model_config.ARCHITECTURES,
            "default_architecture": "simple"
        }
    
    def _create_prediction_job(self, job_id: str, file_id: str, model_config: Dict[str, Any]) -> PredictionJob:
        """Create a new prediction job record"""
        try:
            db = get_db_session()
            try:
                job_record = PredictionJob(
                    job_id=job_id,
                    file_id=file_id,
                    model_type=model_config.get("model_type", "neural_network"),
                    model_architecture=model_config.get("architecture", "simple"),
                    status="pending"
                )
                
                db.add(job_record)
                db.commit()
                db.refresh(job_record)
                
                return job_record
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to create prediction job: {e}")
            raise
    
    def _update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and other fields"""
        try:
            db = get_db_session()
            try:
                job_record = db.query(PredictionJob).filter(PredictionJob.job_id == job_id).first()
                
                if job_record:
                    job_record.status = status
                    job_record.updated_at = datetime.now()
                    
                    # Update additional fields
                    for key, value in kwargs.items():
                        if hasattr(job_record, key):
                            setattr(job_record, key, value)
                    
                    db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def _update_job_results(self, job_id: str, training_results: Dict[str, Any], model_path: str):
        """Update job with training results"""
        try:
            db = get_db_session()
            try:
                job_record = db.query(PredictionJob).filter(PredictionJob.job_id == job_id).first()
                
                if job_record:
                    job_record.status = "completed"
                    job_record.completed_at = datetime.now()
                    job_record.results_file_path = model_path
                    job_record.model_metrics = json.dumps(training_results)
                    
                    # Extract specific metrics
                    if "final_training_loss" in training_results:
                        job_record.loss_value = training_results["final_training_loss"]
                    
                    if "validation_metrics" in training_results:
                        val_metrics = training_results["validation_metrics"]
                        if "accuracy" in val_metrics:
                            job_record.accuracy_score = val_metrics["accuracy"]
                    
                    db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to update job results: {e}")
    
    def _save_model_info(
        self,
        model: NeuralNetworkModel,
        job_id: str,
        file_id: str,
        saved_paths: Dict[str, str],
        training_results: Dict[str, Any]
    ) -> ModelInfo:
        """Save model information to database"""
        try:
            db = get_db_session()
            try:
                model_record = ModelInfo(
                    model_id=model.model_id,
                    model_name=f"Model_{model.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type="neural_network",
                    architecture=model.architecture,
                    model_file_path=saved_paths["model_path"],
                    weights_file_path=saved_paths.get("weights_path"),
                    scaler_file_path=saved_paths["preprocessor_path"],
                    training_file_id=file_id,
                    epochs_trained=training_results.get("epochs_completed", 0),
                    final_loss=training_results.get("final_training_loss"),
                    final_accuracy=training_results.get("validation_metrics", {}).get("accuracy"),
                    input_features=json.dumps(model.preprocessor.feature_columns),
                    output_features=json.dumps([model.preprocessor.target_column]),
                    preprocessing_config=json.dumps({
                        "is_classification": model.preprocessor.is_classification,
                        "num_classes": model.preprocessor.num_classes
                    })
                )
                
                db.add(model_record)
                db.commit()
                db.refresh(model_record)
                
                return model_record
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to save model info: {e}")
            raise
    
    def _load_model(self, model_id: str) -> NeuralNetworkModel:
        """Load a trained model"""
        # Check cache first
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        try:
            # Get model info from database
            model_info = self.get_model_info(model_id)
            
            # Find model files
            model_base_path = model_info["model_file_path"].replace("_model.h5", "")
            
            # Load model
            model = NeuralNetworkModel.load(model_base_path)
            
            # Cache the model
            self._model_cache[model_id] = model
            
            logger.info(f"Model loaded successfully: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    def _save_predictions(
        self,
        file_id: str,
        model_id: str,
        predictions: Dict[str, Any],
        original_df: pd.DataFrame
    ) -> Path:
        """Save predictions to file"""
        try:
            # Create predictions DataFrame
            pred_df = original_df.copy()
            pred_df["predictions"] = predictions["predictions"]
            
            if "probabilities" in predictions:
                pred_df["probabilities"] = predictions["probabilities"]
            
            if "predicted_classes" in predictions:
                pred_df["predicted_classes"] = predictions["predicted_classes"]
            
            # Save to file
            predictions_dir = Path(settings.PROCESSED_DIR) / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_file = predictions_dir / f"predictions_{file_id}_{model_id}_{timestamp}.csv"
            
            pred_df.to_csv(predictions_file, index=False)
            
            logger.info(f"Predictions saved to: {predictions_file}")
            return predictions_file
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise

# Global instance
model_service = ModelService()

# Export main components
__all__ = ["ModelService", "model_service"]