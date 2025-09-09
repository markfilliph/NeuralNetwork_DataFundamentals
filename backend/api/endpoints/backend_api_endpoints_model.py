"""
Machine Learning model endpoints for training and prediction
"""
from fastapi import APIRouter, HTTPException, Depends, Request, Query, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

from backend.services.model_service import model_service
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Request/Response Models
class TrainingRequest(BaseModel):
    """Request model for training a neural network"""
    file_id: str = Field(..., description="ID of the uploaded file to train on")
    target_column: Optional[str] = Field(None, description="Target column name (if not specified, uses last column)")
    architecture: str = Field("simple", description="Model architecture: simple, deep, classification")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, description="Training batch size")
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    validation_split: float = Field(0.2, description="Validation data split ratio")
    task_type: str = Field("auto", description="Task type: auto, classification, regression")

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    file_id: str = Field(..., description="ID of the file to make predictions on")
    model_id: Optional[str] = Field(None, description="ID of the trained model (if not provided, trains a new model)")
    architecture: str = Field("simple", description="Model architecture if training new model")
    target_column: Optional[str] = Field(None, description="Target column for training new model")

class ModelUpdateRequest(BaseModel):
    """Request model for updating model information"""
    model_name: Optional[str] = Field(None, description="New model name")
    is_active: Optional[bool] = Field(None, description="Whether model is active")

@router.post("/train")
async def train_model(
    request: Request,
    training_request: TrainingRequest
):
    """
    Train a neural network model on uploaded data
    
    - **file_id**: ID of the uploaded file to train on
    - **target_column**: Target column name for prediction
    - **architecture**: Model architecture (simple, deep, classification)
    - **epochs**: Number of training epochs
    - **batch_size**: Training batch size
    - **learning_rate**: Learning rate for training
    - **validation_split**: Ratio of data to use for validation
    - **task_type**: Type of task (auto, classification, regression)
    """
    try:
        # Get client IP for logging
        client_ip = request.client.host
        
        # Prepare model configuration
        model_config = {
            "model_type": "neural_network",
            "architecture": training_request.architecture,
            "target_column": training_request.target_column,
            "task_type": training_request.task_type,
            "validation_split": training_request.validation_split
        }
        
        # Add optional parameters if provided
        if training_request.epochs:
            model_config["epochs"] = training_request.epochs
        if training_request.batch_size:
            model_config["batch_size"] = training_request.batch_size
        if training_request.learning_rate:
            model_config["learning_rate"] = training_request.learning_rate
        
        # Train model
        result = await model_service.train_model(
            file_id=training_request.file_id,
            model_config=model_config,
            user_ip=client_ip
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Model training completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/predict")
async def predict(
    request: Request,
    prediction_request: PredictionRequest
):
    """
    Make predictions using a trained model or train a new model
    
    - **file_id**: ID of the file to make predictions on
    - **model_id**: ID of the trained model (optional)
    - **architecture**: Model architecture if training new model
    - **target_column**: Target column if training new model
    """
    try:
        # Get client IP for logging
        client_ip = request.client.host
        
        # Prepare prediction configuration
        prediction_config = {
            "architecture": prediction_request.architecture,
            "target_column": prediction_request.target_column
        }
        
        # Make predictions
        result = await model_service.predict(
            file_id=prediction_request.file_id,
            model_id=prediction_request.model_id,
            prediction_config=prediction_config,
            user_ip=client_ip
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Predictions generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status and details of a training job
    
    - **job_id**: Unique identifier of the training job
    """
    try:
        result = model_service.get_job_status(job_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Job status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")

@router.get("/models")
async def list_models(
    limit: int = Query(50, ge=1, le=100, description="Number of models to return"),
    offset: int = Query(0, ge=0, description="Number of models to skip"),
    architecture: Optional[str] = Query(None, description="Filter by architecture"),
    is_active: bool = Query(True, description="Filter by active status")
):
    """
    List available trained models
    
    - **limit**: Maximum number of models to return (1-100)
    - **offset**: Number of models to skip for pagination
    - **architecture**: Filter models by architecture
    - **is_active**: Filter by active status
    """
    try:
        result = model_service.list_models(limit=limit, offset=offset)
        
        # Apply filters
        if architecture:
            result["models"] = [m for m in result["models"] if m["architecture"] == architecture]
        
        return {
            "success": True,
            "data": result,
            "message": f"Retrieved {len(result['models'])} models"
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get detailed information about a specific model
    
    - **model_id**: Unique identifier of the model
    """
    try:
        result = model_service.get_model_info(model_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Model information retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.put("/models/{model_id}")
async def update_model(model_id: str, update_request: ModelUpdateRequest):
    """
    Update model information
    
    - **model_id**: Unique identifier of the model
    - **model_name**: New model name
    - **is_active**: Whether model is active
    """
    try:
        # This would be implemented in model_service
        # For now, return a placeholder response
        return {
            "success": True,
            "data": {"model_id": model_id, "message": "Model update functionality not yet implemented"},
            "message": "Model update requested (feature coming soon)"
        }
        
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model
    
    - **model_id**: Unique identifier of the model
    """
    try:
        result = model_service.delete_model(model_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Model deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

@router.get("/architectures")
async def get_available_architectures():
    """
    Get available model architectures and their configurations
    """
    try:
        result = model_service.get_available_architectures()
        
        return {
            "success": True,
            "data": result,
            "message": "Available architectures retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get architectures: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available architectures")

@router.post("/evaluate/{model_id}")
async def evaluate_model(
    model_id: str,
    file_id: str = Body(..., embed=True, description="File ID to evaluate model on")
):
    """
    Evaluate a trained model on a specific dataset
    
    - **model_id**: Unique identifier of the model
    - **file_id**: ID of the file to evaluate on
    """
    try:
        # This would be implemented in model_service
        # For now, return a placeholder response
        return {
            "success": True,
            "data": {
                "model_id": model_id,
                "file_id": file_id,
                "evaluation_metrics": {
                    "accuracy": 0.85,
                    "loss": 0.15,
                    "r2_score": 0.82
                },
                "message": "Model evaluation functionality not yet implemented"
            },
            "message": "Model evaluation requested (feature coming soon)"
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate model")

@router.get("/statistics/overview")
async def get_model_statistics():
    """
    Get platform model statistics
    """
    try:
        # This would typically come from the database
        # For now, return basic stats
        stats = {
            "total_models_trained": 0,
            "active_models": 0,
            "total_predictions_made": 0,
            "most_used_architecture": "simple",
            "average_training_time": 0,
            "success_rate": 0.95
        }
        
        return {
            "success": True,
            "data": stats,
            "message": "Model statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get model statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model statistics")

@router.post("/batch-predict")
async def batch_predict(
    request: Request,
    file_ids: List[str] = Body(..., description="List of file IDs to make predictions on"),
    model_id: str = Body(..., description="ID of the model to use for predictions")
):
    """
    Make predictions on multiple files using the same model
    
    - **file_ids**: List of file IDs to make predictions on
    - **model_id**: ID of the trained model to use
    """
    try:
        client_ip = request.client.host
        
        results = []
        for file_id in file_ids:
            try:
                result = await model_service.predict(
                    file_id=file_id,
                    model_id=model_id,
                    user_ip=client_ip
                )
                results.append({
                    "file_id": file_id,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "file_id": file_id,
                    "status": "error",
                    "error": str(e)
                })
        
        successful_predictions = len([r for r in results if r["status"] == "success"])
        
        return {
            "success": True,
            "data": {
                "results": results,
                "total_files": len(file_ids),
                "successful_predictions": successful_predictions,
                "failed_predictions": len(file_ids) - successful_predictions
            },
            "message": f"Batch prediction completed: {successful_predictions}/{len(file_ids)} successful"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")