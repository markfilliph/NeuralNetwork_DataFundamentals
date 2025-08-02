"""Machine learning model routes."""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from backend.services.rbac_service_db import db_rbac_service, Permission
from backend.services.model_service import model_service, ModelTrainingConfig, ModelInfo
from backend.services.data_service import data_service
from backend.api.routes.auth import get_current_user
from backend.api.middleware import require_permission, rate_limiter
from backend.core.logging import audit_logger, EventType

# Initialize router
router = APIRouter(prefix="/models", tags=["models"])

# Pydantic models
class ModelTrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_type: str = "linear_regression"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    hyperparameters: Optional[Dict[str, Any]] = None

class ModelResponse(BaseModel):
    model_id: str
    name: str
    model_type: str
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    status: str
    owner_id: str
    created_at: str
    trained_at: Optional[str]
    metrics: Dict[str, float]

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    dataset_id: str
    output_format: str = "json"  # 'json', 'csv'

@router.post("/train", response_model=ModelResponse)
@rate_limiter(calls=5, period=3600)  # 5 training jobs per hour
@require_permission(Permission.TRAIN_MODELS)
async def train_model(
    request: ModelTrainingRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Train a machine learning model."""
    try:
        # Verify dataset exists and access
        dataset_info = data_service.get_dataset_info(request.dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Check dataset access
        if dataset_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to dataset"
                )
        
        # Create training config
        config = ModelTrainingConfig(
            model_type=request.model_type,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            validation_size=request.validation_size,
            random_state=request.random_state,
            hyperparameters=request.hyperparameters or {}
        )
        
        # Train model
        model_info = model_service.train_model_from_dataset(
            dataset_id=request.dataset_id,
            config=config,
            owner_id=current_user["user_id"],
            model_name=f"{request.model_type}_{dataset_info.name}"
        )
        
        # Log training
        audit_logger.log_event(
            EventType.MODEL_TRAINED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=model_info.model_id,
            details={
                "model_type": request.model_type,
                "dataset_id": request.dataset_id,
                "target_column": request.target_column
            }
        )
        
        return ModelResponse(
            model_id=model_info.model_id,
            name=model_info.name,
            model_type=model_info.model_type,
            dataset_id=model_info.dataset_id,
            target_column=model_info.target_column,
            feature_columns=model_info.feature_columns,
            status=model_info.status,
            owner_id=model_info.owner_id,
            created_at=model_info.created_at,
            trained_at=model_info.trained_at,
            metrics=model_info.metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.MODEL_TRAINED,
            user_id=current_user["user_id"],
            outcome="failure",
            details={
                "error": str(e),
                "dataset_id": request.dataset_id,
                "model_type": request.model_type
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )

@router.get("/", response_model=List[ModelResponse])
@require_permission(Permission.VIEW_MODELS)
async def list_models(current_user: Dict = Depends(get_current_user)):
    """List user's trained models."""
    try:
        models = model_service.list_user_models(current_user["user_id"])
        
        return [
            ModelResponse(
                model_id=model.model_id,
                name=model.name,
                model_type=model.model_type,
                dataset_id=model.dataset_id,
                target_column=model.target_column,
                feature_columns=model.feature_columns,
                status=model.status,
                owner_id=model.owner_id,
                created_at=model.created_at,
                trained_at=model.trained_at,
                metrics=model.metrics
            )
            for model in models
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

@router.get("/{model_id}", response_model=ModelResponse)
@require_permission(Permission.VIEW_MODELS)
async def get_model(
    model_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get model information."""
    try:
        model_info = model_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Check access
        if model_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        return ModelResponse(
            model_id=model_info.model_id,
            name=model_info.name,
            model_type=model_info.model_type,
            dataset_id=model_info.dataset_id,
            target_column=model_info.target_column,
            feature_columns=model_info.feature_columns,
            status=model_info.status,
            owner_id=model_info.owner_id,
            created_at=model_info.created_at,
            trained_at=model_info.trained_at,
            metrics=model_info.metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )

@router.post("/{model_id}/predict")
@require_permission(Permission.VIEW_MODELS)
async def predict(
    model_id: str,
    request: PredictionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Make a prediction using the model."""
    try:
        # Verify model access
        model_info = model_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        if model_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Make prediction
        prediction = model_service.predict_single(model_id, request.features)
        
        # Log prediction
        audit_logger.log_event(
            EventType.MODEL_USED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=model_id,
            action="predict_single"
        )
        
        return {
            "model_id": model_id,
            "prediction": prediction,
            "features": request.features
        }
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.MODEL_USED,
            user_id=current_user["user_id"],
            outcome="failure",
            resource=model_id,
            action="predict_single",
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/{model_id}/predict-batch")
@require_permission(Permission.VIEW_MODELS)
async def predict_batch(
    model_id: str,
    request: BatchPredictionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Make batch predictions on a dataset."""
    try:
        # Verify model and dataset access
        model_info = model_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        dataset_info = data_service.get_dataset_info(request.dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Check access
        if (model_info.owner_id != current_user["user_id"] or 
            dataset_info.owner_id != current_user["user_id"]):
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Make batch predictions
        predictions = model_service.predict_batch(
            model_id=model_id,
            dataset_id=request.dataset_id,
            output_format=request.output_format
        )
        
        # Log batch prediction
        audit_logger.log_event(
            EventType.MODEL_USED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=model_id,
            action="predict_batch",
            details={"dataset_id": request.dataset_id}
        )
        
        return {
            "model_id": model_id,
            "dataset_id": request.dataset_id,
            "predictions": predictions,
            "format": request.output_format
        }
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.MODEL_USED,
            user_id=current_user["user_id"],
            outcome="failure",
            resource=model_id,
            action="predict_batch",
            details={"error": str(e), "dataset_id": request.dataset_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.delete("/{model_id}")
@require_permission(Permission.DELETE_MODELS)
async def delete_model(
    model_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a trained model."""
    try:
        # Verify model exists and ownership
        model_info = model_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        if model_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Delete model
        success = model_service.delete_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete model"
            )
        
        # Log deletion
        audit_logger.log_event(
            EventType.MODEL_DELETED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=model_id
        )
        
        return {"message": "Model deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.MODEL_DELETED,
            user_id=current_user["user_id"],
            outcome="failure",
            resource=model_id,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model deletion failed: {str(e)}"
        )