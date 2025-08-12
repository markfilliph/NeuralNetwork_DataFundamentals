"""Data upload and analysis routes."""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel

from backend.services.rbac_service_db import db_rbac_service, Permission
from backend.services.data_service import data_service, DataAnalysis
from backend.services.encryption_service import encryption_service
from backend.utils.file_handlers import SecureFileHandler
from backend.utils.sanitizers import DataSanitizer
from backend.api.routes.auth import get_current_user
from backend.api.middleware import require_permission, rate_limiter
from backend.core.logging import audit_logger, EventType
from backend.core.config import settings

# Initialize router
router = APIRouter(prefix="/data", tags=["data"])

# Pydantic models
class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    file_size: int
    owner_id: str
    is_encrypted: bool
    created_at: str
    metadata: Dict[str, Any]

class DataAnalysisRequest(BaseModel):
    include_correlation: bool = True
    detect_outliers: bool = True

class DataSampleRequest(BaseModel):
    n_rows: int = 100
    random: bool = False

class DataCleaningRequest(BaseModel):
    remove_duplicates: bool = False
    missing_values_strategy: str = "none"  # 'none', 'drop_rows', 'drop_columns', 'fill_numeric'
    missing_threshold: float = 0.5
    fill_method: str = "mean"  # 'mean', 'median', 'mode'
    remove_outliers: bool = False

@router.post("/upload", response_model=DatasetResponse)
@rate_limiter(calls=10, period=3600)  # 10 uploads per hour
@require_permission(Permission.UPLOAD_DATA)
async def upload_file(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    """Upload and process a data file."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Process file upload
        file_handler = SecureFileHandler()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Validate and sanitize file
            SecureFileHandler.validate_file(Path(temp_file_path))
            validated_path = temp_file_path
            sanitized_path = DataSanitizer.sanitize_file(validated_path)
            
            # Store file securely
            stored_info = data_service.store_dataset(
                file_path=sanitized_path,
                dataset_id=dataset_id,
                owner_id=current_user["user_id"],
                original_filename=file.filename,
                encrypt=True
            )
            
            # Log successful upload
            audit_logger.log_event(
                EventType.DATA_UPLOADED,
                user_id=current_user["user_id"],
                outcome="success",
                resource=dataset_id,
                details={
                    "filename": file.filename,
                    "file_size": file.size,
                    "encrypted": True
                }
            )
            
            return DatasetResponse(
                dataset_id=dataset_id,
                name=file.filename,
                file_size=stored_info.file_size,
                owner_id=current_user["user_id"],
                is_encrypted=stored_info.is_encrypted,
                created_at=stored_info.created_at,
                metadata=stored_info.metadata
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.DATA_UPLOADED,
            user_id=current_user.get("user_id"),
            outcome="failure",
            details={"error": str(e), "filename": file.filename if file else "unknown"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

@router.get("/datasets")
@require_permission(Permission.READ_DATA)
async def list_datasets(current_user: Dict = Depends(get_current_user)):
    """List user's datasets."""
    try:
        datasets = data_service.list_user_datasets(current_user["user_id"])
        return {"datasets": datasets}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )

@router.get("/datasets/{dataset_id}")
@require_permission(Permission.READ_DATA)
async def get_dataset_info(
    dataset_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get dataset information."""
    try:
        dataset_info = data_service.get_dataset_info(dataset_id)
        
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Check ownership or admin permission
        user = db_rbac_service.get_user(current_user["user_id"])
        if (dataset_info.owner_id != current_user["user_id"] and 
            not db_rbac_service.has_permission(user, Permission.MANAGE_USERS)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset info: {str(e)}"
        )

@router.post("/datasets/{dataset_id}/analyze")
@require_permission(Permission.READ_DATA)
async def analyze_dataset(
    dataset_id: str,
    request: DataAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Perform exploratory data analysis on dataset."""
    try:
        # Verify dataset access
        dataset_info = data_service.get_dataset_info(dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Check ownership
        if dataset_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Perform analysis
        analysis = data_service.analyze_dataset(
            dataset_id=dataset_id,
            include_correlation=request.include_correlation,
            detect_outliers=request.detect_outliers
        )
        
        # Log analysis
        audit_logger.log_event(
            EventType.DATA_ANALYZED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=dataset_id
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.DATA_ANALYZED,
            user_id=current_user["user_id"],
            outcome="failure",
            resource=dataset_id,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data analysis failed: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/sample")
@require_permission(Permission.READ_DATA)
async def get_data_sample(
    dataset_id: str,
    request: DataSampleRequest = Depends(),
    current_user: Dict = Depends(get_current_user)
):
    """Get a sample of the dataset."""
    try:
        # Verify dataset access (same logic as analyze)
        dataset_info = data_service.get_dataset_info(dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        if dataset_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Get sample
        sample_data = data_service.get_dataset_sample(
            dataset_id=dataset_id,
            n_rows=request.n_rows,
            random=request.random
        )
        
        return {"sample": sample_data}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get data sample: {str(e)}"
        )

@router.delete("/datasets/{dataset_id}")
@require_permission(Permission.DELETE_DATA)
async def delete_dataset(
    dataset_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a dataset."""
    try:
        # Verify dataset exists and ownership
        dataset_info = data_service.get_dataset_info(dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        if dataset_info.owner_id != current_user["user_id"]:
            user = db_rbac_service.get_user(current_user["user_id"])
            if not db_rbac_service.has_permission(user, Permission.MANAGE_USERS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        
        # Delete dataset
        success = data_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset"
            )
        
        # Log deletion
        audit_logger.log_event(
            EventType.DATA_DELETED,
            user_id=current_user["user_id"],
            outcome="success",
            resource=dataset_id
        )
        
        return {"message": "Dataset deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.DATA_DELETED,
            user_id=current_user["user_id"],
            outcome="failure",
            resource=dataset_id,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset deletion failed: {str(e)}"
        )