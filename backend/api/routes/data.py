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
    id: str  # Frontend expects 'id' not 'dataset_id'
    filename: str  # Frontend expects 'filename' not 'name'
    description: str = ""  # Frontend expects description
    file_size: int
    owner_id: str
    is_encrypted: bool
    created_at: str
    status: str = "ready"  # Frontend expects status field
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
async def upload_file(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    """Upload and process a data file."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"File upload started - User: {current_user.get('username', 'unknown')}, File: {file.filename}, Size: {file.size}")
        
        # Validate file
        if not file.filename:
            logger.error("No filename provided")
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
            logger.info(f"Validating file: {file.filename} at {temp_file_path}")
            # Validate file
            SecureFileHandler.validate_file(Path(temp_file_path), file.filename)
            validated_path = temp_file_path
            logger.info(f"File validation successful")
            
            # Store file securely
            logger.info(f"Storing dataset with ID: {dataset_id}")
            stored_info = data_service.store_dataset(
                file_path=validated_path,
                dataset_id=dataset_id,
                owner_id=current_user["user_id"],
                original_filename=file.filename,
                encrypt=True
            )
            logger.info(f"Dataset stored successfully: {stored_info.dataset_id}")
            
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
                id=dataset_id,
                filename=file.filename,
                description=f"Uploaded {file.filename}",
                file_size=stored_info.file_size,
                owner_id=current_user["user_id"],
                is_encrypted=stored_info.is_encrypted,
                created_at=stored_info.created_at,
                status="ready",
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
        logger.error(f"File upload failed: {str(e)}", exc_info=True)
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

class PaginatedDatasetResponse(BaseModel):
    items: List[DatasetResponse]
    total: int
    page: int
    per_page: int
    pages: int

@router.get("/datasets", response_model=PaginatedDatasetResponse)
async def list_datasets(
    page: int = 1,
    per_page: int = 20,
    current_user: Dict = Depends(get_current_user)
):
    """List user's datasets with pagination."""
    try:
        # Get all datasets for the user
        all_datasets = data_service.list_user_datasets(current_user["user_id"])
        
        # Calculate pagination
        total = len(all_datasets)
        pages = (total + per_page - 1) // per_page  # Ceiling division
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get page slice
        page_datasets = all_datasets[start_idx:end_idx]
        
        # Convert to frontend format
        formatted_datasets = []
        for dataset in page_datasets:
            formatted_datasets.append(DatasetResponse(
                id=dataset["dataset_id"],
                filename=dataset.get("metadata", {}).get("original_filename", dataset["name"]),
                description=f"Dataset uploaded on {dataset['created_at'][:10]}",
                file_size=dataset["file_size"],
                owner_id=dataset["owner_id"],
                is_encrypted=dataset["is_encrypted"],
                created_at=dataset["created_at"],
                status="ready",
                metadata=dataset["metadata"]
            ))
        
        return PaginatedDatasetResponse(
            items=formatted_datasets,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )

@router.get("/datasets/{dataset_id}")
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
        analysis = await data_service.analyze_dataset(
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
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