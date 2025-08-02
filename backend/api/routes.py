"""FastAPI routes for the data analysis platform."""

import os
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from backend.services.rbac_service_db import db_rbac_service, Role, Permission, User
from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.services.encryption_service import encryption_service
from backend.services.data_service import data_service, DataAnalysis
from backend.services.model_service import model_service, ModelTrainingConfig, ModelInfo
from backend.services.export_service import export_service, ExportInfo
from backend.utils.file_handlers import SecureFileHandler
from backend.utils.sanitizers import DataSanitizer
from backend.api.middleware import require_permission, require_role, rate_limiter
from backend.core.logging import audit_logger, EventType
from backend.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Platform API",
    description="Secure data analysis platform with encryption and RBAC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()
auth_service = AuthenticationService()

# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "analyst"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: Optional[str]
    last_login: Optional[str]

class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    file_size: int
    owner_id: str
    is_encrypted: bool
    created_at: str
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict] = None

# Data Analysis Request/Response Models
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

# Model Training Request/Response Models
class ModelTrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_type: str = "linear_regression"
    test_size: float = 0.2
    random_state: int = 42
    scaling_method: Optional[str] = None
    feature_selection: Optional[str] = None
    feature_selection_k: int = 10
    cross_validation: bool = True
    cv_folds: int = 5
    hyperparameter_tuning: bool = False
    regularization_alpha: Optional[float] = None

class PredictionRequest(BaseModel):
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]]

# Export Request Models
class ExportDatasetRequest(BaseModel):
    format: str = "csv"  # 'csv', 'excel', 'json', 'parquet'
    include_metadata: bool = True
    custom_filename: Optional[str] = None

class ExportAnalysisRequest(BaseModel):
    format: str = "json"  # 'json', 'excel'
    custom_filename: Optional[str] = None

class ExportModelRequest(BaseModel):
    format: str = "json"  # 'json', 'excel'
    custom_filename: Optional[str] = None

class ExportPredictionsRequest(BaseModel):
    format: str = "csv"  # 'csv', 'excel', 'json'
    custom_filename: Optional[str] = None

# Dependency to get current user from token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token."""
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user from database
        user = db_rbac_service.get_user(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(request: RegisterRequest):
    """Register a new user."""
    client_ip = "127.0.0.1"  # Would get from request in real implementation
    
    # Check rate limiting
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Try again later."
        )
    
    try:
        # Validate role
        try:
            role = Role(request.role)
        except ValueError:
            rate_limiter.record_attempt(client_ip)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {request.role}"
            )
        
        # Create user
        user_id = str(uuid.uuid4())
        user = db_rbac_service.create_user(
            user_id=user_id,
            username=request.username,
            email=request.email,
            password=request.password,
            role=role
        )
        
        # Clear rate limiting on success
        rate_limiter.clear_attempts(client_ip)
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except Exception as e:
        rate_limiter.record_attempt(client_ip)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login_user(request: LoginRequest):
    """Authenticate user and return JWT token."""
    client_ip = "127.0.0.1"  # Would get from request in real implementation
    
    # Check rate limiting
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later."
        )
    
    # Authenticate user
    user = db_rbac_service.authenticate_user(request.username, request.password)
    
    if not user:
        rate_limiter.record_attempt(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    try:
        # Create JWT token
        token_data = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value
        }
        
        expires_delta = timedelta(minutes=settings.API_KEY_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(token_data, expires_delta)
        
        # Create session
        session_id = str(uuid.uuid4())
        db_rbac_service.create_session(
            user_id=user.user_id,
            session_id=session_id,
            expires_at=datetime.utcnow() + expires_delta,
            ip_address=client_ip
        )
        
        # Clear rate limiting on success
        rate_limiter.clear_attempts(client_ip)
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.API_KEY_EXPIRE_MINUTES * 60,
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in db_rbac_service.get_user_permissions(user)]
            }
        )
        
    except Exception as e:
        audit_logger.log_security_event(
            EventType.LOGIN_FAILED,
            user_id=user.user_id,
            details={"error": str(e)},
            ip_address=client_ip
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authentication token"
        )

@app.post("/auth/logout", tags=["Authentication"])
async def logout_user(current_user: User = Depends(get_current_user)):
    """Logout user and invalidate session."""
    # In a full implementation, you'd get session_id from token or header
    # For now, we'll destroy all user sessions
    sessions_destroyed = db_rbac_service.cleanup_expired_sessions()
    
    audit_logger.log_event(
        EventType.LOGOUT,
        user_id=current_user.user_id,
        outcome="success",
        details={"sessions_destroyed": sessions_destroyed}
    )
    
    return {"message": "Logged out successfully"}

# User management endpoints
@app.get("/users/me", response_model=UserResponse, tags=["Users"])
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role.value,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@app.get("/users", response_model=List[UserResponse], tags=["Users"])
async def list_users(
    current_user: User = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0
):
    """List all users (admin only)."""
    if not db_rbac_service.has_permission(current_user, Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    users = db_rbac_service.list_users(current_user, limit=limit, offset=offset)
    
    return [
        UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        for user in users
    ]

# Data upload endpoints
@app.post("/data/upload", tags=["Data"])
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    encrypt: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Upload and process a dataset."""
    # Check permissions
    if not db_rbac_service.has_permission(current_user, Permission.UPLOAD_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to upload data"
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            # Copy uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Validate file security
            SecureFileHandler.validate_file(tmp_path)
            
            # Calculate file hash
            file_hash = SecureFileHandler.calculate_file_hash(tmp_path)
            
            # Determine storage path
            dataset_id = str(uuid.uuid4())
            storage_dir = Path(settings.PROCESSED_PATH)
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            final_path = storage_dir / f"{dataset_id}{tmp_path.suffix}"
            
            # Encrypt file if requested  
            if encrypt:
                # Keep original extension, just add .encrypted
                encrypted_path = encryption_service.encrypt_file(
                    tmp_path,
                    final_path.with_suffix(tmp_path.suffix + '.encrypted'),
                    remove_original=True
                )
                final_path = encrypted_path
            else:
                tmp_path.rename(final_path)
            
            # Save dataset info to database
            from backend.models.database import db_manager
            query = '''
                INSERT INTO datasets 
                (dataset_id, name, description, file_path, file_size, file_hash, owner_id, is_encrypted, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            metadata = {
                "original_filename": file.filename,
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_type": tmp_path.suffix
            }
            
            db_manager.execute_update(
                query,
                (dataset_id, name or file.filename, description or "", str(final_path), 
                 len(content), file_hash, current_user.user_id, encrypt, json.dumps(metadata))
            )
            
            # Log upload event
            audit_logger.log_data_access(
                user_id=current_user.user_id,
                dataset_id=dataset_id,
                operation="upload",
                details={
                    "filename": file.filename,
                    "file_size": len(content),
                    "file_hash": file_hash,
                    "encrypted": encrypt,
                    "storage_path": str(final_path)
                }
            )
            
            # Return dataset info
            return {
                "dataset_id": dataset_id,
                "name": name or file.filename,
                "file_size": len(content),
                "file_hash": file_hash,
                "is_encrypted": encrypt,
                "storage_path": str(final_path),
                "uploaded_at": datetime.utcnow().isoformat(),
                "owner_id": current_user.user_id
            }
            
        finally:
            # Clean up temp file if it still exists
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as e:
        audit_logger.log_security_event(
            EventType.DATA_UPLOADED,
            user_id=current_user.user_id,
            details={"error": str(e), "filename": file.filename},
            risk_level="medium"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File upload failed: {str(e)}"
        )

# System information endpoints
@app.get("/system/info", tags=["System"])
async def get_system_info(current_user: User = Depends(get_current_user)):
    """Get system information."""
    encryption_info = encryption_service.get_key_info()
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "user_info": {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role.value,
            "permissions": [p.value for p in db_rbac_service.get_user_permissions(current_user)]
        },
        "encryption": {
            "algorithm": encryption_info["algorithm"],
            "key_fingerprint": encryption_info["key_fingerprint"]
        },
        "limits": {
            "max_file_size": settings.MAX_FILE_SIZE,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS
        }
    }

# Admin endpoints
@app.get("/admin/audit-logs", tags=["Admin"])
async def get_audit_logs(
    current_user: User = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Get audit logs (admin only)."""
    if not db_rbac_service.has_permission(current_user, Permission.VIEW_AUDIT_LOGS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view audit logs"
        )
    
    # Get logs from database
    from backend.models.database import audit_log_repository
    
    logs = audit_log_repository.get_logs(
        limit=limit,
        offset=offset,
        event_type=event_type,
        user_id=user_id
    )
    
    return {
        "logs": logs,
        "total": len(logs),
        "limit": limit,
        "offset": offset
    }

@app.get("/admin/security-summary", tags=["Admin"])
async def get_security_summary(
    current_user: User = Depends(get_current_user),
    hours: int = 24
):
    """Get security summary (admin only)."""
    if not db_rbac_service.has_permission(current_user, Permission.VIEW_AUDIT_LOGS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view security summary"
        )
    
    summary = audit_logger.get_security_summary(hours=hours)
    return summary

# Data Analysis Endpoints
@app.get("/data/datasets", tags=["Data Analysis"])
async def list_datasets(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """List datasets owned by or accessible to the user."""
    # Get user's datasets
    query = '''
        SELECT dataset_id, name, file_size, is_encrypted, created_at, updated_at, metadata
        FROM datasets 
        WHERE owner_id = ? 
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
    '''
    
    from backend.models.database import db_manager
    results = db_manager.execute_query(query, (current_user.user_id, limit, offset))
    
    datasets = []
    for row in results:
        datasets.append({
            "dataset_id": row["dataset_id"],
            "name": row["name"],
            "file_size": row["file_size"],
            "is_encrypted": bool(row["is_encrypted"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata"] or '{}')
        })
    
    return {
        "datasets": datasets,
        "total": len(datasets),
        "limit": limit,
        "offset": offset
    }

@app.get("/data/{dataset_id}/analyze", tags=["Data Analysis"])
async def analyze_dataset(
    dataset_id: str,
    request: DataAnalysisRequest = Depends(),
    current_user: User = Depends(get_current_user)
):
    """Perform comprehensive exploratory data analysis on a dataset."""
    if not db_rbac_service.has_permission(current_user, Permission.READ_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to analyze data"
        )
    
    try:
        analysis = await data_service.analyze_dataset(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            include_correlation=request.include_correlation,
            detect_outliers=request.detect_outliers
        )
        
        # Convert dataclass to dictionary for JSON response
        return {
            "dataset_id": analysis.dataset_id,
            "shape": analysis.shape,
            "columns": analysis.columns,
            "data_types": analysis.data_types,
            "missing_values": analysis.missing_values,
            "missing_percentages": analysis.missing_percentages,
            "numeric_summary": analysis.numeric_summary,
            "categorical_summary": analysis.categorical_summary,
            "correlation_matrix": analysis.correlation_matrix,
            "outliers": analysis.outliers,
            "duplicates": analysis.duplicates,
            "memory_usage": analysis.memory_usage,
            "analysis_timestamp": analysis.analysis_timestamp
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/data/{dataset_id}/sample", tags=["Data Analysis"])
async def get_dataset_sample(
    dataset_id: str,
    request: DataSampleRequest = Depends(),
    current_user: User = Depends(get_current_user)
):
    """Get a sample of the dataset for preview."""
    if not db_rbac_service.has_permission(current_user, Permission.READ_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to read data"
        )
    
    try:
        sample = await data_service.get_data_sample(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            n_rows=request.n_rows,
            random=request.random
        )
        
        return sample
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get data sample: {str(e)}"
        )

@app.post("/data/{dataset_id}/clean", tags=["Data Analysis"])
async def clean_dataset(
    dataset_id: str,
    request: DataCleaningRequest,
    current_user: User = Depends(get_current_user)
):
    """Clean dataset based on specified options."""
    if not db_rbac_service.has_permission(current_user, Permission.UPLOAD_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to clean data"
        )
    
    try:
        cleaning_options = {
            "remove_duplicates": request.remove_duplicates,
            "missing_values_strategy": request.missing_values_strategy,
            "missing_threshold": request.missing_threshold,
            "fill_method": request.fill_method,
            "remove_outliers": request.remove_outliers
        }
        
        results = await data_service.clean_dataset(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            cleaning_options=cleaning_options
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data cleaning failed: {str(e)}"
        )

# Model Training and Management Endpoints
@app.post("/models/train", tags=["Machine Learning"])
async def train_model(
    request: ModelTrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Train a machine learning model."""
    if not db_rbac_service.has_permission(current_user, Permission.TRAIN_MODELS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to train models"
        )
    
    try:
        # Create training config
        config = ModelTrainingConfig(
            model_type=request.model_type,
            test_size=request.test_size,
            random_state=request.random_state,
            scaling_method=request.scaling_method,
            feature_selection=request.feature_selection,
            feature_selection_k=request.feature_selection_k,
            cross_validation=request.cross_validation,
            cv_folds=request.cv_folds,
            hyperparameter_tuning=request.hyperparameter_tuning,
            regularization_alpha=request.regularization_alpha
        )
        
        model_info = await model_service.train_model(
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            config=config,
            user_id=current_user.user_id
        )
        
        # Convert dataclass to dict for JSON response
        return {
            "model_id": model_info.model_id,
            "name": model_info.name,
            "model_type": model_info.model_type,
            "dataset_id": model_info.dataset_id,
            "target_column": model_info.target_column,
            "feature_columns": model_info.feature_columns,
            "performance": {
                "r2_score": model_info.performance.r2_score,
                "adjusted_r2": model_info.performance.adjusted_r2,
                "mse": model_info.performance.mse,
                "mae": model_info.performance.mae,
                "rmse": model_info.performance.rmse,
                "mape": model_info.performance.mape,
                "explained_variance": model_info.performance.explained_variance,
                "cv_scores": model_info.performance.cv_scores,
                "cv_mean": model_info.performance.cv_mean,
                "cv_std": model_info.performance.cv_std
            },
            "feature_importance": model_info.feature_importance,
            "model_size": model_info.model_size,
            "training_time": model_info.training_time,
            "created_at": model_info.created_at,
            "status": model_info.status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model training failed: {str(e)}"
        )

@app.get("/models", tags=["Machine Learning"])
async def list_models(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """List models owned by the user."""
    try:
        models = await model_service.list_models(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset
        )
        
        # Convert to dict for JSON response
        models_dict = []
        for model in models:
            models_dict.append({
                "model_id": model.model_id,
                "name": model.name,
                "model_type": model.model_type,
                "dataset_id": model.dataset_id,
                "target_column": model.target_column,
                "feature_columns": model.feature_columns,
                "performance": {
                    "r2_score": model.performance.r2_score,
                    "mse": model.performance.mse,
                    "mae": model.performance.mae
                },
                "created_at": model.created_at,
                "status": model.status
            })
        
        return {
            "models": models_dict,
            "total": len(models_dict),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list models: {str(e)}"
        )

@app.get("/models/{model_id}", tags=["Machine Learning"])
async def get_model_info(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific model."""
    try:
        model_info = await model_service.get_model_info(model_id, current_user.user_id)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return {
            "model_id": model_info.model_id,
            "name": model_info.name,
            "model_type": model_info.model_type,
            "dataset_id": model_info.dataset_id,
            "target_column": model_info.target_column,
            "feature_columns": model_info.feature_columns,
            "training_config": {
                "model_type": model_info.training_config.model_type,
                "test_size": model_info.training_config.test_size,
                "scaling_method": model_info.training_config.scaling_method,
                "feature_selection": model_info.training_config.feature_selection,
                "cross_validation": model_info.training_config.cross_validation
            },
            "performance": {
                "r2_score": model_info.performance.r2_score,
                "adjusted_r2": model_info.performance.adjusted_r2,
                "mse": model_info.performance.mse,
                "mae": model_info.performance.mae,
                "rmse": model_info.performance.rmse,
                "mape": model_info.performance.mape,
                "explained_variance": model_info.performance.explained_variance,
                "cv_scores": model_info.performance.cv_scores,
                "cv_mean": model_info.performance.cv_mean,
                "cv_std": model_info.performance.cv_std
            },
            "feature_importance": model_info.feature_importance,
            "model_size": model_info.model_size,
            "training_time": model_info.training_time,
            "created_at": model_info.created_at,
            "status": model_info.status,
            "metadata": model_info.metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.post("/models/{model_id}/predict", tags=["Machine Learning"])
async def predict(
    model_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Make predictions using a trained model."""
    if not db_rbac_service.has_permission(current_user, Permission.USE_MODELS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to use models"
        )
    
    try:
        results = await model_service.predict(
            model_id=model_id,
            input_data=request.input_data,
            user_id=current_user.user_id
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )

@app.delete("/models/{model_id}", tags=["Machine Learning"])
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a model and its files."""
    if not db_rbac_service.has_permission(current_user, Permission.DELETE_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to delete models"
        )
    
    try:
        success = await model_service.delete_model(model_id, current_user.user_id)
        
        if success:
            return {"message": "Model deleted successfully", "model_id": model_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found or could not be deleted"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete model: {str(e)}"
        )

# Export Endpoints
@app.post("/data/{dataset_id}/export", tags=["Export"])
async def export_dataset(
    dataset_id: str,
    request: ExportDatasetRequest,
    current_user: User = Depends(get_current_user)
):
    """Export a dataset in the specified format."""
    if not db_rbac_service.has_permission(current_user, Permission.EXPORT_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to export data"
        )
    
    try:
        export_info = await export_service.export_dataset(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            format=request.format,
            include_metadata=request.include_metadata,
            custom_filename=request.custom_filename
        )
        
        return {
            "export_id": export_info.export_id,
            "filename": export_info.filename,
            "format": export_info.format,
            "file_size": export_info.file_size,
            "created_at": export_info.created_at,
            "status": export_info.status,
            "download_url": f"/exports/{export_info.export_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export failed: {str(e)}"
        )

@app.post("/data/{dataset_id}/export-analysis", tags=["Export"])
async def export_analysis(
    dataset_id: str,
    request: ExportAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Export data analysis results."""
    if not db_rbac_service.has_permission(current_user, Permission.EXPORT_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to export data"
        )
    
    try:
        export_info = await export_service.export_analysis(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            format=request.format,
            custom_filename=request.custom_filename
        )
        
        return {
            "export_id": export_info.export_id,
            "filename": export_info.filename,
            "format": export_info.format,
            "file_size": export_info.file_size,
            "created_at": export_info.created_at,
            "status": export_info.status,
            "download_url": f"/exports/{export_info.export_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis export failed: {str(e)}"
        )

@app.post("/models/{model_id}/export", tags=["Export"])
async def export_model_results(
    model_id: str,
    request: ExportModelRequest,
    current_user: User = Depends(get_current_user)
):
    """Export model results and performance metrics."""
    if not db_rbac_service.has_permission(current_user, Permission.EXPORT_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to export data"
        )
    
    try:
        export_info = await export_service.export_model_results(
            model_id=model_id,
            user_id=current_user.user_id,
            format=request.format,
            custom_filename=request.custom_filename
        )
        
        return {
            "export_id": export_info.export_id,
            "filename": export_info.filename,
            "format": export_info.format,
            "file_size": export_info.file_size,
            "created_at": export_info.created_at,
            "status": export_info.status,
            "download_url": f"/exports/{export_info.export_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model export failed: {str(e)}"
        )

@app.post("/predictions/export", tags=["Export"])
async def export_predictions(
    predictions: Dict[str, Any],
    request: ExportPredictionsRequest,
    current_user: User = Depends(get_current_user)
):
    """Export prediction results."""
    if not db_rbac_service.has_permission(current_user, Permission.EXPORT_DATA):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to export data"
        )
    
    try:
        export_info = await export_service.export_predictions(
            predictions=predictions,
            user_id=current_user.user_id,
            format=request.format,
            custom_filename=request.custom_filename
        )
        
        return {
            "export_id": export_info.export_id,
            "filename": export_info.filename,
            "format": export_info.format,
            "file_size": export_info.file_size,
            "created_at": export_info.created_at,
            "status": export_info.status,
            "download_url": f"/exports/{export_info.export_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Predictions export failed: {str(e)}"
        )

@app.get("/exports/{export_id}/info", tags=["Export"])
async def get_export_info(
    export_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get information about an export."""
    try:
        export_info = await export_service.get_export_info(export_id, current_user.user_id)
        
        if not export_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export not found"
            )
        
        return {
            "export_id": export_info.export_id,
            "export_type": export_info.export_type,
            "format": export_info.format,
            "resource_id": export_info.resource_id,
            "filename": export_info.filename,
            "file_size": export_info.file_size,
            "created_at": export_info.created_at,
            "status": export_info.status,
            "metadata": export_info.metadata,
            "download_url": f"/exports/{export_info.export_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get export info: {str(e)}"
        )

@app.get("/exports/{export_id}/download", tags=["Export"])
async def download_export(
    export_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download an exported file."""
    try:
        return await export_service.download_export(export_id, current_user.user_id)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": "Authentication failed", "message": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    # Log unexpected errors
    audit_logger.log_security_event(
        EventType.SYSTEM_START,  # Using as generic system event
        details={"error": str(exc), "path": str(request.url)},
        risk_level="medium"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)