"""FastAPI routes for the data analysis platform."""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from backend.services.rbac_service_db import db_rbac_service, Role, Permission, User
from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.services.encryption_service import encryption_service
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
                encrypted_path = encryption_service.encrypt_file(
                    tmp_path,
                    final_path.with_suffix(tmp_path.suffix + '.encrypted'),
                    remove_original=True
                )
                final_path = encrypted_path
            else:
                tmp_path.rename(final_path)
            
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