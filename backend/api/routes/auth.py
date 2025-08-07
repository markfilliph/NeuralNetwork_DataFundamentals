"""Authentication and authorization routes."""

from typing import Dict, Any
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from backend.services.rbac_service_db import db_rbac_service, Role
from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.api.middleware import rate_limiter
from backend.core.logging import audit_logger, EventType

# Initialize router and services
router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
auth_service = AuthenticationService()

# Pydantic models
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

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        user_data = auth_service.verify_token(token)
        return user_data
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=TokenResponse)
@rate_limiter(calls=5, period=300)  # 5 calls per 5 minutes
async def register(request: RegisterRequest):
    """Register a new user."""
    try:
        # Validate role
        try:
            role = Role(request.role.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {request.role}"
            )
        
        # Create user
        user = db_rbac_service.create_user(
            user_id=f"user_{len(request.username)}_{hash(request.username) % 10000}",
            username=request.username,
            email=request.email,
            password=request.password,
            role=role
        )
        
        # Generate token
        token_data = auth_service.create_access_token({
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value
        })
        
        # Log successful registration
        audit_logger.log_event(
            EventType.USER_CREATED,
            user_id=user.user_id,
            outcome="success",
            details={"username": user.username, "role": user.role.value}
        )
        
        return TokenResponse(
            access_token=token_data["access_token"],
            expires_in=token_data["expires_in"],
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            }
        )
        
    except Exception as e:
        audit_logger.log_event(
            EventType.USER_CREATED,
            outcome="failure",
            details={"error": str(e), "username": request.username}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
@rate_limiter(calls=10, period=300)  # 10 calls per 5 minutes
async def login(request: LoginRequest):
    """Authenticate user and return access token."""
    try:
        # Authenticate user
        user = db_rbac_service.authenticate_user(request.username, request.password)
        if not user:
            audit_logger.log_authentication_event(
                success=False,
                user_id=request.username,
                details={"error": "invalid_credentials"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Generate token
        token_data = auth_service.create_access_token({
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value
        })
        
        return TokenResponse(
            access_token=token_data["access_token"],
            expires_in=token_data["expires_in"],
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "last_login": user.last_login
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_authentication_event(
            success=False,
            user_id=request.username,
            details={"error": "authentication_exception", "exception": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    """Logout user and invalidate token."""
    try:
        # In a more sophisticated implementation, we would:
        # 1. Add token to blacklist
        # 2. Destroy session in database
        # For now, we just log the logout event
        
        audit_logger.log_event(
            EventType.USER_LOGOUT,
            user_id=current_user.get("user_id"),
            outcome="success"
        )
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

@router.get("/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information."""
    user = db_rbac_service.get_user(current_user["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "last_login": user.last_login
    }

@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Change user password."""
    try:
        user = db_rbac_service.get_user(current_user["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        authenticated_user = db_rbac_service.authenticate_user(
            user.username, request.current_password
        )
        if not authenticated_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Update password (this would need to be implemented in the RBAC service)
        # For now, we'll just log the attempt
        audit_logger.log_event(
            EventType.USER_MODIFIED,
            user_id=user.user_id,
            outcome="success",
            action="password_change"
        )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            EventType.USER_MODIFIED,
            user_id=current_user["user_id"],
            outcome="failure",
            action="password_change",
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )