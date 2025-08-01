"""Authentication service for JWT-based authentication."""

import os
import hmac
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from functools import wraps

from jose import JWTError, jwt
from passlib.context import CryptContext
from backend.core.config import settings
from backend.core.exceptions import SecurityError


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token is expired."""
    pass


class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid."""
    pass


class AuthenticationService:
    """Handles JWT-based authentication and password management."""
    
    SECRET_KEY = settings.SECRET_KEY
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = settings.API_KEY_EXPIRE_MINUTES
    
    # Initialize password context with bcrypt
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    @classmethod
    def create_access_token(cls, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token using jose library.
        
        Args:
            data: Payload data to encode in token
            expires_delta: Optional custom expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=cls.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        return jwt.encode(to_encode, cls.SECRET_KEY, algorithm=cls.ALGORITHM)
    
    @classmethod
    def verify_token(cls, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token using jose library.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is malformed
            TokenExpiredError: If token is expired
        """
        try:
            payload = jwt.decode(token, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
            return payload
            
        except JWTError as e:
            if "expired" in str(e).lower():
                raise TokenExpiredError("Token has expired")
            else:
                raise InvalidTokenError(f"Token verification failed: {e}")
        except Exception as e:
            raise InvalidTokenError(f"Token verification failed: {e}")
    
    
    @classmethod
    def hash_password(cls, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password using bcrypt via passlib.
        
        Args:
            password: Plain text password
            salt: Optional salt (ignored - bcrypt handles salt internally)
            
        Returns:
            Tuple of (hashed_password, salt) - salt is empty string for bcrypt
        """
        # bcrypt handles salt internally, so we don't need to manage it
        hashed = cls.pwd_context.hash(password)
        return hashed, ""
    
    @classmethod
    def verify_password(cls, password: str, hashed_password: str, salt: str = "") -> bool:
        """Verify a password against its bcrypt hash.
        
        Args:
            password: Plain text password
            hashed_password: Stored bcrypt hashed password
            salt: Salt (ignored for bcrypt compatibility)
            
        Returns:
            True if password matches, False otherwise
        """
        return cls.pwd_context.verify(password, hashed_password)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key.
        
        Returns:
            Random API key string
        """
        return os.urandom(32).hex()


def require_auth(f):
    """Decorator to require authentication for endpoints.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function that checks authentication
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # This would be implemented with actual request context
        # For now, it's a placeholder that shows the pattern
        token = kwargs.get('token') or (args[0] if args else None)
        
        if not token:
            raise AuthenticationError("Authentication token required")
        
        try:
            payload = AuthenticationService.verify_token(token)
            kwargs['current_user'] = payload
            return f(*args, **kwargs)
        except (InvalidTokenError, TokenExpiredError) as e:
            raise AuthenticationError(f"Authentication failed: {e}")
    
    return decorated_function