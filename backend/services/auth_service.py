"""Authentication service for JWT-based authentication."""

import os
import hashlib
import hmac
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from functools import wraps

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
    
    @classmethod
    def create_access_token(cls, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token.
        
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
        
        to_encode.update({"exp": expire.timestamp(), "iat": datetime.utcnow().timestamp()})
        
        return cls._encode_jwt(to_encode)
    
    @classmethod
    def verify_token(cls, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is malformed
            TokenExpiredError: If token is expired
        """
        try:
            payload = cls._decode_jwt(token)
            
            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                raise TokenExpiredError("Token has expired")
            
            return payload
            
        except json.JSONDecodeError:
            raise InvalidTokenError("Invalid token format")
        except Exception as e:
            raise InvalidTokenError(f"Token verification failed: {e}")
    
    @classmethod
    def _encode_jwt(cls, payload: Dict[str, Any]) -> str:
        """Encode JWT token (simplified implementation).
        
        Args:
            payload: Data to encode
            
        Returns:
            JWT token string
        """
        # Create header
        header = {"alg": cls.ALGORITHM, "typ": "JWT"}
        
        # Base64 encode header and payload
        header_encoded = base64.urlsafe_b64encode(
            json.dumps(header, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        payload_encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        # Create signature
        message = f"{header_encoded}.{payload_encoded}"
        signature = hmac.new(
            cls.SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_encoded = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{message}.{signature_encoded}"
    
    @classmethod
    def _decode_jwt(cls, token: str) -> Dict[str, Any]:
        """Decode JWT token (simplified implementation).
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload
            
        Raises:
            InvalidTokenError: If token is invalid
        """
        try:
            parts = token.split('.')
            if len(parts) != 3:
                raise InvalidTokenError("Invalid token format")
            
            header_encoded, payload_encoded, signature_encoded = parts
            
            # Verify signature
            message = f"{header_encoded}.{payload_encoded}"
            expected_signature = hmac.new(
                cls.SECRET_KEY.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            
            # Add padding if needed
            signature_encoded += '=' * (4 - len(signature_encoded) % 4)
            provided_signature = base64.urlsafe_b64decode(signature_encoded)
            
            if not hmac.compare_digest(expected_signature, provided_signature):
                raise InvalidTokenError("Invalid token signature")
            
            # Decode payload
            payload_encoded += '=' * (4 - len(payload_encoded) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_encoded)
            payload = json.loads(payload_bytes.decode())
            
            return payload
            
        except Exception as e:
            raise InvalidTokenError(f"Token decoding failed: {e}")
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password with salt.
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = os.urandom(32).hex()
        
        # Use PBKDF2 for password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed_password: Stored hashed password
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = AuthenticationService.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hashed_password)
    
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