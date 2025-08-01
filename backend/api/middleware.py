"""Authentication and authorization middleware."""

import re
from typing import Dict, Any, Optional, Callable
from functools import wraps

from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.services.rbac_service import rbac_service, Permission, Role
from backend.core.exceptions import SecurityError


class AuthMiddleware:
    """Authentication middleware for request processing."""
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/health",
        "/login",
        "/register",
        "/docs",
        "/openapi.json"
    }
    
    def __init__(self):
        """Initialize authentication middleware."""
        self.auth_service = AuthenticationService()
        self.rbac_service = rbac_service
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request for authentication.
        
        Args:
            request_data: Request data containing headers, path, etc.
            
        Returns:
            Request data with user context added
            
        Raises:
            AuthenticationError: If authentication fails
        """
        path = request_data.get("path", "")
        
        # Skip authentication for public endpoints
        if self._is_public_endpoint(path):
            return request_data
        
        # Extract token from headers
        headers = request_data.get("headers", {})
        token = self._extract_token(headers)
        
        if not token:
            raise AuthenticationError("Authentication token required")
        
        # Verify token
        try:
            payload = self.auth_service.verify_token(token)
            user_id = payload.get("user_id")
            
            if not user_id:
                raise AuthenticationError("Invalid token payload")
            
            # Get user from RBAC service
            user = self.rbac_service.get_user(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            if not user.is_active:
                raise AuthenticationError("User account is deactivated")
            
            # Add user context to request
            request_data["current_user"] = payload
            request_data["current_user_obj"] = user
            
            return request_data
            
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public.
        
        Args:
            path: Request path
            
        Returns:
            True if endpoint is public
        """
        # Exact match
        if path in self.PUBLIC_ENDPOINTS:
            return True
        
        # Pattern matching for dynamic routes
        public_patterns = [
            r"^/static/.*",
            r"^/docs.*",
            r"^/health.*"
        ]
        
        for pattern in public_patterns:
            if re.match(pattern, path):
                return True
        
        return False
    
    def _extract_token(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract JWT token from request headers.
        
        Args:
            headers: Request headers
            
        Returns:
            JWT token string or None
        """
        # Check Authorization header
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check custom header
        return headers.get("X-API-Token")


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # In a real implementation, this would extract from request context
        # For testing, we expect token in kwargs
        token = kwargs.get('token')
        
        if not token:
            raise AuthenticationError("Authentication token required")
        
        try:
            auth_service = AuthenticationService()
            payload = auth_service.verify_token(token)
            
            user_id = payload.get("user_id")
            if not user_id:
                raise AuthenticationError("Invalid token payload")
            
            user = rbac_service.get_user(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Add user context
            kwargs['current_user'] = payload
            kwargs['current_user_obj'] = user
            
            return func(*args, **kwargs)
            
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {e}")
    
    return wrapper


def require_permission(permission: Permission) -> Callable:
    """Decorator to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            user = kwargs.get('current_user_obj')
            
            if not user:
                raise AuthenticationError("User context not found")
            
            if not rbac_service.has_permission(user, permission):
                raise SecurityError(f"Permission denied: {permission.value} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: Role) -> Callable:
    """Decorator to require specific role or higher.
    
    Args:
        role: Minimum required role
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            user = kwargs.get('current_user_obj')
            
            if not user:
                raise AuthenticationError("User context not found")
            
            # Role hierarchy check
            role_hierarchy = {
                Role.VIEWER: 1,
                Role.ANALYST: 2,
                Role.ADMIN: 3,
                Role.SUPER_ADMIN: 4,
            }
            
            user_level = role_hierarchy.get(user.role, 0)
            required_level = role_hierarchy.get(role, 5)
            
            if user_level < required_level:
                raise SecurityError(f"Role denied: {role.value} or higher required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def admin_required(func: Callable) -> Callable:
    """Decorator to require admin role or higher.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    return require_role(Role.ADMIN)(func)


def analyst_required(func: Callable) -> Callable:
    """Decorator to require analyst role or higher.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    return require_role(Role.ANALYST)(func)


class RateLimiter:
    """Simple rate limiting for authentication endpoints."""
    
    def __init__(self, max_attempts: int = 5, window_minutes: int = 15):
        """Initialize rate limiter.
        
        Args:
            max_attempts: Maximum attempts allowed
            window_minutes: Time window in minutes
        """
        self.max_attempts = max_attempts
        self.window_minutes = window_minutes
        self._attempts: Dict[str, list] = {}
    
    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited.
        
        Args:
            identifier: IP address or user identifier
            
        Returns:
            True if rate limited
        """
        import time
        current_time = time.time()
        window_start = current_time - (self.window_minutes * 60)
        
        # Clean old attempts
        if identifier in self._attempts:
            self._attempts[identifier] = [
                attempt_time for attempt_time in self._attempts[identifier]
                if attempt_time > window_start
            ]
        
        # Check current attempts
        attempts = self._attempts.get(identifier, [])
        return len(attempts) >= self.max_attempts
    
    def record_attempt(self, identifier: str) -> None:
        """Record a failed attempt.
        
        Args:
            identifier: IP address or user identifier
        """
        import time
        current_time = time.time()
        
        if identifier not in self._attempts:
            self._attempts[identifier] = []
        
        self._attempts[identifier].append(current_time)
    
    def clear_attempts(self, identifier: str) -> None:
        """Clear attempts for identifier.
        
        Args:
            identifier: IP address or user identifier
        """
        self._attempts.pop(identifier, None)


# Global rate limiter instance
rate_limiter = RateLimiter()