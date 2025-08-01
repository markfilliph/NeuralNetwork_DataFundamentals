"""Authentication, authorization, and scaling middleware."""

import os
import re
import time
import hashlib
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.services.rbac_service import rbac_service, Permission, Role
from backend.core.exceptions import SecurityError
from backend.core.scaling import load_balancer, session_affinity, health_checker
from backend.services.cache_service import cache_service
from backend.core.logging import audit_logger, EventType


class AuthMiddleware:
    """Authentication middleware for request processing."""
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/health",
        "/login",
        "/register",
        "/docs",
        "/openapi.json",
        "/cluster/health",
        "/cluster/stats"
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


class LoadBalancingMiddleware:
    """Middleware for load balancing and horizontal scaling."""
    
    def __init__(self):
        """Initialize load balancing middleware."""
        self.load_balancer = load_balancer
        self.session_affinity = session_affinity
        self.health_checker = health_checker
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request for load balancing.
        
        Args:
            request_data: Request data including headers, path, etc.
            
        Returns:
            Updated request data with routing information
        """
        headers = request_data.get('headers', {})
        session_id = self._extract_session_id(headers)
        
        # Get appropriate instance for request
        if session_id:
            instance = self.session_affinity.get_instance_for_session(session_id)
        else:
            instance = self.load_balancer.get_next_instance()
        
        if instance:
            request_data['target_instance'] = {
                'instance_id': instance.instance_id,
                'host': instance.host,
                'port': instance.port,
                'load_score': instance.load_score
            }
            
            # Log routing decision
            audit_logger.log_event(
                EventType.DATA_ACCESS,
                outcome="success",
                details={
                    "action": "request_routed",
                    "instance_id": instance.instance_id,
                    "session_id": session_id,
                    "path": request_data.get('path', 'unknown')
                }
            )
        else:
            # No healthy instances available
            audit_logger.log_event(
                EventType.DATA_ACCESS,
                outcome="error",
                details={
                    "action": "no_instances_available",
                    "path": request_data.get('path', 'unknown')
                },
                risk_level="high"
            )
            
            request_data['error'] = 'No healthy instances available'
        
        return request_data
    
    def _extract_session_id(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract session ID from request headers.
        
        Args:
            headers: Request headers
            
        Returns:
            Session ID or None
        """
        # Try different header names
        session_headers = ['X-Session-ID', 'Session-ID', 'Authorization']
        
        for header_name in session_headers:
            if header_name in headers:
                if header_name == 'Authorization':
                    # Extract from JWT token
                    token = headers[header_name].replace('Bearer ', '')
                    try:
                        auth_service = AuthenticationService()
                        payload = auth_service.verify_token(token)
                        return payload.get('session_id')
                    except:
                        continue
                else:
                    return headers[header_name]
        
        return None


class CacheMiddleware:
    """Middleware for response caching."""
    
    def __init__(self):
        """Initialize cache middleware."""
        self.cache_service = cache_service
        self.cacheable_methods = {'GET'}
        self.cacheable_paths = {
            '/api/users',
            '/api/datasets',
            '/api/models',
            '/cluster/stats',
            '/cluster/health'
        }
    
    def should_cache_request(self, request_data: Dict[str, Any]) -> bool:
        """Determine if request should be cached.
        
        Args:
            request_data: Request data
            
        Returns:
            True if request should be cached
        """
        method = request_data.get('method', 'GET')
        path = request_data.get('path', '')
        
        # Only cache GET requests
        if method not in self.cacheable_methods:
            return False
        
        # Check if path is cacheable
        for cacheable_path in self.cacheable_paths:
            if path.startswith(cacheable_path):
                return True
        
        return False
    
    def get_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request.
        
        Args:
            request_data: Request data
            
        Returns:
            Cache key
        """
        method = request_data.get('method', 'GET')
        path = request_data.get('path', '')
        query_params = request_data.get('query_params', {})
        user_id = request_data.get('user_id', 'anonymous')
        
        # Create cache key components
        key_components = {
            'method': method,
            'path': path,
            'params': sorted(query_params.items()) if query_params else [],
            'user': user_id
        }
        
        # Generate hash-based key
        key_str = str(key_components)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"api_response:{key_hash}"
    
    def get_cached_response(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request.
        
        Args:
            request_data: Request data
            
        Returns:
            Cached response or None
        """
        if not self.should_cache_request(request_data):
            return None
        
        cache_key = self.get_cache_key(request_data)
        return self.cache_service.get(cache_key)
    
    def cache_response(self, request_data: Dict[str, Any], response_data: Dict[str, Any], ttl: int = 300) -> bool:
        """Cache response for request.
        
        Args:
            request_data: Request data
            response_data: Response data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        if not self.should_cache_request(request_data):
            return False
        
        cache_key = self.get_cache_key(request_data)
        
        # Add cache metadata
        cached_response = {
            'data': response_data,
            'cached_at': datetime.utcnow().isoformat(),
            'cache_key': cache_key
        }
        
        return self.cache_service.set(cache_key, cached_response, ttl)


class HealthCheckMiddleware:
    """Middleware for health checks and monitoring."""
    
    def __init__(self):
        """Initialize health check middleware."""
        self.health_checker = health_checker
        self.instance_id = self._get_instance_id()
        self.start_time = datetime.utcnow()
        
    def _get_instance_id(self) -> str:
        """Get current instance ID."""
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}_{pid}"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current instance health status.
        
        Returns:
            Health status information
        """
        current_time = datetime.utcnow()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Get system metrics (simplified)
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        else:
            # Fallback metrics when psutil not available
            cpu_percent = 50.0  # Default moderate load
            memory_percent = 60.0
        
        # Calculate load score
        load_score = (cpu_percent + memory_percent) / 2
        
        # Determine status
        if load_score < 70:
            status = 'healthy'
        elif load_score < 90:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'instance_id': self.instance_id,
            'status': status,
            'uptime_seconds': uptime,
            'load_score': round(load_score, 2),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'timestamp': current_time.isoformat(),
            'cache_stats': cache_service.get_stats()
        }
    
    def update_instance_health(self) -> bool:
        """Update instance health in load balancer.
        
        Returns:
            True if updated successfully
        """
        health_status = self.get_health_status()
        
        return self.load_balancer.update_instance_health(
            self.instance_id,
            health_status['load_score'],
            health_status['status']
        )


# Global middleware instances
rate_limiter = RateLimiter()
load_balancing_middleware = LoadBalancingMiddleware()
cache_middleware = CacheMiddleware()
health_check_middleware = HealthCheckMiddleware()