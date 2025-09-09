"""Cache utilities and decorators for database operations."""

import hashlib
import json
import functools
from typing import Any, Callable, Optional, Union, Dict
from datetime import timedelta

from backend.services.cache_service_simple import cache_service


def cache_query(ttl: Union[int, timedelta] = 300, key_prefix: str = "query"):
    """Decorator to cache database query results.
    
    Args:
        ttl: Time to live in seconds or timedelta
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key_data = {
                'function': func.__name__,
                'args': str(args[1:]),  # Skip self
                'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
            }
            
            cache_key_str = json.dumps(cache_key_data, sort_keys=True)
            cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
            cache_key = f"{key_prefix}:{cache_key_hash}"
            
            # Try to get from cache first
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cache_user_data(ttl: Union[int, timedelta] = 1800):
    """Decorator to cache user-specific data.
    
    Args:
        ttl: Time to live in seconds or timedelta
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, user_id: str, *args, **kwargs):
            cache_key = f"user_data:{func.__name__}:{user_id}"
            
            # Try cache first
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(self, user_id, *args, **kwargs)
            if result is not None:
                cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def invalidate_user_cache(user_id: str) -> bool:
    """Invalidate all cache entries for a specific user.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if cache invalidated successfully
    """
    return cache_service.invalidate_user_cache(user_id)


def warm_user_cache(user_id: str, user_data: Dict[str, Any]) -> bool:
    """Warm cache with user data for better performance.
    
    Args:
        user_id: User identifier
        user_data: User data to cache
        
    Returns:
        True if cache warmed successfully
    """
    return cache_service.cache_user_data(user_id, user_data)


class CachedRepositoryMixin:
    """Mixin to add caching capabilities to repository classes."""
    
    def cache_result(self, key: str, result: Any, ttl: int = 600) -> bool:
        """Cache a result with the given key.
        
        Args:
            key: Cache key
            result: Result to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        return cache_service.set(key, result, ttl)
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        return cache_service.get(key)
    
    def invalidate_cache(self, pattern: str) -> bool:
        """Invalidate cache entries matching pattern.
        
        Args:
            pattern: Cache key pattern
            
        Returns:
            True if invalidated successfully
        """
        # For simple implementation, we'll handle specific patterns
        if pattern.startswith("user:"):
            user_id = pattern.split(":")[1]
            return cache_service.invalidate_user_cache(user_id)
        
        return cache_service.delete(pattern)
    
    def get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Key components
            
        Returns:
            Generated cache key
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)


def cache_api_response(ttl: Union[int, timedelta] = 600):
    """Decorator to cache API responses.
    
    Args:
        ttl: Time to live in seconds or timedelta
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from endpoint and parameters
            endpoint = func.__name__
            params_str = json.dumps({
                'args': str(args),
                'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
            }, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Check cache first
            cached_response = cache_service.get_cached_api_response(endpoint, params_hash)
            if cached_response is not None:
                return cached_response
            
            # Execute function and cache response
            response = func(*args, **kwargs)
            cache_service.cache_api_response(endpoint, params_hash, response, ttl)
            
            return response
        return wrapper
    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache statistics and health information
    """
    return cache_service.get_cache_health()


def clear_all_cache() -> bool:
    """Clear all cache entries.
    
    Returns:
        True if cleared successfully
    """
    return cache_service.clear()