"""Caching service with Redis integration and fallback to in-memory cache."""

import json
import time
import hashlib
from typing import Any, Optional, Dict, Union
from datetime import timedelta

import redis
from backend.core.config import settings
from backend.core.logging import audit_logger, EventType


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass


class CacheService:
    """Multi-level caching service with Redis backend and in-memory fallback."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize cache service.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self.redis_url = redis_url or getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'redis_hits': 0,
            'memory_hits': 0,
            'errors': 0
        }
        
        self._connect_redis()
    
    def _connect_redis(self) -> None:
        """Connect to Redis server with fallback handling."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="success",
                details={"cache": "Redis connected successfully"}
            )
            
        except Exception as e:
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="warning",
                details={"cache": f"Redis connection failed, using in-memory fallback: {e}"},
                risk_level="low"
            )
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis first, then memory fallback).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value is not None:
                        self.cache_stats['hits'] += 1
                        self.cache_stats['redis_hits'] += 1
                        return json.loads(value)
                except Exception:
                    # Redis failed, continue to memory cache
                    pass
            
            # Fallback to memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > time.time():
                    self.cache_stats['hits'] += 1
                    self.cache_stats['memory_hits'] += 1
                    return entry['value']
                else:
                    # Expired entry
                    del self.memory_cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            audit_logger.log_event(
                EventType.SYSTEM_START,  # Using as general system event
                outcome="error",
                details={"cache_error": f"Get operation failed: {e}"}
            )
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = None) -> bool:
        """Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds or timedelta
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            # Convert ttl to seconds
            if isinstance(ttl, timedelta):
                ttl_seconds = int(ttl.total_seconds())
            else:
                ttl_seconds = ttl or 3600  # Default 1 hour
            
            serialized_value = json.dumps(value)
            
            # Try Redis first
            if self.redis_client:
                try:
                    self.redis_client.setex(key, ttl_seconds, serialized_value)
                    return True
                except Exception:
                    # Redis failed, continue to memory cache
                    pass
            
            # Fallback to memory cache
            self.memory_cache[key] = {
                'value': value,
                'expires': time.time() + ttl_seconds
            }
            
            # Clean up expired entries periodically
            if len(self.memory_cache) > 1000:
                self._cleanup_memory_cache()
            
            return True
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            audit_logger.log_event(
                EventType.SYSTEM_START,  # Using as general system event
                outcome="error",
                details={"cache_error": f"Set operation failed: {e}"}
            )
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            success = False
            
            # Delete from Redis
            if self.redis_client:
                try:
                    deleted = self.redis_client.delete(key)
                    success = success or (deleted > 0)
                except Exception:
                    pass
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                success = True
            
            return success
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            audit_logger.log_event(
                EventType.SYSTEM_START,  # Using as general system event
                outcome="error",
                details={"cache_error": f"Delete operation failed: {e}"}
            )
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if cleared successfully
        """
        try:
            success = False
            
            # Clear Redis
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                    success = True
                except Exception:
                    pass
            
            # Clear memory cache
            self.memory_cache.clear()
            success = True
            
            return success
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_cache_size': len(self.memory_cache),
            'redis_connected': self.redis_client is not None
        }
        
        return stats
    
    def _cleanup_memory_cache(self) -> None:
        """Remove expired entries from memory cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry['expires'] <= current_time
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
    
    def cache_user_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache user session data.
        
        Args:
            user_id: User identifier
            session_data: Session data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        return self.set(f"session:{user_id}", session_data, ttl)
    
    def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session data or None if not found
        """
        return self.get(f"session:{user_id}")
    
    def invalidate_user_session(self, user_id: str) -> bool:
        """Invalidate user session cache.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if invalidated successfully
        """
        return self.delete(f"session:{user_id}")
    
    def cache_user_permissions(self, user_id: str, permissions: list, ttl: int = 1800) -> bool:
        """Cache user permissions.
        
        Args:
            user_id: User identifier
            permissions: List of permission strings
            ttl: Time to live in seconds (default 30 minutes)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"permissions:{user_id}", permissions, ttl)
    
    def get_user_permissions(self, user_id: str) -> Optional[list]:
        """Get cached user permissions.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of permissions or None if not found
        """
        return self.get(f"permissions:{user_id}")
    
    def cache_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any], ttl: int = 7200) -> bool:
        """Cache dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
            metadata: Dataset metadata
            ttl: Time to live in seconds (default 2 hours)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"dataset:{dataset_id}", metadata, ttl)
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset metadata or None if not found
        """
        return self.get(f"dataset:{dataset_id}")
    
    def cache_query_result(self, query_hash: str, result: Any, ttl: int = 300) -> bool:
        """Cache database query result.
        
        Args:
            query_hash: Hash of the query and parameters
            result: Query result to cache
            ttl: Time to live in seconds (default 5 minutes)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"query:{query_hash}", result, ttl)
    
    def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result.
        
        Args:
            query_hash: Hash of the query and parameters
            
        Returns:
            Cached result or None if not found
        """
        return self.get(f"query:{query_hash}")
    
    def cache_user_data(self, user_id: str, data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache comprehensive user data (profile, permissions, etc).
        
        Args:
            user_id: User identifier
            data: User data to cache
            ttl: Time to live in seconds (default 30 minutes)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"user_data:{user_id}", data, ttl)
    
    def get_cached_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Cached user data or None if not found
        """
        return self.get(f"user_data:{user_id}")
    
    def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate all cache entries for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if cache invalidated successfully
        """
        success = True
        cache_keys = [
            f"session:{user_id}",
            f"permissions:{user_id}",
            f"user_data:{user_id}"
        ]
        
        for key in cache_keys:
            if not self.delete(key):
                success = False
        
        return success
    
    def cache_api_response(self, endpoint: str, params_hash: str, response: Any, ttl: int = 600) -> bool:
        """Cache API response.
        
        Args:
            endpoint: API endpoint
            params_hash: Hash of request parameters
            response: Response data to cache
            ttl: Time to live in seconds (default 10 minutes)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"api:{endpoint}:{params_hash}", response, ttl)
    
    def get_cached_api_response(self, endpoint: str, params_hash: str) -> Optional[Any]:
        """Get cached API response.
        
        Args:
            endpoint: API endpoint
            params_hash: Hash of request parameters
            
        Returns:
            Cached response or None if not found
        """
        return self.get(f"api:{endpoint}:{params_hash}")
    
    def cache_file_metadata(self, file_hash: str, metadata: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache file metadata and processing results.
        
        Args:
            file_hash: Hash of the file
            metadata: File metadata and processing results
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            True if cached successfully
        """
        return self.set(f"file:{file_hash}", metadata, ttl)
    
    def get_cached_file_metadata(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached file metadata.
        
        Args:
            file_hash: Hash of the file
            
        Returns:
            Cached metadata or None if not found
        """
        return self.get(f"file:{file_hash}")
    
    def warm_cache(self, user_id: str, session_data: Dict[str, Any], permissions: list) -> bool:
        """Warm cache with frequently accessed user data.
        
        Args:
            user_id: User identifier
            session_data: Session data
            permissions: User permissions
            
        Returns:
            True if cache warmed successfully
        """
        success = True
        
        # Cache session data
        if not self.cache_user_session(user_id, session_data):
            success = False
        
        # Cache permissions
        if not self.cache_user_permissions(user_id, permissions):
            success = False
        
        return success
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get comprehensive cache health metrics.
        
        Returns:
            Dictionary with cache health information
        """
        stats = self.get_stats()
        
        # Calculate health score
        health_score = 100
        
        if stats['hit_rate_percent'] < 50:
            health_score -= 30
        elif stats['hit_rate_percent'] < 70:
            health_score -= 15
        
        if stats['errors'] > stats['total_requests'] * 0.05:  # > 5% error rate
            health_score -= 25
        
        if stats['memory_cache_size'] > 5000:  # Memory cache too large
            health_score -= 10
        
        return {
            **stats,
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'unhealthy'
        }


# Global cache service instance
cache_service = CacheService()