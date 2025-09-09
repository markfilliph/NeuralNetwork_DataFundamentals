"""Simple in-memory caching service without Redis dependency."""

import json
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class SimpleCacheService:
    """Simple in-memory caching service."""
    
    def __init__(self):
        """Initialize simple cache service."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = 1000  # Maximum number of cache entries
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        
        # Check if expired
        if entry['expires_at'] and datetime.utcnow() > entry['expires_at']:
            del self._cache[key]
            return None
            
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        # Clean up if cache is too large
        if len(self._cache) >= self._max_size:
            self._cleanup_expired()
            
        # If still too large, remove oldest entries
        if len(self._cache) >= self._max_size:
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k]['created_at']
            )[:100]  # Remove 100 oldest entries
            for old_key in oldest_keys:
                del self._cache[old_key]
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
        
        self._cache[key] = {
            'value': value,
            'created_at': datetime.utcnow(),
            'expires_at': expires_at
        }
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key existed and was deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if successful
        """
        self._cache.clear()
        return True
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry['expires_at'] and now > entry['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        self._cleanup_expired()
        return {
            'total_entries': len(self._cache),
            'max_size': self._max_size,
            'backend': 'memory'
        }


# Create global cache service instance
cache_service = SimpleCacheService()