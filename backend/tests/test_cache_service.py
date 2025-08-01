"""Tests for caching service."""

import sys
import time
from pathlib import Path
from datetime import timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.cache_service import CacheService, CacheError


class TestCacheService:
    """Test suite for CacheService."""
    
    def setup_method(self):
        """Set up test environment."""
        # Use in-memory only cache for testing (no Redis dependency)
        self.cache_service = CacheService(redis_url="redis://invalid:9999")
        # Clear any existing cache
        self.cache_service.clear()
    
    def test_basic_get_set(self):
        """Test basic get/set operations."""
        key = "test_key"
        value = {"test": "data", "number": 42}
        
        # Initially should not exist
        assert self.cache_service.get(key) is None
        assert not self.cache_service.exists(key)
        
        # Set value
        success = self.cache_service.set(key, value)
        assert success is True
        
        # Get value
        cached_value = self.cache_service.get(key)
        assert cached_value == value
        assert self.cache_service.exists(key)
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        key = "ttl_test"
        value = "test_value"
        
        # Set with 1 second TTL
        success = self.cache_service.set(key, value, ttl=1)
        assert success is True
        
        # Should exist immediately
        assert self.cache_service.get(key) == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert self.cache_service.get(key) is None
    
    def test_timedelta_ttl(self):
        """Test TTL with timedelta."""
        key = "timedelta_test"
        value = "test_value"
        
        # Set with timedelta TTL
        success = self.cache_service.set(key, value, ttl=timedelta(seconds=2))
        assert success is True
        
        # Should exist
        assert self.cache_service.get(key) == value
        
        # Wait for expiration
        time.sleep(2.1)
        
        # Should be expired
        assert self.cache_service.get(key) is None
    
    def test_delete(self):
        """Test delete operation."""
        key = "delete_test"
        value = "test_value"
        
        # Set value
        self.cache_service.set(key, value)
        assert self.cache_service.get(key) == value
        
        # Delete value
        success = self.cache_service.delete(key)
        assert success is True
        
        # Should not exist
        assert self.cache_service.get(key) is None
        assert not self.cache_service.exists(key)
    
    def test_clear(self):
        """Test clear operation."""
        # Set multiple values
        for i in range(5):
            self.cache_service.set(f"key_{i}", f"value_{i}")
        
        # All should exist
        for i in range(5):
            assert self.cache_service.get(f"key_{i}") == f"value_{i}"
        
        # Clear cache
        success = self.cache_service.clear()
        assert success is True
        
        # All should be gone
        for i in range(5):
            assert self.cache_service.get(f"key_{i}") is None
    
    def test_complex_data_types(self):
        """Test caching complex data types."""
        test_cases = [
            ("string", "simple string"),
            ("number", 42),
            ("float", 3.14159),
            ("boolean", True),
            ("list", [1, 2, 3, "a", "b"]),
            ("dict", {"nested": {"data": [1, 2, 3]}, "key": "value"}),
            ("none", None),
            ("empty_list", []),
            ("empty_dict", {})
        ]
        
        for key, value in test_cases:
            success = self.cache_service.set(key, value)
            assert success is True
            
            cached_value = self.cache_service.get(key)
            assert cached_value == value
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Clear stats
        self.cache_service.cache_stats = {
            'hits': 0,
            'misses': 0,
            'redis_hits': 0,
            'memory_hits': 0,
            'errors': 0
        }
        
        key = "stats_test"
        value = "test_value"
        
        # Miss
        result = self.cache_service.get(key)
        assert result is None
        
        # Set
        self.cache_service.set(key, value)
        
        # Hit
        result = self.cache_service.get(key)
        assert result == value
        
        # Get stats
        stats = self.cache_service.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        assert stats['total_requests'] >= 2
        assert stats['hit_rate_percent'] > 0
        assert 'memory_cache_size' in stats
        assert 'redis_connected' in stats
    
    def test_user_session_caching(self):
        """Test user session caching helpers."""
        user_id = "test_user_123"
        session_data = {
            "username": "testuser",
            "role": "analyst",
            "login_time": "2024-01-01T12:00:00Z"
        }
        
        # Cache session
        success = self.cache_service.cache_user_session(user_id, session_data)
        assert success is True
        
        # Get session
        cached_session = self.cache_service.get_user_session(user_id)
        assert cached_session == session_data
        
        # Invalidate session
        success = self.cache_service.invalidate_user_session(user_id)
        assert success is True
        
        # Should be gone
        cached_session = self.cache_service.get_user_session(user_id)
        assert cached_session is None
    
    def test_user_permissions_caching(self):
        """Test user permissions caching helpers."""
        user_id = "test_user_456"
        permissions = ["read_data", "upload_data", "view_reports"]
        
        # Cache permissions
        success = self.cache_service.cache_user_permissions(user_id, permissions)
        assert success is True
        
        # Get permissions
        cached_permissions = self.cache_service.get_user_permissions(user_id)
        assert cached_permissions == permissions
    
    def test_dataset_metadata_caching(self):
        """Test dataset metadata caching helpers."""
        dataset_id = "dataset_789"
        metadata = {
            "name": "Test Dataset",
            "size": 1024,
            "format": "CSV",
            "columns": ["id", "name", "value"]
        }
        
        # Cache metadata
        success = self.cache_service.cache_dataset_metadata(dataset_id, metadata)
        assert success is True
        
        # Get metadata
        cached_metadata = self.cache_service.get_dataset_metadata(dataset_id)
        assert cached_metadata == metadata
    
    def test_memory_cache_cleanup(self):
        """Test memory cache cleanup for expired entries."""
        # Add many entries to trigger cleanup
        for i in range(10):
            self.cache_service.set(f"temp_{i}", f"value_{i}", ttl=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Add one more to trigger cleanup
        self.cache_service.set("trigger_cleanup", "value", ttl=3600)
        
        # Force cleanup by calling internal method
        self.cache_service._cleanup_memory_cache()
        
        # Check that expired entries are cleaned up
        for i in range(10):
            assert self.cache_service.get(f"temp_{i}") is None
        
        # New entry should still exist
        assert self.cache_service.get("trigger_cleanup") == "value"


def run_cache_tests():
    """Run all cache service tests."""
    print("Running Cache Service Tests...")
    
    test_cache = TestCacheService()
    
    test_cache.setup_method()
    test_cache.test_basic_get_set()
    print("✓ Basic get/set operations")
    
    test_cache.setup_method()
    test_cache.test_ttl_expiration()
    print("✓ TTL expiration")
    
    test_cache.setup_method()
    test_cache.test_timedelta_ttl()
    print("✓ Timedelta TTL")
    
    test_cache.setup_method()
    test_cache.test_delete()
    print("✓ Delete operations")
    
    test_cache.setup_method()
    test_cache.test_clear()
    print("✓ Clear operations")
    
    test_cache.setup_method()
    test_cache.test_complex_data_types()
    print("✓ Complex data types")
    
    test_cache.setup_method()
    test_cache.test_cache_stats()
    print("✓ Cache statistics")
    
    test_cache.setup_method()
    test_cache.test_user_session_caching()
    print("✓ User session caching")
    
    test_cache.setup_method()
    test_cache.test_user_permissions_caching()
    print("✓ User permissions caching")
    
    test_cache.setup_method()
    test_cache.test_dataset_metadata_caching()
    print("✓ Dataset metadata caching")
    
    test_cache.setup_method()
    test_cache.test_memory_cache_cleanup()
    print("✓ Memory cache cleanup")
    
    print("✅ All cache service tests passed!")


if __name__ == "__main__":
    run_cache_tests()