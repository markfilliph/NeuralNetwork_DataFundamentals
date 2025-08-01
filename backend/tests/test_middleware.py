"""Tests for authentication middleware and decorators."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.api.middleware import (
    AuthMiddleware,
    require_auth,
    require_permission,
    require_role,
    admin_required,
    analyst_required,
    RateLimiter
)
from backend.services.auth_service import AuthenticationService, AuthenticationError
from backend.services.rbac_service import rbac_service, Permission, Role
from backend.core.exceptions import SecurityError


class TestAuthMiddleware:
    """Test suite for AuthMiddleware."""
    
    def setup_method(self):
        """Set up test environment."""
        self.middleware = AuthMiddleware()
        self.auth_service = AuthenticationService()
        
        # Create test user
        self.test_user = rbac_service.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=Role.ANALYST
        )
        
        # Create test token
        self.test_token = self.auth_service.create_access_token({
            "user_id": "test_user",
            "username": "testuser"
        })
    
    def test_is_public_endpoint(self):
        """Test public endpoint detection."""
        # Exact matches
        assert self.middleware._is_public_endpoint("/health") is True
        assert self.middleware._is_public_endpoint("/login") is True
        assert self.middleware._is_public_endpoint("/docs") is True
        
        # Pattern matches
        assert self.middleware._is_public_endpoint("/static/css/style.css") is True
        assert self.middleware._is_public_endpoint("/docs/swagger") is True
        
        # Private endpoints
        assert self.middleware._is_public_endpoint("/api/data") is False
        assert self.middleware._is_public_endpoint("/api/models") is False
    
    def test_extract_token_bearer(self):
        """Test token extraction from Bearer header."""
        headers = {"Authorization": "Bearer test_token_123"}
        
        token = self.middleware._extract_token(headers)
        assert token == "test_token_123"
    
    def test_extract_token_custom_header(self):
        """Test token extraction from custom header."""
        headers = {"X-API-Token": "custom_token_456"}
        
        token = self.middleware._extract_token(headers)
        assert token == "custom_token_456"
    
    def test_extract_token_no_token(self):
        """Test token extraction when no token present."""
        headers = {}
        
        token = self.middleware._extract_token(headers)
        assert token is None
    
    def test_process_request_public_endpoint(self):
        """Test processing public endpoint request."""
        request_data = {
            "path": "/health",
            "headers": {}
        }
        
        result = self.middleware.process_request(request_data)
        
        # Should pass through unchanged
        assert result == request_data
        assert "current_user" not in result
    
    def test_process_request_valid_token(self):
        """Test processing request with valid token."""
        request_data = {
            "path": "/api/data",
            "headers": {"Authorization": f"Bearer {self.test_token}"}
        }
        
        result = self.middleware.process_request(request_data)
        
        assert "current_user" in result
        assert "current_user_obj" in result
        assert result["current_user"]["user_id"] == "test_user"
        assert result["current_user_obj"].username == "testuser"
    
    def test_process_request_no_token(self):
        """Test processing request without token."""
        request_data = {
            "path": "/api/data",
            "headers": {}
        }
        
        try:
            self.middleware.process_request(request_data)
            assert False, "Should have raised AuthenticationError"
        except AuthenticationError as e:
            assert "token required" in str(e).lower()
    
    def test_process_request_invalid_token(self):
        """Test processing request with invalid token."""
        request_data = {
            "path": "/api/data",
            "headers": {"Authorization": "Bearer invalid_token"}
        }
        
        try:
            self.middleware.process_request(request_data)
            assert False, "Should have raised AuthenticationError"
        except AuthenticationError as e:
            assert "Authentication failed" in str(e)


class TestDecorators:
    """Test suite for authentication decorators."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auth_service = AuthenticationService()
        
        # Create test users with different roles
        self.viewer = rbac_service.create_user(
            user_id="viewer_user",
            username="viewer",
            email="viewer@example.com",
            role=Role.VIEWER
        )
        
        self.analyst = rbac_service.create_user(
            user_id="analyst_user", 
            username="analyst",
            email="analyst@example.com",
            role=Role.ANALYST
        )
        
        self.admin = rbac_service.create_user(
            user_id="admin_user",
            username="admin",
            email="admin@example.com",
            role=Role.ADMIN
        )
        
        # Create tokens
        self.viewer_token = self.auth_service.create_access_token({
            "user_id": "viewer_user",
            "username": "viewer"
        })
        
        self.analyst_token = self.auth_service.create_access_token({
            "user_id": "analyst_user",
            "username": "analyst"
        })
        
        self.admin_token = self.auth_service.create_access_token({
            "user_id": "admin_user",
            "username": "admin"
        })
    
    def test_require_auth_decorator(self):
        """Test require_auth decorator."""
        @require_auth
        def protected_function(message, **kwargs):
            user = kwargs.get('current_user_obj')
            return f"{message} - User: {user.username}"
        
        # Should work with valid token
        result = protected_function("Hello", token=self.analyst_token)
        assert "Hello - User: analyst" == result
        
        # Should fail without token
        try:
            protected_function("Hello")
            assert False, "Should have raised AuthenticationError"
        except AuthenticationError:
            pass
        
        # Should fail with invalid token
        try:
            protected_function("Hello", token="invalid_token")
            assert False, "Should have raised AuthenticationError"
        except AuthenticationError:
            pass
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator."""
        @require_permission(Permission.TRAIN_MODELS)
        def train_model(**kwargs):
            return "Model trained successfully"
        
        # Analyst should have permission
        result = train_model(token=self.analyst_token)
        assert result == "Model trained successfully"
        
        # Viewer should not have permission
        try:
            train_model(token=self.viewer_token)
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Permission denied" in str(e)
    
    def test_require_role_decorator(self):
        """Test require_role decorator."""
        @require_role(Role.ADMIN)
        def admin_function(**kwargs):
            return "Admin operation completed"
        
        # Admin should have access
        result = admin_function(token=self.admin_token)
        assert result == "Admin operation completed"
        
        # Analyst should not have access
        try:
            admin_function(token=self.analyst_token)
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Role denied" in str(e)
        
        # Viewer should not have access
        try:
            admin_function(token=self.viewer_token)
            assert False, "Should have raised SecurityError"
        except SecurityError:
            pass
    
    def test_admin_required_decorator(self):
        """Test admin_required decorator."""
        @admin_required
        def admin_only_function(**kwargs):
            return "Admin only operation"
        
        # Admin should have access
        result = admin_only_function(token=self.admin_token)
        assert result == "Admin only operation"
        
        # Non-admin should not have access
        try:
            admin_only_function(token=self.analyst_token)
            assert False, "Should have raised SecurityError"
        except SecurityError:
            pass
    
    def test_analyst_required_decorator(self):
        """Test analyst_required decorator."""
        @analyst_required
        def analyst_function(**kwargs):
            return "Analyst operation"
        
        # Admin should have access (higher role)
        result = analyst_function(token=self.admin_token)
        assert result == "Analyst operation"
        
        # Analyst should have access
        result = analyst_function(token=self.analyst_token)
        assert result == "Analyst operation"
        
        # Viewer should not have access
        try:
            analyst_function(token=self.viewer_token)
            assert False, "Should have raised SecurityError"
        except SecurityError:
            pass


class TestRateLimiter:
    """Test suite for RateLimiter."""
    
    def setup_method(self):
        """Set up test environment."""
        self.rate_limiter = RateLimiter(max_attempts=3, window_minutes=15)
    
    def test_rate_limiter_normal_usage(self):
        """Test rate limiter under normal usage."""
        identifier = "test_user"
        
        # Should not be rate limited initially
        assert self.rate_limiter.is_rate_limited(identifier) is False
        
        # Record attempts
        self.rate_limiter.record_attempt(identifier)
        self.rate_limiter.record_attempt(identifier)
        
        # Should still not be rate limited
        assert self.rate_limiter.is_rate_limited(identifier) is False
        
        # Third attempt should trigger rate limiting
        self.rate_limiter.record_attempt(identifier)
        assert self.rate_limiter.is_rate_limited(identifier) is True
    
    def test_rate_limiter_clear_attempts(self):
        """Test clearing rate limiter attempts."""
        identifier = "test_user"
        
        # Record max attempts
        for _ in range(3):
            self.rate_limiter.record_attempt(identifier)
        
        assert self.rate_limiter.is_rate_limited(identifier) is True
        
        # Clear attempts
        self.rate_limiter.clear_attempts(identifier)
        assert self.rate_limiter.is_rate_limited(identifier) is False
    
    def test_rate_limiter_multiple_identifiers(self):
        """Test rate limiter with multiple identifiers."""
        user1 = "user1"
        user2 = "user2"
        
        # Max out attempts for user1
        for _ in range(3):
            self.rate_limiter.record_attempt(user1)
        
        # user1 should be rate limited, user2 should not
        assert self.rate_limiter.is_rate_limited(user1) is True
        assert self.rate_limiter.is_rate_limited(user2) is False
        
        # user2 can still make attempts
        self.rate_limiter.record_attempt(user2)
        assert self.rate_limiter.is_rate_limited(user2) is False


def run_middleware_tests():
    """Run all middleware tests."""
    print("Running Phase 2 Middleware Tests...")
    
    # Test AuthMiddleware
    print("✓ Testing AuthMiddleware...")
    test_middleware = TestAuthMiddleware()
    test_middleware.setup_method()
    test_middleware.test_is_public_endpoint()
    test_middleware.test_extract_token_bearer()
    test_middleware.test_extract_token_custom_header()
    test_middleware.test_extract_token_no_token()
    test_middleware.test_process_request_public_endpoint()
    test_middleware.test_process_request_valid_token()
    test_middleware.test_process_request_no_token()
    test_middleware.test_process_request_invalid_token()
    
    # Test Decorators
    print("✓ Testing decorators...")
    test_decorators = TestDecorators()
    test_decorators.setup_method()
    test_decorators.test_require_auth_decorator()
    test_decorators.test_require_permission_decorator()
    test_decorators.test_require_role_decorator()
    test_decorators.test_admin_required_decorator()
    test_decorators.test_analyst_required_decorator()
    
    # Test RateLimiter
    print("✓ Testing rate limiter...")
    test_rate_limiter = TestRateLimiter()
    test_rate_limiter.setup_method()
    test_rate_limiter.test_rate_limiter_normal_usage()
    
    test_rate_limiter.setup_method()
    test_rate_limiter.test_rate_limiter_clear_attempts()
    
    test_rate_limiter.setup_method()
    test_rate_limiter.test_rate_limiter_multiple_identifiers()
    
    print("✅ All middleware tests passed!")


if __name__ == "__main__":
    run_middleware_tests()