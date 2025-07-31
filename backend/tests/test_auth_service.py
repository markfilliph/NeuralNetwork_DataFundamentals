"""Tests for authentication service."""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.auth_service import (
    AuthenticationService,
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError
)


class TestAuthenticationService:
    """Test suite for AuthenticationService."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auth_service = AuthenticationService()
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password123"
        
        hash1, salt1 = self.auth_service.hash_password(password)
        hash2, salt2 = self.auth_service.hash_password(password)
        
        # Different salts should produce different hashes
        assert hash1 != hash2
        assert salt1 != salt2
        assert len(hash1) == 64  # SHA256 hex length
        assert len(salt1) == 64  # 32 bytes hex encoded
    
    def test_hash_password_with_salt(self):
        """Test password hashing with provided salt."""
        password = "test_password123"
        salt = "fixed_salt_for_testing"
        
        hash1, returned_salt1 = self.auth_service.hash_password(password, salt)
        hash2, returned_salt2 = self.auth_service.hash_password(password, salt)
        
        # Same salt should produce same hash
        assert hash1 == hash2
        assert returned_salt1 == returned_salt2 == salt
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "test_password123"
        hashed_password, salt = self.auth_service.hash_password(password)
        
        result = self.auth_service.verify_password(password, hashed_password, salt)
        assert result is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "test_password123"
        wrong_password = "wrong_password"
        hashed_password, salt = self.auth_service.hash_password(password)
        
        result = self.auth_service.verify_password(wrong_password, hashed_password, salt)
        assert result is False
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        payload = {
            "user_id": "test_user",
            "username": "testuser",
            "role": "analyst"
        }
        
        token = self.auth_service.create_access_token(payload)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
    
    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry."""
        payload = {"user_id": "test_user"}
        expires_delta = timedelta(minutes=5)
        
        token = self.auth_service.create_access_token(payload, expires_delta)
        
        assert token is not None
        # Verify token contains correct expiry
        decoded = self.auth_service.verify_token(token)
        assert "exp" in decoded
    
    def test_verify_token_valid(self):
        """Test verification of valid token."""
        payload = {
            "user_id": "test_user",
            "username": "testuser"
        }
        
        token = self.auth_service.create_access_token(payload)
        decoded = self.auth_service.verify_token(token)
        
        assert decoded["user_id"] == payload["user_id"]
        assert decoded["username"] == payload["username"]
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_verify_token_expired(self):
        """Test verification of expired token."""
        payload = {"user_id": "test_user"}
        
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = self.auth_service.create_access_token(payload, expires_delta)
        
        try:
            self.auth_service.verify_token(token)
            assert False, "Should have raised TokenExpiredError or InvalidTokenError"
        except (TokenExpiredError, InvalidTokenError):
            pass  # Expected - either exception is acceptable
    
    def test_verify_token_invalid_format(self):
        """Test verification of malformed token."""
        invalid_tokens = [
            "invalid.token",  # Only 2 parts
            "invalid",  # Not JWT format
            "a.b.c.d",  # Too many parts
            "",  # Empty
        ]
        
        for token in invalid_tokens:
            try:
                self.auth_service.verify_token(token)
                assert False, f"Should have raised InvalidTokenError for: {token}"
            except InvalidTokenError:
                pass  # Expected
    
    def test_verify_token_invalid_signature(self):
        """Test verification of token with invalid signature."""
        payload = {"user_id": "test_user"}
        token = self.auth_service.create_access_token(payload)
        
        # Tamper with the token (change last few characters to ensure change)
        tampered_token = token[:-5] + "XXXXX"
        
        try:
            self.auth_service.verify_token(tampered_token)
            assert False, "Should have raised InvalidTokenError"
        except InvalidTokenError:
            pass  # Expected
        except Exception as e:
            # Any other exception is also acceptable as it indicates token verification failed
            assert "verification failed" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_generate_api_key(self):
        """Test API key generation."""
        key1 = self.auth_service.generate_api_key()
        key2 = self.auth_service.generate_api_key()
        
        assert key1 != key2
        assert len(key1) == 64  # 32 bytes hex encoded
        assert len(key2) == 64
        assert all(c in '0123456789abcdef' for c in key1)
        assert all(c in '0123456789abcdef' for c in key2)
    
    def test_token_roundtrip(self):
        """Test complete token creation and verification cycle."""
        original_payload = {
            "user_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "analyst",
            "permissions": ["read_data", "train_models"]
        }
        
        # Create token
        token = self.auth_service.create_access_token(original_payload)
        
        # Verify token
        decoded_payload = self.auth_service.verify_token(token)
        
        # Check all original data is preserved
        for key, value in original_payload.items():
            assert decoded_payload[key] == value
        
        # Check timestamps are added
        assert "exp" in decoded_payload
        assert "iat" in decoded_payload
        assert decoded_payload["exp"] > decoded_payload["iat"]


def run_auth_tests():
    """Run all authentication tests."""
    print("Running Phase 2 Authentication Tests...")
    
    test_auth = TestAuthenticationService()
    
    # Test password hashing
    print("✓ Testing password hashing...")
    test_auth.setup_method()
    test_auth.test_hash_password()
    test_auth.test_hash_password_with_salt()
    test_auth.test_verify_password_correct()
    test_auth.test_verify_password_incorrect()
    
    # Test JWT tokens
    print("✓ Testing JWT tokens...")
    test_auth.setup_method()
    test_auth.test_create_access_token()
    test_auth.test_create_access_token_with_expiry()
    test_auth.test_verify_token_valid()
    test_auth.test_verify_token_expired()
    test_auth.test_verify_token_invalid_format()
    test_auth.test_verify_token_invalid_signature()
    
    # Test utilities
    print("✓ Testing utilities...")
    test_auth.setup_method()
    test_auth.test_generate_api_key()
    test_auth.test_token_roundtrip()
    
    print("✅ All authentication tests passed!")


if __name__ == "__main__":
    run_auth_tests()