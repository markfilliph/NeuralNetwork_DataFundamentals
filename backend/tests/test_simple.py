"""Simple tests that don't require external dependencies."""

import sys
import os
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.exceptions import SecurityError, FileValidationError, InvalidFileTypeError
from backend.core.config import Settings
from backend.utils.validators import BasicValidator


class TestExceptions:
    """Test custom exceptions."""
    
    def test_security_error(self):
        """Test SecurityError can be raised."""
        try:
            raise SecurityError("Test security error")
        except SecurityError as e:
            assert str(e) == "Test security error"
    
    def test_file_validation_error_inheritance(self):
        """Test FileValidationError inherits from SecurityError."""
        try:
            raise FileValidationError("Test file validation error")
        except SecurityError:
            pass  # Should be caught as SecurityError
        except Exception:
            raise AssertionError("FileValidationError should inherit from SecurityError")


class TestConfig:
    """Test configuration settings."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.APP_NAME == "Data Analysis Platform"
        assert settings.VERSION == "1.0.0"
        assert settings.MAX_FILE_SIZE == 100 * 1024 * 1024
        assert '.xlsx' in settings.ALLOWED_EXTENSIONS
        assert '.xls' in settings.ALLOWED_EXTENSIONS
        assert '.csv' in settings.ALLOWED_EXTENSIONS


class TestBasicValidator:
    """Test basic validation functions."""
    
    def test_is_safe_string_clean(self):
        """Test validation of clean string."""
        assert BasicValidator.is_safe_string("This is a clean string") is True
    
    def test_is_safe_string_formula(self):
        """Test detection of formula strings."""
        assert BasicValidator.is_safe_string("=SUM(A1:A10)") is False
        assert BasicValidator.is_safe_string("+B1*2") is False
        assert BasicValidator.is_safe_string("-C1") is False
        assert BasicValidator.is_safe_string("@D1") is False
    
    def test_is_safe_string_script(self):
        """Test detection of script tags."""
        assert BasicValidator.is_safe_string('<script>alert("xss")</script>') is False
        assert BasicValidator.is_safe_string('<SCRIPT>malicious</SCRIPT>') is False
    
    def test_is_safe_string_sql_injection(self):
        """Test detection of SQL injection."""
        assert BasicValidator.is_safe_string("'; DROP TABLE users; --") is False
        assert BasicValidator.is_safe_string("SELECT * FROM passwords") is False
        assert BasicValidator.is_safe_string("UPDATE users SET admin=1") is False
    
    def test_sanitize_string_formulas(self):
        """Test sanitization of formulas."""
        result = BasicValidator.sanitize_string("=SUM(A1:A10)")
        assert not result.startswith("=")
    
    def test_sanitize_string_scripts(self):
        """Test sanitization of scripts."""
        result = BasicValidator.sanitize_string('<script>alert("xss")</script>')
        assert '<script>' not in result.lower()
    
    def test_validate_column_name_valid(self):
        """Test validation of valid column names."""
        assert BasicValidator.validate_column_name("valid_column") is True
        assert BasicValidator.validate_column_name("_private_col") is True
        assert BasicValidator.validate_column_name("col123") is True
    
    def test_validate_column_name_invalid(self):
        """Test validation of invalid column names."""
        assert BasicValidator.validate_column_name("123invalid") is False
        assert BasicValidator.validate_column_name("col@name") is False
        assert BasicValidator.validate_column_name("col-with-dash") is False
    
    def test_sanitize_column_name(self):
        """Test column name sanitization."""
        assert BasicValidator.sanitize_column_name("col@name") == "col_name"
        assert BasicValidator.sanitize_column_name("123col") == "col_123col"
        assert BasicValidator.sanitize_column_name("") == "column_0"
        assert BasicValidator.sanitize_column_name("_") == "column_0"


def run_tests():
    """Run all tests manually."""
    print("Running Phase 1 Security Tests...")
    
    # Test exceptions
    print("✓ Testing exceptions...")
    test_exceptions = TestExceptions()
    test_exceptions.test_security_error()
    test_exceptions.test_file_validation_error_inheritance()
    
    # Test config
    print("✓ Testing configuration...")
    test_config = TestConfig()
    test_config.test_settings_defaults()
    
    # Test validators
    print("✓ Testing validators...")
    test_validator = TestBasicValidator()
    test_validator.test_is_safe_string_clean()
    test_validator.test_is_safe_string_formula()
    test_validator.test_is_safe_string_script()
    test_validator.test_is_safe_string_sql_injection()
    test_validator.test_sanitize_string_formulas()
    test_validator.test_sanitize_string_scripts()
    test_validator.test_validate_column_name_valid()
    test_validator.test_validate_column_name_invalid()
    test_validator.test_sanitize_column_name()
    
    print("✅ All Phase 1 tests passed!")


if __name__ == "__main__":
    run_tests()