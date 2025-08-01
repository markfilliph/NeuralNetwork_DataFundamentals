"""Tests for secure file handling utilities."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from backend.utils.file_handlers import SecureFileHandler
from backend.core.exceptions import (
    InvalidFileTypeError,
    FileSizeExceededError,
    MalwareDetectedError
)


class TestSecureFileHandler:
    """Test suite for SecureFileHandler class."""
    
    def test_validate_file_valid_excel(self, tmp_path):
        """Test validation of valid Excel file."""
        # Create a temporary file with valid extension
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"valid excel content")
        
        with patch('magic.from_file', return_value='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            with patch.object(SecureFileHandler, '_scan_for_malware', return_value=True):
                result = SecureFileHandler.validate_file(test_file)
                assert result is True
    
    def test_validate_file_size_exceeded(self, tmp_path):
        """Test file size validation."""
        test_file = tmp_path / "large_file.xlsx"
        
        # Mock file size to exceed limit
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = SecureFileHandler.MAX_FILE_SIZE + 1
            
            with pytest.raises(FileSizeExceededError):
                SecureFileHandler.validate_file(test_file)
    
    def test_validate_file_invalid_extension(self, tmp_path):
        """Test invalid file extension validation."""
        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"content")
        
        with pytest.raises(InvalidFileTypeError) as exc_info:
            SecureFileHandler.validate_file(test_file)
        
        assert "Invalid file extension" in str(exc_info.value)
    
    def test_validate_file_invalid_mime_type(self, tmp_path):
        """Test invalid MIME type validation."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"content")
        
        with patch('magic.from_file', return_value='application/x-executable'):
            with pytest.raises(InvalidFileTypeError) as exc_info:
                SecureFileHandler.validate_file(test_file)
            
            assert "Invalid MIME type" in str(exc_info.value)
    
    def test_validate_file_malware_detected(self, tmp_path):
        """Test malware detection."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"content")
        
        with patch('magic.from_file', return_value='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            with patch.object(SecureFileHandler, '_scan_for_malware', return_value=False):
                with pytest.raises(MalwareDetectedError):
                    SecureFileHandler.validate_file(test_file)
    
    def test_scan_for_malware_suspicious_filename(self, tmp_path):
        """Test malware scanning with suspicious filename."""
        test_file = tmp_path / "cmd.exe.xlsx"
        test_file.write_bytes(b"content")
        
        result = SecureFileHandler._scan_for_malware(test_file)
        assert result is False
    
    def test_scan_for_malware_pe_header(self, tmp_path):
        """Test malware scanning with PE header."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"MZ" + b"x" * 100)
        
        result = SecureFileHandler._scan_for_malware(test_file)
        assert result is False
    
    def test_scan_for_malware_elf_header(self, tmp_path):
        """Test malware scanning with ELF header."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"ELF" + b"x" * 100)
        
        result = SecureFileHandler._scan_for_malware(test_file)
        assert result is False
    
    def test_scan_for_malware_clean_file(self, tmp_path):
        """Test malware scanning with clean file."""
        test_file = tmp_path / "clean.xlsx"
        test_file.write_bytes(b"clean excel content")
        
        result = SecureFileHandler._scan_for_malware(test_file)
        assert result is True
    
    def test_calculate_file_hash(self, tmp_path):
        """Test file hash calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)
        
        hash_result = SecureFileHandler.calculate_file_hash(test_file)
        
        # Verify it's a valid SHA256 hash
        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)
    
    def test_sanitize_filename_directory_traversal(self):
        """Test filename sanitization against directory traversal."""
        dangerous_filename = "../../../etc/passwd"
        
        result = SecureFileHandler.sanitize_filename(dangerous_filename)
        
        assert result == "passwd"
        assert "../" not in result
    
    def test_sanitize_filename_dangerous_chars(self):
        """Test filename sanitization of dangerous characters."""
        dangerous_filename = 'file<>:"|?*\\.txt'
        
        result = SecureFileHandler.sanitize_filename(dangerous_filename)
        
        assert result == "file_________.txt"
    
    def test_sanitize_filename_long_name(self):
        """Test filename sanitization of overly long names."""
        long_filename = "a" * 300 + ".txt"
        
        result = SecureFileHandler.sanitize_filename(long_filename)
        
        assert len(result) <= 255
        assert result.endswith(".txt")
    
    def test_sanitize_filename_empty(self):
        """Test sanitization of empty filename."""
        result = SecureFileHandler.sanitize_filename("")
        
        assert result == "uploaded_file.txt"
    
    def test_sanitize_filename_hidden_file(self):
        """Test sanitization of hidden file."""
        result = SecureFileHandler.sanitize_filename(".hidden")
        
        assert result == "uploaded_file.hidden"