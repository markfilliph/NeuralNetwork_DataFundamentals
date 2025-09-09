"""Secure file handling utilities."""

import os
import hashlib
from pathlib import Path
from typing import Optional

from backend.core.config import settings
from backend.core.exceptions import (
    SecurityError,
    InvalidFileTypeError,
    FileSizeExceededError,
    MalwareDetectedError
)


class SecureFileHandler:
    """Handles secure file validation and processing."""
    
    MAX_FILE_SIZE = settings.MAX_FILE_SIZE
    ALLOWED_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)
    ALLOWED_MIMETYPES = set(settings.ALLOWED_MIMETYPES)
    
    @staticmethod
    def validate_file(file_path: Path, original_filename: Optional[str] = None) -> bool:
        """Validate file security and integrity.
        
        Args:
            file_path: Path to the file to validate
            original_filename: Original filename with extension (for temp files)
            
        Returns:
            True if file passes all validation checks
            
        Raises:
            FileSizeExceededError: If file size exceeds limit
            InvalidFileTypeError: If file type is not allowed
            MalwareDetectedError: If malware is detected
        """
        # Size validation
        if file_path.stat().st_size > SecureFileHandler.MAX_FILE_SIZE:
            raise FileSizeExceededError(
                f"File size {file_path.stat().st_size} exceeds limit "
                f"{SecureFileHandler.MAX_FILE_SIZE}"
            )
        
        # Extension validation
        if original_filename:
            # Use original filename for extension validation (for temp files)
            ext = Path(original_filename).suffix.lower()
        else:
            # Use file path extension
            ext = file_path.suffix.lower()
            
        if ext not in SecureFileHandler.ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(f"Invalid file extension: {ext}")
        
        # MIME type validation (simplified - would use python-magic in production)
        try:
            # Basic file content check instead of magic
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # Basic file signature detection
                if ext == '.xlsx' and not (b'PK' in header):
                    raise InvalidFileTypeError("File does not appear to be a valid Excel file")
                elif ext in ['.csv', '.tsv']:
                    # For CSV/TSV files, check if it's readable text
                    f.seek(0)
                    try:
                        f.read(100).decode('utf-8')
                    except UnicodeDecodeError:
                        raise InvalidFileTypeError(f"File does not appear to be valid {ext[1:].upper()}")
        except IOError as e:
            raise InvalidFileTypeError(f"Could not read file: {e}")
        
        # Basic malware scanning (simplified for demo)
        if not SecureFileHandler._scan_for_malware(file_path):
            raise MalwareDetectedError("Potential malware detected")
        
        return True
    
    @staticmethod
    def _scan_for_malware(file_path: Path) -> bool:
        """Basic malware scanning.
        
        Note: This is a simplified implementation. In production,
        integrate with ClamAV or similar antivirus solution.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            True if file appears safe, False if suspicious
        """
        # Check for suspicious file names
        suspicious_patterns = [
            'cmd.exe', 'powershell', '*.bat', '*.exe', '*.scr', '*.vbs'
        ]
        
        file_name = file_path.name.lower()
        for pattern in suspicious_patterns:
            if pattern in file_name:
                return False
        
        # Check file content for suspicious patterns (basic check)
        try:
            with open(file_path, 'rb') as f:
                # Read first 1KB to check for executable signatures
                header = f.read(1024)
                
                # Check for PE header (Windows executable)
                if b'MZ' in header[:2]:
                    return False
                    
                # Check for ELF header (Linux executable)
                if b'ELF' in header[:4]:
                    return False
        except Exception:
            # If we can't read the file, consider it suspicious
            return False
        
        return True
    
    @staticmethod
    def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity verification.
        
        Args:
            file_path: Path to file
            algorithm: Hashing algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem
        """
        # Remove directory traversal attempts
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit filename length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Ensure filename is not empty
        if not filename or filename.startswith('.'):
            filename = 'uploaded_file' + (Path(filename).suffix if Path(filename).suffix else '.txt')
        
        return filename