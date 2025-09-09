"""
Validation utilities for file uploads and data processing
"""
import re
from pathlib import Path
from typing import Dict, Any, List
from fastapi import UploadFile
import magic

from backend.core.config import settings, file_validation
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

def validate_file_upload(file: UploadFile) -> Dict[str, Any]:
    """
    Comprehensive validation for uploaded files
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Dict containing validation results
    """
    try:
        # Get file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        # Basic validation
        validation_errors = []
        
        # Validate filename
        filename_validation = validate_filename(file.filename)
        if not filename_validation["valid"]:
            validation_errors.extend(filename_validation["errors"])
        
        # Validate file size
        size_validation = validate_file_size(file_size)
        if not size_validation["valid"]:
            validation_errors.extend(size_validation["errors"])
        
        # Validate file extension
        extension_validation = validate_file_extension(file.filename)
        if not extension_validation["valid"]:
            validation_errors.extend(extension_validation["errors"])
        
        # Validate MIME type if possible
        try:
            mime_validation = validate_mime_type(file)
            if not mime_validation["valid"]:
                validation_errors.extend(mime_validation["errors"])
        except Exception as e:
            logger.warning(f"MIME type validation failed: {e}")
            # Don't fail validation if MIME type check fails
        
        # Security validation
        security_validation = validate_file_security(file.filename)
        if not security_validation["valid"]:
            validation_errors.extend(security_validation["errors"])
        
        is_valid = len(validation_errors) == 0
        
        return {
            "valid": is_valid,
            "errors": validation_errors,
            "warnings": [],
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "extension": Path(file.filename).suffix.lower(),
                "content_type": file.content_type
            }
        }
        
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
            "file_info": {}
        }

def validate_filename(filename: str) -> Dict[str, Any]:
    """Validate filename for security and format"""
    errors = []
    
    if not filename or not filename.strip():
        errors.append("Filename cannot be empty")
        return {"valid": False, "errors": errors}
    
    # Check filename length
    if len(filename) > 255:
        errors.append("Filename too long (max 255 characters)")
    
    # Check for dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
    if any(char in filename for char in dangerous_chars):
        errors.append("Filename contains invalid characters")
    
    # Check for path traversal attempts
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        errors.append("Filename contains path traversal patterns")
    
    # Check for reserved names (Windows)
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                     'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                     'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
    
    name_without_ext = Path(filename).stem.upper()
    if name_without_ext in reserved_names:
        errors.append("Filename uses a reserved system name")
    
    # Check for reasonable filename pattern
    if not re.match(r'^[a-zA-Z0-9._\-\s()[\]]+$', filename):
        errors.append("Filename contains unusual characters")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_file_size(file_size: int) -> Dict[str, Any]:
    """Validate file size against limits"""
    errors = []
    
    if file_size <= 0:
        errors.append("File appears to be empty")
    
    if file_size > file_validation.MAX_FILE_SIZE:
        max_mb = file_validation.MAX_FILE_SIZE / (1024 * 1024)
        current_mb = file_size / (1024 * 1024)
        errors.append(f"File too large: {current_mb:.2f}MB (max: {max_mb:.2f}MB)")
    
    # Warn if file is very small
    warnings = []
    if file_size < 1024:  # Less than 1KB
        warnings.append("File is very small, may not contain useful data")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def validate_file_extension(filename: str) -> Dict[str, Any]:
    """Validate file extension against allowed types"""
    errors = []
    
    file_ext = Path(filename).suffix.lower()
    
    if not file_ext:
        errors.append("File has no extension")
        return {"valid": False, "errors": errors}
    
    if file_ext not in file_validation.ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(file_validation.ALLOWED_EXTENSIONS)
        errors.append(f"Unsupported file type: {file_ext}. Allowed: {allowed_str}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_mime_type(file: UploadFile) -> Dict[str, Any]:
    """Validate MIME type matches file extension"""
    errors = []
    warnings = []
    
    try:
        # Read first few bytes for MIME detection
        file.file.seek(0)
        file_header = file.file.read(1024)
        file.file.seek(0)
        
        # Use python-magic if available, fallback to content_type
        try:
            import magic
            detected_mime = magic.from_buffer(file_header, mime=True)
        except (ImportError, Exception):
            detected_mime = file.content_type
        
        file_ext = Path(file.filename).suffix.lower()
        
        # Expected MIME types for supported extensions
        expected_mimes = {
            '.csv': ['text/csv', 'text/plain', 'application/csv'],
            '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
            '.xls': ['application/vnd.ms-excel'],
            '.pdf': ['application/pdf']
        }
        
        if file_ext in expected_mimes:
            expected = expected_mimes[file_ext]
            if detected_mime not in expected:
                warnings.append(f"MIME type mismatch: detected '{detected_mime}' but expected one of {expected}")
        
        # Check for potentially dangerous MIME types
        dangerous_mimes = [
            'application/x-executable',
            'application/x-msdownload',
            'application/x-msdos-program',
            'application/x-winexe',
            'application/javascript',
            'text/javascript'
        ]
        
        if detected_mime in dangerous_mimes:
            errors.append(f"Dangerous file type detected: {detected_mime}")
        
    except Exception as e:
        logger.warning(f"MIME type validation failed: {e}")
        warnings.append("Could not validate MIME type")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def validate_file_security(filename: str) -> Dict[str, Any]:
    """Additional security validation for filenames"""
    errors = []
    warnings = []
    
    # Check for executable extensions
    executable_extensions = [
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
        '.app', '.deb', '.pkg', '.dmg', '.run', '.bin', '.sh', '.ps1'
    ]
    
    file_ext = Path(filename).suffix.lower()
    if file_ext in executable_extensions:
        errors.append(f"Executable file type not allowed: {file_ext}")
    
    # Check for double extensions (potential disguised executables)
    if filename.count('.') > 1:
        parts = filename.split('.')
        if len(parts) > 2 and parts[-2].lower() in ['exe', 'bat', 'cmd', 'scr']:
            errors.append("File appears to have disguised executable extension")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.php\.',
        r'\.asp\.',
        r'\.jsp\.',
        r'\.cgi\.',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, filename.lower()):
            warnings.append("Filename contains suspicious patterns")
            break
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def validate_column_names(columns: List[str]) -> Dict[str, Any]:
    """Validate DataFrame column names"""
    errors = []
    warnings = []
    
    # Check for empty column names
    empty_cols = [i for i, col in enumerate(columns) if not col or not str(col).strip()]
    if empty_cols:
        errors.append(f"Empty column names found at positions: {empty_cols}")
    
    # Check for duplicate column names
    seen = set()
    duplicates = set()
    for col in columns:
        if col in seen:
            duplicates.add(col)
        seen.add(col)
    
    if duplicates:
        errors.append(f"Duplicate column names found: {list(duplicates)}")
    
    # Check for problematic characters in column names
    problematic_chars = ['<', '>', ':', '"', '|', '?', '*', '\n', '\r', '\t']
    problematic_columns = []
    
    for col in columns:
        if any(char in str(col) for char in problematic_chars):
            problematic_columns.append(col)
    
    if problematic_columns:
        warnings.append(f"Column names contain problematic characters: {problematic_columns}")
    
    # Check for very long column names
    long_columns = [col for col in columns if len(str(col)) > 100]
    if long_columns:
        warnings.append(f"Very long column names found: {[str(col)[:50] + '...' for col in long_columns]}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def validate_data_quality(df_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data quality metrics"""
    errors = []
    warnings = []
    
    # Check for reasonable data dimensions
    if df_info.get("rows", 0) == 0:
        errors.append("Dataset contains no data rows")
    elif df_info.get("rows", 0) < 10:
        warnings.append("Dataset has very few rows, may not be suitable for machine learning")
    
    if df_info.get("columns", 0) == 0:
        errors.append("Dataset contains no columns")
    elif df_info.get("columns", 0) == 1:
        warnings.append("Dataset has only one column, may not be suitable for prediction tasks")
    
    # Check missing values
    missing_values = df_info.get("missing_values", {})
    if missing_values:
        total_cells = df_info.get("rows", 0) * df_info.get("columns", 0)
        total_missing = sum(missing_values.values())
        missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_percentage > 50:
            errors.append(f"Dataset has {missing_percentage:.1f}% missing values")
        elif missing_percentage > 20:
            warnings.append(f"Dataset has {missing_percentage:.1f}% missing values")
    
    # Check for data type diversity
    column_types = df_info.get("column_types", {})
    if column_types:
        numeric_cols = len([t for t in column_types.values() if 'int' in str(t) or 'float' in str(t)])
        if numeric_cols == 0:
            warnings.append("Dataset contains no numeric columns")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

# Export main validation functions
__all__ = [
    "validate_file_upload",
    "validate_filename",
    "validate_file_size",
    "validate_file_extension",
    "validate_mime_type",
    "validate_file_security",
    "validate_column_names",
    "validate_data_quality"
]