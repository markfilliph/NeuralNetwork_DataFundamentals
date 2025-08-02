"""Custom exceptions for the data analysis platform."""


class SecurityError(Exception):
    """Raised when a security validation fails."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class FileValidationError(SecurityError):
    """Raised when file validation fails."""
    pass


class DataSanitizationError(Exception):
    """Raised when data sanitization fails."""
    pass


class InvalidFileTypeError(FileValidationError):
    """Raised when an invalid file type is uploaded."""
    pass


class FileSizeExceededError(FileValidationError):
    """Raised when file size exceeds the allowed limit."""
    pass


class MalwareDetectedError(SecurityError):
    """Raised when malware is detected in uploaded file."""
    pass