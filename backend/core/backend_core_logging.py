"""
Logging configuration for the application
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

from backend.core.config import settings

def setup_logging():
    """Setup application logging configuration"""
    
    # Remove default loguru handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler for general logs
    log_file = Path(settings.LOG_DIR) / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # Separate file for errors
    error_log_file = Path(settings.LOG_DIR) / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        error_log_file,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\nStacktrace: {exception}",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # Security logs
    security_log_file = Path(settings.LOG_DIR) / f"security_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        security_log_file,
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | SECURITY | {message}",
        filter=lambda record: "SECURITY" in record["message"],
        rotation="1 day",
        retention="90 days"
    )
    
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Replace standard logging handlers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set specific loggers
    for name in ["uvicorn", "uvicorn.access", "fastapi"]:
        logging.getLogger(name).handlers = [InterceptHandler()]
    
    logger.info("🔧 Logging system configured successfully")

def get_logger(name: str):
    """Get a logger instance for a specific module"""
    return logger.bind(name=name)

# Security logging functions
def log_security_event(event_type: str, details: str, user_ip: str = None, user_id: str = None):
    """Log security events"""
    message = f"SECURITY | {event_type} | {details}"
    if user_ip:
        message += f" | IP: {user_ip}"
    if user_id:
        message += f" | User: {user_id}"
    
    logger.warning(message)

def log_file_upload(filename: str, file_size: int, user_ip: str = None):
    """Log file upload events"""
    log_security_event(
        "FILE_UPLOAD",
        f"File: {filename}, Size: {file_size} bytes",
        user_ip=user_ip
    )

def log_prediction_request(file_id: str, model_type: str, user_ip: str = None):
    """Log prediction request events"""
    log_security_event(
        "PREDICTION_REQUEST",
        f"FileID: {file_id}, Model: {model_type}",
        user_ip=user_ip
    )

def log_error_with_context(error: Exception, context: dict = None):
    """Log errors with additional context"""
    context_str = ""
    if context:
        context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
    
    logger.error(f"Error occurred: {str(error)} | Context: {context_str}", exception=error)

# Performance logging
class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"🏁 Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type:
            logger.error(f"❌ Operation failed: {self.operation_name} | Duration: {duration.total_seconds():.2f}s")
        else:
            logger.info(f"✅ Operation completed: {self.operation_name} | Duration: {duration.total_seconds():.2f}s")

# Export main components
__all__ = [
    "setup_logging",
    "get_logger",
    "log_security_event",
    "log_file_upload",
    "log_prediction_request",
    "log_error_with_context",
    "PerformanceLogger"
]