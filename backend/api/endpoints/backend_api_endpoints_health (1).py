"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import psutil
import os

from backend.core.database import get_db
from backend.core.config import settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production"
    }

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with system information"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
        db_error = None
    except Exception as e:
        db_status = "unhealthy"
        db_error = str(e)
        logger.error(f"Database health check failed: {e}")
    
    # Get system information
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check critical directories
    directories_status = {}
    for dir_name, dir_path in [
        ("uploads", settings.UPLOAD_DIR),
        ("processed", settings.PROCESSED_DIR),
        ("models", settings.MODEL_SAVE_DIR),
        ("logs", settings.LOG_DIR)
    ]:
        directories_status[dir_name] = {
            "exists": os.path.exists(dir_path),
            "writable": os.access(dir_path, os.W_OK) if os.path.exists(dir_path) else False
        }
    
    health_data = {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "database": {
            "status": db_status,
            "error": db_error
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        },
        "directories": directories_status,
        "configuration": {
            "max_file_size": settings.MAX_FILE_SIZE,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "debug_mode": settings.DEBUG
        }
    }
    
    return health_data

@router.get("/readiness")
async def readiness_check(db: Session = Depends(get_db)):
    """Kubernetes readiness probe endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        
        # Check if critical directories exist and are writable
        critical_dirs = [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.MODEL_SAVE_DIR]
        for dir_path in critical_dirs:
            if not os.path.exists(dir_path) or not os.access(dir_path, os.W_OK):
                return {"status": "not_ready", "reason": f"Directory not accessible: {dir_path}"}, 503
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "reason": str(e)}, 503

@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    # Simple check to ensure the application is running
    return {"status": "alive", "timestamp": datetime.now().isoformat()}