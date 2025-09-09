"""
Data management endpoints for file upload, processing, and data operations
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Query
from fastapi.responses import FileResponse
from typing import Optional, List
import os
from pathlib import Path

from backend.services.data_service import data_service
from backend.core.config import settings
from backend.core.logging_config import get_logger
from backend.utils.validators import validate_file_upload

logger = get_logger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    description: Optional[str] = None
):
    """
    Upload a data file (CSV, XLSX, PDF) for processing
    
    - **file**: The file to upload (CSV, XLSX, XLS, PDF)
    - **description**: Optional description for the file
    """
    try:
        # Get client IP for logging
        client_ip = request.client.host
        
        # Validate file upload
        validation_result = validate_file_upload(file)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Process upload
        result = await data_service.upload_file(file, user_ip=client_ip)
        
        return {
            "success": True,
            "data": result,
            "message": "File uploaded and processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@router.get("/files")
async def list_files(
    limit: int = Query(50, ge=1, le=100, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    status: Optional[str] = Query(None, description="Filter by status (uploaded, processing, processed, error)")
):
    """
    List uploaded files with pagination
    
    - **limit**: Maximum number of files to return (1-100)
    - **offset**: Number of files to skip for pagination
    - **status**: Filter files by processing status
    """
    try:
        result = data_service.list_files(limit=limit, offset=offset)
        
        # Apply status filter if provided
        if status:
            result["files"] = [f for f in result["files"] if f["status"] == status]
            result["total_count"] = len(result["files"])
        
        return {
            "success": True,
            "data": result,
            "message": f"Retrieved {len(result['files'])} files"
        }
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

@router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """
    Get detailed information about a specific file
    
    - **file_id**: Unique identifier of the uploaded file
    """
    try:
        result = data_service.get_file_info(file_id)
        
        return {
            "success": True,
            "data": result,
            "message": "File information retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file information")

@router.get("/files/{file_id}/data")
async def get_file_data(
    file_id: str,
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of rows to return")
):
    """
    Get processed data from a file
    
    - **file_id**: Unique identifier of the uploaded file
    - **limit**: Maximum number of data rows to return (1-10000)
    """
    try:
        result = data_service.get_file_data(file_id, limit=limit)
        
        return {
            "success": True,
            "data": result,
            "message": f"Retrieved {result['returned_rows']} rows of data"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file data")

@router.get("/files/{file_id}/summary")
async def get_file_summary(file_id: str):
    """
    Get summary statistics for a processed file
    
    - **file_id**: Unique identifier of the uploaded file
    """
    try:
        result = data_service.get_summary_statistics(file_id)
        
        return {
            "success": True,
            "data": result,
            "message": "File summary statistics generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate summary statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary statistics")

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file and its processed data
    
    - **file_id**: Unique identifier of the uploaded file
    """
    try:
        result = data_service.delete_file(file_id)
        
        return {
            "success": True,
            "data": result,
            "message": "File deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed file or prediction results
    
    - **filename**: Name of the file to download
    """
    try:
        # Security check - only allow downloads from processed directory
        safe_filename = Path(filename).name  # Remove any path traversal
        file_path = Path(settings.PROCESSED_DIR) / safe_filename
        
        # Check if file exists in processed directory or predictions subdirectory
        if not file_path.exists():
            predictions_path = Path(settings.PROCESSED_DIR) / "predictions" / safe_filename
            if predictions_path.exists():
                file_path = predictions_path
            else:
                raise HTTPException(status_code=404, detail="File not found")
        
        # Security check - ensure file is within allowed directories
        allowed_dirs = [
            Path(settings.PROCESSED_DIR).resolve(),
            Path(settings.PROCESSED_DIR).resolve() / "predictions"
        ]
        
        file_path_resolved = file_path.resolve()
        if not any(str(file_path_resolved).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            path=file_path,
            filename=safe_filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats and their specifications
    """
    try:
        from backend.utils.file_handlers import FileProcessor
        
        formats_info = {
            "supported_extensions": FileProcessor.get_supported_extensions(),
            "max_file_size": settings.MAX_FILE_SIZE,
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "format_details": {
                ".csv": {
                    "description": "Comma-separated values",
                    "supported_delimiters": [",", ";", "\t", "|"],
                    "encoding_support": ["utf-8", "latin-1", "cp1252"],
                    "max_rows": 1000000
                },
                ".xlsx": {
                    "description": "Excel spreadsheet (modern format)",
                    "max_sheets": 10,
                    "max_rows_per_sheet": 1000000
                },
                ".xls": {
                    "description": "Excel spreadsheet (legacy format)",
                    "max_sheets": 10,
                    "max_rows_per_sheet": 1000000
                },
                ".pdf": {
                    "description": "Portable Document Format",
                    "extraction_types": ["tables", "text"],
                    "max_pages": 1000
                }
            }
        }
        
        return {
            "success": True,
            "data": formats_info,
            "message": "Supported formats retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported formats")

@router.post("/validate")
async def validate_file_endpoint(file: UploadFile = File(...)):
    """
    Validate a file without uploading it
    
    - **file**: The file to validate
    """
    try:
        validation_result = validate_file_upload(file)
        
        # Reset file pointer
        await file.seek(0)
        
        return {
            "success": validation_result["valid"],
            "data": validation_result,
            "message": "File validation completed"
        }
        
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        raise HTTPException(status_code=500, detail="File validation failed")

@router.get("/statistics/overview")
async def get_platform_statistics():
    """
    Get platform usage statistics
    """
    try:
        # This would typically come from the database
        # For now, return basic stats
        stats = {
            "total_files_uploaded": 0,
            "total_files_processed": 0,
            "total_predictions_made": 0,
            "supported_formats": len(settings.ALLOWED_EXTENSIONS),
            "average_processing_time": 0,
            "platform_uptime": "N/A"
        }
        
        return {
            "success": True,
            "data": stats,
            "message": "Platform statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get platform statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get platform statistics")