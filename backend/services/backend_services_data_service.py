"""
Data service for handling file uploads, processing, and data management
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from backend.core.config import settings, file_validation
from backend.core.database import UploadedFile, get_db_session
from backend.core.logging_config import get_logger, log_file_upload, PerformanceLogger
from backend.utils.file_handlers import FileProcessor, FileProcessingError

logger = get_logger(__name__)

class DataService:
    """Service for data operations"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.processed_dir = Path(settings.PROCESSED_DIR)
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(self, file: UploadFile, user_ip: str = None) -> Dict[str, Any]:
        """Upload and process a file"""
        with PerformanceLogger(f"File upload: {file.filename}"):
            try:
                # Validate file
                validation_result = self._validate_file(file)
                if not validation_result["valid"]:
                    raise HTTPException(status_code=400, detail=validation_result["error"])
                
                # Generate unique file ID and save file
                file_id = str(uuid.uuid4())
                file_info = await self._save_uploaded_file(file, file_id)
                
                # Log security event
                log_file_upload(file.filename, file_info["file_size"], user_ip)
                
                # Process file
                processing_result = self._process_uploaded_file(file_info)
                
                # Save to database
                db_record = self._save_to_database(file_info, processing_result)
                
                logger.info(f"File uploaded and processed successfully: {file_id}")
                
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "file_size": file_info["file_size"],
                    "file_type": file_info["file_type"],
                    "status": processing_result["status"],
                    "rows_processed": processing_result.get("metadata", {}).get("rows", 0),
                    "columns_count": processing_result.get("metadata", {}).get("columns", 0),
                    "processing_info": processing_result.get("metadata", {}),
                    "message": "File uploaded and processed successfully"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"File upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get information about an uploaded file"""
        try:
            db = get_db_session()
            try:
                file_record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                
                if not file_record:
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Parse columns info if available
                columns_info = {}
                if file_record.columns_info:
                    try:
                        columns_info = json.loads(file_record.columns_info)
                    except json.JSONDecodeError:
                        pass
                
                return {
                    "file_id": file_record.file_id,
                    "filename": file_record.original_filename,
                    "file_type": file_record.file_type,
                    "file_size": file_record.file_size,
                    "status": file_record.status,
                    "rows_count": file_record.rows_count,
                    "columns_count": file_record.columns_count,
                    "columns_info": columns_info,
                    "created_at": file_record.created_at.isoformat(),
                    "processed_at": file_record.processed_at.isoformat() if file_record.processed_at else None,
                    "error_message": file_record.error_message
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve file information")
    
    def get_file_data(self, file_id: str, limit: int = 1000) -> Dict[str, Any]:
        """Get processed data from a file"""
        try:
            db = get_db_session()
            try:
                file_record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                
                if not file_record:
                    raise HTTPException(status_code=404, detail="File not found")
                
                if file_record.status != "processed":
                    raise HTTPException(status_code=400, detail="File not yet processed")
                
                # Find processed file
                processed_files = list(self.processed_dir.glob(f"*{Path(file_record.filename).stem}_processed.csv"))
                
                if not processed_files:
                    raise HTTPException(status_code=404, detail="Processed file not found")
                
                # Load data
                df = pd.read_csv(processed_files[0])
                
                # Limit rows if necessary
                if len(df) > limit:
                    df_sample = df.head(limit)
                    is_truncated = True
                else:
                    df_sample = df
                    is_truncated = False
                
                return {
                    "file_id": file_id,
                    "total_rows": len(df),
                    "returned_rows": len(df_sample),
                    "is_truncated": is_truncated,
                    "columns": df.columns.tolist(),
                    "data": df_sample.to_dict('records'),
                    "dtypes": df.dtypes.astype(str).to_dict()
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get file data: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve file data")
    
    def list_files(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List uploaded files"""
        try:
            db = get_db_session()
            try:
                query = db.query(UploadedFile).order_by(UploadedFile.created_at.desc())
                
                total_count = query.count()
                files = query.offset(offset).limit(limit).all()
                
                file_list = []
                for file_record in files:
                    file_list.append({
                        "file_id": file_record.file_id,
                        "filename": file_record.original_filename,
                        "file_type": file_record.file_type,
                        "file_size": file_record.file_size,
                        "status": file_record.status,
                        "rows_count": file_record.rows_count,
                        "columns_count": file_record.columns_count,
                        "created_at": file_record.created_at.isoformat(),
                        "processed_at": file_record.processed_at.isoformat() if file_record.processed_at else None
                    })
                
                return {
                    "files": file_list,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_next": offset + limit < total_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise HTTPException(status_code=500, detail="Failed to list files")
    
    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """Delete an uploaded file and its processed data"""
        try:
            db = get_db_session()
            try:
                file_record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                
                if not file_record:
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Delete physical files
                try:
                    if os.path.exists(file_record.file_path):
                        os.remove(file_record.file_path)
                    
                    # Delete processed files
                    processed_files = list(self.processed_dir.glob(f"*{Path(file_record.filename).stem}_processed.*"))
                    for pfile in processed_files:
                        os.remove(pfile)
                        
                except OSError as e:
                    logger.warning(f"Failed to delete physical files: {e}")
                
                # Delete database record
                db.delete(file_record)
                db.commit()
                
                logger.info(f"File deleted successfully: {file_id}")
                
                return {
                    "file_id": file_id,
                    "message": "File deleted successfully"
                }
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete file")
    
    def get_summary_statistics(self, file_id: str) -> Dict[str, Any]:
        """Get summary statistics for a processed file"""
        try:
            # Get file data
            file_data = self.get_file_data(file_id, limit=10000)  # Use larger sample for stats
            
            df = pd.DataFrame(file_data["data"])
            
            # Generate summary statistics
            summary = {
                "file_id": file_id,
                "total_rows": file_data["total_rows"],
                "total_columns": len(df.columns),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict()
            }
            
            # Numeric statistics
            numeric_stats = {}
            for col in summary["numeric_columns"]:
                numeric_stats[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "quartile_25": float(df[col].quantile(0.25)),
                    "quartile_75": float(df[col].quantile(0.75))
                }
            
            summary["numeric_statistics"] = numeric_stats
            
            # Categorical statistics
            categorical_stats = {}
            for col in summary["categorical_columns"]:
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                categorical_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": value_counts.to_dict(),
                    "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_value_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                }
            
            summary["categorical_statistics"] = categorical_stats
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate summary statistics")
    
    def _validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if file_size > file_validation.MAX_FILE_SIZE:
                return {
                    "valid": False,
                    "error": f"File too large: {file_size} bytes (max: {file_validation.MAX_FILE_SIZE})"
                }
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in file_validation.ALLOWED_EXTENSIONS:
                return {
                    "valid": False,
                    "error": f"Unsupported file type: {file_ext}"
                }
            
            # Basic filename validation
            if not file.filename or len(file.filename.strip()) == 0:
                return {
                    "valid": False,
                    "error": "Invalid filename"
                }
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    async def _save_uploaded_file(self, file: UploadFile, file_id: str) -> Dict[str, Any]:
        """Save uploaded file to disk"""
        try:
            # Generate safe filename
            file_ext = Path(file.filename).suffix.lower()
            safe_filename = f"{file_id}{file_ext}"
            file_path = self.upload_dir / safe_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get file info
            file_size = file_path.stat().st_size
            
            return {
                "file_id": file_id,
                "original_filename": file.filename,
                "filename": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_type": file_ext[1:],  # Remove dot
                "mime_type": file.content_type
            }
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise
    
    def _process_uploaded_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process uploaded file using FileProcessor"""
        try:
            file_path = file_info["file_path"]
            
            # Process file based on type
            processing_result = FileProcessor.process_file(file_path)
            
            if processing_result["status"] == "error":
                logger.error(f"File processing failed: {processing_result.get('error')}")
                return processing_result
            
            # Extract metadata for database
            metadata = processing_result.get("metadata", {})
            
            return {
                "status": "processed",
                "metadata": metadata,
                "processed_path": processing_result.get("processed_path"),
                "dataframe": processing_result.get("dataframe")
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": None,
                "processed_path": None
            }
    
    def _save_to_database(self, file_info: Dict[str, Any], processing_result: Dict[str, Any]) -> UploadedFile:
        """Save file information to database"""
        try:
            db = get_db_session()
            try:
                metadata = processing_result.get("metadata", {})
                
                # Create database record
                db_record = UploadedFile(
                    file_id=file_info["file_id"],
                    filename=file_info["filename"],
                    original_filename=file_info["original_filename"],
                    file_path=file_info["file_path"],
                    file_size=file_info["file_size"],
                    file_type=file_info["file_type"],
                    mime_type=file_info.get("mime_type"),
                    status=processing_result["status"],
                    error_message=processing_result.get("error"),
                    rows_count=metadata.get("rows"),
                    columns_count=metadata.get("columns"),
                    columns_info=json.dumps(metadata) if metadata else None,
                    processed_at=datetime.now() if processing_result["status"] == "processed" else None
                )
                
                db.add(db_record)
                db.commit()
                db.refresh(db_record)
                
                return db_record
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            raise

# Global instance
data_service = DataService()

# Export main components
__all__ = ["DataService", "data_service"]