"""Export service for data and results in various formats."""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO, StringIO

import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse

from backend.core.config import settings
from backend.core.logging import audit_logger, EventType
from backend.core.exceptions import ValidationError, SecurityError
from backend.services.data_service import data_service
from backend.services.model_service import model_service
from backend.services.cache_service import cache_service


@dataclass
class ExportRequest:
    """Export request configuration."""
    export_type: str  # 'dataset', 'analysis', 'model_results', 'predictions'
    format: str  # 'csv', 'excel', 'json', 'parquet'
    resource_id: str  # dataset_id, model_id, etc.
    include_metadata: bool = True
    custom_filename: Optional[str] = None
    export_options: Dict[str, Any] = None


@dataclass
class ExportInfo:
    """Information about an export operation."""
    export_id: str
    export_type: str
    format: str
    resource_id: str
    filename: str
    file_size: int
    created_at: str
    user_id: str
    status: str
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = None


class ExportService:
    """Service for exporting data and results in various formats."""
    
    SUPPORTED_FORMATS = {
        'csv': 'text/csv',
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'json': 'application/json',
        'parquet': 'application/octet-stream'
    }
    
    MAX_EXPORT_SIZE = 100 * 1024 * 1024  # 100MB
    EXPORT_CACHE_TTL = 3600  # 1 hour
    
    def __init__(self):
        """Initialize export service."""
        self.exports_storage = Path(settings.PROCESSED_PATH) / "exports"
        self.exports_storage.mkdir(parents=True, exist_ok=True)
    
    async def export_dataset(self, dataset_id: str, user_id: str,
                           format: str = 'csv', 
                           include_metadata: bool = True,
                           custom_filename: Optional[str] = None) -> ExportInfo:
        """Export a dataset in the specified format.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User requesting export
            format: Export format ('csv', 'excel', 'json', 'parquet')
            include_metadata: Whether to include metadata
            custom_filename: Custom filename for export
            
        Returns:
            ExportInfo with export details
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValidationError(f"Unsupported export format: {format}")
        
        # Load dataset
        df = await data_service.load_dataset(dataset_id, user_id)
        
        # Generate export filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            filename = f"{custom_filename}_{timestamp}.{format}"
        else:
            filename = f"dataset_{dataset_id[:8]}_{timestamp}.{format}"
        
        # Create export
        export_id = str(uuid.uuid4())
        export_path = self.exports_storage / filename
        
        # Export data based on format
        if format == 'csv':
            df.to_csv(export_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                if include_metadata:
                    # Add metadata sheet
                    metadata_df = pd.DataFrame([
                        ['Dataset ID', dataset_id],
                        ['Export Date', datetime.utcnow().isoformat()],
                        ['Rows', len(df)],
                        ['Columns', len(df.columns)],
                        ['Data Types', str(df.dtypes.to_dict())],
                        ['Memory Usage (MB)', round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)]
                    ], columns=['Property', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        elif format == 'json':
            export_data = {
                'data': df.to_dict('records'),
                'metadata': {
                    'dataset_id': dataset_id,
                    'export_date': datetime.utcnow().isoformat(),
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                } if include_metadata else None
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'parquet':
            df.to_parquet(export_path, index=False)
        
        # Create export info
        export_info = ExportInfo(
            export_id=export_id,
            export_type='dataset',
            format=format,
            resource_id=dataset_id,
            filename=filename,
            file_size=export_path.stat().st_size,
            created_at=datetime.utcnow().isoformat(),
            user_id=user_id,
            status='completed',
            metadata={
                'original_shape': df.shape,
                'include_metadata': include_metadata
            }
        )
        
        # Cache export info
        await cache_service.set(f"export:{export_id}", export_info, ttl=self.EXPORT_CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="dataset_exported",
            details={
                "export_id": export_id,
                "format": format,
                "file_size": export_info.file_size,
                "filename": filename
            }
        )
        
        return export_info
    
    async def export_analysis(self, dataset_id: str, user_id: str,
                            format: str = 'json',
                            custom_filename: Optional[str] = None) -> ExportInfo:
        """Export data analysis results.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User requesting export
            format: Export format
            custom_filename: Custom filename
            
        Returns:
            ExportInfo with export details
        """
        if format not in ['json', 'excel']:
            raise ValidationError(f"Analysis export only supports json and excel formats")
        
        # Get analysis results
        analysis = await data_service.analyze_dataset(dataset_id, user_id)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            filename = f"{custom_filename}_analysis_{timestamp}.{format}"
        else:
            filename = f"analysis_{dataset_id[:8]}_{timestamp}.{format}"
        
        export_id = str(uuid.uuid4())
        export_path = self.exports_storage / filename
        
        if format == 'json':
            # Convert analysis to JSON-serializable format
            analysis_dict = {
                'dataset_id': analysis.dataset_id,
                'shape': analysis.shape,
                'columns': analysis.columns,
                'data_types': analysis.data_types,
                'missing_values': analysis.missing_values,
                'missing_percentages': analysis.missing_percentages,
                'numeric_summary': analysis.numeric_summary,
                'categorical_summary': analysis.categorical_summary,
                'correlation_matrix': analysis.correlation_matrix,
                'outliers': analysis.outliers,
                'duplicates': analysis.duplicates,
                'memory_usage': analysis.memory_usage,
                'analysis_timestamp': analysis.analysis_timestamp
            }
            
            with open(export_path, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
        
        elif format == 'excel':
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = [
                    ['Dataset ID', analysis.dataset_id],
                    ['Analysis Date', analysis.analysis_timestamp],
                    ['Shape (Rows, Cols)', f"{analysis.shape[0]}, {analysis.shape[1]}"],
                    ['Total Duplicates', analysis.duplicates],
                    ['Memory Usage (MB)', analysis.memory_usage]
                ]
                summary_df = pd.DataFrame(summary_data, columns=['Property', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Numeric summary
                if analysis.numeric_summary:
                    numeric_df = pd.DataFrame(analysis.numeric_summary).T
                    numeric_df.to_excel(writer, sheet_name='Numeric_Summary')
                
                # Categorical summary
                if analysis.categorical_summary:
                    cat_data = []
                    for col, stats in analysis.categorical_summary.items():
                        cat_data.append([
                            col, stats['count'], stats['unique'], 
                            stats['top_value'], stats['top_frequency']
                        ])
                    
                    cat_df = pd.DataFrame(cat_data, columns=[
                        'Column', 'Count', 'Unique', 'Top_Value', 'Top_Frequency'
                    ])
                    cat_df.to_excel(writer, sheet_name='Categorical_Summary', index=False)
                
                # Missing values
                missing_df = pd.DataFrame(list(analysis.missing_values.items()), 
                                        columns=['Column', 'Missing_Count'])
                missing_df['Missing_Percentage'] = [analysis.missing_percentages.get(col, 0) 
                                                  for col in missing_df['Column']]
                missing_df.to_excel(writer, sheet_name='Missing_Values', index=False)
                
                # Correlation matrix
                if analysis.correlation_matrix:
                    corr_df = pd.DataFrame(analysis.correlation_matrix)
                    corr_df.to_excel(writer, sheet_name='Correlation_Matrix')
        
        export_info = ExportInfo(
            export_id=export_id,
            export_type='analysis',
            format=format,
            resource_id=dataset_id,
            filename=filename,
            file_size=export_path.stat().st_size,
            created_at=datetime.utcnow().isoformat(),
            user_id=user_id,
            status='completed',
            metadata={
                'analysis_timestamp': analysis.analysis_timestamp,
                'shape': analysis.shape
            }
        )
        
        await cache_service.set(f"export:{export_id}", export_info, ttl=self.EXPORT_CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="analysis_exported",
            details={
                "export_id": export_id,
                "format": format,
                "file_size": export_info.file_size
            }
        )
        
        return export_info
    
    async def export_model_results(self, model_id: str, user_id: str,
                                 format: str = 'json',
                                 custom_filename: Optional[str] = None) -> ExportInfo:
        """Export model training results and performance metrics.
        
        Args:
            model_id: Model identifier
            user_id: User requesting export
            format: Export format
            custom_filename: Custom filename
            
        Returns:
            ExportInfo with export details
        """
        # Get model info
        model_info = await model_service.get_model_info(model_id, user_id)
        if not model_info:
            raise SecurityError("Model not found or access denied")
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            filename = f"{custom_filename}_model_{timestamp}.{format}"
        else:
            filename = f"model_{model_id[:8]}_{timestamp}.{format}"
        
        export_id = str(uuid.uuid4())
        export_path = self.exports_storage / filename
        
        if format == 'json':
            model_dict = {
                'model_id': model_info.model_id,
                'name': model_info.name,
                'model_type': model_info.model_type,
                'dataset_id': model_info.dataset_id,
                'target_column': model_info.target_column,
                'feature_columns': model_info.feature_columns,
                'training_config': {
                    'model_type': model_info.training_config.model_type,
                    'test_size': model_info.training_config.test_size,
                    'random_state': model_info.training_config.random_state,
                    'scaling_method': model_info.training_config.scaling_method,
                    'feature_selection': model_info.training_config.feature_selection,
                    'cross_validation': model_info.training_config.cross_validation
                },
                'performance': {
                    'r2_score': model_info.performance.r2_score,
                    'adjusted_r2': model_info.performance.adjusted_r2,
                    'mse': model_info.performance.mse,
                    'mae': model_info.performance.mae,
                    'rmse': model_info.performance.rmse,
                    'mape': model_info.performance.mape,
                    'explained_variance': model_info.performance.explained_variance,
                    'cv_scores': model_info.performance.cv_scores,
                    'cv_mean': model_info.performance.cv_mean,
                    'cv_std': model_info.performance.cv_std
                },
                'feature_importance': model_info.feature_importance,
                'model_size': model_info.model_size,
                'training_time': model_info.training_time,
                'created_at': model_info.created_at,
                'metadata': model_info.metadata
            }
            
            with open(export_path, 'w') as f:
                json.dump(model_dict, f, indent=2, default=str)
        
        elif format == 'excel':
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Model info sheet
                info_data = [
                    ['Model ID', model_info.model_id],
                    ['Name', model_info.name],
                    ['Type', model_info.model_type],
                    ['Dataset ID', model_info.dataset_id],
                    ['Target Column', model_info.target_column],
                    ['Feature Count', len(model_info.feature_columns)],
                    ['Training Time (s)', model_info.training_time],
                    ['Model Size (bytes)', model_info.model_size],
                    ['Created At', model_info.created_at]
                ]
                info_df = pd.DataFrame(info_data, columns=['Property', 'Value'])
                info_df.to_excel(writer, sheet_name='Model_Info', index=False)
                
                # Performance sheet
                perf_data = [
                    ['Metric', 'Value'],
                    ['R² Score', model_info.performance.r2_score],
                    ['Adjusted R²', model_info.performance.adjusted_r2],
                    ['MSE', model_info.performance.mse],
                    ['MAE', model_info.performance.mae],
                    ['RMSE', model_info.performance.rmse],
                    ['MAPE', model_info.performance.mape],
                    ['Explained Variance', model_info.performance.explained_variance],
                    ['CV Mean', model_info.performance.cv_mean],
                    ['CV Std', model_info.performance.cv_std]
                ]
                perf_df = pd.DataFrame(perf_data, columns=['Metric', 'Value'])
                perf_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Feature importance
                if model_info.feature_importance:
                    importance_df = pd.DataFrame(list(model_info.feature_importance.items()),
                                               columns=['Feature', 'Importance'])
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        export_info = ExportInfo(
            export_id=export_id,
            export_type='model_results',
            format=format,
            resource_id=model_id,
            filename=filename,
            file_size=export_path.stat().st_size,
            created_at=datetime.utcnow().isoformat(),
            user_id=user_id,
            status='completed',
            metadata={
                'model_type': model_info.model_type,
                'r2_score': model_info.performance.r2_score
            }
        )
        
        await cache_service.set(f"export:{export_id}", export_info, ttl=self.EXPORT_CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=model_info.dataset_id,
            operation="model_results_exported",
            details={
                "export_id": export_id,
                "model_id": model_id,
                "format": format
            }
        )
        
        return export_info
    
    async def export_predictions(self, predictions: Dict[str, Any], user_id: str,
                               format: str = 'csv',
                               custom_filename: Optional[str] = None) -> ExportInfo:
        """Export prediction results.
        
        Args:
            predictions: Prediction results from model service
            user_id: User requesting export
            format: Export format
            custom_filename: Custom filename
            
        Returns:
            ExportInfo with export details
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            filename = f"{custom_filename}_predictions_{timestamp}.{format}"
        else:
            filename = f"predictions_{predictions['model_id'][:8]}_{timestamp}.{format}"
        
        export_id = str(uuid.uuid4())
        export_path = self.exports_storage / filename
        
        # Create predictions DataFrame
        pred_data = {
            'prediction': predictions['predictions']
        }
        
        if predictions.get('confidence'):
            pred_data['confidence'] = predictions['confidence']
        
        # Add feature columns if available in input
        for i, feature in enumerate(predictions['feature_names']):
            pred_data[f'feature_{feature}'] = [None] * len(predictions['predictions'])
        
        pred_df = pd.DataFrame(pred_data)
        
        if format == 'csv':
            pred_df.to_csv(export_path, index=False)
        
        elif format == 'excel':
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                pred_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Metadata sheet
                metadata_data = [
                    ['Model ID', predictions['model_id']],
                    ['Prediction Count', predictions['n_samples']],
                    ['Timestamp', predictions['timestamp']],
                    ['Features Used', ', '.join(predictions['feature_names'])]
                ]
                metadata_df = pd.DataFrame(metadata_data, columns=['Property', 'Value'])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        elif format == 'json':
            export_data = {
                'predictions': predictions,
                'export_metadata': {
                    'export_id': export_id,
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'user_id': user_id
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        export_info = ExportInfo(
            export_id=export_id,
            export_type='predictions',
            format=format,
            resource_id=predictions['model_id'],
            filename=filename,
            file_size=export_path.stat().st_size,
            created_at=datetime.utcnow().isoformat(),
            user_id=user_id,
            status='completed',
            metadata={
                'prediction_count': predictions['n_samples'],
                'model_id': predictions['model_id']
            }
        )
        
        await cache_service.set(f"export:{export_id}", export_info, ttl=self.EXPORT_CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id="predictions",
            operation="predictions_exported",
            details={
                "export_id": export_id,
                "model_id": predictions['model_id'],
                "format": format,
                "prediction_count": predictions['n_samples']
            }
        )
        
        return export_info
    
    async def get_export_info(self, export_id: str, user_id: str) -> Optional[ExportInfo]:
        """Get information about an export.
        
        Args:
            export_id: Export identifier
            user_id: User requesting info
            
        Returns:
            ExportInfo or None if not found
        """
        export_info = await cache_service.get(f"export:{export_id}")
        
        if export_info and export_info.user_id == user_id:
            return export_info
        
        return None
    
    async def download_export(self, export_id: str, user_id: str) -> StreamingResponse:
        """Download an exported file.
        
        Args:
            export_id: Export identifier
            user_id: User requesting download
            
        Returns:
            StreamingResponse with file content
        """
        export_info = await self.get_export_info(export_id, user_id)
        
        if not export_info:
            raise SecurityError("Export not found or access denied")
        
        export_path = self.exports_storage / export_info.filename
        
        if not export_path.exists():
            raise ValueError("Export file not found")
        
        # Get media type
        media_type = self.SUPPORTED_FORMATS.get(export_info.format, 'application/octet-stream')
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=export_info.resource_id,
            operation="export_downloaded",
            details={
                "export_id": export_id,
                "filename": export_info.filename,
                "format": export_info.format
            }
        )
        
        def file_generator():
            with open(export_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_generator(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={export_info.filename}",
                "Content-Length": str(export_info.file_size)
            }
        )
    
    async def cleanup_expired_exports(self, max_age_hours: int = 24) -> int:
        """Clean up expired export files.
        
        Args:
            max_age_hours: Maximum age of exports to keep
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for export_file in self.exports_storage.iterdir():
            if export_file.is_file() and export_file.stat().st_mtime < cutoff_time:
                try:
                    export_file.unlink()
                    cleaned_count += 1
                except Exception:
                    pass  # Continue cleanup even if some files fail
        
        audit_logger.log_event(
            EventType.SYSTEM_START,  # Using as cleanup event
            outcome="success",
            details={
                "action": "export_cleanup",
                "files_cleaned": cleaned_count,
                "max_age_hours": max_age_hours
            }
        )
        
        return cleaned_count


# Global export service instance
export_service = ExportService()