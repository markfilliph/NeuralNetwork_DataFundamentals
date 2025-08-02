"""Data processing service with pandas integration for exploratory data analysis."""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from scipy import stats

from backend.core.config import settings
from backend.core.logging import audit_logger, EventType
from backend.core.exceptions import SecurityError, ValidationError
from backend.services.encryption_service import encryption_service
from backend.services.cache_service import cache_service
from backend.utils.sanitizers import DataSanitizer
from backend.utils.validators import DataValidator
from backend.models.database import db_manager


@dataclass
class DatasetInfo:
    """Dataset information structure."""
    dataset_id: str
    name: str
    file_path: str
    file_size: int
    file_hash: str
    owner_id: str
    is_encrypted: bool
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


@dataclass
class DataAnalysis:
    """Data analysis results structure."""
    dataset_id: str
    shape: Tuple[int, int]
    columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    numeric_summary: Dict[str, Dict[str, float]]
    categorical_summary: Dict[str, Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    outliers: Dict[str, List[Any]]
    duplicates: int
    memory_usage: float
    analysis_timestamp: str


class DataProcessingService:
    """Service for data processing and exploratory data analysis."""
    
    MAX_DATASET_SIZE = 1024 * 1024 * 1024  # 1GB
    MAX_CORRELATION_COLUMNS = 50  # Limit correlation matrix size
    CACHE_TTL = 3600  # 1 hour cache
    
    def __init__(self):
        """Initialize data processing service."""
        self.sanitizer = DataSanitizer()
        self.validator = DataValidator()
    
    async def load_dataset(self, dataset_id: str, user_id: str) -> pd.DataFrame:
        """Load dataset from storage with security checks.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User requesting the data
            
        Returns:
            Loaded pandas DataFrame
            
        Raises:
            SecurityError: If user lacks permissions
            ValueError: If dataset not found or corrupted
        """
        # Get dataset metadata from database
        dataset_info = self._get_dataset_info(dataset_id)
        
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Check ownership or permissions
        if dataset_info.owner_id != user_id:
            # TODO: Check if user has read permissions for this dataset
            audit_logger.log_security_event(
                EventType.DATA_ACCESS,
                user_id=user_id,
                details={
                    "action": "unauthorized_dataset_access",
                    "dataset_id": dataset_id,
                    "owner_id": dataset_info.owner_id
                },
                risk_level="high"
            )
            raise SecurityError("Insufficient permissions to access dataset")
        
        # Check cache first
        cache_key = f"dataset:{dataset_id}:{dataset_info.file_hash}"
        cached_df = cache_service.get(cache_key)
        
        if cached_df is not None:
            audit_logger.log_data_access(
                user_id=user_id,
                dataset_id=dataset_id,
                operation="load_cached",
                details={"cache_hit": True}
            )
            return cached_df
        
        try:
            # Load from file
            file_path = Path(dataset_info.file_path)
            
            if not file_path.exists():
                raise ValueError(f"Dataset file not found: {file_path}")
            
            # Decrypt if necessary
            if dataset_info.is_encrypted:
                decrypted_path = encryption_service.decrypt_file(
                    file_path,
                    file_path.with_suffix('.tmp')
                )
                load_path = decrypted_path
                # Use original file extension from metadata
                original_extension = dataset_info.metadata.get('file_type', '.csv')
            else:
                load_path = file_path
                original_extension = file_path.suffix
            
            # Load based on original file extension
            df = self._load_file_by_extension_with_type(load_path, original_extension)
            
            # Clean up temporary decrypted file
            if dataset_info.is_encrypted and load_path.exists():
                load_path.unlink()
            
            # Validate and sanitize data
            df = self.sanitizer.sanitize_dataframe(df)
            df = self.sanitizer.sanitize_column_names(df)
            
            # Cache the loaded dataset
            cache_service.set(cache_key, df, ttl=self.CACHE_TTL)
            
            audit_logger.log_data_access(
                user_id=user_id,
                dataset_id=dataset_id,
                operation="load_success",
                details={
                    "shape": df.shape,
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            )
            
            return df
            
        except Exception as e:
            audit_logger.log_security_event(
                EventType.DATA_ACCESS,
                user_id=user_id,
                details={
                    "action": "dataset_load_failed",
                    "dataset_id": dataset_id,
                    "error": str(e)
                },
                risk_level="medium"
            )
            raise ValueError(f"Failed to load dataset: {str(e)}")
    
    def _load_file_by_extension(self, file_path: Path) -> pd.DataFrame:
        """Load file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded DataFrame
        """
        # Handle encrypted files - look at the extension before .encrypted
        if file_path.suffix.lower() == '.encrypted':
            # Get the actual file extension before .encrypted
            actual_extension = file_path.with_suffix('').suffix.lower()
        else:
            actual_extension = file_path.suffix.lower()
        
        extension = actual_extension
        return self._load_by_extension_type(file_path, extension)
    
    def _load_file_by_extension_with_type(self, file_path: Path, extension: str) -> pd.DataFrame:
        """Load file based on provided extension type.
        
        Args:
            file_path: Path to the file
            extension: File extension to use for loading
            
        Returns:
            Loaded DataFrame
        """
        return self._load_by_extension_type(file_path, extension)
    
    def _load_by_extension_type(self, file_path: Path, extension: str) -> pd.DataFrame:
        """Load file based on extension type.
        
        Args:
            file_path: Path to the file
            extension: File extension
            
        Returns:
            Loaded DataFrame
        """
        extension = extension.lower()
        
        if extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif extension == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode CSV file with any supported encoding")
        elif extension == '.json':
            return pd.read_json(file_path)
        elif extension == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    async def analyze_dataset(self, dataset_id: str, user_id: str, 
                            include_correlation: bool = True,
                            detect_outliers: bool = True) -> DataAnalysis:
        """Perform comprehensive exploratory data analysis.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User performing analysis
            include_correlation: Whether to compute correlation matrix
            detect_outliers: Whether to detect outliers
            
        Returns:
            DataAnalysis object with complete analysis results
        """
        # Check cache first
        cache_key = f"analysis:{dataset_id}:{include_correlation}:{detect_outliers}"
        cached_analysis = cache_service.get(cache_key)
        
        if cached_analysis:
            return cached_analysis
        
        # Load dataset
        df = await self.load_dataset(dataset_id, user_id)
        
        # Basic info
        shape = df.shape
        columns = df.columns.tolist()
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentages = {
            col: round((missing / len(df)) * 100, 2) 
            for col, missing in missing_values.items()
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_summary = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                numeric_summary[col] = {
                    'count': len(series),
                    'mean': round(series.mean(), 4),
                    'std': round(series.std(), 4),
                    'min': round(series.min(), 4),
                    'q25': round(series.quantile(0.25), 4),
                    'median': round(series.median(), 4),
                    'q75': round(series.quantile(0.75), 4),
                    'max': round(series.max(), 4),
                    'skewness': round(stats.skew(series), 4),
                    'kurtosis': round(stats.kurtosis(series), 4)
                }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_summary = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                categorical_summary[col] = {
                    'count': len(series),
                    'unique': series.nunique(),
                    'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'top_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'value_counts': value_counts.head(10).to_dict()  # Top 10 values
                }
        
        # Correlation matrix (for numeric columns only)
        correlation_matrix = None
        if include_correlation and len(numeric_cols) > 1:
            # Limit correlation matrix size for performance
            correlation_cols = numeric_cols[:self.MAX_CORRELATION_COLUMNS]
            corr_df = df[correlation_cols].corr()
            correlation_matrix = {
                col: corr_df[col].to_dict() 
                for col in corr_df.columns
            }
        
        # Outlier detection using IQR method
        outliers = {}
        if detect_outliers:
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) > 0:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (series < lower_bound) | (series > upper_bound)
                    outlier_values = series[outlier_mask].tolist()
                    
                    if outlier_values:
                        outliers[col] = {
                            'count': len(outlier_values),
                            'percentage': round((len(outlier_values) / len(series)) * 100, 2),
                            'values': outlier_values[:20]  # Limit to first 20 outliers
                        }
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        
        # Memory usage
        memory_usage = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)  # MB
        
        # Create analysis object
        analysis = DataAnalysis(
            dataset_id=dataset_id,
            shape=shape,
            columns=columns,
            data_types=data_types,
            missing_values=missing_values,
            missing_percentages=missing_percentages,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            outliers=outliers,
            duplicates=int(duplicates),
            memory_usage=memory_usage,
            analysis_timestamp=datetime.utcnow().isoformat()
        )
        
        # Cache the analysis
        cache_service.set(cache_key, analysis, ttl=self.CACHE_TTL)
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="analyze_complete",
            details={
                "shape": shape,
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "memory_mb": memory_usage
            }
        )
        
        return analysis
    
    async def get_data_sample(self, dataset_id: str, user_id: str, 
                            n_rows: int = 100, random: bool = False) -> Dict[str, Any]:
        """Get a sample of the dataset for preview.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User requesting sample
            n_rows: Number of rows to return
            random: Whether to sample randomly
            
        Returns:
            Dictionary with sample data and metadata
        """
        df = await self.load_dataset(dataset_id, user_id)
        
        # Get sample
        if random and len(df) > n_rows:
            sample_df = df.sample(n=n_rows, random_state=42)
        else:
            sample_df = df.head(n_rows)
        
        # Convert to serializable format
        sample_data = {
            'data': sample_df.to_dict('records'),
            'columns': sample_df.columns.tolist(),
            'shape': sample_df.shape,
            'total_rows': len(df),
            'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            'sample_type': 'random' if random else 'head',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="sample_data",
            details={
                "sample_rows": len(sample_df),
                "sample_type": sample_data['sample_type']
            }
        )
        
        return sample_data
    
    async def clean_dataset(self, dataset_id: str, user_id: str,
                          cleaning_options: Dict[str, Any]) -> Dict[str, Any]:
        """Clean dataset based on specified options.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User performing cleaning
            cleaning_options: Dictionary with cleaning parameters
            
        Returns:
            Dictionary with cleaning results and statistics
        """
        df = await self.load_dataset(dataset_id, user_id)
        original_shape = df.shape
        
        cleaning_log = []
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            duplicates_before = df.duplicated().sum()
            df = df.drop_duplicates()
            duplicates_removed = duplicates_before - df.duplicated().sum()
            cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_strategy = cleaning_options.get('missing_values_strategy', 'none')
        if missing_strategy != 'none':
            missing_before = df.isnull().sum().sum()
            
            if missing_strategy == 'drop_rows':
                df = df.dropna()
            elif missing_strategy == 'drop_columns':
                threshold = cleaning_options.get('missing_threshold', 0.5)
                missing_ratio = df.isnull().sum() / len(df)
                cols_to_drop = missing_ratio[missing_ratio > threshold].index
                df = df.drop(columns=cols_to_drop)
                cleaning_log.append(f"Dropped columns with >{threshold*100}% missing: {list(cols_to_drop)}")
            elif missing_strategy == 'fill_numeric':
                fill_method = cleaning_options.get('fill_method', 'mean')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_method == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif fill_method == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                        elif fill_method == 'mode':
                            df[col].fillna(df[col].mode().iloc[0], inplace=True)
            
            missing_after = df.isnull().sum().sum()
            cleaning_log.append(f"Handled {missing_before - missing_after} missing values")
        
        # Remove outliers
        if cleaning_options.get('remove_outliers', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_removed = 0
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_in_col = outlier_mask.sum()
                outliers_removed += outliers_in_col
                
                df = df[~outlier_mask]
            
            cleaning_log.append(f"Removed {outliers_removed} outlier rows")
        
        final_shape = df.shape
        
        # Save cleaned dataset (create new dataset ID)
        cleaned_dataset_id = f"{dataset_id}_cleaned_{int(datetime.utcnow().timestamp())}"
        
        # Store cleaned dataset
        storage_path = Path(settings.PROCESSED_PATH) / f"{cleaned_dataset_id}.parquet"
        df.to_parquet(storage_path, index=False)
        
        # Update database with cleaned dataset info
        self._save_dataset_info(
            dataset_id=cleaned_dataset_id,
            name=f"Cleaned_{dataset_id}",
            file_path=str(storage_path),
            file_size=storage_path.stat().st_size,
            owner_id=user_id,
            is_encrypted=False,
            metadata={
                'parent_dataset': dataset_id,
                'cleaning_options': cleaning_options,
                'cleaning_log': cleaning_log
            }
        )
        
        cleaning_results = {
            'cleaned_dataset_id': cleaned_dataset_id,
            'original_shape': original_shape,
            'cleaned_shape': final_shape,
            'rows_removed': original_shape[0] - final_shape[0],
            'columns_removed': original_shape[1] - final_shape[1],
            'cleaning_log': cleaning_log,
            'storage_path': str(storage_path),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        audit_logger.log_data_access(
            user_id=user_id,
            dataset_id=dataset_id,
            operation="dataset_cleaned",
            details=cleaning_results
        )
        
        return cleaning_results
    
    def _get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get dataset information from database.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            DatasetInfo object or None if not found
        """
        query = "SELECT * FROM datasets WHERE dataset_id = ?"
        results = db_manager.execute_query(query, (dataset_id,))
        
        if results:
            row = results[0]
            return DatasetInfo(
                dataset_id=row['dataset_id'],
                name=row['name'],
                file_path=row['file_path'],
                file_size=row['file_size'],
                file_hash=row['file_hash'],
                owner_id=row['owner_id'],
                is_encrypted=bool(row['is_encrypted']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                metadata=json.loads(row['metadata'] or '{}')
            )
        
        return None
    
    def _save_dataset_info(self, dataset_id: str, name: str, file_path: str,
                               file_size: int, owner_id: str, is_encrypted: bool,
                               metadata: Dict[str, Any]) -> bool:
        """Save dataset information to database.
        
        Args:
            dataset_id: Dataset identifier
            name: Dataset name
            file_path: Path to dataset file
            file_size: File size in bytes
            owner_id: Owner user ID
            is_encrypted: Whether file is encrypted
            metadata: Additional metadata
            
        Returns:
            True if saved successfully
        """
        # Calculate file hash
        file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        
        query = '''
            INSERT INTO datasets 
            (dataset_id, name, file_path, file_size, file_hash, owner_id, is_encrypted, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        affected = db_manager.execute_update(
            query,
            (dataset_id, name, file_path, file_size, file_hash, owner_id, is_encrypted, json.dumps(metadata))
        )
        
        return affected > 0


# Global data service instance
data_service = DataProcessingService()