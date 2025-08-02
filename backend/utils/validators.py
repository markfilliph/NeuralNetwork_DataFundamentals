"""Basic validation utilities that don't require external dependencies."""

import re
from typing import List, Dict, Any


class BasicValidator:
    """Basic validation functions."""
    
    FORMULA_PATTERN = re.compile(r'^[=+\-@]')
    SCRIPT_PATTERN = re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL)
    # Multiple patterns for different SQL injection types
    SQL_DROP_PATTERN = re.compile(r'DROP\s+(TABLE|DATABASE)', re.IGNORECASE)
    SQL_SELECT_PATTERN = re.compile(r'SELECT\s+\*\s+FROM', re.IGNORECASE)
    SQL_UPDATE_PATTERN = re.compile(r'UPDATE\s+\w+\s+SET', re.IGNORECASE)
    
    @staticmethod
    def is_safe_string(value: str) -> bool:
        """Check if a string is safe from common attacks.
        
        Args:
            value: String to validate
            
        Returns:
            True if string appears safe, False otherwise
        """
        if not isinstance(value, str):
            return True
            
        # Check for formulas
        if BasicValidator.FORMULA_PATTERN.match(value):
            return False
            
        # Check for scripts
        if BasicValidator.SCRIPT_PATTERN.search(value):
            return False
            
        # Check for SQL injection patterns
        if (BasicValidator.SQL_DROP_PATTERN.search(value) or
            BasicValidator.SQL_SELECT_PATTERN.search(value) or
            BasicValidator.SQL_UPDATE_PATTERN.search(value)):
            return False
            
        return True
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize a string by removing dangerous content.
        
        Args:
            value: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove formula prefixes
        value = BasicValidator.FORMULA_PATTERN.sub('', value)
        
        # Remove script tags
        value = BasicValidator.SCRIPT_PATTERN.sub('', value)
        
        # Remove SQL injection patterns
        value = BasicValidator.SQL_DROP_PATTERN.sub('', value)
        value = BasicValidator.SQL_SELECT_PATTERN.sub('', value)
        value = BasicValidator.SQL_UPDATE_PATTERN.sub('', value)
        
        # Remove control characters
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        return value
    
    @staticmethod
    def validate_column_name(name: str) -> bool:
        """Validate if a column name is safe.
        
        Args:
            name: Column name to validate
            
        Returns:
            True if valid, False otherwise
        """
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', str(name)))
    
    @staticmethod
    def sanitize_column_name(name: str) -> str:
        """Sanitize a column name.
        
        Args:
            name: Column name to sanitize
            
        Returns:
            Sanitized column name
        """
        # Replace special characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        
        # Ensure name doesn't start with number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'col_' + clean_name
        
        # Ensure name is not empty
        if not clean_name or clean_name == '_':
            clean_name = 'column_0'
        
        return clean_name


class DataValidator:
    """Data validation utilities for ML workflows."""
    
    @staticmethod
    def validate_dataset_shape(df_shape: tuple, min_samples: int = 10, max_features: int = 1000) -> bool:
        """Validate dataset shape constraints.
        
        Args:
            df_shape: DataFrame shape (rows, columns)
            min_samples: Minimum required samples
            max_features: Maximum allowed features
            
        Returns:
            True if shape is valid
        """
        rows, cols = df_shape
        return rows >= min_samples and cols <= max_features
    
    @staticmethod
    def validate_column_exists(df_columns: List[str], column_name: str) -> bool:
        """Validate that a column exists in the dataset.
        
        Args:
            df_columns: List of column names
            column_name: Column name to check
            
        Returns:
            True if column exists
        """
        return column_name in df_columns
    
    @staticmethod
    def validate_numeric_column(series_dtype: str) -> bool:
        """Validate that a column is numeric.
        
        Args:
            series_dtype: Pandas dtype string
            
        Returns:
            True if numeric
        """
        return 'int' in str(series_dtype) or 'float' in str(series_dtype)
    
    @staticmethod
    def validate_no_all_null(series_null_count: int, series_length: int) -> bool:
        """Validate that a column is not entirely null.
        
        Args:
            series_null_count: Number of null values
            series_length: Total length of series
            
        Returns:
            True if not all null
        """
        return series_null_count < series_length