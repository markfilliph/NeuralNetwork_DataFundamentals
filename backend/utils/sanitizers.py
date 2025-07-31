"""Data sanitization utilities."""

import re
import pandas as pd
from typing import Dict, Any, List
import logging

from backend.core.exceptions import DataSanitizationError

logger = logging.getLogger(__name__)


class DataSanitizer:
    """Sanitizes data to prevent security vulnerabilities."""
    
    # Patterns for dangerous content
    FORMULA_PATTERN = re.compile(r'^[=+\-@]')
    SCRIPT_PATTERN = re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL)
    SQL_INJECTION_PATTERN = re.compile(
        r'(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|UNION|SELECT)\s+(TABLE|DATABASE|INTO|FROM)',
        re.IGNORECASE
    )
    COMMENT_PATTERN = re.compile(r'--.*$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize a pandas DataFrame.
        
        Args:
            df: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame
            
        Raises:
            DataSanitizationError: If sanitization fails
        """
        try:
            df_clean = df.copy()
            
            # Sanitize string columns
            string_columns = df_clean.select_dtypes(include=['object']).columns
            
            for col in string_columns:
                df_clean[col] = df_clean[col].apply(
                    lambda x: DataSanitizer._sanitize_cell_value(x) if pd.notna(x) else x
                )
            
            # Sanitize column names
            df_clean = DataSanitizer.sanitize_column_names(df_clean)
            
            # Log sanitization results
            original_cells = len(df) * len(df.columns)
            sanitized_cells = DataSanitizer._count_sanitized_cells(df, df_clean)
            
            if sanitized_cells > 0:
                logger.warning(
                    f"Sanitized {sanitized_cells} cells out of {original_cells} total cells"
                )
            
            return df_clean
            
        except Exception as e:
            raise DataSanitizationError(f"Failed to sanitize DataFrame: {e}")
    
    @staticmethod
    def _sanitize_cell_value(value: Any) -> str:
        """Sanitize a single cell value.
        
        Args:
            value: Cell value to sanitize
            
        Returns:
            Sanitized string value
        """
        if not isinstance(value, str):
            return str(value)
        
        original_value = value
        
        # Remove potentially dangerous formulas
        if DataSanitizer.FORMULA_PATTERN.match(value):
            value = value.lstrip('=+-@')
            logger.debug(f"Removed formula prefix from: {original_value}")
        
        # Remove script tags
        value = DataSanitizer.SCRIPT_PATTERN.sub('', value)
        
        # Remove SQL injection attempts
        value = DataSanitizer.SQL_INJECTION_PATTERN.sub('', value)
        
        # Remove SQL comments
        value = DataSanitizer.COMMENT_PATTERN.sub('', value)
        
        # Remove null bytes and other control characters
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Limit string length to prevent DoS
        max_length = 10000
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"Truncated long string to {max_length} characters")
        
        return value
    
    @staticmethod
    def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame column names.
        
        Args:
            df: DataFrame with potentially unsafe column names
            
        Returns:
            DataFrame with sanitized column names
        """
        df_clean = df.copy()
        
        # Replace special characters with underscores
        clean_columns = []
        for col in df.columns:
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            
            # Ensure column name doesn't start with number
            if clean_col and clean_col[0].isdigit():
                clean_col = 'col_' + clean_col
            
            # Ensure column name is not empty
            if not clean_col or clean_col == '_':
                clean_col = f'column_{len(clean_columns)}'
            
            # Handle duplicate column names
            original_clean_col = clean_col
            counter = 1
            while clean_col in clean_columns:
                clean_col = f"{original_clean_col}_{counter}"
                counter += 1
            
            clean_columns.append(clean_col)
        
        df_clean.columns = clean_columns
        return df_clean
    
    @staticmethod
    def _count_sanitized_cells(original_df: pd.DataFrame, sanitized_df: pd.DataFrame) -> int:
        """Count the number of cells that were modified during sanitization.
        
        Args:
            original_df: Original DataFrame
            sanitized_df: Sanitized DataFrame
            
        Returns:
            Number of cells that were modified
        """
        try:
            # Compare string columns only
            string_columns = original_df.select_dtypes(include=['object']).columns
            sanitized_count = 0
            
            for col in string_columns:
                if col in sanitized_df.columns:
                    # Convert to string for comparison
                    orig_series = original_df[col].astype(str)
                    sanitized_series = sanitized_df[col].astype(str)
                    
                    # Count differences
                    sanitized_count += (orig_series != sanitized_series).sum()
            
            return sanitized_count
        except Exception:
            # If comparison fails, return 0
            return 0
    
    @staticmethod
    def validate_sanitization(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that DataFrame has been properly sanitized.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check column names
        for i, col in enumerate(df.columns):
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', str(col)):
                issues.append({
                    'type': 'invalid_column_name',
                    'column_index': i,
                    'column_name': col,
                    'message': f'Column name "{col}" contains invalid characters'
                })
        
        # Check for dangerous patterns in string data
        string_columns = df.select_dtypes(include=['object']).columns
        
        for col in string_columns:
            for idx, value in df[col].items():
                if pd.isna(value):
                    continue
                    
                value_str = str(value)
                
                # Check for formulas
                if DataSanitizer.FORMULA_PATTERN.match(value_str):
                    issues.append({
                        'type': 'formula_detected',
                        'column': col,
                        'row': idx,
                        'value': value_str[:100],  # Truncate for logging
                        'message': 'Formula detected in cell'
                    })
                
                # Check for scripts
                if DataSanitizer.SCRIPT_PATTERN.search(value_str):
                    issues.append({
                        'type': 'script_detected',
                        'column': col,
                        'row': idx,
                        'message': 'Script tag detected in cell'
                    })
                
                # Check for SQL injection patterns
                if DataSanitizer.SQL_INJECTION_PATTERN.search(value_str):
                    issues.append({
                        'type': 'sql_injection_detected',
                        'column': col,
                        'row': idx,
                        'message': 'Potential SQL injection pattern detected'
                    })
        
        return issues