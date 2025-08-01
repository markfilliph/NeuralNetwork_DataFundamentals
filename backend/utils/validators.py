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