"""Tests for data sanitization utilities."""

import pytest
import pandas as pd
from hypothesis import given, strategies as st

from backend.utils.sanitizers import DataSanitizer
from backend.core.exceptions import DataSanitizationError


class TestDataSanitizer:
    """Test suite for DataSanitizer class."""
    
    def test_sanitize_dataframe_clean_data(self):
        """Test sanitization of clean DataFrame."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        result = DataSanitizer.sanitize_dataframe(df)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_sanitize_dataframe_formulas(self):
        """Test sanitization of Excel formulas."""
        df = pd.DataFrame({
            'values': ['=SUM(A1:A10)', '+B1*2', '-C1', '@D1', 'normal_value']
        })
        
        result = DataSanitizer.sanitize_dataframe(df)
        
        expected_values = ['SUM(A1:A10)', 'B1*2', 'C1', 'D1', 'normal_value']
        assert result['values'].tolist() == expected_values
    
    def test_sanitize_dataframe_scripts(self):
        """Test sanitization of script tags."""
        df = pd.DataFrame({
            'content': [
                '<script>alert("xss")</script>',
                'normal content',
                '<SCRIPT>malicious code</SCRIPT>'
            ]
        })
        
        result = DataSanitizer.sanitize_dataframe(df)
        
        expected_content = ['', 'normal content', '']
        assert result['content'].tolist() == expected_content
    
    def test_sanitize_dataframe_sql_injection(self):
        """Test sanitization of SQL injection attempts."""
        df = pd.DataFrame({
            'user_input': [
                "'; DROP TABLE users; --",
                "normal input",
                "SELECT * FROM passwords WHERE id=1",
                "UPDATE users SET admin=1"
            ]
        })
        
        result = DataSanitizer.sanitize_dataframe(df)
        
        # Check that dangerous SQL keywords are removed
        for value in result['user_input']:
            if pd.notna(value):
                assert 'DROP TABLE' not in value.upper()
                assert 'SELECT *' not in value.upper()
                assert 'UPDATE' not in value.upper() or 'users' not in value.lower()
    
    def test_sanitize_column_names_special_chars(self):
        """Test sanitization of column names with special characters."""
        df = pd.DataFrame({
            'col@1': [1, 2],
            'col 2': [3, 4],
            'col-3': [5, 6],
            'col.4': [7, 8]
        })
        
        result = DataSanitizer.sanitize_column_names(df)
        
        expected_columns = ['col_1', 'col_2', 'col_3', 'col_4']
        assert list(result.columns) == expected_columns
    
    def test_sanitize_column_names_numeric_start(self):
        """Test sanitization of column names starting with numbers."""
        df = pd.DataFrame({
            '1col': [1, 2],
            '2nd_column': [3, 4]
        })
        
        result = DataSanitizer.sanitize_column_names(df)
        
        expected_columns = ['col_1col', 'col_2nd_column']
        assert list(result.columns) == expected_columns
    
    def test_sanitize_column_names_empty(self):
        """Test sanitization of empty column names."""
        df = pd.DataFrame({
            '': [1, 2],
            '_': [3, 4],
            'valid': [5, 6]
        })
        
        result = DataSanitizer.sanitize_column_names(df)
        
        expected_columns = ['column_0', 'column_1', 'valid']
        assert list(result.columns) == expected_columns
    
    def test_sanitize_column_names_duplicates(self):
        """Test handling of duplicate column names."""
        df = pd.DataFrame({
            'col': [1, 2],
            'col@': [3, 4],
            'col#': [5, 6]
        })
        
        result = DataSanitizer.sanitize_column_names(df)
        
        expected_columns = ['col', 'col_1', 'col_2']
        assert list(result.columns) == expected_columns
    
    def test_sanitize_cell_value_long_string(self):
        """Test sanitization of overly long strings."""
        long_string = "x" * 15000
        
        result = DataSanitizer._sanitize_cell_value(long_string)
        
        assert len(result) == 10000
    
    def test_sanitize_cell_value_control_chars(self):
        """Test sanitization of control characters."""
        value_with_nulls = "test\x00\x01\x02content"
        
        result = DataSanitizer._sanitize_cell_value(value_with_nulls)
        
        assert result == "testcontent"
        assert '\x00' not in result
    
    def test_validate_sanitization_clean_data(self):
        """Test validation of properly sanitized data."""
        df = pd.DataFrame({
            'clean_col': ['safe', 'data', 'here']
        })
        
        issues = DataSanitizer.validate_sanitization(df)
        
        assert len(issues) == 0
    
    def test_validate_sanitization_formulas(self):
        """Test validation detects remaining formulas."""
        df = pd.DataFrame({
            'formula_col': ['=SUM(A1:A10)', 'normal_value']
        })
        
        issues = DataSanitizer.validate_sanitization(df)
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'formula_detected'
    
    def test_validate_sanitization_invalid_columns(self):
        """Test validation detects invalid column names."""
        df = pd.DataFrame({
            'col@1': [1, 2],
            'valid_col': [3, 4]
        })
        
        issues = DataSanitizer.validate_sanitization(df)
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'invalid_column_name'
    
    @given(
        n_rows=st.integers(min_value=1, max_value=100),
        n_cols=st.integers(min_value=1, max_value=10)
    )
    def test_sanitize_dataframe_property_based(self, n_rows, n_cols):
        """Property-based test: sanitized DataFrames should have same shape."""
        # Generate random data
        data = {}
        for i in range(n_cols):
            data[f'col_{i}'] = ['test_value'] * n_rows
        
        df = pd.DataFrame(data)
        result = DataSanitizer.sanitize_dataframe(df)
        
        # Properties that should hold
        assert result.shape == df.shape
        assert len(result.columns) == len(df.columns)
        assert not result.empty
    
    def test_count_sanitized_cells(self):
        """Test counting of sanitized cells."""
        original_df = pd.DataFrame({
            'col1': ['=formula', 'normal', '<script>'],
            'col2': [1, 2, 3]  # Numeric column should not be counted
        })
        
        sanitized_df = pd.DataFrame({
            'col1': ['formula', 'normal', ''],
            'col2': [1, 2, 3]
        })
        
        count = DataSanitizer._count_sanitized_cells(original_df, sanitized_df)
        
        # Should count 2 sanitized cells in col1
        assert count == 2