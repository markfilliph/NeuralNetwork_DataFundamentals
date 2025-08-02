"""Tests for data processing service."""

import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.data_service import DataService, DatasetInfo, DataAnalysis
from backend.core.exceptions import ValidationError, SecurityError


class TestDataService:
    """Test suite for DataService."""

    def setup_method(self):
        """Set up test environment."""
        self.data_service = DataService()
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': [25, 30, 35, 28, None],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })

    def test_analyze_dataframe(self):
        """Test DataFrame analysis."""
        dataset_id = "test-dataset-123"
        analysis = self.data_service.analyze_dataframe(self.test_data, dataset_id)
        
        assert isinstance(analysis, DataAnalysis)
        assert analysis.dataset_id == dataset_id
        assert analysis.shape == (5, 4)
        assert set(analysis.columns) == {'id', 'name', 'age', 'salary'}
        assert analysis.missing_values['name'] == 1
        assert analysis.missing_values['age'] == 1

    def test_get_data_types(self):
        """Test data type detection."""
        analysis = self.data_service.analyze_dataframe(self.test_data, "test")
        
        assert 'int64' in analysis.data_types['id']
        assert 'object' in analysis.data_types['name']
        assert 'float64' in analysis.data_types['age']  # Contains NaN

    def test_numeric_summary(self):
        """Test numeric column summary statistics."""
        analysis = self.data_service.analyze_dataframe(self.test_data, "test")
        
        salary_stats = analysis.numeric_summary['salary']
        assert salary_stats['mean'] == 60000.0
        assert salary_stats['min'] == 50000.0
        assert salary_stats['max'] == 70000.0

    def test_categorical_summary(self):
        """Test categorical column summary."""
        analysis = self.data_service.analyze_dataframe(self.test_data, "test")
        
        name_stats = analysis.categorical_summary['name']
        assert name_stats['unique_count'] == 4  # Including NaN
        assert name_stats['most_frequent'] in ['Alice', 'Bob', 'David', 'Eve']

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        analysis = self.data_service.analyze_dataframe(empty_df, "empty")
        
        assert analysis.shape == (0, 0)
        assert analysis.columns == []
        assert analysis.data_types == {}

    def test_single_column_dataframe(self):
        """Test single column DataFrame."""
        single_col_df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        analysis = self.data_service.analyze_dataframe(single_col_df, "single")
        
        assert analysis.shape == (5, 1)
        assert analysis.columns == ['values']
        assert 'values' in analysis.numeric_summary

    def test_all_null_column(self):
        """Test column with all null values."""
        null_df = pd.DataFrame({'all_null': [None, None, None]})
        analysis = self.data_service.analyze_dataframe(null_df, "null")
        
        assert analysis.missing_values['all_null'] == 3
        assert analysis.missing_percentages['all_null'] == 100.0