"""
File processing utilities for different file types
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import PyPDF2
import pdfplumber
import openpyxl
from io import StringIO, BytesIO
import chardet
import json
import re

from backend.core.config import settings, file_validation
from backend.core.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass

class CSVHandler:
    """Handler for CSV file processing"""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    @staticmethod
    def detect_delimiter(file_path: str, encoding: str = 'utf-8') -> str:
        """Detect CSV delimiter"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                sample = file.read(1024)
                
            for delimiter in file_validation.CSV_SETTINGS["delimiters"]:
                if sample.count(delimiter) > 0:
                    # Check if this delimiter gives consistent column counts
                    lines = sample.split('\n')[:5]  # Check first 5 lines
                    counts = [line.count(delimiter) for line in lines if line.strip()]
                    if len(set(counts)) == 1:  # All lines have same delimiter count
                        return delimiter
            
            return ','  # Default fallback
        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}, using comma")
            return ','
    
    @classmethod
    def process_file(cls, file_path: str) -> Dict[str, Any]:
        """Process CSV file and return metadata and DataFrame"""
        with PerformanceLogger(f"CSV processing: {Path(file_path).name}"):
            try:
                # Detect encoding and delimiter
                encoding = cls.detect_encoding(file_path)
                delimiter = cls.detect_delimiter(file_path, encoding)
                
                logger.info(f"Processing CSV with encoding: {encoding}, delimiter: '{delimiter}'")
                
                # Read CSV
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    low_memory=False
                )
                
                # Validate size
                if len(df) > file_validation.CSV_SETTINGS["max_rows"]:
                    raise FileProcessingError(f"CSV file too large: {len(df)} rows (max: {file_validation.CSV_SETTINGS['max_rows']})")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Generate metadata
                metadata = cls._generate_metadata(df, encoding, delimiter)
                
                # Save processed file
                processed_path = cls._save_processed_file(df, file_path)
                
                return {
                    "dataframe": df,
                    "metadata": metadata,
                    "processed_path": processed_path,
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"CSV processing failed: {e}")
                return {
                    "dataframe": None,
                    "metadata": None,
                    "processed_path": None,
                    "status": "error",
                    "error": str(e)
                }
    
    @staticmethod
    def _generate_metadata(df: pd.DataFrame, encoding: str, delimiter: str) -> Dict[str, Any]:
        """Generate metadata for CSV file"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "encoding": encoding,
            "delimiter": delimiter,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "sample_data": df.head(3).to_dict('records')
        }
    
    @staticmethod
    def _save_processed_file(df: pd.DataFrame, original_path: str) -> str:
        """Save processed DataFrame"""
        processed_dir = Path(settings.PROCESSED_DIR)
        processed_dir.mkdir(exist_ok=True)
        
        filename = Path(original_path).stem
        processed_path = processed_dir / f"{filename}_processed.csv"
        
        df.to_csv(processed_path, index=False)
        return str(processed_path)

class ExcelHandler:
    """Handler for Excel file processing"""
    
    @classmethod
    def process_file(cls, file_path: str) -> Dict[str, Any]:
        """Process Excel file and return metadata and DataFrames"""
        with PerformanceLogger(f"Excel processing: {Path(file_path).name}"):
            try:
                # Read Excel file
                excel_file = pd.ExcelFile(file_path)
                
                if len(excel_file.sheet_names) > file_validation.EXCEL_SETTINGS["max_sheets"]:
                    raise FileProcessingError(f"Too many sheets: {len(excel_file.sheet_names)} (max: {file_validation.EXCEL_SETTINGS['max_sheets']})")
                
                sheets_data = {}
                combined_df = None
                
                for sheet_name in excel_file.sheet_names:
                    logger.info(f"Processing sheet: {sheet_name}")
                    
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if len(df) > file_validation.EXCEL_SETTINGS["max_rows_per_sheet"]:
                        logger.warning(f"Sheet {sheet_name} truncated to {file_validation.EXCEL_SETTINGS['max_rows_per_sheet']} rows")
                        df = df.head(file_validation.EXCEL_SETTINGS["max_rows_per_sheet"])
                    
                    # Clean column names
                    df.columns = df.columns.astype(str).str.strip()
                    
                    sheets_data[sheet_name] = {
                        "dataframe": df,
                        "metadata": cls._generate_sheet_metadata(df, sheet_name)
                    }
                    
                    # Combine sheets if multiple
                    if combined_df is None:
                        combined_df = df.copy()
                    else:
                        # Try to concatenate if columns match
                        if list(combined_df.columns) == list(df.columns):
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                # Use first sheet if only one, otherwise use combined
                main_df = combined_df if len(sheets_data) > 1 else list(sheets_data.values())[0]["dataframe"]
                
                # Generate overall metadata
                metadata = cls._generate_metadata(sheets_data, main_df)
                
                # Save processed file
                processed_path = cls._save_processed_file(main_df, file_path)
                
                return {
                    "dataframe": main_df,
                    "sheets_data": sheets_data,
                    "metadata": metadata,
                    "processed_path": processed_path,
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Excel processing failed: {e}")
                return {
                    "dataframe": None,
                    "sheets_data": None,
                    "metadata": None,
                    "processed_path": None,
                    "status": "error",
                    "error": str(e)
                }
    
    @staticmethod
    def _generate_sheet_metadata(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Generate metadata for a single sheet"""
        return {
            "sheet_name": sheet_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
    
    @staticmethod
    def _generate_metadata(sheets_data: Dict[str, Any], main_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall metadata for Excel file"""
        return {
            "total_sheets": len(sheets_data),
            "sheet_names": list(sheets_data.keys()),
            "main_dataframe": {
                "rows": len(main_df),
                "columns": len(main_df.columns),
                "column_names": main_df.columns.tolist(),
                "column_types": main_df.dtypes.astype(str).to_dict(),
                "missing_values": main_df.isnull().sum().to_dict(),
                "numeric_columns": main_df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": main_df.select_dtypes(include=['object']).columns.tolist(),
            },
            "sheets_metadata": {name: data["metadata"] for name, data in sheets_data.items()}
        }
    
    @staticmethod
    def _save_processed_file(df: pd.DataFrame, original_path: str) -> str:
        """Save processed DataFrame"""
        processed_dir = Path(settings.PROCESSED_DIR)
        processed_dir.mkdir(exist_ok=True)
        
        filename = Path(original_path).stem
        processed_path = processed_dir / f"{filename}_processed.csv"
        
        df.to_csv(processed_path, index=False)
        return str(processed_path)

class PDFHandler:
    """Handler for PDF file processing"""
    
    @classmethod
    def process_file(cls, file_path: str) -> Dict[str, Any]:
        """Process PDF file and extract data"""
        with PerformanceLogger(f"PDF processing: {Path(file_path).name}"):
            try:
                # Try to extract tables first
                tables_data = cls._extract_tables(file_path)
                
                if tables_data and tables_data["dataframes"]:
                    # Use table data if found
                    main_df = tables_data["dataframes"][0]  # Use first table
                    metadata = cls._generate_table_metadata(tables_data)
                    
                    # Save processed file
                    processed_path = cls._save_processed_file(main_df, file_path)
                    
                    return {
                        "dataframe": main_df,
                        "metadata": metadata,
                        "processed_path": processed_path,
                        "extraction_type": "tables",
                        "status": "success"
                    }
                else:
                    # Fallback to text extraction
                    text_data = cls._extract_text(file_path)
                    
                    return {
                        "dataframe": None,
                        "metadata": text_data["metadata"],
                        "processed_path": None,
                        "extraction_type": "text",
                        "text_content": text_data["text"],
                        "status": "partial",
                        "message": "No tables found, extracted text only"
                    }
                    
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return {
                    "dataframe": None,
                    "metadata": None,
                    "processed_path": None,
                    "status": "error",
                    "error": str(e)
                }
    
    @staticmethod
    def _extract_tables(file_path: str) -> Dict[str, Any]:
        """Extract tables from PDF using pdfplumber"""
        try:
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) > file_validation.PDF_SETTINGS["max_pages"]:
                    logger.warning(f"PDF truncated to {file_validation.PDF_SETTINGS['max_pages']} pages")
                    pages_to_process = pdf.pages[:file_validation.PDF_SETTINGS["max_pages"]]
                else:
                    pages_to_process = pdf.pages
                
                all_tables = []
                dataframes = []
                
                for page_num, page in enumerate(pages_to_process):
                    tables = page.extract_tables()
                    
                    for table_num, table in enumerate(tables):
                        if table and len(table) > 1:  # Has header and data
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # Clean and process
                            df = df.dropna(how='all')  # Remove empty rows
                            df.columns = df.columns.astype(str).str.strip()
                            
                            # Try to convert numeric columns
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            
                            dataframes.append(df)
                            all_tables.append({
                                "page": page_num + 1,
                                "table": table_num + 1,
                                "rows": len(df),
                                "columns": len(df.columns)
                            })
                
                return {
                    "dataframes": dataframes,
                    "tables_info": all_tables,
                    "total_pages": len(pdf.pages),
                    "processed_pages": len(pages_to_process)
                }
                
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            return {"dataframes": [], "tables_info": []}
    
    @staticmethod
    def _extract_text(file_path: str) -> Dict[str, Any]:
        """Extract text from PDF"""
        try:
            text_content = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                pages_to_process = min(len(pdf_reader.pages), file_validation.PDF_SETTINGS["max_pages"])
                
                for page_num in range(pages_to_process):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n\n"
            
            # Basic text analysis
            lines = text_content.split('\n')
            words = text_content.split()
            
            return {
                "text": text_content,
                "metadata": {
                    "total_pages": len(pdf_reader.pages),
                    "processed_pages": pages_to_process,
                    "total_lines": len(lines),
                    "total_words": len(words),
                    "character_count": len(text_content),
                    "non_empty_lines": len([line for line in lines if line.strip()])
                }
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "text": "",
                "metadata": {"error": str(e)}
            }
    
    @staticmethod
    def _generate_table_metadata(tables_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for PDF tables"""
        dataframes = tables_data["dataframes"]
        
        if not dataframes:
            return {"tables_found": 0}
        
        main_df = dataframes[0]
        
        return {
            "tables_found": len(dataframes),
            "main_table": {
                "rows": len(main_df),
                "columns": len(main_df.columns),
                "column_names": main_df.columns.tolist(),
                "column_types": main_df.dtypes.astype(str).to_dict(),
                "missing_values": main_df.isnull().sum().to_dict(),
                "sample_data": main_df.head(3).to_dict('records')
            },
            "extraction_info": tables_data["tables_info"],
            "total_pages": tables_data["total_pages"],
            "processed_pages": tables_data["processed_pages"]
        }
    
    @staticmethod
    def _save_processed_file(df: pd.DataFrame, original_path: str) -> str:
        """Save processed DataFrame from PDF"""
        processed_dir = Path(settings.PROCESSED_DIR)
        processed_dir.mkdir(exist_ok=True)
        
        filename = Path(original_path).stem
        processed_path = processed_dir / f"{filename}_processed.csv"
        
        df.to_csv(processed_path, index=False)
        return str(processed_path)

# File processor factory
class FileProcessor:
    """Main file processor that handles different file types"""
    
    handlers = {
        '.csv': CSVHandler,
        '.xlsx': ExcelHandler,
        '.xls': ExcelHandler,
        '.pdf': PDFHandler
    }
    
    @classmethod
    def process_file(cls, file_path: str) -> Dict[str, Any]:
        """Process file based on its extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in cls.handlers:
            raise FileProcessingError(f"Unsupported file type: {file_ext}")
        
        handler = cls.handlers[file_ext]
        return handler.process_file(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions"""
        return list(cls.handlers.keys())

# Export main components
__all__ = [
    "FileProcessor",
    "CSVHandler",
    "ExcelHandler", 
    "PDFHandler",
    "FileProcessingError"
]