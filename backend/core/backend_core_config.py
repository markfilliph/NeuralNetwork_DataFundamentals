"""
Configuration settings for the application
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from decouple import config

class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    APP_NAME: str = "Neural Network Data Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = config("DEBUG", default=False, cast=bool)
    HOST: str = config("HOST", default="0.0.0.0")
    PORT: int = config("PORT", default=8000, cast=int)
    
    # Security
    SECRET_KEY: str = config("SECRET_KEY", default="your-super-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
    
    # Database
    DATABASE_URL: str = config("DATABASE_URL", default="sqlite:///./data/app.db")
    
    # File Upload Settings
    MAX_FILE_SIZE: int = config("MAX_FILE_SIZE", default=100 * 1024 * 1024, cast=int)  # 100MB
    UPLOAD_DIR: str = config("UPLOAD_DIR", default="data/uploads")
    PROCESSED_DIR: str = config("PROCESSED_DIR", default="data/processed")
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls", ".pdf"]
    
    # Neural Network Settings
    DEFAULT_MODEL_TYPE: str = config("DEFAULT_MODEL_TYPE", default="neural_network")
    MODEL_SAVE_DIR: str = config("MODEL_SAVE_DIR", default="models/saved")
    BATCH_SIZE: int = config("BATCH_SIZE", default=32, cast=int)
    EPOCHS: int = config("EPOCHS", default=100, cast=int)
    LEARNING_RATE: float = config("LEARNING_RATE", default=0.001, cast=float)
    
    # Logging
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    LOG_DIR: str = config("LOG_DIR", default="logs")
    
    # CORS Settings
    CORS_ORIGINS: list = config("CORS_ORIGINS", default="*", cast=lambda v: v.split(","))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.MODEL_SAVE_DIR,
            self.LOG_DIR,
            "data",
            "static"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

# File validation settings
class FileValidationSettings:
    """File validation configuration"""
    
    MAX_FILE_SIZE = settings.MAX_FILE_SIZE
    ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS
    
    # File type specific settings
    CSV_SETTINGS = {
        "encoding": ["utf-8", "latin-1", "cp1252"],
        "delimiters": [",", ";", "\t", "|"],
        "max_rows": 1000000,
    }
    
    EXCEL_SETTINGS = {
        "max_sheets": 10,
        "max_rows_per_sheet": 1000000,
    }
    
    PDF_SETTINGS = {
        "max_pages": 1000,
        "extract_tables": True,
        "extract_text": True,
    }

# Neural Network Model Configuration
class ModelConfig:
    """Neural network model configuration"""
    
    TENSORFLOW_SETTINGS = {
        "random_seed": 42,
        "validation_split": 0.2,
        "test_split": 0.1,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
    }
    
    ARCHITECTURES = {
        "simple": {
            "layers": [64, 32, 16],
            "activation": "relu",
            "dropout": 0.2,
            "output_activation": "linear"
        },
        "deep": {
            "layers": [128, 64, 32, 16, 8],
            "activation": "relu", 
            "dropout": 0.3,
            "output_activation": "linear"
        },
        "classification": {
            "layers": [64, 32, 16],
            "activation": "relu",
            "dropout": 0.2,
            "output_activation": "softmax"
        }
    }
    
    PREPROCESSING = {
        "scale_features": True,
        "handle_missing": "mean",  # mean, median, drop
        "encode_categorical": True,
        "remove_outliers": False,
    }

# Export configurations
model_config = ModelConfig()
file_validation = FileValidationSettings()