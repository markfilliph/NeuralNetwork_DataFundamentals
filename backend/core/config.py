"""Configuration settings for the data analysis platform."""

import os
from typing import List, Optional


class Settings:
    """Application settings."""
    
    def __init__(self):
        # Application settings
        self.APP_NAME: str = "Data Analysis Platform"
        self.VERSION: str = "1.0.0"
        self.DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
        
        # File handling
        self.MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
        self.ALLOWED_EXTENSIONS: List[str] = ['.xlsx', '.xls', '.csv', '.mdb', '.accdb', '.txt', '.tsv']
        self.ALLOWED_MIMETYPES: List[str] = [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/csv',
            'application/x-msaccess',
            'text/plain',
            'text/tab-separated-values'
        ]
        self.UPLOAD_PATH: str = "./data/uploads"
        self.PROCESSED_PATH: str = "./data/processed"
        
        # Model settings
        self.RANDOM_STATE: int = 42
        self.TEST_SIZE: float = 0.2
        
        # Cache settings
        self.CACHE_TTL: int = 3600  # 1 hour
        self.REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
        
        # Security
        self.SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.API_KEY_EXPIRE_MINUTES: int = 60
        self.ENCRYPTION_KEY: Optional[str] = os.getenv("ENCRYPTION_KEY")


settings = Settings()