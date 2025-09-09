"""
Database configuration and session management
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Generator
import uuid

from backend.core.config import settings

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Database Models
class UploadedFile(Base):
    """Model for uploaded files"""
    __tablename__ = "uploaded_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)  # csv, xlsx, pdf
    mime_type = Column(String)
    
    # Processing status
    status = Column(String, default="uploaded")  # uploaded, processing, processed, error
    error_message = Column(Text, nullable=True)
    
    # Metadata
    rows_count = Column(Integer, nullable=True)
    columns_count = Column(Integer, nullable=True)
    columns_info = Column(Text, nullable=True)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime, nullable=True)

class PredictionJob(Base):
    """Model for prediction jobs"""
    __tablename__ = "prediction_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    file_id = Column(String, nullable=False)  # Reference to uploaded file
    
    # Model configuration
    model_type = Column(String, default="neural_network")
    model_architecture = Column(String, default="simple")  # simple, deep, classification
    
    # Job status
    status = Column(String, default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    error_message = Column(Text, nullable=True)
    
    # Results
    predictions_count = Column(Integer, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    loss_value = Column(Float, nullable=True)
    model_metrics = Column(Text, nullable=True)  # JSON string
    results_file_path = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

class ModelInfo(Base):
    """Model for storing trained model information"""
    __tablename__ = "model_info"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    
    # Model file paths
    model_file_path = Column(String, nullable=False)
    weights_file_path = Column(String, nullable=True)
    scaler_file_path = Column(String, nullable=True)
    
    # Training info
    training_file_id = Column(String, nullable=False)
    epochs_trained = Column(Integer, nullable=False)
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    
    # Model metadata
    input_features = Column(Text, nullable=True)  # JSON string
    output_features = Column(Text, nullable=True)  # JSON string
    preprocessing_config = Column(Text, nullable=True)  # JSON string
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class UserSession(Base):
    """Model for user sessions (simple session management)"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_identifier = Column(String, nullable=True)  # IP or user ID
    
    # Session data
    uploaded_files = Column(Text, nullable=True)  # JSON string of file IDs
    active_jobs = Column(Text, nullable=True)  # JSON string of job IDs
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get a database session (for non-FastAPI usage)"""
    return SessionLocal()

# Database utility functions
def init_database():
    """Initialize database with tables and initial data"""
    create_tables()
    
    # Add any initial data here if needed
    db = get_db_session()
    try:
        # Check if we need to add default models or configurations
        pass
    finally:
        db.close()

def reset_database():
    """Reset database (drop and recreate all tables)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("🔄 Database reset successfully")

# Export main components
__all__ = [
    "engine",
    "SessionLocal", 
    "Base",
    "get_db",
    "get_db_session",
    "create_tables",
    "init_database",
    "reset_database",
    "UploadedFile",
    "PredictionJob", 
    "ModelInfo",
    "UserSession"
]