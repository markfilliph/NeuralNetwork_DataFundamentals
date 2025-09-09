## 3. Backend.md

```markdown
# Backend Architecture Documentation

## 1. Overview

The backend architecture provides data processing, model training, and API services for the data analysis platform. It follows a modular, scalable design with clear separation of concerns.

## 2. Architecture Design

### 2.1 High-Level Architecture
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Layer    │────▶│  Service Layer  │────▶│    API Layer    │
│                 │     │                 │     │                 │
│ - File Storage  │     │ - Data Service  │     │ - REST API      │
│ - Cache         │     │ - Model Service │     │ - WebSocket     │
│ - Database      │     │ - Export Service│     │ - GraphQL       │
└─────────────────┘     └─────────────────┘     └─────────────────┘

### 2.2 Directory Structure
backend/
├── api/
│   ├── init.py
│   ├── routes.py
│   └── middleware.py
├── core/
│   ├── init.py
│   ├── config.py
│   ├── exceptions.py
│   └── logging.py
├── models/
│   ├── init.py
│   ├── regression.py
│   └── evaluation.py
├── services/
│   ├── init.py
│   ├── data_service.py
│   ├── model_service.py
│   └── export_service.py
├── utils/
│   ├── init.py
│   ├── validators.py
│   ├── transformers.py
│   └── file_handlers.py
├── tests/
│   └── [test files]
└── main.py

## 3. Core Components

### 3.1 Configuration Management
```python
# core/config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Data Analysis Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # File handling
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = ['.xlsx', '.xls', '.csv', '.mdb', '.accdb']
    UPLOAD_PATH: str = "./uploads"
    
    # Model settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    REDIS_URL: Optional[str] = None
    
    # Security
    SECRET_KEY: str
    API_KEY_EXPIRE_MINUTES: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
3.2 Data Service Layer
python# services/data_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pyodbc

class DataService:
    def __init__(self):
        self.cache = {}
        
    def load_excel(self, filepath: Path) -> pd.DataFrame:
        """Load data from Excel file with validation"""
        try:
            # Check file size
            if filepath.stat().st_size > settings.MAX_FILE_SIZE:
                raise ValueError(f"File size exceeds {settings.MAX_FILE_SIZE} bytes")
            
            # Load with pandas
            df = pd.read_excel(filepath, engine='openpyxl')
            
            # Basic validation
            self._validate_dataframe(df)
            
            # Cache the data
            self.cache[str(filepath)] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def load_access(self, filepath: Path, table_name: str) -> pd.DataFrame:
        """Load data from Access database"""
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            f'DBQ={filepath};'
        )
        
        try:
            conn = pyodbc.connect(conn_str)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            
            self._validate_dataframe(df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading Access file: {e}")
            raise
    
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       options: Dict) -> pd.DataFrame:
        """Preprocess data based on options"""
        df_processed = df.copy()
        
        # Handle missing values
        if options.get('handle_missing'):
            strategy = options.get('missing_strategy', 'drop')
            if strategy == 'drop':
                df_processed = df_processed.dropna()
            elif strategy == 'mean':
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                    df_processed[numeric_cols].mean()
                )
        
        # Encode categorical variables
        if options.get('encode_categorical'):
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_processed[f'{col}_encoded'] = pd.Categorical(
                    df_processed[col]
                ).codes
        
        # Scale features
        if options.get('scale_features'):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = scaler.fit_transform(
                df_processed[numeric_cols]
            )
        
        return df_processed
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data information"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict(),
            'memory_usage': df.memory_usage().sum()
        }
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate dataframe structure"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        if len(df.columns) == 0:
            raise ValueError("DataFrame has no columns")
        if len(df) > 1_000_000:
            logger.warning("Large dataset detected. Performance may be impacted.")
3.3 Model Service Layer
python# services/model_service.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime
import uuid

class ModelService:
    def __init__(self):
        self.models = {}
        
    def train_linear_regression(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Dict:
        """Train linear regression model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'mse': mean_squared_error(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'mse': mean_squared_error(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
        }
        
        # Store model
        model_id = str(uuid.uuid4())
        self.models[model_id] = {
            'model': model,
            'features': list(X.columns),
            'target': y.name,
            'metrics': metrics,
            'created_at': datetime.now(),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_
        }
        
        return {
            'model_id': model_id,
            'metrics': metrics,
            'coefficients': self.models[model_id]['coefficients'],
            'intercept': self.models[model_id]['intercept']
        }
    
    def predict(self, model_id: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stored model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        model = model_info['model']
        
        # Validate features
        expected_features = model_info['features']
        if list(X.columns) != expected_features:
            raise ValueError(f"Expected features: {expected_features}")
        
        return model.predict(X)
    
    def get_feature_importance(self, model_id: str) -> Dict:
        """Get feature importance from linear model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        coefficients = model_info['coefficients']
        
        # Sort by absolute value
        importance = sorted(
            coefficients.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return {
            'features': [x[0] for x in importance],
            'importance': [x[1] for x in importance],
            'absolute_importance': [abs(x[1]) for x in importance]
        }
    
    def save_model(self, model_id: str, filepath: Path):
        """Save model to disk"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        joblib.dump(model_info, filepath)
        
    def load_model(self, filepath: Path) -> str:
        """Load model from disk"""
        model_info = joblib.load(filepath)
        model_id = str(uuid.uuid4())
        self.models[model_id] = model_info
        return model_id
3.4 API Layer
python# api/routes.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import shutil

app = FastAPI(title="Data Analysis API", version="1.0.0")

# Pydantic models
class DataInfo(BaseModel):
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]

class TrainRequest(BaseModel):
    features: List[str]
    target: str
    test_size: float = 0.2
    preprocessing: Optional[Dict] = None

class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, float]]

# Initialize services
data_service = DataService()
model_service = ModelService()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process data file"""
    # Validate file extension
    if not any(file.filename.endswith(ext) for ext in settings.ALLOWED_EXTENSIONS):
        raise HTTPException(400, "Invalid file type")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)
    
    try:
        # Load data based on file type
        if file.filename.endswith(('.xlsx', '.xls')):
            df = data_service.load_excel(tmp_path)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        else:
            raise HTTPException(400, "Unsupported file type")
        
        # Get data info
        info = data_service.get_data_info(df)
        
        # Store in session/cache
        session_id = str(uuid.uuid4())
        app.state.sessions[session_id] = df
        
        return {
            "session_id": session_id,
            "info": info
        }
        
    finally:
        tmp_path.unlink()

@app.post("/train")
async def train_model(session_id: str, request: TrainRequest):
    """Train linear regression model"""
    # Get data from session
    if session_id not in app.state.sessions:
        raise HTTPException(404, "Session not found")
    
    df = app.state.sessions[session_id]
    
    # Preprocess if requested
    if request.preprocessing:
        df = data_service.preprocess_data(df, request.preprocessing)
    
    # Prepare features and target
    X = df[request.features]
    y = df[request.target]
    
    # Train model
    result = model_service.train_linear_regression(
        X, y, test_size=request.test_size
    )
    
    return result

@app.post("/predict")
async def predict(request: PredictRequest):
    """Make predictions"""
    # Convert input data to DataFrame
    X = pd.DataFrame(request.data)
    
    # Make predictions
    try:
        predictions = model_service.predict(request.model_id, X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/model/{model_id}/importance")
async def get_feature_importance(model_id: str):
    """Get feature importance"""
    try:
        importance = model_service.get_feature_importance(model_id)
        return importance
    except Exception as e:
        raise HTTPException(404, str(e))