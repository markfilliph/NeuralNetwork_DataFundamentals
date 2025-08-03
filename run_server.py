"""Simple server runner for testing the Data Analysis Platform."""

import os
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for development
os.environ["SECRET_KEY"] = "development-secret-key-for-testing-32chars"
os.environ["ENCRYPTION_KEY"] = "development-encryption-key-32chars"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all routes
auth, data, models = None, None, None

try:
    from backend.api.routes import auth
    print("✅ Auth routes imported successfully")
except Exception as e:
    print(f"❌ Error importing auth routes: {e}")

try:
    from backend.api.routes import data
    print("✅ Data routes imported successfully")
except Exception as e:
    print(f"❌ Error importing data routes: {e}")

try:
    from backend.api.routes import models
    print("✅ Model routes imported successfully")  
except Exception as e:
    print(f"❌ Error importing model routes: {e}")

# Create app
app = FastAPI(
    title="Data Analysis Platform API",
    description="Secure data analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes if available
if auth:
    app.include_router(auth.router)
if data:
    app.include_router(data.router)
if models:
    app.include_router(models.router)

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "Data Analysis Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Data Analysis Platform"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Data Analysis Platform API...")
    print("📊 API Documentation: http://localhost:8003/docs")
    print("🔐 Health Check: http://localhost:8003/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )