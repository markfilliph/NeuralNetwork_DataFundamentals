"""Main FastAPI application with modular routes."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import auth, data, models
from backend.api.middleware import SecurityMiddleware, LoggingMiddleware
from backend.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Platform API",
    description="Secure data analysis platform with encryption and RBAC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(models.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Data Analysis Platform"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Data Analysis Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)