"""
Main API router that includes all route modules
"""
from fastapi import APIRouter

from backend.api.endpoints import data, model, health

# Create main API router
api_router = APIRouter()

# Include route modules
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health Check"]
)

api_router.include_router(
    data.router,
    prefix="/data",
    tags=["Data Management"]
)

api_router.include_router(
    model.router,
    prefix="/model",
    tags=["Machine Learning Models"]
)