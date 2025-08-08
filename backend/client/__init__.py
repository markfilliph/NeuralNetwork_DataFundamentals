"""
FastAPI client library for Jupyter notebook integration.
Provides seamless API access for data scientists working in notebooks.
"""

from .dapp_client import DAPPClient, AsyncDAPPClient
from .auth_client import AuthClient
from .data_client import DataClient
from .model_client import ModelClient
from .notebook_widgets import (
    FileUploadWidget,
    DataPreviewWidget, 
    ModelTrainingWidget,
    ResultsVisualizationWidget
)

__all__ = [
    "DAPPClient",
    "AsyncDAPPClient", 
    "AuthClient",
    "DataClient",
    "ModelClient",
    "FileUploadWidget",
    "DataPreviewWidget",
    "ModelTrainingWidget", 
    "ResultsVisualizationWidget"
]