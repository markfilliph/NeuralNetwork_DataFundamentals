"""
Main DAPP client for Jupyter notebook integration.
Provides high-level interface to all platform services.
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import httpx
import websockets
from IPython.display import display, HTML

from .auth_client import AuthClient
from .data_client import DataClient  
from .model_client import ModelClient


class DAPPClient:
    """
    Synchronous client for Data Analysis Platform integration in Jupyter notebooks.
    
    Example usage:
        client = DAPPClient("http://localhost:8000")
        client.login("user@example.com", "password")
        
        # Upload and process data
        dataset_id = client.upload_file("data.xlsx")
        df = client.get_processed_data(dataset_id)
        
        # Train model
        model_id = client.train_linear_regression(dataset_id, target_column="price")
        results = client.get_model_results(model_id)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize DAPP client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session: Optional[httpx.Client] = None
        self._token: Optional[str] = None
        
        # Initialize service clients
        self.auth = AuthClient(base_url, timeout)
        self.data = DataClient(base_url, timeout)
        self.model = ModelClient(base_url, timeout)
    
    @property
    def session(self) -> httpx.Client:
        """Get or create HTTP session."""
        if self._session is None:
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._session = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers
            )
        return self._session
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and store authentication token."""
        result = self.auth.login(username, password)
        self._token = result.get("access_token")
        
        # Update session headers
        if self._session:
            self._session.headers["Authorization"] = f"Bearer {self._token}"
        
        # Update service client tokens
        self.auth._token = self._token
        self.data._token = self._token  
        self.model._token = self._token
        
        display(HTML(f"""
        <div style="color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;">
            ✅ Successfully logged in as <strong>{username}</strong>
        </div>
        """))
        
        return result
    
    def upload_file(self, file_path: Union[str, Path], 
                   description: Optional[str] = None,
                   show_progress: bool = True) -> str:
        """
        Upload file and return dataset ID.
        
        Args:
            file_path: Path to file to upload
            description: Optional description for the dataset
            show_progress: Show upload progress in notebook
            
        Returns:
            Dataset ID for further operations
        """
        if show_progress:
            display(HTML("""
            <div style="padding: 10px;">
                📤 Uploading file... <div id="upload-progress"></div>
            </div>
            """))
        
        dataset_id = self.data.upload_file(file_path, description)
        
        if show_progress:
            display(HTML(f"""
            <div style="color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;">
                ✅ File uploaded successfully! Dataset ID: <code>{dataset_id}</code>
            </div>
            """))
        
        return dataset_id
    
    def get_processed_data(self, dataset_id: str) -> pd.DataFrame:
        """Get processed dataset as pandas DataFrame."""
        return self.data.get_processed_data(dataset_id)
    
    def preview_data(self, dataset_id: str, n_rows: int = 10) -> None:
        """Display data preview in notebook."""
        df = self.get_processed_data(dataset_id)
        
        display(HTML(f"""
        <h4>📊 Data Preview (showing {min(n_rows, len(df))} of {len(df)} rows)</h4>
        """))
        
        display(df.head(n_rows))
        
        # Show basic statistics
        display(HTML("<h4>📈 Basic Statistics</h4>"))
        display(df.describe())
    
    def train_linear_regression(self, dataset_id: str, target_column: str,
                              feature_columns: Optional[List[str]] = None,
                              test_size: float = 0.2) -> str:
        """Train linear regression model."""
        return self.model.train_linear_regression(
            dataset_id, target_column, feature_columns, test_size
        )
    
    def get_model_results(self, model_id: str, show_plots: bool = True) -> Dict[str, Any]:
        """Get model results with optional plot display."""
        results = self.model.get_model_results(model_id)
        
        if show_plots:
            self._display_model_results(results)
        
        return results
    
    def _display_model_results(self, results: Dict[str, Any]) -> None:
        """Display model results with plots in notebook."""
        metrics = results.get("metrics", {})
        
        display(HTML(f"""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin: 10px 0;">
            <h3>🎯 Model Performance</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f5f5f5;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>R² Score</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('r2_score', 'N/A'):.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Mean Squared Error</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('mse', 'N/A'):.4f}</td>
                </tr>
                <tr style="background-color: #f5f5f5;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Mean Absolute Error</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('mae', 'N/A'):.4f}</td>
                </tr>
            </table>
        </div>
        """))
        
        # Display plots if available
        if "plots" in results:
            for plot_name, plot_data in results["plots"].items():
                display(HTML(f"<h4>📈 {plot_name.replace('_', ' ').title()}</h4>"))
                # Here you would render the actual plots
                # This would integrate with your existing visualization utils
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all user datasets."""
        return self.data.list_datasets()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all user models."""
        return self.model.list_models()
    
    def health_check(self) -> Dict[str, str]:
        """Check API health status."""
        response = self.session.get("/health")
        response.raise_for_status()
        return response.json()
    
    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None


class AsyncDAPPClient:
    """
    Asynchronous version of DAPP client for advanced users.
    Supports concurrent operations and real-time WebSocket connections.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize async DAPP client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @property
    def session(self) -> httpx.AsyncClient:
        """Get or create async HTTP session."""
        if self._session is None:
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._session = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers
            )
        return self._session
    
    async def connect_realtime(self) -> None:
        """Connect to WebSocket for real-time updates."""
        if not self._token:
            raise ValueError("Must login before connecting to WebSocket")
        
        ws_url = self.base_url.replace('http', 'ws') + f"/ws?token={self._token}"
        self._websocket = await websockets.connect(ws_url)
    
    async def listen_updates(self, callback) -> None:
        """Listen for real-time updates."""
        if not self._websocket:
            await self.connect_realtime()
        
        async for message in self._websocket:
            data = json.loads(message)
            await callback(data)
    
    async def upload_multiple_files(self, file_paths: List[Union[str, Path]],
                                  descriptions: Optional[List[str]] = None) -> List[str]:
        """Upload multiple files concurrently."""
        tasks = []
        for i, file_path in enumerate(file_paths):
            desc = descriptions[i] if descriptions and i < len(descriptions) else None
            # This would use async file upload methods
            tasks.append(self._upload_file_async(file_path, desc))
        
        return await asyncio.gather(*tasks)
    
    async def _upload_file_async(self, file_path: Union[str, Path], 
                               description: Optional[str] = None) -> str:
        """Async file upload helper."""
        # Implementation would be similar to sync version but async
        pass
    
    async def close(self) -> None:
        """Close connections."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        if self._session:
            await self._session.aclose()
            self._session = None