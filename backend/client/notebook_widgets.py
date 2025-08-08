"""
Interactive widgets for Jupyter notebooks integrated with DAPP backend.
Provides rich UI components for data analysis workflows.
"""

import io
import base64
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot

from .dapp_client import DAPPClient


class FileUploadWidget:
    """
    Interactive file upload widget with drag-and-drop support.
    Integrates with DAPP backend for file processing.
    """
    
    def __init__(self, client: DAPPClient, auto_process: bool = True):
        """Initialize file upload widget."""
        self.client = client
        self.auto_process = auto_process
        self.uploaded_files: Dict[str, str] = {}
        
        # Create widgets
        self.upload_widget = widgets.FileUpload(
            accept='.xlsx,.xls,.csv,.txt,.tsv',
            multiple=True,
            description='Upload Files',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.description_input = widgets.Text(
            placeholder='Optional: Describe your dataset',
            description='Description:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.process_button = widgets.Button(
            description='Process Files',
            button_style='primary',
            icon='upload',
            layout=widgets.Layout(width='150px')
        )
        
        self.output_area = widgets.Output()
        
        # Set up event handlers
        self.upload_widget.observe(self._on_upload, names='value')
        self.process_button.on_click(self._on_process)
        
        # Create layout
        self.widget = widgets.VBox([
            HTML('<h3>📁 File Upload</h3>'),
            HTML('<p>Supported formats: Excel (.xlsx, .xls), CSV (.csv), Text (.txt, .tsv)</p>'),
            self.upload_widget,
            self.description_input,
            self.process_button,
            self.output_area
        ])
    
    def _on_upload(self, change):
        """Handle file upload."""
        with self.output_area:
            clear_output(wait=True)
            if self.upload_widget.value:
                display(HTML(f"""
                <div style="color: blue; padding: 10px; border: 1px solid #2196F3; border-radius: 5px;">
                    📁 {len(self.upload_widget.value)} file(s) selected for upload
                </div>
                """))
    
    def _on_process(self, button):
        """Process uploaded files."""
        with self.output_area:
            clear_output(wait=True)
            
            if not self.upload_widget.value:
                display(HTML("""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Please select files to upload first
                </div>
                """))
                return
            
            try:
                display(HTML("🔄 Processing files..."))
                
                for filename, file_info in self.upload_widget.value.items():
                    # Save file temporarily
                    temp_path = Path(f"/tmp/{filename}")
                    with open(temp_path, 'wb') as f:
                        f.write(file_info['content'])
                    
                    # Upload to DAPP backend
                    dataset_id = self.client.upload_file(
                        temp_path, 
                        self.description_input.value or None,
                        show_progress=False
                    )
                    
                    self.uploaded_files[filename] = dataset_id
                    
                    # Clean up temp file
                    temp_path.unlink()
                
                display(HTML(f"""
                <div style="color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;">
                    ✅ Successfully processed {len(self.uploaded_files)} file(s)
                </div>
                """))
                
                # Show dataset IDs
                for filename, dataset_id in self.uploaded_files.items():
                    display(HTML(f"""
                    <div style="margin: 5px 0; padding: 5px; background-color: #f5f5f5;">
                        📊 <strong>{filename}</strong>: <code>{dataset_id}</code>
                    </div>
                    """))
                
            except Exception as e:
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Upload failed: {str(e)}
                </div>
                """))
    
    def display(self):
        """Display the widget."""
        display(self.widget)
    
    def get_dataset_ids(self) -> List[str]:
        """Get list of uploaded dataset IDs."""
        return list(self.uploaded_files.values())


class DataPreviewWidget:
    """
    Interactive data preview widget with filtering and statistics.
    """
    
    def __init__(self, client: DAPPClient):
        """Initialize data preview widget."""
        self.client = client
        self.current_df: Optional[pd.DataFrame] = None
        self.current_dataset_id: Optional[str] = None
        
        # Create widgets
        self.dataset_dropdown = widgets.Dropdown(
            description='Dataset:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.refresh_button = widgets.Button(
            description='Refresh',
            icon='refresh',
            layout=widgets.Layout(width='100px')
        )
        
        self.rows_slider = widgets.IntSlider(
            value=10,
            min=5,
            max=100,
            step=5,
            description='Rows:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='300px')
        )
        
        self.output_area = widgets.Output()
        
        # Set up event handlers
        self.dataset_dropdown.observe(self._on_dataset_change, names='value')
        self.refresh_button.on_click(self._refresh_datasets)
        self.rows_slider.observe(self._on_rows_change, names='value')
        
        # Create layout
        controls = widgets.HBox([self.dataset_dropdown, self.refresh_button])
        self.widget = widgets.VBox([
            HTML('<h3>👀 Data Preview</h3>'),
            controls,
            self.rows_slider,
            self.output_area
        ])
        
        # Load initial datasets
        self._refresh_datasets(None)
    
    def _refresh_datasets(self, button):
        """Refresh dataset list."""
        try:
            datasets = self.client.list_datasets()
            options = [(f"{d['filename']} ({d['id'][:8]}...)", d['id']) for d in datasets]
            self.dataset_dropdown.options = options
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load datasets: {str(e)}
                </div>
                """))
    
    def _on_dataset_change(self, change):
        """Handle dataset selection change."""
        if change['new']:
            self.current_dataset_id = change['new']
            self._load_and_display_data()
    
    def _on_rows_change(self, change):
        """Handle rows slider change."""
        if self.current_df is not None:
            self._display_data()
    
    def _load_and_display_data(self):
        """Load and display current dataset."""
        if not self.current_dataset_id:
            return
        
        try:
            self.current_df = self.client.get_processed_data(self.current_dataset_id)
            self._display_data()
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load data: {str(e)}
                </div>
                """))
    
    def _display_data(self):
        """Display current dataframe."""
        if self.current_df is None:
            return
        
        with self.output_area:
            clear_output(wait=True)
            
            # Show basic info
            display(HTML(f"""
            <div style="padding: 10px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 10px;">
                📊 <strong>Shape:</strong> {self.current_df.shape[0]:,} rows × {self.current_df.shape[1]} columns<br>
                💾 <strong>Memory usage:</strong> {self.current_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            </div>
            """))
            
            # Show data preview
            n_rows = self.rows_slider.value
            display(HTML(f"<h4>🔍 First {min(n_rows, len(self.current_df))} rows:</h4>"))
            display(self.current_df.head(n_rows))
            
            # Show data types
            display(HTML("<h4>📋 Column Information:</h4>"))
            info_df = pd.DataFrame({
                'Column': self.current_df.columns,
                'Data Type': self.current_df.dtypes.values,
                'Non-Null Count': [self.current_df[col].notna().sum() for col in self.current_df.columns],
                'Null Count': [self.current_df[col].isna().sum() for col in self.current_df.columns]
            })
            display(info_df)
    
    def display(self):
        """Display the widget."""
        display(self.widget)
    
    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get currently loaded dataframe."""
        return self.current_df


class ModelTrainingWidget:
    """
    Interactive widget for training linear regression models.
    """
    
    def __init__(self, client: DAPPClient):
        """Initialize model training widget."""
        self.client = client
        self.current_dataset_id: Optional[str] = None
        self.current_columns: List[str] = []
        
        # Create widgets
        self.dataset_dropdown = widgets.Dropdown(
            description='Dataset:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.target_dropdown = widgets.Dropdown(
            description='Target:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='300px')
        )
        
        self.features_select = widgets.SelectMultiple(
            description='Features:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        self.test_size_slider = widgets.FloatSlider(
            value=0.2,
            min=0.1,
            max=0.5,
            step=0.05,
            description='Test Size:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='300px')
        )
        
        self.train_button = widgets.Button(
            description='Train Model',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='150px')
        )
        
        self.output_area = widgets.Output()
        
        # Set up event handlers
        self.dataset_dropdown.observe(self._on_dataset_change, names='value')
        self.train_button.on_click(self._on_train)
        
        # Create layout
        self.widget = widgets.VBox([
            HTML('<h3>🤖 Model Training</h3>'),
            self.dataset_dropdown,
            HTML('<p><strong>Select target column and feature columns:</strong></p>'),
            widgets.HBox([self.target_dropdown, self.features_select]),
            self.test_size_slider,
            self.train_button,
            self.output_area
        ])
        
        # Load initial datasets
        self._refresh_datasets()
    
    def _refresh_datasets(self):
        """Refresh dataset list."""
        try:
            datasets = self.client.list_datasets()
            options = [(f"{d['filename']} ({d['id'][:8]}...)", d['id']) for d in datasets]
            self.dataset_dropdown.options = options
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load datasets: {str(e)}
                </div>
                """))
    
    def _on_dataset_change(self, change):
        """Handle dataset selection change."""
        if change['new']:
            self.current_dataset_id = change['new']
            self._load_columns()
    
    def _load_columns(self):
        """Load column information for current dataset."""
        if not self.current_dataset_id:
            return
        
        try:
            df = self.client.get_processed_data(self.current_dataset_id)
            
            # Filter numeric columns for target and features
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            all_columns = df.columns.tolist()
            
            self.target_dropdown.options = numeric_columns
            self.features_select.options = all_columns
            
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: blue; padding: 10px; border: 1px solid #2196F3; border-radius: 5px;">
                    📊 Dataset loaded: {len(all_columns)} columns, {len(numeric_columns)} numeric
                </div>
                """))
                
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load columns: {str(e)}
                </div>
                """))
    
    def _on_train(self, button):
        """Handle model training."""
        with self.output_area:
            clear_output(wait=True)
            
            if not self.current_dataset_id:
                display(HTML("""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Please select a dataset first
                </div>
                """))
                return
            
            if not self.target_dropdown.value:
                display(HTML("""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Please select a target column
                </div>
                """))
                return
            
            if not self.features_select.value:
                display(HTML("""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Please select at least one feature column
                </div>
                """))
                return
            
            try:
                display(HTML("🔄 Training model..."))
                
                model_id = self.client.train_linear_regression(
                    dataset_id=self.current_dataset_id,
                    target_column=self.target_dropdown.value,
                    feature_columns=list(self.features_select.value),
                    test_size=self.test_size_slider.value
                )
                
                display(HTML(f"""
                <div style="color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;">
                    ✅ Model trained successfully!<br>
                    Model ID: <code>{model_id}</code>
                </div>
                """))
                
                # Get and display results
                results = self.client.get_model_results(model_id, show_plots=True)
                
            except Exception as e:
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Training failed: {str(e)}
                </div>
                """))
    
    def display(self):
        """Display the widget."""
        display(self.widget)


class ResultsVisualizationWidget:
    """
    Interactive widget for visualizing model results and data insights.
    """
    
    def __init__(self, client: DAPPClient):
        """Initialize results visualization widget."""
        self.client = client
        
        # Create widgets
        self.model_dropdown = widgets.Dropdown(
            description='Model:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.refresh_button = widgets.Button(
            description='Refresh',
            icon='refresh',
            layout=widgets.Layout(width='100px')
        )
        
        self.plot_type_dropdown = widgets.Dropdown(
            options=[
                ('Predictions vs Actual', 'predictions'),
                ('Residuals Plot', 'residuals'),
                ('Feature Importance', 'importance'),
                ('Learning Curve', 'learning_curve')
            ],
            value='predictions',
            description='Plot Type:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='300px')
        )
        
        self.output_area = widgets.Output()
        
        # Set up event handlers
        self.model_dropdown.observe(self._on_model_change, names='value')
        self.refresh_button.on_click(self._refresh_models)
        self.plot_type_dropdown.observe(self._on_plot_type_change, names='value')
        
        # Create layout
        controls = widgets.HBox([self.model_dropdown, self.refresh_button])
        self.widget = widgets.VBox([
            HTML('<h3>📊 Results Visualization</h3>'),
            controls,
            self.plot_type_dropdown,
            self.output_area
        ])
        
        # Load initial models
        self._refresh_models(None)
    
    def _refresh_models(self, button):
        """Refresh model list."""
        try:
            models = self.client.list_models()
            options = [(f"Model {m['id'][:8]}... ({m['target_column']})", m['id']) for m in models]
            self.model_dropdown.options = options
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load models: {str(e)}
                </div>
                """))
    
    def _on_model_change(self, change):
        """Handle model selection change."""
        if change['new']:
            self._display_results()
    
    def _on_plot_type_change(self, change):
        """Handle plot type change."""
        if self.model_dropdown.value:
            self._display_results()
    
    def _display_results(self):
        """Display model results and visualizations."""
        if not self.model_dropdown.value:
            return
        
        with self.output_area:
            clear_output(wait=True)
            
            try:
                results = self.client.get_model_results(self.model_dropdown.value, show_plots=False)
                
                # Display metrics
                metrics = results.get('metrics', {})
                display(HTML(f"""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin: 10px 0;">
                    <h4>🎯 Model Performance</h4>
                    <div style="display: flex; justify-content: space-around;">
                        <div><strong>R² Score:</strong> {metrics.get('r2_score', 'N/A'):.4f}</div>
                        <div><strong>MSE:</strong> {metrics.get('mse', 'N/A'):.4f}</div>
                        <div><strong>MAE:</strong> {metrics.get('mae', 'N/A'):.4f}</div>
                    </div>
                </div>
                """))
                
                # Create visualization based on selected plot type
                plot_type = self.plot_type_dropdown.value
                if plot_type == 'predictions' and 'predictions' in results:
                    self._create_predictions_plot(results)
                elif plot_type == 'residuals' and 'predictions' in results:
                    self._create_residuals_plot(results)
                elif plot_type == 'importance' and 'feature_importance' in results:
                    self._create_importance_plot(results)
                
            except Exception as e:
                display(HTML(f"""
                <div style="color: red; padding: 10px; border: 1px solid #f44336; border-radius: 5px;">
                    ❌ Failed to load results: {str(e)}
                </div>
                """))
    
    def _create_predictions_plot(self, results):
        """Create predictions vs actual plot."""
        predictions = results.get('predictions', {})
        y_test = predictions.get('y_test', [])
        y_pred = predictions.get('y_pred', [])
        
        if y_test and y_pred:
            fig = go.Figure()
            
            # Perfect prediction line
            min_val, max_val = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            # Actual predictions
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=6, opacity=0.6)
            ))
            
            fig.update_layout(
                title='Predictions vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=500
            )
            
            fig.show()
    
    def _create_residuals_plot(self, results):
        """Create residuals plot."""
        predictions = results.get('predictions', {})
        y_test = predictions.get('y_test', [])
        y_pred = predictions.get('y_pred', [])
        
        if y_test and y_pred:
            residuals = [actual - pred for actual, pred in zip(y_test, y_pred)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, opacity=0.6)
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title='Residuals Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                height=500
            )
            
            fig.show()
    
    def _create_importance_plot(self, results):
        """Create feature importance plot."""
        importance = results.get('feature_importance', {})
        
        if importance:
            features = list(importance.keys())
            values = list(importance.values())
            
            fig = px.bar(
                x=values,
                y=features,
                orientation='h',
                title='Feature Importance',
                labels={'x': 'Importance', 'y': 'Features'}
            )
            
            fig.update_layout(height=500)
            fig.show()
    
    def display(self):
        """Display the widget."""
        display(self.widget)