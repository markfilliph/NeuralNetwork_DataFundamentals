"""
Visualization Utilities for Data Analysis Platform

This module provides utility functions for creating consistent, professional
visualizations across the platform.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# Color Palettes
COLOR_PALETTES = {
    'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941'],
    'tech': ['#4ECDC4', '#44A08D', '#093637', '#F38BA8', '#FFB3C6'],
    'academic': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
    'corporate': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
    'monochrome': ['#2C3E50', '#34495E', '#7F8C8D', '#BDC3C7', '#ECF0F1']
}

# Chart Configurations
CHART_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'title_size': 16,
    'label_size': 10,
    'legend_size': 10
}


class ChartStyler:
    """Utility class for consistent chart styling."""
    
    @staticmethod
    def set_style(style: str = 'default', palette: str = 'business') -> None:
        """Set global matplotlib and seaborn style."""
        if style == 'presentation':
            plt.style.use('default')
            sns.set_palette(COLOR_PALETTES[palette])
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 20
            })
        elif style == 'report':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette(COLOR_PALETTES[palette])
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14
            })
        else:
            plt.style.use('default')
            sns.set_palette(COLOR_PALETTES[palette])
    
    @staticmethod
    def apply_chart_styling(ax, title: str = None, xlabel: str = None, 
                          ylabel: str = None, remove_spines: bool = True) -> None:
        """Apply consistent styling to matplotlib axes."""
        if title:
            ax.set_title(title, fontsize=CHART_CONFIG['title_size'], 
                        fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=CHART_CONFIG['label_size'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=CHART_CONFIG['label_size'])
        
        if remove_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=CHART_CONFIG['label_size'])


class QuickPlots:
    """Quick plotting functions for common chart types."""
    
    @staticmethod
    def distribution(data: pd.Series, bins: int = 30, 
                    title: str = None, color: str = None) -> plt.Figure:
        """Create a distribution plot with histogram and KDE."""
        fig, ax = plt.subplots(figsize=CHART_CONFIG['figure_size'])
        
        color = color or COLOR_PALETTES['business'][0]
        
        # Histogram
        ax.hist(data, bins=bins, alpha=0.7, color=color, 
               density=True, edgecolor='black', linewidth=0.5)
        
        # KDE
        data_clean = data.dropna()
        if len(data_clean) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data_clean)
            x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
            ax.plot(x_range, kde(x_range), color='darkred', linewidth=2)
        
        ChartStyler.apply_chart_styling(
            ax, title=title or f'Distribution of {data.name}',
            xlabel=data.name, ylabel='Density'
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def comparison_bar(data: pd.Series, title: str = None, 
                      horizontal: bool = False, color_palette: str = 'business') -> plt.Figure:
        """Create a comparison bar chart."""
        fig, ax = plt.subplots(figsize=CHART_CONFIG['figure_size'])
        
        colors = COLOR_PALETTES[color_palette]
        
        if horizontal:
            bars = ax.barh(data.index, data.values, color=colors[:len(data)])
            # Add value labels
            for i, (idx, val) in enumerate(data.items()):
                ax.text(val + max(data.values) * 0.01, i, f'{val:.1f}', 
                       va='center', fontsize=CHART_CONFIG['label_size'])
        else:
            bars = ax.bar(data.index, data.values, color=colors[:len(data)])
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(data.values) * 0.01,
                       f'{height:.1f}', ha='center', va='bottom',
                       fontsize=CHART_CONFIG['label_size'])
        
        ChartStyler.apply_chart_styling(
            ax, title=title or f'Comparison of {data.name}',
            xlabel=data.index.name if not horizontal else data.name,
            ylabel=data.name if not horizontal else data.index.name
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def correlation_heatmap(data: pd.DataFrame, title: str = None) -> plt.Figure:
        """Create a correlation heatmap."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax, fmt='.3f',
                   cbar_kws={'shrink': 0.8})
        
        ChartStyler.apply_chart_styling(
            ax, title=title or 'Feature Correlation Matrix',
            remove_spines=False
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def time_series(data: pd.Series, title: str = None, 
                   show_trend: bool = True, color: str = None) -> plt.Figure:
        """Create a time series plot."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        color = color or COLOR_PALETTES['business'][0]
        
        # Main time series
        ax.plot(data.index, data.values, color=color, linewidth=2, 
               marker='o', markersize=3, alpha=0.8)
        
        # Trend line
        if show_trend and len(data) > 1:
            from scipy.stats import linregress
            x_numeric = np.arange(len(data))
            slope, intercept, _, _, _ = linregress(x_numeric, data.values)
            trend_line = slope * x_numeric + intercept
            ax.plot(data.index, trend_line, '--', color='red', 
                   alpha=0.7, linewidth=1.5, label='Trend')
            ax.legend()
        
        ChartStyler.apply_chart_styling(
            ax, title=title or f'Time Series: {data.name}',
            xlabel='Date', ylabel=data.name
        )
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig


class InteractivePlots:
    """Interactive plotting functions using Plotly."""
    
    @staticmethod
    def scatter_interactive(df: pd.DataFrame, x: str, y: str, 
                          color: str = None, size: str = None,
                          hover_data: List[str] = None,
                          title: str = None) -> go.Figure:
        """Create an interactive scatter plot."""
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            hover_data=hover_data or [],
            title=title or f'{y} vs {x}',
            color_discrete_sequence=COLOR_PALETTES['business']
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            showlegend=True if color else False
        )
        
        return fig
    
    @staticmethod
    def time_series_interactive(df: pd.DataFrame, date_col: str, 
                              value_cols: List[str], title: str = None) -> go.Figure:
        """Create an interactive time series plot."""
        fig = go.Figure()
        
        colors = COLOR_PALETTES['business']
        
        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=title or 'Interactive Time Series',
            xaxis_title=date_col,
            yaxis_title='Value',
            hovermode='x unified',
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    @staticmethod
    def dashboard_subplots(data_dict: Dict[str, Any], 
                          subplot_titles: List[str] = None) -> go.Figure:
        """Create a multi-chart dashboard."""
        n_charts = len(data_dict)
        cols = 2
        rows = (n_charts + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles or list(data_dict.keys())
        )
        
        colors = COLOR_PALETTES['business']
        
        for i, (name, data) in enumerate(data_dict.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            if isinstance(data, pd.Series):
                if data.index.dtype == 'datetime64[ns]':
                    # Time series
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data.values,
                                  mode='lines+markers', name=name,
                                  line=dict(color=colors[i % len(colors)])),
                        row=row, col=col
                    )
                else:
                    # Bar chart
                    fig.add_trace(
                        go.Bar(x=data.index, y=data.values, name=name,
                              marker_color=colors[i % len(colors)]),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
            title_text=\"Multi-Chart Dashboard\",
            title_x=0.5,
            title_font_size=18
        )
        
        return fig


class ReportGenerator:
    """Generate standardized report visualizations."""
    
    @staticmethod
    def executive_summary(df: pd.DataFrame, metrics: Dict[str, str],
                         date_col: str = None) -> Dict[str, go.Figure]:
        """Generate executive summary charts."""
        charts = {}
        
        # KPI Cards (as text for now, could be enhanced)
        kpi_data = {}
        for metric, col in metrics.items():
            if col in df.columns:
                kpi_data[metric] = {
                    'value': df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].count(),
                    'change': np.random.uniform(-10, 15)  # Placeholder
                }
        
        # Trend Analysis
        if date_col and date_col in df.columns:
            monthly_data = df.groupby(pd.Grouper(key=date_col, freq='M')).agg(
                {col: 'sum' if df[col].dtype in ['int64', 'float64'] else 'count' 
                 for col in metrics.values() if col in df.columns}
            ).reset_index()
            
            charts['trends'] = InteractivePlots.time_series_interactive(
                monthly_data, date_col, list(metrics.values()),
                'Key Metrics Trends'
            )
        
        # Distribution Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            charts['correlations'] = go.Figure(data=go.Heatmap(
                z=df[numeric_cols].corr().values,
                x=numeric_cols,
                y=numeric_cols,
                colorscale='RdYlBu_r'
            ))
            charts['correlations'].update_layout(title='Feature Correlations')
        
        return charts
    
    @staticmethod
    def export_chart(fig, filename: str, format: str = 'png', 
                    width: int = 1200, height: int = 800) -> str:
        """Export chart to file."""
        import os
        
        export_dir = 'exports'
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, f"{filename}.{format}")
        
        if hasattr(fig, 'write_image'):  # Plotly figure
            fig.write_image(filepath, width=width, height=height)
        elif hasattr(fig, 'savefig'):  # Matplotlib figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            raise ValueError("Unsupported figure type for export")
        
        return filepath


# Convenience functions for quick access
def quick_histogram(data: pd.Series, **kwargs) -> plt.Figure:
    """Quick histogram with default styling."""
    return QuickPlots.distribution(data, **kwargs)

def quick_bar(data: pd.Series, **kwargs) -> plt.Figure:
    """Quick bar chart with default styling."""
    return QuickPlots.comparison_bar(data, **kwargs)

def quick_correlation(data: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick correlation heatmap."""
    return QuickPlots.correlation_heatmap(data, **kwargs)

def quick_timeseries(data: pd.Series, **kwargs) -> plt.Figure:
    """Quick time series plot."""
    return QuickPlots.time_series(data, **kwargs)

def interactive_scatter(df: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure:
    """Quick interactive scatter plot."""
    return InteractivePlots.scatter_interactive(df, x, y, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'revenue': np.random.normal(1000, 200, 100),
        'customers': np.random.poisson(50, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Test functions
    print("🧪 Testing visualization utilities...")
    
    # Set style
    ChartStyler.set_style('presentation', 'business')
    
    # Test distribution plot
    fig1 = quick_histogram(sample_data['revenue'], title='Revenue Distribution')
    print("✅ Distribution plot created")
    
    # Test bar chart
    region_summary = sample_data.groupby('region')['revenue'].sum()
    fig2 = quick_bar(region_summary, title='Revenue by Region')
    print("✅ Bar chart created")
    
    # Test correlation
    fig3 = quick_correlation(sample_data[['revenue', 'customers']])
    print("✅ Correlation heatmap created")
    
    # Test interactive scatter
    fig4 = interactive_scatter(sample_data, 'customers', 'revenue', 
                              color='region', title='Revenue vs Customers')
    print("✅ Interactive scatter plot created")
    
    print("🎯 All visualization utilities working correctly!")