"""
Comprehensive Visualization Service for creating charts and plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import io
import base64
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from app.core.config import settings
from app.core.exceptions import VisualizationError

logger = logging.getLogger(__name__)


class ComprehensiveVisualizationService:
    """Comprehensive service for creating visualizations"""
    
    def __init__(self):
        self.settings = settings
        self.visualizations_dir = Path(settings.VISUALIZATIONS_DIR)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Set figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
    async def create_comprehensive_eda_plots(
        self, 
        df: pd.DataFrame, 
        session_id: str,
        target_column: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create comprehensive EDA plots for dataset
        
        Args:
            df: Input dataframe
            session_id: Session identifier
            target_column: Target column name
            
        Returns:
            Dictionary of plot types and their file paths
        """
        try:
            plots = {}
            
            # Data overview plots
            plots['data_overview'] = await self._create_data_overview_plot(df, session_id)
            plots['missing_values'] = await self._create_missing_values_plot(df, session_id)
            plots['correlation_matrix'] = await self._create_correlation_matrix(df, session_id)
            
            # Distribution plots
            plots['histograms'] = await self._create_histograms(df, session_id)
            plots['box_plots'] = await self._create_box_plots(df, session_id)
            plots['violin_plots'] = await self._create_violin_plots(df, session_id)
            
            # Categorical plots
            plots['count_plots'] = await self._create_count_plots(df, session_id)
            plots['bar_charts'] = await self._create_bar_charts(df, session_id)
            
            # Relationship plots
            plots['scatter_plots'] = await self._create_scatter_plots(df, session_id, target_column)
            plots['pair_plots'] = await self._create_pair_plots(df, session_id, target_column)
            
            # Statistical plots
            plots['qq_plots'] = await self._create_qq_plots(df, session_id)
            plots['outlier_detection'] = await self._create_outlier_detection_plots(df, session_id)
            
            # Target-specific plots
            if target_column and target_column in df.columns:
                plots['target_distribution'] = await self._create_target_distribution_plot(df, target_column, session_id)
                plots['target_relationships'] = await self._create_target_relationship_plots(df, target_column, session_id)
            
            logger.info(f"Created {len(plots)} comprehensive EDA plots")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating comprehensive EDA plots: {str(e)}")
            raise VisualizationError(f"Failed to create EDA plots: {str(e)}")
    
    async def _create_data_overview_plot(self, df: pd.DataFrame, session_id: str) -> str:
        """Create data overview visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Data Types Distribution')
            
            # Dataset shape info
            ax2.bar(['Rows', 'Columns'], [df.shape[0], df.shape[1]], color=['skyblue', 'lightcoral'])
            ax2.set_title('Dataset Shape')
            for i, v in enumerate([df.shape[0], df.shape[1]]):
                ax2.text(i, v + max(df.shape) * 0.01, str(v), ha='center', va='bottom')
            
            # Missing values overview
            missing_counts = df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0].head(10)
            if len(missing_counts) > 0:
                ax3.barh(range(len(missing_counts)), missing_counts.values)
                ax3.set_yticks(range(len(missing_counts)))
                ax3.set_yticklabels(missing_counts.index)
                ax3.set_xlabel('Missing Values Count')
                ax3.set_title('Top 10 Columns with Missing Values')
            else:
                ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Missing Values Overview')
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            ax4.bar(['Memory Usage (MB)'], [memory_usage], color='lightgreen')
            ax4.set_title('Dataset Memory Usage')
            ax4.text(0, memory_usage + memory_usage * 0.01, f'{memory_usage:.2f} MB', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = f"data_overview_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating data overview plot: {str(e)}")
            raise VisualizationError(f"Failed to create data overview plot: {str(e)}")
    
    async def _create_missing_values_plot(self, df: pd.DataFrame, session_id: str) -> str:
        """Create missing values heatmap"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate missing values percentage
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                missing_percent = (missing_data / len(df)) * 100
                
                # Create heatmap
                missing_matrix = df[missing_data.index].isnull()
                sns.heatmap(missing_matrix.T, cbar=True, cmap='viridis', 
                           yticklabels=missing_data.index, xticklabels=False)
                plt.title('Missing Values Heatmap')
                plt.ylabel('Columns')
                plt.xlabel('Rows')
                
                # Add percentage annotation
                for i, col in enumerate(missing_data.index):
                    plt.text(len(df) + 10, i + 0.5, f'{missing_percent[col]:.1f}%', 
                           va='center', ha='left')
            else:
                plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=16)
                plt.title('Missing Values Analysis')
            
            plt.tight_layout()
            
            filename = f"missing_values_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating missing values plot: {str(e)}")
            raise VisualizationError(f"Failed to create missing values plot: {str(e)}")
    
    async def _create_correlation_matrix(self, df: pd.DataFrame, session_id: str) -> str:
        """Create correlation matrix heatmap"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'Not enough numeric columns for correlation matrix', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Correlation Matrix')
            else:
                corr_matrix = df[numeric_cols].corr()
                
                plt.figure(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols))))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f', cbar_kws={"shrink": .8})
                plt.title('Correlation Matrix Heatmap')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            filename = f"correlation_matrix_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            raise VisualizationError(f"Failed to create correlation matrix: {str(e)}")
    
    async def _create_histograms(self, df: pd.DataFrame, session_id: str) -> str:
        """Create histograms for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No numeric columns for histograms', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Histograms')
            else:
                n_cols = min(4, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(numeric_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    if len(numeric_cols) == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_title(f'{col} Distribution')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    
                    # Add statistics
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                    ax.legend()
                
                # Hide empty subplots
                for i in range(len(numeric_cols), n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"histograms_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating histograms: {str(e)}")
            raise VisualizationError(f"Failed to create histograms: {str(e)}")
    
    async def _create_box_plots(self, df: pd.DataFrame, session_id: str) -> str:
        """Create box plots for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No numeric columns for box plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Box Plots')
            else:
                n_cols = min(4, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(numeric_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    if len(numeric_cols) == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    df.boxplot(column=col, ax=ax)
                    ax.set_title(f'{col} Box Plot')
                    ax.set_ylabel(col)
                
                # Hide empty subplots
                for i in range(len(numeric_cols), n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"box_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating box plots: {str(e)}")
            raise VisualizationError(f"Failed to create box plots: {str(e)}")
    
    async def _create_violin_plots(self, df: pd.DataFrame, session_id: str) -> str:
        """Create violin plots for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No numeric columns for violin plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Violin Plots')
            elif len(numeric_cols) <= 6:
                fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 6))
                if len(numeric_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_cols):
                    sns.violinplot(y=df[col], ax=axes[i])
                    axes[i].set_title(f'{col} Violin Plot')
                    axes[i].set_ylabel(col)
            else:
                # Too many columns, create a summary plot
                plt.figure(figsize=(15, 8))
                
                # Select first 6 most important numeric columns (by variance)
                variances = df[numeric_cols].var()
                top_cols = variances.nlargest(6).index
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                
                for i, col in enumerate(top_cols):
                    sns.violinplot(y=df[col], ax=axes[i])
                    axes[i].set_title(f'{col} Violin Plot')
                    axes[i].set_ylabel(col)
            
            plt.tight_layout()
            
            filename = f"violin_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating violin plots: {str(e)}")
            raise VisualizationError(f"Failed to create violin plots: {str(e)}")
    
    async def _create_count_plots(self, df: pd.DataFrame, session_id: str) -> str:
        """Create count plots for categorical columns"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No categorical columns for count plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Count Plots')
            else:
                # Filter columns with reasonable number of unique values
                plot_cols = []
                for col in categorical_cols:
                    if df[col].nunique() <= 20:  # Only plot if <= 20 unique values
                        plot_cols.append(col)
                
                if len(plot_cols) == 0:
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, 'No categorical columns with reasonable unique values', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                    plt.title('Count Plots')
                else:
                    n_cols = min(3, len(plot_cols))
                    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i, col in enumerate(plot_cols):
                        row = i // n_cols
                        col_idx = i % n_cols
                        
                        if len(plot_cols) == 1:
                            ax = axes[0, 0]
                        else:
                            ax = axes[row, col_idx]
                        
                        sns.countplot(data=df, x=col, ax=ax)
                        ax.set_title(f'{col} Count Plot')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Count')
                        
                        # Rotate x-axis labels if needed
                        if df[col].nunique() > 5:
                            ax.tick_params(axis='x', rotation=45)
                    
                    # Hide empty subplots
                    for i in range(len(plot_cols), n_rows * n_cols):
                        row = i // n_cols
                        col_idx = i % n_cols
                        axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"count_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating count plots: {str(e)}")
            raise VisualizationError(f"Failed to create count plots: {str(e)}")
    
    async def _create_bar_charts(self, df: pd.DataFrame, session_id: str) -> str:
        """Create bar charts for top categories"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No categorical columns for bar charts', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Bar Charts')
            else:
                # Get top categorical columns by number of unique values
                valid_cols = []
                for col in categorical_cols:
                    if 2 <= df[col].nunique() <= 15:
                        valid_cols.append(col)
                
                if len(valid_cols) == 0:
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, 'No categorical columns with 2-15 unique values', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                    plt.title('Bar Charts')
                else:
                    n_cols = min(2, len(valid_cols))
                    n_rows = (len(valid_cols) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i, col in enumerate(valid_cols):
                        row = i // n_cols
                        col_idx = i % n_cols
                        
                        if len(valid_cols) == 1:
                            ax = axes[0, 0]
                        else:
                            ax = axes[row, col_idx]
                        
                        value_counts = df[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=ax)
                        ax.set_title(f'{col} Distribution')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Count')
                        ax.tick_params(axis='x', rotation=45)
                    
                    # Hide empty subplots
                    for i in range(len(valid_cols), n_rows * n_cols):
                        row = i // n_cols
                        col_idx = i % n_cols
                        axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"bar_charts_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating bar charts: {str(e)}")
            raise VisualizationError(f"Failed to create bar charts: {str(e)}")
    
    async def _create_scatter_plots(self, df: pd.DataFrame, session_id: str, target_column: Optional[str] = None) -> str:
        """Create scatter plots for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'Need at least 2 numeric columns for scatter plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Scatter Plots')
            else:
                # Create scatter plots for most correlated pairs
                corr_matrix = df[numeric_cols].corr()
                
                # Get top correlated pairs
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr_val = abs(corr_matrix.loc[col1, col2])
                        if not np.isnan(corr_val):
                            corr_pairs.append((col1, col2, corr_val))
                
                corr_pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = corr_pairs[:6]  # Top 6 most correlated pairs
                
                if len(top_pairs) == 0:
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, 'No valid correlation pairs found', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                    plt.title('Scatter Plots')
                else:
                    n_cols = min(3, len(top_pairs))
                    n_rows = (len(top_pairs) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i, (col1, col2, corr_val) in enumerate(top_pairs):
                        row = i // n_cols
                        col_idx = i % n_cols
                        
                        if len(top_pairs) == 1:
                            ax = axes[0, 0]
                        else:
                            ax = axes[row, col_idx]
                        
                        # Color by target if available
                        if target_column and target_column in df.columns:
                            scatter = ax.scatter(df[col1], df[col2], c=df[target_column], alpha=0.6, cmap='viridis')
                            plt.colorbar(scatter, ax=ax, label=target_column)
                        else:
                            ax.scatter(df[col1], df[col2], alpha=0.6)
                        
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        ax.set_title(f'{col1} vs {col2} (r={corr_val:.3f})')
                        ax.grid(True, alpha=0.3)
                    
                    # Hide empty subplots
                    for i in range(len(top_pairs), n_rows * n_cols):
                        row = i // n_cols
                        col_idx = i % n_cols
                        axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"scatter_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating scatter plots: {str(e)}")
            raise VisualizationError(f"Failed to create scatter plots: {str(e)}")
    
    async def _create_pair_plots(self, df: pd.DataFrame, session_id: str, target_column: Optional[str] = None) -> str:
        """Create pair plots for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'Need at least 2 numeric columns for pair plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Pair Plots')
            else:
                # Limit to top 5 numeric columns to avoid overcrowding
                if len(numeric_cols) > 5:
                    # Select columns with highest variance
                    variances = df[numeric_cols].var()
                    top_cols = variances.nlargest(5).index
                else:
                    top_cols = numeric_cols
                
                # Create pair plot
                if target_column and target_column in df.columns:
                    # Add target column for coloring
                    plot_df = df[list(top_cols) + [target_column]]
                    
                    # For large datasets, sample to avoid performance issues
                    if len(plot_df) > 1000:
                        plot_df = plot_df.sample(1000, random_state=42)
                    
                    sns.pairplot(plot_df, hue=target_column, plot_kws={'alpha': 0.6})
                else:
                    plot_df = df[top_cols]
                    
                    # For large datasets, sample to avoid performance issues
                    if len(plot_df) > 1000:
                        plot_df = plot_df.sample(1000, random_state=42)
                    
                    sns.pairplot(plot_df, plot_kws={'alpha': 0.6})
                
                plt.suptitle('Pair Plot of Numeric Features', y=1.02)
            
            plt.tight_layout()
            
            filename = f"pair_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating pair plots: {str(e)}")
            raise VisualizationError(f"Failed to create pair plots: {str(e)}")
    
    async def _create_qq_plots(self, df: pd.DataFrame, session_id: str) -> str:
        """Create Q-Q plots for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No numeric columns for Q-Q plots', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Q-Q Plots')
            else:
                # Limit to first 6 columns
                plot_cols = numeric_cols[:6]
                
                n_cols = min(3, len(plot_cols))
                n_rows = (len(plot_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(plot_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    if len(plot_cols) == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    data = df[col].dropna()
                    if len(data) > 0:
                        stats.probplot(data, dist="norm", plot=ax)
                        ax.set_title(f'{col} Q-Q Plot')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center', 
                               transform=ax.transAxes)
                
                # Hide empty subplots
                for i in range(len(plot_cols), n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"qq_plots_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating Q-Q plots: {str(e)}")
            raise VisualizationError(f"Failed to create Q-Q plots: {str(e)}")
    
    async def _create_outlier_detection_plots(self, df: pd.DataFrame, session_id: str) -> str:
        """Create outlier detection plots"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No numeric columns for outlier detection', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Outlier Detection')
            else:
                # Create box plots with outlier highlighting
                n_cols = min(4, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(numeric_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    if len(numeric_cols) == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    # Calculate outliers using IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    
                    # Create box plot
                    bp = ax.boxplot(df[col].dropna(), patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    
                    # Highlight outliers
                    if len(outliers) > 0:
                        ax.scatter(np.ones(len(outliers)), outliers, color='red', alpha=0.6, s=50)
                    
                    ax.set_title(f'{col} Outliers ({len(outliers)} detected)')
                    ax.set_ylabel(col)
                    ax.set_xticklabels([col])
                
                # Hide empty subplots
                for i in range(len(numeric_cols), n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"outlier_detection_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating outlier detection plots: {str(e)}")
            raise VisualizationError(f"Failed to create outlier detection plots: {str(e)}")
    
    async def _create_target_distribution_plot(self, df: pd.DataFrame, target_column: str, session_id: str) -> str:
        """Create target distribution plot"""
        try:
            plt.figure(figsize=(12, 5))
            
            # Check if target is numeric or categorical
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 20:
                # Categorical target
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Count plot
                value_counts = df[target_column].value_counts()
                ax1.bar(range(len(value_counts)), value_counts.values)
                ax1.set_xticks(range(len(value_counts)))
                ax1.set_xticklabels(value_counts.index, rotation=45)
                ax1.set_title(f'{target_column} Distribution')
                ax1.set_xlabel(target_column)
                ax1.set_ylabel('Count')
                
                # Add count labels
                for i, v in enumerate(value_counts.values):
                    ax1.text(i, v + max(value_counts.values) * 0.01, str(v), ha='center', va='bottom')
                
                # Pie chart
                ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'{target_column} Proportion')
                
            else:
                # Numeric target
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Histogram
                ax1.hist(df[target_column].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax1.set_title(f'{target_column} Distribution')
                ax1.set_xlabel(target_column)
                ax1.set_ylabel('Frequency')
                
                # Add statistics
                mean_val = df[target_column].mean()
                median_val = df[target_column].median()
                ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax1.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
                ax1.legend()
                
                # Box plot
                ax2.boxplot(df[target_column].dropna())
                ax2.set_title(f'{target_column} Box Plot')
                ax2.set_ylabel(target_column)
                ax2.set_xticklabels([target_column])
            
            plt.tight_layout()
            
            filename = f"target_distribution_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating target distribution plot: {str(e)}")
            raise VisualizationError(f"Failed to create target distribution plot: {str(e)}")
    
    async def _create_target_relationship_plots(self, df: pd.DataFrame, target_column: str, session_id: str) -> str:
        """Create target relationship plots"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Remove target column from features
            numeric_cols = numeric_cols.drop(target_column, errors='ignore')
            categorical_cols = categorical_cols.drop(target_column, errors='ignore')
            
            if len(numeric_cols) == 0 and len(categorical_cols) == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'No feature columns for target relationships', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Target Relationships')
            else:
                # Create subplots
                total_plots = min(6, len(numeric_cols) + len(categorical_cols))
                n_cols = min(3, total_plots)
                n_rows = (total_plots + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                plot_idx = 0
                
                # Plot numeric relationships
                for col in numeric_cols[:3]:  # First 3 numeric columns
                    if plot_idx >= total_plots:
                        break
                    
                    row = plot_idx // n_cols
                    col_idx = plot_idx % n_cols
                    
                    if total_plots == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 20:
                        # Target is categorical - box plot
                        sns.boxplot(data=df, x=target_column, y=col, ax=ax)
                        ax.set_title(f'{col} vs {target_column}')
                        ax.tick_params(axis='x', rotation=45)
                    else:
                        # Target is numeric - scatter plot
                        ax.scatter(df[col], df[target_column], alpha=0.6)
                        ax.set_xlabel(col)
                        ax.set_ylabel(target_column)
                        ax.set_title(f'{col} vs {target_column}')
                        ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
                
                # Plot categorical relationships
                for col in categorical_cols[:3]:  # First 3 categorical columns
                    if plot_idx >= total_plots:
                        break
                    
                    if df[col].nunique() > 15:  # Skip if too many categories
                        continue
                    
                    row = plot_idx // n_cols
                    col_idx = plot_idx % n_cols
                    
                    if total_plots == 1:
                        ax = axes[0, 0]
                    else:
                        ax = axes[row, col_idx]
                    
                    if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 20:
                        # Both categorical - count plot
                        sns.countplot(data=df, x=col, hue=target_column, ax=ax)
                        ax.set_title(f'{col} vs {target_column}')
                        ax.tick_params(axis='x', rotation=45)
                    else:
                        # Target is numeric - box plot
                        sns.boxplot(data=df, x=col, y=target_column, ax=ax)
                        ax.set_title(f'{col} vs {target_column}')
                        ax.tick_params(axis='x', rotation=45)
                    
                    plot_idx += 1
                
                # Hide empty subplots
                for i in range(plot_idx, n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            
            filename = f"target_relationships_{session_id}.png"
            filepath = self.visualizations_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating target relationship plots: {str(e)}")
            raise VisualizationError(f"Failed to create target relationship plots: {str(e)}")