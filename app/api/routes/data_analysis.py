"""
Data Analysis Routes for comprehensive dataset handling
"""

import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
import traceback

from app.models.schemas import (
    DatasetPreview, TaskDetectionResponse, ErrorResponse
)
from app.core.exceptions import DataProcessingError, VisualizationError
from app.core.config import settings
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter()
data_service = DataService()

@router.post("/analyze/{session_id}", response_model=Dict[str, Any])
async def perform_comprehensive_analysis(
    session_id: str,
    target_column: Optional[str] = None
):
    """
    Perform comprehensive data analysis including data inspection and visualization
    """
    try:
        # Get session data
        session_data = await data_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found for session")
        
        # Perform comprehensive data inspection
        inspection_results = await perform_data_inspection(df)
        
        # Generate visualizations
        visualizations = await generate_comprehensive_visualizations(df, session_id, target_column)
        
        # Store results in session
        session_data['analysis_results'] = {
            'inspection': inspection_results,
            'visualizations': visualizations,
            'target_column': target_column
        }
        
        return {
            'session_id': session_id,
            'analysis_completed': True,
            'inspection_results': inspection_results,
            'visualizations': visualizations,
            'target_column': target_column
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/inspection/{session_id}", response_model=Dict[str, Any])
async def get_data_inspection(session_id: str):
    """
    Get comprehensive data inspection results
    """
    try:
        session_data = await data_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found for session")
        
        inspection_results = await perform_data_inspection(df)
        
        return {
            'session_id': session_id,
            'inspection_results': inspection_results
        }
        
    except Exception as e:
        logger.error(f"Error getting data inspection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")

@router.get("/visualizations/{session_id}", response_model=Dict[str, Any])
async def get_visualizations(
    session_id: str,
    target_column: Optional[str] = None,
    chart_types: Optional[str] = None
):
    """
    Generate and return visualization URLs
    """
    try:
        session_data = await data_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found for session")
        
        # Parse chart types if provided
        requested_charts = []
        if chart_types:
            requested_charts = chart_types.split(',')
        
        visualizations = await generate_comprehensive_visualizations(
            df, session_id, target_column, requested_charts
        )
        
        return {
            'session_id': session_id,
            'visualizations': visualizations,
            'target_column': target_column
        }
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@router.get("/visualization/{session_id}/{chart_type}")
async def get_visualization_file(session_id: str, chart_type: str):
    """
    Serve visualization image files
    """
    try:
        file_path = Path(settings.VISUALIZATIONS_DIR) / f"{chart_type}_{session_id}.png"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        return FileResponse(
            path=file_path,
            media_type="image/png",
            filename=f"{chart_type}_{session_id}.png"
        )
        
    except Exception as e:
        logger.error(f"Error serving visualization file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve visualization: {str(e)}")

@router.post("/preprocess/{session_id}", response_model=Dict[str, Any])
async def preprocess_dataset(
    session_id: str,
    preprocessing_config: Dict[str, Any] = None
):
    """
    Apply preprocessing steps to the dataset
    """
    try:
        session_data = await data_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found for session")
        
        # Apply preprocessing
        preprocessed_df = await apply_preprocessing(df, preprocessing_config or {})
        
        # Update session with preprocessed data
        session_data['preprocessed_dataframe'] = preprocessed_df
        session_data['preprocessing_applied'] = True
        session_data['preprocessing_config'] = preprocessing_config
        
        # Generate new inspection results
        inspection_results = await perform_data_inspection(preprocessed_df)
        
        return {
            'session_id': session_id,
            'preprocessing_completed': True,
            'original_shape': df.shape,
            'preprocessed_shape': preprocessed_df.shape,
            'preprocessing_config': preprocessing_config,
            'inspection_results': inspection_results
        }
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

async def perform_data_inspection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data inspection
    """
    try:
        inspection = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage': df.memory_usage(deep=True).sum(),
            },
            'data_samples': {
                'head': df.head().fillna('NaN').to_dict('records'),
                'tail': df.tail().fillna('NaN').to_dict('records'),
                'sample': df.sample(n=min(5, len(df))).fillna('NaN').to_dict('records') if len(df) > 0 else []
            },
            'missing_values': {
                'count': df.isnull().sum().to_dict(),
                'percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'duplicates': {
                'count': df.duplicated().sum(),
                'percentage': df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
            },
            'column_analysis': {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().sum() / len(df) * 100 if len(df) > 0 else 0
            }
            
            if df[col].dtype in ['object', 'category']:
                # Categorical analysis
                value_counts = df[col].value_counts()
                col_info.update({
                    'type': 'categorical',
                    'top_values': value_counts.head(10).to_dict(),
                    'unique_values': df[col].unique().tolist() if df[col].nunique() <= 20 else f"Too many ({df[col].nunique()})"
                })
            else:
                # Numerical analysis
                desc = df[col].describe()
                col_info.update({
                    'type': 'numerical',
                    'statistics': {
                        'mean': desc['mean'] if 'mean' in desc else None,
                        'std': desc['std'] if 'std' in desc else None,
                        'min': desc['min'] if 'min' in desc else None,
                        'max': desc['max'] if 'max' in desc else None,
                        'median': desc['50%'] if '50%' in desc else None,
                        'q1': desc['25%'] if '25%' in desc else None,
                        'q3': desc['75%'] if '75%' in desc else None
                    }
                })
                
                # Outlier detection using IQR
                if df[col].dtype in [np.number] and len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    col_info['outliers'] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(df) * 100
                    }
            
            inspection['column_analysis'][col] = col_info
        
        # Data quality assessment
        quality_score = 100.0
        issues = []
        recommendations = []
        
        # Missing values assessment
        high_missing_cols = [col for col, pct in inspection['missing_values']['percentage'].items() if pct > 50]
        if high_missing_cols:
            quality_score -= 20
            issues.append(f"High missing values in columns: {high_missing_cols}")
            recommendations.append("Consider dropping columns with >50% missing values")
        
        # Duplicates assessment
        if inspection['duplicates']['percentage'] > 5:
            quality_score -= 15
            issues.append(f"High duplicate rows: {inspection['duplicates']['percentage']:.1f}%")
            recommendations.append("Remove duplicate rows")
        
        inspection['data_quality'] = {
            'score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
        
        return inspection
        
    except Exception as e:
        logger.error(f"Error in data inspection: {str(e)}")
        raise DataProcessingError(f"Failed to perform data inspection: {str(e)}")

async def generate_comprehensive_visualizations(
    df: pd.DataFrame, 
    session_id: str, 
    target_column: Optional[str] = None,
    requested_charts: List[str] = None
) -> Dict[str, str]:
    """
    Generate comprehensive visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        # Set up visualization directory
        viz_dir = Path(settings.VISUALIZATIONS_DIR)
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        visualizations = {}
        
        # Data overview
        if not requested_charts or 'data_overview' in requested_charts:
            viz_path = await create_data_overview_plot(df, session_id, viz_dir)
            visualizations['data_overview'] = f"/api/v1/data-analysis/visualization/{session_id}/data_overview"
        
        # Missing values
        if not requested_charts or 'missing_values' in requested_charts:
            viz_path = await create_missing_values_plot(df, session_id, viz_dir)
            visualizations['missing_values'] = f"/api/v1/data-analysis/visualization/{session_id}/missing_values"
        
        # Correlation matrix
        if not requested_charts or 'correlation_matrix' in requested_charts:
            viz_path = await create_correlation_matrix(df, session_id, viz_dir)
            visualizations['correlation_matrix'] = f"/api/v1/data-analysis/visualization/{session_id}/correlation_matrix"
        
        # Histograms
        if not requested_charts or 'histograms' in requested_charts:
            viz_path = await create_histograms(df, session_id, viz_dir)
            visualizations['histograms'] = f"/api/v1/data-analysis/visualization/{session_id}/histograms"
        
        # Box plots
        if not requested_charts or 'box_plots' in requested_charts:
            viz_path = await create_box_plots(df, session_id, viz_dir)
            visualizations['box_plots'] = f"/api/v1/data-analysis/visualization/{session_id}/box_plots"
        
        # Count plots
        if not requested_charts or 'count_plots' in requested_charts:
            viz_path = await create_count_plots(df, session_id, viz_dir)
            visualizations['count_plots'] = f"/api/v1/data-analysis/visualization/{session_id}/count_plots"
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise VisualizationError(f"Failed to generate visualizations: {str(e)}")

async def create_data_overview_plot(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create data overview visualization"""
    import matplotlib.pyplot as plt
    
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
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating data overview plot: {str(e)}")
        raise VisualizationError(f"Failed to create data overview plot: {str(e)}")

async def create_missing_values_plot(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create missing values visualization"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        plt.figure(figsize=(12, 8))
        
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
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16)
            plt.title('Missing Values Analysis')
        
        plt.tight_layout()
        
        filename = f"missing_values_{session_id}.png"
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating missing values plot: {str(e)}")
        raise VisualizationError(f"Failed to create missing values plot: {str(e)}")

async def create_correlation_matrix(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create correlation matrix"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")
        raise VisualizationError(f"Failed to create correlation matrix: {str(e)}")

async def create_histograms(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create histograms for numeric columns"""
    import matplotlib.pyplot as plt
    
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
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols
                col_idx = i % n_cols
                
                if n_rows == 1 and n_cols == 1:
                    ax = axes[0]
                elif n_rows == 1:
                    ax = axes[0, col_idx]
                else:
                    ax = axes[row, col_idx]
                
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{col} Distribution')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_val = df[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.legend()
            
            # Hide empty subplots
            for i in range(len(numeric_cols), n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                if n_rows > 1:
                    axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        
        filename = f"histograms_{session_id}.png"
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating histograms: {str(e)}")
        raise VisualizationError(f"Failed to create histograms: {str(e)}")

async def create_box_plots(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create box plots for numeric columns"""
    import matplotlib.pyplot as plt
    
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
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols
                col_idx = i % n_cols
                
                if n_rows == 1 and n_cols == 1:
                    ax = axes[0]
                elif n_rows == 1:
                    ax = axes[0, col_idx]
                else:
                    ax = axes[row, col_idx]
                
                df.boxplot(column=col, ax=ax)
                ax.set_title(f'{col} Box Plot')
                ax.set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                if n_rows > 1:
                    axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        
        filename = f"box_plots_{session_id}.png"
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating box plots: {str(e)}")
        raise VisualizationError(f"Failed to create box plots: {str(e)}")

async def create_count_plots(df: pd.DataFrame, session_id: str, viz_dir: Path) -> str:
    """Create count plots for categorical columns"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(plot_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    
                    if n_rows == 1 and n_cols == 1:
                        ax = axes[0]
                    elif n_rows == 1:
                        ax = axes[0, col_idx]
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
                    if n_rows > 1:
                        axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        
        filename = f"count_plots_{session_id}.png"
        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error creating count plots: {str(e)}")
        raise VisualizationError(f"Failed to create count plots: {str(e)}")

async def apply_preprocessing(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply basic preprocessing steps
    """
    try:
        processed_df = df.copy()
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            removed_rows = initial_rows - len(processed_df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        if config.get('handle_missing_values', True):
            # Drop columns with too many missing values
            missing_threshold = config.get('missing_threshold', 0.5)
            missing_pct = processed_df.isnull().sum() / len(processed_df)
            cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
            
            if cols_to_drop:
                processed_df = processed_df.drop(columns=cols_to_drop)
                logger.info(f"Dropped columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
            
            # Simple imputation
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
            
            # Fill numeric with median
            for col in numeric_cols:
                if processed_df[col].isnull().any():
                    median_val = processed_df[col].median()
                    processed_df[col].fillna(median_val, inplace=True)
            
            # Fill categorical with mode
            for col in categorical_cols:
                if processed_df[col].isnull().any():
                    mode_val = processed_df[col].mode()
                    if len(mode_val) > 0:
                        processed_df[col].fillna(mode_val[0], inplace=True)
                    else:
                        processed_df[col].fillna('Unknown', inplace=True)
        
        logger.info(f"Preprocessing completed. Shape: {processed_df.shape}")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise DataProcessingError(f"Failed to preprocess data: {str(e)}")