"""
Data Service for handling dataset operations
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from app.models.schemas import DatasetPreview, TaskDetectionResponse
from app.core.exceptions import DataProcessingError, FileProcessingError
from app.core.config import settings
from app.utils.file_utils import detect_file_type, read_file_to_dataframe

logger = logging.getLogger(__name__)


class DataService:
    """Service for data operations"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    async def load_and_preview_dataset(
        self, 
        file_path: str, 
        session_id: str,
        target_column: Optional[str] = None
    ) -> DatasetPreview:
        """
        Load dataset and create preview
        
        Args:
            file_path: Path to the uploaded file
            session_id: Unique session identifier
            target_column: Optional target column override
            
        Returns:
            DatasetPreview: Basic dataset information
        """
        try:
            # Load dataset
            df = read_file_to_dataframe(file_path)
            
            # Detect target column
            if target_column and target_column not in df.columns:
                raise DataProcessingError(f"Target column '{target_column}' not found in dataset")
            
            if not target_column:
                target_column = df.columns[-1]  # Default to last column
            
            # Store session data
            self.sessions[session_id] = {
                "file_path": file_path,
                "dataframe": df,
                "target_column": target_column,
                "original_shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            }
            
            # Create preview
            preview = DatasetPreview(
                session_id=session_id,
                filename=os.path.basename(file_path),
                shape=df.shape,
                columns=df.columns.tolist(),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                target_column=target_column,
                sample_data=df.head(5).to_dict(orient="records"),
                missing_values=df.isnull().sum().to_dict(),
                memory_usage=df.memory_usage(deep=True).sum(),
                numerical_columns=df.select_dtypes(include=[np.number]).columns.tolist(),
                categorical_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
            )
            
            return preview
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise DataProcessingError(f"Failed to load dataset: {str(e)}")
    
    async def get_dataset_preview(self, session_id: str) -> Optional[DatasetPreview]:
        """Get existing dataset preview"""
        if session_id not in self.sessions:
            return None
        
        session_data = self.sessions[session_id]
        df = session_data["dataframe"]
        
        preview = DatasetPreview(
            session_id=session_id,
            filename=os.path.basename(session_data["file_path"]),
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            target_column=session_data["target_column"],
            sample_data=df.head(5).to_dict(orient="records"),
            missing_values=df.isnull().sum().to_dict(),
            memory_usage=df.memory_usage(deep=True).sum(),
            numerical_columns=df.select_dtypes(include=[np.number]).columns.tolist(),
            categorical_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
        )
        
        return preview
    
    async def detect_task_type(
        self, 
        session_id: str, 
        target_column: Optional[str] = None
    ) -> TaskDetectionResponse:
        """
        Detect whether the task is classification or regression
        
        Args:
            session_id: Unique session identifier
            target_column: Optional target column override
            
        Returns:
            TaskDetectionResponse: Task type and analysis
        """
        if session_id not in self.sessions:
            raise DataProcessingError("Session not found")
        
        session_data = self.sessions[session_id]
        df = session_data["dataframe"]
        
        # Update target column if provided
        if target_column:
            if target_column not in df.columns:
                raise DataProcessingError(f"Target column '{target_column}' not found")
            session_data["target_column"] = target_column
        
        target_col = session_data["target_column"]
        target_series = df[target_col]
        
        # Analyze target column
        task_type = self._determine_task_type(target_series)
        
        # Get target statistics
        target_stats = self._get_target_statistics(target_series, task_type)
        
        # Feature analysis
        features = [col for col in df.columns if col != target_col]
        feature_analysis = self._analyze_features(df[features])
        
        response = TaskDetectionResponse(
            session_id=session_id,
            task_type=task_type,
            target_column=target_col,
            target_stats=target_stats,
            confidence_score=self._calculate_confidence_score(target_series, task_type),
            feature_count=len(features),
            feature_analysis=feature_analysis,
            recommendations=self._generate_recommendations(df, target_series, task_type)
        )
        
        # Store task type in session
        session_data["task_type"] = task_type
        
        return response
    
    def _determine_task_type(self, target_series: pd.Series) -> str:
        """Determine if task is classification or regression"""
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target_series):
            # Check unique values ratio
            unique_ratio = target_series.nunique() / len(target_series)
            
            # If less than 10% unique values or less than 20 unique values, likely classification
            if unique_ratio < 0.1 or target_series.nunique() < 20:
                return "classification"
            else:
                return "regression"
        else:
            # Non-numeric target is classification
            return "classification"
    
    def _get_target_statistics(self, target_series: pd.Series, task_type: str) -> Dict[str, Any]:
        """Get statistics for target column"""
        stats = {
            "unique_values": target_series.nunique(),
            "missing_values": target_series.isnull().sum(),
            "data_type": str(target_series.dtype)
        }
        
        if task_type == "classification":
            stats["class_distribution"] = target_series.value_counts().to_dict()
            stats["class_balance"] = target_series.value_counts(normalize=True).to_dict()
        else:
            stats["mean"] = target_series.mean()
            stats["median"] = target_series.median()
            stats["std"] = target_series.std()
            stats["min"] = target_series.min()
            stats["max"] = target_series.max()
        
        return stats
    
    def _analyze_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature columns"""
        analysis = {
            "numerical_features": features_df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_features": features_df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_values_per_feature": features_df.isnull().sum().to_dict(),
            "high_cardinality_features": [],
            "constant_features": []
        }
        
        # Check for high cardinality categorical features
        for col in analysis["categorical_features"]:
            if features_df[col].nunique() > 50:
                analysis["high_cardinality_features"].append(col)
        
        # Check for constant features
        for col in features_df.columns:
            if features_df[col].nunique() == 1:
                analysis["constant_features"].append(col)
        
        return analysis
    
    def _calculate_confidence_score(self, target_series: pd.Series, task_type: str) -> float:
        """Calculate confidence score for task type detection"""
        if task_type == "classification":
            unique_ratio = target_series.nunique() / len(target_series)
            if unique_ratio < 0.05:
                return 0.95
            elif unique_ratio < 0.1:
                return 0.85
            elif target_series.nunique() < 20:
                return 0.75
            else:
                return 0.6
        else:
            unique_ratio = target_series.nunique() / len(target_series)
            if unique_ratio > 0.5:
                return 0.95
            elif unique_ratio > 0.2:
                return 0.85
            else:
                return 0.7
    
    def _generate_recommendations(
        self, 
        df: pd.DataFrame, 
        target_series: pd.Series, 
        task_type: str
    ) -> List[str]:
        """Generate recommendations based on data analysis"""
        recommendations = []
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.1:
            recommendations.append("High proportion of missing values detected. Consider data cleaning.")
        
        # Check for class imbalance in classification
        if task_type == "classification":
            class_counts = target_series.value_counts()
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 10:
                    recommendations.append("Significant class imbalance detected. Consider resampling techniques.")
        
        # Check for high cardinality features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_series.name and df[col].nunique() > 50:
                recommendations.append(f"High cardinality feature '{col}' detected. Consider feature engineering.")
        
        # Check dataset size
        if df.shape[0] < 1000:
            recommendations.append("Small dataset detected. Consider collecting more data for better model performance.")
        
        return recommendations
    
    async def get_columns_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get column information for the dataset"""
        if session_id not in self.sessions:
            return None
        
        session_data = self.sessions[session_id]
        df = session_data["dataframe"]
        
        columns_info = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "target_column": session_data["target_column"],
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return columns_info
    
    async def get_dataframe(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get the dataframe for a session"""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id]["dataframe"]
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get complete session data"""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id]
    
    async def cleanup_session(self, session_id: str):
        """Clean up session data and files"""
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            
            # Remove uploaded file
            if "file_path" in session_data:
                try:
                    os.remove(session_data["file_path"])
                except OSError:
                    pass
            
            # Remove session data
            del self.sessions[session_id]
        
        # Clean up related files
        for directory in [settings.MODELS_DIR, settings.VISUALIZATIONS_DIR, settings.REPORTS_DIR]:
            session_dir = os.path.join(directory, session_id)
            if os.path.exists(session_dir):
                import shutil
                shutil.rmtree(session_dir)
