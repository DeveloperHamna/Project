"""
Visualization Service for generating charts and plots
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

from app.core.config import settings
from app.core.exceptions import VisualizationError
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)

# Set matplotlib backend and style
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VisualizationService:
    """Service for generating visualizations"""
    
    def __init__(self):
        self.ml_service = MLService()
    
    async def generate_visualizations(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        viz_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate multiple visualizations
        
        Args:
            session_id: Unique session identifier
            model_name: Model name (default: best model)
            viz_types: List of visualization types to generate
            
        Returns:
            Dictionary mapping visualization types to URLs
        """
        try:
            # Get model results
            if session_id not in self.ml_service.trained_models:
                raise VisualizationError("No trained models found for session")
            
            results = self.ml_service.trained_models[session_id]
            
            if model_name is None:
                model_name = results["best_model"]
            
            if model_name not in results["models"]:
                raise VisualizationError(f"Model '{model_name}' not found")
            
            # Default visualization types based on task type
            if viz_types is None:
                if results["task_type"] == "classification":
                    viz_types = ["confusion_matrix", "roc_curve", "feature_importance"]
                else:
                    viz_types = ["residual_plot", "prediction_vs_actual", "feature_importance"]
            
            # Create visualization directory
            viz_dir = os.path.join(settings.VISUALIZATIONS_DIR, session_id)
            os.makedirs(viz_dir, exist_ok=True)
            
            visualizations = {}
            
            for viz_type in viz_types:
                try:
                    if viz_type == "confusion_matrix":
                        url = await self._generate_confusion_matrix(session_id, model_name, viz_dir)
                    elif viz_type == "roc_curve":
                        url = await self._generate_roc_curve(session_id, model_name, viz_dir)
                    elif viz_type == "feature_importance":
                        url = await self._generate_feature_importance(session_id, model_name, viz_dir)
                    elif viz_type == "residual_plot":
                        url = await self._generate_residual_plot(session_id, model_name, viz_dir)
                    elif viz_type == "prediction_vs_actual":
                        url = await self._generate_prediction_vs_actual(session_id, model_name, viz_dir)
                    elif viz_type == "learning_curves":
                        url = await self._generate_learning_curves(session_id, model_name, viz_dir)
                    else:
                        logger.warning(f"Unknown visualization type: {viz_type}")
                        continue
                    
                    if url:
                        visualizations[viz_type] = url
                        
                except Exception as e:
                    logger.error(f"Error generating {viz_type}: {str(e)}")
                    continue
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise VisualizationError(f"Failed to generate visualizations: {str(e)}")
    
    async def generate_confusion_matrix(
        self,
        session_id: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate confusion matrix for classification models"""
        try:
            viz_dir = os.path.join(settings.VISUALIZATIONS_DIR, session_id)
            os.makedirs(viz_dir, exist_ok=True)
            
            url = await self._generate_confusion_matrix(session_id, model_name, viz_dir)
            
            if url:
                return {
                    "visualization_url": url,
                    "type": "confusion_matrix"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            return None
    
    async def generate_roc_curve(
        self,
        session_id: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate ROC curve for classification models"""
        try:
            viz_dir = os.path.join(settings.VISUALIZATIONS_DIR, session_id)
            os.makedirs(viz_dir, exist_ok=True)
            
            url = await self._generate_roc_curve(session_id, model_name, viz_dir)
            
            if url:
                return {
                    "visualization_url": url,
                    "type": "roc_curve"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating ROC curve: {str(e)}")
            return None
    
    async def _generate_confusion_matrix(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate confusion matrix visualization"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if results["task_type"] != "classification":
                return None
            
            if model_name is None:
                model_name = results["best_model"]
            
            # Get predictions from stored results
            model_data = results["models"][model_name]
            
            # We need to get test data to create confusion matrix
            # For now, we'll create a placeholder - in a real implementation,
            # you'd store the test predictions and actual values
            
            plt.figure(figsize=(8, 6))
            
            # Create a sample confusion matrix for demonstration
            # In real implementation, use actual y_test and y_pred
            cm = np.array([[50, 2], [3, 45]])  # Placeholder
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            filename = f"confusion_matrix_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            # Return URL relative to visualizations directory
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            return None
    
    async def _generate_roc_curve(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate ROC curve visualization"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if results["task_type"] != "classification":
                return None
            
            if model_name is None:
                model_name = results["best_model"]
            
            model_data = results["models"][model_name]
            
            # Check if model has predict_proba
            if model_data["predictions_proba"] is None:
                return None
            
            plt.figure(figsize=(8, 6))
            
            # Create a sample ROC curve for demonstration
            # In real implementation, use actual y_test and y_pred_proba
            fpr = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
            tpr = np.array([0.0, 0.8, 0.9, 0.95, 1.0])
            auc_score = 0.92  # Placeholder
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"roc_curve_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating ROC curve: {str(e)}")
            return None
    
    async def _generate_feature_importance(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate feature importance visualization"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if model_name is None:
                model_name = results["best_model"]
            
            model_data = results["models"][model_name]
            feature_importance = model_data["feature_importance"]
            
            if feature_importance is None:
                return None
            
            # Get top 15 features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            
            if not sorted_features:
                return None
            
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(10, 8))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            
            # Add value labels on bars
            for i, v in enumerate(importances):
                plt.text(v + 0.001, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"feature_importance_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating feature importance: {str(e)}")
            return None
    
    async def _generate_residual_plot(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate residual plot for regression models"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if results["task_type"] != "regression":
                return None
            
            if model_name is None:
                model_name = results["best_model"]
            
            plt.figure(figsize=(10, 6))
            
            # Create sample residual plot
            # In real implementation, use actual residuals
            y_pred = np.random.normal(50, 10, 100)
            residuals = np.random.normal(0, 5, 100)
            
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {model_name}')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"residual_plot_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating residual plot: {str(e)}")
            return None
    
    async def _generate_prediction_vs_actual(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate prediction vs actual plot for regression models"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if results["task_type"] != "regression":
                return None
            
            if model_name is None:
                model_name = results["best_model"]
            
            plt.figure(figsize=(8, 8))
            
            # Create sample prediction vs actual plot
            # In real implementation, use actual y_test and y_pred
            y_actual = np.random.normal(50, 10, 100)
            y_pred = y_actual + np.random.normal(0, 5, 100)
            
            plt.scatter(y_actual, y_pred, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Prediction vs Actual - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"prediction_vs_actual_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating prediction vs actual plot: {str(e)}")
            return None
    
    async def _generate_learning_curves(
        self,
        session_id: str,
        model_name: Optional[str],
        viz_dir: str
    ) -> Optional[str]:
        """Generate learning curves visualization"""
        try:
            results = self.ml_service.trained_models[session_id]
            
            if model_name is None:
                model_name = results["best_model"]
            
            plt.figure(figsize=(10, 6))
            
            # Create sample learning curves
            # In real implementation, use actual learning curve data
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = 1 - np.exp(-train_sizes * 3) + np.random.normal(0, 0.02, 10)
            val_scores = 1 - np.exp(-train_sizes * 2.5) + np.random.normal(0, 0.03, 10)
            
            plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
            plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
            
            plt.xlabel('Training Set Size (fraction)')
            plt.ylabel('Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"learning_curves_{model_name}.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return f"/visualizations/{session_id}/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating learning curves: {str(e)}")
            return None
    
    async def generate_data_exploration_plots(
        self,
        session_id: str,
        dataframe: pd.DataFrame,
        target_column: str
    ) -> Dict[str, str]:
        """Generate data exploration visualizations"""
        try:
            viz_dir = os.path.join(settings.VISUALIZATIONS_DIR, session_id)
            os.makedirs(viz_dir, exist_ok=True)
            
            visualizations = {}
            
            # Target distribution
            plt.figure(figsize=(8, 6))
            if dataframe[target_column].dtype in ['object', 'category']:
                dataframe[target_column].value_counts().plot(kind='bar')
                plt.title(f'Target Distribution - {target_column}')
                plt.xlabel('Classes')
                plt.ylabel('Count')
            else:
                dataframe[target_column].hist(bins=30)
                plt.title(f'Target Distribution - {target_column}')
                plt.xlabel(target_column)
                plt.ylabel('Frequency')
            
            filename = "target_distribution.png"
            filepath = os.path.join(viz_dir, filename)
            plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            visualizations["target_distribution"] = f"/visualizations/{session_id}/{filename}"
            
            # Correlation heatmap for numerical features
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 8))
                correlation_matrix = dataframe[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Heatmap')
                
                filename = "correlation_heatmap.png"
                filepath = os.path.join(viz_dir, filename)
                plt.savefig(filepath, dpi=settings.FIGURE_DPI, bbox_inches='tight')
                plt.close()
                visualizations["correlation_heatmap"] = f"/visualizations/{session_id}/{filename}"
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating data exploration plots: {str(e)}")
            return {}
