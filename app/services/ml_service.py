"""
Machine Learning Service for model training and evaluation
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb

from app.core.config import settings
from app.core.exceptions import ModelTrainingError, DataProcessingError
from app.models.schemas import EvaluationResponse, ModelComparison
from app.services.data_service import DataService

logger = logging.getLogger(__name__)


class MLService:
    """Service for machine learning operations"""
    
    def __init__(self):
        self.data_service = DataService()
        self.model_configs = {
            "classification": {
                "logistic_regression": {
                    "model": LogisticRegression,
                    "params": {
                        "C": [0.1, 1, 10],
                        "penalty": ["l1", "l2"],
                        "solver": ["liblinear", "lbfgs"]
                    }
                },
                "decision_tree": {
                    "model": DecisionTreeClassifier,
                    "params": {
                        "max_depth": [3, 5, 7, 10],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "random_forest": {
                    "model": RandomForestClassifier,
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7, 10],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "xgboost": {
                    "model": xgb.XGBClassifier,
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1, 0.2]
                    }
                }
            },
            "regression": {
                "linear_regression": {
                    "model": LinearRegression,
                    "params": {}
                },
                "decision_tree": {
                    "model": DecisionTreeRegressor,
                    "params": {
                        "max_depth": [3, 5, 7, 10],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "random_forest": {
                    "model": RandomForestRegressor,
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7, 10],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "xgboost": {
                    "model": xgb.XGBRegressor,
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1, 0.2]
                    }
                }
            }
        }
        
        # Storage for trained models and results
        self.trained_models: Dict[str, Dict[str, Any]] = {}
    
    async def train_models(
        self,
        session_id: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring_metric: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Train multiple models and select the best one
        
        Args:
            session_id: Unique session identifier
            target_column: Target column name
            test_size: Test set size
            random_state: Random state for reproducibility
            cv_folds: Cross-validation folds
            scoring_metric: Scoring metric for model selection
            progress_callback: Callback for progress updates
        """
        try:
            # Get session data
            session_data = await self.data_service.get_session_data(session_id)
            if not session_data:
                raise ModelTrainingError("Session not found")
            
            df = session_data["dataframe"]
            task_type = session_data.get("task_type")
            
            if not task_type:
                # Auto-detect task type
                task_detection = await self.data_service.detect_task_type(session_id, target_column)
                task_type = task_detection.task_type
            
            # Update progress
            if progress_callback:
                progress_callback("preprocessing", 10, "Preprocessing data...")
            
            # Prepare data
            X, y, preprocessor = await self._prepare_data(df, target_column, task_type)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == "classification" else None
            )
            
            # Update progress
            if progress_callback:
                progress_callback("training", 20, "Training models...")
            
            # Train models
            models = await self._train_all_models(
                X_train, X_test, y_train, y_test, task_type, cv_folds, scoring_metric, progress_callback
            )
            
            # Update progress
            if progress_callback:
                progress_callback("evaluation", 80, "Evaluating models...")
            
            # Evaluate models
            evaluation_results = await self._evaluate_models(models, X_test, y_test, task_type)
            
            # Select best model
            best_model_name = await self._select_best_model(evaluation_results, task_type)
            
            # Save results
            model_results = {
                "session_id": session_id,
                "task_type": task_type,
                "target_column": target_column or session_data["target_column"],
                "models": models,
                "evaluation_results": evaluation_results,
                "best_model": best_model_name,
                "preprocessor": preprocessor,
                "feature_names": X.columns.tolist(),
                "training_params": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "cv_folds": cv_folds,
                    "scoring_metric": scoring_metric
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.trained_models[session_id] = model_results
            
            # Save models to disk
            await self._save_models(session_id, model_results)
            
            # Update progress
            if progress_callback:
                progress_callback("completed", 100, "Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    async def _prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str], 
        task_type: str
    ) -> tuple:
        """Prepare data for training"""
        # Determine target column
        if not target_column:
            target_column = df.columns[-1]
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values in target
        if y.isnull().any():
            # Remove rows with missing target
            missing_mask = y.isnull()
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        # Encode target for classification
        if task_type == "classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Create preprocessor
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Preprocessing pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = (
            numeric_features + 
            list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
        )
        
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        return X_processed, y, preprocessor
    
    async def _train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        cv_folds: int,
        scoring_metric: Optional[str],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train all models for the given task type"""
        models = {}
        model_configs = self.model_configs[task_type]
        
        total_models = len(model_configs)
        
        for i, (model_name, config) in enumerate(model_configs.items()):
            try:
                if progress_callback:
                    progress = 20 + (i / total_models) * 60
                    progress_callback("training", progress, f"Training {model_name}...")
                
                # Create model
                model_class = config["model"]
                model = model_class(random_state=42, n_jobs=settings.N_JOBS if hasattr(model_class(), 'n_jobs') else None)
                
                # Hyperparameter tuning
                if config["params"]:
                    # Use RandomizedSearchCV for faster training
                    search = RandomizedSearchCV(
                        model,
                        config["params"],
                        n_iter=10,
                        cv=cv_folds,
                        scoring=scoring_metric,
                        random_state=42,
                        n_jobs=settings.N_JOBS
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    cv_score = search.best_score_
                else:
                    # No hyperparameters to tune
                    model.fit(X_train, y_train)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring_metric)
                    cv_score = cv_scores.mean()
                    best_model = model
                    best_params = {}
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                if hasattr(best_model, 'predict_proba'):
                    y_pred_proba = best_model.predict_proba(X_test)
                else:
                    y_pred_proba = None
                
                models[model_name] = {
                    "model": best_model,
                    "params": best_params,
                    "cv_score": cv_score,
                    "predictions": y_pred,
                    "predictions_proba": y_pred_proba,
                    "feature_importance": self._get_feature_importance(best_model, X_train.columns)
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                # Continue training other models
                continue
        
        return models
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficient values
                if model.coef_.ndim == 1:
                    importances = np.abs(model.coef_)
                else:
                    # For multi-class, take mean of absolute coefficients
                    importances = np.mean(np.abs(model.coef_), axis=0)
                return dict(zip(feature_names, importances))
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None
    
    async def _evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Evaluate all trained models"""
        evaluation_results = {}
        
        for model_name, model_data in models.items():
            try:
                y_pred = model_data["predictions"]
                y_pred_proba = model_data["predictions_proba"]
                
                if task_type == "classification":
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                evaluation_results[model_name] = {
                    "metrics": metrics,
                    "cv_score": model_data["cv_score"],
                    "feature_importance": model_data["feature_importance"]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return evaluation_results
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        return metrics
    
    async def _select_best_model(
        self, 
        evaluation_results: Dict[str, Any], 
        task_type: str
    ) -> str:
        """Select the best model based on evaluation results"""
        if not evaluation_results:
            raise ModelTrainingError("No models were successfully trained")
        
        # Define scoring criteria
        if task_type == "classification":
            score_key = "f1"
        else:
            score_key = "r2"
        
        # Find best model
        best_model = max(
            evaluation_results.items(),
            key=lambda x: x[1]["metrics"].get(score_key, -np.inf)
        )
        
        return best_model[0]
    
    async def _save_models(self, session_id: str, model_results: Dict[str, Any]):
        """Save trained models to disk"""
        models_dir = os.path.join(settings.MODELS_DIR, session_id)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save each model
        for model_name, model_data in model_results["models"].items():
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model_data["model"], model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        joblib.dump(model_results["preprocessor"], preprocessor_path)
        
        # Save metadata
        metadata_path = os.path.join(models_dir, "metadata.json")
        metadata = {
            "session_id": session_id,
            "task_type": model_results["task_type"],
            "target_column": model_results["target_column"],
            "feature_names": model_results["feature_names"],
            "best_model": model_results["best_model"],
            "training_params": model_results["training_params"],
            "timestamp": model_results["timestamp"]
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    async def get_training_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training results for a session"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        # Format results for API response
        formatted_results = {
            "session_id": session_id,
            "task_type": results["task_type"],
            "target_column": results["target_column"],
            "best_model": results["best_model"],
            "models": {
                name: {
                    "cv_score": data["cv_score"],
                    "params": data["params"],
                    "feature_importance": data["feature_importance"]
                }
                for name, data in results["models"].items()
            },
            "evaluation_results": results["evaluation_results"],
            "training_params": results["training_params"],
            "timestamp": results["timestamp"]
        }
        
        return formatted_results
    
    async def get_evaluation_results(self, session_id: str) -> Optional[EvaluationResponse]:
        """Get evaluation results for a session"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        evaluation_response = EvaluationResponse(
            session_id=session_id,
            task_type=results["task_type"],
            best_model=results["best_model"],
            model_results=results["evaluation_results"],
            training_timestamp=results["timestamp"]
        )
        
        return evaluation_response
    
    async def get_best_model_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about the best model"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        best_model_name = results["best_model"]
        
        best_model_info = {
            "model_name": best_model_name,
            "model_type": results["task_type"],
            "cv_score": results["models"][best_model_name]["cv_score"],
            "parameters": results["models"][best_model_name]["params"],
            "metrics": results["evaluation_results"][best_model_name]["metrics"],
            "feature_importance": results["evaluation_results"][best_model_name]["feature_importance"]
        }
        
        return best_model_info
    
    async def compare_models(
        self, 
        session_id: str, 
        models: Optional[List[str]] = None
    ) -> Optional[ModelComparison]:
        """Compare multiple models"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        if models is None:
            models = list(results["models"].keys())
        
        # Filter to only include requested models
        available_models = [m for m in models if m in results["models"]]
        
        if not available_models:
            return None
        
        comparison_data = {}
        for model_name in available_models:
            comparison_data[model_name] = {
                "cv_score": results["models"][model_name]["cv_score"],
                "metrics": results["evaluation_results"][model_name]["metrics"],
                "parameters": results["models"][model_name]["params"]
            }
        
        comparison = ModelComparison(
            session_id=session_id,
            task_type=results["task_type"],
            models=comparison_data,
            best_model=results["best_model"]
        )
        
        return comparison
    
    async def get_model_metrics(self, session_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific model"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        if model_name not in results["models"]:
            return None
        
        return results["evaluation_results"][model_name]
    
    async def get_feature_importance(
        self, 
        session_id: str, 
        model_name: Optional[str] = None, 
        top_n: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Get feature importance for a model"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        if model_name is None:
            model_name = results["best_model"]
        
        if model_name not in results["models"]:
            return None
        
        feature_importance = results["evaluation_results"][model_name]["feature_importance"]
        
        if feature_importance is None:
            return None
        
        # Sort by importance and get top N
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            "model_name": model_name,
            "feature_importance": dict(sorted_features),
            "total_features": len(feature_importance)
        }
    
    async def make_predictions(
        self, 
        session_id: str, 
        prediction_data: Dict[str, Any], 
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Make predictions using a trained model"""
        if session_id not in self.trained_models:
            return None
        
        results = self.trained_models[session_id]
        
        if model_name is None:
            model_name = results["best_model"]
        
        if model_name not in results["models"]:
            return None
        
        try:
            # Load model and preprocessor
            model = results["models"][model_name]["model"]
            preprocessor = results["preprocessor"]
            
            # Prepare input data
            input_df = pd.DataFrame([prediction_data])
            
            # Preprocess input
            X_processed = preprocessor.transform(input_df)
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_processed)
            else:
                prediction_proba = None
            
            return {
                "predictions": predictions.tolist(),
                "prediction_proba": prediction_proba.tolist() if prediction_proba is not None else None,
                "model_name": model_name,
                "input_features": list(prediction_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
