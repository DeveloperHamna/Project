"""
Machine Learning utility functions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
    """
    Detect outliers in numerical columns
    
    Args:
        df: DataFrame to analyze
        method: Method to use ('iqr', 'zscore', 'isolation_forest')
        
    Returns:
        Dict mapping column names to lists of outlier indices
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df[outlier_mask].index.tolist()
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > 3
            outliers[col] = df[outlier_mask].index.tolist()
    
    return outliers


def calculate_class_weights(y: pd.Series) -> Dict[Any, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target variable
        
    Returns:
        Dict mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def analyze_feature_correlations(df: pd.DataFrame, threshold: float = 0.8) -> Dict[str, List[str]]:
    """
    Find highly correlated features
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold
        
    Returns:
        Dict mapping features to their highly correlated counterparts
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Apply mask and threshold
    high_corr_pairs = np.where((corr_matrix > threshold) & ~mask)
    
    correlations = {}
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        feature1 = corr_matrix.index[i]
        feature2 = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]
        
        if feature1 not in correlations:
            correlations[feature1] = []
        correlations[feature1].append((feature2, corr_value))
    
    return correlations


def suggest_feature_engineering(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Suggest feature engineering techniques based on data characteristics
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name
        
    Returns:
        List of suggested feature engineering techniques
    """
    suggestions = []
    
    # Analyze numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Check for skewed distributions
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            suggestions.append(f"Consider log transformation for '{col}' (skewness: {skewness:.2f})")
    
    # Check for categorical features with high cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 20:
            suggestions.append(f"Consider grouping rare categories in '{col}' ({unique_count} unique values)")
    
    # Check for potential date/time features
    for col in df.columns:
        if col != target_col and df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if any(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in sample_values):
                suggestions.append(f"Consider extracting date/time features from '{col}'")
    
    # Check for potential interaction features
    if len(numeric_cols) >= 2:
        suggestions.append("Consider creating interaction features between numerical variables")
    
    return suggestions


def calculate_feature_stability(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Population Stability Index (PSI) for features
    
    Args:
        df: DataFrame to analyze
        n_bins: Number of bins for discretization
        
    Returns:
        Dict mapping feature names to PSI values
    """
    psi_values = {}
    
    # Split data into two halves for comparison
    split_point = len(df) // 2
    df_base = df.iloc[:split_point]
    df_test = df.iloc[split_point:]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            # Create bins based on base data
            _, bins = pd.cut(df_base[col], bins=n_bins, retbins=True, duplicates='drop')
            
            # Calculate distributions
            base_dist = pd.cut(df_base[col], bins=bins, include_lowest=True).value_counts(normalize=True)
            test_dist = pd.cut(df_test[col], bins=bins, include_lowest=True).value_counts(normalize=True)
            
            # Align distributions
            base_dist = base_dist.reindex(test_dist.index, fill_value=0.001)
            test_dist = test_dist.fillna(0.001)
            
            # Calculate PSI
            psi = np.sum((test_dist - base_dist) * np.log(test_dist / base_dist))
            psi_values[col] = psi
            
        except Exception as e:
            logger.warning(f"Could not calculate PSI for {col}: {str(e)}")
            psi_values[col] = np.nan
    
    return psi_values


def identify_categorical_encoding_strategy(df: pd.DataFrame) -> Dict[str, str]:
    """
    Suggest encoding strategy for categorical features
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict mapping column names to suggested encoding strategies
    """
    strategies = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if unique_count == 2:
            strategies[col] = "label_encoding"
        elif unique_count <= 10:
            strategies[col] = "one_hot_encoding"
        elif unique_count <= 50:
            strategies[col] = "target_encoding"
        else:
            strategies[col] = "frequency_encoding"
    
    return strategies


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Validate train-test split for data leakage and distribution shifts
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dict containing validation results
    """
    validation_results = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X_train.shape[1],
        "warnings": []
    }
    
    # Check for data leakage (identical rows)
    if hasattr(X_train, 'merge'):
        duplicates = X_train.merge(X_test, how='inner')
        if len(duplicates) > 0:
            validation_results["warnings"].append(f"Found {len(duplicates)} identical rows between train and test")
    
    # Check target distribution
    if hasattr(y_train, 'value_counts'):
        # Classification
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        # Calculate distribution difference
        common_classes = set(train_dist.index) & set(test_dist.index)
        if len(common_classes) < len(train_dist.index):
            validation_results["warnings"].append("Some classes missing in test set")
        
        # Check for significant distribution shifts
        for cls in common_classes:
            diff = abs(train_dist[cls] - test_dist[cls])
            if diff > 0.1:  # 10% threshold
                validation_results["warnings"].append(f"Class '{cls}' distribution shift: {diff:.2f}")
    
    return validation_results


def calculate_model_complexity_score(model: Any) -> Dict[str, Any]:
    """
    Calculate complexity score for a model
    
    Args:
        model: Trained model
        
    Returns:
        Dict containing complexity metrics
    """
    complexity = {
        "model_type": type(model).__name__,
        "complexity_score": 0,
        "interpretability": "unknown",
        "parameters": 0
    }
    
    # Model-specific complexity calculation
    if hasattr(model, 'n_estimators'):
        # Tree-based ensemble
        complexity["parameters"] = getattr(model, 'n_estimators', 0)
        complexity["complexity_score"] = min(complexity["parameters"] / 100, 1.0)
        complexity["interpretability"] = "medium"
    elif hasattr(model, 'coef_'):
        # Linear model
        if hasattr(model.coef_, 'shape'):
            complexity["parameters"] = np.prod(model.coef_.shape)
        else:
            complexity["parameters"] = len(model.coef_)
        complexity["complexity_score"] = min(complexity["parameters"] / 1000, 1.0)
        complexity["interpretability"] = "high"
    elif hasattr(model, 'max_depth'):
        # Single decision tree
        depth = getattr(model, 'max_depth', 10)
        complexity["parameters"] = 2 ** depth if depth else 1024
        complexity["complexity_score"] = min(depth / 20, 1.0) if depth else 0.5
        complexity["interpretability"] = "high"
    
    return complexity


def generate_learning_curves(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Generate learning curves for model evaluation
    
    Args:
        estimator: Model to evaluate
        X: Features
        y: Target
        cv: Cross-validation folds
        train_sizes: Training sizes to evaluate
        
    Returns:
        Dict containing learning curve data
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        
        return {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "val_scores_mean": np.mean(val_scores, axis=1),
            "val_scores_std": np.std(val_scores, axis=1)
        }
    except Exception as e:
        logger.error(f"Error generating learning curves: {str(e)}")
        return {}


def calculate_prediction_intervals(
    model: Any,
    X: pd.DataFrame,
    confidence: float = 0.95
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate prediction intervals for regression models
    
    Args:
        model: Trained regression model
        X: Features for prediction
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) or None if not applicable
    """
    try:
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            # For models with uncertainty estimation
            predictions = model.predict(X)
            
            # Simple approach using standard deviation
            if hasattr(model, 'estimators_'):
                # For ensemble methods
                all_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                std_predictions = np.std(all_predictions, axis=0)
                
                # Calculate confidence intervals
                alpha = 1 - confidence
                z_score = 1.96  # For 95% confidence
                
                lower_bounds = predictions - z_score * std_predictions
                upper_bounds = predictions + z_score * std_predictions
                
                return lower_bounds, upper_bounds
        
        return None
    except Exception as e:
        logger.error(f"Error calculating prediction intervals: {str(e)}")
        return None


def assess_model_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.DataFrame
) -> Dict[str, Any]:
    """
    Assess model fairness across different groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Features to check fairness across
        
    Returns:
        Dict containing fairness metrics
    """
    fairness_metrics = {}
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        for feature in sensitive_features.columns:
            feature_fairness = {}
            
            # Calculate metrics for each group
            for group in sensitive_features[feature].unique():
                if pd.isna(group):
                    continue
                    
                group_mask = sensitive_features[feature] == group
                if group_mask.sum() < 10:  # Skip small groups
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                feature_fairness[str(group)] = {
                    "accuracy": accuracy_score(group_y_true, group_y_pred),
                    "precision": precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                    "sample_size": len(group_y_true)
                }
            
            # Calculate fairness metrics
            if len(feature_fairness) > 1:
                accuracies = [metrics["accuracy"] for metrics in feature_fairness.values()]
                fairness_metrics[feature] = {
                    "group_metrics": feature_fairness,
                    "accuracy_disparity": max(accuracies) - min(accuracies),
                    "overall_accuracy": overall_accuracy
                }
    
    except Exception as e:
        logger.error(f"Error assessing model fairness: {str(e)}")
    
    return fairness_metrics


def generate_model_summary(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str
) -> Dict[str, Any]:
    """
    Generate comprehensive model summary
    
    Args:
        model: Trained model
        X: Features
        y: Target
        task_type: 'classification' or 'regression'
        
    Returns:
        Dict containing model summary
    """
    summary = {
        "model_type": type(model).__name__,
        "task_type": task_type,
        "feature_count": X.shape[1],
        "training_samples": X.shape[0],
        "model_size": "unknown",
        "complexity": calculate_model_complexity_score(model),
        "interpretability_score": 0
    }
    
    # Try to estimate model size
    try:
        import pickle
        model_bytes = pickle.dumps(model)
        summary["model_size"] = f"{len(model_bytes) / 1024:.2f} KB"
    except Exception:
        pass
    
    # Calculate interpretability score
    if hasattr(model, 'feature_importances_'):
        summary["interpretability_score"] = 0.8
    elif hasattr(model, 'coef_'):
        summary["interpretability_score"] = 0.9
    else:
        summary["interpretability_score"] = 0.3
    
    return summary
