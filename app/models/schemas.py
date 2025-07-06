"""
Pydantic schemas for request/response validation
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class DatasetPreview(BaseModel):
    """Schema for dataset preview response"""
    session_id: str
    filename: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    target_column: str
    sample_data: List[Dict[str, Any]]
    missing_values: Dict[str, int]
    memory_usage: int
    numerical_columns: List[str]
    categorical_columns: List[str]


class TaskDetectionResponse(BaseModel):
    """Schema for task detection response"""
    session_id: str
    task_type: str = Field(..., description="Either 'classification' or 'regression'")
    target_column: str
    target_stats: Dict[str, Any]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    feature_count: int
    feature_analysis: Dict[str, Any]
    recommendations: List[str]


class TrainingRequest(BaseModel):
    """Schema for training request"""
    target_column: Optional[str] = None
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0)
    cv_folds: int = Field(5, ge=3, le=10)
    scoring_metric: Optional[str] = None
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError('test_size must be between 0.1 and 0.5')
        return v


class TrainingResponse(BaseModel):
    """Schema for training response"""
    job_id: str
    status: str
    message: str


class TrainingStatus(BaseModel):
    """Schema for training status response"""
    job_id: str
    status: str
    progress: int = Field(..., ge=0, le=100)
    message: str
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    estimated_time_remaining: Optional[int] = None


class ModelMetrics(BaseModel):
    """Schema for model metrics"""
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    cv_score: float
    parameters: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Schema for evaluation response"""
    session_id: str
    task_type: str
    best_model: str
    model_results: Dict[str, Any]
    training_timestamp: str


class ModelComparison(BaseModel):
    """Schema for model comparison response"""
    session_id: str
    task_type: str
    models: Dict[str, Dict[str, Any]]
    best_model: str


class FeatureImportance(BaseModel):
    """Schema for feature importance response"""
    model_name: str
    feature_importance: Dict[str, float]
    total_features: int


class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    input_data: Dict[str, Any]
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predictions: List[Union[float, int, str]]
    prediction_proba: Optional[List[List[float]]] = None
    model_name: str
    input_features: List[str]


class ReportRequest(BaseModel):
    """Schema for report generation request"""
    format: str = Field("pdf", description="Report format: pdf, html, json")
    include_visualizations: bool = Field(True)
    include_raw_data: bool = Field(False)
    
    @validator('format')
    def validate_format(cls, v):
        if v not in ['pdf', 'html', 'json']:
            raise ValueError('format must be one of: pdf, html, json')
        return v


class ReportResponse(BaseModel):
    """Schema for report generation response"""
    report_id: str
    format: str
    filename: str
    download_url: str
    generated_at: str


class VisualizationResponse(BaseModel):
    """Schema for visualization response"""
    session_id: str
    visualizations: Dict[str, str]
    model_name: str


class ConfusionMatrixResponse(BaseModel):
    """Schema for confusion matrix response"""
    visualization_url: str
    matrix_data: List[List[int]]
    class_labels: List[str]
    model_name: str


class ROCCurveResponse(BaseModel):
    """Schema for ROC curve response"""
    visualization_url: str
    fpr: List[float]
    tpr: List[float]
    auc_score: float
    model_name: str


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    message: str
    type: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Schema for health check response"""
    status: str
    service: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionInfo(BaseModel):
    """Schema for session information"""
    session_id: str
    created_at: str
    dataset_filename: str
    status: str
    task_type: Optional[str] = None
    target_column: Optional[str] = None


class DataQualityReport(BaseModel):
    """Schema for data quality report"""
    session_id: str
    total_rows: int
    total_columns: int
    missing_values_summary: Dict[str, int]
    duplicate_rows: int
    numerical_columns_stats: Dict[str, Dict[str, float]]
    categorical_columns_stats: Dict[str, Dict[str, Any]]
    data_quality_score: float
    recommendations: List[str]


class ModelExportResponse(BaseModel):
    """Schema for model export response"""
    model_name: str
    export_format: str
    file_size: int
    download_url: str
    exported_at: str


class TrainingProgress(BaseModel):
    """Schema for training progress updates"""
    session_id: str
    current_step: str
    progress_percentage: int
    estimated_remaining_time: Optional[int] = None
    current_model: Optional[str] = None
    completed_models: List[str]
    message: str


class DatasetStatistics(BaseModel):
    """Schema for dataset statistics"""
    session_id: str
    numerical_stats: Dict[str, Dict[str, float]]
    categorical_stats: Dict[str, Dict[str, Any]]
    correlation_matrix: Dict[str, Dict[str, float]]
    outliers_detected: Dict[str, int]
    class_distribution: Optional[Dict[str, int]] = None


class ModelInsights(BaseModel):
    """Schema for model insights"""
    model_name: str
    task_type: str
    feature_importance: Dict[str, float]
    model_complexity: str
    training_time: float
    prediction_speed: float
    memory_usage: int
    strengths: List[str]
    limitations: List[str]


class CrossValidationResults(BaseModel):
    """Schema for cross-validation results"""
    model_name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    best_fold: int
    worst_fold: int
    score_distribution: Dict[str, float]


class HyperparameterTuningResults(BaseModel):
    """Schema for hyperparameter tuning results"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    param_grid: Dict[str, List[Any]]
    tuning_method: str
    total_iterations: int
    best_iteration: int
    param_importance: Optional[Dict[str, float]] = None


class ModelInterpretation(BaseModel):
    """Schema for model interpretation"""
    model_name: str
    interpretation_method: str
    global_importance: Dict[str, float]
    sample_explanations: List[Dict[str, Any]]
    interpretation_plots: List[str]
    summary: str


class DeploymentPackage(BaseModel):
    """Schema for deployment package"""
    session_id: str
    model_name: str
    package_format: str
    includes_preprocessing: bool
    includes_feature_names: bool
    deployment_instructions: str
    requirements: List[str]
    download_url: str
    created_at: str
