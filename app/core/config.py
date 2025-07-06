"""
Configuration Settings
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Credit Scoring Model Backend Service"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: set = {".csv", ".xlsx", ".xls", ".json", ".tsv", ".zip"}
    UPLOAD_DIR: str = "uploads"
    
    # Model Configuration
    MODELS_DIR: str = "models"
    VISUALIZATIONS_DIR: str = "visualizations"
    REPORTS_DIR: str = "reports"
    
    # ML Configuration
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    MAX_FEATURES_FOR_SHAP: int = 20
    
    # Preprocessing Configuration
    AUTO_DETECT_ENCODING: bool = True
    AUTO_DETECT_DELIMITER: bool = True
    MISSING_VALUE_THRESHOLD: float = 0.5  # Drop columns with >50% missing values
    LOW_VARIANCE_THRESHOLD: float = 0.01  # Drop features with variance < 0.01
    CORRELATION_THRESHOLD: float = 0.95  # Drop highly correlated features
    OUTLIER_DETECTION_METHOD: str = "IQR"  # IQR or Z-Score
    OUTLIER_THRESHOLD: float = 3.0  # Standard deviations for outlier detection
    
    # Imbalanced Data Configuration
    IMBALANCE_THRESHOLD: float = 0.1  # Minimum class proportion to consider balanced
    DEFAULT_SAMPLING_STRATEGY: str = "auto"  # auto, minority, majority, all
    SMOTE_K_NEIGHBORS: int = 5
    
    # Feature Engineering Configuration
    AUTO_FEATURE_ENGINEERING: bool = True
    CREATE_POLYNOMIAL_FEATURES: bool = False
    POLYNOMIAL_DEGREE: int = 2
    CREATE_INTERACTION_FEATURES: bool = True
    
    # Scaling Configuration
    DEFAULT_SCALER: str = "StandardScaler"  # StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    
    # Dataset Split Configuration
    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    
    # Performance Configuration
    N_JOBS: int = -1  # Use all available cores
    JOBLIB_CACHE_SIZE: str = "1G"
    
    # Visualization Configuration
    FIGURE_DPI: int = 300
    FIGURE_SIZE: tuple = (10, 8)
    
    # Report Configuration
    REPORT_TEMPLATE_DIR: str = "templates"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "automl-secret-key-change-in-production")
    
    # External Services (if needed)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Global settings instance
settings = Settings()
