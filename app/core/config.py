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
    ALLOWED_EXTENSIONS: set = {".csv", ".xlsx", ".xls"}
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
