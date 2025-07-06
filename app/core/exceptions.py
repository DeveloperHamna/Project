"""
Custom Exceptions and Exception Handlers
"""
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)


class AutoMLException(Exception):
    """Base exception for AutoML operations"""
    
    def __init__(self, message: str, error_code: str = "AUTOML_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DataProcessingError(AutoMLException):
    """Exception for data processing errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_PROCESSING_ERROR")


class ModelTrainingError(AutoMLException):
    """Exception for model training errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "MODEL_TRAINING_ERROR")


class ValidationError(AutoMLException):
    """Exception for validation errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")


class FileProcessingError(AutoMLException):
    """Exception for file processing errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "FILE_PROCESSING_ERROR")


class VisualizationError(AutoMLException):
    """Exception for visualization errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "VISUALIZATION_ERROR")


class ReportGenerationError(AutoMLException):
    """Exception for report generation errors"""
    
    def __init__(self, message: str):
        super().__init__(message, "REPORT_GENERATION_ERROR")


def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers for the FastAPI app"""
    
    @app.exception_handler(AutoMLException)
    async def automl_exception_handler(request: Request, exc: AutoMLException):
        """Handle AutoML specific exceptions"""
        logger.error(f"AutoML Exception: {exc.message}")
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "type": "automl_error"
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        logger.error(f"Validation Error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "type": "validation_error"
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        logger.error(f"HTTP Exception: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "type": "http_error"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "type": "internal_error"
            }
        )
