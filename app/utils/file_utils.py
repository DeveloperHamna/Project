"""
File utility functions for handling uploads and file operations
"""
import os
import shutil
import pandas as pd
from typing import Optional, List
from pathlib import Path
import magic
import logging
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import FileProcessingError

logger = logging.getLogger(__name__)


def validate_file_extension(filename: str) -> bool:
    """
    Validate file extension against allowed types
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """
    Validate file size against maximum allowed size
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        bool: True if size is within limits, False otherwise
    """
    return file_size <= settings.MAX_UPLOAD_SIZE


def detect_file_type(file_path: str) -> str:
    """
    Detect file type using python-magic
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Detected file type
    """
    try:
        mime_type = magic.from_file(file_path, mime=True)
        return mime_type
    except Exception as e:
        logger.warning(f"Could not detect file type: {str(e)}")
        # Fallback to extension-based detection
        extension = Path(file_path).suffix.lower()
        if extension in ['.csv']:
            return 'text/csv'
        elif extension in ['.xlsx', '.xls']:
            return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            return 'application/octet-stream'


def read_file_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read file into pandas DataFrame
    
    Args:
        file_path: Path to the file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileProcessingError: If file cannot be read
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                raise FileProcessingError("Could not read CSV file with any supported encoding")
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise FileProcessingError(f"Unsupported file format: {file_extension}")
        
        # Basic validation
        if df.empty:
            raise FileProcessingError("File is empty or contains no data")
        
        if df.shape[1] < 2:
            raise FileProcessingError("Dataset must have at least 2 columns (features and target)")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise FileProcessingError("File is empty or contains no data")
    except pd.errors.ParserError as e:
        raise FileProcessingError(f"Error parsing file: {str(e)}")
    except Exception as e:
        raise FileProcessingError(f"Error reading file: {str(e)}")


async def save_uploaded_file(file: UploadFile, session_id: str) -> str:
    """
    Save uploaded file to the uploads directory
    
    Args:
        file: Uploaded file object
        session_id: Unique session identifier
        
    Returns:
        str: Path to the saved file
        
    Raises:
        FileProcessingError: If file cannot be saved
    """
    try:
        # Validate file extension
        if not validate_file_extension(file.filename):
            raise FileProcessingError(f"Invalid file extension. Allowed: {settings.ALLOWED_EXTENSIONS}")
        
        # Create uploads directory if it doesn't exist
        uploads_dir = Path(settings.UPLOAD_DIR)
        uploads_dir.mkdir(exist_ok=True)
        
        # Create session-specific directory
        session_dir = uploads_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Generate safe filename
        safe_filename = secure_filename(file.filename)
        file_path = session_dir / safe_filename
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        if not validate_file_size(len(content)):
            raise FileProcessingError(f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes")
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"File saved successfully: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise FileProcessingError(f"Failed to save file: {str(e)}")


def secure_filename(filename: str) -> str:
    """
    Generate a secure filename by removing dangerous characters
    
    Args:
        filename: Original filename
        
    Returns:
        str: Secure filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    import re
    filename = re.sub(r'[^\w\s\-_\.]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def get_file_info(file_path: str) -> dict:
    """
    Get detailed information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: File information
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileProcessingError(f"File does not exist: {file_path}")
        
        stat = file_path.stat()
        
        return {
            "filename": file_path.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "extension": file_path.suffix.lower(),
            "mime_type": detect_file_type(str(file_path))
        }
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise FileProcessingError(f"Failed to get file info: {str(e)}")


def cleanup_session_files(session_id: str):
    """
    Clean up all files associated with a session
    
    Args:
        session_id: Session identifier
    """
    try:
        # Clean up uploads
        uploads_dir = Path(settings.UPLOAD_DIR) / session_id
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir)
        
        # Clean up models
        models_dir = Path(settings.MODELS_DIR) / session_id
        if models_dir.exists():
            shutil.rmtree(models_dir)
        
        # Clean up visualizations
        viz_dir = Path(settings.VISUALIZATIONS_DIR) / session_id
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        
        # Clean up reports
        reports_dir = Path(settings.REPORTS_DIR) / session_id
        if reports_dir.exists():
            shutil.rmtree(reports_dir)
        
        logger.info(f"Cleaned up files for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up session files: {str(e)}")


def get_directory_size(directory: str) -> int:
    """
    Get total size of a directory in bytes
    
    Args:
        directory: Directory path
        
    Returns:
        int: Total size in bytes
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    except Exception as e:
        logger.error(f"Error calculating directory size: {str(e)}")
        return 0


def create_directory_structure():
    """Create necessary directory structure for the application"""
    directories = [
        settings.UPLOAD_DIR,
        settings.MODELS_DIR,
        settings.VISUALIZATIONS_DIR,
        settings.REPORTS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_csv_format(file_path: str) -> bool:
    """
    Validate CSV file format and structure
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        bool: True if valid CSV format
    """
    try:
        # Try to read just the first few rows to validate structure
        df = pd.read_csv(file_path, nrows=5)
        
        # Check if it has data
        if df.empty:
            return False
        
        # Check if it has at least 2 columns
        if df.shape[1] < 2:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"CSV validation error: {str(e)}")
        return False


def validate_excel_format(file_path: str) -> bool:
    """
    Validate Excel file format and structure
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        bool: True if valid Excel format
    """
    try:
        # Try to read just the first few rows to validate structure
        df = pd.read_excel(file_path, nrows=5)
        
        # Check if it has data
        if df.empty:
            return False
        
        # Check if it has at least 2 columns
        if df.shape[1] < 2:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Excel validation error: {str(e)}")
        return False


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats
    
    Returns:
        List[str]: Supported file extensions
    """
    return list(settings.ALLOWED_EXTENSIONS)


def estimate_memory_usage(file_path: str) -> dict:
    """
    Estimate memory usage for loading the file
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: Memory usage estimation
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # Rough estimates based on file type
        if file_path.endswith('.csv'):
            # CSV files typically use 2-3x their file size in memory
            estimated_memory = file_size * 2.5
        elif file_path.endswith(('.xlsx', '.xls')):
            # Excel files typically use 3-4x their file size in memory
            estimated_memory = file_size * 3.5
        else:
            estimated_memory = file_size * 2
        
        return {
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "estimated_memory_mb": round(estimated_memory / (1024 * 1024), 2),
            "recommendation": "acceptable" if estimated_memory < 1024 * 1024 * 1024 else "large_file"
        }
        
    except Exception as e:
        logger.error(f"Error estimating memory usage: {str(e)}")
        return {
            "file_size_mb": 0,
            "estimated_memory_mb": 0,
            "recommendation": "unknown"
        }
