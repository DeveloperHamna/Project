"""
File Upload and Dataset Preview Routes
"""
import os
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.data_service import DataService
from app.models.schemas import DatasetPreview, TaskDetectionResponse
from app.core.exceptions import AutoMLException
from app.utils.file_utils import validate_file_extension, save_uploaded_file

router = APIRouter()
data_service = DataService()


@router.post("/upload", response_model=DatasetPreview)
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(None, description="Override target column name")
):
    """
    Upload a dataset file and get preview information
    
    Args:
        file: CSV or Excel file containing the dataset
        target_column: Optional override for target column (default: last column)
        
    Returns:
        DatasetPreview: Basic information about the uploaded dataset
    """
    try:
        # Validate file
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only CSV and Excel files are supported."
            )
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = await save_uploaded_file(file, session_id)
        
        # Load and preview dataset
        preview = await data_service.load_and_preview_dataset(
            file_path=file_path,
            session_id=session_id,
            target_column=target_column
        )
        
        return preview
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dataset/{session_id}/preview", response_model=DatasetPreview)
async def get_dataset_preview(session_id: str):
    """
    Get preview of an already uploaded dataset
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        DatasetPreview: Basic information about the dataset
    """
    try:
        preview = await data_service.get_dataset_preview(session_id)
        if not preview:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return preview
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/dataset/{session_id}/detect-task", response_model=TaskDetectionResponse)
async def detect_task_type(
    session_id: str,
    target_column: Optional[str] = Query(None, description="Override target column name")
):
    """
    Detect whether the dataset represents a classification or regression task
    
    Args:
        session_id: Unique session identifier
        target_column: Optional override for target column
        
    Returns:
        TaskDetectionResponse: Detected task type and analysis
    """
    try:
        task_info = await data_service.detect_task_type(session_id, target_column)
        return task_info
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dataset/{session_id}/columns")
async def get_dataset_columns(session_id: str):
    """
    Get column information for the dataset
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict containing column names and types
    """
    try:
        columns_info = await data_service.get_columns_info(session_id)
        if not columns_info:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return columns_info
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/dataset/{session_id}")
async def delete_dataset(session_id: str):
    """
    Delete uploaded dataset and associated files
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Success message
    """
    try:
        await data_service.cleanup_session(session_id)
        return {"message": "Dataset deleted successfully"}
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
