"""
Model Training Routes
"""
import asyncio
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from app.services.ml_service import MLService
from app.models.schemas import TrainingRequest, TrainingResponse, TrainingStatus
from app.core.exceptions import AutoMLException

router = APIRouter()
ml_service = MLService()

# In-memory storage for training status (in production, use Redis or database)
training_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/train/{session_id}", response_model=TrainingResponse)
async def start_training(
    session_id: str,
    background_tasks: BackgroundTasks,
    target_column: Optional[str] = Query(None, description="Target column name"),
    test_size: float = Query(0.2, description="Test set size (0.1-0.5)"),
    random_state: int = Query(42, description="Random state for reproducibility"),
    cv_folds: int = Query(5, description="Cross-validation folds"),
    scoring_metric: Optional[str] = Query(None, description="Scoring metric for model selection")
):
    """
    Start model training process
    
    Args:
        session_id: Unique session identifier
        target_column: Target column name
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        cv_folds: Number of cross-validation folds
        scoring_metric: Scoring metric for model selection
        
    Returns:
        TrainingResponse: Training job information
    """
    try:
        # Validate test_size
        if not 0.1 <= test_size <= 0.5:
            raise HTTPException(
                status_code=400,
                detail="Test size must be between 0.1 and 0.5"
            )
        
        # Initialize training job
        job_id = f"training_{session_id}"
        training_jobs[job_id] = {
            "status": "initializing",
            "progress": 0,
            "message": "Initializing training process...",
            "session_id": session_id
        }
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            session_id=session_id,
            job_id=job_id,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric
        )
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            message="Training process started successfully"
        )
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/training/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get training job status
    
    Args:
        job_id: Training job identifier
        
    Returns:
        TrainingStatus: Current training status and progress
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        job_info = training_jobs[job_id]
        
        return TrainingStatus(
            job_id=job_id,
            status=job_info["status"],
            progress=job_info["progress"],
            message=job_info["message"],
            current_step=job_info.get("current_step"),
            total_steps=job_info.get("total_steps"),
            estimated_time_remaining=job_info.get("estimated_time_remaining")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/training/{job_id}/results")
async def get_training_results(job_id: str):
    """
    Get training results
    
    Args:
        job_id: Training job identifier
        
    Returns:
        Training results including model performance metrics
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        job_info = training_jobs[job_id]
        
        if job_info["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Training not completed. Current status: {job_info['status']}"
            )
        
        session_id = job_info["session_id"]
        results = await ml_service.get_training_results(session_id)
        
        return results
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/training/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a running training job
    
    Args:
        job_id: Training job identifier
        
    Returns:
        Success message
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        job_info = training_jobs[job_id]
        
        if job_info["status"] in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job_info['status']}"
            )
        
        # Update job status
        training_jobs[job_id]["status"] = "cancelled"
        training_jobs[job_id]["message"] = "Training cancelled by user"
        
        return {"message": "Training job cancelled successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def run_training_job(
    session_id: str,
    job_id: str,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    scoring_metric: Optional[str] = None
):
    """
    Background task for running the training job
    """
    try:
        # Update job status
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 5
        training_jobs[job_id]["message"] = "Loading and preprocessing data..."
        training_jobs[job_id]["current_step"] = "preprocessing"
        training_jobs[job_id]["total_steps"] = 5
        
        # Run training
        await ml_service.train_models(
            session_id=session_id,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric,
            progress_callback=lambda step, progress, message: update_job_progress(
                job_id, step, progress, message
            )
        )
        
        # Training completed
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["message"] = "Training completed successfully"
        training_jobs[job_id]["current_step"] = "completed"
        
    except Exception as e:
        # Training failed
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        training_jobs[job_id]["error"] = str(e)


def update_job_progress(job_id: str, step: str, progress: int, message: str):
    """Update training job progress"""
    if job_id in training_jobs:
        training_jobs[job_id]["progress"] = progress
        training_jobs[job_id]["message"] = message
        training_jobs[job_id]["current_step"] = step
