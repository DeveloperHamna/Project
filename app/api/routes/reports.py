"""
Report Generation Routes
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.report_service import ReportService
from app.models.schemas import ReportRequest, ReportResponse
from app.core.exceptions import AutoMLException

router = APIRouter()
report_service = ReportService()


@router.post("/reports/{session_id}/generate", response_model=ReportResponse)
async def generate_report(
    session_id: str,
    format: str = Query("pdf", description="Report format (pdf, html, json)"),
    include_visualizations: bool = Query(True, description="Include visualizations in report"),
    include_raw_data: bool = Query(False, description="Include raw data sample in report")
):
    """
    Generate a comprehensive report
    
    Args:
        session_id: Unique session identifier
        format: Report format (pdf, html, json)
        include_visualizations: Whether to include visualizations
        include_raw_data: Whether to include raw data sample
        
    Returns:
        ReportResponse: Report generation information
    """
    try:
        if format not in ["pdf", "html", "json"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Supported formats: pdf, html, json"
            )
        
        report_info = await report_service.generate_report(
            session_id=session_id,
            format=format,
            include_visualizations=include_visualizations,
            include_raw_data=include_raw_data
        )
        
        return report_info
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/reports/{session_id}/download/{report_id}")
async def download_report(session_id: str, report_id: str):
    """
    Download a generated report
    
    Args:
        session_id: Unique session identifier
        report_id: Report identifier
        
    Returns:
        File download response
    """
    try:
        file_path = await report_service.get_report_file_path(session_id, report_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get file info
        report_info = await report_service.get_report_info(session_id, report_id)
        filename = report_info.get("filename", f"automl_report_{report_id}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/reports/{session_id}/summary")
async def get_report_summary(session_id: str):
    """
    Get a summary of the analysis for quick overview
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Summary of the analysis results
    """
    try:
        summary = await report_service.get_analysis_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Analysis summary not available")
        
        return summary
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/reports/{session_id}/export-model")
async def export_model(
    session_id: str,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)"),
    format: str = Query("joblib", description="Export format (joblib, pickle)")
):
    """
    Export a trained model for deployment
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model (default: best model)
        format: Export format
        
    Returns:
        Model file download
    """
    try:
        if format not in ["joblib", "pickle"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Supported formats: joblib, pickle"
            )
        
        file_path = await report_service.export_model(session_id, model_name, format)
        if not file_path:
            raise HTTPException(status_code=404, detail="Model not found")
        
        filename = f"model_{model_name or 'best'}_{session_id}.{format}"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/reports/{session_id}/preprocessing-pipeline")
async def export_preprocessing_pipeline(session_id: str):
    """
    Export the preprocessing pipeline
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Preprocessing pipeline file download
    """
    try:
        file_path = await report_service.export_preprocessing_pipeline(session_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Preprocessing pipeline not found")
        
        filename = f"preprocessing_pipeline_{session_id}.joblib"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/reports/{session_id}/list")
async def list_reports(session_id: str):
    """
    List all generated reports for a session
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        List of available reports
    """
    try:
        reports = await report_service.list_reports(session_id)
        return {"reports": reports}
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/reports/{session_id}/{report_id}")
async def delete_report(session_id: str, report_id: str):
    """
    Delete a specific report
    
    Args:
        session_id: Unique session identifier
        report_id: Report identifier
        
    Returns:
        Success message
    """
    try:
        await report_service.delete_report(session_id, report_id)
        return {"message": "Report deleted successfully"}
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
