"""
Model Evaluation Routes
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.ml_service import MLService
from app.services.visualization_service import VisualizationService
from app.models.schemas import EvaluationResponse, ModelComparison
from app.core.exceptions import AutoMLException

router = APIRouter()
ml_service = MLService()
viz_service = VisualizationService()


@router.get("/evaluation/{session_id}", response_model=EvaluationResponse)
async def get_evaluation_results(session_id: str):
    """
    Get evaluation results for all trained models
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        EvaluationResponse: Comprehensive evaluation results
    """
    try:
        results = await ml_service.get_evaluation_results(session_id)
        if not results:
            raise HTTPException(status_code=404, detail="No evaluation results found")
        
        return results
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/best-model")
async def get_best_model(session_id: str):
    """
    Get the best performing model for the session
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Information about the best model
    """
    try:
        best_model_info = await ml_service.get_best_model_info(session_id)
        if not best_model_info:
            raise HTTPException(status_code=404, detail="No trained models found")
        
        return best_model_info
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/compare", response_model=ModelComparison)
async def compare_models(
    session_id: str,
    models: Optional[List[str]] = Query(None, description="List of model names to compare")
):
    """
    Compare performance of multiple models
    
    Args:
        session_id: Unique session identifier
        models: List of model names to compare (default: all models)
        
    Returns:
        ModelComparison: Side-by-side comparison of model performance
    """
    try:
        comparison = await ml_service.compare_models(session_id, models)
        if not comparison:
            raise HTTPException(status_code=404, detail="No models to compare")
        
        return comparison
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/metrics/{model_name}")
async def get_model_metrics(session_id: str, model_name: str):
    """
    Get detailed metrics for a specific model
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model
        
    Returns:
        Detailed metrics for the specified model
    """
    try:
        metrics = await ml_service.get_model_metrics(session_id, model_name)
        if not metrics:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return metrics
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/feature-importance")
async def get_feature_importance(
    session_id: str,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)"),
    top_n: int = Query(10, description="Number of top features to return")
):
    """
    Get feature importance for a model
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model (default: best model)
        top_n: Number of top features to return
        
    Returns:
        Feature importance information
    """
    try:
        importance = await ml_service.get_feature_importance(session_id, model_name, top_n)
        if not importance:
            raise HTTPException(status_code=404, detail="Feature importance not available")
        
        return importance
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/visualizations")
async def get_visualizations(
    session_id: str,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)"),
    viz_types: Optional[List[str]] = Query(None, description="Types of visualizations to generate")
):
    """
    Generate and return visualization URLs
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model (default: best model)
        viz_types: List of visualization types to generate
        
    Returns:
        URLs to generated visualizations
    """
    try:
        visualizations = await viz_service.generate_visualizations(
            session_id, model_name, viz_types
        )
        
        return visualizations
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/confusion-matrix")
async def get_confusion_matrix(
    session_id: str,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)")
):
    """
    Get confusion matrix for classification models
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model (default: best model)
        
    Returns:
        Confusion matrix data and visualization URL
    """
    try:
        confusion_matrix = await viz_service.generate_confusion_matrix(session_id, model_name)
        if not confusion_matrix:
            raise HTTPException(
                status_code=400,
                detail="Confusion matrix not available for this model type"
            )
        
        return confusion_matrix
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/{session_id}/roc-curve")
async def get_roc_curve(
    session_id: str,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)")
):
    """
    Get ROC curve for classification models
    
    Args:
        session_id: Unique session identifier
        model_name: Name of the model (default: best model)
        
    Returns:
        ROC curve data and visualization URL
    """
    try:
        roc_curve = await viz_service.generate_roc_curve(session_id, model_name)
        if not roc_curve:
            raise HTTPException(
                status_code=400,
                detail="ROC curve not available for this model type"
            )
        
        return roc_curve
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/{session_id}/predict")
async def make_prediction(
    session_id: str,
    prediction_data: dict,
    model_name: Optional[str] = Query(None, description="Model name (default: best model)")
):
    """
    Make predictions using a trained model
    
    Args:
        session_id: Unique session identifier
        prediction_data: Input data for prediction
        model_name: Name of the model (default: best model)
        
    Returns:
        Prediction results
    """
    try:
        predictions = await ml_service.make_predictions(
            session_id, prediction_data, model_name
        )
        
        return predictions
        
    except AutoMLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
