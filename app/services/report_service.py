"""
Report Service for generating comprehensive reports and model exports
"""
import os
import json
import uuid
import joblib
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
from jinja2 import Template
from weasyprint import HTML, CSS

from app.core.config import settings
from app.core.exceptions import ReportGenerationError
from app.models.schemas import ReportResponse
from app.services.ml_service import MLService
from app.services.data_service import DataService

logger = logging.getLogger(__name__)


class ReportService:
    """Service for generating reports and exporting models"""
    
    def __init__(self):
        self.ml_service = MLService()
        self.data_service = DataService()
        self.generated_reports: Dict[str, Dict[str, Any]] = {}
    
    async def generate_report(
        self,
        session_id: str,
        format: str = "pdf",
        include_visualizations: bool = True,
        include_raw_data: bool = False
    ) -> ReportResponse:
        """
        Generate a comprehensive AutoML report
        
        Args:
            session_id: Unique session identifier
            format: Report format (pdf, html, json)
            include_visualizations: Include visualizations in report
            include_raw_data: Include raw data sample
            
        Returns:
            ReportResponse: Report generation information
        """
        try:
            # Generate unique report ID
            report_id = str(uuid.uuid4())
            
            # Get session data
            session_data = await self.data_service.get_session_data(session_id)
            if not session_data:
                raise ReportGenerationError("Session not found")
            
            # Get training results
            training_results = await self.ml_service.get_training_results(session_id)
            if not training_results:
                raise ReportGenerationError("No training results found")
            
            # Prepare report data
            report_data = await self._prepare_report_data(
                session_id, session_data, training_results, include_raw_data
            )
            
            # Create reports directory
            reports_dir = os.path.join(settings.REPORTS_DIR, session_id)
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate report based on format
            if format == "pdf":
                file_path = await self._generate_pdf_report(
                    report_data, reports_dir, report_id, include_visualizations
                )
                filename = f"automl_report_{report_id}.pdf"
            elif format == "html":
                file_path = await self._generate_html_report(
                    report_data, reports_dir, report_id, include_visualizations
                )
                filename = f"automl_report_{report_id}.html"
            elif format == "json":
                file_path = await self._generate_json_report(
                    report_data, reports_dir, report_id
                )
                filename = f"automl_report_{report_id}.json"
            else:
                raise ReportGenerationError(f"Unsupported format: {format}")
            
            # Store report information
            self.generated_reports[f"{session_id}_{report_id}"] = {
                "session_id": session_id,
                "report_id": report_id,
                "format": format,
                "file_path": file_path,
                "filename": filename,
                "generated_at": datetime.now().isoformat(),
                "include_visualizations": include_visualizations,
                "include_raw_data": include_raw_data
            }
            
            return ReportResponse(
                report_id=report_id,
                format=format,
                filename=filename,
                download_url=f"/api/v1/reports/{session_id}/download/{report_id}",
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise ReportGenerationError(f"Failed to generate report: {str(e)}")
    
    async def _prepare_report_data(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        training_results: Dict[str, Any],
        include_raw_data: bool
    ) -> Dict[str, Any]:
        """Prepare comprehensive report data"""
        
        # Get dataset info
        df = session_data["dataframe"]
        
        # Get evaluation results
        evaluation_results = await self.ml_service.get_evaluation_results(session_id)
        
        # Get best model info
        best_model_info = await self.ml_service.get_best_model_info(session_id)
        
        # Get feature importance
        feature_importance = await self.ml_service.get_feature_importance(session_id)
        
        report_data = {
            "session_info": {
                "session_id": session_id,
                "generated_at": datetime.now().isoformat(),
                "dataset_filename": os.path.basename(session_data["file_path"])
            },
            "dataset_summary": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "target_column": session_data["target_column"],
                "task_type": training_results["task_type"],
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "numerical_features": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_features": df.select_dtypes(include=['object', 'category']).columns.tolist()
            },
            "preprocessing_steps": self._get_preprocessing_steps(training_results["task_type"]),
            "model_results": {
                "best_model": training_results["best_model"],
                "all_models": training_results["models"],
                "evaluation_metrics": evaluation_results.model_results if evaluation_results else {}
            },
            "best_model_details": best_model_info,
            "feature_importance": feature_importance,
            "training_parameters": training_results["training_params"],
            "visualizations": [],  # Will be populated if visualizations are included
            "recommendations": self._generate_model_recommendations(training_results, best_model_info)
        }
        
        # Add raw data sample if requested
        if include_raw_data:
            report_data["raw_data_sample"] = {
                "first_5_rows": df.head().to_dict(orient="records"),
                "last_5_rows": df.tail().to_dict(orient="records"),
                "statistical_summary": df.describe().to_dict()
            }
        
        return report_data
    
    def _get_preprocessing_steps(self, task_type: str) -> List[str]:
        """Get list of preprocessing steps performed"""
        steps = [
            "Missing value imputation using median for numerical features",
            "Missing value imputation using most frequent for categorical features",
            "One-hot encoding for categorical features",
            "Standard scaling for numerical features"
        ]
        
        if task_type == "classification":
            steps.append("Label encoding for target variable")
        
        return steps
    
    def _generate_model_recommendations(
        self,
        training_results: Dict[str, Any],
        best_model_info: Dict[str, Any]
    ) -> List[str]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        best_model = training_results["best_model"]
        task_type = training_results["task_type"]
        
        # Performance-based recommendations
        if task_type == "classification":
            f1_score = best_model_info["metrics"].get("f1", 0)
            if f1_score < 0.7:
                recommendations.append("Consider collecting more data or feature engineering to improve model performance")
            elif f1_score > 0.9:
                recommendations.append("Excellent model performance. Consider validating on additional test data")
        else:
            r2_score = best_model_info["metrics"].get("r2", 0)
            if r2_score < 0.7:
                recommendations.append("Consider feature engineering or trying ensemble methods")
            elif r2_score > 0.9:
                recommendations.append("High R² score indicates good model fit. Monitor for overfitting")
        
        # Model-specific recommendations
        if "xgboost" in best_model:
            recommendations.append("XGBoost selected as best model. Consider hyperparameter tuning for production")
        elif "random_forest" in best_model:
            recommendations.append("Random Forest provides good interpretability. Feature importance is reliable")
        elif "linear" in best_model:
            recommendations.append("Linear model suggests linear relationships. Check for feature interactions")
        
        return recommendations
    
    async def _generate_pdf_report(
        self,
        report_data: Dict[str, Any],
        reports_dir: str,
        report_id: str,
        include_visualizations: bool
    ) -> str:
        """Generate PDF report"""
        # Generate HTML first
        html_content = self._generate_html_content(report_data, include_visualizations)
        
        # Convert HTML to PDF
        html_file = HTML(string=html_content)
        css_file = CSS(string=self._get_pdf_styles())
        
        pdf_path = os.path.join(reports_dir, f"automl_report_{report_id}.pdf")
        html_file.write_pdf(pdf_path, stylesheets=[css_file])
        
        return pdf_path
    
    async def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        reports_dir: str,
        report_id: str,
        include_visualizations: bool
    ) -> str:
        """Generate HTML report"""
        html_content = self._generate_html_content(report_data, include_visualizations)
        
        html_path = os.path.join(reports_dir, f"automl_report_{report_id}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    async def _generate_json_report(
        self,
        report_data: Dict[str, Any],
        reports_dir: str,
        report_id: str
    ) -> str:
        """Generate JSON report"""
        json_path = os.path.join(reports_dir, f"automl_report_{report_id}.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return json_path
    
    def _generate_html_content(
        self,
        report_data: Dict[str, Any],
        include_visualizations: bool
    ) -> str:
        """Generate HTML content for the report"""
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>AutoML Report - {{ session_info.session_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin: 20px 0; }
        .section h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .recommendation { background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .best-model { background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AutoML Analysis Report</h1>
        <p>Generated on: {{ session_info.generated_at }}</p>
        <p>Dataset: {{ session_info.dataset_filename }}</p>
    </div>
    
    <div class="section">
        <h2>Dataset Summary</h2>
        <p><strong>Shape:</strong> {{ dataset_summary.shape[0] }} rows, {{ dataset_summary.shape[1] }} columns</p>
        <p><strong>Task Type:</strong> {{ dataset_summary.task_type|title }}</p>
        <p><strong>Target Column:</strong> {{ dataset_summary.target_column }}</p>
        <p><strong>Numerical Features:</strong> {{ dataset_summary.numerical_features|length }}</p>
        <p><strong>Categorical Features:</strong> {{ dataset_summary.categorical_features|length }}</p>
    </div>
    
    <div class="section">
        <h2>Preprocessing Steps</h2>
        <ul>
        {% for step in preprocessing_steps %}
            <li>{{ step }}</li>
        {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h2>Model Results</h2>
        <div class="best-model">
            <h3>Best Model: {{ model_results.best_model|title }}</h3>
            {% if best_model_details %}
            <div>
                <h4>Performance Metrics:</h4>
                {% for metric, value in best_model_details.metrics.items() %}
                <div class="metric">
                    <strong>{{ metric|title }}:</strong> {{ "%.4f"|format(value) }}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <h3>All Models Performance</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>CV Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
            {% for model_name, model_data in model_results.all_models.items() %}
                <tr>
                    <td>{{ model_name|title }}</td>
                    <td>{{ "%.4f"|format(model_data.cv_score) }}</td>
                    <td>{% if model_name == model_results.best_model %}🏆 Best{% else %}✓ Trained{% endif %}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    
    {% if feature_importance %}
    <div class="section">
        <h2>Feature Importance</h2>
        <p>Top {{ feature_importance.feature_importance|length }} most important features:</p>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
            </thead>
            <tbody>
            {% for feature, importance in feature_importance.feature_importance.items() %}
                <tr>
                    <td>{{ feature }}</td>
                    <td>{{ "%.6f"|format(importance) }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in recommendations %}
        <div class="recommendation">
            {{ recommendation }}
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Training Parameters</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
            {% for param, value in training_parameters.items() %}
                <tr>
                    <td>{{ param|title }}</td>
                    <td>{{ value }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """)
        
        return template.render(**report_data)
    
    def _get_pdf_styles(self) -> str:
        """Get CSS styles for PDF generation"""
        return """
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: 'Arial', sans-serif;
            font-size: 10pt;
            line-height: 1.4;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .section {
            margin: 15px 0;
        }
        .section h2 {
            color: #333;
            border-bottom: 1px solid #007bff;
            padding-bottom: 3px;
            font-size: 14pt;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 9pt;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 5px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .best-model {
            background-color: #d4edda;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        .metric {
            display: inline-block;
            margin: 5px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        """
    
    async def get_analysis_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get quick analysis summary"""
        try:
            # Get training results
            training_results = await self.ml_service.get_training_results(session_id)
            if not training_results:
                return None
            
            # Get best model info
            best_model_info = await self.ml_service.get_best_model_info(session_id)
            
            # Get session data
            session_data = await self.data_service.get_session_data(session_id)
            if not session_data:
                return None
            
            df = session_data["dataframe"]
            
            summary = {
                "session_id": session_id,
                "dataset_shape": df.shape,
                "task_type": training_results["task_type"],
                "target_column": training_results["target_column"],
                "best_model": training_results["best_model"],
                "model_performance": best_model_info["metrics"] if best_model_info else {},
                "total_models_trained": len(training_results["models"]),
                "training_completed_at": training_results["timestamp"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}")
            return None
    
    async def export_model(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        format: str = "joblib"
    ) -> Optional[str]:
        """Export trained model"""
        try:
            # Get training results
            training_results = await self.ml_service.get_training_results(session_id)
            if not training_results:
                return None
            
            if model_name is None:
                model_name = training_results["best_model"]
            
            # Load model from storage
            models_dir = os.path.join(settings.MODELS_DIR, session_id)
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            
            if not os.path.exists(model_path):
                return None
            
            # Create export directory
            export_dir = os.path.join(settings.REPORTS_DIR, session_id)
            os.makedirs(export_dir, exist_ok=True)
            
            # Export model in requested format
            if format == "joblib":
                export_path = os.path.join(export_dir, f"model_{model_name}.joblib")
                # Copy existing joblib file
                import shutil
                shutil.copy2(model_path, export_path)
            elif format == "pickle":
                export_path = os.path.join(export_dir, f"model_{model_name}.pkl")
                # Load and save as pickle
                model = joblib.load(model_path)
                with open(export_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                return None
            
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None
    
    async def export_preprocessing_pipeline(self, session_id: str) -> Optional[str]:
        """Export preprocessing pipeline"""
        try:
            # Load preprocessor
            models_dir = os.path.join(settings.MODELS_DIR, session_id)
            preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
            
            if not os.path.exists(preprocessor_path):
                return None
            
            # Create export directory
            export_dir = os.path.join(settings.REPORTS_DIR, session_id)
            os.makedirs(export_dir, exist_ok=True)
            
            # Copy preprocessor
            export_path = os.path.join(export_dir, "preprocessing_pipeline.joblib")
            import shutil
            shutil.copy2(preprocessor_path, export_path)
            
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting preprocessing pipeline: {str(e)}")
            return None
    
    async def get_report_file_path(self, session_id: str, report_id: str) -> Optional[str]:
        """Get file path for a generated report"""
        report_key = f"{session_id}_{report_id}"
        if report_key in self.generated_reports:
            return self.generated_reports[report_key]["file_path"]
        return None
    
    async def get_report_info(self, session_id: str, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report information"""
        report_key = f"{session_id}_{report_id}"
        if report_key in self.generated_reports:
            return self.generated_reports[report_key]
        return None
    
    async def list_reports(self, session_id: str) -> List[Dict[str, Any]]:
        """List all reports for a session"""
        reports = []
        for report_key, report_info in self.generated_reports.items():
            if report_info["session_id"] == session_id:
                reports.append({
                    "report_id": report_info["report_id"],
                    "format": report_info["format"],
                    "filename": report_info["filename"],
                    "generated_at": report_info["generated_at"],
                    "download_url": f"/api/v1/reports/{session_id}/download/{report_info['report_id']}"
                })
        return reports
    
    async def delete_report(self, session_id: str, report_id: str):
        """Delete a specific report"""
        report_key = f"{session_id}_{report_id}"
        if report_key in self.generated_reports:
            report_info = self.generated_reports[report_key]
            
            # Delete file
            if os.path.exists(report_info["file_path"]):
                os.remove(report_info["file_path"])
            
            # Remove from memory
            del self.generated_reports[report_key]
