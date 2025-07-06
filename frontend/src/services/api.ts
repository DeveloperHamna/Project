import axios, { AxiosResponse } from 'axios';
import {
  ApiResponse,
  DatasetPreview,
  TaskDetectionResponse,
  TrainingRequest,
  TrainingResponse,
  TrainingStatus,
  EvaluationResponse,
  ModelComparison,
  FeatureImportance,
  PredictionRequest,
  PredictionResponse,
  VisualizationResponse,
  ConfusionMatrixResponse,
  ROCCurveResponse,
  ReportRequest,
  ReportResponse,
  SessionInfo,
  DataQualityReport,
  FileUpload,
} from '../types';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const API_VERSION = '/api/v1';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    // Handle common errors
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    
    // Format error message
    const errorMessage = error.response?.data?.message || error.message || 'An error occurred';
    return Promise.reject(new Error(errorMessage));
  }
);

// API Service Class
class ApiService {
  // Dataset Management
  async uploadDataset(fileUpload: FileUpload): Promise<DatasetPreview> {
    const formData = new FormData();
    formData.append('file', fileUpload.file);
    if (fileUpload.target_column) {
      formData.append('target_column', fileUpload.target_column);
    }

    const response = await apiClient.post<DatasetPreview>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / (progressEvent.total || 1)
        );
        // Emit progress event
        window.dispatchEvent(
          new CustomEvent('uploadProgress', { detail: { progress: percentCompleted } })
        );
      },
    });

    return response.data;
  }

  async getDatasetPreview(sessionId: string): Promise<DatasetPreview> {
    const response = await apiClient.get<DatasetPreview>(`/dataset/${sessionId}/preview`);
    return response.data;
  }

  async detectTaskType(sessionId: string, targetColumn?: string): Promise<TaskDetectionResponse> {
    const response = await apiClient.post<TaskDetectionResponse>(
      `/dataset/${sessionId}/detect-task`,
      { target_column: targetColumn }
    );
    return response.data;
  }

  async getDataQualityReport(sessionId: string): Promise<DataQualityReport> {
    const response = await apiClient.get<DataQualityReport>(`/dataset/${sessionId}/quality-report`);
    return response.data;
  }

  // Training Management
  async startTraining(sessionId: string, request: TrainingRequest): Promise<TrainingResponse> {
    const response = await apiClient.post<TrainingResponse>(`/train/${sessionId}`, request);
    return response.data;
  }

  async getTrainingStatus(jobId: string): Promise<TrainingStatus> {
    const response = await apiClient.get<TrainingStatus>(`/training/${jobId}/status`);
    return response.data;
  }

  async cancelTraining(jobId: string): Promise<void> {
    await apiClient.delete(`/training/${jobId}`);
  }

  // Model Evaluation
  async getEvaluationResults(sessionId: string): Promise<EvaluationResponse> {
    const response = await apiClient.get<EvaluationResponse>(`/evaluation/${sessionId}`);
    return response.data;
  }

  async getModelComparison(sessionId: string): Promise<ModelComparison> {
    const response = await apiClient.get<ModelComparison>(`/evaluation/${sessionId}/comparison`);
    return response.data;
  }

  async getFeatureImportance(sessionId: string, modelName?: string): Promise<FeatureImportance> {
    const params = modelName ? { model_name: modelName } : {};
    const response = await apiClient.get<FeatureImportance>(
      `/evaluation/${sessionId}/feature-importance`,
      { params }
    );
    return response.data;
  }

  // Predictions
  async makePrediction(sessionId: string, request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post<PredictionResponse>(`/predict/${sessionId}`, request);
    return response.data;
  }

  async batchPredict(sessionId: string, inputData: Record<string, any>[]): Promise<PredictionResponse> {
    const response = await apiClient.post<PredictionResponse>(`/predict/${sessionId}/batch`, {
      input_data: inputData,
    });
    return response.data;
  }

  // Visualizations
  async getVisualizations(sessionId: string): Promise<VisualizationResponse> {
    const response = await apiClient.get<VisualizationResponse>(`/evaluation/${sessionId}/visualizations`);
    return response.data;
  }

  async getConfusionMatrix(sessionId: string, modelName?: string): Promise<ConfusionMatrixResponse> {
    const params = modelName ? { model_name: modelName } : {};
    const response = await apiClient.get<ConfusionMatrixResponse>(
      `/evaluation/${sessionId}/confusion-matrix`,
      { params }
    );
    return response.data;
  }

  async getROCCurve(sessionId: string, modelName?: string): Promise<ROCCurveResponse> {
    const params = modelName ? { model_name: modelName } : {};
    const response = await apiClient.get<ROCCurveResponse>(
      `/evaluation/${sessionId}/roc-curve`,
      { params }
    );
    return response.data;
  }

  // Reports
  async generateReport(sessionId: string, request: ReportRequest): Promise<ReportResponse> {
    const response = await apiClient.post<ReportResponse>(`/reports/${sessionId}/generate`, request);
    return response.data;
  }

  async downloadReport(sessionId: string, reportId: string): Promise<Blob> {
    const response = await apiClient.get(`/reports/${sessionId}/download/${reportId}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async getReportsList(sessionId: string): Promise<ReportResponse[]> {
    const response = await apiClient.get<ReportResponse[]>(`/reports/${sessionId}/list`);
    return response.data;
  }

  // Session Management
  async getSessionInfo(sessionId: string): Promise<SessionInfo> {
    const response = await apiClient.get<SessionInfo>(`/session/${sessionId}`);
    return response.data;
  }

  async getAllSessions(): Promise<SessionInfo[]> {
    const response = await apiClient.get<SessionInfo[]>('/sessions');
    return response.data;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await apiClient.delete(`/session/${sessionId}`);
  }

  // Health Check
  async healthCheck(): Promise<{ status: string; service: string }> {
    const response = await apiClient.get<{ status: string; service: string }>('/health');
    return response.data;
  }

  // Model Export
  async exportModel(sessionId: string, modelName: string, format: string): Promise<Blob> {
    const response = await apiClient.post(
      `/model/${sessionId}/export`,
      { model_name: modelName, format },
      { responseType: 'blob' }
    );
    return response.data;
  }

  // Utilities
  async getColumnSuggestions(sessionId: string): Promise<string[]> {
    const response = await apiClient.get<string[]>(`/dataset/${sessionId}/columns`);
    return response.data;
  }

  async validateDataset(file: File): Promise<{ valid: boolean; errors: string[] }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<{ valid: boolean; errors: string[] }>(
      '/dataset/validate',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();

// Export individual methods for convenience
export const {
  uploadDataset,
  getDatasetPreview,
  detectTaskType,
  getDataQualityReport,
  startTraining,
  getTrainingStatus,
  cancelTraining,
  getEvaluationResults,
  getModelComparison,
  getFeatureImportance,
  makePrediction,
  batchPredict,
  getVisualizations,
  getConfusionMatrix,
  getROCCurve,
  generateReport,
  downloadReport,
  getReportsList,
  getSessionInfo,
  getAllSessions,
  deleteSession,
  healthCheck,
  exportModel,
  getColumnSuggestions,
  validateDataset,
} = apiService;

// Export types for use in components
export type { ApiResponse };

// Error handling utility
export const handleApiError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

// Success notification utility
export const handleApiSuccess = (message: string): void => {
  window.dispatchEvent(
    new CustomEvent('apiSuccess', { detail: { message } })
  );
};

// Loading state utility
export const setLoadingState = (isLoading: boolean, message?: string): void => {
  window.dispatchEvent(
    new CustomEvent('loadingStateChange', { 
      detail: { isLoading, message } 
    })
  );
};