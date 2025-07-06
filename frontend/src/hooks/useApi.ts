import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { 
  DatasetPreview, 
  TaskDetectionResponse, 
  TrainingStatus, 
  EvaluationResponse,
  ModelComparison,
  FeatureImportance,
  VisualizationResponse,
  ReportResponse,
  SessionInfo,
  DataQualityReport,
  FileUpload,
  TrainingRequest,
  PredictionRequest,
  ReportRequest,
} from '../types';

// Upload Hook
export const useUpload = () => {
  const [progress, setProgress] = useState(0);
  const queryClient = useQueryClient();

  useEffect(() => {
    const handleProgress = (event: CustomEvent) => {
      setProgress(event.detail.progress);
    };

    window.addEventListener('uploadProgress', handleProgress as EventListener);
    return () => window.removeEventListener('uploadProgress', handleProgress as EventListener);
  }, []);

  return useMutation({
    mutationFn: async (fileUpload: FileUpload) => {
      setProgress(0);
      return await apiService.uploadDataset(fileUpload);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['dataset', data.session_id], data);
      setProgress(100);
    },
    onError: () => {
      setProgress(0);
    },
  });
};

// Dataset Preview Hook
export const useDatasetPreview = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['dataset', sessionId],
    queryFn: () => apiService.getDatasetPreview(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Task Detection Hook
export const useTaskDetection = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['task-detection', sessionId],
    queryFn: () => apiService.detectTaskType(sessionId!),
    enabled: !!sessionId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Data Quality Report Hook
export const useDataQualityReport = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['data-quality', sessionId],
    queryFn: () => apiService.getDataQualityReport(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Training Hook
export const useTraining = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ sessionId, request }: { sessionId: string; request: TrainingRequest }) => {
      return await apiService.startTraining(sessionId, request);
    },
    onSuccess: (data, variables) => {
      queryClient.setQueryData(['training', data.job_id], data);
      queryClient.invalidateQueries({ queryKey: ['training-status', data.job_id] });
    },
  });
};

// Training Status Hook with Polling
export const useTrainingStatus = (jobId: string | null, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['training-status', jobId],
    queryFn: () => apiService.getTrainingStatus(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (data) => {
      // Poll every 2 seconds if training is in progress
      if (data?.status === 'running' || data?.status === 'pending') {
        return 2000;
      }
      return false;
    },
    staleTime: 1000, // Always refetch when requested
  });
};

// Evaluation Results Hook
export const useEvaluationResults = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['evaluation', sessionId],
    queryFn: () => apiService.getEvaluationResults(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Model Comparison Hook
export const useModelComparison = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['model-comparison', sessionId],
    queryFn: () => apiService.getModelComparison(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Feature Importance Hook
export const useFeatureImportance = (sessionId: string | null, modelName?: string) => {
  return useQuery({
    queryKey: ['feature-importance', sessionId, modelName],
    queryFn: () => apiService.getFeatureImportance(sessionId!, modelName),
    enabled: !!sessionId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Visualizations Hook
export const useVisualizations = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['visualizations', sessionId],
    queryFn: () => apiService.getVisualizations(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Confusion Matrix Hook
export const useConfusionMatrix = (sessionId: string | null, modelName?: string) => {
  return useQuery({
    queryKey: ['confusion-matrix', sessionId, modelName],
    queryFn: () => apiService.getConfusionMatrix(sessionId!, modelName),
    enabled: !!sessionId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// ROC Curve Hook
export const useROCCurve = (sessionId: string | null, modelName?: string) => {
  return useQuery({
    queryKey: ['roc-curve', sessionId, modelName],
    queryFn: () => apiService.getROCCurve(sessionId!, modelName),
    enabled: !!sessionId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Prediction Hook
export const usePrediction = () => {
  return useMutation({
    mutationFn: async ({ sessionId, request }: { sessionId: string; request: PredictionRequest }) => {
      return await apiService.makePrediction(sessionId, request);
    },
  });
};

// Report Generation Hook
export const useReportGeneration = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ sessionId, request }: { sessionId: string; request: ReportRequest }) => {
      return await apiService.generateReport(sessionId, request);
    },
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['reports', variables.sessionId] });
    },
  });
};

// Reports List Hook
export const useReportsList = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['reports', sessionId],
    queryFn: () => apiService.getReportsList(sessionId!),
    enabled: !!sessionId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Session Info Hook
export const useSessionInfo = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => apiService.getSessionInfo(sessionId!),
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// All Sessions Hook
export const useAllSessions = () => {
  return useQuery({
    queryKey: ['sessions'],
    queryFn: () => apiService.getAllSessions(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Delete Session Hook
export const useDeleteSession = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      return await apiService.deleteSession(sessionId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
};

// Health Check Hook
export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.healthCheck(),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 30 * 1000, // Refetch every 30 seconds
  });
};

// Model Export Hook
export const useModelExport = () => {
  return useMutation({
    mutationFn: async ({ sessionId, modelName, format }: { sessionId: string; modelName: string; format: string }) => {
      return await apiService.exportModel(sessionId, modelName, format);
    },
  });
};

// Column Suggestions Hook
export const useColumnSuggestions = (sessionId: string | null) => {
  return useQuery({
    queryKey: ['column-suggestions', sessionId],
    queryFn: () => apiService.getColumnSuggestions(sessionId!),
    enabled: !!sessionId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Dataset Validation Hook
export const useDatasetValidation = () => {
  return useMutation({
    mutationFn: async (file: File) => {
      return await apiService.validateDataset(file);
    },
  });
};

// Custom Hook for Managing Training Pipeline
export const useTrainingPipeline = (sessionId: string | null) => {
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  const trainingMutation = useTraining();
  const trainingStatus = useTrainingStatus(currentJobId, isTraining);

  const startTraining = useCallback(async (request: TrainingRequest) => {
    if (!sessionId) return;

    setIsTraining(true);
    try {
      const result = await trainingMutation.mutateAsync({ sessionId, request });
      setCurrentJobId(result.job_id);
    } catch (error) {
      setIsTraining(false);
      setCurrentJobId(null);
    }
  }, [sessionId, trainingMutation]);

  const stopTraining = useCallback(() => {
    setIsTraining(false);
    setCurrentJobId(null);
  }, []);

  // Auto-stop training when completed or failed
  useEffect(() => {
    if (trainingStatus.data?.status === 'completed' || trainingStatus.data?.status === 'failed') {
      setIsTraining(false);
    }
  }, [trainingStatus.data?.status]);

  return {
    startTraining,
    stopTraining,
    isTraining,
    trainingStatus: trainingStatus.data,
    trainingError: trainingMutation.error || trainingStatus.error,
    isStartingTraining: trainingMutation.isPending,
  };
};

// Custom Hook for Managing Complete ML Pipeline
export const useMLPipeline = (sessionId: string | null) => {
  const datasetPreview = useDatasetPreview(sessionId);
  const taskDetection = useTaskDetection(sessionId);
  const dataQuality = useDataQualityReport(sessionId);
  const trainingPipeline = useTrainingPipeline(sessionId);
  const evaluationResults = useEvaluationResults(sessionId);
  const modelComparison = useModelComparison(sessionId);
  const visualizations = useVisualizations(sessionId);

  const isDataReady = !!(datasetPreview.data && taskDetection.data);
  const isTrainingComplete = trainingPipeline.trainingStatus?.status === 'completed';
  const isEvaluationReady = !!(isTrainingComplete && evaluationResults.data);

  return {
    // Data
    datasetPreview: datasetPreview.data,
    taskDetection: taskDetection.data,
    dataQuality: dataQuality.data,
    
    // Training
    ...trainingPipeline,
    
    // Evaluation
    evaluationResults: evaluationResults.data,
    modelComparison: modelComparison.data,
    visualizations: visualizations.data,
    
    // Loading states
    isLoadingData: datasetPreview.isLoading || taskDetection.isLoading,
    isLoadingEvaluation: evaluationResults.isLoading || modelComparison.isLoading,
    
    // Pipeline states
    isDataReady,
    isTrainingComplete,
    isEvaluationReady,
    
    // Errors
    dataError: datasetPreview.error || taskDetection.error,
    evaluationError: evaluationResults.error || modelComparison.error,
  };
};