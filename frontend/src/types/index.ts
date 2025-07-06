// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Dataset Types
export interface DatasetPreview {
  session_id: string;
  filename: string;
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  target_column: string;
  sample_data: Record<string, any>[];
  missing_values: Record<string, number>;
  memory_usage: number;
  numerical_columns: string[];
  categorical_columns: string[];
}

export interface TaskDetectionResponse {
  session_id: string;
  task_type: 'classification' | 'regression';
  target_column: string;
  target_stats: Record<string, any>;
  confidence_score: number;
  feature_count: number;
  feature_analysis: Record<string, any>;
  recommendations: string[];
}

// Training Types
export interface TrainingRequest {
  target_column?: string;
  test_size?: number;
  random_state?: number;
  cv_folds?: number;
  scoring_metric?: string;
}

export interface TrainingResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainingStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  current_step?: string;
  total_steps?: number;
  estimated_time_remaining?: number;
}

// Model Types
export interface ModelMetrics {
  model_name: string;
  task_type: string;
  metrics: Record<string, number>;
  cv_score: number;
  parameters: Record<string, any>;
}

export interface EvaluationResponse {
  session_id: string;
  task_type: string;
  best_model: string;
  model_results: Record<string, any>;
  training_timestamp: string;
}

export interface ModelComparison {
  session_id: string;
  task_type: string;
  models: Record<string, Record<string, any>>;
  best_model: string;
}

export interface FeatureImportance {
  model_name: string;
  feature_importance: Record<string, number>;
  total_features: number;
}

// Prediction Types
export interface PredictionRequest {
  input_data: Record<string, any>;
  model_name?: string;
}

export interface PredictionResponse {
  predictions: (number | string)[];
  prediction_proba?: number[][];
  model_name: string;
  input_features: string[];
}

// Visualization Types
export interface VisualizationResponse {
  session_id: string;
  visualizations: Record<string, string>;
  model_name: string;
}

export interface ConfusionMatrixResponse {
  visualization_url: string;
  matrix_data: number[][];
  class_labels: string[];
  model_name: string;
}

export interface ROCCurveResponse {
  visualization_url: string;
  fpr: number[];
  tpr: number[];
  auc_score: number;
  model_name: string;
}

// Report Types
export interface ReportRequest {
  format: 'pdf' | 'html' | 'json';
  include_visualizations?: boolean;
  include_raw_data?: boolean;
}

export interface ReportResponse {
  report_id: string;
  format: string;
  filename: string;
  download_url: string;
  generated_at: string;
}

// Session Types
export interface SessionInfo {
  session_id: string;
  created_at: string;
  dataset_filename: string;
  status: string;
  task_type?: string;
  target_column?: string;
}

// Data Quality Types
export interface DataQualityReport {
  session_id: string;
  total_rows: number;
  total_columns: number;
  missing_values_summary: Record<string, number>;
  duplicate_rows: number;
  numerical_columns_stats: Record<string, Record<string, number>>;
  categorical_columns_stats: Record<string, Record<string, any>>;
  data_quality_score: number;
  recommendations: string[];
}

// UI State Types
export interface UploadState {
  isUploading: boolean;
  progress: number;
  error?: string;
  success?: boolean;
}

export interface ThemeState {
  isDark: boolean;
  toggleTheme: () => void;
}

export interface StepperState {
  currentStep: number;
  completedSteps: number[];
  steps: StepInfo[];
}

export interface StepInfo {
  id: number;
  title: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'error';
}

// Chart Data Types
export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
    fill?: boolean;
  }[];
}

export interface ChartOptions {
  responsive: boolean;
  maintainAspectRatio: boolean;
  plugins?: {
    title?: {
      display: boolean;
      text: string;
    };
    legend?: {
      display: boolean;
      position?: 'top' | 'bottom' | 'left' | 'right';
    };
  };
  scales?: {
    x?: {
      display: boolean;
      title?: {
        display: boolean;
        text: string;
      };
    };
    y?: {
      display: boolean;
      title?: {
        display: boolean;
        text: string;
      };
    };
  };
}

// Error Types
export interface ErrorResponse {
  error: string;
  message: string;
  type: string;
  details?: Record<string, any>;
}

// File Upload Types
export interface FileUpload {
  file: File;
  target_column?: string;
}

// Navigation Types
export interface NavigationItem {
  id: string;
  label: string;
  icon: any;
  path: string;
  disabled?: boolean;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  duration?: number;
}

// Loading States
export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
}

// Feature Engineering Types
export interface FeatureEngineeringOptions {
  handle_missing: 'drop' | 'fill_mean' | 'fill_median' | 'fill_mode';
  encoding_method: 'one_hot' | 'label' | 'target';
  scaling_method: 'standard' | 'minmax' | 'robust' | 'none';
  feature_selection: boolean;
  create_polynomial_features: boolean;
  remove_outliers: boolean;
}

// Model Configuration Types
export interface ModelConfig {
  model_name: string;
  parameters: Record<string, any>;
  enabled: boolean;
}

// Hyperparameter Tuning Types
export interface HyperparameterTuningConfig {
  method: 'grid_search' | 'random_search' | 'bayesian';
  cv_folds: number;
  scoring_metric: string;
  n_iterations?: number;
  param_distributions?: Record<string, any>;
}

// Export Types
export interface ExportOptions {
  format: 'joblib' | 'pickle' | 'onnx';
  include_preprocessing: boolean;
  include_feature_names: boolean;
  deployment_ready: boolean;
}

// Dashboard Types
export interface DashboardStats {
  total_sessions: number;
  successful_trainings: number;
  average_accuracy: number;
  most_used_algorithm: string;
  total_datasets_processed: number;
}

// User Preferences Types
export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  default_test_size: number;
  default_cv_folds: number;
  auto_download_reports: boolean;
  notification_preferences: {
    training_complete: boolean;
    errors: boolean;
    warnings: boolean;
  };
}