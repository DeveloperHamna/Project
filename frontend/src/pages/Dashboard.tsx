import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast, { Toaster } from 'react-hot-toast';
import { UploadZone } from '../components/UploadZone';
import { StepperProgress, DEFAULT_STEPS } from '../components/StepperProgress';
import { DataPreviewTable } from '../components/DataPreviewTable';
import { TrainingProgress } from '../components/TrainingProgress';
import { EvaluationDashboard } from '../components/EvaluationDashboard';
import { ReportsPanel } from '../components/ReportsPanel';
import { useMLPipeline } from '../hooks/useApi';
import { StepInfo } from '../types';

export const Dashboard: React.FC = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [steps, setSteps] = useState<StepInfo[]>(DEFAULT_STEPS);

  const {
    datasetPreview,
    taskDetection,
    dataQuality,
    startTraining,
    stopTraining,
    isTraining,
    trainingStatus,
    trainingError,
    isStartingTraining,
    evaluationResults,
    modelComparison,
    visualizations,
    isDataReady,
    isTrainingComplete,
    isEvaluationReady,
    isLoadingData,
    isLoadingEvaluation,
    dataError,
    evaluationError,
  } = useMLPipeline(sessionId);

  // Handle upload success
  const handleUploadSuccess = (newSessionId: string) => {
    setSessionId(newSessionId);
    setCurrentStep(2);
    setCompletedSteps([1]);
    toast.success('Dataset uploaded successfully!');
  };

  // Handle upload error
  const handleUploadError = (error: string) => {
    toast.error(`Upload failed: ${error}`);
    setCurrentStep(1);
    setCompletedSteps([]);
  };

  // Handle training start
  const handleStartTraining = () => {
    if (!sessionId) return;
    
    startTraining({
      test_size: 0.2,
      cv_folds: 5,
      random_state: 42,
    });
    
    setCurrentStep(3);
    setCompletedSteps([1, 2]);
    toast.success('Training started!');
  };

  // Update steps based on data availability
  useEffect(() => {
    if (isDataReady && !completedSteps.includes(2)) {
      setCompletedSteps(prev => [...prev, 2]);
      if (currentStep === 2) {
        // Auto-advance to step 3 when data analysis is complete
        setTimeout(() => setCurrentStep(3), 1000);
      }
    }
  }, [isDataReady, completedSteps, currentStep]);

  // Update steps based on training completion
  useEffect(() => {
    if (isTrainingComplete && !completedSteps.includes(3)) {
      setCompletedSteps(prev => [...prev, 3]);
      setCurrentStep(4);
      toast.success('Training completed successfully!');
    }
  }, [isTrainingComplete, completedSteps]);

  // Update steps based on evaluation readiness
  useEffect(() => {
    if (isEvaluationReady && !completedSteps.includes(4)) {
      setCompletedSteps(prev => [...prev, 4]);
    }
  }, [isEvaluationReady, completedSteps]);

  // Handle training errors
  useEffect(() => {
    if (trainingError) {
      toast.error(`Training failed: ${trainingError.message}`);
      setSteps(prev => prev.map(step => 
        step.id === 3 
          ? { ...step, status: 'error' }
          : step
      ));
    }
  }, [trainingError]);

  // Handle data errors
  useEffect(() => {
    if (dataError) {
      toast.error(`Data loading failed: ${dataError.message}`);
      setSteps(prev => prev.map(step => 
        step.id === 2 
          ? { ...step, status: 'error' }
          : step
      ));
    }
  }, [dataError]);

  const resetWorkflow = () => {
    setSessionId(null);
    setCurrentStep(1);
    setCompletedSteps([]);
    setSteps(DEFAULT_STEPS);
    toast.success('Workflow reset. You can start with a new dataset.');
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Welcome to <span className="gradient-text">AutoML Studio</span>
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
          Upload your dataset and let our AI automatically build, train, and evaluate 
          machine learning models for you. No coding required.
        </p>
      </motion.div>

      {/* Progress Stepper */}
      <StepperProgress
        currentStep={currentStep}
        completedSteps={completedSteps}
        steps={steps}
      />

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Primary Actions */}
        <div className="lg:col-span-2 space-y-6">
          {/* Step 1: Upload */}
          {currentStep === 1 && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="card p-6">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
                  Step 1: Upload Your Dataset
                </h2>
                <UploadZone
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={handleUploadError}
                />
              </div>
            </motion.div>
          )}

          {/* Step 2: Data Preview & Analysis */}
          {currentStep >= 2 && datasetPreview && (
            <motion.div
              key="data-analysis"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                    Step 2: Data Analysis
                  </h2>
                  {isDataReady && currentStep === 2 && (
                    <motion.button
                      onClick={handleStartTraining}
                      className="btn-primary"
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.3 }}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Start Training →
                    </motion.button>
                  )}
                </div>
                
                <DataPreviewTable
                  datasetPreview={datasetPreview}
                  taskDetection={taskDetection}
                />
              </div>
            </motion.div>
          )}

          {/* Step 3: Training */}
          {currentStep >= 3 && (
            <motion.div
              key="training"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="card p-6">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
                  Step 3: Model Training
                </h2>
                <TrainingProgress
                  isTraining={isTraining}
                  trainingStatus={trainingStatus}
                  onStop={stopTraining}
                  sessionId={sessionId}
                />
              </div>
            </motion.div>
          )}

          {/* Step 4: Evaluation & Results */}
          {currentStep >= 4 && isTrainingComplete && (
            <motion.div
              key="evaluation"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="card p-6">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
                  Step 4: Results & Evaluation
                </h2>
                <EvaluationDashboard
                  sessionId={sessionId}
                  evaluationResults={evaluationResults}
                  modelComparison={modelComparison}
                  taskType={taskDetection?.task_type}
                />
              </div>
            </motion.div>
          )}
        </div>

        {/* Right Column - Sidebar */}
        <div className="space-y-6">
          {/* Session Info */}
          {sessionId && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="card p-4"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                Session Info
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Session ID:</span>
                  <span className="font-mono text-xs">{sessionId.substring(0, 8)}...</span>
                </div>
                {datasetPreview && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Dataset:</span>
                      <span className="truncate ml-2">{datasetPreview.filename}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Rows:</span>
                      <span>{datasetPreview.shape[0].toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Columns:</span>
                      <span>{datasetPreview.shape[1]}</span>
                    </div>
                  </>
                )}
                {taskDetection && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Task Type:</span>
                    <span className="capitalize">{taskDetection.task_type}</span>
                  </div>
                )}
              </div>
              
              <button
                onClick={resetWorkflow}
                className="btn-secondary w-full mt-4 text-sm"
              >
                Start New Session
              </button>
            </motion.div>
          )}

          {/* Quick Actions */}
          {sessionId && isEvaluationReady && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="card p-4"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                Quick Actions
              </h3>
              <div className="space-y-2">
                <button className="btn-primary w-full text-sm">
                  Download Best Model
                </button>
                <button className="btn-secondary w-full text-sm">
                  Generate Report
                </button>
                <button className="btn-secondary w-full text-sm">
                  Export Predictions
                </button>
              </div>
            </motion.div>
          )}

          {/* Reports Panel */}
          {sessionId && isTrainingComplete && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <ReportsPanel sessionId={sessionId} />
            </motion.div>
          )}

          {/* Tips & Help */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
            className="card p-4"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              💡 Tips
            </h3>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p>• Use clean, well-structured data for best results</p>
              <p>• Ensure your target column is clearly defined</p>
              <p>• Larger datasets generally produce better models</p>
              <p>• Check for data quality issues before training</p>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'var(--toast-bg)',
            color: 'var(--toast-color)',
          },
        }}
      />
    </div>
  );
};

export default Dashboard;