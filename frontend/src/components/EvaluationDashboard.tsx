import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon,
  TrophyIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { EvaluationResponse, ModelComparison } from '../types';
import { ModelComparisonTable } from './ModelComparisonTable';
import { VisualizationPanel } from './VisualizationPanel';
import { MetricsCards } from './MetricsCards';

interface EvaluationDashboardProps {
  sessionId: string | null;
  evaluationResults?: EvaluationResponse | null;
  modelComparison?: ModelComparison | null;
  taskType?: string;
  className?: string;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  sessionId,
  evaluationResults,
  modelComparison,
  taskType,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'comparison' | 'visualizations'>('overview');

  if (!evaluationResults || !modelComparison) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="space-y-4">
          <div className="w-16 h-16 mx-auto">
            <div className="w-full h-full border-4 border-gray-200 dark:border-gray-600 border-t-primary-500 rounded-full animate-spin"></div>
          </div>
          <p className="text-gray-500 dark:text-gray-400">
            Loading evaluation results...
          </p>
        </div>
      </div>
    );
  }

  const bestModel = evaluationResults.best_model;
  const bestModelResults = modelComparison.models[bestModel];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: ChartBarIcon },
    { id: 'comparison', label: 'Model Comparison', icon: TrophyIcon },
    { id: 'visualizations', label: 'Visualizations', icon: EyeIcon },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Best Model Info */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-green-200 dark:border-green-800"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
              <TrophyIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Best Model: {bestModel}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Task: {taskType?.charAt(0).toUpperCase()}{taskType?.slice(1)} • 
                Trained on {new Date(evaluationResults.training_timestamp).toLocaleDateString()}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button className="btn-primary flex items-center space-x-2">
              <ArrowDownTrayIcon className="w-4 h-4" />
              <span>Download Model</span>
            </button>
            <button className="btn-secondary flex items-center space-x-2">
              <InformationCircleIcon className="w-4 h-4" />
              <span>Model Details</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors
                  ${activeTab === tab.id
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Metrics Cards */}
            <MetricsCards
              modelResults={bestModelResults}
              taskType={taskType}
            />
            
            {/* Quick Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Total Models Trained */}
              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Models Trained</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {Object.keys(modelComparison.models).length}
                    </p>
                  </div>
                  <ChartBarIcon className="w-8 h-8 text-primary-500" />
                </div>
              </div>

              {/* Best Score */}
              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Best {taskType === 'classification' ? 'Accuracy' : 'R² Score'}
                    </p>
                    <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {taskType === 'classification' 
                        ? `${(bestModelResults?.metrics?.accuracy * 100 || 0).toFixed(1)}%`
                        : `${(bestModelResults?.metrics?.r2_score || 0).toFixed(3)}`
                      }
                    </p>
                  </div>
                  <TrophyIcon className="w-8 h-8 text-green-500" />
                </div>
              </div>

              {/* Training Time */}
              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Training Time</p>
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {bestModelResults?.training_time 
                        ? `${bestModelResults.training_time.toFixed(1)}s`
                        : '---'
                      }
                    </p>
                  </div>
                  <svg className="w-8 h-8 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
              </div>

              {/* Model Complexity */}
              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Complexity</p>
                    <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                      {bestModelResults?.parameters?.max_depth || 
                       bestModelResults?.parameters?.n_estimators || 
                       'Auto'}
                    </p>
                  </div>
                  <svg className="w-8 h-8 text-purple-500" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                </div>
              </div>
            </div>

            {/* Model Performance Summary */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Performance Summary
              </h3>
              <div className="prose dark:prose-invert max-w-none">
                <p className="text-gray-600 dark:text-gray-400">
                  Your best performing model is <strong>{bestModel}</strong> with{' '}
                  {taskType === 'classification' ? (
                    <>an accuracy of <strong>{(bestModelResults?.metrics?.accuracy * 100 || 0).toFixed(1)}%</strong></>
                  ) : (
                    <>an R² score of <strong>{(bestModelResults?.metrics?.r2_score || 0).toFixed(3)}</strong></>
                  )}
                  . This model demonstrates{' '}
                  {(bestModelResults?.metrics?.accuracy || bestModelResults?.metrics?.r2_score || 0) > 0.8 
                    ? 'excellent' 
                    : (bestModelResults?.metrics?.accuracy || bestModelResults?.metrics?.r2_score || 0) > 0.6
                    ? 'good'
                    : 'moderate'
                  } performance on your dataset.
                </p>
                
                {taskType === 'classification' && bestModelResults?.metrics && (
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Precision:</span>
                      <span className="ml-2 font-medium">{(bestModelResults.metrics.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Recall:</span>
                      <span className="ml-2 font-medium">{(bestModelResults.metrics.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">F1-Score:</span>
                      <span className="ml-2 font-medium">{(bestModelResults.metrics.f1_score * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">AUC:</span>
                      <span className="ml-2 font-medium">{(bestModelResults.metrics.roc_auc * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )}

                {taskType === 'regression' && bestModelResults?.metrics && (
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">RMSE:</span>
                      <span className="ml-2 font-medium">{bestModelResults.metrics.rmse?.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">MAE:</span>
                      <span className="ml-2 font-medium">{bestModelResults.metrics.mae?.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">MAPE:</span>
                      <span className="ml-2 font-medium">{bestModelResults.metrics.mape?.toFixed(1)}%</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'comparison' && (
          <ModelComparisonTable
            modelComparison={modelComparison}
            taskType={taskType}
            bestModel={bestModel}
          />
        )}

        {activeTab === 'visualizations' && sessionId && (
          <VisualizationPanel
            sessionId={sessionId}
            taskType={taskType}
            bestModel={bestModel}
          />
        )}
      </motion.div>
    </div>
  );
};

export default EvaluationDashboard;