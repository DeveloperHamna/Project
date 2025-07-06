import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon,
  EyeIcon,
  ArrowsPointingOutIcon,
  PhotoIcon
} from '@heroicons/react/24/outline';
import { useConfusionMatrix, useROCCurve, useFeatureImportance } from '../hooks/useApi';

interface VisualizationPanelProps {
  sessionId: string;
  taskType?: string;
  bestModel: string;
  className?: string;
}

export const VisualizationPanel: React.FC<VisualizationPanelProps> = ({
  sessionId,
  taskType,
  bestModel,
  className = '',
}) => {
  const [selectedVisualization, setSelectedVisualization] = useState<string>('feature-importance');
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { data: confusionMatrix, isLoading: isLoadingConfusion } = useConfusionMatrix(
    sessionId, 
    taskType === 'classification' ? bestModel : undefined
  );
  
  const { data: rocCurve, isLoading: isLoadingROC } = useROCCurve(
    sessionId, 
    taskType === 'classification' ? bestModel : undefined
  );
  
  const { data: featureImportance, isLoading: isLoadingFeatures } = useFeatureImportance(
    sessionId, 
    bestModel
  );

  const visualizations = [
    {
      id: 'feature-importance',
      title: 'Feature Importance',
      description: 'Shows which features contribute most to predictions',
      icon: ChartBarIcon,
      available: true,
    },
    ...(taskType === 'classification' ? [
      {
        id: 'confusion-matrix',
        title: 'Confusion Matrix',
        description: 'Shows prediction accuracy by class',
        icon: PhotoIcon,
        available: true,
      },
      {
        id: 'roc-curve',
        title: 'ROC Curve',
        description: 'Shows model performance across thresholds',
        icon: ChartBarIcon,
        available: true,
      },
    ] : [
      {
        id: 'residual-plot',
        title: 'Residual Plot',
        description: 'Shows prediction errors vs actual values',
        icon: ChartBarIcon,
        available: true,
      },
      {
        id: 'prediction-scatter',
        title: 'Prediction vs Actual',
        description: 'Scatter plot of predictions vs actual values',
        icon: PhotoIcon,
        available: true,
      },
    ]),
  ];

  const renderFeatureImportance = () => {
    if (isLoadingFeatures) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        </div>
      );
    }

    if (!featureImportance?.feature_importance) {
      return (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No feature importance data available
        </div>
      );
    }

    const features = Object.entries(featureImportance.feature_importance)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10); // Show top 10 features

    const maxImportance = Math.max(...Object.values(featureImportance.feature_importance));

    return (
      <div className="space-y-4">
        <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
          Top 10 Most Important Features
        </h4>
        <div className="space-y-3">
          {features.map(([feature, importance], index) => (
            <motion.div
              key={feature}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="flex items-center space-x-3"
            >
              <div className="w-24 text-sm text-gray-600 dark:text-gray-400 truncate">
                {feature}
              </div>
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(importance / maxImportance) * 100}%` }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                />
              </div>
              <div className="w-16 text-sm font-medium text-gray-900 dark:text-white text-right">
                {(importance * 100).toFixed(1)}%
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    );
  };

  const renderConfusionMatrix = () => {
    if (isLoadingConfusion) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        </div>
      );
    }

    if (!confusionMatrix?.matrix_data) {
      return (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No confusion matrix data available
        </div>
      );
    }

    const matrix = confusionMatrix.matrix_data;
    const labels = confusionMatrix.class_labels;
    const total = matrix.flat().reduce((sum, val) => sum + val, 0);

    return (
      <div className="space-y-4">
        <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
          Confusion Matrix - {bestModel}
        </h4>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="p-2"></th>
                <th colSpan={labels.length} className="text-center text-sm font-medium text-gray-600 dark:text-gray-400 p-2">
                  Predicted
                </th>
              </tr>
              <tr>
                <th className="p-2"></th>
                {labels.map((label) => (
                  <th key={label} className="text-sm font-medium text-gray-600 dark:text-gray-400 p-2">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  {i === 0 && (
                    <th 
                      rowSpan={matrix.length} 
                      className="text-sm font-medium text-gray-600 dark:text-gray-400 p-2 -rotate-90"
                    >
                      Actual
                    </th>
                  )}
                  <th className="text-sm font-medium text-gray-600 dark:text-gray-400 p-2">
                    {labels[i]}
                  </th>
                  {row.map((value, j) => (
                    <td key={j} className="p-2">
                      <div 
                        className={`
                          w-16 h-16 flex items-center justify-center rounded text-sm font-medium
                          ${i === j 
                            ? 'bg-green-500 text-white' 
                            : value > 0 
                            ? 'bg-red-200 dark:bg-red-900 text-red-800 dark:text-red-200' 
                            : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                          }
                        `}
                        style={{
                          opacity: value === 0 ? 0.3 : Math.max(0.3, value / Math.max(...matrix.flat()))
                        }}
                      >
                        {value}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Total predictions: {total} • Darker colors indicate higher values
        </div>
      </div>
    );
  };

  const renderROCCurve = () => {
    if (isLoadingROC) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        </div>
      );
    }

    if (!rocCurve?.fpr || !rocCurve?.tpr) {
      return (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No ROC curve data available
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
            ROC Curve - {bestModel}
          </h4>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            AUC: {rocCurve.auc_score.toFixed(3)}
          </div>
        </div>
        
        <div className="relative bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4">
          <svg viewBox="0 0 400 400" className="w-full h-64">
            {/* Grid lines */}
            {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
              <g key={tick}>
                <line
                  x1={tick * 360 + 40}
                  y1={40}
                  x2={tick * 360 + 40}
                  y2={360}
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-gray-200 dark:text-gray-600"
                  strokeDasharray="2,2"
                />
                <line
                  x1={40}
                  y1={360 - tick * 320}
                  x2={400}
                  y2={360 - tick * 320}
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-gray-200 dark:text-gray-600"
                  strokeDasharray="2,2"
                />
              </g>
            ))}

            {/* Axes */}
            <line x1={40} y1={40} x2={40} y2={360} stroke="currentColor" strokeWidth="2" className="text-gray-400" />
            <line x1={40} y1={360} x2={400} y2={360} stroke="currentColor" strokeWidth="2" className="text-gray-400" />

            {/* Diagonal reference line */}
            <line x1={40} y1={360} x2={400} y2={40} stroke="currentColor" strokeWidth="1" strokeDasharray="5,5" className="text-gray-400" />

            {/* ROC Curve */}
            <path
              d={`M 40 360 ${rocCurve.fpr.map((fpr, i) => 
                `L ${40 + fpr * 360} ${360 - rocCurve.tpr[i] * 320}`
              ).join(' ')}`}
              fill="none"
              stroke="#3B82F6"
              strokeWidth="3"
            />

            {/* Labels */}
            <text x="220" y="390" textAnchor="middle" className="text-sm fill-current text-gray-600 dark:text-gray-400">
              False Positive Rate
            </text>
            <text x="20" y="200" textAnchor="middle" className="text-sm fill-current text-gray-600 dark:text-gray-400" transform="rotate(-90 20 200)">
              True Positive Rate
            </text>
          </svg>
        </div>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
          <div className="text-sm text-blue-800 dark:text-blue-300">
            <strong>AUC Score: {rocCurve.auc_score.toFixed(3)}</strong>
            <p className="text-xs mt-1 text-blue-600 dark:text-blue-400">
              {rocCurve.auc_score > 0.9 ? 'Excellent' : 
               rocCurve.auc_score > 0.8 ? 'Good' : 
               rocCurve.auc_score > 0.7 ? 'Fair' : 'Poor'} classification performance
            </p>
          </div>
        </div>
      </div>
    );
  };

  const renderVisualization = () => {
    switch (selectedVisualization) {
      case 'feature-importance':
        return renderFeatureImportance();
      case 'confusion-matrix':
        return renderConfusionMatrix();
      case 'roc-curve':
        return renderROCCurve();
      default:
        return (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            Visualization not available
          </div>
        );
    }
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Visualization Selector */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Model Visualizations
        </h3>
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
        >
          <ArrowsPointingOutIcon className="w-5 h-5" />
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {visualizations.map((viz) => {
          const Icon = viz.icon;
          return (
            <button
              key={viz.id}
              onClick={() => setSelectedVisualization(viz.id)}
              disabled={!viz.available}
              className={`
                p-4 rounded-lg border-2 text-left transition-all duration-200
                ${selectedVisualization === viz.id
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                }
                ${!viz.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              <div className="flex items-center space-x-3">
                <Icon className={`w-5 h-5 ${
                  selectedVisualization === viz.id 
                    ? 'text-primary-600 dark:text-primary-400' 
                    : 'text-gray-500 dark:text-gray-400'
                }`} />
                <div>
                  <div className={`font-medium ${
                    selectedVisualization === viz.id 
                      ? 'text-primary-900 dark:text-primary-100' 
                      : 'text-gray-900 dark:text-white'
                  }`}>
                    {viz.title}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {viz.description}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Visualization Content */}
      <motion.div
        key={selectedVisualization}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={`
          bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 p-6
          ${isFullscreen ? 'fixed inset-4 z-50 overflow-auto' : ''}
        `}
      >
        {isFullscreen && (
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              {visualizations.find(v => v.id === selectedVisualization)?.title}
            </h3>
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        
        {renderVisualization()}
      </motion.div>
    </div>
  );
};

export default VisualizationPanel;