import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronUpIcon,
  ChevronDownIcon,
  TrophyIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { ModelComparison } from '../types';

interface ModelComparisonTableProps {
  modelComparison: ModelComparison;
  taskType?: string;
  bestModel: string;
  className?: string;
}

export const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({
  modelComparison,
  taskType,
  bestModel,
  className = '',
}) => {
  const [sortBy, setSortBy] = useState<string>('score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  const models = Object.entries(modelComparison.models);

  const getMetricValue = (modelData: any, metric: string) => {
    if (metric === 'score') {
      return taskType === 'classification' 
        ? modelData.metrics?.accuracy || 0
        : modelData.metrics?.r2_score || 0;
    }
    return modelData.metrics?.[metric] || modelData[metric] || 0;
  };

  const sortedModels = [...models].sort((a, b) => {
    const aValue = getMetricValue(a[1], sortBy);
    const bValue = getMetricValue(b[1], sortBy);
    
    if (sortDirection === 'asc') {
      return aValue - bValue;
    }
    return bValue - aValue;
  });

  const handleSort = (metric: string) => {
    if (sortBy === metric) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(metric);
      setSortDirection('desc');
    }
  };

  const formatMetric = (value: number, metric: string) => {
    if (value == null) return '---';
    
    if (metric.includes('time')) {
      return `${value.toFixed(2)}s`;
    }
    
    if (metric.includes('accuracy') || metric.includes('precision') || 
        metric.includes('recall') || metric.includes('f1') || metric.includes('auc')) {
      return `${(value * 100).toFixed(1)}%`;
    }
    
    return value.toFixed(3);
  };

  const getMetricColor = (value: number, metric: string, isNegative = false) => {
    if (value == null) return 'text-gray-500';
    
    if (isNegative) {
      return value < 0.1 ? 'text-green-600 dark:text-green-400' : 
             value < 0.2 ? 'text-yellow-600 dark:text-yellow-400' : 
             'text-red-600 dark:text-red-400';
    }
    
    return value > 0.8 ? 'text-green-600 dark:text-green-400' : 
           value > 0.6 ? 'text-yellow-600 dark:text-yellow-400' : 
           'text-red-600 dark:text-red-400';
  };

  const getColumns = () => {
    if (taskType === 'classification') {
      return [
        { key: 'score', label: 'Accuracy', sortable: true },
        { key: 'precision', label: 'Precision', sortable: true },
        { key: 'recall', label: 'Recall', sortable: true },
        { key: 'f1_score', label: 'F1 Score', sortable: true },
        { key: 'roc_auc', label: 'ROC AUC', sortable: true },
        { key: 'training_time', label: 'Training Time', sortable: true },
      ];
    } else {
      return [
        { key: 'score', label: 'R² Score', sortable: true },
        { key: 'rmse', label: 'RMSE', sortable: true },
        { key: 'mae', label: 'MAE', sortable: true },
        { key: 'mape', label: 'MAPE', sortable: true },
        { key: 'training_time', label: 'Training Time', sortable: true },
      ];
    }
  };

  const columns = getColumns();

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Model Performance Comparison
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          {models.length} models trained
        </div>
      </div>

      {/* Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Model
                </th>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className={`px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider ${
                      column.sortable ? 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600' : ''
                    }`}
                    onClick={() => column.sortable && handleSort(column.key)}
                  >
                    <div className="flex items-center space-x-1">
                      <span>{column.label}</span>
                      {column.sortable && (
                        <div className="flex flex-col">
                          <ChevronUpIcon 
                            className={`w-3 h-3 ${
                              sortBy === column.key && sortDirection === 'asc' 
                                ? 'text-primary-500' 
                                : 'text-gray-400'
                            }`} 
                          />
                          <ChevronDownIcon 
                            className={`w-3 h-3 -mt-1 ${
                              sortBy === column.key && sortDirection === 'desc' 
                                ? 'text-primary-500' 
                                : 'text-gray-400'
                            }`} 
                          />
                        </div>
                      )}
                    </div>
                  </th>
                ))}
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {sortedModels.map(([modelName, modelData], index) => (
                <React.Fragment key={modelName}>
                  <motion.tr
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className={`hover:bg-gray-50 dark:hover:bg-gray-700 ${
                      modelName === bestModel ? 'bg-green-50 dark:bg-green-900/20' : ''
                    }`}
                  >
                    {/* Model Name */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        {modelName === bestModel && (
                          <TrophyIcon className="w-5 h-5 text-yellow-500" />
                        )}
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {modelName}
                            {modelName === bestModel && (
                              <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                Best
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            Rank #{index + 1}
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Metrics */}
                    {columns.map((column) => {
                      const value = getMetricValue(modelData, column.key);
                      const isNegative = column.key === 'rmse' || column.key === 'mae' || column.key === 'mape';
                      
                      return (
                        <td key={column.key} className="px-6 py-4 whitespace-nowrap">
                          <div className={`text-sm font-medium ${getMetricColor(value, column.key, isNegative)}`}>
                            {formatMetric(value, column.key)}
                          </div>
                        </td>
                      );
                    })}

                    {/* Actions */}
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                      <button
                        onClick={() => setExpandedModel(expandedModel === modelName ? null : modelName)}
                        className="text-primary-600 hover:text-primary-800 dark:text-primary-400 dark:hover:text-primary-300 font-medium"
                      >
                        {expandedModel === modelName ? 'Hide' : 'Details'}
                      </button>
                    </td>
                  </motion.tr>

                  {/* Expanded Details */}
                  {expandedModel === modelName && (
                    <motion.tr
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                    >
                      <td colSpan={columns.length + 2} className="px-6 py-4 bg-gray-50 dark:bg-gray-700">
                        <div className="space-y-4">
                          {/* Model Parameters */}
                          <div>
                            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                              Model Parameters
                            </h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                              {modelData.parameters && Object.entries(modelData.parameters).map(([param, value]) => (
                                <div key={param} className="text-xs">
                                  <span className="text-gray-500 dark:text-gray-400">{param}:</span>
                                  <span className="ml-1 font-medium text-gray-900 dark:text-white">
                                    {typeof value === 'number' ? value.toFixed(3) : String(value)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Cross-Validation Results */}
                          {modelData.cv_scores && (
                            <div>
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                                Cross-Validation Results
                              </h4>
                              <div className="flex items-center space-x-4 text-xs">
                                <div>
                                  <span className="text-gray-500 dark:text-gray-400">Mean CV Score:</span>
                                  <span className="ml-1 font-medium text-gray-900 dark:text-white">
                                    {formatMetric(modelData.cv_score, 'score')}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-500 dark:text-gray-400">Std Deviation:</span>
                                  <span className="ml-1 font-medium text-gray-900 dark:text-white">
                                    {formatMetric(modelData.cv_std || 0, 'score')}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Model Insights */}
                          <div>
                            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                              Model Insights
                            </h4>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              {modelName === bestModel ? (
                                <p>✅ This is your best performing model with excellent generalization capabilities.</p>
                              ) : (
                                <p>This model shows {
                                  getMetricValue(modelData, 'score') > 0.8 ? 'good' :
                                  getMetricValue(modelData, 'score') > 0.6 ? 'moderate' : 'poor'
                                } performance compared to the best model.</p>
                              )}
                            </div>
                          </div>
                        </div>
                      </td>
                    </motion.tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Insights */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <InformationCircleIcon className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300">
              Performance Analysis
            </h4>
            <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
              The table shows all trained models ranked by performance. The {bestModel} achieved the highest {
                taskType === 'classification' ? 'accuracy' : 'R² score'
              } and is recommended for deployment. Consider the trade-offs between accuracy and training time when selecting your final model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelComparisonTable;