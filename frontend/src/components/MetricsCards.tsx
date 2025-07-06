import React from 'react';
import { motion } from 'framer-motion';
import { 
  TrophyIcon,
  ClockIcon,
  ChartBarIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';

interface MetricsCardsProps {
  modelResults: any;
  taskType?: string;
  className?: string;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({
  modelResults,
  taskType,
  className = '',
}) => {
  if (!modelResults?.metrics) {
    return null;
  }

  const metrics = modelResults.metrics;

  const getMetricCards = () => {
    if (taskType === 'classification') {
      return [
        {
          title: 'Accuracy',
          value: `${(metrics.accuracy * 100).toFixed(1)}%`,
          icon: TrophyIcon,
          color: 'green',
          description: 'Overall prediction accuracy',
        },
        {
          title: 'Precision',
          value: `${(metrics.precision * 100).toFixed(1)}%`,
          icon: ChartBarIcon,
          color: 'blue',
          description: 'Precision of positive predictions',
        },
        {
          title: 'Recall',
          value: `${(metrics.recall * 100).toFixed(1)}%`,
          icon: ChartBarIcon,
          color: 'purple',
          description: 'Sensitivity to positive cases',
        },
        {
          title: 'F1 Score',
          value: `${(metrics.f1_score * 100).toFixed(1)}%`,
          icon: ChartBarIcon,
          color: 'orange',
          description: 'Harmonic mean of precision and recall',
        },
      ];
    } else {
      return [
        {
          title: 'R² Score',
          value: metrics.r2_score?.toFixed(3) || '---',
          icon: TrophyIcon,
          color: 'green',
          description: 'Coefficient of determination',
        },
        {
          title: 'RMSE',
          value: metrics.rmse?.toFixed(3) || '---',
          icon: ChartBarIcon,
          color: 'red',
          description: 'Root mean squared error',
        },
        {
          title: 'MAE',
          value: metrics.mae?.toFixed(3) || '---',
          icon: ChartBarIcon,
          color: 'blue',
          description: 'Mean absolute error',
        },
        {
          title: 'MAPE',
          value: metrics.mape ? `${metrics.mape.toFixed(1)}%` : '---',
          icon: ChartBarIcon,
          color: 'orange',
          description: 'Mean absolute percentage error',
        },
      ];
    }
  };

  const additionalCards = [
    {
      title: 'Training Time',
      value: modelResults.training_time ? `${modelResults.training_time.toFixed(2)}s` : '---',
      icon: ClockIcon,
      color: 'gray',
      description: 'Time taken to train the model',
    },
    {
      title: 'Model Complexity',
      value: modelResults.parameters?.n_estimators || 
             modelResults.parameters?.max_depth || 
             'Auto',
      icon: CpuChipIcon,
      color: 'indigo',
      description: 'Model complexity indicator',
    },
  ];

  const cards = [...getMetricCards(), ...additionalCards];

  const getColorClasses = (color: string) => {
    const colorMap = {
      green: 'bg-green-500 text-green-100',
      blue: 'bg-blue-500 text-blue-100',
      purple: 'bg-purple-500 text-purple-100',
      orange: 'bg-orange-500 text-orange-100',
      red: 'bg-red-500 text-red-100',
      gray: 'bg-gray-500 text-gray-100',
      indigo: 'bg-indigo-500 text-indigo-100',
    };
    return colorMap[color as keyof typeof colorMap] || 'bg-gray-500 text-gray-100';
  };

  const getBgColorClasses = (color: string) => {
    const colorMap = {
      green: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      purple: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
      orange: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
      red: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
      gray: 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800',
      indigo: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800',
    };
    return colorMap[color as keyof typeof colorMap] || 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
  };

  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ${className}`}>
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <motion.div
            key={card.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className={`
              border rounded-lg p-6 transition-all duration-300 hover:shadow-lg
              ${getBgColorClasses(card.color)}
            `}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  {card.title}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {card.value}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {card.description}
                </p>
              </div>
              
              <div className={`
                w-12 h-12 rounded-lg flex items-center justify-center
                ${getColorClasses(card.color)}
              `}>
                <Icon className="w-6 h-6" />
              </div>
            </div>

            {/* Progress Bar for Percentage Metrics */}
            {card.value.includes('%') && (
              <div className="mt-4">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    className={`h-2 rounded-full ${
                      card.color === 'green' ? 'bg-green-500' :
                      card.color === 'blue' ? 'bg-blue-500' :
                      card.color === 'purple' ? 'bg-purple-500' :
                      card.color === 'orange' ? 'bg-orange-500' :
                      'bg-gray-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ 
                      width: `${parseFloat(card.value.replace('%', ''))}%` 
                    }}
                    transition={{ duration: 1, delay: index * 0.1 + 0.5 }}
                  />
                </div>
              </div>
            )}

            {/* Performance Indicator */}
            <div className="mt-3 flex items-center justify-between">
              <div className="flex items-center space-x-1">
                {card.value.includes('%') && (
                  <>
                    {parseFloat(card.value.replace('%', '')) >= 80 && (
                      <div className="flex items-center space-x-1 text-green-600 dark:text-green-400">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-xs font-medium">Excellent</span>
                      </div>
                    )}
                    {parseFloat(card.value.replace('%', '')) >= 60 && 
                     parseFloat(card.value.replace('%', '')) < 80 && (
                      <div className="flex items-center space-x-1 text-yellow-600 dark:text-yellow-400">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                        <span className="text-xs font-medium">Good</span>
                      </div>
                    )}
                    {parseFloat(card.value.replace('%', '')) < 60 && (
                      <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        <span className="text-xs font-medium">Needs Improvement</span>
                      </div>
                    )}
                  </>
                )}
              </div>
              
              {/* Trend Indicator */}
              {index === 0 && (
                <div className="flex items-center space-x-1 text-green-600 dark:text-green-400">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M5.293 7.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L6.707 7.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                  </svg>
                  <span className="text-xs font-medium">Best</span>
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

export default MetricsCards;