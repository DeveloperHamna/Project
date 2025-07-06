import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PlayIcon,
  StopIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { TrainingStatus } from '../types';

interface TrainingProgressProps {
  isTraining: boolean;
  trainingStatus?: TrainingStatus | null;
  onStop: () => void;
  sessionId: string | null;
  className?: string;
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  isTraining,
  trainingStatus,
  onStop,
  sessionId,
  className = '',
}) => {
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [estimatedTotal, setEstimatedTotal] = useState(0);

  // Timer for elapsed time
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isTraining) {
      interval = setInterval(() => {
        setTimeElapsed(prev => prev + 1);
      }, 1000);
    } else {
      setTimeElapsed(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining]);

  // Update estimated total time
  useEffect(() => {
    if (trainingStatus?.estimated_time_remaining) {
      setEstimatedTotal(timeElapsed + trainingStatus.estimated_time_remaining);
    }
  }, [trainingStatus?.estimated_time_remaining, timeElapsed]);

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 dark:text-green-400';
      case 'failed':
        return 'text-red-600 dark:text-red-400';
      case 'running':
        return 'text-primary-600 dark:text-primary-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <ExclamationTriangleIcon className="w-6 h-6 text-red-500" />;
      case 'running':
        return <CpuChipIcon className="w-6 h-6 text-primary-500 animate-pulse" />;
      default:
        return <ClockIcon className="w-6 h-6 text-gray-500" />;
    }
  };

  const progress = trainingStatus?.progress || 0;
  const status = trainingStatus?.status || 'pending';

  if (!sessionId) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <p className="text-gray-500 dark:text-gray-400">
          Please upload a dataset first to start training.
        </p>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Training Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon(status)}
          <div>
            <h3 className={`text-lg font-semibold ${getStatusColor(status)}`}>
              {status === 'completed' && 'Training Completed!'}
              {status === 'failed' && 'Training Failed'}
              {status === 'running' && 'Training in Progress...'}
              {status === 'pending' && 'Ready to Train'}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {trainingStatus?.message || 'Click start to begin training multiple models'}
            </p>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center space-x-2">
          {!isTraining && status !== 'completed' && (
            <motion.button
              onClick={() => {/* Start training logic handled by parent */}}
              className="btn-primary flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              disabled={!sessionId}
            >
              <PlayIcon className="w-4 h-4" />
              <span>Start Training</span>
            </motion.button>
          )}

          {isTraining && (
            <motion.button
              onClick={onStop}
              className="btn-error flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <StopIcon className="w-4 h-4" />
              <span>Stop Training</span>
            </motion.button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600 dark:text-gray-400">
            Overall Progress
          </span>
          <span className="font-medium text-gray-900 dark:text-gray-100">
            {progress}%
          </span>
        </div>
        
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full flex items-center justify-end pr-1"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            {progress > 10 && (
              <div className="w-2 h-2 bg-white rounded-full opacity-80 animate-pulse"></div>
            )}
          </motion.div>
        </div>
      </div>

      {/* Current Step Information */}
      {trainingStatus && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Current Step */}
            <div className="flex items-center space-x-2">
              <ChartBarIcon className="w-5 h-5 text-primary-500" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Current Step</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {trainingStatus.current_step || 'Initializing...'}
                </p>
              </div>
            </div>

            {/* Time Information */}
            <div className="flex items-center space-x-2">
              <ClockIcon className="w-5 h-5 text-gray-500" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Time Elapsed</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {formatTime(timeElapsed)}
                </p>
              </div>
            </div>

            {/* Estimated Remaining */}
            {trainingStatus.estimated_time_remaining && (
              <div className="flex items-center space-x-2">
                <ClockIcon className="w-5 h-5 text-orange-500" />
                <div>
                  <p className="text-xs text-gray-600 dark:text-gray-400">Est. Remaining</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">
                    {formatTime(trainingStatus.estimated_time_remaining)}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Step Progress */}
          {trainingStatus.total_steps && (
            <div className="mt-4">
              <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                <span>Step Progress</span>
                <span>
                  Step {Math.ceil((progress / 100) * trainingStatus.total_steps)} of {trainingStatus.total_steps}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1">
                <div 
                  className="h-1 bg-primary-400 rounded-full transition-all duration-300"
                  style={{ width: `${(Math.ceil((progress / 100) * trainingStatus.total_steps) / trainingStatus.total_steps) * 100}%` }}
                />
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Training Models List */}
      {isTraining && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
        >
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
            Training Models
          </h4>
          
          <div className="space-y-2">
            {[
              { name: 'Random Forest', status: progress > 20 ? 'completed' : progress > 0 ? 'running' : 'pending' },
              { name: 'XGBoost', status: progress > 40 ? 'completed' : progress > 20 ? 'running' : 'pending' },
              { name: 'Linear Model', status: progress > 60 ? 'completed' : progress > 40 ? 'running' : 'pending' },
              { name: 'Support Vector Machine', status: progress > 80 ? 'completed' : progress > 60 ? 'running' : 'pending' },
              { name: 'Neural Network', status: progress > 95 ? 'completed' : progress > 80 ? 'running' : 'pending' },
            ].map((model, index) => (
              <div key={model.name} className="flex items-center justify-between py-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{model.name}</span>
                <div className="flex items-center space-x-2">
                  {model.status === 'completed' && (
                    <CheckCircleIcon className="w-4 h-4 text-green-500" />
                  )}
                  {model.status === 'running' && (
                    <div className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"></div>
                  )}
                  {model.status === 'pending' && (
                    <div className="w-4 h-4 border-2 border-gray-300 dark:border-gray-600 rounded-full"></div>
                  )}
                  <span className={`text-xs font-medium ${
                    model.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                    model.status === 'running' ? 'text-primary-600 dark:text-primary-400' :
                    'text-gray-500 dark:text-gray-400'
                  }`}>
                    {model.status === 'completed' ? 'Done' :
                     model.status === 'running' ? 'Training...' :
                     'Waiting'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Training Complete Message */}
      {status === 'completed' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4"
        >
          <div className="flex items-center space-x-3">
            <CheckCircleIcon className="w-8 h-8 text-green-500" />
            <div>
              <h4 className="text-lg font-semibold text-green-800 dark:text-green-300">
                Training Completed Successfully!
              </h4>
              <p className="text-sm text-green-600 dark:text-green-400">
                All models have been trained and evaluated. You can now view the results and download reports.
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Training Failed Message */}
      {status === 'failed' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
        >
          <div className="flex items-center space-x-3">
            <ExclamationTriangleIcon className="w-8 h-8 text-red-500" />
            <div>
              <h4 className="text-lg font-semibold text-red-800 dark:text-red-300">
                Training Failed
              </h4>
              <p className="text-sm text-red-600 dark:text-red-400">
                {trainingStatus?.message || 'An error occurred during training. Please try again.'}
              </p>
              <button className="btn-primary text-sm mt-2">
                Retry Training
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default TrainingProgress;