import React from 'react';
import { motion } from 'framer-motion';
import { 
  CloudArrowUpIcon,
  CogIcon,
  ChartBarIcon,
  DocumentArrowDownIcon,
  CheckIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { StepperState, StepInfo } from '../types';

interface StepperProgressProps {
  currentStep: number;
  completedSteps: number[];
  steps: StepInfo[];
  className?: string;
}

export const StepperProgress: React.FC<StepperProgressProps> = ({
  currentStep,
  completedSteps,
  steps,
  className = '',
}) => {
  const getStepIcon = (step: StepInfo) => {
    switch (step.id) {
      case 1:
        return <CloudArrowUpIcon className="w-5 h-5" />;
      case 2:
        return <CogIcon className="w-5 h-5" />;
      case 3:
        return <ChartBarIcon className="w-5 h-5" />;
      case 4:
        return <DocumentArrowDownIcon className="w-5 h-5" />;
      default:
        return <div className="w-5 h-5" />;
    }
  };

  const getStepStatus = (step: StepInfo) => {
    if (completedSteps.includes(step.id)) {
      return 'completed';
    }
    if (step.id === currentStep) {
      return step.status === 'error' ? 'error' : 'active';
    }
    return 'inactive';
  };

  const getStepClasses = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500 border-green-500 text-white';
      case 'active':
        return 'bg-primary-500 border-primary-500 text-white';
      case 'error':
        return 'bg-red-500 border-red-500 text-white';
      default:
        return 'bg-gray-200 border-gray-200 text-gray-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-400';
    }
  };

  const getConnectorClasses = (stepIndex: number) => {
    const isCompleted = completedSteps.includes(steps[stepIndex].id);
    const isActive = steps[stepIndex].id === currentStep;
    
    if (isCompleted || (isActive && stepIndex < steps.length - 1)) {
      return 'bg-primary-500';
    }
    return 'bg-gray-200 dark:bg-gray-600';
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const status = getStepStatus(step);
          const stepClasses = getStepClasses(status);
          
          return (
            <div key={step.id} className="flex items-center flex-1">
              {/* Step Circle */}
              <div className="relative">
                <motion.div
                  className={`
                    w-12 h-12 rounded-full flex items-center justify-center text-sm font-medium border-2
                    ${stepClasses}
                    transition-all duration-300
                  `}
                  initial={{ scale: 0.8 }}
                  animate={{ 
                    scale: status === 'active' ? 1.1 : 1,
                    rotate: status === 'active' ? 360 : 0
                  }}
                  transition={{ 
                    duration: 0.3,
                    rotate: { duration: 0.5 }
                  }}
                >
                  {status === 'completed' ? (
                    <CheckIcon className="w-6 h-6" />
                  ) : status === 'error' ? (
                    <ExclamationTriangleIcon className="w-6 h-6" />
                  ) : (
                    getStepIcon(step)
                  )}
                </motion.div>
                
                {/* Active Step Glow */}
                {status === 'active' && (
                  <motion.div
                    className="absolute inset-0 rounded-full bg-primary-500 opacity-20"
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.2, 0.4, 0.2],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                )}
              </div>
              
              {/* Step Info */}
              <div className="ml-4 flex-1">
                <div className="flex items-center">
                  <h3 className={`
                    text-sm font-medium
                    ${status === 'active' ? 'text-primary-600 dark:text-primary-400' : ''}
                    ${status === 'completed' ? 'text-green-600 dark:text-green-400' : ''}
                    ${status === 'error' ? 'text-red-600 dark:text-red-400' : ''}
                    ${status === 'inactive' ? 'text-gray-500 dark:text-gray-400' : ''}
                  `}>
                    {step.title}
                  </h3>
                  
                  {/* Status Badge */}
                  {status === 'active' && (
                    <motion.div
                      className="ml-2 px-2 py-1 text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200 rounded-full"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      In Progress
                    </motion.div>
                  )}
                  
                  {status === 'completed' && (
                    <motion.div
                      className="ml-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded-full"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      Completed
                    </motion.div>
                  )}
                  
                  {status === 'error' && (
                    <motion.div
                      className="ml-2 px-2 py-1 text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 rounded-full"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      Error
                    </motion.div>
                  )}
                </div>
                
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  {step.description}
                </p>
              </div>
              
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div className="mx-4 flex-1 h-0.5 relative">
                  <div className="absolute inset-0 bg-gray-200 dark:bg-gray-600 rounded-full"></div>
                  <motion.div
                    className={`absolute inset-0 rounded-full ${getConnectorClasses(index)}`}
                    initial={{ scaleX: 0 }}
                    animate={{ 
                      scaleX: completedSteps.includes(steps[index].id) ? 1 : 0 
                    }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    style={{ transformOrigin: 'left' }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Progress Bar */}
      <div className="mt-8 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <motion.div
          className="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full"
          initial={{ width: 0 }}
          animate={{ 
            width: `${(completedSteps.length / steps.length) * 100}%` 
          }}
          transition={{ duration: 0.5 }}
        />
      </div>
      
      {/* Progress Text */}
      <div className="mt-2 flex justify-between text-xs text-gray-600 dark:text-gray-400">
        <span>Progress: {completedSteps.length} of {steps.length} steps completed</span>
        <span>{Math.round((completedSteps.length / steps.length) * 100)}%</span>
      </div>
    </div>
  );
};

// Default steps configuration
export const DEFAULT_STEPS: StepInfo[] = [
  {
    id: 1,
    title: 'Upload Dataset',
    description: 'Upload your CSV or Excel file',
    status: 'pending',
  },
  {
    id: 2,
    title: 'Data Analysis',
    description: 'Analyze data quality and detect task type',
    status: 'pending',
  },
  {
    id: 3,
    title: 'Model Training',
    description: 'Train multiple ML models automatically',
    status: 'pending',
  },
  {
    id: 4,
    title: 'Results & Reports',
    description: 'View evaluation results and download reports',
    status: 'pending',
  },
];

export default StepperProgress;