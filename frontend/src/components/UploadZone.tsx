import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CloudArrowUpIcon, 
  DocumentTextIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XMarkIcon 
} from '@heroicons/react/24/outline';
import { useUpload } from '../hooks/useApi';
import { FileUpload } from '../types';

interface UploadZoneProps {
  onUploadSuccess: (sessionId: string) => void;
  onUploadError: (error: string) => void;
  className?: string;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onUploadSuccess,
  onUploadError,
  className = '',
}) => {
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const uploadMutation = useUpload();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const fileUpload: FileUpload = {
        file,
        target_column: targetColumn || undefined,
      };

      uploadMutation.mutate(fileUpload, {
        onSuccess: (data) => {
          onUploadSuccess(data.session_id);
        },
        onError: (error) => {
          onUploadError(error.message);
        },
      });
    }
  }, [targetColumn, uploadMutation, onUploadSuccess, onUploadError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    onDropAccepted: () => setDragActive(false),
    onDropRejected: () => setDragActive(false),
  });

  const getFileTypeIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'csv':
        return <DocumentTextIcon className="w-6 h-6 text-green-500" />;
      case 'xlsx':
      case 'xls':
        return <DocumentTextIcon className="w-6 h-6 text-blue-500" />;
      default:
        return <DocumentTextIcon className="w-6 h-6 text-gray-500" />;
    }
  };

  const isUploading = uploadMutation.isPending;
  const isSuccess = uploadMutation.isSuccess;
  const isError = uploadMutation.isError;

  return (
    <div className={`w-full ${className}`}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Target Column Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Target Column (Optional)
          </label>
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="Enter target column name"
            className="input"
            disabled={isUploading}
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Leave empty to auto-detect the target column
          </p>
        </div>

        {/* Upload Zone */}
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-all duration-300 ease-in-out
            ${isDragActive || dragActive 
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 scale-105' 
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }
            ${isUploading ? 'cursor-not-allowed opacity-75' : ''}
            ${isSuccess ? 'border-green-500 bg-green-50 dark:bg-green-900/20' : ''}
            ${isError ? 'border-red-500 bg-red-50 dark:bg-red-900/20' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <AnimatePresence mode="wait">
            {isUploading ? (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="space-y-4"
              >
                <div className="w-16 h-16 mx-auto">
                  <div className="w-full h-full border-4 border-primary-200 border-t-primary-500 rounded-full animate-spin"></div>
                </div>
                <div>
                  <p className="text-lg font-semibold text-primary-600 dark:text-primary-400">
                    Uploading your dataset...
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Please wait while we process your file
                  </p>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    className="bg-primary-500 h-2 rounded-full progress-bar"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {uploadProgress}% complete
                </p>
              </motion.div>
            ) : isSuccess ? (
              <motion.div
                key="success"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="space-y-4"
              >
                <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto" />
                <div>
                  <p className="text-lg font-semibold text-green-600 dark:text-green-400">
                    Upload successful!
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Your dataset has been processed and is ready for analysis
                  </p>
                </div>
              </motion.div>
            ) : isError ? (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="space-y-4"
              >
                <ExclamationTriangleIcon className="w-16 h-16 text-red-500 mx-auto" />
                <div>
                  <p className="text-lg font-semibold text-red-600 dark:text-red-400">
                    Upload failed
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {uploadMutation.error?.message || 'An error occurred during upload'}
                  </p>
                </div>
                <button
                  onClick={() => uploadMutation.reset()}
                  className="btn-secondary text-sm"
                >
                  Try Again
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="default"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="space-y-4"
              >
                <CloudArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto" />
                <div>
                  <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    {isDragActive ? 'Drop your file here' : 'Upload your dataset'}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Drag and drop your CSV or Excel file, or click to browse
                  </p>
                </div>
                
                {/* Supported formats */}
                <div className="flex items-center justify-center space-x-6 text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center space-x-1">
                    <DocumentTextIcon className="w-4 h-4" />
                    <span>CSV</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <DocumentTextIcon className="w-4 h-4" />
                    <span>Excel</span>
                  </div>
                  <div className="text-gray-300 dark:text-gray-600">•</div>
                  <span>Max 100MB</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* File Requirements */}
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
            File Requirements:
          </h4>
          <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
            <li>• Supported formats: CSV, Excel (.xlsx, .xls)</li>
            <li>• Maximum file size: 100MB</li>
            <li>• First row should contain column headers</li>
            <li>• Data should be in tabular format</li>
            <li>• Missing values are automatically handled</li>
          </ul>
        </div>
      </motion.div>
    </div>
  );
};

export default UploadZone;