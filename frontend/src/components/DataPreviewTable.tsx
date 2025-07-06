import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpDownIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { DatasetPreview, TaskDetectionResponse } from '../types';

interface DataPreviewTableProps {
  datasetPreview: DatasetPreview;
  taskDetection?: TaskDetectionResponse;
  className?: string;
}

export const DataPreviewTable: React.FC<DataPreviewTableProps> = ({
  datasetPreview,
  taskDetection,
  className = '',
}) => {
  const [currentPage, setCurrentPage] = useState(0);
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);

  const itemsPerPage = 10;
  const totalPages = Math.ceil(datasetPreview.sample_data.length / itemsPerPage);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortColumn) return datasetPreview.sample_data;
    
    return [...datasetPreview.sample_data].sort((a, b) => {
      const aValue = a[sortColumn];
      const bValue = b[sortColumn];
      
      if (aValue == null && bValue == null) return 0;
      if (aValue == null) return 1;
      if (bValue == null) return -1;
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      const aStr = String(aValue);
      const bStr = String(bValue);
      
      return sortDirection === 'asc' 
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
  }, [datasetPreview.sample_data, sortColumn, sortDirection]);

  // Paginate data
  const paginatedData = useMemo(() => {
    const start = currentPage * itemsPerPage;
    const end = start + itemsPerPage;
    return sortedData.slice(start, end);
  }, [sortedData, currentPage, itemsPerPage]);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const getColumnType = (column: string) => {
    if (datasetPreview.numerical_columns.includes(column)) return 'numerical';
    if (datasetPreview.categorical_columns.includes(column)) return 'categorical';
    return 'unknown';
  };

  const getColumnTypeIcon = (column: string) => {
    const type = getColumnType(column);
    switch (type) {
      case 'numerical':
        return <span className="text-blue-500 font-mono text-xs">123</span>;
      case 'categorical':
        return <span className="text-purple-500 font-mono text-xs">ABC</span>;
      default:
        return <span className="text-gray-500 font-mono text-xs">?</span>;
    }
  };

  const getColumnBadgeClass = (column: string) => {
    if (column === datasetPreview.target_column) {
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    }
    if (datasetPreview.missing_values[column] > 0) {
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    }
    return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
  };

  const formatValue = (value: any) => {
    if (value == null) return <span className="text-gray-400 italic">null</span>;
    if (typeof value === 'number') {
      return Number.isInteger(value) ? value : value.toFixed(3);
    }
    return String(value);
  };

  return (
    <div className={`w-full ${className}`}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden"
      >
        {/* Header */}
        <div className="px-6 py-4 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Dataset Preview
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {datasetPreview.filename} • {datasetPreview.shape[0]} rows × {datasetPreview.shape[1]} columns
              </p>
            </div>
            
            {/* Task Detection Badge */}
            {taskDetection && (
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1">
                  <CheckCircleIcon className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Task Detected:
                  </span>
                </div>
                <span className={`
                  px-3 py-1 rounded-full text-xs font-medium
                  ${taskDetection.task_type === 'classification' 
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                  }
                `}>
                  {taskDetection.task_type.charAt(0).toUpperCase() + taskDetection.task_type.slice(1)}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  ({Math.round(taskDetection.confidence_score * 100)}% confidence)
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Data Quality Summary */}
        <div className="px-6 py-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <InformationCircleIcon className="w-4 h-4 text-blue-500" />
                <span className="text-gray-600 dark:text-gray-400">
                  Memory: {(datasetPreview.memory_usage / 1024).toFixed(1)} KB
                </span>
              </div>
              
              <div className="flex items-center space-x-1">
                <span className="text-gray-600 dark:text-gray-400">
                  Missing Values: {Object.values(datasetPreview.missing_values).reduce((a, b) => a + b, 0)}
                </span>
                {Object.values(datasetPreview.missing_values).reduce((a, b) => a + b, 0) > 0 && (
                  <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500" />
                )}
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-gray-600 dark:text-gray-400">
                Showing {currentPage * itemsPerPage + 1}-{Math.min((currentPage + 1) * itemsPerPage, datasetPreview.sample_data.length)} of {datasetPreview.sample_data.length} rows
              </span>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                {datasetPreview.columns.map((column) => (
                  <th
                    key={column}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                    onClick={() => handleSort(column)}
                  >
                    <div className="flex items-center space-x-2">
                      <div className="flex items-center space-x-1">
                        {getColumnTypeIcon(column)}
                        <span>{column}</span>
                      </div>
                      
                      {/* Sort Icon */}
                      <ChevronUpDownIcon className={`
                        w-4 h-4 transition-colors
                        ${sortColumn === column 
                          ? 'text-primary-500' 
                          : 'text-gray-400 group-hover:text-gray-600'
                        }
                      `} />
                      
                      {/* Column Badges */}
                      <div className="flex items-center space-x-1">
                        {column === datasetPreview.target_column && (
                          <span className="px-1.5 py-0.5 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded">
                            Target
                          </span>
                        )}
                        
                        {datasetPreview.missing_values[column] > 0 && (
                          <span className="px-1.5 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 rounded">
                            {datasetPreview.missing_values[column]} missing
                          </span>
                        )}
                      </div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-600">
              {paginatedData.map((row, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  {datasetPreview.columns.map((column) => (
                    <td
                      key={column}
                      className={`
                        px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100
                        ${column === datasetPreview.target_column 
                          ? 'bg-green-50 dark:bg-green-900/20 font-medium' 
                          : ''
                        }
                      `}
                    >
                      {formatValue(row[column])}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-6 py-4 bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                <span>Page {currentPage + 1} of {totalPages}</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                  disabled={currentPage === 0}
                  className="p-2 rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeftIcon className="w-4 h-4" />
                </button>
                
                {/* Page numbers */}
                <div className="flex items-center space-x-1">
                  {Array.from({ length: totalPages }, (_, i) => (
                    <button
                      key={i}
                      onClick={() => setCurrentPage(i)}
                      className={`
                        px-3 py-1 rounded-md text-sm font-medium transition-colors
                        ${i === currentPage
                          ? 'bg-primary-500 text-white'
                          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600'
                        }
                      `}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
                
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                  disabled={currentPage === totalPages - 1}
                  className="p-2 rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRightIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default DataPreviewTable;