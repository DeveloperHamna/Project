import React, { useState } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { 
  DocumentArrowDownIcon,
  DocumentTextIcon,
  EyeIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useReportGeneration, useReportsList } from '../hooks/useApi';
import { ReportRequest } from '../types';

interface ReportsPanelProps {
  sessionId: string;
  className?: string;
}

export const ReportsPanel: React.FC<ReportsPanelProps> = ({
  sessionId,
  className = '',
}) => {
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'html' | 'json'>('pdf');
  const [includeVisualizations, setIncludeVisualizations] = useState(true);
  const [includeRawData, setIncludeRawData] = useState(false);

  const generateReportMutation = useReportGeneration();
  const { data: reports, isLoading: isLoadingReports, refetch: refetchReports } = useReportsList(sessionId);

  const handleGenerateReport = async () => {
    const request: ReportRequest = {
      format: selectedFormat,
      include_visualizations: includeVisualizations,
      include_raw_data: includeRawData,
    };

    try {
      await generateReportMutation.mutateAsync({ sessionId, request });
      toast.success(`${selectedFormat.toUpperCase()} report generated successfully!`);
      refetchReports();
    } catch (error) {
      toast.error(`Failed to generate report: ${error.message}`);
    }
  };

  const handleDownloadReport = async (reportId: string, filename: string) => {
    try {
      // This would typically download the file
      toast.success(`Downloading ${filename}...`);
    } catch (error) {
      toast.error(`Failed to download report: ${error.message}`);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf':
        return <DocumentArrowDownIcon className="w-5 h-5 text-red-500" />;
      case 'html':
        return <EyeIcon className="w-5 h-5 text-blue-500" />;
      case 'json':
        return <DocumentTextIcon className="w-5 h-5 text-green-500" />;
      default:
        return <DocumentTextIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const reportFormats = [
    {
      id: 'pdf',
      name: 'PDF Report',
      description: 'Comprehensive report with charts and analysis',
      icon: DocumentArrowDownIcon,
      color: 'red',
    },
    {
      id: 'html',
      name: 'HTML Report',
      description: 'Interactive web-based report',
      icon: EyeIcon,
      color: 'blue',
    },
    {
      id: 'json',
      name: 'JSON Data',
      description: 'Raw data and metrics in JSON format',
      icon: DocumentTextIcon,
      color: 'green',
    },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Generate New Report Section */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Generate Report
        </h3>

        {/* Format Selection */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Report Format
            </label>
            <div className="grid grid-cols-1 gap-2">
              {reportFormats.map((format) => {
                const Icon = format.icon;
                return (
                  <button
                    key={format.id}
                    onClick={() => setSelectedFormat(format.id as any)}
                    className={`
                      p-3 rounded-lg border-2 text-left transition-all duration-200
                      ${selectedFormat === format.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                      }
                    `}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className={`w-5 h-5 ${
                        selectedFormat === format.id 
                          ? 'text-primary-600 dark:text-primary-400' 
                          : 'text-gray-500 dark:text-gray-400'
                      }`} />
                      <div>
                        <div className={`font-medium ${
                          selectedFormat === format.id 
                            ? 'text-primary-900 dark:text-primary-100' 
                            : 'text-gray-900 dark:text-white'
                        }`}>
                          {format.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {format.description}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Options */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Report Options
            </label>
            
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={includeVisualizations}
                  onChange={(e) => setIncludeVisualizations(e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Include visualizations and charts
                </span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={includeRawData}
                  onChange={(e) => setIncludeRawData(e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Include raw dataset
                </span>
              </label>
            </div>
          </div>

          {/* Generate Button */}
          <motion.button
            onClick={handleGenerateReport}
            disabled={generateReportMutation.isPending}
            className="btn-primary w-full flex items-center justify-center space-x-2"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {generateReportMutation.isPending ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <DocumentArrowDownIcon className="w-4 h-4" />
                <span>Generate {selectedFormat.toUpperCase()} Report</span>
              </>
            )}
          </motion.button>
        </div>
      </div>

      {/* Existing Reports Section */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Generated Reports
          </h3>
          <button
            onClick={() => refetchReports()}
            className="text-sm text-primary-600 hover:text-primary-800 dark:text-primary-400 dark:hover:text-primary-300"
          >
            Refresh
          </button>
        </div>

        {isLoadingReports ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
          </div>
        ) : reports && reports.length > 0 ? (
          <div className="space-y-3">
            {reports.map((report, index) => (
              <motion.div
                key={report.report_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getFormatIcon(report.format)}
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {report.filename}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center space-x-3">
                        <span>Generated {new Date(report.generated_at).toLocaleDateString()}</span>
                        <span>•</span>
                        <span>{report.format.toUpperCase()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleDownloadReport(report.report_id, report.filename)}
                      className="btn-secondary text-sm flex items-center space-x-1"
                    >
                      <DocumentArrowDownIcon className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <DocumentTextIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              No reports generated yet. Create your first report above.
            </p>
          </div>
        )}
      </div>

      {/* Report Tips */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <CheckCircleIcon className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300">
              Report Tips
            </h4>
            <ul className="text-sm text-blue-600 dark:text-blue-400 mt-1 space-y-1">
              <li>• PDF reports are best for sharing and presentations</li>
              <li>• HTML reports are interactive and work great for exploration</li>
              <li>• JSON format is perfect for integrating with other tools</li>
              <li>• Including visualizations makes reports more comprehensive</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsPanel;