import React, { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { 
  SunIcon, 
  MoonIcon,
  Cog6ToothIcon,
  BellIcon,
  UserCircleIcon
} from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';
import { useHealthCheck } from '../hooks/useApi';

interface LayoutProps {
  children: ReactNode;
  className?: string;
}

export const Layout: React.FC<LayoutProps> = ({ children, className = '' }) => {
  const { isDark, toggleTheme } = useTheme();
  const { data: healthData } = useHealthCheck();

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300 ${className}`}>
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Title */}
            <div className="flex items-center space-x-4">
              <motion.div
                className="flex items-center space-x-3"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-white"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M9.5 2A1.5 1.5 0 0 0 8 3.5v1A1.5 1.5 0 0 0 9.5 6h5A1.5 1.5 0 0 0 16 4.5v-1A1.5 1.5 0 0 0 14.5 2h-5z"/>
                    <path d="M6.5 6A1.5 1.5 0 0 0 5 7.5V18a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3V7.5A1.5 1.5 0 0 0 17.5 6h-11z"/>
                    <path d="M8.5 10.5A.5.5 0 0 1 9 10h6a.5.5 0 0 1 0 1H9a.5.5 0 0 1-.5-.5z"/>
                    <path d="M8.5 13.5A.5.5 0 0 1 9 13h6a.5.5 0 0 1 0 1H9a.5.5 0 0 1-.5-.5z"/>
                    <path d="M8.5 16.5A.5.5 0 0 1 9 16h3a.5.5 0 0 1 0 1H9a.5.5 0 0 1-.5-.5z"/>
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900 dark:text-white gradient-text">
                    AutoML Studio
                  </h1>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Automated Machine Learning Platform
                  </p>
                </div>
              </motion.div>
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center space-x-4">
              {/* Health Status */}
              {healthData && (
                <motion.div
                  className="flex items-center space-x-2"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-600 dark:text-gray-400 hidden sm:block">
                    Backend Online
                  </span>
                </motion.div>
              )}

              {/* Theme Toggle */}
              <motion.button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDark ? (
                  <SunIcon className="w-5 h-5 text-yellow-500" />
                ) : (
                  <MoonIcon className="w-5 h-5 text-gray-600" />
                )}
              </motion.button>

              {/* Settings Button */}
              <motion.button
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Settings"
              >
                <Cog6ToothIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </motion.button>

              {/* Notifications */}
              <motion.button
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors relative"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Notifications"
              >
                <BellIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                {/* Notification dot */}
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></div>
              </motion.button>

              {/* User Menu */}
              <motion.button
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="User menu"
              >
                <UserCircleIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </motion.button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          {children}
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                © 2025 AutoML Studio. Built with FastAPI & React.
              </p>
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-gray-600 dark:text-gray-400">
              <a 
                href="#" 
                className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
              >
                Documentation
              </a>
              <a 
                href="#" 
                className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
              >
                API Reference
              </a>
              <a 
                href="#" 
                className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
              >
                Support
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;