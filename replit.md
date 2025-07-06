# AutoML Backend Service

## Overview

This is a production-ready FastAPI-based AutoML backend service that provides comprehensive machine learning capabilities for tabular data. The application automatically handles the entire ML pipeline from data upload to model training, evaluation, and report generation.

## System Architecture

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs with automatic OpenAPI documentation
- **Python 3.11**: Core runtime environment with type hints and async support
- **Uvicorn**: ASGI server for production deployment with hot reload capabilities

### Frontend Framework
- **React 18**: Modern frontend framework with TypeScript support
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **Framer Motion**: Animation library for smooth UI transitions
- **React Query**: Data fetching and caching library for API integration

### ML Stack
- **Scikit-learn**: Primary ML library for preprocessing, training, and evaluation
- **XGBoost**: Gradient boosting framework for advanced model training
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn**: Data visualization and plotting
- **Plotly**: Interactive visualizations for web interface

### File Processing
- **Python-magic**: File type detection and validation
- **Openpyxl**: Excel file support (.xlsx, .xls)
- **Python-multipart**: File upload handling

### Report Generation
- **Jinja2**: Template engine for HTML report generation
- **WeasyPrint**: PDF generation from HTML templates

## Key Components

### API Routes Structure
1. **Upload Route** (`/upload`): File upload, validation, and dataset preview
2. **Training Route** (`/train`): Model training orchestration with background tasks
3. **Evaluation Route** (`/evaluation`): Model performance metrics and comparison
4. **Reports Route** (`/reports`): Report generation and export functionality

### Services Layer
1. **DataService**: Dataset loading, preprocessing, validation, and task detection
2. **MLService**: Model training, evaluation, hyperparameter tuning, and comparison
3. **VisualizationService**: Chart generation, feature importance, confusion matrices
4. **ReportService**: Comprehensive report generation with multiple formats

### Frontend Components
1. **UploadZone**: Drag-and-drop file upload with progress tracking
2. **DataPreviewTable**: Dataset preview with pagination and sorting
3. **TrainingProgress**: Real-time training progress with status updates
4. **EvaluationDashboard**: Model comparison and metrics visualization
5. **ReportsPanel**: Report generation and download interface

### Core Components
1. **Configuration Management**: Centralized settings using Pydantic
2. **Exception Handling**: Custom exception classes with proper error responses
3. **Schema Validation**: Pydantic models for request/response validation
4. **Theme Management**: Dark/light mode support with system preference detection

## Data Flow

1. **File Upload**: Client uploads CSV/Excel → Validation → Session creation → Dataset preview
2. **Task Detection**: Automatic classification/regression detection → Feature analysis → Preprocessing recommendations
3. **Training**: Background training process → Multiple model comparison → Best model selection
4. **Evaluation**: Performance metrics calculation → Visualization generation → Model comparison
5. **Reports**: Comprehensive report generation → Multiple format support → Download functionality

## External Dependencies

### Python Backend Dependencies
- **FastAPI**: Web framework with automatic API documentation
- **Uvicorn**: ASGI server for production deployment
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Advanced gradient boosting models
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Statistical data visualization
- **Jinja2**: Template engine for report generation
- **WeasyPrint**: PDF generation from HTML
- **Python-magic**: File type detection
- **Joblib**: Model serialization and parallel processing

### Frontend Dependencies
- **React**: User interface library with hooks and context
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation and gesture library
- **React Query**: Data fetching and state management
- **React Router**: Client-side routing
- **Axios**: HTTP client for API communication
- **Chart.js**: Data visualization library
- **React Dropzone**: File upload interface

## Deployment Strategy

### Development Environment
- **Dev Container**: Configured with Python 3.11, Node.js 20, and required extensions
- **Hot Reload**: Both backend (Uvicorn) and frontend (React) support hot reload
- **Port Configuration**: Backend on port 5000, frontend on port 3000

### Production Considerations
- **File Storage**: Local filesystem with configurable directories
- **Session Management**: In-memory storage (recommend Redis for production)
- **Error Handling**: Comprehensive exception handling with proper HTTP status codes
- **CORS**: Configured for cross-origin requests
- **Static Files**: Served through FastAPI static file mounting

### Scalability Options
- **Database Integration**: Prepared for PostgreSQL/MongoDB integration
- **Cache Layer**: Ready for Redis integration for session management
- **Background Tasks**: Uses FastAPI background tasks (consider Celery for production)
- **File Processing**: Configurable upload limits and file validation

## Changelog
- July 06, 2025: Initial setup
- July 06, 2025: Comprehensive dataset handling and preprocessing pipeline implementation

## Recent Changes

### July 06, 2025 - Comprehensive Dataset Handling Implementation

#### Extended File Format Support
- Added support for multiple dataset types: CSV, Excel (.xlsx, .xls), JSON, TSV, and ZIP files
- Increased upload limit to 500MB with robust error handling
- Auto-detection of file encoding, delimiters, and headers
- Comprehensive file validation and type checking

#### Advanced Data Analysis & Inspection
- Implemented comprehensive data inspection with automated quality assessment
- Added data quality scoring with issue detection and recommendations
- Auto-run data inspection functions including df.head(), df.tail(), df.info(), df.describe()
- Statistical analysis for numeric and categorical columns
- Outlier detection using IQR and Z-score methods
- Missing value analysis with percentage calculations
- Duplicate detection and reporting

#### Interactive Visualization Dashboard
- Created comprehensive visualization suite with toggle controls
- Auto-generated charts: histograms, box plots, violin plots, scatter plots, pair plots
- Correlation matrices with heatmaps for numeric data
- Count plots and bar charts for categorical data
- QQ plots for normality testing
- Missing values heatmaps
- Target distribution and relationship analysis
- Real-time chart generation with matplotlib and seaborn

#### Preprocessing Pipeline
- Comprehensive data cleaning with configurable options
- Missing value handling (drop columns, imputation strategies)
- Duplicate removal with progress tracking
- Data type fixing and validation
- Outlier treatment using IQR and statistical methods
- Basic feature engineering capabilities
- Data validation and schema consistency checks

#### Enhanced User Interface
- Modern sidebar navigation with clickable step progression
- Dark/light theme support with system preference detection
- Real-time progress tracking and status indicators
- Interactive visualization controls with toggle buttons
- Responsive design for mobile, tablet, and desktop
- Professional UI suitable for business presentations
- Session management with unique identifiers

#### API Enhancements
- New comprehensive data analysis endpoints (/api/v1/data-analysis/*)
- Data inspection API with detailed column analysis
- Visualization generation API with selective chart types
- Preprocessing API with configurable options
- File serving endpoints for generated charts
- Enhanced error handling with meaningful messages

#### Technical Improvements
- Modular service architecture with comprehensive preprocessing service
- Advanced visualization service with multiple chart types
- Session-based data management for concurrent users
- Robust error handling and logging throughout pipeline
- Type-safe API endpoints with Pydantic validation
- Memory-efficient data processing for large datasets

## User Preferences

Preferred communication style: Simple, everyday language.