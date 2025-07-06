# AutoML Backend Service

## Overview

This is a production-ready FastAPI-based AutoML backend service that provides comprehensive machine learning capabilities for tabular data. The application automatically handles the entire ML pipeline from data ingestion to model deployment, offering a REST API interface for frontend integration.

## System Architecture

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs with automatic OpenAPI documentation
- **Python 3.8+**: Core runtime environment
- **Uvicorn**: ASGI server for production deployment

### ML Stack
- **Scikit-learn**: Primary ML library for preprocessing, training, and evaluation
- **XGBoost**: Gradient boosting framework for advanced model training
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn**: Data visualization and plotting

### File Processing
- **Python-magic**: File type detection and validation
- **Multiple format support**: CSV, Excel (.xlsx, .xls) file handling

### Report Generation
- **Jinja2**: Template engine for HTML report generation
- **WeasyPrint**: PDF generation from HTML templates

## Key Components

### API Routes Structure
1. **Upload Route** (`/upload`): File upload and dataset preview
2. **Training Route** (`/train`): Model training orchestration
3. **Evaluation Route** (`/evaluation`): Model performance metrics
4. **Reports Route** (`/reports`): Report generation and export

### Services Layer
1. **DataService**: Dataset loading, preprocessing, and validation
2. **MLService**: Model training, evaluation, and comparison
3. **VisualizationService**: Chart and plot generation
4. **ReportService**: Comprehensive report generation

### Core Components
1. **Configuration Management**: Centralized settings using Pydantic
2. **Exception Handling**: Custom exception classes for better error management
3. **Schema Validation**: Pydantic models for request/response validation
4. **Utility Functions**: File handling and ML utility functions

## Data Flow

1. **File Upload**: Client uploads CSV/Excel → Validation → Session creation
2. **Data Preview**: File parsing → Column analysis → Target detection
3. **Task Detection**: Automatic classification/regression determination
4. **Data Preprocessing**: Missing value handling → Feature encoding → Scaling
5. **Model Training**: Multiple algorithms → Hyperparameter tuning → Cross-validation
6. **Model Evaluation**: Performance metrics → Visualization generation
7. **Report Generation**: Comprehensive results → Export formats (PDF/HTML/JSON)

## External Dependencies

### Core ML Libraries
- scikit-learn: Machine learning algorithms and utilities
- xgboost: Gradient boosting framework
- pandas: Data manipulation
- numpy: Numerical computing

### Visualization
- matplotlib: Static plotting
- seaborn: Statistical data visualization

### File Processing
- python-magic: File type detection
- openpyxl: Excel file handling

### Report Generation
- jinja2: Template rendering
- weasyprint: PDF generation

### Web Framework
- fastapi: API framework
- uvicorn: ASGI server
- pydantic: Data validation

## Deployment Strategy

### Development Environment
- Direct Python execution with uvicorn
- Hot-reload enabled for development
- Local file storage for uploads and models

### Production Considerations
- **Scalability**: Session-based architecture supports multiple concurrent users
- **Storage**: File system-based storage (can be migrated to cloud storage)
- **Caching**: In-memory session storage (recommended: Redis for production)
- **Security**: CORS configuration and file validation
- **Monitoring**: Structured logging throughout the application

### Directory Structure
```
uploads/          # Uploaded dataset files
models/           # Trained model artifacts
visualizations/   # Generated charts and plots
reports/          # Generated reports
static/           # Static web assets
```

### Configuration Management
- Environment-based configuration using Pydantic Settings
- Configurable limits for file uploads and processing
- Adjustable ML parameters (CV folds, test size, etc.)

## Recent Changes

### July 06, 2025 - App Rebranding to Credit Scoring Model

#### Application Name Update
- Changed application name from "AutoML Studio" to "Credit Scoring Model"
- Updated main frontend title and branding in static/automl-app.html
- Updated backend service name in configuration and FastAPI app
- Modified backend description to focus on credit scoring model use case
- Changed subtitle from "Automated Machine Learning Platform" to "Created by Hamna"
- Increased maximum upload file size from 100MB to 500MB
- All functionality remains the same, only branding and limits updated

### July 06, 2025 - Complete AutoML Platform with Modern Frontend

#### Frontend Implementation
- Built comprehensive modern frontend interface with Tailwind CSS
- Implemented dark theme by default with light mode toggle
- Created responsive design supporting mobile, tablet, and desktop
- Added smooth animations and transitions throughout the interface
- Integrated real-time progress tracking and status updates

#### Key Frontend Features
- **Interactive Upload Zone**: Drag-and-drop file upload with progress tracking
- **Smart Progress Stepper**: Visual 4-step workflow (Upload → Analysis → Training → Results)
- **Live Data Preview**: Interactive table with sorting, pagination, and data quality indicators
- **Real-time Training Progress**: Animated progress bars with model-by-model status
- **Comprehensive Results Dashboard**: Model comparison, metrics visualization, and performance insights
- **Report Generation**: PDF/HTML/JSON report generation with download capabilities
- **Theme Management**: Dark/light mode with system preference detection
- **Toast Notifications**: User-friendly success/error messaging
- **Session Management**: Multi-session support with session info sidebar

#### Technical Architecture
- **Single-page Application**: Built as modern HTML5 app with vanilla JavaScript
- **API Integration**: Direct integration with FastAPI backend via fetch API
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Real-time Updates**: Polling-based training progress monitoring
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Accessibility**: Keyboard navigation and screen reader support

#### Backend Enhancements
- Updated main route to serve modern frontend (`/automl-app.html`)
- Maintained backward compatibility with documentation interface (`/docs-old`)
- All existing API endpoints remain functional
- Enhanced CORS support for frontend integration

#### File Structure
```
static/
├── automl-app.html       # Modern AutoML frontend interface
└── index.html           # Legacy documentation interface

frontend/                 # React components (development version)
├── src/
│   ├── components/      # Reusable UI components
│   ├── services/        # API service layer
│   ├── hooks/          # Custom React hooks
│   ├── contexts/       # React context providers
│   └── types/          # TypeScript type definitions
└── public/             # Static assets
```

#### User Experience Improvements
- **Zero Configuration**: No setup required - upload dataset and start training
- **Visual Feedback**: Real-time progress indicators and status updates
- **Intuitive Navigation**: Clear step-by-step workflow progression
- **Professional UI**: Clean, modern interface suitable for business use
- **Mobile Ready**: Fully responsive design for all device sizes

#### Performance Features
- **Efficient Polling**: Smart training progress monitoring
- **Optimized Loading**: Fast initial load with progressive enhancement
- **Memory Management**: Proper cleanup of intervals and event listeners
- **Caching Strategy**: Local storage for theme preferences and session data

## Changelog

- July 06, 2025. Complete AutoML platform with modern frontend interface
- July 06, 2025. Initial backend setup with full ML pipeline

## User Preferences

Preferred communication style: Simple, everyday language.