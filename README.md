# Credit Scoring Model - AutoML Platform

## Overview

A comprehensive AutoML backend service designed for credit scoring model development that automatically processes datasets, trains multiple ML models, and provides evaluation insights via REST APIs. The platform features a modern web interface that requires no coding knowledge.

## System Architecture

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs with automatic OpenAPI documentation
- **Python 3.11+**: Core runtime environment
- **Uvicorn**: ASGI server for production deployment

### ML Stack
- **Scikit-learn**: Primary ML library for preprocessing, training, and evaluation
- **XGBoost**: Gradient boosting framework for advanced model training
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn/Plotly**: Data visualization and plotting

### File Processing
- **Python-magic**: File type detection and validation
- **Multiple format support**: CSV, Excel (.xlsx, .xls) file handling up to 500MB

### Report Generation
- **Jinja2**: Template engine for HTML report generation
- **WeasyPrint**: PDF generation from HTML templates

## Getting Started from Scratch

### Prerequisites
- GitHub Codespaces (recommended) or local development environment
- Python 3.11+
- Node.js 20+

### GitHub Codespaces Setup (Recommended)

1. **Open in Codespaces**
   ```bash
   # Fork this repository and open in GitHub Codespaces
   # The devcontainer will automatically set up the environment
   ```

2. **Environment Setup** (Automatic via devcontainer)
   ```bash
   # These commands run automatically in the devcontainer:
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   cd frontend && npm install
   ```

### Local Development Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd credit-scoring-model
   ```

2. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

3. **Install Frontend Dependencies** (Optional for React development)
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

#### Start Backend Server
```bash
# Option 1: Direct Python execution
python main.py

# Option 2: Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

#### Access the Application
- **Main Application**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Legacy Interface**: http://localhost:5000/docs-old

### Testing the Platform

1. **Upload Test Dataset**
   - Navigate to the main application
   - Upload a CSV or Excel file (max 500MB)
   - Supported formats: .csv, .xlsx, .xls

2. **Follow the Workflow**
   - **Step 1**: Upload Dataset → View data preview and quality analysis
   - **Step 2**: Data Analysis → Review detected task type and feature analysis
   - **Step 3**: Model Training → Automatic training of multiple ML models
   - **Step 4**: Results & Reports → Model comparison and performance metrics

3. **Generate Reports**
   - PDF reports with comprehensive model analysis
   - HTML reports for web viewing
   - JSON exports for programmatic access

## Key Features

### Automated ML Pipeline
- **Task Detection**: Automatic classification/regression determination
- **Data Preprocessing**: Missing value handling, feature encoding, scaling
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Cross-validation**: Robust model evaluation
- **Model Comparison**: Performance metrics and visualization

### Modern Web Interface
- **Dark/Light Theme**: System preference detection and manual toggle
- **Responsive Design**: Mobile, tablet, and desktop support
- **Real-time Progress**: Live training status and progress tracking
- **Interactive Navigation**: Step-by-step workflow with clear progression
- **Professional UI**: Clean, modern interface suitable for business use

### API Features
- **RESTful API**: Comprehensive REST endpoints for all functionality
- **OpenAPI Documentation**: Interactive API docs at `/docs`
- **Session Management**: Multi-session support for concurrent users
- **File Upload**: Robust file handling with progress tracking
- **Error Handling**: Comprehensive error messages and recovery

## Project Structure

```
.
├── .devcontainer/              # GitHub Codespaces configuration
│   └── devcontainer.json      # Development container setup
├── app/                       # Backend application
│   ├── api/                   # API route handlers
│   ├── core/                  # Core configuration and exceptions
│   ├── models/                # Pydantic schemas
│   ├── services/              # Business logic services
│   └── utils/                 # Utility functions
├── frontend/                  # React frontend (development version)
│   ├── src/                   # Source code
│   ├── public/                # Static assets
│   └── package.json           # Node.js dependencies
├── static/                    # Production web interface
│   └── automl-app.html       # Main application interface
├── uploads/                   # Uploaded dataset files
├── models/                    # Trained model artifacts
├── visualizations/            # Generated charts and plots
├── reports/                   # Generated reports
├── main.py                    # Application entry point
├── requirements-dev.txt       # Python dependencies
└── README.md                  # This file
```

## API Endpoints

### Dataset Management
- `POST /api/v1/upload` - Upload dataset
- `GET /api/v1/dataset/{session_id}/preview` - Get dataset preview
- `POST /api/v1/dataset/{session_id}/detect-task` - Detect task type

### Model Training
- `POST /api/v1/train/{session_id}` - Start training
- `GET /api/v1/training/{job_id}/status` - Get training status

### Model Evaluation
- `GET /api/v1/evaluation/{session_id}` - Get evaluation results
- `GET /api/v1/evaluation/{session_id}/comparison` - Compare models

### Reports
- `POST /api/v1/reports/{session_id}/generate` - Generate report
- `GET /api/v1/reports/{session_id}/download/{report_id}` - Download report

### Utilities
- `GET /health` - Health check endpoint

## Configuration

### Environment Variables
```bash
# Optional configuration
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
```

### Application Settings
- **File Upload**: 500MB maximum file size
- **Supported Formats**: CSV, Excel (.xlsx, .xls)
- **Model Training**: Automatic hyperparameter tuning
- **Cross-validation**: 5-fold CV by default
- **Feature Handling**: Automatic preprocessing

## Development

### Code Structure
- **Backend**: FastAPI with async/await support
- **Frontend**: Vanilla HTML/CSS/JavaScript for production, React for development
- **Database**: File-based storage (can be migrated to PostgreSQL)
- **Caching**: In-memory session storage (Redis recommended for production)

### Adding New Features
1. Update Pydantic schemas in `app/models/`
2. Add business logic in `app/services/`
3. Create API routes in `app/api/`
4. Update frontend interface in `static/automl-app.html`

### Testing
```bash
# Run backend tests
python -m pytest

# Check API health
curl http://localhost:5000/health

# Test file upload
curl -X POST -F "file=@test.csv" http://localhost:5000/api/v1/upload
```

## Deployment

### GitHub Codespaces (Development)
- Automatic setup via devcontainer
- Hot-reload enabled
- Port forwarding configured

### Production Deployment
```bash
# Using Docker (create Dockerfile)
FROM python:3.11-slim
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY . .
EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

# Build and run
docker build -t credit-scoring-model .
docker run -p 5000:5000 credit-scoring-model
```

### Environment Setup
- **Scalability**: Session-based architecture supports multiple concurrent users
- **Storage**: File system-based storage (cloud storage recommended for production)
- **Caching**: Redis recommended for production session management
- **Security**: CORS configuration and comprehensive file validation
- **Monitoring**: Structured logging throughout the application

## Troubleshooting

### Common Issues

1. **Upload Fails with Network Error**
   - Check backend server is running on port 5000
   - Verify file size is under 500MB
   - Check file format is CSV or Excel

2. **Training Doesn't Start**
   - Ensure dataset was uploaded successfully
   - Check target column detection
   - Verify sufficient data quality

3. **Port Already in Use**
   ```bash
   # Kill process on port 5000
   lsof -ti:5000 | xargs kill -9
   
   # Or use different port
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Python Dependencies Issues**
   ```bash
   # Clean install
   pip uninstall -y -r requirements-dev.txt
   pip install -r requirements-dev.txt
   ```

### Performance Optimization
- **Large Files**: Use streaming for files > 100MB
- **Model Training**: Adjust `N_JOBS` in configuration for CPU usage
- **Memory Usage**: Monitor RAM usage during training with large datasets

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push and create a pull request

## Recent Changes

### July 06, 2025 - GitHub Codespaces Migration

#### Major Refactoring for Codespaces
- Migrated from Replit-specific setup to GitHub Codespaces
- Created comprehensive devcontainer configuration
- Removed all Replit dependencies and references
- Fixed upload network errors with improved error handling
- Enhanced navigation with Next buttons between pipeline steps

#### Upload & Navigation Improvements
- Fixed "network error" issues during dataset upload
- Added clear dataset details display (columns, size, preview)
- Implemented intuitive navigation between pipeline steps
- Added Next buttons for seamless workflow progression
- Enhanced API error handling with meaningful messages

#### DevContainer Setup
- Complete `.devcontainer/devcontainer.json` configuration
- Automatic Python and Node.js environment setup
- Pre-configured VS Code extensions for development
- Port forwarding setup for backend (5000) and frontend (3000)
- Automatic dependency installation on container creation

#### Documentation Overhaul
- Renamed `replit.md` to `README.md`
- Removed all Replit references and instructions
- Added comprehensive "Getting Started from Scratch" section
- Included GitHub Codespaces setup instructions
- Added troubleshooting and deployment sections

#### Technical Improvements
- Enhanced file upload with 500MB limit (updated from 100MB)
- Improved error handling with timeout and network error detection
- Added navigation functions for step-by-step workflow
- Fixed API endpoints and response validation
- Optimized for seamless GitHub Codespaces development

### Previous History
- July 06, 2025: Complete AutoML platform with modern frontend interface
- July 06, 2025: Initial backend setup with full ML pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review GitHub Issues for similar problems
3. Create a new issue with detailed information about your problem

## Acknowledgments

- Created by Hamna
- Built with FastAPI, React, and Scikit-learn
- Designed for GitHub Codespaces development environment