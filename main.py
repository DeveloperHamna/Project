"""
FastAPI AutoML Backend Service
Main application entry point
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.api.routes import upload, training, evaluation, reports
from app.core.config import settings
from app.core.exceptions import setup_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Create necessary directories
    directories = ["uploads", "models", "visualizations", "reports"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    # Cleanup if needed
    pass


# Initialize FastAPI app
app = FastAPI(
    title="AutoML Backend Service",
    description="A comprehensive AutoML backend service for automatic model training and evaluation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(reports.router, prefix="/api/v1", tags=["reports"])


@app.get("/")
async def root():
    """Root endpoint serving the frontend"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AutoML Backend"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
