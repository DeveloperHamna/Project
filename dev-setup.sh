#!/bin/bash

# Development Setup Script for Credit Scoring Model
# This script sets up the development environment for GitHub Codespaces

echo "🚀 Setting up Credit Scoring Model development environment..."

# Check if we're in a codespace
if [ ! -z "$CODESPACE_NAME" ]; then
    echo "✅ Running in GitHub Codespaces: $CODESPACE_NAME"
else
    echo "🏠 Running in local development environment"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p uploads models visualizations reports
touch uploads/.gitkeep models/.gitkeep visualizations/.gitkeep reports/.gitkeep

# Install frontend dependencies (optional)
if [ -d "frontend" ]; then
    echo "🎨 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Check if backend is running
echo "🔍 Checking if backend server is already running..."
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ Backend server is already running on port 5000"
else
    echo "🚀 Starting backend server..."
    python main.py &
    BACKEND_PID=$!
    echo "Backend server started with PID: $BACKEND_PID"
    
    # Wait for server to start
    echo "⏳ Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:5000/health > /dev/null; then
            echo "✅ Backend server is ready!"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📋 Available endpoints:"
echo "   🌐 Main Application: http://localhost:5000"
echo "   📚 API Documentation: http://localhost:5000/docs"
echo "   🏥 Health Check: http://localhost:5000/health"
echo ""
echo "🔧 Development commands:"
echo "   python main.py                    - Start backend server"
echo "   uvicorn main:app --reload         - Start with hot reload"
echo "   curl http://localhost:5000/health - Test API health"
echo ""
echo "Happy coding! 💻"