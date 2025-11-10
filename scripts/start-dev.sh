#!/bin/bash

# AI Lead Management System - Development Server Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found. Please run setup.sh first."
    exit 1
fi

# Function to start backend
start_backend() {
    print_status "Starting Python backend..."
    cd backend
    
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run setup.sh first."
        exit 1
    fi
    
    source venv/bin/activate
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    print_success "Backend started on http://localhost:8000 (PID: $BACKEND_PID)"
    cd ..
}

# Function to start frontend
start_frontend() {
    print_status "Starting React frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        print_error "Node modules not found. Please run setup.sh first."
        exit 1
    fi
    
    npm start &
    FRONTEND_PID=$!
    
    print_success "Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"
    cd ..
}

# Function to start databases
start_databases() {
    print_status "Starting databases..."
    
    if command -v docker &> /dev/null; then
        docker-compose up -d mongodb redis
        print_success "Databases started with Docker"
    else
        print_error "Docker not found. Please start MongoDB and Redis manually."
    fi
}

# Function to check health
check_health() {
    print_status "Checking application health..."
    
    # Wait a bit for services to start
    sleep 5
    
    # Check backend health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "Backend is healthy"
    else
        print_error "Backend health check failed"
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 &> /dev/null; then
        print_success "Frontend is accessible"
    else
        print_error "Frontend is not accessible yet (may still be starting)"
    fi
}

# Cleanup function
cleanup() {
    print_status "Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    print_success "Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo "======================================"
    echo "  Starting AI Lead Management System"
    echo "======================================"
    echo ""
    
    start_databases
    echo ""
    
    start_backend
    echo ""
    
    start_frontend
    echo ""
    
    check_health
    echo ""
    
    print_success "🚀 All services are running!"
    echo ""
    echo "Access points:"
    echo "- Frontend: http://localhost:3000"
    echo "- Backend API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/docs"
    echo "- MongoDB: mongodb://localhost:27017"
    echo "- Redis: redis://localhost:6379"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Keep script running
    while true; do
        sleep 1
    done
}

# Run main function
main