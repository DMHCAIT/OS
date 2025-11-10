#!/bin/bash

# AI Lead Management System - Setup Script
# This script sets up the complete development environment

set -e

echo "🚀 Setting up AI Lead Management & Voice Communication System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please adjust for your operating system."
    exit 1
fi

# Check for required tools
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is required but not installed."
        echo "Install Homebrew from: https://brew.sh/"
        exit 1
    fi
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        print_status "Installing Node.js..."
        brew install node
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        print_status "Installing Python..."
        brew install python@3.11
    fi
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Please install Docker Desktop from: https://docker.com/products/docker-desktop"
        print_warning "You can continue without Docker, but you'll need to set up databases manually."
    fi
    
    print_success "System requirements check completed"
}

# Create environment file
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f .env ]; then
        cp .env.example .env
        print_success "Created .env file from template"
        print_warning "Please update the .env file with your actual API keys and configuration"
    else
        print_warning ".env file already exists, skipping creation"
    fi
}

# Setup Python backend
setup_backend() {
    print_status "Setting up Python backend..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Download spaCy model for NLP
    print_status "Downloading spaCy English language model..."
    python -m spacy download en_core_web_sm
    
    # Download NLTK data
    print_status "Downloading NLTK data..."
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
    
    print_success "Backend dependencies and NLP models installed"
    
    cd ..
}

# Setup React frontend
setup_frontend() {
    print_status "Setting up React frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend dependencies installed"
    
    cd ..
}

# Setup databases with Docker
setup_databases() {
    print_status "Setting up databases..."
    
    if command -v docker &> /dev/null; then
        # Start MongoDB and Redis with Docker
        print_status "Starting MongoDB and Redis with Docker..."
        docker-compose up -d mongodb redis
        
        # Wait for databases to start
        print_status "Waiting for databases to start..."
        sleep 10
        
        print_success "Databases are running"
    else
        print_warning "Docker not found. Please set up MongoDB and Redis manually:"
        echo "  - MongoDB: brew install mongodb-community"
        echo "  - Redis: brew install redis"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p logs
    mkdir -p database/backups
    mkdir -p voice-recordings
    
    print_success "Directories created"
}

# Setup development tools
setup_dev_tools() {
    print_status "Setting up development tools..."
    
    # Install VS Code extensions (if VS Code is installed)
    if command -v code &> /dev/null; then
        print_status "Installing VS Code extensions..."
        code --install-extension ms-python.python
        code --install-extension bradlc.vscode-tailwindcss
        code --install-extension esbenp.prettier-vscode
        code --install-extension ms-vscode.vscode-typescript-next
        print_success "VS Code extensions installed"
    fi
    
    # Git hooks setup
    if [ -d ".git" ]; then
        print_status "Setting up Git hooks..."
        cp scripts/pre-commit .git/hooks/
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks installed"
    fi
}

# Main setup function
main() {
    echo "======================================"
    echo "  AI Lead Management System Setup"
    echo "======================================"
    echo ""
    
    check_requirements
    echo ""
    
    setup_environment
    echo ""
    
    create_directories
    echo ""
    
    setup_backend
    echo ""
    
    setup_frontend
    echo ""
    
    setup_databases
    echo ""
    
    setup_dev_tools
    echo ""
    
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Update your .env file with actual API keys"
    echo "2. Start the development servers:"
    echo "   - Backend: cd backend && source venv/bin/activate && uvicorn main:app --reload"
    echo "   - Frontend: cd frontend && npm start"
    echo "3. Or use Docker: docker-compose up"
    echo ""
    echo "Access the application:"
    echo "- Frontend: http://localhost:3000"
    echo "- Backend API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/docs"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main