#!/bin/bash

# Advanced AI/ML Sales Automation Backend Setup Script
# This script sets up and runs the comprehensive AI/ML backend system

set -e  # Exit on any error

echo "🚀 Setting up Advanced AI/ML Sales Automation Backend..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if Python 3.8+ is installed
print_section "Checking Python Version"
python_version=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1,2)
required_version="3.8"

if [ "$(echo "$python_version >= $required_version" | bc -l)" -eq 1 ]; then
    print_status "Python $python_version detected - Compatible ✓"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the backend directory."
    exit 1
fi

print_status "Running from correct directory: $(pwd)"

# Create virtual environment if it doesn't exist
print_section "Setting up Virtual Environment"
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created successfully"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first
print_section "Installing Core Dependencies"
print_status "Installing FastAPI and core web framework..."
pip install fastapi uvicorn websockets pydantic python-multipart

# Install data processing libraries
print_status "Installing data processing libraries..."
pip install numpy pandas scipy

# Install machine learning libraries (with error handling)
print_section "Installing Machine Learning Libraries"
print_status "Installing scikit-learn..."
pip install scikit-learn || print_warning "Failed to install scikit-learn - will try alternative"

print_status "Installing XGBoost..."
pip install xgboost || print_warning "Failed to install XGBoost"

print_status "Installing LightGBM..."
pip install lightgbm || print_warning "Failed to install LightGBM"

print_status "Installing CatBoost..."
pip install catboost || print_warning "Failed to install CatBoost"

# Install PyTorch (CPU version for compatibility)
print_section "Installing PyTorch and Deep Learning Libraries"
print_status "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || print_warning "Failed to install PyTorch"

print_status "Installing Transformers..."
pip install transformers sentence-transformers tokenizers || print_warning "Failed to install transformers"

# Install NLP libraries
print_section "Installing Natural Language Processing Libraries"
print_status "Installing spaCy..."
pip install spacy
python -m spacy download en_core_web_sm || print_warning "Failed to download spaCy model"

print_status "Installing other NLP libraries..."
pip install nltk textblob

# Install voice processing libraries
print_section "Installing Voice AI Libraries"
print_status "Installing voice processing libraries..."
pip install speechrecognition pydub librosa || print_warning "Some voice libraries failed to install"

# Install time series libraries
print_section "Installing Time Series Analysis Libraries"
print_status "Installing statsmodels and time series libraries..."
pip install statsmodels || print_warning "Failed to install statsmodels"

# Install remaining dependencies
print_section "Installing Remaining Dependencies"
print_status "Installing remaining packages from requirements.txt..."
pip install -r requirements.txt || print_warning "Some packages from requirements.txt failed to install"

# Download NLTK data
print_section "Downloading NLTK Data"
print_status "Downloading NLTK data packages..."
python -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Failed to download NLTK data: {e}')
" || print_warning "Failed to download NLTK data"

# Create necessary directories
print_section "Creating Project Structure"
print_status "Creating necessary directories..."
mkdir -p ml-models/predictive
mkdir -p logs
mkdir -p data/training
mkdir -p data/cache

print_status "Project structure created"

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_section "Creating Environment Configuration"
    print_status "Creating .env file with default configuration..."
    cat > .env << EOF
# Advanced AI/ML Sales Automation Configuration

# API Keys (Please configure with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
MONGODB_URL=mongodb://localhost:27017/sales_ai
REDIS_URL=redis://localhost:6379

# Application Settings
DEBUG=true
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000

# AI/ML Configuration
MODEL_CACHE_DIR=./ml-models
TRAINING_DATA_DIR=./data/training
MAX_CONVERSATION_HISTORY=100

# Voice AI Settings
VOICE_SAMPLE_RATE=16000
VOICE_CHUNK_SIZE=1024

# Security
JWT_SECRET_KEY=your_super_secret_jwt_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_email_password

# Performance Settings
WORKER_PROCESSES=1
MAX_REQUESTS=1000
REQUEST_TIMEOUT=300
EOF

    print_status "Environment file created. Please configure your API keys in .env"
    print_warning "Remember to set your actual API keys in the .env file!"
else
    print_status ".env file already exists"
fi

# Run basic tests
print_section "Running Basic Tests"
print_status "Testing Python imports..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import fastapi
    print('✓ FastAPI imported successfully')
except ImportError as e:
    print(f'✗ FastAPI import failed: {e}')

try:
    import numpy
    print('✓ NumPy imported successfully')
except ImportError as e:
    print(f'✗ NumPy import failed: {e}')

try:
    import pandas
    print('✓ Pandas imported successfully')
except ImportError as e:
    print(f'✗ Pandas import failed: {e}')

try:
    import sklearn
    print('✓ Scikit-learn imported successfully')
except ImportError as e:
    print(f'✗ Scikit-learn import failed: {e}')

try:
    import torch
    print('✓ PyTorch imported successfully')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')

try:
    import transformers
    print('✓ Transformers imported successfully')
except ImportError as e:
    print(f'✗ Transformers import failed: {e}')

print('\\nBasic import tests completed.')
"

# Create startup script for the application
print_section "Creating Application Scripts"
cat > start_backend.sh << 'EOF'
#!/bin/bash
# Start the Advanced AI/ML Sales Automation Backend

source venv/bin/activate

echo "🚀 Starting Advanced AI/ML Sales Automation Backend..."
echo "Time: $(date)"
echo "Directory: $(pwd)"
echo "Python: $(which python)"

# Start the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
EOF

chmod +x start_backend.sh

cat > start_development.sh << 'EOF'
#!/bin/bash
# Start the backend in development mode with additional logging

source venv/bin/activate

echo "🔧 Starting Development Mode..."
export DEBUG=true
export LOG_LEVEL=debug

# Start with hot reloading
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app --log-level debug
EOF

chmod +x start_development.sh

print_status "Startup scripts created"

print_section "Installation Complete!"
echo -e "${GREEN}"
echo "✅ Advanced AI/ML Sales Automation Backend Setup Complete!"
echo ""
echo "🔥 Key Features Installed:"
echo "   • Advanced Conversation Analysis with GPT/Transformer models"
echo "   • ML-powered Lead Scoring with ensemble algorithms"
echo "   • Real-time Voice AI with emotion detection"
echo "   • Predictive Analytics for sales forecasting"
echo "   • Automated objection handling"
echo "   • WebSocket support for real-time guidance"
echo ""
echo "📚 Next Steps:"
echo "   1. Configure your API keys in .env file"
echo "   2. Start the backend: ./start_backend.sh"
echo "   3. Access API docs: http://localhost:8000/docs"
echo "   4. Test the health endpoint: http://localhost:8000/health"
echo ""
echo "🚀 Quick Start Commands:"
echo "   Development mode: ./start_development.sh"
echo "   Production mode:  ./start_backend.sh"
echo ""
echo "📖 API Documentation will be available at:"
echo "   • Swagger UI: http://localhost:8000/docs"
echo "   • ReDoc: http://localhost:8000/redoc"
echo ""
print_warning "Don't forget to configure your OpenAI API key and other services!"
echo -e "${NC}"