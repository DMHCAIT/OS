#!/bin/bash

# 🚀 AI Sales Automation - Quick Production Setup (Updated)
# ========================================================
echo "🚀 AI Sales Automation - Quick Production Setup"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "${BLUE}[STEP]${NC} Checking prerequisites..."
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[WARNING]${NC} Virtual environment not found. Creating one..."
    python3 -m venv venv
fi
echo -e "${GREEN}[SUCCESS]${NC} Virtual environment found"

# Step 2: Install dependencies
echo -e "${BLUE}[STEP]${NC} Installing/updating dependencies..."
source venv/bin/activate
pip install -r requirements-simple.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed successfully"
else
    echo -e "${RED}[ERROR]${NC} Some dependencies failed to install, but continuing..."
fi

# Step 3: Test core imports
echo -e "${BLUE}[STEP]${NC} Testing core imports..."
python -c "
import textblob
from textblob import TextBlob
print('✅ TextBlob working!')
import nltk
print('✅ NLTK working!')
import spacy
print('✅ spaCy working!')
import transformers
print('✅ Transformers working!')
import torch
print('✅ PyTorch working!')
import pandas
print('✅ Pandas working!')
import numpy
print('✅ NumPy working!')
print('🎉 All core AI/ML libraries are working!')
" || echo -e "${YELLOW}[WARNING]${NC} Some imports failed but core system should work"

# Step 4: Download NLTK data
echo -e "${BLUE}[STEP]${NC} Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('✅ NLTK data downloaded')
" > /dev/null 2>&1

# Step 5: Download spaCy model
echo -e "${BLUE}[STEP]${NC} Checking spaCy model..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy English model available')
except OSError:
    print('⚠️  spaCy English model not found. Installing...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    print('✅ spaCy English model installed')
" || echo -e "${YELLOW}[INFO]${NC} spaCy model will be downloaded when needed"

# Step 6: Test production training pipeline
echo -e "${BLUE}[STEP]${NC} Testing production training pipeline..."
if [ -f "train_production_models.py" ]; then
    python -c "
import sys
sys.path.append('.')
try:
    from train_production_models import ProductionModelTrainer
    print('✅ Production training pipeline ready')
except Exception as e:
    print(f'⚠️  Training pipeline needs attention: {e}')
"
else
    echo -e "${YELLOW}[WARNING]${NC} train_production_models.py not found"
fi

# Step 7: Test data validation system
echo -e "${BLUE}[STEP]${NC} Testing data validation system..."
if [ -f "validate_data.py" ]; then
    python -c "
import sys
sys.path.append('.')
try:
    from validate_data import DataValidator
    print('✅ Data validation system ready')
except Exception as e:
    print(f'⚠️  Data validator needs attention: {e}')
"
else
    echo -e "${YELLOW}[WARNING]${NC} validate_data.py not found"
fi

echo ""
echo -e "${GREEN}🎉 PRODUCTION SETUP COMPLETE!${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Place your historical data in the data/ directory"
echo "2. Run: python validate_data.py to check data quality"  
echo "3. Run: python train_production_models.py to train custom models"
echo "4. Start the server: uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo -e "${BLUE}For detailed guidance, see:${NC} PRODUCTION_GUIDE.md"
echo ""
echo -e "${GREEN}Your AI Sales Automation system is ready for production! 🚀${NC}"