#!/bin/bash

# Quick Production Setup Script
# Sets up the AI Sales Automation System for production use with real data

set -e

echo "🚀 AI Sales Automation - Quick Production Setup"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
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

# Step 1: Check Prerequisites
print_step "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found, creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    print_success "Virtual environment found"
    source venv/bin/activate
fi

# Step 2: Install Dependencies
print_step "Installing/updating dependencies..."
pip install -r requirements.txt

# Step 3: Setup Data Directory
print_step "Setting up data directories..."
mkdir -p data/training
mkdir -p data/production
mkdir -p models/production
mkdir -p logs

# Step 4: Check for Real Data
print_step "Checking for training data..."

if [ ! -f "data/training/historical_leads.csv" ]; then
    print_warning "No training data found"
    echo ""
    echo "📋 IMPORTANT: To use this system in production, you need:"
    echo ""
    echo "1. 📊 HISTORICAL LEAD DATA (1000+ leads minimum)"
    echo "   - File: data/training/historical_leads.csv"
    echo "   - Columns: lead_id, company_size, industry, email_engagement_rate, converted, etc."
    echo ""
    echo "2. 💬 CONVERSATION DATA (500+ conversations)"
    echo "   - File: data/training/sales_conversations.csv"
    echo "   - Columns: conversation_id, conversation_text, outcome, etc."
    echo ""
    echo "3. 💰 DEAL DATA (200+ closed deals)"
    echo "   - File: data/training/historical_deals.csv"
    echo "   - Columns: deal_id, deal_value, close_date, etc."
    echo ""
    
    # Create data validation templates
    print_step "Creating data templates..."
    python validate_data.py
    
    echo ""
    print_warning "NEXT STEPS:"
    echo "1. Fill data templates in data/training/ with your real data"
    echo "2. Run: python validate_data.py (to check data quality)"
    echo "3. Run: python train_production_models.py (to train with your data)"
    echo "4. Run: ./start_production.sh (to deploy)"
    echo ""
    
else
    # Validate existing data
    print_step "Validating existing data..."
    python validate_data.py
    
    echo ""
    read -p "🤖 Do you want to train models with your data now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Training models with your data..."
        python train_production_models.py
        
        if [ $? -eq 0 ]; then
            print_success "Models trained successfully!"
            echo ""
            read -p "🚀 Deploy to production now? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ./deploy_production.sh
            fi
        else
            print_error "Model training failed. Check logs for details."
        fi
    fi
fi

# Step 5: Configuration Check
print_step "Checking configuration..."

if [ ! -f ".env" ]; then
    print_warning "Environment file not found, creating template..."
    cat > .env << EOF
# AI Sales Automation - Production Configuration

# AI/ML API Keys (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Database Configuration
MONGODB_URL=mongodb://localhost:27017/sales_ai_production
POSTGRESQL_URL=postgresql://localhost/sales_ai_production
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
DEBUG=false
ENVIRONMENT=production

# Email/Notifications (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# CRM Integration (Optional)
SALESFORCE_API_KEY=your_salesforce_key
HUBSPOT_API_KEY=your_hubspot_key
EOF

    echo ""
    print_warning "IMPORTANT: Update .env file with your actual API keys!"
    echo "Required:"
    echo "  • OPENAI_API_KEY (for GPT-4 conversation analysis)"
    echo "  • Database URLs (if using external databases)"
    echo ""
fi

# Step 6: Test System
print_step "Running system tests..."

if command -v python3 &> /dev/null; then
    python3 -c "
try:
    from textblob import TextBlob
    print('✅ TextBlob working')
    from app.core.ai_conversation_engine import AIConversationEngine
    print('✅ AI services importable')
    print('✅ System dependencies verified')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "System tests passed!"
    else
        print_error "System tests failed"
        exit 1
    fi
fi

# Step 7: Production Summary
echo ""
echo "🎯 PRODUCTION SETUP SUMMARY"
echo "=========================="
echo ""

if [ -f "models/production/lead_scoring_models.pkl" ]; then
    print_success "Lead scoring models trained with your data"
else
    print_warning "Using pre-trained models (train with your data for better accuracy)"
fi

if [ -f ".env" ]; then
    print_success "Environment configuration ready"
else
    print_warning "Environment configuration needs setup"
fi

echo ""
echo "📊 DATA STATUS:"
if [ -f "data/training/historical_leads.csv" ]; then
    lead_count=$(tail -n +2 data/training/historical_leads.csv | wc -l)
    echo "   • Leads: $lead_count records"
else
    echo "   • Leads: No data (using sample data)"
fi

if [ -f "data/training/sales_conversations.csv" ]; then
    conv_count=$(tail -n +2 data/training/sales_conversations.csv | wc -l)
    echo "   • Conversations: $conv_count records"
else
    echo "   • Conversations: No data (using sample data)"
fi

echo ""
echo "🚀 NEXT STEPS FOR PRODUCTION:"
echo ""

if [ ! -f "data/training/historical_leads.csv" ] || [ ! -f ".env" ]; then
    echo "1. 📊 Provide your historical sales data"
    echo "2. 🔑 Update API keys in .env file"
    echo "3. 🤖 Train models: python train_production_models.py"
    echo "4. 🚀 Deploy: ./deploy_production.sh"
else
    echo "1. 🔑 Verify API keys in .env file"
    echo "2. 🚀 Deploy to production: ./deploy_production.sh"
    echo "3. 📊 Monitor performance: ./monitor_system.py"
fi

echo ""
echo "📞 SUPPORT:"
echo "   • Documentation: README_ADVANCED_AI.md"
echo "   • Health Check: http://localhost:8000/health"
echo "   • API Docs: http://localhost:8000/docs"
echo ""

print_success "Quick setup completed! Your AI Sales Automation System is ready."