#!/bin/bash

# Railway Deployment Script for AI Lead Management System
# Run this script to deploy your entire system to Railway

echo "🚀 Starting Railway deployment for AI Lead Management System..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway..."
    railway login
fi

echo "🏗️  Setting up Railway project..."

# Initialize Railway project
railway init

echo "🌱 Creating environment variables template..."

# Create environment variables file for Railway
cat > railway-env.txt << EOF
# Copy these environment variables to your Railway dashboard

## Backend Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here  
AZURE_REGION=eastus
JWT_SECRET=$(openssl rand -base64 32)
ENVIRONMENT=production
PYTHONPATH=/app

## Frontend Environment Variables  
REACT_APP_ENVIRONMENT=production
# REACT_APP_API_URL will be auto-configured by Railway

## Database URLs (Auto-configured by Railway)
# DATABASE_URL=mongodb://...
# REDIS_URL=redis://...
EOF

echo "📋 Environment variables template created: railway-env.txt"

echo "🗄️  Setting up databases..."

# Add MongoDB database
echo "Adding MongoDB database..."
railway add --database mongodb

# Add Redis database  
echo "Adding Redis cache..."
railway add --database redis

echo "🚀 Deploying backend service..."

# Deploy backend
railway up --service backend

echo "🎨 Deploying frontend service..."

# Deploy frontend
railway up --service frontend

echo "✅ Deployment initiated!"

echo "
🎉 Railway deployment setup complete!

📋 Next Steps:
1. Go to your Railway dashboard: https://railway.app/dashboard
2. Add the environment variables from 'railway-env.txt'
3. Get your API keys from the API_KEYS_GUIDE.md
4. Your app will be live at: https://your-project.railway.app

📊 Monitor deployment:
- Backend: railway logs --service backend
- Frontend: railway logs --service frontend  
- Status: railway status

🔧 Useful commands:
- railway open: Open your app in browser
- railway logs: View all logs
- railway env: Manage environment variables
- railway connect: Connect to databases

💰 Expected cost: $15-25/month for your AI system
"

echo "🔗 Opening Railway dashboard..."
railway open --dashboard