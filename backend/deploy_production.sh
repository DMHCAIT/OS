#!/bin/bash

# Advanced AI/ML Sales Automation System - Production Deployment Script
# This script sets up the complete production environment

set -e  # Exit on any error

echo "🚀 Starting Production Deployment of Advanced AI/ML Sales Automation System"
echo "=================================================================="

# Configuration variables
PROJECT_NAME="ai-sales-automation"
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
ENVIRONMENT=${ENVIRONMENT:-production}
DOMAIN=${DOMAIN:-localhost}

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Create necessary directories
print_status "Creating production directories..."
mkdir -p logs
mkdir -p data/models
mkdir -p data/backups
mkdir -p config/production
mkdir -p certificates
mkdir -p monitoring

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if ! command -v python3.8 &> /dev/null && ! command -v python3.9 &> /dev/null && ! command -v python3.10 &> /dev/null; then
    print_error "Python 3.8+ is required"
    exit 1
fi

# Check Node.js version for frontend
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Frontend deployment will be skipped."
    SKIP_FRONTEND=true
else
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    print_success "Node.js version $NODE_VERSION found"
fi

# Check Docker (optional)
if command -v docker &> /dev/null; then
    print_success "Docker found - containerized deployment available"
    DOCKER_AVAILABLE=true
else
    print_warning "Docker not found - using local deployment"
    DOCKER_AVAILABLE=false
fi

# Install system dependencies
print_status "Installing system dependencies..."

# For macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is required on macOS. Please install it first."
        exit 1
    fi
    
    brew update
    brew install redis mongodb-community postgresql nginx
    
    # Start services
    brew services start redis
    brew services start mongodb-community
    brew services start postgresql
    
# For Ubuntu/Debian
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y redis-server mongodb postgresql postgresql-contrib nginx python3-dev build-essential
    
    # Start services
    sudo systemctl start redis-server
    sudo systemctl start mongodb
    sudo systemctl start postgresql
    sudo systemctl enable redis-server
    sudo systemctl enable mongodb
    sudo systemctl enable postgresql
fi

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."
python3 -m venv venv_production
source venv_production/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn uvicorn[standard] supervisor celery redis

# Setup environment variables
print_status "Configuring environment variables..."

cat > .env.production << EOF
# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=$BACKEND_PORT
WORKERS=4
RELOAD=false

# Database Configuration
MONGODB_URL=mongodb://localhost:27017/sales_ai_production
REDIS_URL=redis://localhost:6379/0
POSTGRESQL_URL=postgresql://localhost/sales_ai_production

# AI/ML API Keys (REQUIRED - Please update these)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Security Configuration
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/sales_ai.log
ERROR_LOG_FILE=logs/sales_ai_error.log

# Performance Configuration
MAX_WORKERS=8
WORKER_TIMEOUT=300
KEEPALIVE=120

# Rate Limiting
RATE_LIMIT_CALLS=1000
RATE_LIMIT_PERIOD=3600

# Model Configuration
MODEL_CACHE_SIZE=1000
MODEL_UPDATE_INTERVAL=86400
ENABLE_MODEL_CACHING=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
HEALTH_CHECK_INTERVAL=30

EOF

# Create production configuration
print_status "Creating production configuration files..."

# Gunicorn configuration
cat > gunicorn.conf.py << EOF
import multiprocessing

# Server socket
bind = "0.0.0.0:$BACKEND_PORT"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 300
keepalive = 120

# Restart workers
preload_app = True
reload = False

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%h %l %u %t "%r" %s %b "%{Referer}i" "%{User-Agent}i" %D'

# Process naming
proc_name = 'ai_sales_automation'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment and configure for HTTPS)
# keyfile = "certificates/private.key"
# certfile = "certificates/certificate.crt"
EOF

# Supervisor configuration
cat > config/production/supervisord.conf << EOF
[unix_http_server]
file=/tmp/supervisor.sock

[supervisord]
logfile=logs/supervisord.log
logfile_maxbytes=50MB
logfile_backups=10
loglevel=info
pidfile=logs/supervisord.pid
nodaemon=false
minfds=1024
minprocs=200

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock

[program:ai_sales_backend]
command=$(pwd)/venv_production/bin/gunicorn app.main:app -c gunicorn.conf.py
directory=$(pwd)
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=logs/backend_supervisor.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="$(pwd)/venv_production/bin"

[program:celery_worker]
command=$(pwd)/venv_production/bin/celery -A app.core.celery_app worker --loglevel=info
directory=$(pwd)
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=logs/celery_worker.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="$(pwd)/venv_production/bin"

[program:celery_beat]
command=$(pwd)/venv_production/bin/celery -A app.core.celery_app beat --loglevel=info
directory=$(pwd)
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=logs/celery_beat.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="$(pwd)/venv_production/bin"
EOF

# Nginx configuration
print_status "Configuring Nginx reverse proxy..."

cat > config/production/nginx.conf << EOF
upstream ai_sales_backend {
    server 127.0.0.1:$BACKEND_PORT;
    keepalive 32;
}

server {
    listen 80;
    server_name $DOMAIN;
    
    # Redirect HTTP to HTTPS (uncomment for SSL)
    # return 301 https://\$server_name\$request_uri;
    
    client_max_body_size 50M;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # API endpoints
    location /api/ {
        proxy_pass http://ai_sales_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # WebSocket endpoint
    location /ws/ {
        proxy_pass http://ai_sales_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Health check
    location /health {
        proxy_pass http://ai_sales_backend/health;
        access_log off;
    }
    
    # API Documentation
    location /docs {
        proxy_pass http://ai_sales_backend/docs;
    }
    
    location /redoc {
        proxy_pass http://ai_sales_backend/redoc;
    }
    
    # Frontend (if available)
    location / {
        root /var/www/ai-sales-frontend;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }
    
    # Static files
    location /static/ {
        alias $(pwd)/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Logging
    access_log logs/nginx_access.log;
    error_log logs/nginx_error.log;
}

# HTTPS configuration (uncomment and configure for SSL)
# server {
#     listen 443 ssl http2;
#     server_name $DOMAIN;
#     
#     ssl_certificate certificates/certificate.crt;
#     ssl_certificate_key certificates/private.key;
#     ssl_session_timeout 1d;
#     ssl_session_cache shared:MozTLS:10m;
#     ssl_session_tickets off;
#     
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
#     ssl_prefer_server_ciphers off;
#     
#     # HSTS
#     add_header Strict-Transport-Security "max-age=63072000" always;
#     
#     # Include the same location blocks as the HTTP server above
# }
EOF

# Create systemd service (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/ai-sales-automation.service > /dev/null << EOF
[Unit]
Description=AI Sales Automation System
After=network.target

[Service]
Type=forking
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv_production/bin
ExecStart=$(pwd)/venv_production/bin/supervisord -c config/production/supervisord.conf
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ai-sales-automation
fi

# Create monitoring script
cat > monitor_system.py << 'EOF'
#!/usr/bin/env python3
"""
System monitoring script for AI Sales Automation
Monitors health, performance, and resource usage
"""

import psutil
import requests
import time
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

class SystemMonitor:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.check_interval = 30  # seconds
    
    def check_backend_health(self):
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Backend health check failed: {e}")
            return False
    
    def get_system_metrics(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None,
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
        }
    
    def monitor(self):
        logging.info("Starting system monitoring...")
        
        while True:
            try:
                # Check backend health
                backend_healthy = self.check_backend_health()
                
                # Get system metrics
                metrics = self.get_system_metrics()
                
                # Log status
                status = "HEALTHY" if backend_healthy else "UNHEALTHY"
                logging.info(
                    f"System Status: {status} | "
                    f"CPU: {metrics['cpu_usage']:.1f}% | "
                    f"Memory: {metrics['memory_usage']:.1f}% | "
                    f"Disk: {metrics['disk_usage']:.1f}%"
                )
                
                # Alert on high resource usage
                if metrics['cpu_usage'] > 80:
                    logging.warning(f"High CPU usage: {metrics['cpu_usage']:.1f}%")
                
                if metrics['memory_usage'] > 80:
                    logging.warning(f"High memory usage: {metrics['memory_usage']:.1f}%")
                
                if not backend_healthy:
                    logging.error("Backend is unhealthy!")
                
                # Save metrics to file
                with open('logs/metrics.json', 'a') as f:
                    metrics['backend_healthy'] = backend_healthy
                    json.dump(metrics, f)
                    f.write('\n')
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logging.info("Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor()
EOF

chmod +x monitor_system.py

# Create deployment scripts
cat > start_production.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Sales Automation System (Production Mode)"

# Load environment
source venv_production/bin/activate
export $(cat .env.production | xargs)

# Start supervisor (which manages all services)
supervisord -c config/production/supervisord.conf

echo "✅ System started successfully!"
echo "🌐 API Documentation: http://localhost:8000/docs"
echo "📊 Health Check: http://localhost:8000/health"
echo "📈 Monitoring: Run './monitor_system.py' in another terminal"
EOF

cat > stop_production.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping AI Sales Automation System"

# Stop supervisor
supervisorctl -c config/production/supervisord.conf shutdown

echo "✅ System stopped successfully!"
EOF

cat > restart_production.sh << 'EOF'
#!/bin/bash
echo "🔄 Restarting AI Sales Automation System"

# Stop and start
./stop_production.sh
sleep 5
./start_production.sh
EOF

# Make scripts executable
chmod +x start_production.sh stop_production.sh restart_production.sh

# Database initialization
print_status "Initializing databases..."

# Create PostgreSQL database
if command -v createdb &> /dev/null; then
    createdb sales_ai_production 2>/dev/null || true
fi

# MongoDB initialization (create collections with proper indexes)
python3 << 'EOF'
import pymongo
try:
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client.sales_ai_production
    
    # Create collections with indexes
    db.conversations.create_index([("conversation_id", 1), ("timestamp", -1)])
    db.leads.create_index([("email", 1), ("company", 1)])
    db.analytics.create_index([("date", -1), ("metric_type", 1)])
    db.voice_sessions.create_index([("session_id", 1), ("timestamp", -1)])
    
    print("✅ MongoDB collections and indexes created successfully")
except Exception as e:
    print(f"⚠️  MongoDB setup warning: {e}")
EOF

# Test the installation
print_status "Testing installation..."

# Activate virtual environment and test imports
source venv_production/bin/activate

python3 -c "
import sys
try:
    from app.main import app
    print('✅ Backend imports successful')
except Exception as e:
    print(f'❌ Backend import error: {e}')
    sys.exit(1)
"

# Final setup
print_success "Production deployment completed successfully!"

echo ""
echo "🎉 AI Sales Automation System - Production Deployment Complete!"
echo "=============================================================="
echo ""
echo "📋 Next Steps:"
echo "   1. Update API keys in .env.production file"
echo "   2. Configure SSL certificates (optional)"
echo "   3. Start the system: ./start_production.sh"
echo "   4. Monitor system: ./monitor_system.py"
echo ""
echo "📊 System Information:"
echo "   • Backend URL: http://$DOMAIN:$BACKEND_PORT"
echo "   • API Docs: http://$DOMAIN:$BACKEND_PORT/docs"
echo "   • Health Check: http://$DOMAIN:$BACKEND_PORT/health"
echo "   • Environment: $ENVIRONMENT"
echo ""
echo "🔧 Management Commands:"
echo "   • Start: ./start_production.sh"
echo "   • Stop: ./stop_production.sh"
echo "   • Restart: ./restart_production.sh"
echo "   • Monitor: ./monitor_system.py"
echo "   • Test: python test_ai_system.py"
echo ""
echo "📁 Important Files:"
echo "   • Configuration: .env.production"
echo "   • Logs: logs/ directory"
echo "   • Monitoring: logs/monitoring.log"
echo "   • Backup: Use 'backup_system.sh' script"
echo ""

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux Service Management:"
    echo "   • Enable: sudo systemctl enable ai-sales-automation"
    echo "   • Start: sudo systemctl start ai-sales-automation"
    echo "   • Status: sudo systemctl status ai-sales-automation"
    echo ""
fi

print_warning "Important: Please update the API keys in .env.production before starting the system!"

echo "🚀 Ready for production deployment!"