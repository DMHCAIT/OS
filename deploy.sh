#!/bin/bash

# Advanced AI Sales Automation - Production Deployment Script
# This script sets up the complete production environment

set -e  # Exit on any error

echo "🚀 Starting Advanced AI Sales Automation Deployment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Check system requirements
print_status "Checking system requirements..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check available disk space (minimum 10GB)
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -lt 10 ]; then
    print_error "Insufficient disk space. At least 10GB required."
    exit 1
fi

print_status "System requirements check passed ✓"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads logs models mongodb/init nginx monitoring/grafana/{dashboards,datasources}

# Set up environment variables
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    if [ -f backend/.env.template ]; then
        cp backend/.env.template .env
        print_warning "Please edit .env file with your actual configuration values"
    else
        print_error ".env.template not found. Please create .env file manually."
        exit 1
    fi
fi

# Generate secure secrets if not provided
if ! grep -q "SECRET_KEY.*=" .env || grep -q "your-super-secret" .env; then
    print_status "Generating secure secret key..."
    secret_key=$(openssl rand -hex 32)
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=$secret_key/" .env
fi

# Create nginx configuration
print_status "Setting up Nginx configuration..."
cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name localhost;
        client_max_body_size 100M;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN";
        add_header X-XSS-Protection "1; mode=block";
        add_header X-Content-Type-Options "nosniff";

        # API endpoints
        location /api/ {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Health check
        location /health {
            proxy_pass http://api/health;
        }

        # Static files
        location /static/ {
            alias /app/static/;
        }
    }
}
EOF

# Create Prometheus configuration
print_status "Setting up monitoring configuration..."
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-sales-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Create MongoDB initialization script
print_status "Setting up MongoDB initialization..."
cat > mongodb/init/01-init.js << 'EOF'
// Initialize AI Sales Automation Database
db = db.getSiblingDB('ai_sales_automation');

// Create collections with proper indexes
db.createCollection('leads');
db.createCollection('calls');
db.createCollection('users');
db.createCollection('intelligence_analyses');
db.createCollection('territories');
db.createCollection('sales_history');

// Create indexes for performance
db.leads.createIndex({ "email": 1 }, { unique: true });
db.leads.createIndex({ "created_at": -1 });
db.leads.createIndex({ "score": -1 });

db.calls.createIndex({ "lead_id": 1 });
db.calls.createIndex({ "created_at": -1 });

db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });

db.intelligence_analyses.createIndex({ "user_id": 1, "analysis_type": 1 });
db.intelligence_analyses.createIndex({ "created_at": -1 });

db.territories.createIndex({ "user_id": 1 });
db.sales_history.createIndex({ "user_id": 1, "date": -1 });

print('AI Sales Automation database initialized successfully');
EOF

# Function to wait for services
wait_for_service() {
    local service=$1
    local port=$2
    local timeout=120
    local count=0
    
    print_status "Waiting for $service to be ready..."
    while [ $count -lt $timeout ]; do
        if curl -f http://localhost:$port/health &> /dev/null; then
            print_status "$service is ready! ✓"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    print_error "$service failed to start within $timeout seconds"
    return 1
}

# Build and start services
print_status "Building Docker images..."
docker-compose -f docker-compose.production.yml build

print_status "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
print_status "Waiting for services to initialize..."
sleep 10

# Check if API is ready
if wait_for_service "API" "8000"; then
    print_status "API service started successfully ✓"
else
    print_error "API service failed to start"
    print_status "Checking logs..."
    docker-compose -f docker-compose.production.yml logs api
    exit 1
fi

# Verify database connection
print_status "Verifying database connection..."
if docker-compose -f docker-compose.production.yml exec -T mongodb mongosh --eval "db.adminCommand('ismaster')" &> /dev/null; then
    print_status "Database connection verified ✓"
else
    print_error "Database connection failed"
    exit 1
fi

# Verify Redis connection
print_status "Verifying Redis connection..."
if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping | grep -q PONG; then
    print_status "Redis connection verified ✓"
else
    print_error "Redis connection failed"
    exit 1
fi

# Setup health monitoring
print_status "Setting up health monitoring..."
(
    while true; do
        if ! curl -f http://localhost:8000/health &> /dev/null; then
            echo "$(date): Health check failed" >> logs/health.log
        fi
        sleep 60
    done
) &

# Display final status
print_status "🎉 Deployment completed successfully!"
echo ""
echo "================================================================"
echo "🚀 Advanced AI Sales Automation is now running!"
echo "================================================================"
echo ""
echo "📊 Services:"
echo "  • API Server:      http://localhost:8000"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Grafana Dashboard: http://localhost:3001 (admin/admin123)"
echo "  • Prometheus:      http://localhost:9090"
echo ""
echo "💾 Data:"
echo "  • MongoDB:         localhost:27017"
echo "  • Redis Cache:     localhost:6379"
echo ""
echo "📁 Important Directories:"
echo "  • Logs:           ./logs/"
echo "  • Uploads:        ./uploads/"
echo "  • ML Models:      ./models/"
echo ""
echo "🛠️  Management Commands:"
echo "  • View logs:      docker-compose -f docker-compose.production.yml logs"
echo "  • Stop services:  docker-compose -f docker-compose.production.yml down"
echo "  • Restart:        docker-compose -f docker-compose.production.yml restart"
echo "  • Update:         ./deploy.sh"
echo ""
echo "📋 Next Steps:"
echo "  1. Edit .env file with your API keys and configuration"
echo "  2. Access the API documentation at http://localhost:8000/docs"
echo "  3. Set up monitoring dashboards in Grafana"
echo "  4. Configure your external services (OpenAI, etc.)"
echo ""
echo "⚠️  Security Reminders:"
echo "  • Change default passwords in .env"
echo "  • Set up SSL certificates for production"
echo "  • Configure firewall rules"
echo "  • Set up backup procedures"
echo ""
print_status "🎯 Your AI-powered sales automation system is ready!"