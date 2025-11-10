#!/bin/bash

# AI Sales Automation System - Backup and Maintenance Script
# Handles database backups, log rotation, and system maintenance

set -e

# Configuration
BACKUP_DIR="data/backups"
LOG_DIR="logs"
MAX_LOG_SIZE="100M"
BACKUP_RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Create backup directories
mkdir -p "$BACKUP_DIR/mongodb"
mkdir -p "$BACKUP_DIR/postgresql"
mkdir -p "$BACKUP_DIR/logs"
mkdir -p "$BACKUP_DIR/models"

backup_mongodb() {
    print_status "Backing up MongoDB..."
    
    if command -v mongodump &> /dev/null; then
        mongodump --host localhost --port 27017 --db sales_ai_production --out "$BACKUP_DIR/mongodb/backup_$TIMESTAMP"
        
        # Compress backup
        tar -czf "$BACKUP_DIR/mongodb/mongodb_backup_$TIMESTAMP.tar.gz" -C "$BACKUP_DIR/mongodb" "backup_$TIMESTAMP"
        rm -rf "$BACKUP_DIR/mongodb/backup_$TIMESTAMP"
        
        print_success "MongoDB backup completed: mongodb_backup_$TIMESTAMP.tar.gz"
    else
        print_warning "mongodump not found, skipping MongoDB backup"
    fi
}

backup_postgresql() {
    print_status "Backing up PostgreSQL..."
    
    if command -v pg_dump &> /dev/null; then
        pg_dump sales_ai_production > "$BACKUP_DIR/postgresql/postgresql_backup_$TIMESTAMP.sql"
        
        # Compress backup
        gzip "$BACKUP_DIR/postgresql/postgresql_backup_$TIMESTAMP.sql"
        
        print_success "PostgreSQL backup completed: postgresql_backup_$TIMESTAMP.sql.gz"
    else
        print_warning "pg_dump not found, skipping PostgreSQL backup"
    fi
}

backup_logs() {
    print_status "Backing up logs..."
    
    if [ -d "$LOG_DIR" ]; then
        tar -czf "$BACKUP_DIR/logs/logs_backup_$TIMESTAMP.tar.gz" -C "$LOG_DIR" .
        print_success "Logs backup completed: logs_backup_$TIMESTAMP.tar.gz"
    else
        print_warning "Log directory not found"
    fi
}

backup_models() {
    print_status "Backing up ML models..."
    
    if [ -d "data/models" ]; then
        tar -czf "$BACKUP_DIR/models/models_backup_$TIMESTAMP.tar.gz" -C "data/models" .
        print_success "Models backup completed: models_backup_$TIMESTAMP.tar.gz"
    else
        print_warning "Models directory not found"
    fi
}

rotate_logs() {
    print_status "Rotating logs..."
    
    find "$LOG_DIR" -name "*.log" -size +$MAX_LOG_SIZE -exec gzip {} \;
    
    # Keep only last 10 compressed log files
    find "$LOG_DIR" -name "*.log.gz" -type f -mtime +10 -delete
    
    print_success "Log rotation completed"
}

cleanup_old_backups() {
    print_status "Cleaning up old backups (older than $BACKUP_RETENTION_DAYS days)..."
    
    find "$BACKUP_DIR" -type f -mtime +$BACKUP_RETENTION_DAYS -delete
    
    print_success "Old backups cleaned up"
}

update_dependencies() {
    print_status "Checking for dependency updates..."
    
    if [ -f "venv_production/bin/activate" ]; then
        source venv_production/bin/activate
        
        # Create a backup of current requirements
        pip freeze > "$BACKUP_DIR/requirements_backup_$TIMESTAMP.txt"
        
        # Check for updates (don't auto-update in production)
        pip list --outdated --format=json > "$BACKUP_DIR/outdated_packages_$TIMESTAMP.json"
        
        outdated_count=$(cat "$BACKUP_DIR/outdated_packages_$TIMESTAMP.json" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data))")
        
        if [ "$outdated_count" -gt 0 ]; then
            print_warning "$outdated_count packages have updates available. Review outdated_packages_$TIMESTAMP.json"
        else
            print_success "All packages are up to date"
        fi
        
        deactivate
    else
        print_warning "Production virtual environment not found"
    fi
}

check_disk_space() {
    print_status "Checking disk space..."
    
    # Get disk usage percentage
    disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        print_error "Disk usage is critical: ${disk_usage}%"
        return 1
    elif [ "$disk_usage" -gt 80 ]; then
        print_warning "Disk usage is high: ${disk_usage}%"
    else
        print_success "Disk usage is normal: ${disk_usage}%"
    fi
}

check_memory_usage() {
    print_status "Checking memory usage..."
    
    if command -v python3 &> /dev/null; then
        python3 << 'EOF'
import psutil
memory = psutil.virtual_memory()
usage_percent = memory.percent

if usage_percent > 90:
    print(f"❌ Memory usage is critical: {usage_percent:.1f}%")
elif usage_percent > 80:
    print(f"⚠️  Memory usage is high: {usage_percent:.1f}%")
else:
    print(f"✅ Memory usage is normal: {usage_percent:.1f}%")
EOF
    fi
}

check_service_status() {
    print_status "Checking service status..."
    
    # Check if backend is responding
    if command -v curl &> /dev/null; then
        if curl -f -s http://localhost:8000/health > /dev/null; then
            print_success "Backend service is healthy"
        else
            print_error "Backend service is not responding"
        fi
    fi
    
    # Check supervisor status (if available)
    if command -v supervisorctl &> /dev/null; then
        if supervisorctl -c config/production/supervisord.conf status > /dev/null 2>&1; then
            print_success "Supervisor services are running"
            supervisorctl -c config/production/supervisord.conf status
        else
            print_warning "Supervisor services may not be running"
        fi
    fi
}

generate_maintenance_report() {
    print_status "Generating maintenance report..."
    
    report_file="$BACKUP_DIR/maintenance_report_$TIMESTAMP.txt"
    
    {
        echo "AI Sales Automation System - Maintenance Report"
        echo "Generated: $(date)"
        echo "================================================"
        echo ""
        
        echo "System Information:"
        echo "- Hostname: $(hostname)"
        echo "- OS: $(uname -s) $(uname -r)"
        echo "- Python: $(python3 --version 2>/dev/null || echo 'Not found')"
        echo "- Disk Usage: $(df -h . | tail -1 | awk '{print $5}')"
        echo ""
        
        echo "Backup Summary:"
        echo "- MongoDB: $([ -f "$BACKUP_DIR/mongodb/mongodb_backup_$TIMESTAMP.tar.gz" ] && echo 'Success' || echo 'Failed')"
        echo "- PostgreSQL: $([ -f "$BACKUP_DIR/postgresql/postgresql_backup_$TIMESTAMP.sql.gz" ] && echo 'Success' || echo 'Failed')"
        echo "- Logs: $([ -f "$BACKUP_DIR/logs/logs_backup_$TIMESTAMP.tar.gz" ] && echo 'Success' || echo 'Failed')"
        echo "- Models: $([ -f "$BACKUP_DIR/models/models_backup_$TIMESTAMP.tar.gz" ] && echo 'Success' || echo 'Failed')"
        echo ""
        
        echo "Service Status:"
        if command -v curl &> /dev/null && curl -f -s http://localhost:8000/health > /dev/null; then
            echo "- Backend: Online"
        else
            echo "- Backend: Offline"
        fi
        echo ""
        
        echo "Recent Log Errors (last 24 hours):"
        if [ -f "$LOG_DIR/sales_ai.log" ]; then
            grep -i error "$LOG_DIR/sales_ai.log" | tail -10 || echo "No errors found"
        else
            echo "Log file not found"
        fi
        
    } > "$report_file"
    
    print_success "Maintenance report generated: $report_file"
}

show_usage() {
    echo "AI Sales Automation System - Maintenance Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  backup       Perform full system backup"
    echo "  logs         Backup and rotate logs only"
    echo "  check        Check system health"
    echo "  cleanup      Clean up old backups and logs"
    echo "  update       Check for dependency updates"
    echo "  report       Generate maintenance report"
    echo "  full         Perform all maintenance tasks"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 backup       # Backup databases and models"
    echo "  $0 check        # Check system health"
    echo "  $0 full         # Run all maintenance tasks"
}

# Main execution logic
case "${1:-full}" in
    "backup")
        print_status "Starting backup process..."
        backup_mongodb
        backup_postgresql
        backup_models
        cleanup_old_backups
        print_success "Backup process completed"
        ;;
        
    "logs")
        print_status "Starting log management..."
        backup_logs
        rotate_logs
        print_success "Log management completed"
        ;;
        
    "check")
        print_status "Starting system health check..."
        check_disk_space
        check_memory_usage
        check_service_status
        print_success "System health check completed"
        ;;
        
    "cleanup")
        print_status "Starting cleanup process..."
        cleanup_old_backups
        rotate_logs
        print_success "Cleanup process completed"
        ;;
        
    "update")
        print_status "Starting update check..."
        update_dependencies
        print_success "Update check completed"
        ;;
        
    "report")
        print_status "Generating maintenance report..."
        generate_maintenance_report
        print_success "Report generation completed"
        ;;
        
    "full")
        print_status "Starting full maintenance process..."
        echo ""
        
        check_disk_space
        check_memory_usage
        backup_mongodb
        backup_postgresql
        backup_logs
        backup_models
        rotate_logs
        cleanup_old_backups
        check_service_status
        update_dependencies
        generate_maintenance_report
        
        echo ""
        print_success "Full maintenance process completed"
        ;;
        
    "help"|"--help"|"-h")
        show_usage
        exit 0
        ;;
        
    *)
        print_error "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

print_success "Maintenance script finished successfully!"