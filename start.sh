# Railway Start Script
#!/bin/bash

echo "Starting Railway deployment..."

# Set Python path
export PYTHONPATH="/app:/app/backend"

# Change to backend directory and start the application
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT