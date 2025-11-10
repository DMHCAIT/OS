"""
Simplified Railway Deployment Main File
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Lead Management System",
    description="AI-powered lead management and voice communication system",
    version="1.0.0",
    docs_url="/docs"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 AI Lead Management System is running!",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "service": "ai-lead-management",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_status": "online",
        "features": [
            "🧠 AI Lead Scoring",
            "🎤 Voice Processing", 
            "📊 Analytics Dashboard",
            "🤖 Machine Learning"
        ]
    }

# Basic API routes
@app.post("/api/leads")
async def create_lead(lead_data: dict):
    """Create a new lead"""
    return {"message": "Lead created", "lead_id": "12345", "data": lead_data}

@app.get("/api/leads")
async def get_leads():
    """Get all leads"""
    return {
        "leads": [
            {"id": "1", "name": "John Doe", "email": "john@example.com", "score": 85},
            {"id": "2", "name": "Jane Smith", "email": "jane@example.com", "score": 92}
        ]
    }

@app.post("/api/voice/transcribe")
async def transcribe_audio():
    """Voice transcription endpoint"""
    return {"transcript": "Hello, this is a sample transcription", "confidence": 0.95}

@app.get("/api/analytics/dashboard")
async def get_dashboard_data():
    """Dashboard analytics data"""
    return {
        "total_leads": 1250,
        "conversion_rate": 0.24,
        "calls_today": 47,
        "ai_score_avg": 78.5,
        "chart_data": [65, 75, 80, 85, 90]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)