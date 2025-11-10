"""
API v1 router configuration
"""

from fastapi import APIRouter
from app.api.v1 import auth, leads, voice, nlp

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(leads.router, prefix="/leads", tags=["leads"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(nlp.router, prefix="/nlp", tags=["nlp"])

# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Global health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "running",
            "database": "connected",
            "ml_engine": "active",
            "voice_ai": "ready",
            "nlp_service": "loaded"
        }
    }