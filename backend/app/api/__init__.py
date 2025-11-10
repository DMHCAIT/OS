"""
API Router configuration
Centralizes all API endpoints
"""

from fastapi import APIRouter
from app.api.endpoints import leads, calls, voice, analytics, users, auth

api_router = APIRouter()

# Authentication routes
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# User management routes
api_router.include_router(users.router, prefix="/users", tags=["users"])

# Lead management routes
api_router.include_router(leads.router, prefix="/leads", tags=["leads"])

# Call management routes
api_router.include_router(calls.router, prefix="/calls", tags=["calls"])

# Voice AI routes
api_router.include_router(voice.router, prefix="/voice", tags=["voice-ai"])

# Analytics routes
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])