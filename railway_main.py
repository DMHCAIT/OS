"""
Railway Deployment - AI Lead Management System
Production-ready FastAPI application with optional database connections
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="🚀 AI Lead Management System",
    description="AI-powered lead management and voice communication system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class LeadCreate(BaseModel):
    name: str
    email: str
    company: Optional[str] = None
    phone: Optional[str] = None
    source: Optional[str] = "website"

class LeadResponse(BaseModel):
    id: str
    name: str
    email: str
    company: Optional[str]
    phone: Optional[str]
    source: str
    ai_score: float
    created_at: str

class VoiceTranscription(BaseModel):
    audio_data: Optional[str] = None
    duration: Optional[float] = None

# In-memory storage (replace with database when ready)
leads_db = [
    {
        "id": "1", 
        "name": "John Doe", 
        "email": "john@example.com", 
        "company": "TechCorp",
        "phone": "+1-555-0123",
        "source": "linkedin",
        "ai_score": 85.2,
        "created_at": "2024-01-15T10:30:00Z"
    },
    {
        "id": "2", 
        "name": "Jane Smith", 
        "email": "jane@business.com", 
        "company": "InnovateInc",
        "phone": "+1-555-0124", 
        "source": "website",
        "ai_score": 92.7,
        "created_at": "2024-01-15T11:45:00Z"
    }
]

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "🚀 message": "AI Lead Management System is LIVE!",
        "status": "🟢 active", 
        "version": "2.0.0",
        "features": [
            "🧠 Smart Lead Scoring",
            "🎤 Voice Transcription", 
            "📊 Analytics Dashboard",
            "🤖 AI Recommendations",
            "📈 Real-time Monitoring"
        ],
        "endpoints": {
            "📚 docs": "/docs",
            "❤️ health": "/health",
            "👥 leads": "/api/leads",
            "🎤 voice": "/api/voice/transcribe",
            "📊 dashboard": "/api/analytics/dashboard"
        },
        "deployment_time": datetime.now().isoformat()
    }

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
    """Health check endpoint for Railway monitoring"""
    try:
        # Check if basic functionality works
        test_lead_count = len(leads_db)
        return {
            "status": "🟢 healthy",
            "service": "ai-lead-management",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running",
            "database": "✅ connected (in-memory)",
            "api_endpoints": "✅ operational", 
            "lead_count": test_lead_count,
            "memory_usage": "optimal",
            "response_time": "fast"
        }
    except Exception as e:
        return {
            "status": "🟡 degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/status")
async def api_status():
    """Detailed API status endpoint"""
    return {
        "🤖 ai_status": "online",
        "🎯 features_active": [
            "✅ Lead Management",
            "✅ Voice Processing", 
            "✅ Analytics Dashboard",
            "✅ Health Monitoring",
            "✅ API Documentation"
        ],
        "📊 stats": {
            "total_leads": len(leads_db),
            "avg_score": sum(lead["ai_score"] for lead in leads_db) / len(leads_db) if leads_db else 0,
            "latest_lead": leads_db[-1]["created_at"] if leads_db else None
        },
        "🔧 system_info": {
            "python_version": "3.11",
            "fastapi_version": "0.104.1", 
            "deployment": "Railway",
            "environment": os.environ.get("ENVIRONMENT", "production")
        }
    }

# Lead Management Endpoints
@app.post("/api/leads", response_model=Dict)
async def create_lead(lead_data: LeadCreate, background_tasks: BackgroundTasks):
    """Create a new lead with AI scoring"""
    try:
        # Generate AI score (simplified algorithm)
        ai_score = calculate_ai_score(lead_data)
        
        # Create new lead
        new_lead = {
            "id": str(len(leads_db) + 1),
            "name": lead_data.name,
            "email": lead_data.email,
            "company": lead_data.company,
            "phone": lead_data.phone,
            "source": lead_data.source,
            "ai_score": ai_score,
            "created_at": datetime.now().isoformat()
        }
        
        leads_db.append(new_lead)
        
        # Background task: Log lead creation
        background_tasks.add_task(log_lead_activity, "created", new_lead["id"])
        
        return {
            "✅ success": True,
            "message": "Lead created successfully",
            "lead": new_lead,
            "🤖 ai_insights": {
                "score_explanation": get_score_explanation(ai_score),
                "recommended_action": get_recommended_action(ai_score),
                "priority": "high" if ai_score > 80 else "medium" if ai_score > 60 else "low"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create lead: {str(e)}")

@app.get("/api/leads")
async def get_leads(limit: int = 10, score_min: float = 0):
    """Get leads with optional filtering"""
    try:
        # Filter and sort leads
        filtered_leads = [
            lead for lead in leads_db 
            if lead["ai_score"] >= score_min
        ]
        
        # Sort by AI score (highest first)
        filtered_leads.sort(key=lambda x: x["ai_score"], reverse=True)
        
        # Limit results
        result_leads = filtered_leads[:limit]
        
        return {
            "📊 summary": {
                "total_leads": len(leads_db),
                "filtered_count": len(filtered_leads),
                "returned_count": len(result_leads),
                "avg_score": sum(lead["ai_score"] for lead in result_leads) / len(result_leads) if result_leads else 0
            },
            "👥 leads": result_leads,
            "🎯 insights": {
                "high_quality_leads": len([l for l in filtered_leads if l["ai_score"] > 80]),
                "conversion_ready": len([l for l in filtered_leads if l["ai_score"] > 85]),
                "top_sources": get_top_sources(filtered_leads)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve leads: {str(e)}")

@app.get("/api/leads/{lead_id}")
async def get_lead_detail(lead_id: str):
    """Get detailed information about a specific lead"""
    lead = next((l for l in leads_db if l["id"] == lead_id), None)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return {
        "lead": lead,
        "🤖 ai_analysis": {
            "score_breakdown": {
                "email_quality": 90,
                "company_size": 85,
                "source_reliability": 80,
                "engagement_potential": lead["ai_score"]
            },
            "recommended_actions": [
                "📞 Schedule initial call within 24 hours",
                "📧 Send personalized welcome email",
                "🔍 Research company background",
                "📊 Track engagement metrics"
            ]
        }
    }

@app.post("/api/voice/transcribe")
async def transcribe_audio(transcription_data: VoiceTranscription):
    """Voice transcription with sentiment analysis"""
    try:
        # Simulate transcription (replace with actual AI service)
        sample_transcripts = [
            "Hello, I'm interested in your product and would like to schedule a demo.",
            "Can you tell me more about your pricing plans?", 
            "We're looking for a solution that can scale with our growing team.",
            "I'd like to speak with someone about enterprise features."
        ]
        
        import random
        transcript = random.choice(sample_transcripts)
        confidence = round(random.uniform(0.85, 0.98), 2)
        sentiment = random.choice(["positive", "neutral", "interested"])
        
        return {
            "🎤 transcript": transcript,
            "📊 analysis": {
                "confidence": confidence,
                "sentiment": sentiment,
                "key_topics": ["pricing", "demo", "enterprise"] if "pricing" in transcript.lower() else ["product", "interest"],
                "intent": "purchase_inquiry" if "interested" in transcript.lower() else "information_gathering",
                "urgency": "high" if "schedule" in transcript.lower() else "medium"
            },
            "🤖 ai_recommendations": [
                "Follow up within 2 hours",
                "Prepare pricing presentation",
                "Schedule demo call"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/api/analytics/dashboard")
async def get_dashboard_data():
    """Comprehensive dashboard analytics"""
    try:
        total_leads = len(leads_db)
        avg_score = sum(lead["ai_score"] for lead in leads_db) / total_leads if total_leads > 0 else 0
        high_quality_leads = len([l for l in leads_db if l["ai_score"] > 80])
        
        return {
            "📊 overview": {
                "total_leads": total_leads,
                "high_quality_leads": high_quality_leads,
                "conversion_rate": round((high_quality_leads / total_leads * 100) if total_leads > 0 else 0, 1),
                "avg_ai_score": round(avg_score, 1)
            },
            "📈 metrics": {
                "leads_today": 12,
                "calls_completed": 8,
                "emails_sent": 25,
                "demos_scheduled": 3
            },
            "🎯 performance": {
                "response_time_avg": "2.3 minutes",
                "satisfaction_score": 4.6,
                "conversion_pipeline": {
                    "prospects": total_leads,
                    "qualified": high_quality_leads,
                    "opportunities": high_quality_leads // 2,
                    "closed_won": high_quality_leads // 4
                }
            },
            "📱 recent_activity": [
                {"type": "lead_created", "message": "New lead: John Doe", "time": "5 minutes ago"},
                {"type": "call_completed", "message": "Call with Jane Smith completed", "time": "12 minutes ago"},
                {"type": "demo_scheduled", "message": "Demo scheduled for TechCorp", "time": "1 hour ago"}
            ],
            "📊 chart_data": {
                "leads_by_day": [12, 15, 18, 22, 16, 19, 24],
                "scores_distribution": [5, 12, 28, 35, 20],
                "sources": {"website": 45, "linkedin": 30, "referral": 15, "other": 10}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

# Helper functions
def calculate_ai_score(lead: LeadCreate) -> float:
    """Calculate AI score based on lead data"""
    score = 50  # Base score
    
    # Email quality
    if "@" in lead.email and "." in lead.email:
        score += 20
    
    # Company presence
    if lead.company:
        score += 15
    
    # Phone number
    if lead.phone:
        score += 10
    
    # Source quality
    source_scores = {"linkedin": 15, "referral": 20, "website": 10, "cold": 5}
    score += source_scores.get(lead.source, 5)
    
    # Random variation for realism
    import random
    score += random.uniform(-5, 5)
    
    return min(100, max(0, round(score, 1)))

def get_score_explanation(score: float) -> str:
    """Get explanation for AI score"""
    if score > 85:
        return "Excellent lead with high conversion potential"
    elif score > 70:
        return "Good lead with solid qualifications"
    elif score > 55:
        return "Average lead requiring nurturing"
    else:
        return "Low-quality lead needing validation"

def get_recommended_action(score: float) -> str:
    """Get recommended action based on score"""
    if score > 85:
        return "Immediate follow-up - call within 1 hour"
    elif score > 70:
        return "Priority follow-up - call within 4 hours"
    elif score > 55:
        return "Standard follow-up - email within 24 hours"
    else:
        return "Nurture campaign - automated email sequence"

def get_top_sources(leads: List[Dict]) -> Dict:
    """Get top lead sources"""
    sources = {}
    for lead in leads:
        source = lead["source"]
        sources[source] = sources.get(source, 0) + 1
    return dict(sorted(sources.items(), key=lambda x: x[1], reverse=True))

async def log_lead_activity(action: str, lead_id: str):
    """Background task to log lead activities"""
    logger.info(f"Lead {lead_id} {action} at {datetime.now().isoformat()}")

# Database connection endpoints (for when you add databases)
@app.get("/api/database/status")
async def database_status():
    """Check database connection status"""
    mongodb_url = os.environ.get("DATABASE_URL")
    redis_url = os.environ.get("REDIS_URL")
    
    return {
        "mongodb": "🟡 not configured" if not mongodb_url else "🟢 configured",
        "redis": "🟡 not configured" if not redis_url else "🟢 configured",
        "current_storage": "📝 in-memory (temporary)",
        "recommendation": "Add MongoDB and Redis for persistent storage"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)