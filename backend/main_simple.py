"""
Simple FastAPI application for AI Lead Management System
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting AI Lead Management System...")
    logger.info("🧠 Loading NLP services...")
    
    try:
        # Initialize NLP service
        from app.core.nlp_service import nlp_service
        logger.info("✅ NLP service loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load NLP service: {e}")
    
    yield
    
    logger.info("🛑 Shutting down AI Lead Management System...")

# Create FastAPI application
app = FastAPI(
    title="AI Lead Management & Voice Communication System",
    description="Advanced lead management with AI-powered analysis and voice communication",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "message": "AI Lead Management System is running",
        "version": "1.0.0",
        "services": {
            "api": "running",
            "nlp": "active"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AI Lead Management & Voice Communication System",
        "docs": "/docs",
        "health": "/health",
        "nlp_demo": "/nlp/demo"
    }

# NLP Demo endpoints
@app.get("/nlp/demo")
async def nlp_demo():
    """NLP service demonstration"""
    
    try:
        from app.core.nlp_service import nlp_service
        
        # Demo text
        demo_text = "Hi, I'm very interested in your product but need to know about pricing ASAP. We have a deadline next week and need to make a decision quickly."
        
        # Analyze text
        analysis = nlp_service.analyze_text(demo_text)
        
        return {
            "demo_text": demo_text,
            "analysis": {
                "sentiment": {
                    "score": analysis.sentiment_score,
                    "label": analysis.sentiment_label
                },
                "intent": analysis.intent_classification,
                "urgency_indicators": analysis.urgency_indicators,
                "keywords": analysis.keywords[:5],
                "topics": analysis.topics,
                "entities": [{"text": e["text"], "label": e["label"]} for e in analysis.entities[:3]]
            },
            "message": "NLP service is working! Check /docs for full API documentation."
        }
        
    except Exception as e:
        logger.error(f"NLP demo error: {e}")
        return {
            "error": "NLP service not available",
            "message": str(e),
            "status": "Please check NLP service configuration"
        }

@app.post("/nlp/analyze")
async def analyze_text_simple(text: str):
    """Simple text analysis endpoint"""
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        from app.core.nlp_service import nlp_service
        
        # Analyze the text
        analysis = nlp_service.analyze_text(text)
        
        return {
            "text": text,
            "sentiment_score": analysis.sentiment_score,
            "sentiment_label": analysis.sentiment_label,
            "intent": analysis.intent_classification,
            "confidence": analysis.confidence,
            "urgency_indicators": analysis.urgency_indicators,
            "keywords": analysis.keywords,
            "topics": analysis.topics,
            "entities": analysis.entities,
            "language": analysis.language,
            "readability_score": analysis.readability_score
        }
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/nlp/similarity")
async def calculate_similarity_simple(text1: str, text2: str):
    """Simple text similarity calculation"""
    
    if not text1 or not text2:
        raise HTTPException(status_code=400, detail="Both texts are required")
    
    try:
        from app.core.nlp_service import nlp_service
        
        # Calculate semantic similarity
        similarity = nlp_service.semantic_similarity(text1, text2)
        
        return {
            "text1": text1,
            "text2": text2,
            "semantic_similarity": similarity,
            "interpretation": "Very similar" if similarity > 0.8 else "Similar" if similarity > 0.6 else "Somewhat similar" if similarity > 0.4 else "Different"
        }
        
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return {
        "error": "Internal server error",
        "message": "Please check the server logs for details",
        "type": type(exc).__name__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)