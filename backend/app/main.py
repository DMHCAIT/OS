"""
Integrated AI/ML Backend API
FastAPI application integrating all advanced AI/ML services for sales automation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
import json
from datetime import datetime, timedelta
import uvicorn
from contextlib import asynccontextmanager

# Import our AI/ML services
from app.core.advanced_ai_service import advanced_ai_service, ConversationAnalysis, LeadScoringResult, ObjectionResponse, ConversationHistory
from app.core.ml_lead_scoring import ml_lead_scoring, LeadScoringPrediction, MLModelMetrics
from app.core.ai_conversation_engine import ai_conversation_engine, ConversationContext, AIResponse, ConversationStage, MessageType, ConversationMessage
from app.core.voice_ai_enhancement import voice_ai_enhancement, VoiceMetrics, VoiceInsight, RealTimeGuidance
from app.core.predictive_analytics import predictive_analytics, ForecastResult, PredictionType, ForecastHorizon, SalesPipelineInsight

# Import predictive business intelligence services
from app.core.market_trend_analysis import market_trend_analysis, MarketTrendPrediction, EconomicIndicator, TrendSignal
from app.core.territory_optimization import territory_optimization, TerritoryRecommendation, OptimizationObjective, PerformanceMetrics
from app.core.seasonal_patterns import seasonal_patterns, SeasonalForecast, PatternType, SeasonalStrategy
from app.core.competitive_intelligence import competitive_intelligence, CompetitiveThreat, MarketPosition, StrategicResponse

# Import monitoring and dashboard services
from app.monitoring.performance_monitor import performance_collector, PerformanceMiddleware
from app.monitoring.business_intelligence import business_intelligence
from app.monitoring.predictive_alerts import predictive_alerts
from app.api.dashboard_routes import router as dashboard_router
from app.api.intelligence_routes import router as intelligence_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            await self.send_message(message, connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting AI/ML Backend Services...")
    
    # Initialize services
    try:
        await advanced_ai_service.initialize()
        await ml_lead_scoring.load_models()
        await predictive_analytics.load_models()
        
        # Start monitoring services
        await performance_collector.start_collection()
        await predictive_alerts.start_monitoring()
        
        logger.info("All AI/ML services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI/ML Backend Services...")
    await performance_collector.stop_collection()
    await predictive_alerts.stop_monitoring()

# FastAPI app
app = FastAPI(
    title="Advanced AI/ML Sales Automation API",
    description="Comprehensive AI/ML backend for automated sales processes with advanced conversation analysis, lead scoring, voice AI, and predictive analytics",
    version="2.0.0",
    lifespan=lifespan
)

# Add performance monitoring middleware
app.add_middleware(PerformanceMiddleware, performance_collector)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dashboard and intelligence routes
app.include_router(dashboard_router)
app.include_router(intelligence_router)

# Pydantic models for API
class LeadData(BaseModel):
    id: str
    name: str
    email: str
    company: str
    job_title: str = ""
    phone: str = ""
    industry: str = ""
    company_size: int = 0
    location: str = ""
    lead_source: str = ""
    engagement_score: float = 0.0
    last_contact_date: Optional[datetime] = None
    notes: str = ""
    
class ConversationRequest(BaseModel):
    lead_id: str
    message: str
    conversation_id: str
    speaker: str = "prospect"
    
class VoiceAnalysisRequest(BaseModel):
    session_id: str
    speaker_id: str
    audio_data: str  # Base64 encoded audio
    
class PredictionRequest(BaseModel):
    prediction_type: str
    input_data: Dict[str, Any]
    forecast_horizon: str = "monthly"
    
class BatchLeadScoringRequest(BaseModel):
    leads: List[Dict[str, Any]]
    
class TrainingDataRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    model_type: str = "lead_scoring"

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "advanced_ai": "online",
            "ml_lead_scoring": "online", 
            "conversation_engine": "online",
            "voice_ai": "online",
            "predictive_analytics": "online"
        }
    }

# Advanced AI Service Endpoints

@app.post("/api/v1/conversation/analyze")
async def analyze_conversation(request: ConversationRequest):
    """Analyze conversation with advanced AI"""
    try:
        conversation_history = ConversationHistory(
            conversation_id=request.conversation_id,
            messages=[],
            lead_id=request.lead_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        analysis = await advanced_ai_service.analyze_conversation(
            request.message, conversation_history
        )
        
        return {
            "success": True,
            "analysis": {
                "sentiment": analysis.sentiment_score,
                "emotion": analysis.primary_emotion,
                "intent": analysis.detected_intent,
                "key_insights": analysis.key_insights,
                "buying_signals": analysis.buying_signals,
                "objections": analysis.objections_detected,
                "next_actions": analysis.recommended_actions,
                "urgency_level": analysis.urgency_level
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/conversation/respond")
async def generate_ai_response(request: ConversationRequest):
    """Generate AI response to customer message"""
    try:
        # Create conversation context
        context = ConversationContext(
            conversation_id=request.conversation_id,
            lead_id=request.lead_id,
            current_stage=ConversationStage.DISCOVERY,
            messages=[],
            key_insights={},
            pain_points=[],
            identified_needs=[]
        )
        
        # Generate AI response
        ai_response = await ai_conversation_engine.generate_response(
            request.message, context
        )
        
        return {
            "success": True,
            "response": {
                "content": ai_response.content,
                "confidence": ai_response.confidence_score,
                "suggested_stage": ai_response.suggested_stage.value,
                "follow_up_actions": ai_response.follow_up_actions,
                "opportunities": ai_response.detected_opportunities,
                "risks": ai_response.risk_flags,
                "personalization": ai_response.personalization_elements
            }
        }
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/objections/handle")
async def handle_objection(objection_text: str, lead_data: Optional[Dict] = None):
    """Handle customer objections with AI"""
    try:
        response = await advanced_ai_service.handle_objection(objection_text, lead_data)
        
        return {
            "success": True,
            "response": {
                "objection_type": response.objection_type,
                "suggested_response": response.suggested_response,
                "confidence": response.confidence_score,
                "follow_up_questions": response.follow_up_questions,
                "alternative_approaches": response.alternative_approaches,
                "empathy_statements": response.empathy_statements
            }
        }
    except Exception as e:
        logger.error(f"Error handling objection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML Lead Scoring Endpoints

@app.post("/api/v1/leads/score")
async def score_lead(lead_data: LeadData):
    """Score a single lead using ML algorithms"""
    try:
        lead_dict = lead_data.dict()
        prediction = await ml_lead_scoring.predict_lead_score(lead_dict)
        
        return {
            "success": True,
            "lead_score": {
                "overall_score": prediction.overall_score,
                "conversion_probability": prediction.conversion_probability,
                "confidence_interval": {
                    "lower": prediction.confidence_interval[0],
                    "upper": prediction.confidence_interval[1]
                },
                "predicted_deal_value": prediction.predicted_deal_value,
                "predicted_timeline": prediction.predicted_conversion_timeline,
                "segment": prediction.segment,
                "priority_tier": prediction.priority_tier,
                "feature_contributions": prediction.feature_contributions,
                "model_votes": prediction.model_ensemble_votes,
                "risk_assessment": prediction.risk_assessment,
                "recommendations": prediction.recommended_actions
            }
        }
    except Exception as e:
        logger.error(f"Error scoring lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/leads/batch-score")
async def batch_score_leads(request: BatchLeadScoringRequest):
    """Score multiple leads in batch"""
    try:
        results = []
        
        for lead_data in request.leads:
            try:
                prediction = await ml_lead_scoring.predict_lead_score(lead_data)
                results.append({
                    "lead_id": lead_data.get("id", "unknown"),
                    "success": True,
                    "score": prediction.overall_score,
                    "probability": prediction.conversion_probability,
                    "segment": prediction.segment,
                    "priority": prediction.priority_tier
                })
            except Exception as e:
                results.append({
                    "lead_id": lead_data.get("id", "unknown"),
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total_processed": len(results),
            "successful": sum(1 for r in results if r["success"])
        }
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ml/train")
async def train_ml_models(request: TrainingDataRequest, background_tasks: BackgroundTasks):
    """Train ML models with new data"""
    try:
        if request.model_type == "lead_scoring":
            # Train lead scoring models in background
            background_tasks.add_task(
                ml_lead_scoring.train_models, 
                request.training_data
            )
        elif request.model_type == "predictive":
            # Train predictive models in background
            background_tasks.add_task(
                train_predictive_models_task,
                request.training_data
            )
        
        return {
            "success": True,
            "message": f"Started training {request.model_type} models",
            "data_points": len(request.training_data)
        }
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice AI Endpoints

@app.post("/api/v1/voice/start-session")
async def start_voice_session(session_id: str, speaker_id: str):
    """Start voice analysis session"""
    try:
        success = await voice_ai_enhancement.start_voice_analysis(session_id, speaker_id)
        
        return {
            "success": success,
            "session_id": session_id,
            "message": "Voice analysis session started" if success else "Failed to start session"
        }
    except Exception as e:
        logger.error(f"Error starting voice session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/analyze")
async def analyze_voice(request: VoiceAnalysisRequest):
    """Analyze voice audio chunk"""
    try:
        import base64
        audio_bytes = base64.b64decode(request.audio_data)
        
        metrics = await voice_ai_enhancement.process_audio_chunk(
            request.session_id, audio_bytes, request.speaker_id
        )
        
        if metrics:
            return {
                "success": True,
                "metrics": {
                    "timestamp": metrics.timestamp.isoformat(),
                    "speaker_id": metrics.speaker_id,
                    "volume_level": metrics.volume_level,
                    "clarity_score": metrics.clarity_score,
                    "speech_rate": metrics.speech_rate,
                    "primary_emotion": metrics.primary_emotion.value,
                    "emotion_confidence": metrics.emotion_confidence,
                    "engagement_score": metrics.engagement_score,
                    "stress_indicators": metrics.stress_indicators
                }
            }
        else:
            return {
                "success": False,
                "message": "No speech detected or session not active"
            }
    except Exception as e:
        logger.error(f"Error analyzing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voice/session/{session_id}/summary")
async def get_voice_session_summary(session_id: str):
    """Get voice session summary"""
    try:
        summary = await voice_ai_enhancement.get_session_summary(session_id)
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/stop-session/{session_id}")
async def stop_voice_session(session_id: str):
    """Stop voice analysis session"""
    try:
        summary = await voice_ai_enhancement.stop_voice_analysis(session_id)
        return {
            "success": True,
            "final_summary": summary
        }
    except Exception as e:
        logger.error(f"Error stopping voice session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Predictive Analytics Endpoints

@app.post("/api/v1/predictions/forecast")
async def make_prediction(request: PredictionRequest):
    """Make predictive forecast"""
    try:
        # Map string to enum
        prediction_type_map = {
            "revenue": PredictionType.REVENUE_FORECAST,
            "deal_probability": PredictionType.DEAL_PROBABILITY,
            "pipeline_conversion": PredictionType.PIPELINE_CONVERSION,
            "sales_velocity": PredictionType.SALES_VELOCITY,
            "customer_lifetime_value": PredictionType.CUSTOMER_LIFETIME_VALUE,
            "churn": PredictionType.CHURN_PREDICTION
        }
        
        horizon_map = {
            "weekly": ForecastHorizon.WEEKLY,
            "monthly": ForecastHorizon.MONTHLY,
            "quarterly": ForecastHorizon.QUARTERLY,
            "yearly": ForecastHorizon.YEARLY
        }
        
        prediction_type = prediction_type_map.get(request.prediction_type)
        forecast_horizon = horizon_map.get(request.forecast_horizon, ForecastHorizon.MONTHLY)
        
        if not prediction_type:
            raise HTTPException(status_code=400, detail="Invalid prediction type")
        
        forecast = await predictive_analytics.make_prediction(
            prediction_type, request.input_data, forecast_horizon
        )
        
        return {
            "success": True,
            "forecast": {
                "prediction_type": forecast.prediction_type.value,
                "horizon": forecast.forecast_horizon.value,
                "predicted_value": forecast.predicted_value,
                "confidence_interval": {
                    "lower": forecast.confidence_interval_lower,
                    "upper": forecast.confidence_interval_upper
                },
                "confidence_score": forecast.confidence_score,
                "contributing_factors": forecast.factors_contributing,
                "model_used": forecast.model_used,
                "timestamp": forecast.timestamp.isoformat(),
                "metadata": forecast.metadata
            }
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analytics/sales-insights")
async def generate_sales_insights(historical_data: Dict[str, Any]):
    """Generate sales insights using predictive analytics"""
    try:
        insights = await predictive_analytics.generate_sales_insights(historical_data)
        
        return {
            "success": True,
            "insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "current_value": insight.current_value,
                    "predicted_value": insight.predicted_value,
                    "change_percentage": insight.change_percentage,
                    "confidence": insight.confidence,
                    "recommendations": insight.recommendations,
                    "impact_level": insight.impact_level
                }
                for insight in insights
            ],
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive Dashboard Endpoint

@app.get("/api/v1/dashboard/comprehensive")
async def get_comprehensive_dashboard(lead_id: Optional[str] = None):
    """Get comprehensive dashboard data combining all AI/ML insights"""
    try:
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "ai_services_status": {
                "advanced_ai": "operational",
                "ml_lead_scoring": "operational",
                "conversation_engine": "operational", 
                "voice_ai": "operational",
                "predictive_analytics": "operational"
            }
        }
        
        # If lead_id provided, get specific insights
        if lead_id:
            # Mock lead data for demonstration
            lead_data = {
                "id": lead_id,
                "name": "John Doe",
                "company": "TechCorp",
                "email": "john@techcorp.com",
                "engagement_score": 0.75,
                "company_size": 500
            }
            
            # Get lead scoring
            scoring_prediction = await ml_lead_scoring.predict_lead_score(lead_data)
            dashboard_data["lead_scoring"] = {
                "overall_score": scoring_prediction.overall_score,
                "conversion_probability": scoring_prediction.conversion_probability,
                "segment": scoring_prediction.segment,
                "priority_tier": scoring_prediction.priority_tier,
                "recommendations": scoring_prediction.recommended_actions
            }
        
        # Get general sales insights
        sample_historical_data = {
            "revenue": 100000,
            "pipeline_value": 250000,
            "conversion_rate": 0.25,
            "sales_velocity": 1.5
        }
        
        insights = await predictive_analytics.generate_sales_insights(sample_historical_data)
        dashboard_data["sales_insights"] = [
            {
                "type": insight.insight_type,
                "description": insight.description,
                "impact_level": insight.impact_level,
                "recommendations": insight.recommendations[:2]  # Top 2 recommendations
            }
            for insight in insights[:3]  # Top 3 insights
        ]
        
        return {
            "success": True,
            "dashboard": dashboard_data
        }
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time AI guidance
@app.websocket("/ws/ai-guidance")
async def websocket_ai_guidance(websocket: WebSocket):
    """WebSocket endpoint for real-time AI guidance during sales calls"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "conversation_message":
                # Analyze conversation in real-time
                conversation_analysis = await advanced_ai_service.analyze_conversation(
                    message_data.get("message", ""),
                    ConversationHistory(
                        conversation_id=message_data.get("conversation_id", ""),
                        messages=[],
                        lead_id=message_data.get("lead_id", ""),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                )
                
                # Send real-time guidance
                guidance = {
                    "type": "conversation_guidance",
                    "sentiment": conversation_analysis.sentiment_score,
                    "emotion": conversation_analysis.primary_emotion,
                    "buying_signals": conversation_analysis.buying_signals,
                    "objections": conversation_analysis.objections_detected,
                    "recommendations": conversation_analysis.recommended_actions,
                    "urgency": conversation_analysis.urgency_level,
                    "timestamp": datetime.now().isoformat()
                }
                
                await manager.send_message(guidance, websocket)
                
            elif message_type == "voice_data":
                # Process voice data if available
                session_id = message_data.get("session_id")
                if session_id:
                    # Send voice analysis update
                    voice_update = {
                        "type": "voice_analysis",
                        "session_id": session_id,
                        "status": "processing",
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_message(voice_update, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task functions
async def train_predictive_models_task(training_data: List[Dict[str, Any]]):
    """Background task for training predictive models"""
    try:
        import pandas as pd
        
        # Convert to appropriate format for predictive analytics
        df = pd.DataFrame(training_data)
        training_data_dict = {PredictionType.REVENUE_FORECAST: df}
        
        await predictive_analytics.train_models(training_data_dict)
        logger.info("Predictive models training completed")
    except Exception as e:
        logger.error(f"Error in predictive models training task: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": {
            "code": 500,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )