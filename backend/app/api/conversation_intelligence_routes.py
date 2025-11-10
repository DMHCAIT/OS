"""
Advanced Conversation Intelligence API Routes
FastAPI endpoints for the comprehensive conversation intelligence system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import numpy as np
import io
import librosa

from ..core.advanced_conversation_intelligence import advanced_conversation_intelligence

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/conversation-intelligence", tags=["conversation-intelligence"])

# Pydantic models for request/response
class ConversationMessage(BaseModel):
    conversation_id: str
    participant_id: str
    message: str
    current_script: Optional[str] = None
    conversation_context: Optional[Dict[str, Any]] = None

class ConversationAnalysisRequest(BaseModel):
    conversation_id: str
    participant_id: str
    message: str
    audio_file_path: Optional[str] = None
    current_script: Optional[str] = None
    conversation_context: Optional[Dict[str, Any]] = None

class IntelligenceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class ConversationSummaryRequest(BaseModel):
    conversation_id: str


@router.post("/analyze", response_model=IntelligenceResponse)
async def analyze_conversation_message(request: ConversationAnalysisRequest):
    """
    Comprehensive conversation intelligence analysis
    Processes message through all intelligence systems: language, emotion, adaptation, competitive
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process audio if provided
        audio_data = None
        if request.audio_file_path:
            try:
                # Load audio file (assuming it's been uploaded separately)
                audio_data, _ = librosa.load(request.audio_file_path, sr=16000)
            except Exception as e:
                logger.warning(f"Could not process audio file: {e}")
        
        # Run comprehensive analysis
        result = await advanced_conversation_intelligence.analyze_conversation_message(
            conversation_id=request.conversation_id,
            participant_id=request.participant_id,
            message=request.message,
            audio_data=audio_data,
            current_script=request.current_script,
            conversation_context=request.conversation_context
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Convert dataclass to dict for JSON serialization
        result_dict = {
            'conversation_id': result.conversation_id,
            'participant_id': result.participant_id,
            'message': result.message,
            'analysis_timestamp': result.analysis_timestamp.isoformat(),
            'language_analysis': result.language_analysis,
            'localized_response': result.localized_response,
            'emotion_analysis': result.emotion_analysis,
            'empathy_recommendations': result.empathy_recommendations,
            'behavioral_insights': result.behavioral_insights,
            'script_adaptation': result.script_adaptation,
            'competitive_analysis': result.competitive_analysis,
            'strategic_responses': result.strategic_responses,
            'priority_actions': result.priority_actions,
            'conversation_health_score': result.conversation_health_score,
            'engagement_forecast': result.engagement_forecast,
            'adaptive_strategy': result.adaptive_strategy
        }
        
        return IntelligenceResponse(
            success=True,
            data=result_dict,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in conversation intelligence analysis: {e}")
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return IntelligenceResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )


@router.post("/analyze-with-audio")
async def analyze_conversation_with_audio(
    conversation_id: str,
    participant_id: str,
    message: str,
    audio_file: UploadFile = File(None),
    current_script: Optional[str] = None
):
    """
    Analyze conversation message with audio upload
    Supports real-time audio emotion analysis
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process uploaded audio
        audio_data = None
        if audio_file:
            try:
                audio_content = await audio_file.read()
                audio_array = np.frombuffer(audio_content, dtype=np.float32)
                # Resample to 16kHz if needed (assuming input is proper audio format)
                audio_data = audio_array
            except Exception as e:
                logger.warning(f"Could not process uploaded audio: {e}")
        
        # Run analysis
        result = await advanced_conversation_intelligence.analyze_conversation_message(
            conversation_id=conversation_id,
            participant_id=participant_id,
            message=message,
            audio_data=audio_data,
            current_script=current_script
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "success": True,
            "conversation_id": result.conversation_id,
            "analysis": {
                "priority_actions": result.priority_actions,
                "health_score": result.conversation_health_score,
                "engagement_forecast": result.engagement_forecast,
                "adaptive_strategy": result.adaptive_strategy,
                "emotion_state": result.emotion_analysis.get('combined_analysis', {}).get('emotional_state'),
                "competitive_alerts": len(result.competitive_analysis.get('competitive_mentions', [])) > 0,
                "localization_needed": result.localized_response is not None
            },
            "detailed_analysis": {
                "language": result.language_analysis,
                "emotion": result.emotion_analysis,
                "adaptation": result.script_adaptation,
                "competitive": result.competitive_analysis
            },
            "recommendations": {
                "script_adaptation": result.script_adaptation.get('recommended_content'),
                "empathy_response": result.empathy_recommendations,
                "competitive_responses": result.strategic_responses,
                "localized_response": result.localized_response
            },
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error in audio conversation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{conversation_id}")
async def get_conversation_summary(conversation_id: str):
    """
    Get comprehensive conversation summary and insights
    """
    try:
        summary = await advanced_conversation_intelligence.generate_conversation_summary(conversation_id)
        
        if 'error' in summary:
            raise HTTPException(status_code=404, detail=summary['error'])
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{conversation_id}")
async def get_conversation_insights(conversation_id: str):
    """
    Get real-time conversation insights and recommendations
    """
    try:
        session = advanced_conversation_intelligence.active_sessions.get(conversation_id)
        if not session:
            raise HTTPException(status_code=404, detail="Conversation session not found")
        
        if not session.intelligence_timeline:
            return {
                "success": True,
                "insights": "No analysis data available yet",
                "recommendations": ["Begin conversation analysis"]
            }
        
        latest_analysis = session.intelligence_timeline[-1]
        
        insights = {
            "current_state": {
                "health_score": latest_analysis.conversation_health_score,
                "engagement_forecast": latest_analysis.engagement_forecast,
                "adaptive_strategy": latest_analysis.adaptive_strategy,
                "priority_actions": latest_analysis.priority_actions
            },
            "participant_profile": {
                "language": session.cultural_context.get('detected_language', 'en'),
                "cultural_background": session.cultural_context.get('cultural_background'),
                "emotional_state": session.emotional_journey[-1].get('emotional_state') if session.emotional_journey else 'unknown',
                "engagement_trend": session.engagement_trend
            },
            "competitive_context": session.competitive_context,
            "conversation_flow": {
                "current_stage": session.current_stage,
                "message_count": len(session.conversation_history),
                "timeline_length": len(session.intelligence_timeline)
            },
            "real_time_recommendations": latest_analysis.priority_actions,
            "next_best_actions": [
                action for action in latest_analysis.priority_actions 
                if "URGENT" in action or "Implement" in action
            ]
        }
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "insights": insights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-check")
async def get_system_health():
    """
    Check health and performance of all conversation intelligence systems
    """
    try:
        metrics = await advanced_conversation_intelligence.get_system_performance_metrics()
        
        return {
            "success": True,
            "system_status": "operational",
            "metrics": metrics,
            "capabilities": {
                "multilingual_support": "active",
                "emotion_detection": "active", 
                "script_adaptation": "active",
                "competitive_intelligence": "active",
                "real_time_analysis": "active"
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            "success": False,
            "system_status": "error",
            "error": str(e)
        }


@router.post("/initialize")
async def initialize_conversation_intelligence():
    """
    Initialize all conversation intelligence systems
    """
    try:
        await advanced_conversation_intelligence.initialize()
        
        return {
            "success": True,
            "message": "Advanced Conversation Intelligence System initialized successfully",
            "systems_initialized": [
                "Multilingual Intelligence",
                "Emotion Detection", 
                "Dynamic Script Adaptation",
                "Competitor Intelligence"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error initializing conversation intelligence: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.get("/active-conversations")
async def get_active_conversations():
    """
    Get list of active conversation sessions
    """
    try:
        sessions = advanced_conversation_intelligence.active_sessions
        
        active_conversations = []
        for conversation_id, session in sessions.items():
            latest_message = session.conversation_history[-1] if session.conversation_history else None
            latest_analysis = session.intelligence_timeline[-1] if session.intelligence_timeline else None
            
            conversation_info = {
                "conversation_id": conversation_id,
                "participant_count": len(set([msg.get('participant_id') for msg in session.conversation_history])),
                "message_count": len(session.conversation_history),
                "current_stage": session.current_stage,
                "engagement_trend": session.engagement_trend,
                "health_score": latest_analysis.conversation_health_score if latest_analysis else 0.5,
                "last_activity": latest_message.get('timestamp') if latest_message else None,
                "competitive_active": bool(session.competitive_context.get('active_competitors')),
                "language": session.cultural_context.get('detected_language', 'en')
            }
            active_conversations.append(conversation_info)
        
        return {
            "success": True,
            "total_active_conversations": len(active_conversations),
            "conversations": active_conversations
        }
        
    except Exception as e:
        logger.error(f"Error getting active conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-analyze")
async def quick_conversation_analysis(message: ConversationMessage):
    """
    Quick conversation analysis for real-time feedback
    Returns essential insights without full processing
    """
    try:
        # Run lightweight analysis
        result = await advanced_conversation_intelligence.analyze_conversation_message(
            conversation_id=message.conversation_id,
            participant_id=message.participant_id,
            message=message.message,
            current_script=message.current_script,
            conversation_context=message.conversation_context
        )
        
        # Return simplified response for real-time use
        quick_insights = {
            "health_score": result.conversation_health_score,
            "engagement_forecast": result.engagement_forecast,
            "top_priority_action": result.priority_actions[0] if result.priority_actions else "Continue conversation",
            "emotion_detected": result.emotion_analysis.get('combined_analysis', {}).get('emotional_state', 'neutral'),
            "adaptation_needed": result.script_adaptation.get('adaptation_type') != 'maintain',
            "competitive_alert": len(result.competitive_analysis.get('competitive_mentions', [])) > 0,
            "language_support_needed": result.localized_response is not None
        }
        
        return {
            "success": True,
            "conversation_id": message.conversation_id,
            "quick_insights": quick_insights,
            "recommendations": result.priority_actions[:3]  # Top 3 actions
        }
        
    except Exception as e:
        logger.error(f"Error in quick conversation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream-insights/{conversation_id}")
async def stream_conversation_insights(conversation_id: str):
    """
    Stream real-time conversation insights using Server-Sent Events
    """
    async def generate_insights():
        try:
            while True:
                session = advanced_conversation_intelligence.active_sessions.get(conversation_id)
                if session and session.intelligence_timeline:
                    latest = session.intelligence_timeline[-1]
                    
                    insights = {
                        "timestamp": latest.analysis_timestamp.isoformat(),
                        "health_score": latest.conversation_health_score,
                        "engagement": latest.engagement_forecast,
                        "priority_action": latest.priority_actions[0] if latest.priority_actions else None,
                        "adaptive_strategy": latest.adaptive_strategy
                    }
                    
                    yield f"data: {json.dumps(insights)}\n\n"
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in streaming insights: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_insights(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.get("/performance-metrics")
async def get_performance_metrics():
    """
    Get detailed performance metrics for all conversation intelligence systems
    """
    try:
        metrics = await advanced_conversation_intelligence.get_system_performance_metrics()
        
        # Add additional analytics
        active_sessions = len(advanced_conversation_intelligence.active_sessions)
        total_timeline_entries = sum(
            len(session.intelligence_timeline) 
            for session in advanced_conversation_intelligence.active_sessions.values()
        )
        
        analytics = {
            "usage_statistics": {
                "active_sessions": active_sessions,
                "total_analyses_completed": total_timeline_entries,
                "average_session_length": total_timeline_entries / max(active_sessions, 1)
            },
            "system_performance": metrics,
            "feature_utilization": {
                "multilingual_conversations": sum(
                    1 for session in advanced_conversation_intelligence.active_sessions.values()
                    if session.cultural_context.get('detected_language', 'en') != 'en'
                ),
                "emotional_interventions_active": sum(
                    1 for session in advanced_conversation_intelligence.active_sessions.values()
                    if session.emotional_journey and 
                    session.emotional_journey[-1].get('emotional_state') in ['frustrated', 'worried', 'defensive']
                ),
                "competitive_situations_active": sum(
                    1 for session in advanced_conversation_intelligence.active_sessions.values()
                    if session.competitive_context.get('active_competitors')
                ),
                "high_adaptation_conversations": sum(
                    1 for session in advanced_conversation_intelligence.active_sessions.values()
                    if len(session.adaptation_history) > 5
                )
            }
        }
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))