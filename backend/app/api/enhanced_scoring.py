"""
Enhanced Lead Scoring API Endpoints
FastAPI integration for advanced ML lead scoring system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import asyncio

from ..ml.enhanced_lead_scoring import (
    enhanced_lead_scoring_system,
    ComprehensiveLeadAnalysis,
    MLSystemStatus
)
from ..ml.behavioral_scoring import BehaviorType
from ..ml.churn_prediction import ChurnPrediction

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/enhanced-scoring", tags=["Enhanced Lead Scoring"])

# Pydantic models for request/response
class LeadDataInput(BaseModel):
    lead_id: str = Field(..., description="Unique lead identifier")
    contact_info: Dict[str, Any] = Field(default={}, description="Contact information")
    demographic_data: Dict[str, Any] = Field(default={}, description="Demographic data")
    behavioral_data: Dict[str, Any] = Field(default={}, description="Behavioral data")
    interaction_history: List[Dict[str, Any]] = Field(default=[], description="Interaction history")
    custom_attributes: Dict[str, Any] = Field(default={}, description="Custom attributes")

class BehavioralEventInput(BaseModel):
    lead_id: str = Field(..., description="Lead identifier")
    event_type: str = Field(..., description="Type of behavioral event")
    event_data: Dict[str, Any] = Field(default={}, description="Event data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    source: str = Field(default="api", description="Event source")

class BatchAnalysisRequest(BaseModel):
    leads_data: List[LeadDataInput] = Field(..., description="List of leads to analyze")
    batch_size: int = Field(default=10, description="Batch processing size")
    priority_filter: Optional[str] = Field(None, description="Priority filter")

class ModelRetrainingRequest(BaseModel):
    training_data_path: str = Field(..., description="Path to training data")
    validation_data_path: Optional[str] = Field(None, description="Path to validation data")
    model_types: List[str] = Field(default=["all"], description="Models to retrain")

# Dependency for system initialization
async def get_enhanced_scoring_system():
    """Dependency to ensure system is initialized"""
    if not hasattr(enhanced_lead_scoring_system, '_initialized'):
        await enhanced_lead_scoring_system.initialize()
        enhanced_lead_scoring_system._initialized = True
    return enhanced_lead_scoring_system

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Enhanced Lead Scoring System"
    }

@router.get("/system/status")
async def get_system_status(
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        performance_metrics = await system.get_system_performance_metrics()
        return {
            "status": "success",
            "system_health": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/comprehensive")
async def analyze_lead_comprehensive(
    lead_data: LeadDataInput,
    force_refresh: bool = False,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Comprehensive lead analysis using all ML systems"""
    try:
        logger.info(f"Running comprehensive analysis for lead {lead_data.lead_id}")
        
        # Convert Pydantic model to dict
        lead_dict = lead_data.dict()
        
        # Run comprehensive analysis
        analysis = await system.analyze_lead_comprehensive(
            lead_data.lead_id,
            lead_dict,
            force_refresh=force_refresh
        )
        
        # Convert analysis to serializable format
        result = {
            "status": "success",
            "lead_id": analysis.lead_id,
            "analysis": {
                "final_lead_score": analysis.final_lead_score,
                "confidence_level": analysis.confidence_level,
                "priority_ranking": analysis.priority_ranking,
                "action_recommendations": analysis.action_recommendations,
                "optimal_timing": analysis.optimal_timing,
                "analysis_timestamp": analysis.analysis_timestamp.isoformat()
            },
            "deep_learning_scores": {
                "overall_score": analysis.deep_learning_scores.overall_score if analysis.deep_learning_scores else None,
                "confidence": analysis.deep_learning_scores.confidence if analysis.deep_learning_scores else None,
                "prediction_components": {
                    "conversion_probability": analysis.deep_learning_scores.conversion_probability if analysis.deep_learning_scores else None,
                    "revenue_potential": analysis.deep_learning_scores.revenue_potential if analysis.deep_learning_scores else None,
                    "engagement_score": analysis.deep_learning_scores.engagement_score if analysis.deep_learning_scores else None
                } if analysis.deep_learning_scores else None
            },
            "behavioral_scores": {
                "current_score": analysis.behavioral_scores.current_score if analysis.behavioral_scores else None,
                "score_change": analysis.behavioral_scores.score_change if analysis.behavioral_scores else None,
                "behavioral_profile": {
                    "total_events": analysis.behavioral_scores.behavioral_profile.total_events if analysis.behavioral_scores else None,
                    "engagement_trend": analysis.behavioral_scores.behavioral_profile.engagement_trend if analysis.behavioral_scores else None,
                    "behavioral_stage": analysis.behavioral_scores.behavioral_profile.behavioral_stage if analysis.behavioral_scores else None
                } if analysis.behavioral_scores else None
            },
            "churn_analysis": {
                "churn_probability": analysis.churn_analysis.churn_probability if analysis.churn_analysis else None,
                "risk_level": analysis.churn_analysis.risk_level if analysis.churn_analysis else None,
                "time_to_churn_days": analysis.churn_analysis.time_to_churn_days if analysis.churn_analysis else None,
                "early_warning_signals": analysis.churn_analysis.early_warning_signals if analysis.churn_analysis else None
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in comprehensive lead analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/batch")
async def batch_analyze_leads(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Batch analysis of multiple leads"""
    try:
        logger.info(f"Running batch analysis for {len(request.leads_data)} leads")
        
        # Convert Pydantic models to dicts
        leads_data = [lead.dict() for lead in request.leads_data]
        
        # Run batch analysis
        analyses = await system.batch_analyze_leads(
            leads_data,
            batch_size=request.batch_size
        )
        
        # Filter by priority if specified
        if request.priority_filter:
            analyses = [
                analysis for analysis in analyses
                if analysis.priority_ranking == request.priority_filter
            ]
        
        # Convert to serializable format
        results = []
        for analysis in analyses:
            results.append({
                "lead_id": analysis.lead_id,
                "final_lead_score": analysis.final_lead_score,
                "confidence_level": analysis.confidence_level,
                "priority_ranking": analysis.priority_ranking,
                "top_recommendations": analysis.action_recommendations[:3],
                "analysis_timestamp": analysis.analysis_timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "total_analyzed": len(analyses),
            "results": results,
            "summary": {
                "avg_score": sum(r["final_lead_score"] for r in results) / len(results) if results else 0,
                "priority_distribution": {
                    "critical": len([r for r in results if r["priority_ranking"] == "critical"]),
                    "high": len([r for r in results if r["priority_ranking"] == "high"]),
                    "medium": len([r for r in results if r["priority_ranking"] == "medium"]),
                    "low": len([r for r in results if r["priority_ranking"] == "low"])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavioral/track-event")
async def track_behavioral_event(
    event: BehavioralEventInput,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Track a behavioral event"""
    try:
        logger.info(f"Tracking behavioral event for lead {event.lead_id}")
        
        # Convert event type string to enum
        try:
            behavior_type = BehaviorType(event.event_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid event type: {event.event_type}"
            )
        
        # Track the event
        tracked_event = await system.behavioral_engine.track_behavioral_event(
            lead_id=event.lead_id,
            event_type=behavior_type,
            event_data=event.event_data,
            session_id=event.session_id,
            source=event.source
        )
        
        return {
            "status": "success",
            "message": "Behavioral event tracked successfully",
            "event_id": f"{event.lead_id}_{tracked_event.timestamp.isoformat()}",
            "value_score": tracked_event.value_score,
            "engagement_score": tracked_event.engagement_score,
            "intent_score": tracked_event.intent_score
        }
        
    except Exception as e:
        logger.error(f"Error tracking behavioral event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/leads/top/{limit}")
async def get_top_leads(
    limit: int = 50,
    priority_filter: Optional[str] = None,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get top-scoring leads"""
    try:
        if limit > 200:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 200")
        
        top_leads = await system.get_top_leads(
            limit=limit,
            priority_filter=priority_filter
        )
        
        results = []
        for lead in top_leads:
            results.append({
                "lead_id": lead.lead_id,
                "final_lead_score": lead.final_lead_score,
                "confidence_level": lead.confidence_level,
                "priority_ranking": lead.priority_ranking,
                "top_recommendations": lead.action_recommendations[:3],
                "churn_risk": lead.churn_analysis.churn_probability if lead.churn_analysis else None,
                "analysis_timestamp": lead.analysis_timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "total_leads": len(results),
            "leads": results,
            "filters_applied": {
                "limit": limit,
                "priority_filter": priority_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting top leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/churn/analytics")
async def get_churn_analytics(
    time_period_days: int = 30,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get churn analytics"""
    try:
        if time_period_days > 365:
            raise HTTPException(status_code=400, detail="Time period cannot exceed 365 days")
        
        analytics = await system.churn_engine.get_churn_analytics(
            time_period_days=time_period_days
        )
        
        return {
            "status": "success",
            "time_period_days": time_period_days,
            "analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting churn analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/churn/at-risk-leads/{limit}")
async def get_at_risk_leads(
    limit: int = 50,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get leads at high risk of churning"""
    try:
        if limit > 200:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 200")
        
        at_risk_leads = await system.churn_engine.get_at_risk_leads(limit=limit)
        
        results = []
        for lead_id, current_score, churn_risk in at_risk_leads:
            results.append({
                "lead_id": lead_id,
                "current_score": current_score,
                "churn_risk_score": churn_risk,
                "risk_level": "high" if churn_risk > 0.8 else "critical" if churn_risk > 0.9 else "medium"
            })
        
        return {
            "status": "success",
            "total_at_risk": len(results),
            "leads": results,
            "summary": {
                "avg_churn_risk": sum(r["churn_risk_score"] for r in results) / len(results) if results else 0,
                "critical_count": len([r for r in results if r["churn_risk_score"] > 0.9]),
                "high_count": len([r for r in results if 0.8 < r["churn_risk_score"] <= 0.9])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting at-risk leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/retrain")
async def retrain_models(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Retrain ML models with new data"""
    try:
        # This would typically load data from the provided paths
        # For now, return a placeholder response
        
        background_tasks.add_task(
            _retrain_models_background,
            system,
            request.training_data_path,
            request.validation_data_path,
            request.model_types
        )
        
        return {
            "status": "success",
            "message": "Model retraining started in background",
            "training_data_path": request.training_data_path,
            "validation_data_path": request.validation_data_path,
            "models_to_retrain": request.model_types,
            "estimated_completion_time": "30-60 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/performance")
async def get_performance_analytics(
    time_period_hours: int = 24,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get system performance analytics"""
    try:
        if time_period_hours > 168:  # 1 week max
            raise HTTPException(status_code=400, detail="Time period cannot exceed 168 hours")
        
        performance_metrics = await system.get_system_performance_metrics()
        
        return {
            "status": "success",
            "time_period_hours": time_period_hours,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/behavioral/trending-leads/{limit}")
async def get_trending_leads(
    limit: int = 50,
    system: Any = Depends(get_enhanced_scoring_system)
) -> Dict[str, Any]:
    """Get leads with trending positive behavior"""
    try:
        if limit > 200:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 200")
        
        trending_leads = await system.behavioral_engine.get_trending_leads(limit=limit)
        
        results = []
        for lead_id, current_score, score_change in trending_leads:
            results.append({
                "lead_id": lead_id,
                "current_score": current_score,
                "score_change": score_change,
                "trend_strength": "strong" if score_change > 20 else "moderate" if score_change > 10 else "weak"
            })
        
        return {
            "status": "success",
            "total_trending": len(results),
            "leads": results,
            "summary": {
                "avg_score_change": sum(r["score_change"] for r in results) / len(results) if results else 0,
                "strong_trends": len([r for r in results if r["score_change"] > 20]),
                "moderate_trends": len([r for r in results if 10 < r["score_change"] <= 20])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting trending leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _retrain_models_background(
    system: Any,
    training_data_path: str,
    validation_data_path: Optional[str],
    model_types: List[str]
):
    """Background task for model retraining"""
    try:
        logger.info("Starting background model retraining...")
        
        # This would load actual data and retrain models
        # For now, simulate the process
        await asyncio.sleep(10)  # Simulate training time
        
        logger.info("Background model retraining completed")
        
    except Exception as e:
        logger.error(f"Error in background model retraining: {e}")

# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status": "error",
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }