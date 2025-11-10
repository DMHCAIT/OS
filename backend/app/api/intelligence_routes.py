"""
Predictive Business Intelligence API Routes
FastAPI routes for market trends, territory optimization, seasonal patterns, and competitive intelligence
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import json

from app.core.security import get_current_user
from app.core.database import get_database
from app.core.market_trend_analysis import market_trend_analysis, MarketTrendPrediction, EconomicIndicator, TrendSignal
from app.core.territory_optimization import territory_optimization, TerritoryRecommendation, OptimizationObjective, PerformanceMetrics
from app.core.seasonal_patterns import seasonal_patterns, SeasonalForecast, PatternType, SeasonalStrategy
from app.core.competitive_intelligence import competitive_intelligence, CompetitiveThreat, MarketPosition, StrategicResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/intelligence", tags=["Predictive Intelligence"])

# Pydantic models for request/response
class MarketAnalysisRequest(BaseModel):
    """Market trend analysis request"""
    timeframe_days: int = Field(default=90, ge=1, le=365)
    include_predictions: bool = True
    sectors: Optional[List[str]] = None
    economic_indicators: Optional[List[str]] = None

class TerritoryOptimizationRequest(BaseModel):
    """Territory optimization request"""
    objective: str = Field(default="revenue_maximization")
    include_reps: bool = True
    rebalance_existing: bool = False
    max_territories: Optional[int] = None
    geography_constraint: Optional[Dict[str, Any]] = None

class SeasonalAnalysisRequest(BaseModel):
    """Seasonal pattern analysis request"""
    historical_years: int = Field(default=3, ge=1, le=10)
    pattern_types: Optional[List[str]] = None
    forecast_months: int = Field(default=12, ge=1, le=24)
    include_strategies: bool = True

class CompetitiveIntelligenceRequest(BaseModel):
    """Competitive intelligence request"""
    competitor_tracking: bool = True
    threat_assessment: bool = True
    market_positioning: bool = True
    response_strategies: bool = True
    timeframe_days: int = Field(default=30, ge=1, le=180)

# Market Trend Analysis Routes
@router.post("/market-trends/analyze", response_model=Dict[str, Any])
async def analyze_market_trends(
    request: MarketAnalysisRequest,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Analyze market trends and predict sales impact
    """
    try:
        logger.info(f"Analyzing market trends for user {current_user.get('username', 'unknown')}")
        
        # Perform market trend analysis
        analysis_result = await market_trend_analysis.analyze_trends(
            timeframe_days=request.timeframe_days,
            sectors=request.sectors or ["technology", "healthcare", "finance"],
            include_predictions=request.include_predictions
        )
        
        # Store analysis in database
        analysis_doc = {
            "user_id": current_user.get("user_id"),
            "analysis_type": "market_trends",
            "result": analysis_result,
            "created_at": datetime.utcnow(),
            "parameters": request.dict()
        }
        
        await db.intelligence_analyses.insert_one(analysis_doc)
        
        return {
            "success": True,
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/market-trends/predictions", response_model=Dict[str, Any])
async def get_market_predictions(
    days_ahead: int = Query(default=30, ge=1, le=90),
    confidence_level: float = Query(default=0.8, ge=0.5, le=0.99),
    current_user=Depends(get_current_user)
):
    """
    Get market trend predictions
    """
    try:
        predictions = await market_trend_analysis.get_predictions(
            days_ahead=days_ahead,
            confidence_level=confidence_level
        )
        
        return {
            "success": True,
            "predictions": predictions,
            "forecast_period": f"{days_ahead} days",
            "confidence": confidence_level
        }
        
    except Exception as e:
        logger.error(f"Market prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Territory Optimization Routes
@router.post("/territory/optimize", response_model=Dict[str, Any])
async def optimize_territories(
    request: TerritoryOptimizationRequest,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Optimize sales territories using AI algorithms
    """
    try:
        logger.info(f"Optimizing territories for user {current_user.get('username', 'unknown')}")
        
        # Get current territory data
        territories_data = await db.territories.find({
            "user_id": current_user.get("user_id")
        }).to_list(length=None)
        
        # Perform territory optimization
        optimization_result = await territory_optimization.optimize_territories(
            current_territories=territories_data,
            objective=request.objective,
            include_reps=request.include_reps,
            max_territories=request.max_territories
        )
        
        # Store optimization results
        optimization_doc = {
            "user_id": current_user.get("user_id"),
            "optimization_type": "territory",
            "result": optimization_result,
            "created_at": datetime.utcnow(),
            "parameters": request.dict()
        }
        
        await db.intelligence_analyses.insert_one(optimization_doc)
        
        return {
            "success": True,
            "optimization": optimization_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Territory optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/territory/performance", response_model=Dict[str, Any])
async def get_territory_performance(
    territory_id: Optional[str] = Query(None),
    timeframe_days: int = Query(default=90, ge=1, le=365),
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Get territory performance analytics
    """
    try:
        performance_data = await territory_optimization.analyze_performance(
            territory_id=territory_id,
            timeframe_days=timeframe_days,
            user_id=current_user.get("user_id")
        )
        
        return {
            "success": True,
            "performance": performance_data,
            "timeframe": f"{timeframe_days} days"
        }
        
    except Exception as e:
        logger.error(f"Territory performance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

# Seasonal Pattern Analysis Routes
@router.post("/seasonal/analyze", response_model=Dict[str, Any])
async def analyze_seasonal_patterns(
    request: SeasonalAnalysisRequest,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Analyze seasonal sales patterns and generate forecasts
    """
    try:
        logger.info(f"Analyzing seasonal patterns for user {current_user.get('username', 'unknown')}")
        
        # Get historical sales data
        sales_data = await db.sales_history.find({
            "user_id": current_user.get("user_id"),
            "date": {"$gte": datetime.utcnow() - timedelta(days=365 * request.historical_years)}
        }).to_list(length=None)
        
        # Perform seasonal analysis
        seasonal_analysis = await seasonal_patterns.analyze_patterns(
            sales_data=sales_data,
            historical_years=request.historical_years,
            pattern_types=request.pattern_types or ["monthly", "quarterly", "holiday"],
            forecast_months=request.forecast_months
        )
        
        # Store analysis results
        analysis_doc = {
            "user_id": current_user.get("user_id"),
            "analysis_type": "seasonal_patterns",
            "result": seasonal_analysis,
            "created_at": datetime.utcnow(),
            "parameters": request.dict()
        }
        
        await db.intelligence_analyses.insert_one(analysis_doc)
        
        return {
            "success": True,
            "seasonal_analysis": seasonal_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Seasonal analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Seasonal analysis failed: {str(e)}")

@router.get("/seasonal/forecast", response_model=Dict[str, Any])
async def get_seasonal_forecast(
    months_ahead: int = Query(default=6, ge=1, le=24),
    include_strategies: bool = Query(default=True),
    current_user=Depends(get_current_user)
):
    """
    Get seasonal sales forecasts and strategies
    """
    try:
        forecast_data = await seasonal_patterns.generate_forecast(
            months_ahead=months_ahead,
            include_strategies=include_strategies,
            user_id=current_user.get("user_id")
        )
        
        return {
            "success": True,
            "forecast": forecast_data,
            "forecast_period": f"{months_ahead} months"
        }
        
    except Exception as e:
        logger.error(f"Seasonal forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# Competitive Intelligence Routes
@router.post("/competitive/analyze", response_model=Dict[str, Any])
async def analyze_competitive_landscape(
    request: CompetitiveIntelligenceRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Analyze competitive landscape and generate intelligence reports
    """
    try:
        logger.info(f"Analyzing competitive intelligence for user {current_user.get('username', 'unknown')}")
        
        # Start competitive intelligence analysis
        intelligence_result = await competitive_intelligence.analyze_competition(
            competitor_tracking=request.competitor_tracking,
            threat_assessment=request.threat_assessment,
            market_positioning=request.market_positioning,
            timeframe_days=request.timeframe_days
        )
        
        # Store intelligence results
        intelligence_doc = {
            "user_id": current_user.get("user_id"),
            "analysis_type": "competitive_intelligence",
            "result": intelligence_result,
            "created_at": datetime.utcnow(),
            "parameters": request.dict()
        }
        
        await db.intelligence_analyses.insert_one(intelligence_doc)
        
        # Schedule background monitoring
        if request.competitor_tracking:
            background_tasks.add_task(
                competitive_intelligence.monitor_competitors,
                user_id=current_user.get("user_id")
            )
        
        return {
            "success": True,
            "intelligence": intelligence_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Competitive intelligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intelligence analysis failed: {str(e)}")

@router.get("/competitive/threats", response_model=Dict[str, Any])
async def get_competitive_threats(
    threat_level: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    timeframe_days: int = Query(default=30, ge=1, le=180),
    current_user=Depends(get_current_user)
):
    """
    Get current competitive threats and recommendations
    """
    try:
        threats_data = await competitive_intelligence.get_threats(
            threat_level=threat_level,
            timeframe_days=timeframe_days,
            user_id=current_user.get("user_id")
        )
        
        return {
            "success": True,
            "threats": threats_data,
            "timeframe": f"{timeframe_days} days"
        }
        
    except Exception as e:
        logger.error(f"Competitive threats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Threat analysis failed: {str(e)}")

# Combined Intelligence Dashboard Route
@router.get("/dashboard/overview", response_model=Dict[str, Any])
async def get_intelligence_overview(
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Get comprehensive intelligence dashboard overview
    """
    try:
        # Get latest analyses
        recent_analyses = await db.intelligence_analyses.find({
            "user_id": current_user.get("user_id")
        }).sort("created_at", -1).limit(10).to_list(length=None)
        
        # Get quick insights from all intelligence systems
        market_insights = await market_trend_analysis.get_quick_insights()
        territory_insights = await territory_optimization.get_quick_insights()
        seasonal_insights = await seasonal_patterns.get_quick_insights()
        competitive_insights = await competitive_intelligence.get_quick_insights()
        
        dashboard_data = {
            "market_trends": market_insights,
            "territory_performance": territory_insights,
            "seasonal_patterns": seasonal_insights,
            "competitive_landscape": competitive_insights,
            "recent_analyses": recent_analyses,
            "summary": {
                "total_analyses": len(recent_analyses),
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        return {
            "success": True,
            "dashboard": dashboard_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def intelligence_health_check():
    """
    Health check for all intelligence services
    """
    try:
        health_status = {
            "market_trend_analysis": await market_trend_analysis.health_check(),
            "territory_optimization": await territory_optimization.health_check(),
            "seasonal_patterns": await seasonal_patterns.health_check(),
            "competitive_intelligence": await competitive_intelligence.health_check(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        all_healthy = all(service["status"] == "healthy" for service in health_status.values() if isinstance(service, dict))
        
        return {
            "success": True,
            "overall_health": "healthy" if all_healthy else "degraded",
            "services": health_status
        }
        
    except Exception as e:
        logger.error(f"Intelligence health check error: {str(e)}")
        return {
            "success": False,
            "overall_health": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }