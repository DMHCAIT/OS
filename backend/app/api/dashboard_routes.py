"""
Performance Dashboard API Endpoints
FastAPI routes for real-time dashboard data, performance metrics, and alert management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import logging
from pydantic import BaseModel

from ..monitoring.performance_monitor import (
    performance_collector, 
    APIMetrics, 
    MLModelMetrics, 
    CallMetrics,
    PerformanceMiddleware
)
from ..monitoring.business_intelligence import (
    business_intelligence,
    ConversionMetric,
    ROIMetric,
    SalesRepPerformance,
    ConversionStage
)
from ..monitoring.predictive_alerts import (
    predictive_alerts,
    AlertSeverity,
    AlertCategory
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Performance Dashboard"])

# Pydantic models for request/response
class AlertAcknowledgeRequest(BaseModel):
    alert_id: str
    user_id: str

class AlertResolveRequest(BaseModel):
    alert_id: str
    user_id: str
    resolution_note: Optional[str] = ""

class MetricSubmission(BaseModel):
    metric_name: str
    value: float
    source: str = "api"

class MLMetricSubmission(BaseModel):
    model_name: str
    accuracy: float
    inference_time: float
    confidence_score: float
    prediction_count: int = 1

class CallMetricSubmission(BaseModel):
    call_id: str
    duration: float
    success: bool
    conversion: bool
    sentiment_score: float
    quality_score: float
    disconnect_reason: Optional[str] = None

class ConversionTrackingRequest(BaseModel):
    lead_id: str
    stage: str
    source: str
    value: float
    sales_rep: str

class ROITrackingRequest(BaseModel):
    campaign_id: str
    cost: float
    revenue: float
    conversions: int
    leads: int
    channel: str

# Performance Monitoring Endpoints
@router.get("/performance/current")
async def get_current_performance():
    """Get current real-time performance metrics"""
    try:
        metrics = await performance_collector.get_current_metrics()
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        logger.error(f"Error getting current performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/history")
async def get_performance_history(hours: int = 24):
    """Get performance metrics history"""
    try:
        if hours > 168:  # Limit to 1 week
            hours = 168
        
        history = await performance_collector.get_performance_history(hours)
        return {
            "success": True,
            "data": history
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/metric")
async def submit_performance_metric(metric: MetricSubmission):
    """Submit a custom performance metric"""
    try:
        await predictive_alerts.process_performance_metric(
            metric.metric_name,
            metric.value,
            metric.source
        )
        return {
            "success": True,
            "message": "Metric recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/ml-metric")
async def submit_ml_metric(metric: MLMetricSubmission):
    """Submit ML model performance metric"""
    try:
        ml_metric = MLModelMetrics(
            model_name=metric.model_name,
            prediction_accuracy=metric.accuracy,
            inference_time=metric.inference_time,
            confidence_score=metric.confidence_score,
            timestamp=datetime.now(),
            prediction_count=metric.prediction_count,
            error_rate=0.0  # Could be calculated from failed predictions
        )
        
        await performance_collector.record_ml_metric(ml_metric)
        await predictive_alerts.process_ml_metric(metric.model_name, metric.accuracy)
        
        return {
            "success": True,
            "message": "ML metric recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting ML metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/call-metric")
async def submit_call_metric(metric: CallMetricSubmission):
    """Submit voice call performance metric"""
    try:
        call_metric = CallMetrics(
            call_id=metric.call_id,
            duration=metric.duration,
            success=metric.success,
            conversion=metric.conversion,
            sentiment_score=metric.sentiment_score,
            timestamp=datetime.now(),
            quality_score=metric.quality_score,
            disconnect_reason=metric.disconnect_reason
        )
        
        await performance_collector.record_call_metric(call_metric)
        
        return {
            "success": True,
            "message": "Call metric recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting call metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Business Intelligence Endpoints
@router.get("/business/conversion-funnel")
async def get_conversion_funnel(days: int = 30):
    """Get conversion funnel analytics"""
    try:
        funnel_data = await business_intelligence.calculate_conversion_funnel(days)
        return {
            "success": True,
            "data": funnel_data
        }
    except Exception as e:
        logger.error(f"Error getting conversion funnel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/business/roi-analysis")
async def get_roi_analysis(days: int = 30):
    """Get ROI analysis across campaigns and channels"""
    try:
        roi_data = await business_intelligence.calculate_roi_analysis(days)
        return {
            "success": True,
            "data": roi_data
        }
    except Exception as e:
        logger.error(f"Error getting ROI analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/business/pipeline-analysis")
async def get_pipeline_analysis():
    """Get sales pipeline bottleneck analysis"""
    try:
        pipeline_data = await business_intelligence.analyze_sales_pipeline()
        return {
            "success": True,
            "data": pipeline_data
        }
    except Exception as e:
        logger.error(f"Error getting pipeline analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/business/rep-rankings")
async def get_sales_rep_rankings(days: int = 30):
    """Get sales rep performance rankings"""
    try:
        rankings = await business_intelligence.calculate_sales_rep_rankings(days)
        return {
            "success": True,
            "data": rankings
        }
    except Exception as e:
        logger.error(f"Error getting rep rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/business/predictive-insights")
async def get_predictive_insights():
    """Get AI-powered predictive business insights"""
    try:
        insights = await business_intelligence.generate_predictive_insights()
        return {
            "success": True,
            "data": [insight.__dict__ for insight in insights]
        }
    except Exception as e:
        logger.error(f"Error getting predictive insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/business/comprehensive")
async def get_comprehensive_business_data():
    """Get all business intelligence data in one call"""
    try:
        data = await business_intelligence.get_comprehensive_dashboard_data()
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error getting comprehensive business data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/business/track-conversion")
async def track_conversion(conversion: ConversionTrackingRequest):
    """Track a conversion event"""
    try:
        # Convert string stage to enum
        stage = ConversionStage(conversion.stage.lower())
        
        metric = ConversionMetric(
            lead_id=conversion.lead_id,
            stage=stage,
            timestamp=datetime.now(),
            source=conversion.source,
            value=conversion.value,
            sales_rep=conversion.sales_rep
        )
        
        await business_intelligence.track_conversion(metric)
        
        return {
            "success": True,
            "message": "Conversion tracked successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {conversion.stage}")
    except Exception as e:
        logger.error(f"Error tracking conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/business/track-roi")
async def track_roi(roi: ROITrackingRequest):
    """Track ROI metrics"""
    try:
        roi_percentage = ((roi.revenue - roi.cost) / roi.cost * 100) if roi.cost > 0 else 0
        
        metric = ROIMetric(
            campaign_id=roi.campaign_id,
            cost=roi.cost,
            revenue=roi.revenue,
            conversions=roi.conversions,
            leads=roi.leads,
            timestamp=datetime.now(),
            channel=roi.channel,
            roi_percentage=roi_percentage
        )
        
        await business_intelligence.track_roi(metric)
        
        return {
            "success": True,
            "message": "ROI tracked successfully"
        }
    except Exception as e:
        logger.error(f"Error tracking ROI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Management Endpoints
@router.get("/alerts/active")
async def get_active_alerts():
    """Get all active alerts"""
    try:
        alerts = predictive_alerts.alert_manager.get_active_alerts()
        return {
            "success": True,
            "data": [alert.__dict__ for alert in alerts]
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/summary")
async def get_alert_summary():
    """Get alert summary statistics"""
    try:
        summary = predictive_alerts.alert_manager.get_alert_summary()
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/dashboard")
async def get_alert_dashboard_data():
    """Get comprehensive alert dashboard data"""
    try:
        data = await predictive_alerts.get_alert_dashboard_data()
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error getting alert dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/acknowledge")
async def acknowledge_alert(request: AlertAcknowledgeRequest):
    """Acknowledge an alert"""
    try:
        success = await predictive_alerts.acknowledge_alert(request.alert_id, request.user_id)
        if success:
            return {
                "success": True,
                "message": "Alert acknowledged successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/resolve")
async def resolve_alert(request: AlertResolveRequest):
    """Resolve an alert"""
    try:
        success = await predictive_alerts.resolve_alert(
            request.alert_id, 
            request.user_id, 
            request.resolution_note
        )
        if success:
            return {
                "success": True,
                "message": "Alert resolved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time Streaming Endpoints
@router.get("/stream/performance")
async def stream_performance_metrics():
    """Stream real-time performance metrics via Server-Sent Events"""
    async def event_generator():
        while True:
            try:
                # Get current metrics
                metrics = await performance_collector.get_current_metrics()
                
                # Format as Server-Sent Event
                event_data = json.dumps({
                    "type": "performance",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                })
                
                yield f"data: {event_data}\n\n"
                
                # Wait 5 seconds before next update
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in performance stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/stream/alerts")
async def stream_alert_updates():
    """Stream real-time alert updates via Server-Sent Events"""
    async def event_generator():
        last_check = datetime.now()
        
        while True:
            try:
                # Get active alerts
                alerts = predictive_alerts.alert_manager.get_active_alerts()
                
                # Filter for new alerts since last check
                new_alerts = [
                    alert for alert in alerts 
                    if alert.timestamp > last_check
                ]
                
                if new_alerts:
                    event_data = json.dumps({
                        "type": "new_alerts",
                        "data": [alert.__dict__ for alert in new_alerts],
                        "timestamp": datetime.now().isoformat()
                    })
                    yield f"data: {event_data}\n\n"
                
                # Send periodic heartbeat with alert summary
                summary = predictive_alerts.alert_manager.get_alert_summary()
                heartbeat_data = json.dumps({
                    "type": "alert_summary",
                    "data": summary,
                    "timestamp": datetime.now().isoformat()
                })
                yield f"data: {heartbeat_data}\n\n"
                
                last_check = datetime.now()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(10)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# System Control Endpoints
@router.post("/monitoring/start")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start the monitoring systems"""
    try:
        background_tasks.add_task(performance_collector.start_collection)
        background_tasks.add_task(predictive_alerts.start_monitoring)
        
        return {
            "success": True,
            "message": "Monitoring systems started"
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop the monitoring systems"""
    try:
        await performance_collector.stop_collection()
        await predictive_alerts.stop_monitoring()
        
        return {
            "success": True,
            "message": "Monitoring systems stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        return {
            "success": True,
            "data": {
                "performance_monitoring": performance_collector._running if hasattr(performance_collector, '_running') else False,
                "predictive_alerts": predictive_alerts.monitoring_enabled,
                "metrics_collected": len(performance_collector.metrics_buffer),
                "active_alerts": len(predictive_alerts.alert_manager.active_alerts),
                "last_update": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoints
@router.get("/health")
async def health_check():
    """Dashboard API health check"""
    return {
        "success": True,
        "service": "Performance Dashboard API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including all monitoring systems"""
    try:
        # Check performance collector
        performance_health = {
            "running": getattr(performance_collector, '_running', False),
            "metrics_count": len(performance_collector.metrics_buffer),
            "last_metric": (
                performance_collector.metrics_buffer[-1].timestamp.isoformat() 
                if performance_collector.metrics_buffer else None
            )
        }
        
        # Check predictive alerts
        alerts_health = {
            "monitoring_enabled": predictive_alerts.monitoring_enabled,
            "active_alerts": len(predictive_alerts.alert_manager.active_alerts),
            "metrics_tracked": len(predictive_alerts.anomaly_detector.metric_history)
        }
        
        # Check business intelligence
        bi_health = {
            "conversion_history": len(business_intelligence.conversion_history),
            "roi_history": len(business_intelligence.roi_history),
            "cache_valid": datetime.now() < business_intelligence._cache_expiry
        }
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "performance_collector": performance_health,
                "predictive_alerts": alerts_health,
                "business_intelligence": bi_health
            }
        }
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))