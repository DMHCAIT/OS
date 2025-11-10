"""
Predictive Alerts System
Early warning system for declining model performance, system issues, and anomaly detection
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import logging
import json
from scipy import stats
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertCategory(Enum):
    """Alert categories"""
    PERFORMANCE = "performance"
    ML_MODEL = "ml_model"
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    current_value: float
    expected_range: tuple
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    historical_data: List[float]

@dataclass
class PredictiveTrend:
    """Predictive trend analysis"""
    metric_name: str
    current_trend: str  # 'improving', 'stable', 'declining', 'critical'
    predicted_value_24h: float
    predicted_value_7d: float
    confidence: float
    trend_strength: float

@dataclass
class ModelPerformanceDrift:
    """ML model performance drift detection"""
    model_name: str
    accuracy_drift: float
    drift_rate: float
    significance_level: float
    recommendation: str
    retrain_suggested: bool

class AlertManager:
    """Manages alert lifecycle and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_channels = []
        self.escalation_rules = {}
        
        # Alert rate limiting
        self.alert_counts = defaultdict(int)
        self.rate_limit_window = timedelta(minutes=15)
        self.max_alerts_per_window = 5

    async def create_alert(self, 
                          severity: AlertSeverity, 
                          category: AlertCategory,
                          title: str,
                          message: str,
                          source: str,
                          data: Dict[str, Any]) -> Alert:
        """Create and process a new alert"""
        alert_id = f"{category.value}_{source}_{int(datetime.now().timestamp())}"
        
        # Check rate limiting
        if await self._is_rate_limited(category, source):
            logger.warning(f"Alert rate limited: {title}")
            return None
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            data=data
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Check escalation rules
        await self._check_escalation(alert)
        
        logger.warning(f"Alert created: {severity.value} - {title}")
        return alert

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
            return True
        return False

    async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            alert.data['resolution_note'] = resolution_note
            alert.data['resolved_by'] = user_id
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id} by {user_id}")
            return True
        return False

    async def _is_rate_limited(self, category: AlertCategory, source: str) -> bool:
        """Check if alerts are rate limited for this category/source"""
        key = f"{category.value}_{source}"
        current_time = datetime.now()
        
        # Clean old counts
        cutoff = current_time - self.rate_limit_window
        self.alert_counts = {
            k: v for k, v in self.alert_counts.items() 
            if k.endswith(str(int(cutoff.timestamp())))
        }
        
        # Check current count
        count_key = f"{key}_{int(current_time.timestamp() // 900)}"  # 15-minute windows
        self.alert_counts[count_key] += 1
        
        return self.alert_counts[count_key] > self.max_alerts_per_window

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    async def _check_escalation(self, alert: Alert):
        """Check and apply escalation rules"""
        # Count recent critical alerts
        recent_critical = [
            a for a in self.alert_history 
            if a.severity == AlertSeverity.CRITICAL 
            and a.timestamp >= datetime.now() - timedelta(minutes=30)
        ]
        
        # Escalate to emergency if too many critical alerts
        if len(recent_critical) >= 3 and alert.severity == AlertSeverity.CRITICAL:
            alert.severity = AlertSeverity.EMERGENCY
            alert.escalated = True
            logger.error(f"Alert escalated to EMERGENCY: {alert.title}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active = self.active_alerts.values()
        
        return {
            'total_active': len(active),
            'by_severity': {
                severity.value: len([a for a in active if a.severity == severity])
                for severity in AlertSeverity
            },
            'by_category': {
                category.value: len([a for a in active if a.category == category])
                for category in AlertCategory
            },
            'unacknowledged': len([a for a in active if not a.acknowledged]),
            'escalated': len([a for a in active if a.escalated])
        }


class AnomalyDetector:
    """Detects anomalies in metrics using statistical methods"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict] = {}

    async def add_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add a new metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # Update baseline statistics
        await self._update_baseline(metric_name)

    async def detect_anomaly(self, metric_name: str, value: float) -> AnomalyDetection:
        """Detect if a value is anomalous"""
        if metric_name not in self.baseline_stats:
            return AnomalyDetection(
                metric_name=metric_name,
                current_value=value,
                expected_range=(value, value),
                anomaly_score=0.0,
                is_anomaly=False,
                confidence=0.0,
                historical_data=[]
            )
        
        stats = self.baseline_stats[metric_name]
        mean = stats['mean']
        std = stats['std']
        
        # Calculate z-score
        z_score = abs(value - mean) / std if std > 0 else 0
        
        # Determine if anomaly (beyond 2 standard deviations)
        is_anomaly = z_score > 2.0
        anomaly_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
        confidence = min(z_score / 2.0, 1.0)
        
        # Expected range (mean ± 2 std)
        expected_range = (mean - 2 * std, mean + 2 * std)
        
        # Historical data
        historical_values = [v for _, v in self.metric_history[metric_name]]
        
        return AnomalyDetection(
            metric_name=metric_name,
            current_value=value,
            expected_range=expected_range,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            historical_data=historical_values[-20:]  # Last 20 values
        )

    async def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric"""
        if len(self.metric_history[metric_name]) < 10:
            return  # Need at least 10 data points
        
        values = [v for _, v in self.metric_history[metric_name]]
        
        self.baseline_stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }

    async def analyze_trend(self, metric_name: str, hours: int = 24) -> PredictiveTrend:
        """Analyze trend and predict future values"""
        if metric_name not in self.metric_history:
            return PredictiveTrend(
                metric_name=metric_name,
                current_trend='unknown',
                predicted_value_24h=0.0,
                predicted_value_7d=0.0,
                confidence=0.0,
                trend_strength=0.0
            )
        
        # Get recent data
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_data = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts >= cutoff
        ]
        
        if len(recent_data) < 5:
            return PredictiveTrend(
                metric_name=metric_name,
                current_trend='insufficient_data',
                predicted_value_24h=0.0,
                predicted_value_7d=0.0,
                confidence=0.0,
                trend_strength=0.0
            )
        
        # Extract values and calculate trend
        timestamps = [ts.timestamp() for ts, _ in recent_data]
        values = [val for _, val in recent_data]
        
        # Linear regression
        slope, intercept, r_value, p_value, _ = stats.linregress(
            range(len(values)), values
        )
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving' if slope > 0.05 else 'slight_improvement'
        else:
            trend = 'declining' if slope < -0.05 else 'slight_decline'
        
        # Predict future values
        current_time = len(values) - 1
        predicted_24h = slope * (current_time + 24) + intercept
        predicted_7d = slope * (current_time + 168) + intercept  # 7 days = 168 hours
        
        return PredictiveTrend(
            metric_name=metric_name,
            current_trend=trend,
            predicted_value_24h=predicted_24h,
            predicted_value_7d=predicted_7d,
            confidence=abs(r_value),  # Correlation coefficient as confidence
            trend_strength=abs(slope)
        )


class ModelDriftDetector:
    """Detects performance drift in ML models"""
    
    def __init__(self):
        self.model_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.baseline_accuracy: Dict[str, float] = {}

    async def add_model_performance(self, model_name: str, accuracy: float, timestamp: datetime = None):
        """Add model performance data point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.model_performance_history[model_name].append((timestamp, accuracy))
        
        # Set baseline from first few measurements
        if model_name not in self.baseline_accuracy:
            accuracies = [acc for _, acc in self.model_performance_history[model_name]]
            if len(accuracies) >= 10:
                self.baseline_accuracy[model_name] = np.mean(accuracies[:10])

    async def detect_drift(self, model_name: str) -> ModelPerformanceDrift:
        """Detect if model performance has drifted"""
        if model_name not in self.model_performance_history:
            return ModelPerformanceDrift(
                model_name=model_name,
                accuracy_drift=0.0,
                drift_rate=0.0,
                significance_level=0.0,
                recommendation="No data available",
                retrain_suggested=False
            )
        
        history = self.model_performance_history[model_name]
        if len(history) < 20:
            return ModelPerformanceDrift(
                model_name=model_name,
                accuracy_drift=0.0,
                drift_rate=0.0,
                significance_level=0.0,
                recommendation="Insufficient data for drift detection",
                retrain_suggested=False
            )
        
        # Get baseline and recent performance
        baseline = self.baseline_accuracy.get(model_name, 0.8)
        recent_accuracies = [acc for _, acc in list(history)[-10:]]  # Last 10 measurements
        recent_mean = np.mean(recent_accuracies)
        
        # Calculate drift
        accuracy_drift = baseline - recent_mean
        drift_percentage = (accuracy_drift / baseline) * 100 if baseline > 0 else 0
        
        # Calculate drift rate (change over time)
        if len(history) >= 50:
            early_accuracies = [acc for _, acc in list(history)[:10]]
            early_mean = np.mean(early_accuracies)
            time_span = (list(history)[-1][0] - list(history)[0][0]).total_seconds() / 86400  # days
            drift_rate = (early_mean - recent_mean) / time_span if time_span > 0 else 0
        else:
            drift_rate = 0
        
        # Statistical significance test
        if len(recent_accuracies) >= 5:
            baseline_data = [acc for _, acc in list(history)[:10]]
            t_stat, p_value = stats.ttest_ind(baseline_data, recent_accuracies)
            significance_level = 1 - p_value
        else:
            significance_level = 0
        
        # Generate recommendation
        if drift_percentage > 5:  # 5% performance drop
            if drift_percentage > 10:
                recommendation = "Critical performance degradation - immediate retraining required"
                retrain_suggested = True
            else:
                recommendation = "Moderate performance degradation - schedule retraining"
                retrain_suggested = True
        elif drift_rate > 0.01:  # Trending downward
            recommendation = "Performance trending down - monitor closely"
            retrain_suggested = False
        else:
            recommendation = "Model performance stable"
            retrain_suggested = False
        
        return ModelPerformanceDrift(
            model_name=model_name,
            accuracy_drift=accuracy_drift,
            drift_rate=drift_rate,
            significance_level=significance_level,
            recommendation=recommendation,
            retrain_suggested=retrain_suggested
        )


class PredictiveAlertSystem:
    """Main predictive alert system orchestrator"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = ModelDriftDetector()
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.check_interval = 60  # seconds
        self._monitoring_task = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'api_response_time_high': 2000,  # ms
            'error_rate_high': 0.05,         # 5%
            'ml_accuracy_low': 0.80,         # 80%
            'cpu_usage_high': 0.85,          # 85%
            'memory_usage_high': 0.90,       # 90%
            'conversion_rate_low': 0.10,     # 10%
            'anomaly_score_high': 0.8        # 80%
        }

    async def start_monitoring(self):
        """Start the predictive monitoring system"""
        self.monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Predictive alert system started")

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_enabled = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Predictive alert system stopped")

    async def process_performance_metric(self, metric_name: str, value: float, source: str = "system"):
        """Process a performance metric and check for anomalies"""
        # Add to anomaly detector
        await self.anomaly_detector.add_metric(metric_name, value)
        
        # Check for anomalies
        anomaly = await self.anomaly_detector.detect_anomaly(metric_name, value)
        
        if anomaly.is_anomaly and anomaly.confidence > 0.7:
            severity = AlertSeverity.CRITICAL if anomaly.anomaly_score > 0.9 else AlertSeverity.WARNING
            
            await self.alert_manager.create_alert(
                severity=severity,
                category=AlertCategory.PERFORMANCE,
                title=f"Anomaly Detected: {metric_name}",
                message=f"Metric '{metric_name}' value {value:.2f} is anomalous. Expected range: {anomaly.expected_range[0]:.2f} - {anomaly.expected_range[1]:.2f}",
                source=source,
                data={
                    'metric_name': metric_name,
                    'current_value': value,
                    'expected_range': anomaly.expected_range,
                    'anomaly_score': anomaly.anomaly_score,
                    'confidence': anomaly.confidence
                }
            )
        
        # Check specific thresholds
        await self._check_threshold_alerts(metric_name, value, source)

    async def process_ml_metric(self, model_name: str, accuracy: float, source: str = "ml_service"):
        """Process ML model performance metric"""
        # Add to drift detector
        await self.drift_detector.add_model_performance(model_name, accuracy)
        
        # Check for drift
        drift = await self.drift_detector.detect_drift(model_name)
        
        if drift.retrain_suggested:
            severity = AlertSeverity.CRITICAL if drift.accuracy_drift > 0.1 else AlertSeverity.WARNING
            
            await self.alert_manager.create_alert(
                severity=severity,
                category=AlertCategory.ML_MODEL,
                title=f"Model Drift Detected: {model_name}",
                message=f"Model '{model_name}' performance has degraded. {drift.recommendation}",
                source=source,
                data={
                    'model_name': model_name,
                    'accuracy_drift': drift.accuracy_drift,
                    'drift_rate': drift.drift_rate,
                    'recommendation': drift.recommendation,
                    'significance_level': drift.significance_level
                }
            )
        
        # Check accuracy threshold
        if accuracy < self.alert_thresholds['ml_accuracy_low']:
            await self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.ML_MODEL,
                title=f"Low Model Accuracy: {model_name}",
                message=f"Model '{model_name}' accuracy {accuracy:.3f} below threshold {self.alert_thresholds['ml_accuracy_low']:.3f}",
                source=source,
                data={'model_name': model_name, 'accuracy': accuracy}
            )

    async def _check_threshold_alerts(self, metric_name: str, value: float, source: str):
        """Check if metric exceeds predefined thresholds"""
        threshold_checks = {
            'api_response_time': ('api_response_time_high', AlertCategory.PERFORMANCE, 'High API Response Time'),
            'error_rate': ('error_rate_high', AlertCategory.SYSTEM, 'High Error Rate'),
            'cpu_usage': ('cpu_usage_high', AlertCategory.SYSTEM, 'High CPU Usage'),
            'memory_usage': ('memory_usage_high', AlertCategory.SYSTEM, 'High Memory Usage'),
            'conversion_rate': ('conversion_rate_low', AlertCategory.BUSINESS, 'Low Conversion Rate', True)  # True means lower is worse
        }
        
        for metric_pattern, (threshold_key, category, title, *lower_is_worse) in threshold_checks.items():
            if metric_pattern in metric_name and threshold_key in self.alert_thresholds:
                threshold = self.alert_thresholds[threshold_key]
                is_lower_worse = bool(lower_is_worse)
                
                exceeds = (value < threshold) if is_lower_worse else (value > threshold)
                
                if exceeds:
                    await self.alert_manager.create_alert(
                        severity=AlertSeverity.WARNING,
                        category=category,
                        title=f"{title}: {metric_name}",
                        message=f"Metric '{metric_name}' value {value:.3f} {'below' if is_lower_worse else 'above'} threshold {threshold:.3f}",
                        source=source,
                        data={'metric_name': metric_name, 'value': value, 'threshold': threshold}
                    )

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Generate predictive insights
                await self._generate_predictive_insights()
                
                # Check for system health patterns
                await self._analyze_system_patterns()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _generate_predictive_insights(self):
        """Generate predictive insights and alerts"""
        # Analyze trends for all tracked metrics
        for metric_name in self.anomaly_detector.metric_history.keys():
            trend = await self.anomaly_detector.analyze_trend(metric_name, hours=24)
            
            # Alert on negative trends
            if trend.current_trend in ['declining', 'critical'] and trend.confidence > 0.7:
                await self.alert_manager.create_alert(
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.PERFORMANCE,
                    title=f"Declining Trend: {metric_name}",
                    message=f"Metric '{metric_name}' is trending {trend.current_trend}. Predicted 24h value: {trend.predicted_value_24h:.2f}",
                    source="predictive_analysis",
                    data=asdict(trend)
                )

    async def _analyze_system_patterns(self):
        """Analyze system patterns for early warnings"""
        # This could include analyzing patterns across multiple metrics
        # For now, we'll implement basic correlation analysis
        
        # Example: Check if high error rate correlates with high response time
        error_rate_history = self.anomaly_detector.metric_history.get('error_rate', deque())
        response_time_history = self.anomaly_detector.metric_history.get('api_response_time', deque())
        
        if len(error_rate_history) >= 10 and len(response_time_history) >= 10:
            error_values = [v for _, v in list(error_rate_history)[-10:]]
            response_values = [v for _, v in list(response_time_history)[-10:]]
            
            # Calculate correlation
            if len(error_values) == len(response_values):
                correlation = np.corrcoef(error_values, response_values)[0, 1]
                
                if correlation > 0.7:  # Strong positive correlation
                    await self.alert_manager.create_alert(
                        severity=AlertSeverity.INFO,
                        category=AlertCategory.SYSTEM,
                        title="System Performance Pattern Detected",
                        message=f"High correlation ({correlation:.2f}) detected between error rate and response time",
                        source="pattern_analysis",
                        data={'correlation': correlation, 'pattern': 'error_response_correlation'}
                    )

    async def get_alert_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive alert dashboard data"""
        alert_summary = self.alert_manager.get_alert_summary()
        active_alerts = [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
        
        # Get recent trends for key metrics
        trend_analysis = {}
        for metric_name in list(self.anomaly_detector.metric_history.keys())[:5]:  # Top 5 metrics
            trend = await self.anomaly_detector.analyze_trend(metric_name)
            trend_analysis[metric_name] = asdict(trend)
        
        # Get model drift status
        model_drift_status = {}
        for model_name in self.drift_detector.model_performance_history.keys():
            drift = await self.drift_detector.detect_drift(model_name)
            model_drift_status[model_name] = asdict(drift)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'alert_summary': alert_summary,
            'active_alerts': active_alerts,
            'trend_analysis': trend_analysis,
            'model_drift_status': model_drift_status,
            'system_health': {
                'monitoring_enabled': self.monitoring_enabled,
                'metrics_tracked': len(self.anomaly_detector.metric_history),
                'models_monitored': len(self.drift_detector.model_performance_history),
                'alert_thresholds': self.alert_thresholds
            }
        }

    # Alert management methods
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        return await self.alert_manager.acknowledge_alert(alert_id, user_id)

    async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        return await self.alert_manager.resolve_alert(alert_id, user_id, resolution_note)

    def add_notification_channel(self, channel_func: Callable[[Alert], None]):
        """Add a notification channel"""
        self.alert_manager.notification_channels.append(channel_func)


# Global predictive alert system instance
predictive_alerts = PredictiveAlertSystem()


# Email notification channel
async def email_notification_channel(alert: Alert):
    """Send email notifications for critical alerts"""
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
        # This would integrate with your email service
        logger.info(f"Would send email for: {alert.title}")
        # Implementation would go here


# Slack notification channel  
async def slack_notification_channel(alert: Alert):
    """Send Slack notifications for alerts"""
    if alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
        # This would integrate with Slack API
        logger.info(f"Would send Slack notification for: {alert.title}")
        # Implementation would go here