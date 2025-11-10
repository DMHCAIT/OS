"""
Real-time Performance Monitoring System
Tracks API response times, ML model accuracy, and system health metrics
"""

import asyncio
import time
import logging
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import aioredis
import numpy as np
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    service: str

@dataclass
class SystemHealth:
    """System health snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    response_time_avg: float
    error_rate: float
    ml_accuracy: float
    call_success_rate: float

@dataclass
class APIMetrics:
    """API performance metrics"""
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime
    user_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class MLModelMetrics:
    """ML model performance metrics"""
    model_name: str
    prediction_accuracy: float
    inference_time: float
    confidence_score: float
    timestamp: datetime
    prediction_count: int
    error_rate: float

@dataclass
class CallMetrics:
    """Voice call performance metrics"""
    call_id: str
    duration: float
    success: bool
    conversion: bool
    sentiment_score: float
    timestamp: datetime
    quality_score: float
    disconnect_reason: Optional[str] = None

class PerformanceCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.api_metrics = deque(maxlen=5000)
        self.ml_metrics = deque(maxlen=2000)
        self.call_metrics = deque(maxlen=3000)
        self.system_health = deque(maxlen=1000)
        
        # Real-time aggregates
        self.current_stats = {
            'api_response_times': defaultdict(list),
            'ml_accuracies': defaultdict(list),
            'call_success_rates': [],
            'error_rates': defaultdict(list),
            'active_users': set(),
            'concurrent_calls': 0
        }
        
        # Performance thresholds
        self.thresholds = {
            'api_response_time_ms': 2000,  # 2 seconds
            'ml_accuracy_min': 0.80,      # 80% accuracy
            'error_rate_max': 0.05,       # 5% error rate
            'cpu_usage_max': 0.80,        # 80% CPU
            'memory_usage_max': 0.85,     # 85% memory
            'call_success_rate_min': 0.75  # 75% success rate
        }
        
        self._running = False
        self._collection_task = None

    async def start_collection(self):
        """Start the performance collection process"""
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_system_metrics())
        logger.info("Performance collection started")

    async def stop_collection(self):
        """Stop the performance collection process"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("Performance collection stopped")

    @asynccontextmanager
    async def track_api_call(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Context manager to track API call performance"""
        start_time = time.time()
        error_message = None
        status_code = 200
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
            status_code = 500
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            metric = APIMetrics(
                endpoint=endpoint,
                method=method,
                response_time=response_time,
                status_code=status_code,
                timestamp=datetime.now(),
                user_id=user_id,
                error_message=error_message
            )
            
            await self.record_api_metric(metric)

    async def record_api_metric(self, metric: APIMetrics):
        """Record API performance metric"""
        self.api_metrics.append(metric)
        
        # Update real-time stats
        endpoint_key = f"{metric.method}:{metric.endpoint}"
        self.current_stats['api_response_times'][endpoint_key].append(metric.response_time)
        
        if metric.user_id:
            self.current_stats['active_users'].add(metric.user_id)
            
        if metric.status_code >= 400:
            self.current_stats['error_rates'][endpoint_key].append(1)
        else:
            self.current_stats['error_rates'][endpoint_key].append(0)
            
        logger.debug(f"API metric recorded: {endpoint_key} - {metric.response_time:.2f}ms")

    async def record_ml_metric(self, metric: MLModelMetrics):
        """Record ML model performance metric"""
        self.ml_metrics.append(metric)
        
        # Update real-time stats
        self.current_stats['ml_accuracies'][metric.model_name].append(metric.prediction_accuracy)
        
        logger.debug(f"ML metric recorded: {metric.model_name} - {metric.prediction_accuracy:.3f} accuracy")

    async def record_call_metric(self, metric: CallMetrics):
        """Record voice call performance metric"""
        self.call_metrics.append(metric)
        
        # Update real-time stats
        self.current_stats['call_success_rates'].append(1 if metric.success else 0)
        
        if metric.success:
            self.current_stats['concurrent_calls'] += 1
        else:
            self.current_stats['concurrent_calls'] = max(0, self.current_stats['concurrent_calls'] - 1)
            
        logger.debug(f"Call metric recorded: {metric.call_id} - Success: {metric.success}")

    async def _collect_system_metrics(self):
        """Continuously collect system health metrics"""
        while self._running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Calculate derived metrics
                avg_response_time = self._calculate_avg_response_time()
                error_rate = self._calculate_error_rate()
                ml_accuracy = self._calculate_avg_ml_accuracy()
                call_success_rate = self._calculate_call_success_rate()
                
                # Create system health snapshot
                health = SystemHealth(
                    timestamp=datetime.now(),
                    cpu_usage=cpu_percent / 100.0,
                    memory_usage=memory.percent / 100.0,
                    disk_usage=disk.percent / 100.0,
                    active_connections=len(self.current_stats['active_users']),
                    response_time_avg=avg_response_time,
                    error_rate=error_rate,
                    ml_accuracy=ml_accuracy,
                    call_success_rate=call_success_rate
                )
                
                self.system_health.append(health)
                
                # Clear old stats (keep last 5 minutes)
                self._cleanup_old_stats()
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)

    def _calculate_avg_response_time(self) -> float:
        """Calculate average API response time"""
        all_times = []
        for times in self.current_stats['api_response_times'].values():
            all_times.extend(times[-50:])  # Last 50 calls per endpoint
        return np.mean(all_times) if all_times else 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        all_errors = []
        for errors in self.current_stats['error_rates'].values():
            all_errors.extend(errors[-50:])  # Last 50 calls per endpoint
        return np.mean(all_errors) if all_errors else 0.0

    def _calculate_avg_ml_accuracy(self) -> float:
        """Calculate average ML model accuracy"""
        all_accuracies = []
        for accuracies in self.current_stats['ml_accuracies'].values():
            all_accuracies.extend(accuracies[-20:])  # Last 20 predictions per model
        return np.mean(all_accuracies) if all_accuracies else 0.0

    def _calculate_call_success_rate(self) -> float:
        """Calculate call success rate"""
        recent_calls = self.current_stats['call_success_rates'][-100:]  # Last 100 calls
        return np.mean(recent_calls) if recent_calls else 0.0

    def _cleanup_old_stats(self):
        """Clean up old statistics to prevent memory bloat"""
        cutoff = datetime.now() - timedelta(minutes=5)
        
        # Keep only recent API metrics
        for endpoint in list(self.current_stats['api_response_times'].keys()):
            times = self.current_stats['api_response_times'][endpoint]
            self.current_stats['api_response_times'][endpoint] = times[-100:]
            
        for endpoint in list(self.current_stats['error_rates'].keys()):
            errors = self.current_stats['error_rates'][endpoint]
            self.current_stats['error_rates'][endpoint] = errors[-100:]

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        latest_health = self.system_health[-1] if self.system_health else None
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': asdict(latest_health) if latest_health else None,
            'api_metrics': {
                'avg_response_time': self._calculate_avg_response_time(),
                'error_rate': self._calculate_error_rate(),
                'active_users': len(self.current_stats['active_users']),
                'endpoints_performance': {
                    endpoint: {
                        'avg_response_time': np.mean(times[-20:]) if times else 0,
                        'error_rate': np.mean(errors[-20:]) if errors else 0
                    }
                    for endpoint, times in self.current_stats['api_response_times'].items()
                    for errors in [self.current_stats['error_rates'].get(endpoint, [0])]
                }
            },
            'ml_metrics': {
                'avg_accuracy': self._calculate_avg_ml_accuracy(),
                'models_performance': {
                    model: np.mean(accuracies[-10:]) if accuracies else 0
                    for model, accuracies in self.current_stats['ml_accuracies'].items()
                }
            },
            'call_metrics': {
                'success_rate': self._calculate_call_success_rate(),
                'concurrent_calls': self.current_stats['concurrent_calls']
            }
        }

    async def get_performance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics by time
        recent_health = [h for h in self.system_health if h.timestamp >= cutoff]
        recent_api = [m for m in self.api_metrics if m.timestamp >= cutoff]
        recent_ml = [m for m in self.ml_metrics if m.timestamp >= cutoff]
        recent_calls = [c for c in self.call_metrics if c.timestamp >= cutoff]
        
        return {
            'system_health_history': [asdict(h) for h in recent_health],
            'api_metrics_history': [asdict(m) for m in recent_api],
            'ml_metrics_history': [asdict(m) for m in recent_ml],
            'call_metrics_history': [asdict(c) for c in recent_calls]
        }

    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check if any performance thresholds are exceeded"""
        alerts = []
        
        # Check API response time
        avg_response_time = self._calculate_avg_response_time()
        if avg_response_time > self.thresholds['api_response_time_ms']:
            alerts.append({
                'type': 'api_performance',
                'severity': 'warning',
                'message': f"High API response time: {avg_response_time:.2f}ms",
                'threshold': self.thresholds['api_response_time_ms'],
                'current_value': avg_response_time
            })
        
        # Check ML accuracy
        avg_accuracy = self._calculate_avg_ml_accuracy()
        if avg_accuracy < self.thresholds['ml_accuracy_min']:
            alerts.append({
                'type': 'ml_performance',
                'severity': 'critical',
                'message': f"Low ML accuracy: {avg_accuracy:.3f}",
                'threshold': self.thresholds['ml_accuracy_min'],
                'current_value': avg_accuracy
            })
        
        # Check error rate
        error_rate = self._calculate_error_rate()
        if error_rate > self.thresholds['error_rate_max']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'warning',
                'message': f"High error rate: {error_rate:.3f}",
                'threshold': self.thresholds['error_rate_max'],
                'current_value': error_rate
            })
        
        # Check system resources
        if self.system_health:
            latest_health = self.system_health[-1]
            
            if latest_health.cpu_usage > self.thresholds['cpu_usage_max']:
                alerts.append({
                    'type': 'system_resource',
                    'severity': 'warning',
                    'message': f"High CPU usage: {latest_health.cpu_usage:.1%}",
                    'threshold': self.thresholds['cpu_usage_max'],
                    'current_value': latest_health.cpu_usage
                })
            
            if latest_health.memory_usage > self.thresholds['memory_usage_max']:
                alerts.append({
                    'type': 'system_resource',
                    'severity': 'critical',
                    'message': f"High memory usage: {latest_health.memory_usage:.1%}",
                    'threshold': self.thresholds['memory_usage_max'],
                    'current_value': latest_health.memory_usage
                })
        
        # Check call success rate
        call_success_rate = self._calculate_call_success_rate()
        if call_success_rate < self.thresholds['call_success_rate_min']:
            alerts.append({
                'type': 'call_performance',
                'severity': 'warning',
                'message': f"Low call success rate: {call_success_rate:.1%}",
                'threshold': self.thresholds['call_success_rate_min'],
                'current_value': call_success_rate
            })
        
        return alerts


# Global performance collector instance
performance_collector = PerformanceCollector()


# Middleware for automatic API tracking
class PerformanceMiddleware:
    """FastAPI middleware for automatic performance tracking"""
    
    def __init__(self, app, collector: PerformanceCollector):
        self.app = app
        self.collector = collector
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Extract request info
            method = scope["method"]
            path = scope["path"]
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    response_time = (time.time() - start_time) * 1000
                    
                    metric = APIMetrics(
                        endpoint=path,
                        method=method,
                        response_time=response_time,
                        status_code=status_code,
                        timestamp=datetime.now()
                    )
                    
                    await self.collector.record_api_metric(metric)
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)