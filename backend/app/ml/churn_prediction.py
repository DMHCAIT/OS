"""
Advanced Churn Prediction System
Predict which leads are likely to go cold using time-series analysis and behavioral patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
import json
import asyncio
import redis
import aioredis
from collections import defaultdict, deque
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ChurnRiskFactors:
    """Factors contributing to churn risk"""
    engagement_decline: float  # Decline in engagement over time
    interaction_frequency_drop: float  # Decrease in interaction frequency
    intent_stagnation: float  # Lack of progression in intent signals
    response_delay: float  # Delayed responses to outreach
    competitor_signals: float  # Signals of competitor engagement
    temporal_patterns: float  # Unusual temporal patterns
    value_alignment: float  # Misalignment with value propositions
    support_satisfaction: float  # Satisfaction with support interactions

@dataclass
class ChurnPrediction:
    """Churn prediction result"""
    lead_id: str
    churn_probability: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    time_to_churn_days: Optional[int]
    confidence_interval: Tuple[float, float]
    risk_factors: ChurnRiskFactors
    intervention_recommendations: List[str]
    early_warning_signals: List[str]
    prediction_timestamp: datetime
    next_check_date: datetime

@dataclass
class ChurnTrend:
    """Churn trend analysis"""
    lead_id: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    seasonal_pattern: Optional[Dict[str, float]]
    anomaly_score: float
    change_points: List[datetime]
    forecast_values: List[float]
    forecast_dates: List[datetime]

@dataclass
class InterventionStrategy:
    """Intervention strategy recommendation"""
    strategy_type: str
    priority: int  # 1-5, where 1 is highest priority
    description: str
    expected_effectiveness: float
    implementation_effort: str  # 'low', 'medium', 'high'
    timeline_days: int
    success_metrics: List[str]
    automated_actions: List[str]

class ChurnPredictionEngine:
    """Advanced churn prediction engine using time-series analysis and behavioral patterns"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        models_dir: str = "models/churn",
        lookback_days: int = 90,
        prediction_horizon_days: int = 30
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.lookback_days = lookback_days
        self.prediction_horizon_days = prediction_horizon_days
        
        # Scalers and models
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.churn_model = None
        
        # Thresholds
        self.risk_thresholds = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.7,
            'critical': 0.9
        }
        
        # Engagement patterns cache
        self.engagement_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.churn_predictions_cache: Dict[str, ChurnPrediction] = {}
        
        # Intervention strategies
        self.intervention_strategies = self._initialize_intervention_strategies()
        
        logger.info("ChurnPredictionEngine initialized")
    
    async def initialize(self):
        """Initialize the churn prediction engine"""
        try:
            # Connect to Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            await self.redis_client.ping()
            
            # Load existing models
            await self._load_models()
            
            logger.info("ChurnPredictionEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChurnPredictionEngine: {e}")
            raise
    
    async def predict_churn_risk(
        self,
        lead_id: str,
        force_refresh: bool = False
    ) -> ChurnPrediction:
        """Predict churn risk for a specific lead"""
        try:
            # Check cache if not forcing refresh
            if not force_refresh and lead_id in self.churn_predictions_cache:
                cached_prediction = self.churn_predictions_cache[lead_id]
                # Return cached if less than 4 hours old
                if (datetime.now() - cached_prediction.prediction_timestamp).total_seconds() < 4 * 3600:
                    return cached_prediction
            
            logger.info(f"Predicting churn risk for lead {lead_id}")
            
            # Get engagement history
            engagement_history = await self._get_engagement_history(lead_id)
            
            if len(engagement_history) < 5:  # Need minimum data points
                # Return low-confidence prediction for new leads
                return self._create_new_lead_prediction(lead_id)
            
            # Analyze engagement patterns
            trend_analysis = await self._analyze_engagement_trends(lead_id, engagement_history)
            
            # Calculate risk factors
            risk_factors = await self._calculate_risk_factors(lead_id, engagement_history, trend_analysis)
            
            # Calculate churn probability
            churn_probability = await self._calculate_churn_probability(risk_factors, trend_analysis)
            
            # Determine risk level
            risk_level = self._get_risk_level(churn_probability)
            
            # Estimate time to churn
            time_to_churn = await self._estimate_time_to_churn(trend_analysis, churn_probability)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(churn_probability, len(engagement_history))
            
            # Generate early warning signals
            warning_signals = await self._identify_warning_signals(risk_factors, trend_analysis)
            
            # Generate intervention recommendations
            recommendations = await self._generate_intervention_recommendations(
                risk_level, risk_factors, warning_signals
            )
            
            # Create prediction
            prediction = ChurnPrediction(
                lead_id=lead_id,
                churn_probability=churn_probability,
                risk_level=risk_level,
                time_to_churn_days=time_to_churn,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                intervention_recommendations=recommendations,
                early_warning_signals=warning_signals,
                prediction_timestamp=datetime.now(),
                next_check_date=self._calculate_next_check_date(risk_level)
            )
            
            # Cache prediction
            self.churn_predictions_cache[lead_id] = prediction
            
            # Store in Redis
            await self._store_churn_prediction(prediction)
            
            logger.info(f"Churn risk prediction completed for {lead_id}: {risk_level} ({churn_probability:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting churn risk for {lead_id}: {e}")
            raise
    
    async def batch_predict_churn(
        self,
        lead_ids: Optional[List[str]] = None,
        risk_threshold: float = 0.3
    ) -> List[ChurnPrediction]:
        """Batch predict churn for multiple leads"""
        try:
            # Get all active leads if not specified
            if lead_ids is None:
                lead_ids = await self._get_active_leads()
            
            logger.info(f"Running batch churn prediction for {len(lead_ids)} leads")
            
            # Process in batches to avoid overwhelming the system
            batch_size = 10
            all_predictions = []
            
            for i in range(0, len(lead_ids), batch_size):
                batch_leads = lead_ids[i:i + batch_size]
                
                # Create prediction tasks
                tasks = [
                    self.predict_churn_risk(lead_id)
                    for lead_id in batch_leads
                ]
                
                # Run batch
                batch_predictions = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter successful predictions
                valid_predictions = [
                    pred for pred in batch_predictions
                    if isinstance(pred, ChurnPrediction)
                ]
                
                all_predictions.extend(valid_predictions)
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            # Filter by risk threshold
            high_risk_predictions = [
                pred for pred in all_predictions
                if pred.churn_probability >= risk_threshold
            ]
            
            # Sort by risk level
            high_risk_predictions.sort(key=lambda x: x.churn_probability, reverse=True)
            
            logger.info(f"Found {len(high_risk_predictions)} leads at risk")
            
            return high_risk_predictions
            
        except Exception as e:
            logger.error(f"Error in batch churn prediction: {e}")
            return []
    
    async def get_churn_analytics(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive churn analytics"""
        try:
            # Get all recent predictions
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Get predictions from Redis
            prediction_keys = await self.redis_client.keys("churn_prediction:*")
            predictions = []
            
            for key in prediction_keys:
                prediction_data = await self.redis_client.get(key)
                if prediction_data:
                    data = json.loads(prediction_data)
                    pred_time = datetime.fromisoformat(data['prediction_timestamp'])
                    if pred_time >= cutoff_date:
                        predictions.append(data)
            
            if not predictions:
                return {"message": "No recent churn predictions available"}
            
            # Calculate analytics
            total_leads = len(predictions)
            
            # Risk distribution
            risk_distribution = defaultdict(int)
            for pred in predictions:
                risk_distribution[pred['risk_level']] += 1
            
            # Average churn probability by risk level
            avg_prob_by_risk = defaultdict(list)
            for pred in predictions:
                avg_prob_by_risk[pred['risk_level']].append(pred['churn_probability'])
            
            for risk_level in avg_prob_by_risk:
                avg_prob_by_risk[risk_level] = np.mean(avg_prob_by_risk[risk_level])
            
            # Top risk factors
            all_risk_factors = defaultdict(list)
            for pred in predictions:
                factors = pred['risk_factors']
                for factor, value in factors.items():
                    all_risk_factors[factor].append(value)
            
            top_risk_factors = {
                factor: np.mean(values)
                for factor, values in all_risk_factors.items()
            }
            top_risk_factors = dict(sorted(top_risk_factors.items(), key=lambda x: x[1], reverse=True))
            
            # Intervention recommendations frequency
            intervention_freq = defaultdict(int)
            for pred in predictions:
                for recommendation in pred.get('intervention_recommendations', []):
                    intervention_freq[recommendation] += 1
            
            # Time-based trends
            daily_predictions = defaultdict(list)
            for pred in predictions:
                date = datetime.fromisoformat(pred['prediction_timestamp']).date()
                daily_predictions[date].append(pred['churn_probability'])
            
            trend_data = []
            for date, probabilities in daily_predictions.items():
                trend_data.append({
                    'date': date.isoformat(),
                    'avg_churn_probability': np.mean(probabilities),
                    'high_risk_count': len([p for p in probabilities if p >= 0.7])
                })
            
            analytics = {
                "summary": {
                    "total_leads_analyzed": total_leads,
                    "high_risk_leads": risk_distribution.get('high', 0) + risk_distribution.get('critical', 0),
                    "medium_risk_leads": risk_distribution.get('medium', 0),
                    "low_risk_leads": risk_distribution.get('low', 0),
                    "avg_churn_probability": np.mean([p['churn_probability'] for p in predictions])
                },
                "risk_distribution": dict(risk_distribution),
                "avg_probability_by_risk": dict(avg_prob_by_risk),
                "top_risk_factors": dict(list(top_risk_factors.items())[:10]),
                "common_interventions": dict(list(intervention_freq.items())[:10]),
                "trend_data": sorted(trend_data, key=lambda x: x['date'])
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting churn analytics: {e}")
            return {"error": str(e)}
    
    async def _get_engagement_history(self, lead_id: str) -> List[Dict[str, Any]]:
        """Get engagement history for a lead"""
        try:
            # Check cache first
            if lead_id in self.engagement_cache:
                return list(self.engagement_cache[lead_id])
            
            # Get from Redis (behavioral events)
            events_key = f"lead_events:{lead_id}"
            events_data = await self.redis_client.lrange(events_key, 0, -1)
            
            engagement_points = []
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            
            for event_json in events_data:
                event_data = json.loads(event_json)
                event_time = datetime.fromisoformat(event_data['timestamp'])
                
                if event_time >= cutoff_date:
                    engagement_points.append({
                        'timestamp': event_time,
                        'engagement_score': event_data.get('engagement_score', 0),
                        'value_score': event_data.get('value_score', 0),
                        'intent_score': event_data.get('intent_score', 0),
                        'event_type': event_data.get('event_type', 'unknown')
                    })
            
            # Sort by timestamp
            engagement_points.sort(key=lambda x: x['timestamp'])
            
            # Cache for future use
            self.engagement_cache[lead_id] = deque(engagement_points, maxlen=1000)
            
            return engagement_points
            
        except Exception as e:
            logger.error(f"Error getting engagement history for {lead_id}: {e}")
            return []
    
    async def _analyze_engagement_trends(
        self,
        lead_id: str,
        engagement_history: List[Dict[str, Any]]
    ) -> ChurnTrend:
        """Analyze engagement trends and patterns"""
        try:
            if len(engagement_history) < 5:
                return ChurnTrend(
                    lead_id=lead_id,
                    trend_direction='stable',
                    trend_strength=0.0,
                    seasonal_pattern=None,
                    anomaly_score=0.0,
                    change_points=[],
                    forecast_values=[],
                    forecast_dates=[]
                )
            
            # Convert to time series
            timestamps = [point['timestamp'] for point in engagement_history]
            engagement_scores = [point['engagement_score'] for point in engagement_history]
            
            # Create daily aggregates
            df = pd.DataFrame({
                'timestamp': timestamps,
                'engagement': engagement_scores
            })
            df['date'] = df['timestamp'].dt.date
            daily_engagement = df.groupby('date')['engagement'].mean()
            
            # Trend analysis
            if len(daily_engagement) >= 3:
                # Calculate trend direction and strength
                x = np.arange(len(daily_engagement))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_engagement.values)
                
                trend_strength = abs(r_value)
                if slope > 0.01:
                    trend_direction = 'increasing'
                elif slope < -0.01:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
                trend_strength = 0.0
                slope = 0.0
            
            # Anomaly detection
            if len(engagement_scores) >= 10:
                engagement_array = np.array(engagement_scores).reshape(-1, 1)
                anomaly_scores = self.anomaly_detector.fit_predict(engagement_array)
                anomaly_score = len(anomaly_scores[anomaly_scores == -1]) / len(anomaly_scores)
            else:
                anomaly_score = 0.0
            
            # Change point detection (simplified)
            change_points = []
            if len(daily_engagement) >= 7:
                # Look for significant changes in rolling mean
                rolling_mean = daily_engagement.rolling(window=3).mean()
                rolling_std = daily_engagement.rolling(window=3).std()
                
                for i in range(3, len(rolling_mean)):
                    if abs(rolling_mean.iloc[i] - rolling_mean.iloc[i-1]) > 2 * rolling_std.iloc[i]:
                        change_points.append(daily_engagement.index[i])
            
            # Simple forecasting
            forecast_values = []
            forecast_dates = []
            if len(daily_engagement) >= 5:
                # Linear extrapolation for next 7 days
                last_date = daily_engagement.index[-1]
                last_value = daily_engagement.iloc[-1]
                
                for i in range(1, 8):
                    future_date = last_date + timedelta(days=i)
                    future_value = last_value + (slope * i)
                    forecast_dates.append(future_date)
                    forecast_values.append(max(0, future_value))  # Ensure non-negative
            
            return ChurnTrend(
                lead_id=lead_id,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_pattern=None,  # Could be enhanced with seasonal decomposition
                anomaly_score=anomaly_score,
                change_points=change_points,
                forecast_values=forecast_values,
                forecast_dates=forecast_dates
            )
            
        except Exception as e:
            logger.error(f"Error analyzing engagement trends: {e}")
            return ChurnTrend(
                lead_id=lead_id,
                trend_direction='stable',
                trend_strength=0.0,
                seasonal_pattern=None,
                anomaly_score=0.0,
                change_points=[],
                forecast_values=[],
                forecast_dates=[]
            )
    
    async def _calculate_risk_factors(
        self,
        lead_id: str,
        engagement_history: List[Dict[str, Any]],
        trend_analysis: ChurnTrend
    ) -> ChurnRiskFactors:
        """Calculate individual risk factors"""
        try:
            # Engagement decline factor
            if trend_analysis.trend_direction == 'decreasing':
                engagement_decline = trend_analysis.trend_strength
            else:
                engagement_decline = 0.0
            
            # Interaction frequency drop
            now = datetime.now()
            recent_interactions = [
                point for point in engagement_history
                if (now - point['timestamp']).days <= 7
            ]
            older_interactions = [
                point for point in engagement_history
                if 7 < (now - point['timestamp']).days <= 14
            ]
            
            if older_interactions:
                recent_freq = len(recent_interactions) / 7
                older_freq = len(older_interactions) / 7
                frequency_drop = max(0, (older_freq - recent_freq) / max(0.1, older_freq))
            else:
                frequency_drop = 0.0 if recent_interactions else 1.0
            
            # Intent stagnation
            intent_scores = [point['intent_score'] for point in engagement_history[-10:]]
            if len(intent_scores) >= 5:
                intent_variance = np.var(intent_scores)
                intent_stagnation = 1.0 - min(1.0, intent_variance * 5)  # Low variance = high stagnation
            else:
                intent_stagnation = 0.5
            
            # Response delay (placeholder - would need actual response data)
            response_delay = 0.3  # Default moderate risk
            
            # Competitor signals (placeholder)
            competitor_signals = 0.1  # Default low risk
            
            # Temporal patterns (based on anomaly score)
            temporal_patterns = trend_analysis.anomaly_score
            
            # Value alignment (based on value scores)
            value_scores = [point['value_score'] for point in engagement_history[-10:]]
            if value_scores:
                value_alignment = 1.0 - (np.mean(value_scores) / 100.0)  # Invert: low value = high risk
            else:
                value_alignment = 0.5
            
            # Support satisfaction (placeholder)
            support_satisfaction = 0.2  # Default low-medium risk
            
            return ChurnRiskFactors(
                engagement_decline=engagement_decline,
                interaction_frequency_drop=frequency_drop,
                intent_stagnation=intent_stagnation,
                response_delay=response_delay,
                competitor_signals=competitor_signals,
                temporal_patterns=temporal_patterns,
                value_alignment=value_alignment,
                support_satisfaction=support_satisfaction
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk factors: {e}")
            return ChurnRiskFactors(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)  # Default moderate risk
    
    async def _calculate_churn_probability(
        self,
        risk_factors: ChurnRiskFactors,
        trend_analysis: ChurnTrend
    ) -> float:
        """Calculate overall churn probability"""
        # Weighted combination of risk factors
        weights = {
            'engagement_decline': 0.25,
            'interaction_frequency_drop': 0.20,
            'intent_stagnation': 0.15,
            'response_delay': 0.10,
            'competitor_signals': 0.08,
            'temporal_patterns': 0.07,
            'value_alignment': 0.10,
            'support_satisfaction': 0.05
        }
        
        # Calculate base probability
        base_probability = (
            risk_factors.engagement_decline * weights['engagement_decline'] +
            risk_factors.interaction_frequency_drop * weights['interaction_frequency_drop'] +
            risk_factors.intent_stagnation * weights['intent_stagnation'] +
            risk_factors.response_delay * weights['response_delay'] +
            risk_factors.competitor_signals * weights['competitor_signals'] +
            risk_factors.temporal_patterns * weights['temporal_patterns'] +
            risk_factors.value_alignment * weights['value_alignment'] +
            risk_factors.support_satisfaction * weights['support_satisfaction']
        )
        
        # Adjust based on trend strength
        if trend_analysis.trend_direction == 'decreasing':
            base_probability *= (1 + trend_analysis.trend_strength * 0.3)
        elif trend_analysis.trend_direction == 'increasing':
            base_probability *= (1 - trend_analysis.trend_strength * 0.2)
        
        # Cap at 1.0
        return min(1.0, base_probability)
    
    def _get_risk_level(self, churn_probability: float) -> str:
        """Convert churn probability to risk level"""
        if churn_probability >= self.risk_thresholds['critical']:
            return 'critical'
        elif churn_probability >= self.risk_thresholds['high']:
            return 'high'
        elif churn_probability >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    async def _estimate_time_to_churn(
        self,
        trend_analysis: ChurnTrend,
        churn_probability: float
    ) -> Optional[int]:
        """Estimate time to churn in days"""
        if churn_probability < 0.3:
            return None  # Low risk, no immediate concern
        
        # Simple estimation based on trend
        if trend_analysis.trend_direction == 'decreasing' and trend_analysis.trend_strength > 0:
            # Estimate based on trend strength
            if churn_probability >= 0.8:
                return 7  # Critical - very soon
            elif churn_probability >= 0.6:
                return 14  # High - within 2 weeks
            elif churn_probability >= 0.4:
                return 30  # Medium - within a month
            else:
                return 60  # Lower risk - within 2 months
        
        # Default estimates for stable/increasing trends
        if churn_probability >= 0.7:
            return 21  # 3 weeks
        else:
            return 45  # 6+ weeks
    
    def _calculate_confidence_interval(
        self,
        churn_probability: float,
        data_points: int
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simple confidence interval based on data availability
        confidence = min(0.95, 0.5 + (data_points / 100))  # More data = higher confidence
        margin = (1 - confidence) * churn_probability
        
        lower_bound = max(0.0, churn_probability - margin)
        upper_bound = min(1.0, churn_probability + margin)
        
        return (lower_bound, upper_bound)
    
    async def _identify_warning_signals(
        self,
        risk_factors: ChurnRiskFactors,
        trend_analysis: ChurnTrend
    ) -> List[str]:
        """Identify early warning signals"""
        warnings = []
        
        if risk_factors.engagement_decline > 0.5:
            warnings.append("Significant engagement decline detected")
        
        if risk_factors.interaction_frequency_drop > 0.6:
            warnings.append("Sharp drop in interaction frequency")
        
        if risk_factors.intent_stagnation > 0.7:
            warnings.append("Intent signals showing stagnation")
        
        if trend_analysis.trend_direction == 'decreasing' and trend_analysis.trend_strength > 0.6:
            warnings.append("Strong negative engagement trend")
        
        if trend_analysis.anomaly_score > 0.2:
            warnings.append("Unusual behavioral patterns detected")
        
        if len(trend_analysis.change_points) >= 2:
            warnings.append("Multiple behavioral change points identified")
        
        return warnings
    
    async def _generate_intervention_recommendations(
        self,
        risk_level: str,
        risk_factors: ChurnRiskFactors,
        warning_signals: List[str]
    ) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []
        
        # Base recommendations by risk level
        if risk_level == 'critical':
            recommendations.append("Immediate personal outreach required")
            recommendations.append("Schedule emergency check-in call")
            recommendations.append("Escalate to senior account manager")
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Provide additional value demonstrations")
            recommendations.append("Offer customized solution presentation")
        
        if risk_level in ['medium', 'high', 'critical']:
            recommendations.append("Increase touchpoint frequency")
            recommendations.append("Share relevant success stories")
        
        # Specific recommendations based on risk factors
        if risk_factors.engagement_decline > 0.5:
            recommendations.append("Re-engage with personalized content")
        
        if risk_factors.interaction_frequency_drop > 0.6:
            recommendations.append("Proactive outreach to understand challenges")
        
        if risk_factors.intent_stagnation > 0.7:
            recommendations.append("Provide next-step guidance and clear path forward")
        
        if risk_factors.value_alignment > 0.6:
            recommendations.append("Reassess value proposition alignment")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_next_check_date(self, risk_level: str) -> datetime:
        """Calculate when to next check this lead"""
        now = datetime.now()
        
        if risk_level == 'critical':
            return now + timedelta(hours=12)  # 12 hours
        elif risk_level == 'high':
            return now + timedelta(days=1)    # 1 day
        elif risk_level == 'medium':
            return now + timedelta(days=3)    # 3 days
        else:
            return now + timedelta(days=7)    # 1 week
    
    def _create_new_lead_prediction(self, lead_id: str) -> ChurnPrediction:
        """Create prediction for new leads with minimal data"""
        return ChurnPrediction(
            lead_id=lead_id,
            churn_probability=0.3,  # Default moderate risk for new leads
            risk_level='medium',
            time_to_churn_days=None,
            confidence_interval=(0.1, 0.5),
            risk_factors=ChurnRiskFactors(0.0, 0.0, 0.5, 0.3, 0.1, 0.0, 0.3, 0.2),
            intervention_recommendations=["Monitor initial engagement patterns", "Establish regular touchpoints"],
            early_warning_signals=[],
            prediction_timestamp=datetime.now(),
            next_check_date=datetime.now() + timedelta(days=7)
        )
    
    async def _get_active_leads(self) -> List[str]:
        """Get list of active leads"""
        try:
            # Get all leads with recent activity
            pattern = "lead_events:*"
            keys = await self.redis_client.keys(pattern)
            
            active_leads = []
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for key in keys:
                lead_id = key.split(":")[1]
                
                # Check if lead has recent activity
                recent_events = await self.redis_client.lrange(key, 0, 10)
                if recent_events:
                    latest_event_data = json.loads(recent_events[0])
                    latest_time = datetime.fromisoformat(latest_event_data['timestamp'])
                    
                    if latest_time >= cutoff_date:
                        active_leads.append(lead_id)
            
            return active_leads
            
        except Exception as e:
            logger.error(f"Error getting active leads: {e}")
            return []
    
    async def _store_churn_prediction(self, prediction: ChurnPrediction):
        """Store churn prediction in Redis"""
        try:
            key = f"churn_prediction:{prediction.lead_id}"
            prediction_dict = asdict(prediction)
            
            # Convert datetime objects to strings
            prediction_dict['prediction_timestamp'] = prediction.prediction_timestamp.isoformat()
            prediction_dict['next_check_date'] = prediction.next_check_date.isoformat()
            
            await self.redis_client.setex(
                key,
                30 * 24 * 3600,  # 30 days expiration
                json.dumps(prediction_dict)
            )
            
        except Exception as e:
            logger.error(f"Error storing churn prediction: {e}")
    
    async def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            model_path = self.models_dir / "churn_model.joblib"
            scaler_path = self.models_dir / "churn_scaler.joblib"
            
            if model_path.exists():
                self.churn_model = joblib.load(model_path)
                logger.info("Loaded churn model")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _initialize_intervention_strategies(self) -> Dict[str, List[InterventionStrategy]]:
        """Initialize intervention strategies by risk level"""
        strategies = {
            'low': [
                InterventionStrategy(
                    strategy_type="nurture_campaign",
                    priority=3,
                    description="Continue nurture campaign with valuable content",
                    expected_effectiveness=0.7,
                    implementation_effort="low",
                    timeline_days=7,
                    success_metrics=["email_opens", "content_engagement"],
                    automated_actions=["send_nurture_email", "track_engagement"]
                )
            ],
            'medium': [
                InterventionStrategy(
                    strategy_type="personal_outreach",
                    priority=2,
                    description="Personal outreach to understand current needs",
                    expected_effectiveness=0.8,
                    implementation_effort="medium",
                    timeline_days=3,
                    success_metrics=["response_rate", "meeting_scheduled"],
                    automated_actions=["schedule_follow_up", "log_interaction"]
                )
            ],
            'high': [
                InterventionStrategy(
                    strategy_type="value_demonstration",
                    priority=1,
                    description="Demonstrate additional value through demo or case study",
                    expected_effectiveness=0.85,
                    implementation_effort="high",
                    timeline_days=1,
                    success_metrics=["demo_attendance", "proposal_request"],
                    automated_actions=["book_demo", "prepare_case_study"]
                )
            ],
            'critical': [
                InterventionStrategy(
                    strategy_type="executive_intervention",
                    priority=1,
                    description="Executive-level intervention with special offer",
                    expected_effectiveness=0.9,
                    implementation_effort="high",
                    timeline_days=1,
                    success_metrics=["executive_meeting", "contract_discussion"],
                    automated_actions=["escalate_to_executive", "prepare_special_offer"]
                )
            ]
        }
        
        return strategies


# Global instance
churn_prediction_engine = ChurnPredictionEngine()