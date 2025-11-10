"""
Market Trend Analysis System
Predict market conditions affecting sales through comprehensive trend analysis and forecasting
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Market trend directions"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class TrendConfidence(Enum):
    """Confidence levels for trend predictions"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class MarketSegment(Enum):
    """Market segments for analysis"""
    ENTERPRISE = "enterprise"
    MID_MARKET = "mid_market"
    SMB = "small_medium_business"
    STARTUP = "startup"
    GOVERNMENT = "government"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"

class EconomicIndicator(Enum):
    """Economic indicators affecting sales"""
    GDP_GROWTH = "gdp_growth"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    INFLATION_RATE = "inflation_rate"
    INTEREST_RATES = "interest_rates"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    BUSINESS_CONFIDENCE = "business_confidence"
    STOCK_MARKET_INDEX = "stock_market_index"
    CURRENCY_EXCHANGE = "currency_exchange"
    COMMODITY_PRICES = "commodity_prices"
    TECH_SPENDING_INDEX = "tech_spending_index"

@dataclass
class MarketTrend:
    """Market trend prediction result"""
    trend_id: str
    segment: MarketSegment
    direction: TrendDirection
    confidence: TrendConfidence
    predicted_impact: float  # -1.0 to 1.0 (negative impact to positive impact)
    prediction_horizon_days: int
    key_factors: List[str]
    supporting_data: Dict[str, Any]
    risk_factors: List[str]
    opportunities: List[str]
    created_at: datetime
    expires_at: datetime

@dataclass
class EconomicIndicatorData:
    """Economic indicator data point"""
    indicator: EconomicIndicator
    value: float
    previous_value: float
    change_percentage: float
    trend_direction: str
    last_updated: datetime
    source: str
    reliability_score: float

@dataclass
class MarketSentiment:
    """Market sentiment analysis result"""
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str
    news_volume: int
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    key_themes: List[str]
    sentiment_drivers: List[str]
    analysis_date: datetime

@dataclass
class SalesImpactPrediction:
    """Predicted impact of market trends on sales"""
    prediction_id: str
    market_segment: MarketSegment
    predicted_revenue_change: float  # Percentage change
    predicted_deal_volume_change: float
    predicted_cycle_length_change: float
    confidence_interval: Tuple[float, float]
    key_risk_factors: List[str]
    mitigation_strategies: List[str]
    prediction_date: datetime
    forecast_period_days: int

class MarketTrendAnalysisEngine:
    """Advanced market trend analysis and prediction system"""
    
    def __init__(self, config_path: str = "config/market_analysis_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Data storage
        self.economic_indicators: Dict[str, EconomicIndicatorData] = {}
        self.market_trends: Dict[str, MarketTrend] = {}
        self.sentiment_data: Dict[str, MarketSentiment] = {}
        self.sales_impact_predictions: Dict[str, SalesImpactPrediction] = {}
        
        # ML models
        self.trend_prediction_model = None
        self.impact_prediction_model = None
        self.scaler = StandardScaler()
        
        # External data sources
        self.data_sources = {
            "economic_api": "https://api.economicdata.com",
            "news_api": "https://api.newsdata.com",
            "market_api": "https://api.marketdata.com"
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("MarketTrendAnalysisEngine initialized")
    
    async def analyze_market_trends(
        self,
        segments: List[MarketSegment] = None,
        prediction_horizon_days: int = 30,
        include_economic_indicators: bool = True,
        include_sentiment_analysis: bool = True
    ) -> Dict[str, List[MarketTrend]]:
        """Comprehensive market trend analysis"""
        try:
            if segments is None:
                segments = list(MarketSegment)
            
            # Collect economic indicator data
            economic_data = {}
            if include_economic_indicators:
                economic_data = await self._collect_economic_indicators()
            
            # Collect market sentiment data
            sentiment_data = {}
            if include_sentiment_analysis:
                sentiment_data = await self._analyze_market_sentiment(segments)
            
            # Analyze trends for each segment
            trend_results = {}
            
            for segment in segments:
                segment_trends = await self._analyze_segment_trends(
                    segment,
                    economic_data,
                    sentiment_data.get(segment.value, {}),
                    prediction_horizon_days
                )
                trend_results[segment.value] = segment_trends
            
            # Store results
            for segment_name, trends in trend_results.items():
                for trend in trends:
                    self.market_trends[trend.trend_id] = trend
            
            logger.info(f"Analyzed market trends for {len(segments)} segments")
            return trend_results
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {}
    
    async def predict_sales_impact(
        self,
        market_trends: Dict[str, List[MarketTrend]],
        historical_sales_data: Dict[str, Any],
        forecast_period_days: int = 90
    ) -> Dict[str, SalesImpactPrediction]:
        """Predict the impact of market trends on sales performance"""
        try:
            impact_predictions = {}
            
            for segment_name, trends in market_trends.items():
                segment = MarketSegment(segment_name)
                
                # Prepare feature data
                feature_data = await self._prepare_impact_features(
                    trends, historical_sales_data.get(segment_name, {}), segment
                )
                
                # Make predictions
                prediction = await self._predict_segment_impact(
                    segment, feature_data, trends, forecast_period_days
                )
                
                if prediction:
                    impact_predictions[segment_name] = prediction
                    self.sales_impact_predictions[prediction.prediction_id] = prediction
            
            logger.info(f"Generated sales impact predictions for {len(impact_predictions)} segments")
            return impact_predictions
            
        except Exception as e:
            logger.error(f"Error predicting sales impact: {e}")
            return {}
    
    async def get_market_intelligence_dashboard(
        self,
        time_period_days: int = 30,
        segments: List[MarketSegment] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive market intelligence dashboard"""
        try:
            if segments is None:
                segments = list(MarketSegment)[:5]  # Limit for performance
            
            dashboard = {
                "generated_at": datetime.now().isoformat(),
                "period": f"last_{time_period_days}_days",
                "market_overview": {},
                "trend_summary": {},
                "economic_indicators": {},
                "sentiment_analysis": {},
                "sales_impact_forecast": {},
                "risk_assessment": {},
                "opportunities": {},
                "recommendations": []
            }
            
            # Market overview
            dashboard["market_overview"] = await self._get_market_overview(segments)
            
            # Trend summary
            dashboard["trend_summary"] = await self._get_trend_summary(segments, time_period_days)
            
            # Economic indicators
            dashboard["economic_indicators"] = await self._get_economic_summary()
            
            # Sentiment analysis
            dashboard["sentiment_analysis"] = await self._get_sentiment_summary(segments)
            
            # Sales impact forecast
            dashboard["sales_impact_forecast"] = await self._get_impact_forecast_summary()
            
            # Risk assessment
            dashboard["risk_assessment"] = await self._assess_market_risks(segments)
            
            # Opportunities
            dashboard["opportunities"] = await self._identify_market_opportunities(segments)
            
            # Strategic recommendations
            dashboard["recommendations"] = await self._generate_strategic_recommendations(
                dashboard["trend_summary"],
                dashboard["risk_assessment"],
                dashboard["opportunities"]
            )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating market intelligence dashboard: {e}")
            return {"error": str(e)}
    
    async def monitor_market_changes(
        self,
        alert_thresholds: Dict[str, float] = None,
        notification_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Monitor real-time market changes and generate alerts"""
        try:
            if alert_thresholds is None:
                alert_thresholds = {
                    "sentiment_change": 0.3,  # 30% change
                    "economic_indicator_change": 0.15,  # 15% change
                    "trend_confidence_drop": 0.2  # 20% drop in confidence
                }
            
            alerts = []
            
            # Monitor sentiment changes
            sentiment_alerts = await self._monitor_sentiment_changes(
                alert_thresholds["sentiment_change"]
            )
            alerts.extend(sentiment_alerts)
            
            # Monitor economic indicator changes
            economic_alerts = await self._monitor_economic_changes(
                alert_thresholds["economic_indicator_change"]
            )
            alerts.extend(economic_alerts)
            
            # Monitor trend confidence changes
            confidence_alerts = await self._monitor_confidence_changes(
                alert_thresholds["trend_confidence_drop"]
            )
            alerts.extend(confidence_alerts)
            
            # Send notifications if callback provided
            if notification_callback and alerts:
                for alert in alerts:
                    await notification_callback(alert)
            
            logger.info(f"Generated {len(alerts)} market change alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring market changes: {e}")
            return []
    
    async def get_trend_forecast(
        self,
        segment: MarketSegment,
        forecast_horizon_days: int = 90,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Get detailed trend forecast for a specific market segment"""
        try:
            # Get current trends for the segment
            current_trends = [
                trend for trend in self.market_trends.values()
                if trend.segment == segment and trend.expires_at > datetime.now()
            ]
            
            if not current_trends:
                # Generate new trends
                trends = await self._analyze_segment_trends(segment, {}, {}, forecast_horizon_days)
                current_trends = trends
            
            # Generate detailed forecast
            forecast = {
                "segment": segment.value,
                "forecast_period_days": forecast_horizon_days,
                "confidence_level": confidence_level,
                "current_trends": [asdict(trend) for trend in current_trends],
                "trend_trajectory": await self._calculate_trend_trajectory(
                    current_trends, forecast_horizon_days
                ),
                "key_milestones": await self._identify_trend_milestones(
                    current_trends, forecast_horizon_days
                ),
                "scenario_analysis": await self._generate_scenario_analysis(
                    segment, current_trends, forecast_horizon_days
                ),
                "actionable_insights": await self._generate_actionable_insights(
                    segment, current_trends
                ),
                "generated_at": datetime.now().isoformat()
            }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating trend forecast: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _collect_economic_indicators(self) -> Dict[str, EconomicIndicatorData]:
        """Collect current economic indicator data"""
        try:
            indicators = {}
            
            # In production, this would call real APIs
            # For demo, we'll simulate data
            for indicator in EconomicIndicator:
                # Simulate economic data
                current_value = self._simulate_economic_value(indicator)
                previous_value = current_value * (1 + np.random.normal(0, 0.02))
                change_pct = ((current_value - previous_value) / previous_value) * 100
                
                indicator_data = EconomicIndicatorData(
                    indicator=indicator,
                    value=current_value,
                    previous_value=previous_value,
                    change_percentage=change_pct,
                    trend_direction="up" if change_pct > 0 else "down",
                    last_updated=datetime.now(),
                    source="economic_data_api",
                    reliability_score=np.random.uniform(0.7, 0.95)
                )
                
                indicators[indicator.value] = indicator_data
                self.economic_indicators[indicator.value] = indicator_data
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error collecting economic indicators: {e}")
            return {}
    
    async def _analyze_market_sentiment(
        self,
        segments: List[MarketSegment]
    ) -> Dict[str, MarketSentiment]:
        """Analyze market sentiment for different segments"""
        try:
            sentiment_results = {}
            
            for segment in segments:
                # Simulate sentiment analysis
                # In production, this would analyze news, social media, etc.
                
                sentiment_score = np.random.uniform(-0.5, 0.8)  # Slightly positive bias
                positive_mentions = np.random.randint(50, 200)
                negative_mentions = np.random.randint(10, 80)
                neutral_mentions = np.random.randint(100, 300)
                
                sentiment = MarketSentiment(
                    sentiment_score=sentiment_score,
                    sentiment_label=self._get_sentiment_label(sentiment_score),
                    news_volume=positive_mentions + negative_mentions + neutral_mentions,
                    positive_mentions=positive_mentions,
                    negative_mentions=negative_mentions,
                    neutral_mentions=neutral_mentions,
                    key_themes=self._generate_key_themes(segment),
                    sentiment_drivers=self._generate_sentiment_drivers(sentiment_score),
                    analysis_date=datetime.now()
                )
                
                sentiment_results[segment.value] = sentiment
                self.sentiment_data[segment.value] = sentiment
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {}
    
    async def _analyze_segment_trends(
        self,
        segment: MarketSegment,
        economic_data: Dict[str, EconomicIndicatorData],
        sentiment_data: Dict[str, Any],
        prediction_horizon_days: int
    ) -> List[MarketTrend]:
        """Analyze trends for a specific market segment"""
        try:
            trends = []
            
            # Analyze economic impact on segment
            economic_impact = self._calculate_economic_impact(segment, economic_data)
            
            # Analyze sentiment impact
            sentiment_impact = sentiment_data.get("sentiment_score", 0.0) if sentiment_data else 0.0
            
            # Combine factors to determine trend direction
            combined_impact = (economic_impact * 0.7) + (sentiment_impact * 0.3)
            
            # Determine trend direction
            if combined_impact > 0.3:
                direction = TrendDirection.STRONG_BULLISH
            elif combined_impact > 0.1:
                direction = TrendDirection.BULLISH
            elif combined_impact > -0.1:
                direction = TrendDirection.NEUTRAL
            elif combined_impact > -0.3:
                direction = TrendDirection.BEARISH
            else:
                direction = TrendDirection.STRONG_BEARISH
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(economic_data, sentiment_data)
            
            # Create trend
            trend = MarketTrend(
                trend_id=f"trend_{segment.value}_{int(datetime.now().timestamp())}",
                segment=segment,
                direction=direction,
                confidence=confidence,
                predicted_impact=combined_impact,
                prediction_horizon_days=prediction_horizon_days,
                key_factors=self._identify_key_factors(economic_data, sentiment_data),
                supporting_data={
                    "economic_impact": economic_impact,
                    "sentiment_impact": sentiment_impact,
                    "data_quality_score": np.mean([
                        ind.reliability_score for ind in economic_data.values()
                    ]) if economic_data else 0.8
                },
                risk_factors=self._identify_risk_factors(segment, economic_data),
                opportunities=self._identify_opportunities(segment, direction, economic_data),
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=prediction_horizon_days)
            )
            
            trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing segment trends: {e}")
            return []
    
    async def _predict_segment_impact(
        self,
        segment: MarketSegment,
        feature_data: Dict[str, Any],
        trends: List[MarketTrend],
        forecast_period_days: int
    ) -> Optional[SalesImpactPrediction]:
        """Predict sales impact for a specific segment"""
        try:
            # Calculate predicted changes based on trends and historical data
            revenue_change = 0.0
            deal_volume_change = 0.0
            cycle_length_change = 0.0
            
            for trend in trends:
                impact = trend.predicted_impact
                
                # Revenue impact
                if trend.direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                    revenue_change += abs(impact) * 15  # Positive impact
                elif trend.direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                    revenue_change -= abs(impact) * 10  # Negative impact
                
                # Deal volume impact
                if trend.direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                    deal_volume_change += abs(impact) * 8
                elif trend.direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                    deal_volume_change -= abs(impact) * 6
                
                # Cycle length impact (bearish markets typically increase cycle length)
                if trend.direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                    cycle_length_change += abs(impact) * 12  # Longer cycles
                elif trend.direction in [TrendDirection.STRONG_BULLISH]:
                    cycle_length_change -= abs(impact) * 8  # Shorter cycles
            
            # Calculate confidence intervals
            confidence_range = 0.15  # 15% uncertainty
            revenue_ci = (
                revenue_change - abs(revenue_change) * confidence_range,
                revenue_change + abs(revenue_change) * confidence_range
            )
            
            # Identify risk factors and mitigation strategies
            risk_factors = []
            mitigation_strategies = []
            
            for trend in trends:
                risk_factors.extend(trend.risk_factors)
                
                if trend.direction == TrendDirection.BEARISH:
                    mitigation_strategies.append(f"Focus on value-based selling for {segment.value}")
                elif trend.direction == TrendDirection.STRONG_BEARISH:
                    mitigation_strategies.append(f"Implement defensive pricing strategies for {segment.value}")
            
            prediction = SalesImpactPrediction(
                prediction_id=f"impact_{segment.value}_{int(datetime.now().timestamp())}",
                market_segment=segment,
                predicted_revenue_change=revenue_change,
                predicted_deal_volume_change=deal_volume_change,
                predicted_cycle_length_change=cycle_length_change,
                confidence_interval=revenue_ci,
                key_risk_factors=list(set(risk_factors)),
                mitigation_strategies=list(set(mitigation_strategies)),
                prediction_date=datetime.now(),
                forecast_period_days=forecast_period_days
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting segment impact: {e}")
            return None
    
    def _calculate_economic_impact(
        self,
        segment: MarketSegment,
        economic_data: Dict[str, EconomicIndicatorData]
    ) -> float:
        """Calculate economic impact on a market segment"""
        impact = 0.0
        
        # Different segments are affected differently by economic indicators
        weights = self._get_segment_economic_weights(segment)
        
        for indicator_name, indicator_data in economic_data.items():
            if indicator_name in weights:
                # Positive change in most indicators is good (except unemployment, inflation)
                change = indicator_data.change_percentage / 100
                
                if indicator_name in ['unemployment_rate', 'inflation_rate']:
                    change = -change  # Invert for negative indicators
                
                impact += change * weights[indicator_name] * indicator_data.reliability_score
        
        return np.clip(impact, -1.0, 1.0)
    
    def _get_segment_economic_weights(self, segment: MarketSegment) -> Dict[str, float]:
        """Get economic indicator weights for different market segments"""
        weights = {
            MarketSegment.ENTERPRISE: {
                'gdp_growth': 0.3,
                'business_confidence': 0.25,
                'tech_spending_index': 0.2,
                'interest_rates': 0.15,
                'unemployment_rate': 0.1
            },
            MarketSegment.SMB: {
                'business_confidence': 0.3,
                'gdp_growth': 0.25,
                'unemployment_rate': 0.2,
                'interest_rates': 0.15,
                'inflation_rate': 0.1
            },
            MarketSegment.STARTUP: {
                'interest_rates': 0.4,
                'business_confidence': 0.25,
                'tech_spending_index': 0.2,
                'gdp_growth': 0.15
            }
        }
        
        return weights.get(segment, {
            'gdp_growth': 0.25,
            'business_confidence': 0.25,
            'interest_rates': 0.25,
            'unemployment_rate': 0.25
        })
    
    def _calculate_trend_confidence(
        self,
        economic_data: Dict[str, EconomicIndicatorData],
        sentiment_data: Dict[str, Any]
    ) -> TrendConfidence:
        """Calculate confidence level for trend prediction"""
        
        # Calculate data quality score
        data_quality = 0.5  # Base score
        
        if economic_data:
            avg_reliability = np.mean([ind.reliability_score for ind in economic_data.values()])
            data_quality += avg_reliability * 0.3
        
        if sentiment_data:
            news_volume = sentiment_data.get("news_volume", 0)
            if news_volume > 200:
                data_quality += 0.2
            elif news_volume > 100:
                data_quality += 0.1
        
        # Convert to confidence enum
        if data_quality >= 0.9:
            return TrendConfidence.VERY_HIGH
        elif data_quality >= 0.75:
            return TrendConfidence.HIGH
        elif data_quality >= 0.6:
            return TrendConfidence.MEDIUM
        elif data_quality >= 0.4:
            return TrendConfidence.LOW
        else:
            return TrendConfidence.VERY_LOW
    
    def _simulate_economic_value(self, indicator: EconomicIndicator) -> float:
        """Simulate economic indicator values for demo purposes"""
        base_values = {
            EconomicIndicator.GDP_GROWTH: 2.5,
            EconomicIndicator.UNEMPLOYMENT_RATE: 4.2,
            EconomicIndicator.INFLATION_RATE: 3.1,
            EconomicIndicator.INTEREST_RATES: 5.25,
            EconomicIndicator.CONSUMER_CONFIDENCE: 68.5,
            EconomicIndicator.BUSINESS_CONFIDENCE: 72.3,
            EconomicIndicator.STOCK_MARKET_INDEX: 4250.0,
            EconomicIndicator.CURRENCY_EXCHANGE: 1.08,
            EconomicIndicator.COMMODITY_PRICES: 85.2,
            EconomicIndicator.TECH_SPENDING_INDEX: 112.8
        }
        
        base = base_values.get(indicator, 100.0)
        # Add some random variation
        return base * (1 + np.random.normal(0, 0.05))
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score >= 0.6:
            return "Very Positive"
        elif sentiment_score >= 0.2:
            return "Positive"
        elif sentiment_score >= -0.2:
            return "Neutral"
        elif sentiment_score >= -0.6:
            return "Negative"
        else:
            return "Very Negative"
    
    def _generate_key_themes(self, segment: MarketSegment) -> List[str]:
        """Generate key themes for market segment sentiment"""
        themes = {
            MarketSegment.ENTERPRISE: ["digital transformation", "cost optimization", "security concerns"],
            MarketSegment.SMB: ["growth challenges", "cash flow", "technology adoption"],
            MarketSegment.TECHNOLOGY: ["AI adoption", "cloud migration", "innovation"],
            MarketSegment.HEALTHCARE: ["regulatory compliance", "patient outcomes", "cost reduction"]
        }
        
        return themes.get(segment, ["market conditions", "business growth", "technology trends"])
    
    def _initialize_models(self):
        """Initialize ML models for trend prediction"""
        # Initialize trend prediction model
        self.trend_prediction_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Initialize sales impact prediction model
        self.impact_prediction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        logger.info("ML models initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load market analysis configuration"""
        return {
            "update_frequency_hours": 6,
            "prediction_accuracy_threshold": 0.7,
            "confidence_threshold": 0.6,
            "max_trends_per_segment": 5,
            "economic_data_sources": ["fed_reserve", "bureau_stats", "market_apis"]
        }


# Global instance
market_trend_engine = MarketTrendAnalysisEngine()