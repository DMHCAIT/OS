"""
Seasonal Pattern Recognition System
Automatic adjustment for seasonal sales patterns with historical analysis and forecasting
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SeasonalPeriod(Enum):
    """Seasonal period types"""
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    HOLIDAY = "holiday"
    FISCAL_YEAR = "fiscal_year"
    INDUSTRY_SPECIFIC = "industry_specific"

class PatternType(Enum):
    """Types of seasonal patterns"""
    CYCLICAL = "cyclical"
    TRENDING = "trending"
    VOLATILE = "volatile"
    STABLE = "stable"
    DECLINING = "declining"
    GROWTH = "growth"

class SeasonalityStrength(Enum):
    """Strength of seasonal patterns"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEGLIGIBLE = "negligible"

class AdjustmentType(Enum):
    """Types of seasonal adjustments"""
    QUOTA_ADJUSTMENT = "quota_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    MARKETING_SPEND = "marketing_spend"
    PRICING_STRATEGY = "pricing_strategy"
    TEAM_CAPACITY = "team_capacity"
    TERRITORY_FOCUS = "territory_focus"

@dataclass
class SeasonalPattern:
    """Detected seasonal pattern"""
    pattern_id: str
    name: str
    period: SeasonalPeriod
    pattern_type: PatternType
    strength: SeasonalityStrength
    peak_months: List[int]
    trough_months: List[int]
    average_variation: float  # Percentage variation from baseline
    confidence_score: float
    historical_data_points: int
    detected_date: datetime
    next_occurrence: datetime
    impact_factors: List[str]
    business_drivers: List[str]

@dataclass
class SeasonalForecast:
    """Seasonal forecast for future periods"""
    forecast_id: str
    period_start: datetime
    period_end: datetime
    baseline_value: float
    seasonal_multiplier: float
    forecasted_value: float
    confidence_interval: Tuple[float, float]
    pattern_ids: List[str]  # Contributing patterns
    risk_factors: List[str]
    assumptions: List[str]
    forecast_date: datetime

@dataclass
class SeasonalAdjustment:
    """Recommended seasonal adjustment"""
    adjustment_id: str
    adjustment_type: AdjustmentType
    target_period: datetime
    current_value: float
    recommended_value: float
    adjustment_percentage: float
    rationale: str
    expected_impact: Dict[str, float]
    implementation_timeline: List[Dict[str, Any]]
    success_metrics: List[str]
    risk_assessment: str

@dataclass
class SeasonalStrategy:
    """Comprehensive seasonal strategy"""
    strategy_id: str
    strategy_name: str
    target_season: str
    key_patterns: List[str]  # Pattern IDs
    strategic_objectives: List[str]
    tactical_adjustments: List[SeasonalAdjustment]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, datetime]
    success_criteria: Dict[str, float]
    contingency_plans: List[Dict[str, Any]]
    created_date: datetime

class SeasonalPatternEngine:
    """Advanced seasonal pattern recognition and adaptation system"""
    
    def __init__(self, config_path: str = "config/seasonal_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Data storage
        self.patterns: Dict[str, SeasonalPattern] = {}
        self.forecasts: Dict[str, SeasonalForecast] = {}
        self.adjustments: Dict[str, SeasonalAdjustment] = {}
        self.strategies: Dict[str, SeasonalStrategy] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # ML models
        self.forecasting_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trend_model = LinearRegression()
        self.scaler = StandardScaler()
        
        # Pattern detection parameters
        self.min_data_points = 24  # Minimum months for pattern detection
        self.confidence_threshold = 0.7
        
        logger.info("SeasonalPatternEngine initialized")
    
    async def detect_seasonal_patterns(
        self,
        sales_data: pd.DataFrame,
        segment_filters: Dict[str, Any] = None,
        detect_holidays: bool = True
    ) -> Dict[str, List[SeasonalPattern]]:
        """Detect seasonal patterns in sales data"""
        try:
            if sales_data.empty or len(sales_data) < self.min_data_points:
                logger.warning("Insufficient data for pattern detection")
                return {}
            
            # Prepare data
            processed_data = await self._prepare_sales_data(sales_data, segment_filters)
            
            detected_patterns = {}
            
            # Detect patterns for different segments
            for segment, data in processed_data.items():
                segment_patterns = []
                
                # Monthly patterns
                monthly_patterns = await self._detect_monthly_patterns(data)
                segment_patterns.extend(monthly_patterns)
                
                # Quarterly patterns
                quarterly_patterns = await self._detect_quarterly_patterns(data)
                segment_patterns.extend(quarterly_patterns)
                
                # Holiday patterns (if enabled)
                if detect_holidays:
                    holiday_patterns = await self._detect_holiday_patterns(data)
                    segment_patterns.extend(holiday_patterns)
                
                # Industry-specific patterns
                industry_patterns = await self._detect_industry_patterns(data, segment)
                segment_patterns.extend(industry_patterns)
                
                # Store patterns
                for pattern in segment_patterns:
                    self.patterns[pattern.pattern_id] = pattern
                
                detected_patterns[segment] = segment_patterns
            
            logger.info(f"Detected {sum(len(p) for p in detected_patterns.values())} seasonal patterns")
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {e}")
            return {}
    
    async def generate_seasonal_forecasts(
        self,
        patterns: Dict[str, List[SeasonalPattern]],
        forecast_horizon_months: int = 12,
        confidence_level: float = 0.95
    ) -> Dict[str, List[SeasonalForecast]]:
        """Generate seasonal forecasts based on detected patterns"""
        try:
            forecasts = {}
            
            for segment, segment_patterns in patterns.items():
                segment_forecasts = []
                
                # Generate monthly forecasts for the horizon
                current_date = datetime.now()
                
                for month_offset in range(forecast_horizon_months):
                    target_date = current_date + timedelta(days=30 * month_offset)
                    
                    # Calculate baseline value (trend-adjusted)
                    baseline = await self._calculate_baseline_value(segment, target_date)
                    
                    # Apply seasonal multipliers from all relevant patterns
                    seasonal_multiplier = await self._calculate_seasonal_multiplier(
                        segment_patterns, target_date
                    )
                    
                    # Generate forecast
                    forecast = await self._create_seasonal_forecast(
                        segment, target_date, baseline, seasonal_multiplier,
                        segment_patterns, confidence_level
                    )
                    
                    if forecast:
                        segment_forecasts.append(forecast)
                        self.forecasts[forecast.forecast_id] = forecast
                
                forecasts[segment] = segment_forecasts
            
            logger.info(f"Generated seasonal forecasts for {len(forecasts)} segments")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating seasonal forecasts: {e}")
            return {}
    
    async def recommend_seasonal_adjustments(
        self,
        forecasts: Dict[str, List[SeasonalForecast]],
        current_metrics: Dict[str, Any],
        business_constraints: Dict[str, Any] = None
    ) -> Dict[str, List[SeasonalAdjustment]]:
        """Recommend seasonal adjustments based on forecasts"""
        try:
            if business_constraints is None:
                business_constraints = self._get_default_constraints()
            
            adjustments = {}
            
            for segment, segment_forecasts in forecasts.items():
                segment_adjustments = []
                
                # Analyze each forecast period
                for forecast in segment_forecasts:
                    period_adjustments = await self._analyze_period_adjustments(
                        forecast, current_metrics.get(segment, {}), business_constraints
                    )
                    segment_adjustments.extend(period_adjustments)
                
                # Prioritize and optimize adjustments
                optimized_adjustments = await self._optimize_adjustments(
                    segment_adjustments, business_constraints
                )
                
                # Store adjustments
                for adjustment in optimized_adjustments:
                    self.adjustments[adjustment.adjustment_id] = adjustment
                
                adjustments[segment] = optimized_adjustments
            
            logger.info(f"Generated seasonal adjustments for {len(adjustments)} segments")
            return adjustments
            
        except Exception as e:
            logger.error(f"Error recommending seasonal adjustments: {e}")
            return {}
    
    async def create_seasonal_strategy(
        self,
        target_season: str,
        patterns: Dict[str, List[SeasonalPattern]],
        adjustments: Dict[str, List[SeasonalAdjustment]],
        strategic_objectives: List[str] = None
    ) -> SeasonalStrategy:
        """Create comprehensive seasonal strategy"""
        try:
            if strategic_objectives is None:
                strategic_objectives = [
                    "Maximize seasonal revenue opportunities",
                    "Optimize resource allocation",
                    "Minimize seasonal volatility impact"
                ]
            
            # Identify key patterns for the target season
            key_patterns = await self._identify_key_patterns_for_season(
                target_season, patterns
            )
            
            # Aggregate relevant adjustments
            tactical_adjustments = []
            for segment_adjustments in adjustments.values():
                for adj in segment_adjustments:
                    if self._is_adjustment_relevant_for_season(adj, target_season):
                        tactical_adjustments.append(adj)
            
            # Calculate resource requirements
            resource_requirements = await self._calculate_resource_requirements(
                tactical_adjustments
            )
            
            # Create implementation timeline
            timeline = await self._create_implementation_timeline(
                target_season, tactical_adjustments
            )
            
            # Define success criteria
            success_criteria = await self._define_success_criteria(
                key_patterns, strategic_objectives
            )
            
            # Generate contingency plans
            contingency_plans = await self._generate_contingency_plans(
                key_patterns, tactical_adjustments
            )
            
            strategy = SeasonalStrategy(
                strategy_id=f"strategy_{target_season}_{int(datetime.now().timestamp())}",
                strategy_name=f"Seasonal Strategy for {target_season}",
                target_season=target_season,
                key_patterns=[p.pattern_id for patterns_list in key_patterns.values() 
                            for p in patterns_list],
                strategic_objectives=strategic_objectives,
                tactical_adjustments=tactical_adjustments,
                resource_requirements=resource_requirements,
                timeline=timeline,
                success_criteria=success_criteria,
                contingency_plans=contingency_plans,
                created_date=datetime.now()
            )
            
            self.strategies[strategy.strategy_id] = strategy
            
            logger.info(f"Created seasonal strategy for {target_season}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating seasonal strategy: {e}")
            raise
    
    async def monitor_seasonal_performance(
        self,
        strategy_id: str,
        actual_metrics: Dict[str, Any],
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Monitor and analyze seasonal strategy performance"""
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            strategy = self.strategies[strategy_id]
            
            monitoring_report = {
                "strategy_id": strategy_id,
                "monitoring_period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                "performance_analysis": {},
                "pattern_accuracy": {},
                "adjustment_effectiveness": {},
                "variance_analysis": {},
                "recommendations": [],
                "generated_at": datetime.now().isoformat()
            }
            
            # Analyze overall performance against success criteria
            monitoring_report["performance_analysis"] = await self._analyze_strategy_performance(
                strategy, actual_metrics
            )
            
            # Check pattern prediction accuracy
            monitoring_report["pattern_accuracy"] = await self._evaluate_pattern_accuracy(
                strategy.key_patterns, actual_metrics, period_start, period_end
            )
            
            # Evaluate adjustment effectiveness
            monitoring_report["adjustment_effectiveness"] = await self._evaluate_adjustment_effectiveness(
                strategy.tactical_adjustments, actual_metrics
            )
            
            # Variance analysis
            monitoring_report["variance_analysis"] = await self._analyze_performance_variance(
                strategy, actual_metrics, period_start, period_end
            )
            
            # Generate improvement recommendations
            monitoring_report["recommendations"] = await self._generate_improvement_recommendations(
                monitoring_report["performance_analysis"],
                monitoring_report["pattern_accuracy"],
                monitoring_report["adjustment_effectiveness"]
            )
            
            return monitoring_report
            
        except Exception as e:
            logger.error(f"Error monitoring seasonal performance: {e}")
            return {"error": str(e)}
    
    async def get_seasonal_insights_dashboard(
        self,
        time_horizon_months: int = 12,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive seasonal insights dashboard"""
        try:
            dashboard = {
                "generated_at": datetime.now().isoformat(),
                "horizon_months": time_horizon_months,
                "pattern_summary": {},
                "seasonal_calendar": {},
                "forecast_summary": {},
                "adjustment_opportunities": {},
                "risk_assessment": {},
                "strategic_recommendations": []
            }
            
            # Pattern summary
            dashboard["pattern_summary"] = await self._get_pattern_summary()
            
            # Seasonal calendar
            dashboard["seasonal_calendar"] = await self._create_seasonal_calendar(
                time_horizon_months
            )
            
            # Forecast summary
            dashboard["forecast_summary"] = await self._get_forecast_summary()
            
            # Adjustment opportunities
            dashboard["adjustment_opportunities"] = await self._identify_adjustment_opportunities()
            
            # Risk assessment
            dashboard["risk_assessment"] = await self._assess_seasonal_risks()
            
            # Strategic recommendations
            if include_recommendations:
                dashboard["strategic_recommendations"] = await self._generate_strategic_recommendations()
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating seasonal insights dashboard: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _prepare_sales_data(
        self,
        sales_data: pd.DataFrame,
        segment_filters: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Prepare and segment sales data for analysis"""
        processed_data = {}
        
        # Ensure date column is datetime
        if 'date' not in sales_data.columns:
            if 'Date' in sales_data.columns:
                sales_data['date'] = pd.to_datetime(sales_data['Date'])
            else:
                sales_data['date'] = pd.to_datetime(sales_data.index)
        else:
            sales_data['date'] = pd.to_datetime(sales_data['date'])
        
        # Add time-based features
        sales_data['year'] = sales_data['date'].dt.year
        sales_data['month'] = sales_data['date'].dt.month
        sales_data['quarter'] = sales_data['date'].dt.quarter
        sales_data['week'] = sales_data['date'].dt.isocalendar().week
        
        # Segment data
        if segment_filters:
            for segment_name, filters in segment_filters.items():
                segment_data = sales_data.copy()
                for column, values in filters.items():
                    if column in segment_data.columns:
                        segment_data = segment_data[segment_data[column].isin(values)]
                processed_data[segment_name] = segment_data
        else:
            processed_data['overall'] = sales_data
        
        # Store for future reference
        for segment, data in processed_data.items():
            self.historical_data[segment] = data
        
        return processed_data
    
    async def _detect_monthly_patterns(self, data: pd.DataFrame) -> List[SeasonalPattern]:
        """Detect monthly seasonal patterns"""
        patterns = []
        
        if 'month' not in data.columns or len(data) < 12:
            return patterns
        
        # Group by month and calculate statistics
        monthly_stats = data.groupby('month').agg({
            'revenue': ['mean', 'std', 'count']
        }).reset_index()
        monthly_stats.columns = ['month', 'mean_revenue', 'std_revenue', 'count']
        
        # Calculate seasonal indices
        overall_mean = monthly_stats['mean_revenue'].mean()
        monthly_stats['seasonal_index'] = monthly_stats['mean_revenue'] / overall_mean
        
        # Identify significant variations
        variation_threshold = 0.15  # 15% variation
        significant_months = monthly_stats[
            abs(monthly_stats['seasonal_index'] - 1) > variation_threshold
        ]
        
        if len(significant_months) >= 3:  # Need at least 3 months of significant variation
            # Determine pattern characteristics
            peak_months = significant_months[
                significant_months['seasonal_index'] > 1
            ]['month'].tolist()
            
            trough_months = significant_months[
                significant_months['seasonal_index'] < 1
            ]['month'].tolist()
            
            average_variation = abs(monthly_stats['seasonal_index'] - 1).mean()
            
            # Determine strength
            strength = self._calculate_pattern_strength(average_variation)
            
            # Determine pattern type
            pattern_type = self._determine_pattern_type(monthly_stats['seasonal_index'].values)
            
            pattern = SeasonalPattern(
                pattern_id=f"monthly_pattern_{int(datetime.now().timestamp())}",
                name="Monthly Seasonal Pattern",
                period=SeasonalPeriod.MONTHLY,
                pattern_type=pattern_type,
                strength=strength,
                peak_months=peak_months,
                trough_months=trough_months,
                average_variation=average_variation * 100,  # Convert to percentage
                confidence_score=min(0.9, len(data) / 100),  # Confidence based on data points
                historical_data_points=len(data),
                detected_date=datetime.now(),
                next_occurrence=self._calculate_next_occurrence(peak_months),
                impact_factors=["seasonal_demand", "budget_cycles"],
                business_drivers=["year_end_spending", "quarterly_planning"]
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_quarterly_patterns(self, data: pd.DataFrame) -> List[SeasonalPattern]:
        """Detect quarterly seasonal patterns"""
        patterns = []
        
        if 'quarter' not in data.columns or len(data) < 8:  # Need at least 2 years
            return patterns
        
        # Group by quarter
        quarterly_stats = data.groupby('quarter').agg({
            'revenue': ['mean', 'std', 'count']
        }).reset_index()
        quarterly_stats.columns = ['quarter', 'mean_revenue', 'std_revenue', 'count']
        
        # Calculate seasonal indices
        overall_mean = quarterly_stats['mean_revenue'].mean()
        quarterly_stats['seasonal_index'] = quarterly_stats['mean_revenue'] / overall_mean
        
        # Check for significant quarterly variation
        variation_threshold = 0.20  # 20% variation for quarterly
        significant_quarters = quarterly_stats[
            abs(quarterly_stats['seasonal_index'] - 1) > variation_threshold
        ]
        
        if len(significant_quarters) >= 2:
            peak_quarters = significant_quarters[
                significant_quarters['seasonal_index'] > 1
            ]['quarter'].tolist()
            
            trough_quarters = significant_quarters[
                significant_quarters['seasonal_index'] < 1
            ]['quarter'].tolist()
            
            average_variation = abs(quarterly_stats['seasonal_index'] - 1).mean()
            strength = self._calculate_pattern_strength(average_variation)
            pattern_type = self._determine_pattern_type(quarterly_stats['seasonal_index'].values)
            
            pattern = SeasonalPattern(
                pattern_id=f"quarterly_pattern_{int(datetime.now().timestamp())}",
                name="Quarterly Seasonal Pattern",
                period=SeasonalPeriod.QUARTERLY,
                pattern_type=pattern_type,
                strength=strength,
                peak_months=[q*3 for q in peak_quarters],  # Convert quarters to months
                trough_months=[q*3 for q in trough_quarters],
                average_variation=average_variation * 100,
                confidence_score=min(0.85, len(data) / 50),
                historical_data_points=len(data),
                detected_date=datetime.now(),
                next_occurrence=self._calculate_next_occurrence([q*3 for q in peak_quarters]),
                impact_factors=["quarterly_budgets", "business_cycles"],
                business_drivers=["quarterly_targets", "fiscal_year_planning"]
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_holiday_patterns(self, data: pd.DataFrame) -> List[SeasonalPattern]:
        """Detect holiday-related seasonal patterns"""
        patterns = []
        
        # Define holiday periods (simplified)
        holiday_periods = {
            "year_end": [11, 12],  # November, December
            "new_year": [1],  # January
            "summer": [6, 7, 8],  # June, July, August
            "spring": [3, 4, 5]  # March, April, May
        }
        
        for holiday_name, months in holiday_periods.items():
            # Check if there's a significant pattern in holiday months
            holiday_data = data[data['month'].isin(months)]
            non_holiday_data = data[~data['month'].isin(months)]
            
            if len(holiday_data) > 0 and len(non_holiday_data) > 0:
                holiday_mean = holiday_data['revenue'].mean()
                non_holiday_mean = non_holiday_data['revenue'].mean()
                
                seasonal_index = holiday_mean / non_holiday_mean if non_holiday_mean > 0 else 1
                
                if abs(seasonal_index - 1) > 0.25:  # 25% difference
                    pattern_type = PatternType.CYCLICAL
                    strength = self._calculate_pattern_strength(abs(seasonal_index - 1))
                    
                    pattern = SeasonalPattern(
                        pattern_id=f"holiday_{holiday_name}_{int(datetime.now().timestamp())}",
                        name=f"{holiday_name.title()} Holiday Pattern",
                        period=SeasonalPeriod.HOLIDAY,
                        pattern_type=pattern_type,
                        strength=strength,
                        peak_months=months if seasonal_index > 1 else [],
                        trough_months=months if seasonal_index < 1 else [],
                        average_variation=abs(seasonal_index - 1) * 100,
                        confidence_score=0.75,
                        historical_data_points=len(holiday_data),
                        detected_date=datetime.now(),
                        next_occurrence=self._calculate_next_occurrence(months),
                        impact_factors=[f"{holiday_name}_holidays", "consumer_behavior"],
                        business_drivers=[f"{holiday_name}_buying_patterns", "seasonal_marketing"]
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_industry_patterns(self, data: pd.DataFrame, segment: str) -> List[SeasonalPattern]:
        """Detect industry-specific seasonal patterns"""
        patterns = []
        
        # Industry-specific seasonal behaviors (simplified)
        industry_patterns = {
            "education": {"peak": [8, 9], "trough": [6, 7]},  # Back to school
            "retail": {"peak": [11, 12], "trough": [1, 2]},  # Holiday season
            "construction": {"peak": [4, 5, 6], "trough": [12, 1, 2]},  # Weather dependent
            "agriculture": {"peak": [3, 4, 5], "trough": [11, 12, 1]}  # Planting season
        }
        
        # Try to match segment to industry patterns
        for industry, pattern_info in industry_patterns.items():
            if industry.lower() in segment.lower():
                peak_months = pattern_info["peak"]
                trough_months = pattern_info["trough"]
                
                # Verify pattern exists in data
                peak_data = data[data['month'].isin(peak_months)]
                trough_data = data[data['month'].isin(trough_months)]
                
                if len(peak_data) > 0 and len(trough_data) > 0:
                    peak_mean = peak_data['revenue'].mean()
                    trough_mean = trough_data['revenue'].mean()
                    
                    seasonal_index = peak_mean / trough_mean if trough_mean > 0 else 1
                    
                    if seasonal_index > 1.2:  # 20% difference
                        average_variation = (seasonal_index - 1) * 0.5  # Simplified calculation
                        strength = self._calculate_pattern_strength(average_variation)
                        
                        pattern = SeasonalPattern(
                            pattern_id=f"industry_{industry}_{int(datetime.now().timestamp())}",
                            name=f"{industry.title()} Industry Pattern",
                            period=SeasonalPeriod.INDUSTRY_SPECIFIC,
                            pattern_type=PatternType.CYCLICAL,
                            strength=strength,
                            peak_months=peak_months,
                            trough_months=trough_months,
                            average_variation=average_variation * 100,
                            confidence_score=0.65,
                            historical_data_points=len(data),
                            detected_date=datetime.now(),
                            next_occurrence=self._calculate_next_occurrence(peak_months),
                            impact_factors=[f"{industry}_seasonality", "industry_cycles"],
                            business_drivers=[f"{industry}_demand_patterns", "market_dynamics"]
                        )
                        
                        patterns.append(pattern)
                        break  # Only add one industry pattern per segment
        
        return patterns
    
    def _calculate_pattern_strength(self, variation: float) -> SeasonalityStrength:
        """Calculate the strength of a seasonal pattern"""
        if variation >= 0.5:
            return SeasonalityStrength.VERY_STRONG
        elif variation >= 0.3:
            return SeasonalityStrength.STRONG
        elif variation >= 0.15:
            return SeasonalityStrength.MODERATE
        elif variation >= 0.05:
            return SeasonalityStrength.WEAK
        else:
            return SeasonalityStrength.NEGLIGIBLE
    
    def _determine_pattern_type(self, seasonal_indices: np.ndarray) -> PatternType:
        """Determine the type of seasonal pattern"""
        # Simple heuristics for pattern type determination
        max_index = np.max(seasonal_indices)
        min_index = np.min(seasonal_indices)
        variation = max_index - min_index
        trend = np.polyfit(range(len(seasonal_indices)), seasonal_indices, 1)[0]
        
        if abs(trend) > 0.1:
            return PatternType.TRENDING if trend > 0 else PatternType.DECLINING
        elif variation > 0.5:
            return PatternType.VOLATILE
        elif variation > 0.2:
            return PatternType.CYCLICAL
        elif variation < 0.1:
            return PatternType.STABLE
        else:
            return PatternType.CYCLICAL  # Default
    
    def _calculate_next_occurrence(self, months: List[int]) -> datetime:
        """Calculate the next occurrence of peak months"""
        if not months:
            return datetime.now() + timedelta(days=365)
        
        current_month = datetime.now().month
        next_month = min([m for m in months if m > current_month] + 
                        [m + 12 for m in months])
        
        if next_month > 12:
            next_month -= 12
            year_offset = 1
        else:
            year_offset = 0
        
        return datetime.now().replace(
            month=next_month,
            year=datetime.now().year + year_offset,
            day=1
        )
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """Get default business constraints for adjustments"""
        return {
            "max_quota_adjustment": 0.25,  # 25% max adjustment
            "max_resource_reallocation": 0.30,  # 30% max reallocation
            "budget_flexibility": 0.15,  # 15% budget flexibility
            "team_capacity_limit": 1.2  # 20% over normal capacity
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load seasonal pattern configuration"""
        return {
            "pattern_detection_threshold": 0.15,
            "confidence_threshold": 0.7,
            "forecast_accuracy_target": 0.80,
            "adjustment_implementation_window_days": 45
        }


# Global instance
seasonal_pattern_engine = SeasonalPatternEngine()