"""
Performance Benchmarking Engine
Compare sales rep performance against top performers with detailed metrics and analytics
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque
import pandas as pd

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Types of performance metrics"""
    CONVERSION_RATE = "conversion_rate"
    AVERAGE_DEAL_SIZE = "average_deal_size"
    SALES_CYCLE_LENGTH = "sales_cycle_length"
    CALL_TO_CLOSE_RATIO = "call_to_close_ratio"
    OBJECTION_HANDLING_SUCCESS = "objection_handling_success"
    RAPPORT_BUILDING_SCORE = "rapport_building_score"
    PRODUCT_KNOWLEDGE_SCORE = "product_knowledge_score"
    COMMUNICATION_EFFECTIVENESS = "communication_effectiveness"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    FOLLOW_UP_CONSISTENCY = "follow_up_consistency"
    PIPELINE_MANAGEMENT = "pipeline_management"
    ACTIVITY_VOLUME = "activity_volume"

class BenchmarkCategory(Enum):
    """Categories for benchmarking"""
    TOP_PERFORMER = "top_performer"
    HIGH_PERFORMER = "high_performer"
    AVERAGE_PERFORMER = "average_performer"
    DEVELOPING_PERFORMER = "developing_performer"

class PerformancePeriod(Enum):
    """Time periods for performance analysis"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class SalesRepProfile:
    """Sales representative profile"""
    rep_id: str
    name: str
    team: str
    role: str
    start_date: datetime
    territory: str
    product_focus: List[str]
    manager_id: str
    experience_level: str  # junior, mid, senior
    quota: float
    current_performance_category: BenchmarkCategory

@dataclass
class PerformanceMetricData:
    """Individual performance metric data"""
    metric: PerformanceMetric
    value: float
    period: PerformancePeriod
    date: datetime
    rep_id: str
    context: Dict[str, Any]
    benchmark_percentile: float  # Where this sits vs all reps (0-100)
    category_percentile: float   # Where this sits vs same category (0-100)

@dataclass
class BenchmarkData:
    """Benchmark data for comparison"""
    metric: PerformanceMetric
    period: PerformancePeriod
    top_performer_avg: float
    top_performer_min: float
    high_performer_avg: float
    average_performer_avg: float
    developing_performer_avg: float
    overall_avg: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    sample_size: int
    last_updated: datetime

@dataclass
class PerformanceComparison:
    """Comparison result between rep and benchmarks"""
    rep_id: str
    metric: PerformanceMetric
    rep_value: float
    benchmark_data: BenchmarkData
    performance_gap: float  # Difference from top performer average
    percentile_rank: float  # Overall percentile rank
    category_rank: float   # Rank within performance category
    improvement_potential: float  # Potential improvement to reach next category
    comparison_insights: List[str]
    recommended_actions: List[str]

class PerformanceBenchmarkingEngine:
    """Advanced performance benchmarking and comparison system"""
    
    def __init__(self, config_path: str = "config/benchmarking_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Performance data storage
        self.sales_reps: Dict[str, SalesRepProfile] = {}
        self.performance_history: Dict[str, List[PerformanceMetricData]] = defaultdict(list)
        self.benchmark_cache: Dict[str, BenchmarkData] = {}
        
        # Performance thresholds for categorization
        self.category_thresholds = self._initialize_category_thresholds()
        
        # Metric weights for overall scoring
        self.metric_weights = self._initialize_metric_weights()
        
        logger.info("PerformanceBenchmarkingEngine initialized")
    
    async def add_sales_rep(
        self,
        rep_data: Dict[str, Any]
    ) -> SalesRepProfile:
        """Add or update sales representative profile"""
        try:
            rep_profile = SalesRepProfile(
                rep_id=rep_data["rep_id"],
                name=rep_data["name"],
                team=rep_data["team"],
                role=rep_data["role"],
                start_date=datetime.fromisoformat(rep_data["start_date"]),
                territory=rep_data["territory"],
                product_focus=rep_data["product_focus"],
                manager_id=rep_data["manager_id"],
                experience_level=rep_data["experience_level"],
                quota=rep_data["quota"],
                current_performance_category=BenchmarkCategory(rep_data.get("performance_category", "average_performer"))
            )
            
            self.sales_reps[rep_profile.rep_id] = rep_profile
            
            logger.info(f"Added sales rep profile: {rep_profile.name}")
            return rep_profile
            
        except Exception as e:
            logger.error(f"Error adding sales rep: {e}")
            raise
    
    async def record_performance_metric(
        self,
        rep_id: str,
        metric: PerformanceMetric,
        value: float,
        period: PerformancePeriod,
        date: datetime = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Record a performance metric for a sales rep"""
        try:
            if rep_id not in self.sales_reps:
                logger.warning(f"Sales rep {rep_id} not found")
                return False
            
            # Calculate benchmark percentiles
            benchmark_percentile = await self._calculate_benchmark_percentile(
                metric, value, period
            )
            
            category_percentile = await self._calculate_category_percentile(
                rep_id, metric, value, period
            )
            
            # Create metric data
            metric_data = PerformanceMetricData(
                metric=metric,
                value=value,
                period=period,
                date=date or datetime.now(),
                rep_id=rep_id,
                context=context or {},
                benchmark_percentile=benchmark_percentile,
                category_percentile=category_percentile
            )
            
            # Store the data
            self.performance_history[rep_id].append(metric_data)
            
            # Update rep's performance category if needed
            await self._update_rep_category(rep_id)
            
            # Invalidate relevant benchmark cache
            cache_key = f"{metric.value}_{period.value}"
            if cache_key in self.benchmark_cache:
                del self.benchmark_cache[cache_key]
            
            logger.debug(f"Recorded {metric.value} for rep {rep_id}: {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
    
    async def get_performance_comparison(
        self,
        rep_id: str,
        metrics: List[PerformanceMetric] = None,
        period: PerformancePeriod = PerformancePeriod.MONTHLY,
        include_insights: bool = True
    ) -> List[PerformanceComparison]:
        """Get comprehensive performance comparison for a sales rep"""
        try:
            if rep_id not in self.sales_reps:
                logger.warning(f"Sales rep {rep_id} not found")
                return []
            
            if metrics is None:
                metrics = list(PerformanceMetric)
            
            comparisons = []
            
            for metric in metrics:
                # Get rep's latest performance for this metric
                rep_performance = await self._get_latest_metric_value(rep_id, metric, period)
                
                if rep_performance is None:
                    continue
                
                # Get benchmark data
                benchmark_data = await self._get_benchmark_data(metric, period)
                
                if benchmark_data is None:
                    continue
                
                # Calculate comparison
                comparison = await self._create_performance_comparison(
                    rep_id, metric, rep_performance, benchmark_data, include_insights
                )
                
                if comparison:
                    comparisons.append(comparison)
            
            logger.info(f"Generated {len(comparisons)} performance comparisons for rep {rep_id}")
            return comparisons
            
        except Exception as e:
            logger.error(f"Error getting performance comparison: {e}")
            return []
    
    async def get_top_performers(
        self,
        metric: PerformanceMetric,
        period: PerformancePeriod,
        limit: int = 10,
        team_filter: str = None,
        territory_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Get top performers for a specific metric"""
        try:
            # Collect all performance data for the metric
            performer_data = []
            
            for rep_id, rep_profile in self.sales_reps.items():
                # Apply filters
                if team_filter and rep_profile.team != team_filter:
                    continue
                if territory_filter and rep_profile.territory != territory_filter:
                    continue
                
                # Get latest metric value
                metric_value = await self._get_latest_metric_value(rep_id, metric, period)
                
                if metric_value is not None:
                    performer_data.append({
                        "rep_id": rep_id,
                        "name": rep_profile.name,
                        "team": rep_profile.team,
                        "territory": rep_profile.territory,
                        "metric_value": metric_value,
                        "experience_level": rep_profile.experience_level,
                        "quota": rep_profile.quota
                    })
            
            # Sort by metric value (descending for most metrics)
            reverse_sort = metric not in [PerformanceMetric.SALES_CYCLE_LENGTH]  # Lower is better for cycle length
            
            top_performers = sorted(
                performer_data,
                key=lambda x: x["metric_value"],
                reverse=reverse_sort
            )[:limit]
            
            logger.info(f"Retrieved {len(top_performers)} top performers for {metric.value}")
            return top_performers
            
        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return []
    
    async def get_benchmark_report(
        self,
        rep_id: str,
        period: PerformancePeriod = PerformancePeriod.MONTHLY,
        include_trends: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report for a sales rep"""
        try:
            if rep_id not in self.sales_reps:
                return {"error": "Sales rep not found"}
            
            rep_profile = self.sales_reps[rep_id]
            
            # Get performance comparisons
            comparisons = await self.get_performance_comparison(
                rep_id, period=period, include_insights=True
            )
            
            # Calculate overall performance score
            overall_score = await self._calculate_overall_performance_score(rep_id, period)
            
            # Get peer comparison
            peer_ranking = await self._get_peer_ranking(rep_id, period)
            
            # Get trends if requested
            trends = {}
            if include_trends:
                trends = await self._calculate_performance_trends(rep_id, period)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = await self._identify_strengths_weaknesses(comparisons)
            
            # Generate improvement recommendations
            recommendations = await self._generate_improvement_recommendations(
                rep_id, comparisons, trends
            )
            
            report = {
                "rep_profile": {
                    "rep_id": rep_profile.rep_id,
                    "name": rep_profile.name,
                    "team": rep_profile.team,
                    "territory": rep_profile.territory,
                    "experience_level": rep_profile.experience_level,
                    "current_category": rep_profile.current_performance_category.value
                },
                "overall_performance": {
                    "score": overall_score,
                    "percentile_rank": peer_ranking.get("percentile_rank", 0),
                    "team_rank": peer_ranking.get("team_rank", 0),
                    "total_team_size": peer_ranking.get("total_team_size", 0)
                },
                "metric_comparisons": [
                    {
                        "metric": comp.metric.value,
                        "rep_value": comp.rep_value,
                        "top_performer_avg": comp.benchmark_data.top_performer_avg,
                        "percentile_rank": comp.percentile_rank,
                        "performance_gap": comp.performance_gap,
                        "improvement_potential": comp.improvement_potential
                    }
                    for comp in comparisons
                ],
                "strengths": strengths,
                "weaknesses": weaknesses,
                "trends": trends,
                "recommendations": recommendations,
                "report_date": datetime.now().isoformat(),
                "period": period.value
            }
            
            logger.info(f"Generated benchmark report for rep {rep_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating benchmark report: {e}")
            return {"error": str(e)}
    
    async def get_team_performance_analysis(
        self,
        team: str,
        period: PerformancePeriod = PerformancePeriod.MONTHLY,
        include_individual_details: bool = False
    ) -> Dict[str, Any]:
        """Analyze performance for an entire team"""
        try:
            # Get all reps in the team
            team_reps = [
                rep for rep in self.sales_reps.values()
                if rep.team == team
            ]
            
            if not team_reps:
                return {"error": f"No sales reps found for team {team}"}
            
            team_analysis = {
                "team": team,
                "period": period.value,
                "total_reps": len(team_reps),
                "team_metrics": {},
                "performance_distribution": {},
                "top_performers": [],
                "improvement_opportunities": [],
                "team_benchmarks": {}
            }
            
            # Calculate team metrics
            for metric in PerformanceMetric:
                metric_values = []
                
                for rep in team_reps:
                    value = await self._get_latest_metric_value(rep.rep_id, metric, period)
                    if value is not None:
                        metric_values.append(value)
                
                if metric_values:
                    team_analysis["team_metrics"][metric.value] = {
                        "average": statistics.mean(metric_values),
                        "median": statistics.median(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "std_dev": statistics.stdev(metric_values) if len(metric_values) > 1 else 0
                    }
            
            # Performance distribution by category
            category_counts = defaultdict(int)
            for rep in team_reps:
                category_counts[rep.current_performance_category.value] += 1
            
            team_analysis["performance_distribution"] = dict(category_counts)
            
            # Get top 3 performers in team
            if len(team_reps) >= 3:
                # Use conversion rate as primary metric for ranking
                team_performance_scores = []
                for rep in team_reps:
                    score = await self._calculate_overall_performance_score(rep.rep_id, period)
                    team_performance_scores.append({"rep_id": rep.rep_id, "name": rep.name, "score": score})
                
                top_3 = sorted(team_performance_scores, key=lambda x: x["score"], reverse=True)[:3]
                team_analysis["top_performers"] = top_3
            
            # Individual details if requested
            if include_individual_details:
                team_analysis["individual_details"] = []
                for rep in team_reps:
                    rep_comparisons = await self.get_performance_comparison(
                        rep.rep_id, period=period, include_insights=False
                    )
                    
                    team_analysis["individual_details"].append({
                        "rep_id": rep.rep_id,
                        "name": rep.name,
                        "performance_category": rep.current_performance_category.value,
                        "comparisons": [
                            {
                                "metric": comp.metric.value,
                                "percentile_rank": comp.percentile_rank,
                                "performance_gap": comp.performance_gap
                            }
                            for comp in rep_comparisons
                        ]
                    })
            
            logger.info(f"Generated team analysis for {team} with {len(team_reps)} reps")
            return team_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing team performance: {e}")
            return {"error": str(e)}
    
    async def _get_latest_metric_value(
        self,
        rep_id: str,
        metric: PerformanceMetric,
        period: PerformancePeriod,
        lookback_days: int = 30
    ) -> Optional[float]:
        """Get the latest metric value for a rep"""
        try:
            if rep_id not in self.performance_history:
                return None
            
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Filter for the specific metric and period
            relevant_metrics = [
                m for m in self.performance_history[rep_id]
                if (m.metric == metric and 
                    m.period == period and 
                    m.date >= cutoff_date)
            ]
            
            if not relevant_metrics:
                return None
            
            # Return the most recent value
            latest_metric = max(relevant_metrics, key=lambda x: x.date)
            return latest_metric.value
            
        except Exception as e:
            logger.error(f"Error getting latest metric value: {e}")
            return None
    
    async def _get_benchmark_data(
        self,
        metric: PerformanceMetric,
        period: PerformancePeriod
    ) -> Optional[BenchmarkData]:
        """Get or calculate benchmark data for a metric"""
        try:
            cache_key = f"{metric.value}_{period.value}"
            
            # Check cache first
            if cache_key in self.benchmark_cache:
                cached_data = self.benchmark_cache[cache_key]
                # Use cache if less than 1 hour old
                if (datetime.now() - cached_data.last_updated).total_seconds() < 3600:
                    return cached_data
            
            # Calculate fresh benchmark data
            all_values = []
            category_values = defaultdict(list)
            
            for rep_id in self.sales_reps:
                value = await self._get_latest_metric_value(rep_id, metric, period)
                if value is not None:
                    all_values.append(value)
                    rep_category = self.sales_reps[rep_id].current_performance_category
                    category_values[rep_category].append(value)
            
            if not all_values:
                return None
            
            # Calculate percentiles
            percentiles = np.percentile(all_values, [25, 50, 75, 90, 95])
            
            # Calculate category averages
            category_avgs = {}
            for category, values in category_values.items():
                if values:
                    category_avgs[category] = statistics.mean(values)
            
            benchmark_data = BenchmarkData(
                metric=metric,
                period=period,
                top_performer_avg=category_avgs.get(BenchmarkCategory.TOP_PERFORMER, 0),
                top_performer_min=min(category_values.get(BenchmarkCategory.TOP_PERFORMER, [0])),
                high_performer_avg=category_avgs.get(BenchmarkCategory.HIGH_PERFORMER, 0),
                average_performer_avg=category_avgs.get(BenchmarkCategory.AVERAGE_PERFORMER, 0),
                developing_performer_avg=category_avgs.get(BenchmarkCategory.DEVELOPING_PERFORMER, 0),
                overall_avg=statistics.mean(all_values),
                percentile_25=percentiles[0],
                percentile_50=percentiles[1],
                percentile_75=percentiles[2],
                percentile_90=percentiles[3],
                percentile_95=percentiles[4],
                sample_size=len(all_values),
                last_updated=datetime.now()
            )
            
            # Cache the result
            self.benchmark_cache[cache_key] = benchmark_data
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            return None
    
    async def _create_performance_comparison(
        self,
        rep_id: str,
        metric: PerformanceMetric,
        rep_value: float,
        benchmark_data: BenchmarkData,
        include_insights: bool
    ) -> Optional[PerformanceComparison]:
        """Create a performance comparison object"""
        try:
            # Calculate performance gap
            performance_gap = rep_value - benchmark_data.top_performer_avg
            
            # Calculate percentile rank
            all_values = [benchmark_data.percentile_25, benchmark_data.percentile_50, 
                         benchmark_data.percentile_75, benchmark_data.percentile_90, 
                         benchmark_data.percentile_95]
            percentile_rank = self._calculate_percentile_rank(rep_value, all_values)
            
            # Get rep's current category
            rep_category = self.sales_reps[rep_id].current_performance_category
            
            # Calculate category rank (simplified)
            category_rank = percentile_rank  # Would be more sophisticated in real implementation
            
            # Calculate improvement potential
            improvement_potential = await self._calculate_improvement_potential(
                rep_value, rep_category, benchmark_data
            )
            
            # Generate insights and recommendations
            insights = []
            recommendations = []
            
            if include_insights:
                insights, recommendations = await self._generate_metric_insights(
                    metric, rep_value, benchmark_data, performance_gap, percentile_rank
                )
            
            comparison = PerformanceComparison(
                rep_id=rep_id,
                metric=metric,
                rep_value=rep_value,
                benchmark_data=benchmark_data,
                performance_gap=performance_gap,
                percentile_rank=percentile_rank,
                category_rank=category_rank,
                improvement_potential=improvement_potential,
                comparison_insights=insights,
                recommended_actions=recommendations
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error creating performance comparison: {e}")
            return None
    
    async def _calculate_overall_performance_score(
        self,
        rep_id: str,
        period: PerformancePeriod
    ) -> float:
        """Calculate overall performance score using weighted metrics"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for metric, weight in self.metric_weights.items():
                metric_value = await self._get_latest_metric_value(rep_id, metric, period)
                if metric_value is not None:
                    # Normalize metric value to 0-100 scale based on benchmarks
                    benchmark_data = await self._get_benchmark_data(metric, period)
                    if benchmark_data:
                        normalized_score = self._normalize_metric_score(
                            metric_value, benchmark_data
                        )
                        total_score += normalized_score * weight
                        total_weight += weight
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall performance score: {e}")
            return 0.0
    
    def _normalize_metric_score(
        self,
        value: float,
        benchmark_data: BenchmarkData
    ) -> float:
        """Normalize a metric value to 0-100 scale"""
        try:
            # Use percentile-based normalization
            if value >= benchmark_data.percentile_95:
                return 100.0
            elif value >= benchmark_data.percentile_90:
                return 90.0
            elif value >= benchmark_data.percentile_75:
                return 75.0
            elif value >= benchmark_data.percentile_50:
                return 50.0
            elif value >= benchmark_data.percentile_25:
                return 25.0
            else:
                return 10.0
                
        except Exception as e:
            logger.error(f"Error normalizing metric score: {e}")
            return 50.0  # Default to median
    
    async def _calculate_benchmark_percentile(
        self,
        metric: PerformanceMetric,
        value: float,
        period: PerformancePeriod
    ) -> float:
        """Calculate what percentile this value represents"""
        try:
            benchmark_data = await self._get_benchmark_data(metric, period)
            if not benchmark_data:
                return 50.0  # Default to median
            
            percentiles = [
                benchmark_data.percentile_25,
                benchmark_data.percentile_50,
                benchmark_data.percentile_75,
                benchmark_data.percentile_90,
                benchmark_data.percentile_95
            ]
            
            return self._calculate_percentile_rank(value, percentiles)
            
        except Exception as e:
            logger.error(f"Error calculating benchmark percentile: {e}")
            return 50.0
    
    def _calculate_percentile_rank(self, value: float, percentiles: List[float]) -> float:
        """Calculate percentile rank for a value"""
        try:
            if value <= percentiles[0]:  # 25th percentile
                return 25.0
            elif value <= percentiles[1]:  # 50th percentile
                return 50.0
            elif value <= percentiles[2]:  # 75th percentile
                return 75.0
            elif value <= percentiles[3]:  # 90th percentile
                return 90.0
            elif value <= percentiles[4]:  # 95th percentile
                return 95.0
            else:
                return 99.0
                
        except Exception as e:
            logger.error(f"Error calculating percentile rank: {e}")
            return 50.0
    
    async def _calculate_category_percentile(
        self,
        rep_id: str,
        metric: PerformanceMetric,
        value: float,
        period: PerformancePeriod
    ) -> float:
        """Calculate percentile within the rep's performance category"""
        try:
            rep_category = self.sales_reps[rep_id].current_performance_category
            
            # Get all values for reps in the same category
            category_values = []
            for other_rep_id, rep_profile in self.sales_reps.items():
                if rep_profile.current_performance_category == rep_category:
                    other_value = await self._get_latest_metric_value(other_rep_id, metric, period)
                    if other_value is not None:
                        category_values.append(other_value)
            
            if len(category_values) <= 1:
                return 50.0  # Default if no comparison data
            
            # Calculate percentile within category
            sorted_values = sorted(category_values)
            position = sorted_values.index(min(sorted_values, key=lambda x: abs(x - value)))
            percentile = (position / (len(sorted_values) - 1)) * 100
            
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating category percentile: {e}")
            return 50.0
    
    async def _update_rep_category(self, rep_id: str):
        """Update rep's performance category based on recent metrics"""
        try:
            overall_score = await self._calculate_overall_performance_score(
                rep_id, PerformancePeriod.MONTHLY
            )
            
            # Determine new category based on score
            if overall_score >= self.category_thresholds["top_performer"]:
                new_category = BenchmarkCategory.TOP_PERFORMER
            elif overall_score >= self.category_thresholds["high_performer"]:
                new_category = BenchmarkCategory.HIGH_PERFORMER
            elif overall_score >= self.category_thresholds["average_performer"]:
                new_category = BenchmarkCategory.AVERAGE_PERFORMER
            else:
                new_category = BenchmarkCategory.DEVELOPING_PERFORMER
            
            # Update if changed
            if self.sales_reps[rep_id].current_performance_category != new_category:
                old_category = self.sales_reps[rep_id].current_performance_category
                self.sales_reps[rep_id].current_performance_category = new_category
                
                logger.info(f"Updated rep {rep_id} category: {old_category.value} -> {new_category.value}")
            
        except Exception as e:
            logger.error(f"Error updating rep category: {e}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load benchmarking configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        return {
            "cache_timeout_minutes": 60,
            "lookback_days": 30,
            "min_sample_size": 5,
            "update_frequency_hours": 24
        }
    
    def _initialize_category_thresholds(self) -> Dict[str, float]:
        """Initialize performance category thresholds"""
        return {
            "top_performer": 85.0,     # Top 15%
            "high_performer": 70.0,    # Top 30%
            "average_performer": 40.0,  # Middle 60%
            "developing_performer": 0.0 # Bottom 40%
        }
    
    def _initialize_metric_weights(self) -> Dict[PerformanceMetric, float]:
        """Initialize metric weights for overall scoring"""
        return {
            PerformanceMetric.CONVERSION_RATE: 0.25,
            PerformanceMetric.AVERAGE_DEAL_SIZE: 0.20,
            PerformanceMetric.SALES_CYCLE_LENGTH: 0.10,
            PerformanceMetric.CALL_TO_CLOSE_RATIO: 0.15,
            PerformanceMetric.OBJECTION_HANDLING_SUCCESS: 0.10,
            PerformanceMetric.COMMUNICATION_EFFECTIVENESS: 0.10,
            PerformanceMetric.CUSTOMER_SATISFACTION: 0.10
        }
    
    # Additional helper methods for insights and recommendations would be implemented here
    async def _generate_metric_insights(
        self,
        metric: PerformanceMetric,
        rep_value: float,
        benchmark_data: BenchmarkData,
        performance_gap: float,
        percentile_rank: float
    ) -> Tuple[List[str], List[str]]:
        """Generate insights and recommendations for a metric"""
        insights = []
        recommendations = []
        
        try:
            if percentile_rank >= 90:
                insights.append(f"Excellent performance - in top 10% for {metric.value}")
                recommendations.append("Share your best practices with the team")
            elif percentile_rank >= 75:
                insights.append(f"Strong performance - above average for {metric.value}")
                recommendations.append("Focus on consistency and small improvements")
            elif percentile_rank >= 50:
                insights.append(f"Average performance for {metric.value}")
                recommendations.append("Identify specific areas for improvement")
            else:
                insights.append(f"Below average performance for {metric.value}")
                recommendations.append("Consider additional training and support")
            
            # Metric-specific insights
            if metric == PerformanceMetric.CONVERSION_RATE:
                if performance_gap < -0.1:  # 10% below top performers
                    recommendations.append("Focus on qualification and closing techniques")
            elif metric == PerformanceMetric.AVERAGE_DEAL_SIZE:
                if performance_gap < -5000:  # $5k below top performers
                    recommendations.append("Work on upselling and value proposition")
            
        except Exception as e:
            logger.error(f"Error generating metric insights: {e}")
        
        return insights, recommendations


# Global instance
performance_benchmarking_engine = PerformanceBenchmarkingEngine()