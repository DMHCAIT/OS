"""
Business Intelligence Module
Advanced analytics for conversion tracking, ROI measurement, and sales pipeline optimization
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ConversionStage(Enum):
    """Sales pipeline stages"""
    LEAD = "lead"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"

@dataclass
class ConversionMetric:
    """Conversion tracking metric"""
    lead_id: str
    stage: ConversionStage
    timestamp: datetime
    source: str
    value: float
    sales_rep: str
    duration_in_stage: Optional[timedelta] = None

@dataclass
class ROIMetric:
    """ROI measurement data"""
    campaign_id: str
    cost: float
    revenue: float
    conversions: int
    leads: int
    timestamp: datetime
    channel: str
    roi_percentage: float

@dataclass
class SalesPipelineMetric:
    """Sales pipeline optimization data"""
    stage: ConversionStage
    count: int
    value: float
    avg_duration: timedelta
    conversion_rate: float
    timestamp: datetime

@dataclass
class BusinessMetrics:
    """Comprehensive business metrics"""
    timestamp: datetime
    total_revenue: float
    conversion_rate: float
    avg_deal_size: float
    sales_cycle_length: timedelta
    customer_acquisition_cost: float
    lifetime_value: float
    churn_rate: float
    growth_rate: float

@dataclass
class SalesRepPerformance:
    """Individual sales rep performance metrics"""
    rep_id: str
    rep_name: str
    calls_made: int
    leads_generated: int
    deals_closed: int
    revenue: float
    conversion_rate: float
    avg_deal_size: float
    avg_call_duration: timedelta
    sentiment_score: float
    timestamp: datetime

@dataclass
class PipelineAnalysis:
    """Pipeline bottleneck analysis"""
    bottleneck_stage: ConversionStage
    avg_time_in_stage: timedelta
    conversion_drop_rate: float
    recommendations: List[str]
    predicted_impact: float

@dataclass
class PredictiveInsight:
    """AI-powered predictive business insight"""
    insight_type: str
    prediction: str
    confidence: float
    impact_score: float
    recommended_actions: List[str]
    expected_roi: float
    timeframe: str

class BusinessIntelligenceEngine:
    """Advanced business intelligence and analytics engine"""
    
    def __init__(self):
        self.conversion_history = []
        self.roi_history = []
        self.pipeline_history = []
        self.rep_performance_history = []
        self.business_metrics_history = []
        
        # Cached analytics for performance
        self._analytics_cache = {}
        self._cache_expiry = datetime.now()
        self._cache_duration = timedelta(minutes=5)
        
        # Benchmark data for comparisons
        self.industry_benchmarks = {
            'conversion_rate': 0.15,  # 15% industry average
            'avg_deal_size': 10000,   # $10k average
            'sales_cycle_days': 45,   # 45 days average
            'cac_ltv_ratio': 3.0,     # 3:1 LTV:CAC ratio
            'churn_rate': 0.05        # 5% monthly churn
        }

    async def track_conversion(self, metric: ConversionMetric):
        """Track conversion through sales pipeline"""
        self.conversion_history.append(metric)
        
        # Update pipeline stage durations
        await self._update_stage_durations(metric)
        
        logger.info(f"Conversion tracked: {metric.lead_id} -> {metric.stage.value}")

    async def track_roi(self, metric: ROIMetric):
        """Track ROI metrics for campaigns and channels"""
        self.roi_history.append(metric)
        logger.info(f"ROI tracked: {metric.campaign_id} - {metric.roi_percentage:.1f}% ROI")

    async def track_rep_performance(self, metric: SalesRepPerformance):
        """Track individual sales rep performance"""
        self.rep_performance_history.append(metric)
        logger.info(f"Rep performance tracked: {metric.rep_name}")

    async def _update_stage_durations(self, metric: ConversionMetric):
        """Calculate time spent in each pipeline stage"""
        # Find previous stage for this lead
        previous_stages = [
            m for m in self.conversion_history 
            if m.lead_id == metric.lead_id and m.timestamp < metric.timestamp
        ]
        
        if previous_stages:
            last_stage = max(previous_stages, key=lambda x: x.timestamp)
            duration = metric.timestamp - last_stage.timestamp
            metric.duration_in_stage = duration

    async def calculate_conversion_funnel(self, days: int = 30) -> Dict[str, Any]:
        """Calculate conversion funnel analytics"""
        if self._is_cache_valid('conversion_funnel'):
            return self._analytics_cache['conversion_funnel']
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_conversions = [
            c for c in self.conversion_history 
            if c.timestamp >= cutoff
        ]
        
        # Count leads at each stage
        stage_counts = defaultdict(int)
        stage_values = defaultdict(float)
        
        # Get unique leads per stage (latest stage for each lead)
        lead_stages = {}
        for conv in sorted(recent_conversions, key=lambda x: x.timestamp):
            lead_stages[conv.lead_id] = conv
        
        for conv in lead_stages.values():
            stage_counts[conv.stage.value] += 1
            stage_values[conv.stage.value] += conv.value
        
        # Calculate conversion rates between stages
        stages = [stage.value for stage in ConversionStage]
        conversion_rates = {}
        
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if stage_counts[current_stage] > 0:
                rate = stage_counts[next_stage] / stage_counts[current_stage]
                conversion_rates[f"{current_stage}_to_{next_stage}"] = rate
        
        funnel_data = {
            'stage_counts': dict(stage_counts),
            'stage_values': dict(stage_values),
            'conversion_rates': conversion_rates,
            'total_leads': len(lead_stages),
            'won_deals': stage_counts['closed_won'],
            'overall_conversion_rate': (
                stage_counts['closed_won'] / len(lead_stages) 
                if len(lead_stages) > 0 else 0
            )
        }
        
        self._analytics_cache['conversion_funnel'] = funnel_data
        return funnel_data

    async def calculate_roi_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Calculate ROI analysis across campaigns and channels"""
        if self._is_cache_valid('roi_analysis'):
            return self._analytics_cache['roi_analysis']
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_roi = [r for r in self.roi_history if r.timestamp >= cutoff]
        
        if not recent_roi:
            return {'error': 'No ROI data available'}
        
        # Aggregate by channel
        channel_performance = defaultdict(lambda: {
            'cost': 0, 'revenue': 0, 'conversions': 0, 'leads': 0
        })
        
        for roi in recent_roi:
            channel = roi.channel
            channel_performance[channel]['cost'] += roi.cost
            channel_performance[channel]['revenue'] += roi.revenue
            channel_performance[channel]['conversions'] += roi.conversions
            channel_performance[channel]['leads'] += roi.leads
        
        # Calculate channel ROI
        channel_roi = {}
        for channel, data in channel_performance.items():
            roi_pct = ((data['revenue'] - data['cost']) / data['cost'] * 100) if data['cost'] > 0 else 0
            conversion_rate = data['conversions'] / data['leads'] if data['leads'] > 0 else 0
            cost_per_conversion = data['cost'] / data['conversions'] if data['conversions'] > 0 else float('inf')
            
            channel_roi[channel] = {
                'roi_percentage': roi_pct,
                'conversion_rate': conversion_rate,
                'cost_per_conversion': cost_per_conversion,
                'total_revenue': data['revenue'],
                'total_cost': data['cost'],
                'total_conversions': data['conversions']
            }
        
        # Overall metrics
        total_cost = sum(r.cost for r in recent_roi)
        total_revenue = sum(r.revenue for r in recent_roi)
        total_conversions = sum(r.conversions for r in recent_roi)
        total_leads = sum(r.leads for r in recent_roi)
        
        overall_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        roi_analysis = {
            'overall_roi_percentage': overall_roi,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_conversions': total_conversions,
            'total_leads': total_leads,
            'channel_performance': dict(channel_roi),
            'best_performing_channel': max(channel_roi.keys(), key=lambda x: channel_roi[x]['roi_percentage']) if channel_roi else None,
            'most_cost_effective': min(channel_roi.keys(), key=lambda x: channel_roi[x]['cost_per_conversion']) if channel_roi else None
        }
        
        self._analytics_cache['roi_analysis'] = roi_analysis
        return roi_analysis

    async def analyze_sales_pipeline(self) -> Dict[str, Any]:
        """Analyze sales pipeline for bottlenecks and optimization"""
        if self._is_cache_valid('pipeline_analysis'):
            return self._analytics_cache['pipeline_analysis']
        
        # Calculate average time in each stage
        stage_durations = defaultdict(list)
        stage_conversions = defaultdict(lambda: {'entered': 0, 'converted': 0})
        
        for conv in self.conversion_history:
            if conv.duration_in_stage:
                stage_durations[conv.stage.value].append(conv.duration_in_stage.total_seconds() / 86400)  # Convert to days
        
        # Identify bottlenecks
        avg_durations = {}
        for stage, durations in stage_durations.items():
            avg_durations[stage] = np.mean(durations) if durations else 0
        
        # Find the stage with longest average duration
        bottleneck_stage = max(avg_durations.keys(), key=lambda x: avg_durations[x]) if avg_durations else None
        
        # Generate recommendations based on bottlenecks
        recommendations = await self._generate_pipeline_recommendations(avg_durations)
        
        pipeline_analysis = {
            'average_stage_durations_days': avg_durations,
            'bottleneck_stage': bottleneck_stage,
            'bottleneck_duration_days': avg_durations.get(bottleneck_stage, 0),
            'recommendations': recommendations,
            'total_pipeline_length_days': sum(avg_durations.values()),
            'vs_industry_benchmark': {
                'current_cycle': sum(avg_durations.values()),
                'industry_average': self.industry_benchmarks['sales_cycle_days'],
                'performance_vs_benchmark': 'above' if sum(avg_durations.values()) < self.industry_benchmarks['sales_cycle_days'] else 'below'
            }
        }
        
        self._analytics_cache['pipeline_analysis'] = pipeline_analysis
        return pipeline_analysis

    async def _generate_pipeline_recommendations(self, avg_durations: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on pipeline analysis"""
        recommendations = []
        
        # Check for specific bottlenecks
        if avg_durations.get('qualified', 0) > 7:
            recommendations.append("Consider automating lead qualification with ML scoring to reduce qualification time")
        
        if avg_durations.get('proposal', 0) > 14:
            recommendations.append("Implement proposal templates and automation to speed up proposal generation")
        
        if avg_durations.get('negotiation', 0) > 21:
            recommendations.append("Provide negotiation training and decision-making frameworks to sales reps")
        
        # General recommendations
        total_cycle = sum(avg_durations.values())
        if total_cycle > self.industry_benchmarks['sales_cycle_days'] * 1.5:
            recommendations.append("Sales cycle is significantly longer than industry average - consider process optimization")
        
        if not recommendations:
            recommendations.append("Pipeline performance looks good - maintain current processes")
        
        return recommendations

    async def calculate_sales_rep_rankings(self, days: int = 30) -> Dict[str, Any]:
        """Calculate sales rep performance rankings"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_performance = [
            p for p in self.rep_performance_history 
            if p.timestamp >= cutoff
        ]
        
        if not recent_performance:
            return {'error': 'No sales rep performance data available'}
        
        # Group by rep and calculate aggregates
        rep_stats = defaultdict(lambda: {
            'calls': 0, 'leads': 0, 'deals': 0, 'revenue': 0,
            'conversions': [], 'deal_sizes': [], 'call_durations': [],
            'sentiments': [], 'name': ''
        })
        
        for perf in recent_performance:
            rep_id = perf.rep_id
            rep_stats[rep_id]['name'] = perf.rep_name
            rep_stats[rep_id]['calls'] += perf.calls_made
            rep_stats[rep_id]['leads'] += perf.leads_generated
            rep_stats[rep_id]['deals'] += perf.deals_closed
            rep_stats[rep_id]['revenue'] += perf.revenue
            rep_stats[rep_id]['conversions'].append(perf.conversion_rate)
            rep_stats[rep_id]['deal_sizes'].append(perf.avg_deal_size)
            rep_stats[rep_id]['sentiments'].append(perf.sentiment_score)
        
        # Calculate final metrics and rankings
        rep_rankings = []
        for rep_id, stats in rep_stats.items():
            conversion_rate = np.mean(stats['conversions']) if stats['conversions'] else 0
            avg_deal_size = np.mean(stats['deal_sizes']) if stats['deal_sizes'] else 0
            avg_sentiment = np.mean(stats['sentiments']) if stats['sentiments'] else 0
            
            # Calculate performance score
            performance_score = (
                (conversion_rate * 0.4) +
                (min(avg_deal_size / 10000, 1.0) * 0.3) +  # Normalize deal size
                (avg_sentiment * 0.2) +
                (min(stats['deals'] / 10, 1.0) * 0.1)  # Normalize deal count
            ) * 100
            
            rep_rankings.append({
                'rep_id': rep_id,
                'rep_name': stats['name'],
                'calls_made': stats['calls'],
                'leads_generated': stats['leads'],
                'deals_closed': stats['deals'],
                'total_revenue': stats['revenue'],
                'conversion_rate': conversion_rate,
                'avg_deal_size': avg_deal_size,
                'avg_sentiment': avg_sentiment,
                'performance_score': performance_score
            })
        
        # Sort by performance score
        rep_rankings.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Add rankings
        for i, rep in enumerate(rep_rankings):
            rep['rank'] = i + 1
        
        return {
            'rep_rankings': rep_rankings,
            'top_performer': rep_rankings[0] if rep_rankings else None,
            'team_averages': {
                'avg_conversion_rate': np.mean([r['conversion_rate'] for r in rep_rankings]) if rep_rankings else 0,
                'avg_deal_size': np.mean([r['avg_deal_size'] for r in rep_rankings]) if rep_rankings else 0,
                'total_team_revenue': sum(r['total_revenue'] for r in rep_rankings),
                'total_team_deals': sum(r['deals_closed'] for r in rep_rankings)
            }
        }

    async def generate_predictive_insights(self) -> List[PredictiveInsight]:
        """Generate AI-powered predictive business insights"""
        insights = []
        
        # Analyze conversion trends
        conversion_trend = await self._analyze_conversion_trend()
        if conversion_trend['trend'] == 'declining':
            insights.append(PredictiveInsight(
                insight_type='conversion_trend',
                prediction=f"Conversion rate declining by {conversion_trend['decline_rate']:.1%} per week",
                confidence=0.8,
                impact_score=0.9,
                recommended_actions=[
                    "Review lead qualification criteria",
                    "Enhance sales training programs",
                    "Analyze lost deal reasons"
                ],
                expected_roi=conversion_trend['potential_recovery'] * 15000,  # Estimated revenue per conversion
                timeframe="2-4 weeks"
            ))
        
        # Analyze seasonal patterns
        seasonal_pattern = await self._analyze_seasonal_patterns()
        if seasonal_pattern['has_pattern']:
            insights.append(PredictiveInsight(
                insight_type='seasonal_forecast',
                prediction=f"Expected {seasonal_pattern['expected_change']:.1%} change in next month based on seasonal patterns",
                confidence=0.7,
                impact_score=0.6,
                recommended_actions=[
                    "Adjust marketing spend for seasonal trends",
                    "Prepare sales team for volume changes",
                    "Optimize inventory/resource planning"
                ],
                expected_roi=seasonal_pattern['revenue_impact'],
                timeframe="1 month"
            ))
        
        # Pipeline capacity analysis
        pipeline_capacity = await self._analyze_pipeline_capacity()
        if pipeline_capacity['over_capacity']:
            insights.append(PredictiveInsight(
                insight_type='capacity_warning',
                prediction=f"Pipeline approaching capacity limit - potential {pipeline_capacity['overflow']:.0f} leads may be under-served",
                confidence=0.85,
                impact_score=0.8,
                recommended_actions=[
                    "Hire additional sales staff",
                    "Implement lead prioritization",
                    "Consider automated qualification"
                ],
                expected_roi=pipeline_capacity['lost_revenue_risk'],
                timeframe="immediate"
            ))
        
        return insights

    async def _analyze_conversion_trend(self) -> Dict[str, Any]:
        """Analyze conversion rate trends"""
        # Get last 8 weeks of data
        weekly_conversions = defaultdict(lambda: {'leads': 0, 'conversions': 0})
        
        for conv in self.conversion_history:
            week = conv.timestamp.strftime('%Y-W%U')
            if conv.stage == ConversionStage.LEAD:
                weekly_conversions[week]['leads'] += 1
            elif conv.stage == ConversionStage.CLOSED_WON:
                weekly_conversions[week]['conversions'] += 1
        
        # Calculate weekly conversion rates
        weeks = sorted(weekly_conversions.keys())[-8:]  # Last 8 weeks
        rates = []
        
        for week in weeks:
            data = weekly_conversions[week]
            rate = data['conversions'] / data['leads'] if data['leads'] > 0 else 0
            rates.append(rate)
        
        if len(rates) >= 4:
            # Calculate trend using linear regression
            x = np.arange(len(rates))
            slope = np.polyfit(x, rates, 1)[0]
            
            return {
                'trend': 'declining' if slope < -0.01 else 'stable' if slope < 0.01 else 'improving',
                'decline_rate': abs(slope),
                'potential_recovery': 0.05 if slope < -0.01 else 0
            }
        
        return {'trend': 'insufficient_data', 'decline_rate': 0, 'potential_recovery': 0}

    async def _analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """Analyze seasonal business patterns"""
        # Group data by month
        monthly_revenue = defaultdict(float)
        
        for conv in self.conversion_history:
            if conv.stage == ConversionStage.CLOSED_WON:
                month = conv.timestamp.month
                monthly_revenue[month] += conv.value
        
        if len(monthly_revenue) >= 6:  # Need at least 6 months of data
            current_month = datetime.now().month
            avg_revenue = np.mean(list(monthly_revenue.values()))
            
            # Predict next month based on historical pattern
            next_month = (current_month % 12) + 1
            historical_next = monthly_revenue.get(next_month, avg_revenue)
            
            expected_change = (historical_next - avg_revenue) / avg_revenue
            
            return {
                'has_pattern': True,
                'expected_change': expected_change,
                'revenue_impact': historical_next - avg_revenue
            }
        
        return {'has_pattern': False, 'expected_change': 0, 'revenue_impact': 0}

    async def _analyze_pipeline_capacity(self) -> Dict[str, Any]:
        """Analyze pipeline capacity and potential bottlenecks"""
        # Calculate current pipeline load
        current_leads = len([
            c for c in self.conversion_history 
            if c.stage not in [ConversionStage.CLOSED_WON, ConversionStage.CLOSED_LOST]
        ])
        
        # Estimate capacity based on recent performance
        recent_performance = self.rep_performance_history[-10:]  # Last 10 data points
        if recent_performance:
            avg_calls_per_rep = np.mean([p.calls_made for p in recent_performance])
            total_capacity = avg_calls_per_rep * len(set(p.rep_id for p in recent_performance)) * 0.8  # 80% utilization
            
            overflow = max(0, current_leads - total_capacity)
            
            return {
                'over_capacity': overflow > 0,
                'overflow': overflow,
                'lost_revenue_risk': overflow * 5000  # Estimated revenue per lead
            }
        
        return {'over_capacity': False, 'overflow': 0, 'lost_revenue_risk': 0}

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analytics are still valid"""
        return (
            cache_key in self._analytics_cache and
            datetime.now() < self._cache_expiry
        )

    async def get_comprehensive_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data in one call"""
        conversion_funnel = await self.calculate_conversion_funnel()
        roi_analysis = await self.calculate_roi_analysis()
        pipeline_analysis = await self.analyze_sales_pipeline()
        rep_rankings = await self.calculate_sales_rep_rankings()
        predictive_insights = await self.generate_predictive_insights()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'conversion_funnel': conversion_funnel,
            'roi_analysis': roi_analysis,
            'pipeline_analysis': pipeline_analysis,
            'sales_rep_rankings': rep_rankings,
            'predictive_insights': [asdict(insight) for insight in predictive_insights],
            'business_health_score': await self._calculate_business_health_score(),
            'key_metrics': await self._get_key_metrics()
        }

    async def _calculate_business_health_score(self) -> float:
        """Calculate overall business health score (0-100)"""
        scores = []
        
        # Conversion rate score
        conversion_data = await self.calculate_conversion_funnel()
        conversion_rate = conversion_data.get('overall_conversion_rate', 0)
        benchmark_conversion = self.industry_benchmarks['conversion_rate']
        conversion_score = min(100, (conversion_rate / benchmark_conversion) * 100)
        scores.append(conversion_score)
        
        # ROI score
        roi_data = await self.calculate_roi_analysis()
        if 'overall_roi_percentage' in roi_data:
            roi_score = min(100, max(0, roi_data['overall_roi_percentage']))
            scores.append(roi_score)
        
        # Pipeline efficiency score
        pipeline_data = await self.analyze_sales_pipeline()
        cycle_length = pipeline_data.get('total_pipeline_length_days', 0)
        benchmark_cycle = self.industry_benchmarks['sales_cycle_days']
        pipeline_score = min(100, (benchmark_cycle / max(cycle_length, 1)) * 100)
        scores.append(pipeline_score)
        
        return np.mean(scores) if scores else 50.0

    async def _get_key_metrics(self) -> Dict[str, Any]:
        """Get key business metrics summary"""
        # Calculate from recent data
        recent_conversions = [
            c for c in self.conversion_history 
            if c.timestamp >= datetime.now() - timedelta(days=30)
        ]
        
        won_deals = [c for c in recent_conversions if c.stage == ConversionStage.CLOSED_WON]
        total_leads = len(set(c.lead_id for c in recent_conversions))
        
        return {
            'total_revenue_30d': sum(c.value for c in won_deals),
            'deals_closed_30d': len(won_deals),
            'conversion_rate_30d': len(won_deals) / total_leads if total_leads > 0 else 0,
            'avg_deal_size_30d': np.mean([c.value for c in won_deals]) if won_deals else 0,
            'total_leads_30d': total_leads,
            'pipeline_value': sum(
                c.value for c in recent_conversions 
                if c.stage not in [ConversionStage.CLOSED_WON, ConversionStage.CLOSED_LOST]
            )
        }


# Global business intelligence engine instance
business_intelligence = BusinessIntelligenceEngine()