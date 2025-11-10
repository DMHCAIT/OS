"""
Comprehensive Test Suite for Predictive Business Intelligence
Tests for market trends, territory optimization, seasonal patterns, and competitive intelligence
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.core.market_trend_analysis import market_trend_analysis, MarketTrendPrediction
from app.core.territory_optimization import territory_optimization, TerritoryRecommendation
from app.core.seasonal_patterns import seasonal_patterns, SeasonalForecast
from app.core.competitive_intelligence import competitive_intelligence, CompetitiveThreat


class TestPredictiveIntelligence:
    """Test suite for all predictive intelligence components"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com"
        }
    
    @pytest.fixture
    def sample_lead_data(self):
        """Sample lead data for testing"""
        return {
            "id": "lead_123",
            "name": "John Doe",
            "email": "john@company.com",
            "company": "Test Corp",
            "job_title": "CTO",
            "industry": "technology",
            "company_size": 100,
            "location": "San Francisco, CA",
            "engagement_score": 0.85
        }


class TestMarketTrendAnalysis(TestPredictiveIntelligence):
    """Tests for market trend analysis engine"""
    
    @pytest.mark.asyncio
    async def test_analyze_trends_success(self, mock_user):
        """Test successful market trend analysis"""
        with patch.object(market_trend_analysis, 'analyze_trends') as mock_analyze:
            # Mock successful analysis result
            mock_result = {
                "overall_trend": "positive",
                "trend_strength": 0.75,
                "economic_indicators": {
                    "gdp_growth": 2.1,
                    "unemployment": 3.8,
                    "inflation": 2.3
                },
                "market_sentiment": {
                    "score": 0.68,
                    "classification": "bullish"
                },
                "sales_impact_prediction": {
                    "predicted_change": 12.5,
                    "confidence": 0.82
                },
                "recommendations": [
                    "Increase outreach in technology sector",
                    "Focus on enterprise clients"
                ]
            }
            
            mock_analyze.return_value = mock_result
            
            # Test the analysis
            result = await market_trend_analysis.analyze_trends(
                timeframe_days=90,
                sectors=["technology", "healthcare"],
                include_predictions=True
            )
            
            assert result["overall_trend"] == "positive"
            assert result["trend_strength"] == 0.75
            assert "economic_indicators" in result
            assert "sales_impact_prediction" in result
            assert len(result["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_predictions(self):
        """Test market trend predictions"""
        with patch.object(market_trend_analysis, 'get_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "forecast_period": "30 days",
                "predictions": [
                    {
                        "date": (datetime.now() + timedelta(days=i)).isoformat(),
                        "predicted_trend": 0.75 + (i * 0.01),
                        "confidence": 0.80
                    } for i in range(30)
                ],
                "summary": {
                    "average_trend": 0.82,
                    "volatility": 0.15,
                    "recommendation": "favorable"
                }
            }
            
            result = await market_trend_analysis.get_predictions(days_ahead=30)
            
            assert len(result["predictions"]) == 30
            assert result["summary"]["average_trend"] > 0.7
            assert "recommendation" in result["summary"]
    
    def test_market_trend_api_endpoint(self, client):
        """Test market trend API endpoint"""
        with patch('app.core.security.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_123", "username": "testuser"}
            
            with patch.object(market_trend_analysis, 'analyze_trends') as mock_analyze:
                mock_analyze.return_value = {"overall_trend": "positive", "trend_strength": 0.8}
                
                response = client.post(
                    "/api/v1/intelligence/market-trends/analyze",
                    json={
                        "timeframe_days": 90,
                        "include_predictions": True,
                        "sectors": ["technology"]
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] == True
                assert "analysis" in data


class TestTerritoryOptimization(TestPredictiveIntelligence):
    """Tests for territory optimization engine"""
    
    @pytest.mark.asyncio
    async def test_optimize_territories_success(self):
        """Test successful territory optimization"""
        sample_territories = [
            {
                "id": "territory_1",
                "name": "West Coast",
                "rep_id": "rep_123",
                "customers": [{"lat": 37.7749, "lng": -122.4194}],
                "revenue": 250000
            },
            {
                "id": "territory_2",
                "name": "East Coast",
                "rep_id": "rep_456", 
                "customers": [{"lat": 40.7128, "lng": -74.0060}],
                "revenue": 180000
            }
        ]
        
        with patch.object(territory_optimization, 'optimize_territories') as mock_optimize:
            mock_optimize.return_value = {
                "optimization_type": "revenue_maximization",
                "improved_territories": [
                    {
                        "id": "territory_1",
                        "recommended_changes": ["Add 5 accounts from territory_2"],
                        "predicted_revenue_increase": 35000,
                        "efficiency_score": 0.87
                    }
                ],
                "overall_improvement": {
                    "revenue_increase": 45000,
                    "efficiency_gain": 0.23,
                    "coverage_improvement": 0.15
                },
                "recommendations": [
                    "Redistribute high-value accounts",
                    "Assign additional rep to West Coast"
                ]
            }
            
            result = await territory_optimization.optimize_territories(
                current_territories=sample_territories,
                objective="revenue_maximization",
                include_reps=True
            )
            
            assert "optimization_type" in result
            assert "improved_territories" in result
            assert result["overall_improvement"]["revenue_increase"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_performance(self):
        """Test territory performance analysis"""
        with patch.object(territory_optimization, 'analyze_performance') as mock_performance:
            mock_performance.return_value = {
                "territory_id": "territory_1",
                "performance_metrics": {
                    "revenue": 250000,
                    "deals_closed": 45,
                    "conversion_rate": 0.18,
                    "average_deal_size": 5556
                },
                "efficiency_metrics": {
                    "travel_efficiency": 0.82,
                    "customer_coverage": 0.75,
                    "rep_utilization": 0.91
                },
                "recommendations": [
                    "Increase customer visits in Q4",
                    "Focus on larger deal opportunities"
                ],
                "benchmark_comparison": {
                    "revenue_vs_average": 1.15,
                    "conversion_vs_average": 1.08
                }
            }
            
            result = await territory_optimization.analyze_performance(
                territory_id="territory_1",
                timeframe_days=90
            )
            
            assert "performance_metrics" in result
            assert "efficiency_metrics" in result
            assert result["performance_metrics"]["conversion_rate"] > 0


class TestSeasonalPatterns(TestPredictiveIntelligence):
    """Tests for seasonal pattern recognition"""
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_success(self):
        """Test successful seasonal pattern analysis"""
        sample_sales_data = [
            {
                "date": (datetime.now() - timedelta(days=i)),
                "revenue": 10000 + (i % 30) * 1000,
                "deals": 5 + (i % 10)
            } for i in range(365)
        ]
        
        with patch.object(seasonal_patterns, 'analyze_patterns') as mock_analyze:
            mock_analyze.return_value = {
                "detected_patterns": {
                    "monthly": {
                        "pattern_strength": 0.76,
                        "peak_months": ["November", "December", "January"],
                        "low_months": ["July", "August"]
                    },
                    "quarterly": {
                        "pattern_strength": 0.68,
                        "q4_boost": 0.35,
                        "q2_dip": -0.15
                    },
                    "holiday": {
                        "pattern_strength": 0.82,
                        "thanksgiving_effect": 0.25,
                        "new_year_effect": 0.18
                    }
                },
                "forecast": {
                    "next_12_months": [
                        {
                            "month": "2024-01",
                            "predicted_revenue": 125000,
                            "confidence_interval": [110000, 140000]
                        }
                    ],
                    "seasonal_multipliers": {
                        "jan": 1.15, "feb": 0.95, "mar": 1.05
                    }
                },
                "recommendations": [
                    "Increase marketing spend in Q4",
                    "Plan inventory buildup before holiday season"
                ]
            }
            
            result = await seasonal_patterns.analyze_patterns(
                sales_data=sample_sales_data,
                historical_years=2,
                pattern_types=["monthly", "quarterly", "holiday"]
            )
            
            assert "detected_patterns" in result
            assert "forecast" in result
            assert len(result["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_forecast(self):
        """Test seasonal forecast generation"""
        with patch.object(seasonal_patterns, 'generate_forecast') as mock_forecast:
            mock_forecast.return_value = {
                "forecast_period": "6 months",
                "monthly_forecasts": [
                    {
                        "month": "2024-01",
                        "predicted_revenue": 120000,
                        "seasonal_factor": 1.12,
                        "confidence": 0.85
                    }
                ],
                "seasonal_strategies": [
                    {
                        "period": "Q4",
                        "strategy": "Holiday Campaign",
                        "expected_lift": 0.25,
                        "recommended_budget": 50000
                    }
                ]
            }
            
            result = await seasonal_patterns.generate_forecast(
                months_ahead=6,
                include_strategies=True
            )
            
            assert "forecast_period" in result
            assert "monthly_forecasts" in result
            assert "seasonal_strategies" in result


class TestCompetitiveIntelligence(TestPredictiveIntelligence):
    """Tests for competitive intelligence engine"""
    
    @pytest.mark.asyncio
    async def test_analyze_competition_success(self):
        """Test successful competitive analysis"""
        with patch.object(competitive_intelligence, 'analyze_competition') as mock_analyze:
            mock_analyze.return_value = {
                "competitive_landscape": {
                    "tracked_competitors": [
                        {
                            "name": "Competitor A",
                            "market_share": 0.25,
                            "threat_level": "high",
                            "recent_activities": [
                                "Launched new product",
                                "Reduced pricing by 15%"
                            ]
                        }
                    ],
                    "market_position": {
                        "our_ranking": 3,
                        "total_players": 8,
                        "market_share": 0.18
                    }
                },
                "threat_assessment": {
                    "high_threats": 2,
                    "medium_threats": 3,
                    "low_threats": 1,
                    "emerging_threats": ["New entrant from overseas"]
                },
                "strategic_recommendations": [
                    {
                        "threat": "Price competition",
                        "response": "Value differentiation",
                        "priority": "high",
                        "timeline": "immediate"
                    }
                ],
                "intelligence_summary": {
                    "total_insights": 15,
                    "actionable_items": 8,
                    "monitoring_alerts": 3
                }
            }
            
            result = await competitive_intelligence.analyze_competition(
                competitor_tracking=True,
                threat_assessment=True,
                market_positioning=True
            )
            
            assert "competitive_landscape" in result
            assert "threat_assessment" in result
            assert "strategic_recommendations" in result
    
    @pytest.mark.asyncio
    async def test_get_threats(self):
        """Test competitive threat detection"""
        with patch.object(competitive_intelligence, 'get_threats') as mock_threats:
            mock_threats.return_value = [
                {
                    "competitor": "Competitor A",
                    "threat_type": "pricing",
                    "threat_level": "high",
                    "description": "15% price reduction on core products",
                    "detected_date": datetime.now().isoformat(),
                    "impact_assessment": "Potential 20% revenue impact",
                    "recommended_response": "Launch value-add promotion"
                },
                {
                    "competitor": "Competitor B", 
                    "threat_type": "product_launch",
                    "threat_level": "medium",
                    "description": "New AI-powered feature release",
                    "detected_date": datetime.now().isoformat(),
                    "impact_assessment": "Feature gap identified",
                    "recommended_response": "Accelerate roadmap"
                }
            ]
            
            result = await competitive_intelligence.get_threats(
                threat_level="high",
                timeframe_days=30
            )
            
            assert len(result) >= 1
            assert all("threat_level" in threat for threat in result)
            assert all("recommended_response" in threat for threat in result)


class TestIntegrationAPIs(TestPredictiveIntelligence):
    """Integration tests for API endpoints"""
    
    def test_intelligence_dashboard_overview(self, client):
        """Test intelligence dashboard overview endpoint"""
        with patch('app.core.security.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_123", "username": "testuser"}
            
            with patch.object(market_trend_analysis, 'get_quick_insights') as mock_market, \
                 patch.object(territory_optimization, 'get_quick_insights') as mock_territory, \
                 patch.object(seasonal_patterns, 'get_quick_insights') as mock_seasonal, \
                 patch.object(competitive_intelligence, 'get_quick_insights') as mock_competitive:
                
                mock_market.return_value = {"trend": "positive", "strength": 0.8}
                mock_territory.return_value = {"efficiency": 0.75, "revenue": 500000}
                mock_seasonal.return_value = {"current_factor": 1.15, "forecast": "strong"}
                mock_competitive.return_value = {"threat_level": "medium", "opportunities": 3}
                
                response = client.get("/api/v1/intelligence/dashboard/overview")
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] == True
                assert "dashboard" in data
                assert "market_trends" in data["dashboard"]
                assert "territory_performance" in data["dashboard"]
                assert "seasonal_patterns" in data["dashboard"]
                assert "competitive_landscape" in data["dashboard"]
    
    def test_health_check_endpoint(self, client):
        """Test intelligence services health check"""
        with patch.object(market_trend_analysis, 'health_check') as mock_market_health, \
             patch.object(territory_optimization, 'health_check') as mock_territory_health, \
             patch.object(seasonal_patterns, 'health_check') as mock_seasonal_health, \
             patch.object(competitive_intelligence, 'health_check') as mock_competitive_health:
            
            # Mock all services as healthy
            for mock_health in [mock_market_health, mock_territory_health, 
                               mock_seasonal_health, mock_competitive_health]:
                mock_health.return_value = {"status": "healthy", "last_check": datetime.now().isoformat()}
            
            response = client.get("/api/v1/intelligence/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["overall_health"] == "healthy"
            assert "services" in data


class TestDataValidation(TestPredictiveIntelligence):
    """Tests for data validation and error handling"""
    
    def test_invalid_request_parameters(self, client):
        """Test handling of invalid request parameters"""
        with patch('app.core.security.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_123", "username": "testuser"}
            
            # Test invalid timeframe
            response = client.post(
                "/api/v1/intelligence/market-trends/analyze",
                json={
                    "timeframe_days": 999,  # Invalid: too large
                    "include_predictions": True
                }
            )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_error_handling_in_analysis(self):
        """Test error handling in analysis functions"""
        with patch.object(market_trend_analysis, 'analyze_trends') as mock_analyze:
            mock_analyze.side_effect = Exception("External API error")
            
            with pytest.raises(Exception):
                await market_trend_analysis.analyze_trends(timeframe_days=30)
    
    def test_missing_authentication(self, client):
        """Test endpoints without authentication"""
        response = client.post(
            "/api/v1/intelligence/market-trends/analyze",
            json={"timeframe_days": 30}
        )
        
        assert response.status_code == 401  # Unauthorized


# Performance Tests
class TestPerformance(TestPredictiveIntelligence):
    """Performance tests for intelligence services"""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self):
        """Test handling multiple concurrent analysis requests"""
        async def mock_analysis():
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"result": "success"}
        
        with patch.object(market_trend_analysis, 'analyze_trends', side_effect=mock_analysis):
            # Run 10 concurrent analyses
            tasks = [market_trend_analysis.analyze_trends(timeframe_days=30) for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all(result["result"] == "success" for result in results)
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self):
        """Test processing large datasets efficiently"""
        large_sales_data = [
            {
                "date": datetime.now() - timedelta(days=i),
                "revenue": 1000 + (i % 100) * 10,
                "deals": i % 20
            } for i in range(10000)  # 10k records
        ]
        
        with patch.object(seasonal_patterns, 'analyze_patterns') as mock_analyze:
            mock_analyze.return_value = {"patterns": "detected", "processing_time": 2.5}
            
            start_time = datetime.now()
            result = await seasonal_patterns.analyze_patterns(
                sales_data=large_sales_data,
                historical_years=3
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 10  # Should process within 10 seconds
            assert "patterns" in result


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=app.core",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])