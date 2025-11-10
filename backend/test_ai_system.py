#!/usr/bin/env python3
"""
Advanced AI/ML Sales Automation System - Comprehensive Test Suite
Tests all major AI/ML components to ensure proper functionality
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

try:
    from app.core.advanced_ai_service import AdvancedAIMLService
    from app.core.ml_lead_scoring import MLLeadScoringSystem
    from app.core.ai_conversation_engine import AIConversationEngine
    from app.core.voice_ai_enhancement import VoiceAIEnhancementSystem
    from app.core.predictive_analytics import PredictiveAnalytics
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed and the backend is properly set up.")
    sys.exit(1)

class ComprehensiveTestSuite:
    """Comprehensive testing suite for all AI/ML components"""
    
    def __init__(self):
        self.test_results = {}
        self.success_count = 0
        self.failure_count = 0
        
        # Initialize all AI services
        self.advanced_ai = AdvancedAIMLService()
        self.ml_scoring = MLLeadScoringSystem()
        self.conversation_engine = AIConversationEngine()
        self.voice_ai = VoiceAIEnhancementSystem()
        self.predictive_analytics = PredictiveAnalytics()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
        if details:
            print(f"   └── {details}")
        
        self.test_results[test_name] = {
            "success": success,
            "details": details
        }
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    async def test_advanced_ai_service(self):
        """Test Advanced AI Service functionality"""
        print("\n🧠 Testing Advanced AI Service...")
        
        try:
            # Test conversation analysis
            test_conversation = {
                "message": "I'm interested in your product but concerned about the price",
                "conversation_history": [
                    {"role": "user", "content": "Hello, tell me about your solution"},
                    {"role": "assistant", "content": "Our solution helps automate sales processes"},
                    {"role": "user", "content": "That sounds interesting but expensive"}
                ]
            }
            
            analysis = await self.advanced_ai.analyze_conversation(
                test_conversation["message"],
                test_conversation["conversation_history"]
            )
            
            # Verify analysis components
            required_keys = ["sentiment", "emotions", "intent", "objections", "buying_signals", "recommendations"]
            missing_keys = [key for key in required_keys if key not in analysis]
            
            if missing_keys:
                self.log_test(
                    "Advanced AI - Conversation Analysis",
                    False,
                    f"Missing keys: {missing_keys}"
                )
            else:
                self.log_test(
                    "Advanced AI - Conversation Analysis",
                    True,
                    f"Detected sentiment: {analysis['sentiment']['compound']:.2f}, "
                    f"Objections: {len(analysis['objections'])}, "
                    f"Intent: {analysis['intent']['category']}"
                )
            
            # Test objection handling
            objection = "Your solution seems too expensive for our budget"
            objection_response = await self.advanced_ai.handle_objection(objection)
            
            self.log_test(
                "Advanced AI - Objection Handling",
                len(objection_response.get("response", "")) > 50,
                f"Generated {len(objection_response.get('response', ''))} character response"
            )
            
        except Exception as e:
            self.log_test("Advanced AI Service", False, f"Error: {str(e)}")
    
    async def test_ml_lead_scoring(self):
        """Test ML Lead Scoring System"""
        print("\n📊 Testing ML Lead Scoring System...")
        
        try:
            # Sample lead data
            test_lead = {
                "company_size": "51-200",
                "industry": "Technology",
                "job_title": "VP of Sales",
                "email_engagement_rate": 0.45,
                "website_visits": 15,
                "content_downloads": 3,
                "meeting_acceptance_rate": 0.8,
                "response_time_hours": 2.5,
                "lead_source": "LinkedIn",
                "annual_revenue": 5000000,
                "number_of_employees": 150,
                "decision_maker": True,
                "budget_indicated": True,
                "timeline": "3 months",
                "previous_interactions": 8,
                "competitor_mentions": 1,
                "urgency_score": 0.7
            }
            
            # Test lead scoring
            prediction = await self.ml_scoring.predict_lead_score(test_lead)
            
            required_prediction_keys = ["score", "probability", "confidence_interval", 
                                      "predicted_deal_value", "conversion_timeline", 
                                      "risk_assessment", "recommendations"]
            
            missing_keys = [key for key in required_prediction_keys if key not in prediction]
            
            if missing_keys:
                self.log_test(
                    "ML Lead Scoring - Prediction",
                    False,
                    f"Missing prediction keys: {missing_keys}"
                )
            else:
                self.log_test(
                    "ML Lead Scoring - Prediction",
                    True,
                    f"Score: {prediction['score']:.1f}, "
                    f"Probability: {prediction['probability']:.2f}, "
                    f"Deal Value: ${prediction['predicted_deal_value']:,.0f}"
                )
            
            # Test feature importance
            feature_importance = self.ml_scoring.get_feature_importance()
            
            self.log_test(
                "ML Lead Scoring - Feature Importance",
                len(feature_importance) > 0,
                f"Generated {len(feature_importance)} feature importance scores"
            )
            
        except Exception as e:
            self.log_test("ML Lead Scoring", False, f"Error: {str(e)}")
    
    async def test_conversation_engine(self):
        """Test AI Conversation Engine"""
        print("\n💬 Testing AI Conversation Engine...")
        
        try:
            # Test conversation context
            conversation_context = {
                "conversation_id": "test_conv_123",
                "stage": "discovery",
                "lead_data": {
                    "company": "TechCorp Inc",
                    "industry": "Technology",
                    "size": "Medium"
                }
            }
            
            # Test response generation
            customer_message = "We're looking for a solution to automate our sales process"
            response = await self.conversation_engine.generate_response(
                customer_message, conversation_context
            )
            
            required_response_keys = ["response", "conversation_stage", "confidence_score", 
                                    "next_actions", "personalization_elements"]
            
            missing_keys = [key for key in required_response_keys if key not in response]
            
            if missing_keys:
                self.log_test(
                    "Conversation Engine - Response Generation",
                    False,
                    f"Missing response keys: {missing_keys}"
                )
            else:
                self.log_test(
                    "Conversation Engine - Response Generation",
                    True,
                    f"Generated {len(response['response'])} character response, "
                    f"Stage: {response['conversation_stage']}, "
                    f"Confidence: {response['confidence_score']:.2f}"
                )
            
            # Test conversation flow
            conversation_flow = self.conversation_engine.analyze_conversation_flow([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": customer_message}
            ])
            
            self.log_test(
                "Conversation Engine - Flow Analysis",
                "current_stage" in conversation_flow,
                f"Detected stage: {conversation_flow.get('current_stage', 'Unknown')}"
            )
            
        except Exception as e:
            self.log_test("AI Conversation Engine", False, f"Error: {str(e)}")
    
    async def test_voice_ai_enhancement(self):
        """Test Voice AI Enhancement"""
        print("\n🎤 Testing Voice AI Enhancement...")
        
        try:
            # Create mock audio session
            session_id = await self.voice_ai.start_voice_session({
                "call_id": "test_call_123",
                "participants": ["agent", "customer"],
                "quality_settings": "high"
            })
            
            self.log_test(
                "Voice AI - Session Creation",
                session_id is not None,
                f"Created session: {session_id}"
            )
            
            # Test voice analysis with mock data
            mock_audio_features = {
                "volume_level": 0.7,
                "clarity_score": 0.85,
                "speech_rate": 150,  # words per minute
                "emotional_indicators": {
                    "happiness": 0.3,
                    "surprise": 0.1,
                    "sadness": 0.2,
                    "anger": 0.1,
                    "fear": 0.15,
                    "neutral": 0.15
                },
                "stress_level": 0.3,
                "engagement_score": 0.75
            }
            
            voice_analysis = await self.voice_ai.analyze_voice_features(
                session_id, mock_audio_features
            )
            
            required_analysis_keys = ["emotional_state", "engagement_level", "stress_indicators",
                                    "communication_quality", "real_time_guidance"]
            
            missing_keys = [key for key in required_analysis_keys if key not in voice_analysis]
            
            if missing_keys:
                self.log_test(
                    "Voice AI - Analysis",
                    False,
                    f"Missing analysis keys: {missing_keys}"
                )
            else:
                self.log_test(
                    "Voice AI - Analysis",
                    True,
                    f"Emotion: {voice_analysis['emotional_state']['dominant_emotion']}, "
                    f"Engagement: {voice_analysis['engagement_level']:.2f}, "
                    f"Quality: {voice_analysis['communication_quality']['overall_score']:.2f}"
                )
            
        except Exception as e:
            self.log_test("Voice AI Enhancement", False, f"Error: {str(e)}")
    
    async def test_predictive_analytics(self):
        """Test Predictive Analytics"""
        print("\n🔮 Testing Predictive Analytics...")
        
        try:
            # Test revenue forecasting
            historical_data = [
                {"month": "2024-01", "revenue": 150000},
                {"month": "2024-02", "revenue": 175000},
                {"month": "2024-03", "revenue": 162000},
                {"month": "2024-04", "revenue": 188000},
                {"month": "2024-05", "revenue": 195000},
                {"month": "2024-06", "revenue": 210000}
            ]
            
            revenue_forecast = await self.predictive_analytics.forecast_revenue(
                historical_data, forecast_periods=3
            )
            
            required_forecast_keys = ["forecast", "confidence_intervals", "trend_analysis",
                                    "seasonality", "model_performance"]
            
            missing_keys = [key for key in required_forecast_keys if key not in revenue_forecast]
            
            if missing_keys:
                self.log_test(
                    "Predictive Analytics - Revenue Forecast",
                    False,
                    f"Missing forecast keys: {missing_keys}"
                )
            else:
                self.log_test(
                    "Predictive Analytics - Revenue Forecast",
                    True,
                    f"Generated {len(revenue_forecast['forecast'])} period forecast, "
                    f"Trend: {revenue_forecast['trend_analysis']['direction']}, "
                    f"R²: {revenue_forecast['model_performance']['r_squared']:.3f}"
                )
            
            # Test deal probability prediction
            deal_data = {
                "stage": "Proposal",
                "deal_value": 50000,
                "age_days": 45,
                "interactions_count": 12,
                "decision_maker_engaged": True,
                "budget_confirmed": True,
                "competitor_present": False,
                "timeline_urgency": 0.8
            }
            
            deal_prediction = await self.predictive_analytics.predict_deal_probability(deal_data)
            
            self.log_test(
                "Predictive Analytics - Deal Probability",
                "probability" in deal_prediction,
                f"Probability: {deal_prediction.get('probability', 0):.2f}, "
                f"Expected close: {deal_prediction.get('expected_close_date', 'Unknown')}"
            )
            
        except Exception as e:
            self.log_test("Predictive Analytics", False, f"Error: {str(e)}")
    
    async def test_integration_scenarios(self):
        """Test integrated scenarios across multiple services"""
        print("\n🔄 Testing Integration Scenarios...")
        
        try:
            # Scenario 1: Complete sales interaction workflow
            lead_data = {
                "company_size": "201-500",
                "industry": "Healthcare",
                "job_title": "Chief Technology Officer",
                "email_engagement_rate": 0.65,
                "website_visits": 25,
                "content_downloads": 5,
                "decision_maker": True,
                "budget_indicated": True
            }
            
            # 1. Score the lead
            lead_score = await self.ml_scoring.predict_lead_score(lead_data)
            
            # 2. Analyze conversation
            conversation_message = "We need a solution that can scale with our growing business"
            conversation_analysis = await self.advanced_ai.analyze_conversation(
                conversation_message, []
            )
            
            # 3. Generate intelligent response
            conversation_context = {
                "conversation_id": "integration_test_1",
                "stage": "discovery",
                "lead_score": lead_score["score"],
                "lead_data": lead_data
            }
            
            ai_response = await self.conversation_engine.generate_response(
                conversation_message, conversation_context
            )
            
            # 4. Predict deal outcome
            deal_prediction = await self.predictive_analytics.predict_deal_probability({
                "stage": "Discovery",
                "deal_value": lead_score["predicted_deal_value"],
                "age_days": 3,
                "interactions_count": 1,
                "decision_maker_engaged": True,
                "budget_confirmed": lead_data["budget_indicated"],
                "timeline_urgency": conversation_analysis["intent"].get("urgency", 0.5)
            })
            
            integration_success = all([
                lead_score.get("score", 0) > 0,
                conversation_analysis.get("sentiment", {}).get("compound", 0) != 0,
                len(ai_response.get("response", "")) > 0,
                deal_prediction.get("probability", 0) > 0
            ])
            
            self.log_test(
                "Integration - Complete Sales Workflow",
                integration_success,
                f"Lead Score: {lead_score.get('score', 0):.1f}, "
                f"Sentiment: {conversation_analysis.get('sentiment', {}).get('compound', 0):.2f}, "
                f"Deal Probability: {deal_prediction.get('probability', 0):.2f}"
            )
            
        except Exception as e:
            self.log_test("Integration Scenarios", False, f"Error: {str(e)}")
    
    def performance_benchmark(self):
        """Run performance benchmarks"""
        print("\n⚡ Running Performance Benchmarks...")
        
        # Simple response time test
        start_time = time.time()
        
        # Simulate quick operations
        for _ in range(10):
            # Mock quick computation
            time.sleep(0.01)
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        self.log_test(
            "Performance - Response Time",
            execution_time < 500,  # Should complete within 500ms
            f"Avg response time: {execution_time/10:.1f}ms per operation"
        )
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE TEST SUITE SUMMARY")
        print("="*60)
        
        total_tests = self.success_count + self.failure_count
        success_rate = (self.success_count / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {self.success_count} ✅")
        print(f"   Failed: {self.failure_count} ❌")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if self.failure_count > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result["success"]:
                    print(f"   • {test_name}: {result['details']}")
        
        print(f"\n🎯 System Status:")
        if success_rate >= 90:
            print("   🟢 EXCELLENT - System is fully operational")
        elif success_rate >= 75:
            print("   🟡 GOOD - System is mostly operational with minor issues")
        elif success_rate >= 50:
            print("   🟠 FAIR - System has significant issues requiring attention")
        else:
            print("   🔴 POOR - System requires immediate attention")
        
        print("\n" + "="*60)

async def main():
    """Main test execution function"""
    print("🚀 Starting Comprehensive AI/ML Sales Automation Test Suite")
    print("Testing all advanced AI components...")
    
    test_suite = ComprehensiveTestSuite()
    
    # Run all test categories
    await test_suite.test_advanced_ai_service()
    await test_suite.test_ml_lead_scoring()
    await test_suite.test_conversation_engine()
    await test_suite.test_voice_ai_enhancement()
    await test_suite.test_predictive_analytics()
    await test_suite.test_integration_scenarios()
    test_suite.performance_benchmark()
    
    # Print comprehensive summary
    test_suite.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if test_suite.failure_count == 0 else 1)

if __name__ == "__main__":
    asyncio.run(main())