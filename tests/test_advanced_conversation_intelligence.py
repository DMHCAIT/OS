"""
Unit Tests for Advanced Conversation Intelligence System
Comprehensive testing for all conversation intelligence components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import numpy as np

from app.core.advanced_conversation_intelligence import (
    AdvancedConversationIntelligence,
    ConversationAnalysisResult,
    ConversationSession
)
from app.ai.multilingual_intelligence import MultilingualIntelligence
from app.ai.emotion_detection import EmotionManager
from app.ai.dynamic_script_adaptation import DynamicScriptAdaptationEngine
from app.ai.competitor_intelligence import CompetitorIntelligenceManager


class TestAdvancedConversationIntelligence:
    """Test suite for the main conversation intelligence system"""
    
    @pytest.fixture
    def intelligence_system(self):
        """Create a test instance of the conversation intelligence system"""
        return AdvancedConversationIntelligence()
    
    @pytest.fixture
    def sample_conversation_data(self):
        """Sample conversation data for testing"""
        return {
            'conversation_id': 'test_conv_123',
            'participant_id': 'user_456',
            'message': 'Hello, I am interested in your product but I heard competitor X has better features.',
            'audio_data': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'conversation_context': {
                'stage': 'discovery',
                'previous_messages': 5
            }
        }
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, intelligence_system):
        """Test that the system initializes correctly"""
        await intelligence_system.initialize()
        
        assert intelligence_system.multilingual_intelligence is not None
        assert intelligence_system.emotion_manager is not None
        assert intelligence_system.script_adaptation_engine is not None
        assert intelligence_system.competitor_intelligence is not None
        assert intelligence_system.system_metrics['initialization_time'] is not None
    
    @pytest.mark.asyncio
    async def test_conversation_message_analysis(self, intelligence_system, sample_conversation_data):
        """Test comprehensive conversation message analysis"""
        await intelligence_system.initialize()
        
        # Mock the individual intelligence systems
        with patch.object(intelligence_system.multilingual_intelligence, 'analyze_message') as mock_multilingual, \
             patch.object(intelligence_system.emotion_manager, 'analyze_emotion') as mock_emotion, \
             patch.object(intelligence_system.script_adaptation_engine, 'analyze_and_adapt') as mock_adaptation, \
             patch.object(intelligence_system.competitor_intelligence, 'analyze_competitive_context') as mock_competitive:
            
            # Setup mock returns
            mock_multilingual.return_value = {
                'detected_language': 'en',
                'confidence': 0.98,
                'cultural_context': {'formality': 'professional'}
            }
            
            mock_emotion.return_value = {
                'combined_analysis': {
                    'emotional_state': 'interested',
                    'confidence': 0.85
                }
            }
            
            mock_adaptation.return_value = {
                'adaptation_type': 'competitive_response',
                'recommended_content': 'Address competitor comparison',
                'confidence': 0.90
            }
            
            mock_competitive.return_value = {
                'competitive_mentions': [{'competitor': 'competitor X', 'sentiment': 'neutral'}],
                'threat_level': 'medium'
            }
            
            result = await intelligence_system.analyze_conversation_message(
                conversation_id=sample_conversation_data['conversation_id'],
                participant_id=sample_conversation_data['participant_id'],
                message=sample_conversation_data['message'],
                audio_data=sample_conversation_data['audio_data'],
                conversation_context=sample_conversation_data['conversation_context']
            )
            
            # Verify result structure
            assert isinstance(result, ConversationAnalysisResult)
            assert result.conversation_id == sample_conversation_data['conversation_id']
            assert result.participant_id == sample_conversation_data['participant_id']
            assert result.message == sample_conversation_data['message']
            assert result.language_analysis is not None
            assert result.emotion_analysis is not None
            assert result.script_adaptation is not None
            assert result.competitive_analysis is not None
            assert isinstance(result.conversation_health_score, float)
            assert 0 <= result.conversation_health_score <= 1
            assert result.priority_actions is not None
            assert isinstance(result.priority_actions, list)
    
    @pytest.mark.asyncio
    async def test_conversation_session_management(self, intelligence_system, sample_conversation_data):
        """Test conversation session creation and management"""
        await intelligence_system.initialize()
        
        conversation_id = sample_conversation_data['conversation_id']
        
        # First message should create a session
        with patch.object(intelligence_system.multilingual_intelligence, 'analyze_message') as mock_multilingual, \
             patch.object(intelligence_system.emotion_manager, 'analyze_emotion') as mock_emotion, \
             patch.object(intelligence_system.script_adaptation_engine, 'analyze_and_adapt') as mock_adaptation, \
             patch.object(intelligence_system.competitor_intelligence, 'analyze_competitive_context') as mock_competitive:
            
            mock_multilingual.return_value = {'detected_language': 'en', 'confidence': 0.98}
            mock_emotion.return_value = {'combined_analysis': {'emotional_state': 'neutral'}}
            mock_adaptation.return_value = {'adaptation_type': 'maintain', 'recommended_content': ''}
            mock_competitive.return_value = {'competitive_mentions': [], 'threat_level': 'none'}
            
            await intelligence_system.analyze_conversation_message(
                conversation_id=conversation_id,
                participant_id=sample_conversation_data['participant_id'],
                message=sample_conversation_data['message']
            )
            
            # Verify session was created
            assert conversation_id in intelligence_system.active_sessions
            session = intelligence_system.active_sessions[conversation_id]
            assert isinstance(session, ConversationSession)
            assert len(session.conversation_history) == 1
            assert len(session.intelligence_timeline) == 1
    
    @pytest.mark.asyncio
    async def test_health_score_calculation(self, intelligence_system):
        """Test conversation health score calculation logic"""
        await intelligence_system.initialize()
        
        # Test scenarios with different analysis results
        test_cases = [
            {
                'emotion': {'combined_analysis': {'emotional_state': 'happy', 'confidence': 0.9}},
                'competitive': {'competitive_mentions': [], 'threat_level': 'none'},
                'adaptation': {'adaptation_type': 'maintain', 'confidence': 0.8},
                'expected_range': (0.8, 1.0)
            },
            {
                'emotion': {'combined_analysis': {'emotional_state': 'frustrated', 'confidence': 0.9}},
                'competitive': {'competitive_mentions': [{'competitor': 'X'}], 'threat_level': 'high'},
                'adaptation': {'adaptation_type': 'urgent_intervention', 'confidence': 0.9},
                'expected_range': (0.0, 0.4)
            },
            {
                'emotion': {'combined_analysis': {'emotional_state': 'neutral', 'confidence': 0.7}},
                'competitive': {'competitive_mentions': [], 'threat_level': 'low'},
                'adaptation': {'adaptation_type': 'gentle_adjustment', 'confidence': 0.6},
                'expected_range': (0.5, 0.8)
            }
        ]
        
        for test_case in test_cases:
            health_score = intelligence_system._calculate_health_score(
                test_case['emotion'],
                test_case['competitive'],
                test_case['adaptation']
            )
            
            min_expected, max_expected = test_case['expected_range']
            assert min_expected <= health_score <= max_expected, \
                f"Health score {health_score} not in expected range {test_case['expected_range']}"
    
    @pytest.mark.asyncio
    async def test_priority_actions_generation(self, intelligence_system):
        """Test generation of priority actions based on analysis"""
        await intelligence_system.initialize()
        
        # Test urgent competitive scenario
        competitive_analysis = {
            'competitive_mentions': [{'competitor': 'CompetitorX', 'sentiment': 'positive'}],
            'threat_level': 'high'
        }
        emotion_analysis = {
            'combined_analysis': {'emotional_state': 'confused', 'confidence': 0.8}
        }
        adaptation_analysis = {
            'adaptation_type': 'competitive_response',
            'recommended_content': 'Address competitor strengths'
        }
        
        actions = intelligence_system._generate_priority_actions(
            emotion_analysis,
            competitive_analysis,
            adaptation_analysis
        )
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        # Should include urgent competitive action
        urgent_actions = [action for action in actions if 'URGENT' in action.upper()]
        assert len(urgent_actions) > 0
        
        # Should include competitive response
        competitive_actions = [action for action in actions if 'competitive' in action.lower()]
        assert len(competitive_actions) > 0
    
    @pytest.mark.asyncio
    async def test_conversation_summary_generation(self, intelligence_system, sample_conversation_data):
        """Test conversation summary generation"""
        await intelligence_system.initialize()
        
        conversation_id = sample_conversation_data['conversation_id']
        
        # First create a conversation with some history
        with patch.object(intelligence_system.multilingual_intelligence, 'analyze_message'), \
             patch.object(intelligence_system.emotion_manager, 'analyze_emotion'), \
             patch.object(intelligence_system.script_adaptation_engine, 'analyze_and_adapt'), \
             patch.object(intelligence_system.competitor_intelligence, 'analyze_competitive_context'):
            
            # Add multiple messages to create history
            for i in range(3):
                await intelligence_system.analyze_conversation_message(
                    conversation_id=conversation_id,
                    participant_id=f"user_{i}",
                    message=f"Test message {i}"
                )
            
            summary = await intelligence_system.generate_conversation_summary(conversation_id)
            
            assert 'error' not in summary
            assert 'conversation_overview' in summary
            assert 'participant_insights' in summary
            assert 'conversation_flow' in summary
            assert 'recommendations' in summary
            
            # Check summary content
            assert summary['conversation_overview']['total_messages'] == 3
            assert summary['conversation_overview']['unique_participants'] == 3
            assert isinstance(summary['conversation_flow']['health_progression'], list)
    
    @pytest.mark.asyncio
    async def test_system_performance_metrics(self, intelligence_system):
        """Test system performance metrics collection"""
        await intelligence_system.initialize()
        
        metrics = await intelligence_system.get_system_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_analyses_completed' in metrics
        assert 'average_processing_time' in metrics
        assert 'system_health' in metrics
        assert 'component_performance' in metrics
        
        # Verify component performance tracking
        components = ['multilingual_intelligence', 'emotion_manager', 
                     'script_adaptation_engine', 'competitor_intelligence']
        for component in components:
            assert component in metrics['component_performance']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, intelligence_system, sample_conversation_data):
        """Test error handling in various failure scenarios"""
        await intelligence_system.initialize()
        
        # Test with invalid conversation data
        with pytest.raises(ValueError):
            await intelligence_system.analyze_conversation_message(
                conversation_id="",  # Invalid empty ID
                participant_id=sample_conversation_data['participant_id'],
                message=sample_conversation_data['message']
            )
        
        with pytest.raises(ValueError):
            await intelligence_system.analyze_conversation_message(
                conversation_id=sample_conversation_data['conversation_id'],
                participant_id="",  # Invalid empty participant ID
                message=sample_conversation_data['message']
            )
        
        # Test with component failure
        with patch.object(intelligence_system.multilingual_intelligence, 'analyze_message') as mock_multilingual:
            mock_multilingual.side_effect = Exception("Simulated component failure")
            
            # Should handle gracefully and still return a result
            result = await intelligence_system.analyze_conversation_message(
                conversation_id=sample_conversation_data['conversation_id'],
                participant_id=sample_conversation_data['participant_id'],
                message=sample_conversation_data['message']
            )
            
            assert result is not None
            assert 'error' not in result.language_analysis or result.language_analysis.get('error') is not None


class TestConversationIntelligenceComponents:
    """Test suite for individual conversation intelligence components"""
    
    @pytest.mark.asyncio
    async def test_multilingual_intelligence_integration(self):
        """Test multilingual intelligence component integration"""
        multilingual = MultilingualIntelligence()
        await multilingual.initialize()
        
        # Test English message
        result = await multilingual.analyze_message("Hello, how are you?")
        assert result['detected_language'] == 'en'
        assert result['confidence'] > 0.8
        
        # Test Spanish message
        result = await multilingual.analyze_message("Hola, ¿cómo estás?")
        assert result['detected_language'] == 'es'
        assert result['confidence'] > 0.8
    
    @pytest.mark.asyncio
    async def test_emotion_detection_integration(self):
        """Test emotion detection component integration"""
        emotion_manager = EmotionManager()
        await emotion_manager.initialize()
        
        # Test positive message
        result = await emotion_manager.analyze_emotion(
            text="I'm really excited about this product!",
            audio_data=None
        )
        assert 'combined_analysis' in result
        assert result['combined_analysis']['emotional_state'] in ['happy', 'excited']
        
        # Test negative message
        result = await emotion_manager.analyze_emotion(
            text="I'm frustrated with the poor service",
            audio_data=None
        )
        assert 'combined_analysis' in result
        assert result['combined_analysis']['emotional_state'] in ['frustrated', 'angry']
    
    @pytest.mark.asyncio
    async def test_script_adaptation_integration(self):
        """Test script adaptation component integration"""
        adaptation_engine = DynamicScriptAdaptationEngine()
        await adaptation_engine.initialize()
        
        # Test script adaptation for concerned prospect
        result = await adaptation_engine.analyze_and_adapt(
            message="I'm not sure if this is worth the price",
            conversation_history=[
                {'message': 'Tell me about pricing', 'timestamp': datetime.now()},
                {'message': 'That seems expensive', 'timestamp': datetime.now()}
            ],
            current_script="Standard pricing presentation"
        )
        
        assert result['adaptation_type'] in ['price_concerns', 'value_reinforcement']
        assert 'recommended_content' in result
        assert len(result['recommended_content']) > 0
    
    @pytest.mark.asyncio
    async def test_competitor_intelligence_integration(self):
        """Test competitor intelligence component integration"""
        competitor_intel = CompetitorIntelligenceManager()
        await competitor_intel.initialize()
        
        # Test competitive mention detection
        result = await competitor_intel.analyze_competitive_context(
            message="I've been looking at Salesforce and HubSpot as alternatives",
            conversation_history=[]
        )
        
        assert 'competitive_mentions' in result
        assert len(result['competitive_mentions']) >= 2  # Should detect Salesforce and HubSpot
        
        competitor_names = [mention['competitor'].lower() for mention in result['competitive_mentions']]
        assert any('salesforce' in name for name in competitor_names)
        assert any('hubspot' in name for name in competitor_names)


class TestConversationAnalysisResult:
    """Test suite for ConversationAnalysisResult data structure"""
    
    def test_conversation_analysis_result_creation(self):
        """Test creation and validation of ConversationAnalysisResult"""
        result = ConversationAnalysisResult(
            conversation_id="test_123",
            participant_id="user_456",
            message="Test message",
            analysis_timestamp=datetime.now(),
            language_analysis={'detected_language': 'en'},
            emotion_analysis={'emotional_state': 'neutral'},
            behavioral_insights={'engagement': 0.7},
            script_adaptation={'adaptation_type': 'maintain'},
            competitive_analysis={'competitive_mentions': []},
            priority_actions=['Continue conversation'],
            conversation_health_score=0.8,
            engagement_forecast="positive",
            adaptive_strategy="maintain_current_approach"
        )
        
        assert result.conversation_id == "test_123"
        assert result.participant_id == "user_456" 
        assert result.message == "Test message"
        assert isinstance(result.analysis_timestamp, datetime)
        assert result.conversation_health_score == 0.8
        assert result.engagement_forecast == "positive"


class TestConversationSession:
    """Test suite for ConversationSession management"""
    
    def test_conversation_session_creation(self):
        """Test conversation session creation and initialization"""
        session = ConversationSession(conversation_id="test_session")
        
        assert session.conversation_id == "test_session"
        assert session.conversation_history == []
        assert session.intelligence_timeline == []
        assert session.cultural_context == {}
        assert session.emotional_journey == []
        assert session.competitive_context == {}
        assert session.adaptation_history == []
        assert session.current_stage == "initial"
        assert session.engagement_trend == 0.5
    
    def test_conversation_session_message_tracking(self):
        """Test conversation session message and analysis tracking"""
        session = ConversationSession(conversation_id="test_session")
        
        # Add messages to history
        message1 = {
            'participant_id': 'user_1',
            'message': 'Hello',
            'timestamp': datetime.now()
        }
        message2 = {
            'participant_id': 'agent_1', 
            'message': 'Hi there!',
            'timestamp': datetime.now()
        }
        
        session.conversation_history.append(message1)
        session.conversation_history.append(message2)
        
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]['message'] == 'Hello'
        assert session.conversation_history[1]['message'] == 'Hi there!'


# Integration Tests
class TestEndToEndIntegration:
    """End-to-end integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self):
        """Test a complete conversation flow from start to finish"""
        intelligence = AdvancedConversationIntelligence()
        await intelligence.initialize()
        
        conversation_id = "integration_test_conv"
        participant_id = "test_user"
        
        # Conversation flow: greeting -> interest -> concern -> competitive -> resolution
        conversation_messages = [
            "Hello, I'm interested in learning about your product",
            "That sounds good, but I'm concerned about the cost",
            "I've heard that your competitor offers similar features for less money",
            "Okay, let me think about it. Can you send me more information?",
            "Thanks, I'm ready to move forward with a trial"
        ]
        
        results = []
        
        for i, message in enumerate(conversation_messages):
            with patch.object(intelligence.multilingual_intelligence, 'analyze_message') as mock_multilingual, \
                 patch.object(intelligence.emotion_manager, 'analyze_emotion') as mock_emotion, \
                 patch.object(intelligence.script_adaptation_engine, 'analyze_and_adapt') as mock_adaptation, \
                 patch.object(intelligence.competitor_intelligence, 'analyze_competitive_context') as mock_competitive:
                
                # Mock progressive conversation states
                mock_multilingual.return_value = {'detected_language': 'en', 'confidence': 0.98}
                
                if i == 1:  # Concern message
                    mock_emotion.return_value = {'combined_analysis': {'emotional_state': 'concerned'}}
                    mock_adaptation.return_value = {'adaptation_type': 'address_concerns'}
                elif i == 2:  # Competitive message
                    mock_emotion.return_value = {'combined_analysis': {'emotional_state': 'analytical'}}
                    mock_adaptation.return_value = {'adaptation_type': 'competitive_response'}
                    mock_competitive.return_value = {
                        'competitive_mentions': [{'competitor': 'competitor', 'sentiment': 'neutral'}],
                        'threat_level': 'medium'
                    }
                else:
                    mock_emotion.return_value = {'combined_analysis': {'emotional_state': 'neutral'}}
                    mock_adaptation.return_value = {'adaptation_type': 'maintain'}
                
                mock_competitive.return_value = mock_competitive.return_value if i == 2 else {
                    'competitive_mentions': [], 'threat_level': 'none'
                }
                
                result = await intelligence.analyze_conversation_message(
                    conversation_id=conversation_id,
                    participant_id=participant_id,
                    message=message
                )
                
                results.append(result)
        
        # Verify conversation progression
        assert len(results) == 5
        
        # Check that session was maintained throughout
        session = intelligence.active_sessions[conversation_id]
        assert len(session.conversation_history) == 5
        assert len(session.intelligence_timeline) == 5
        
        # Verify competitive detection in message 3
        competitive_result = results[2]
        assert len(competitive_result.competitive_analysis.get('competitive_mentions', [])) > 0
        
        # Generate final summary
        summary = await intelligence.generate_conversation_summary(conversation_id)
        assert 'conversation_overview' in summary
        assert summary['conversation_overview']['total_messages'] == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])