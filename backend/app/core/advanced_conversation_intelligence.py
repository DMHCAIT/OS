"""
Advanced Conversation Intelligence Integration System
Integrates multi-language support, emotion detection, dynamic script adaptation, and competitor intelligence
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

# Import our advanced conversation intelligence modules
from .multilingual_intelligence import multilingual_intelligence
from .emotion_detection import emotion_manager
from .dynamic_script_adaptation import script_adaptation_engine
from .competitor_intelligence import competitor_intelligence

logger = logging.getLogger(__name__)

@dataclass
class ConversationIntelligenceResult:
    """Comprehensive conversation intelligence analysis result"""
    conversation_id: str
    participant_id: str
    message: str
    analysis_timestamp: datetime
    
    # Multi-language analysis
    language_analysis: Dict[str, Any]
    localized_response: Optional[Dict[str, Any]]
    
    # Emotion analysis
    emotion_analysis: Dict[str, Any]
    empathy_recommendations: Dict[str, Any]
    
    # Script adaptation
    behavioral_insights: Dict[str, Any]
    script_adaptation: Dict[str, Any]
    
    # Competitive intelligence
    competitive_analysis: Dict[str, Any]
    strategic_responses: List[Dict[str, Any]]
    
    # Integrated recommendations
    priority_actions: List[str]
    conversation_health_score: float
    engagement_forecast: str
    adaptive_strategy: str

@dataclass
class ConversationSession:
    """Comprehensive conversation session state"""
    session_id: str
    participant_profile: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    intelligence_timeline: List[ConversationIntelligenceResult]
    current_stage: str
    engagement_trend: str
    competitive_context: Dict[str, Any]
    cultural_context: Dict[str, Any]
    emotional_journey: List[Dict[str, Any]]
    adaptation_history: List[Dict[str, Any]]


class AdvancedConversationIntelligence:
    """Master integration system for advanced conversation intelligence"""
    
    def __init__(self):
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.intelligence_cache: Dict[str, Any] = {}
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_adaptations': 0,
            'competitive_mentions_detected': 0,
            'emotion_interventions': 0,
            'language_translations': 0
        }
    
    async def initialize(self):
        """Initialize all conversation intelligence systems"""
        logger.info("Initializing Advanced Conversation Intelligence System...")
        
        try:
            # Initialize all subsystems
            await multilingual_intelligence.initialize()
            await emotion_manager.initialize()
            await script_adaptation_engine.initialize()
            await competitor_intelligence.initialize()
            
            logger.info("Advanced Conversation Intelligence System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing conversation intelligence: {e}")
            raise
    
    async def analyze_conversation_message(self,
                                         conversation_id: str,
                                         participant_id: str,
                                         message: str,
                                         audio_data: Optional[np.ndarray] = None,
                                         current_script: Optional[str] = None,
                                         conversation_context: Optional[Dict[str, Any]] = None) -> ConversationIntelligenceResult:
        """Comprehensive analysis of conversation message using all intelligence systems"""
        
        start_time = datetime.now()
        
        try:
            # Get or create session
            session = await self._get_or_create_session(conversation_id, participant_id)
            
            # Parallel analysis across all systems
            analysis_tasks = [
                self._analyze_language(conversation_id, message, participant_id),
                self._analyze_emotions(conversation_id, participant_id, message, audio_data),
                self._analyze_script_adaptation(conversation_id, message, session.conversation_history, current_script),
                self._analyze_competitive_intelligence(conversation_id, message, conversation_context or {})
            ]
            
            # Execute analyses in parallel
            language_result, emotion_result, adaptation_result, competitive_result = await asyncio.gather(*analysis_tasks)
            
            # Integrate results
            integrated_result = await self._integrate_analysis_results(
                conversation_id, participant_id, message, 
                language_result, emotion_result, adaptation_result, competitive_result,
                session
            )
            
            # Update session
            await self._update_session(session, integrated_result)
            
            # Update performance metrics
            self._update_performance_metrics(integrated_result)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Conversation intelligence analysis completed in {analysis_time:.2f}s")
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in conversation intelligence analysis: {e}")
            # Return minimal result
            return ConversationIntelligenceResult(
                conversation_id=conversation_id,
                participant_id=participant_id,
                message=message,
                analysis_timestamp=datetime.now(),
                language_analysis={'error': str(e)},
                localized_response=None,
                emotion_analysis={'error': str(e)},
                empathy_recommendations={},
                behavioral_insights={'error': str(e)},
                script_adaptation={},
                competitive_analysis={'error': str(e)},
                strategic_responses=[],
                priority_actions=['Error in analysis - manual intervention recommended'],
                conversation_health_score=0.5,
                engagement_forecast='uncertain',
                adaptive_strategy='maintain_current_approach'
            )
    
    async def _get_or_create_session(self, conversation_id: str, participant_id: str) -> ConversationSession:
        """Get existing session or create new one"""
        if conversation_id not in self.active_sessions:
            self.active_sessions[conversation_id] = ConversationSession(
                session_id=conversation_id,
                participant_profile={'id': participant_id},
                conversation_history=[],
                intelligence_timeline=[],
                current_stage='discovery',
                engagement_trend='stable',
                competitive_context={},
                cultural_context={},
                emotional_journey=[],
                adaptation_history=[]
            )
        
        return self.active_sessions[conversation_id]
    
    async def _analyze_language(self, conversation_id: str, message: str, participant_id: str) -> Dict[str, Any]:
        """Analyze language and cultural context"""
        try:
            language_result = await multilingual_intelligence.process_multilingual_message(
                conversation_id, message, participant_id
            )
            return language_result
        except Exception as e:
            logger.error(f"Language analysis error: {e}")
            return {'error': str(e)}
    
    async def _analyze_emotions(self, conversation_id: str, participant_id: str, message: str, audio_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze emotional intelligence"""
        try:
            emotion_result = await emotion_manager.analyze_conversation_emotions(
                conversation_id, participant_id, message, audio_data
            )
            return emotion_result
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {'error': str(e)}
    
    async def _analyze_script_adaptation(self, conversation_id: str, message: str, conversation_history: List[Dict[str, Any]], current_script: Optional[str]) -> Dict[str, Any]:
        """Analyze behavioral cues and script adaptation needs"""
        try:
            adaptation_result = await script_adaptation_engine.analyze_and_adapt(
                conversation_id, message, conversation_history, current_script or "Continue the conversation naturally."
            )
            return asdict(adaptation_result)
        except Exception as e:
            logger.error(f"Script adaptation error: {e}")
            return {'error': str(e)}
    
    async def _analyze_competitive_intelligence(self, conversation_id: str, message: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive mentions and generate strategic responses"""
        try:
            competitive_result = await competitor_intelligence.analyze_competitive_conversation(
                conversation_id, message, conversation_context
            )
            return competitive_result
        except Exception as e:
            logger.error(f"Competitive intelligence error: {e}")
            return {'error': str(e)}
    
    async def _integrate_analysis_results(self,
                                        conversation_id: str,
                                        participant_id: str,
                                        message: str,
                                        language_result: Dict[str, Any],
                                        emotion_result: Dict[str, Any],
                                        adaptation_result: Dict[str, Any],
                                        competitive_result: Dict[str, Any],
                                        session: ConversationSession) -> ConversationIntelligenceResult:
        """Integrate all analysis results into comprehensive intelligence"""
        
        # Extract key insights
        language_detected = language_result.get('language_detection', {}).get('detected_language', 'en')
        emotion_state = emotion_result.get('combined_analysis', {}).get('emotional_state', 'neutral')
        adaptation_strategy = adaptation_result.get('adaptation_type', 'maintain')
        competitive_mentions = len(competitive_result.get('competitive_mentions', []))
        
        # Generate localized response if needed
        localized_response = None
        if language_detected != 'en':
            try:
                localized_response = await multilingual_intelligence.generate_multilingual_response(
                    conversation_id,
                    adaptation_result.get('recommended_content', ''),
                    language_detected
                )
                localized_response = asdict(localized_response)
            except:
                pass
        
        # Calculate conversation health score
        health_score = await self._calculate_conversation_health(
            emotion_result, adaptation_result, competitive_result, session
        )
        
        # Predict engagement forecast
        engagement_forecast = await self._predict_engagement_forecast(
            emotion_result, adaptation_result, session
        )
        
        # Determine adaptive strategy
        adaptive_strategy = await self._determine_adaptive_strategy(
            emotion_result, adaptation_result, competitive_result
        )
        
        # Generate priority actions
        priority_actions = await self._generate_priority_actions(
            language_result, emotion_result, adaptation_result, competitive_result
        )
        
        return ConversationIntelligenceResult(
            conversation_id=conversation_id,
            participant_id=participant_id,
            message=message,
            analysis_timestamp=datetime.now(),
            language_analysis=language_result,
            localized_response=localized_response,
            emotion_analysis=emotion_result,
            empathy_recommendations=emotion_result.get('empathy_response', {}),
            behavioral_insights=adaptation_result,
            script_adaptation=adaptation_result,
            competitive_analysis=competitive_result,
            strategic_responses=competitive_result.get('strategic_responses', []),
            priority_actions=priority_actions,
            conversation_health_score=health_score,
            engagement_forecast=engagement_forecast,
            adaptive_strategy=adaptive_strategy
        )
    
    async def _calculate_conversation_health(self,
                                           emotion_result: Dict[str, Any],
                                           adaptation_result: Dict[str, Any],
                                           competitive_result: Dict[str, Any],
                                           session: ConversationSession) -> float:
        """Calculate overall conversation health score"""
        
        health_factors = []
        
        # Emotion health factor
        emotion_analysis = emotion_result.get('combined_analysis', {})
        emotion_score = emotion_analysis.get('sentiment_score', 0.0)
        emotion_stability = emotion_analysis.get('stability', 0.5)
        emotion_health = (emotion_score + 1) / 2 * 0.7 + emotion_stability * 0.3  # Normalize sentiment to 0-1
        health_factors.append(emotion_health)
        
        # Adaptation health factor
        adaptation_confidence = adaptation_result.get('confidence', 0.5)
        success_probability = adaptation_result.get('success_probability', 0.5)
        adaptation_health = (adaptation_confidence + success_probability) / 2
        health_factors.append(adaptation_health)
        
        # Competitive health factor (fewer competitors = better health)
        competitive_mentions = len(competitive_result.get('competitive_mentions', []))
        competitive_intensity = competitive_result.get('competitive_landscape', {}).get('competitive_intensity', 0.0)
        competitive_health = max(1.0 - competitive_intensity, 0.0)
        health_factors.append(competitive_health)
        
        # Historical trend factor
        if len(session.intelligence_timeline) > 2:
            recent_scores = [result.conversation_health_score for result in session.intelligence_timeline[-3:]]
            trend_factor = 1.0 if len(recent_scores) < 2 else (recent_scores[-1] / max(recent_scores[-2], 0.1))
            health_factors.append(min(trend_factor, 1.0))
        
        # Calculate weighted average
        overall_health = sum(health_factors) / len(health_factors)
        return min(max(overall_health, 0.0), 1.0)
    
    async def _predict_engagement_forecast(self,
                                         emotion_result: Dict[str, Any],
                                         adaptation_result: Dict[str, Any],
                                         session: ConversationSession) -> str:
        """Predict future engagement based on current trends"""
        
        # Extract engagement indicators
        emotion_analysis = emotion_result.get('combined_analysis', {})
        emotional_state = emotion_analysis.get('emotional_state', 'neutral')
        intensity = emotion_analysis.get('intensity', 0.5)
        
        adaptation_success = adaptation_result.get('success_probability', 0.5)
        
        # Positive indicators
        positive_emotions = ['enthusiastic', 'excited', 'confident', 'interested']
        negative_emotions = ['frustrated', 'defensive', 'disengaged', 'worried']
        
        score = 0.5  # Neutral baseline
        
        if emotional_state in positive_emotions:
            score += 0.3
        elif emotional_state in negative_emotions:
            score -= 0.3
        
        # Intensity modifier
        if emotional_state in positive_emotions:
            score += intensity * 0.2
        elif emotional_state in negative_emotions:
            score -= intensity * 0.2
        
        # Adaptation success influence
        score += (adaptation_success - 0.5) * 0.4
        
        # Historical trend
        if len(session.intelligence_timeline) >= 3:
            recent_emotions = [result.emotion_analysis.get('combined_analysis', {}).get('emotional_state', 'neutral') 
                             for result in session.intelligence_timeline[-3:]]
            positive_trend = sum(1 for emotion in recent_emotions if emotion in positive_emotions)
            negative_trend = sum(1 for emotion in recent_emotions if emotion in negative_emotions)
            
            if positive_trend > negative_trend:
                score += 0.1
            elif negative_trend > positive_trend:
                score -= 0.1
        
        # Classify forecast
        if score >= 0.7:
            return 'highly_positive'
        elif score >= 0.55:
            return 'positive'
        elif score >= 0.45:
            return 'stable'
        elif score >= 0.3:
            return 'concerning'
        else:
            return 'critical'
    
    async def _determine_adaptive_strategy(self,
                                         emotion_result: Dict[str, Any],
                                         adaptation_result: Dict[str, Any],
                                         competitive_result: Dict[str, Any]) -> str:
        """Determine overall adaptive strategy"""
        
        # Priority order: Competitive > Emotion > Behavioral
        
        # Check for competitive situations first
        competitive_mentions = competitive_result.get('competitive_mentions', [])
        if competitive_mentions:
            high_urgency = any(mention.get('response_urgency') == 'high' for mention in competitive_mentions)
            if high_urgency:
                return 'competitive_response_urgent'
            else:
                return 'competitive_positioning'
        
        # Check emotional state
        emotion_analysis = emotion_result.get('combined_analysis', {})
        emotional_state = emotion_analysis.get('emotional_state', 'neutral')
        intensity = emotion_analysis.get('intensity', 0.5)
        
        if emotional_state in ['frustrated', 'worried', 'defensive'] and intensity > 0.6:
            return 'emotion_intervention'
        elif emotional_state in ['enthusiastic', 'excited'] and intensity > 0.7:
            return 'momentum_acceleration'
        
        # Default to behavioral adaptation
        adaptation_type = adaptation_result.get('adaptation_type', 'maintain')
        return f"behavioral_{adaptation_type}"
    
    async def _generate_priority_actions(self,
                                       language_result: Dict[str, Any],
                                       emotion_result: Dict[str, Any],
                                       adaptation_result: Dict[str, Any],
                                       competitive_result: Dict[str, Any]) -> List[str]:
        """Generate prioritized action recommendations"""
        
        actions = []
        
        # High priority: Competitive threats
        competitive_mentions = competitive_result.get('competitive_mentions', [])
        high_urgency_competitive = [m for m in competitive_mentions if m.get('response_urgency') == 'high']
        if high_urgency_competitive:
            competitors = [m.get('competitor_name') for m in high_urgency_competitive]
            actions.append(f"URGENT: Address competitive threat from {', '.join(competitors)}")
        
        # High priority: Emotional crises
        emotion_analysis = emotion_result.get('combined_analysis', {})
        emotional_state = emotion_analysis.get('emotional_state', 'neutral')
        if emotional_state in ['frustrated', 'defensive'] and emotion_analysis.get('intensity', 0) > 0.7:
            actions.append("URGENT: Deploy empathetic response to address emotional concerns")
        
        # Medium priority: Script adaptation
        adaptation_type = adaptation_result.get('adaptation_type', 'maintain')
        if adaptation_type != 'maintain' and adaptation_result.get('confidence', 0) > 0.7:
            actions.append(f"Implement {adaptation_type.replace('_', ' ')} script adaptation")
        
        # Medium priority: Language localization
        language_detection = language_result.get('language_detection', {})
        if language_detection.get('needs_translation', False):
            detected_lang = language_detection.get('detected_language', 'unknown')
            actions.append(f"Use localized response for {detected_lang} speaker")
        
        # Low priority: Engagement optimization
        if emotion_analysis.get('sentiment_score', 0) < -0.2:
            actions.append("Focus on relationship building and value reinforcement")
        
        # Strategic recommendations
        strategic_responses = competitive_result.get('strategic_responses', [])
        if strategic_responses and len(actions) < 3:
            for response in strategic_responses[:2]:  # Top 2 strategic responses
                strategy = response.get('response_strategy', '').replace('_', ' ')
                actions.append(f"Strategic approach: {strategy}")
        
        return actions[:5]  # Limit to top 5 actions
    
    async def _update_session(self, session: ConversationSession, result: ConversationIntelligenceResult):
        """Update session with latest intelligence result"""
        
        # Add to timeline
        session.intelligence_timeline.append(result)
        
        # Keep only recent timeline (last 50 entries)
        if len(session.intelligence_timeline) > 50:
            session.intelligence_timeline = session.intelligence_timeline[-50:]
        
        # Update conversation history
        session.conversation_history.append({
            'timestamp': result.analysis_timestamp.isoformat(),
            'message': result.message,
            'participant_id': result.participant_id,
            'intelligence_summary': {
                'health_score': result.conversation_health_score,
                'engagement_forecast': result.engagement_forecast,
                'adaptive_strategy': result.adaptive_strategy,
                'priority_actions': result.priority_actions
            }
        })
        
        # Update session state
        if result.script_adaptation.get('adaptation_type'):
            session.current_stage = result.script_adaptation.get('adaptation_type', session.current_stage)
        
        session.engagement_trend = result.engagement_forecast
        
        # Update cultural context
        if result.language_analysis.get('language_detection'):
            session.cultural_context.update({
                'detected_language': result.language_analysis['language_detection'].get('detected_language'),
                'cultural_background': result.language_analysis.get('conversation_context', {}).get('cultural_background')
            })
        
        # Update emotional journey
        if result.emotion_analysis.get('combined_analysis'):
            session.emotional_journey.append({
                'timestamp': result.analysis_timestamp.isoformat(),
                'emotional_state': result.emotion_analysis['combined_analysis'].get('emotional_state'),
                'sentiment_score': result.emotion_analysis['combined_analysis'].get('sentiment_score'),
                'intensity': result.emotion_analysis['combined_analysis'].get('intensity')
            })
        
        # Update competitive context
        if result.competitive_analysis.get('competitive_mentions'):
            session.competitive_context.update({
                'active_competitors': [m.get('competitor_name') for m in result.competitive_analysis['competitive_mentions']],
                'competitive_intensity': result.competitive_analysis.get('competitive_landscape', {}).get('competitive_intensity', 0),
                'last_competitive_mention': result.analysis_timestamp.isoformat()
            })
    
    def _update_performance_metrics(self, result: ConversationIntelligenceResult):
        """Update system performance metrics"""
        self.performance_metrics['total_analyses'] += 1
        
        if result.script_adaptation.get('adaptation_type', 'maintain') != 'maintain':
            self.performance_metrics['successful_adaptations'] += 1
        
        if result.competitive_analysis.get('competitive_mentions'):
            self.performance_metrics['competitive_mentions_detected'] += len(result.competitive_analysis['competitive_mentions'])
        
        if result.empathy_recommendations.get('empathy_level', 'low') in ['high', 'medium']:
            self.performance_metrics['emotion_interventions'] += 1
        
        if result.language_analysis.get('translation'):
            self.performance_metrics['language_translations'] += 1
    
    async def generate_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Generate comprehensive conversation summary"""
        
        session = self.active_sessions.get(conversation_id)
        if not session:
            return {'error': 'Conversation session not found'}
        
        if not session.intelligence_timeline:
            return {'summary': 'No conversation intelligence data available'}
        
        # Analyze conversation journey
        timeline = session.intelligence_timeline
        
        # Health score trend
        health_scores = [result.conversation_health_score for result in timeline]
        health_trend = 'improving' if len(health_scores) > 1 and health_scores[-1] > health_scores[0] else 'stable'
        
        # Emotional journey
        emotional_states = [result.emotion_analysis.get('combined_analysis', {}).get('emotional_state', 'neutral') 
                          for result in timeline if 'error' not in result.emotion_analysis]
        
        # Competitive landscape
        all_competitors = []
        for result in timeline:
            mentions = result.competitive_analysis.get('competitive_mentions', [])
            all_competitors.extend([m.get('competitor_name') for m in mentions])
        
        # Language patterns
        languages_detected = [result.language_analysis.get('language_detection', {}).get('detected_language', 'en') 
                            for result in timeline if 'error' not in result.language_analysis]
        
        # Adaptation patterns
        adaptations = [result.script_adaptation.get('adaptation_type', 'maintain') 
                      for result in timeline if 'error' not in result.script_adaptation and result.script_adaptation.get('adaptation_type') != 'maintain']
        
        summary = {
            'conversation_id': conversation_id,
            'session_duration': len(timeline),
            'current_health_score': health_scores[-1] if health_scores else 0.5,
            'health_trend': health_trend,
            'engagement_forecast': timeline[-1].engagement_forecast if timeline else 'unknown',
            'conversation_insights': {
                'emotional_journey': {
                    'dominant_emotions': list(set(emotional_states)),
                    'emotional_stability': len(set(emotional_states)) <= 3,
                    'current_state': emotional_states[-1] if emotional_states else 'neutral'
                },
                'competitive_context': {
                    'competitors_mentioned': list(set(all_competitors)),
                    'competitive_pressure': len(all_competitors) > 0,
                    'most_mentioned_competitor': max(set(all_competitors), key=all_competitors.count) if all_competitors else None
                },
                'cultural_adaptation': {
                    'languages_detected': list(set(languages_detected)),
                    'primary_language': max(set(languages_detected), key=languages_detected.count) if languages_detected else 'en',
                    'localization_needed': len(set(languages_detected)) > 1 or 'en' not in languages_detected
                },
                'behavioral_patterns': {
                    'adaptations_made': len(adaptations),
                    'adaptation_types': list(set(adaptations)),
                    'adaptation_effectiveness': sum(result.script_adaptation.get('success_probability', 0.5) for result in timeline if 'error' not in result.script_adaptation) / len(timeline) if timeline else 0.5
                }
            },
            'recommendations': await self._generate_conversation_recommendations(session),
            'next_best_actions': timeline[-1].priority_actions if timeline else [],
            'success_indicators': await self._assess_conversation_success(session)
        }
        
        return summary
    
    async def _generate_conversation_recommendations(self, session: ConversationSession) -> List[str]:
        """Generate strategic recommendations for conversation"""
        recommendations = []
        
        if not session.intelligence_timeline:
            return ['Continue conversation with active listening']
        
        latest = session.intelligence_timeline[-1]
        
        # Health-based recommendations
        if latest.conversation_health_score < 0.4:
            recommendations.append("Conversation health is low - focus on relationship repair and value reinforcement")
        elif latest.conversation_health_score > 0.8:
            recommendations.append("Conversation is healthy - consider advancing to next stage")
        
        # Engagement-based recommendations
        if latest.engagement_forecast in ['concerning', 'critical']:
            recommendations.append("Engagement is declining - implement re-engagement strategy")
        elif latest.engagement_forecast == 'highly_positive':
            recommendations.append("High engagement detected - opportunity for acceleration")
        
        # Competitive recommendations
        if session.competitive_context.get('competitive_intensity', 0) > 0.5:
            recommendations.append("Active competitive situation - deploy battlecard and differentiation messaging")
        
        # Cultural recommendations
        if session.cultural_context.get('detected_language', 'en') != 'en':
            recommendations.append("Non-English speaker detected - ensure cultural sensitivity and consider localized materials")
        
        # Emotional recommendations
        if session.emotional_journey:
            recent_emotion = session.emotional_journey[-1].get('emotional_state', 'neutral')
            if recent_emotion in ['frustrated', 'worried']:
                recommendations.append("Address emotional concerns with empathy before proceeding")
            elif recent_emotion in ['excited', 'enthusiastic']:
                recommendations.append("Leverage positive emotional state to drive momentum")
        
        return recommendations[:5]
    
    async def _assess_conversation_success(self, session: ConversationSession) -> List[str]:
        """Assess conversation success indicators"""
        indicators = []
        
        if not session.intelligence_timeline:
            return ['Insufficient data for assessment']
        
        # Positive health trend
        health_scores = [result.conversation_health_score for result in session.intelligence_timeline]
        if len(health_scores) > 1 and health_scores[-1] > health_scores[0]:
            indicators.append("Conversation health improving over time")
        
        # High engagement
        latest = session.intelligence_timeline[-1]
        if latest.engagement_forecast in ['positive', 'highly_positive']:
            indicators.append("Strong participant engagement")
        
        # Successful adaptations
        adaptations = [result.script_adaptation.get('success_probability', 0.5) 
                      for result in session.intelligence_timeline if 'error' not in result.script_adaptation]
        if adaptations and sum(adaptations) / len(adaptations) > 0.7:
            indicators.append("Script adaptations showing high success probability")
        
        # Competitive advantage
        if not session.competitive_context.get('active_competitors'):
            indicators.append("No competitive threats detected")
        elif session.competitive_context.get('competitive_intensity', 0) < 0.3:
            indicators.append("Low competitive pressure")
        
        # Emotional positivity
        if session.emotional_journey:
            recent_emotions = [journey['emotional_state'] for journey in session.emotional_journey[-3:]]
            positive_emotions = ['enthusiastic', 'excited', 'confident', 'interested']
            if any(emotion in positive_emotions for emotion in recent_emotions):
                indicators.append("Positive emotional engagement")
        
        return indicators if indicators else ['Conversation proceeding normally']
    
    async def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        
        # Calculate additional metrics
        active_sessions_count = len(self.active_sessions)
        total_messages_analyzed = sum(len(session.intelligence_timeline) for session in self.active_sessions.values())
        
        # Performance rates
        adaptation_rate = (self.performance_metrics['successful_adaptations'] / 
                          max(self.performance_metrics['total_analyses'], 1))
        
        competitive_detection_rate = (self.performance_metrics['competitive_mentions_detected'] / 
                                    max(self.performance_metrics['total_analyses'], 1))
        
        return {
            'system_status': 'operational',
            'active_conversations': active_sessions_count,
            'total_messages_processed': total_messages_analyzed,
            'performance_metrics': self.performance_metrics,
            'performance_rates': {
                'adaptation_rate': adaptation_rate,
                'competitive_detection_rate': competitive_detection_rate,
                'emotion_intervention_rate': (self.performance_metrics['emotion_interventions'] / 
                                            max(self.performance_metrics['total_analyses'], 1)),
                'translation_rate': (self.performance_metrics['language_translations'] / 
                                   max(self.performance_metrics['total_analyses'], 1))
            },
            'system_health': {
                'multilingual_system': 'operational',
                'emotion_detection': 'operational',
                'script_adaptation': 'operational',
                'competitive_intelligence': 'operational'
            },
            'cache_status': {
                'cache_size': len(self.intelligence_cache),
                'active_sessions': active_sessions_count
            }
        }


# Global advanced conversation intelligence instance
advanced_conversation_intelligence = AdvancedConversationIntelligence()