"""
Dynamic Script Adaptation Engine
AI system that adapts conversation flow based on real-time prospect responses, behavioral cues, and engagement patterns
"""

import asyncio
import logging
import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import math
from collections import defaultdict, deque

# Machine learning and NLP
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ConversationStage(Enum):
    """Conversation stages in sales process"""
    OPENING = "opening"
    RAPPORT_BUILDING = "rapport_building"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"

class EngagementLevel(Enum):
    """Prospect engagement levels"""
    HIGHLY_ENGAGED = "highly_engaged"
    ENGAGED = "engaged"
    NEUTRAL = "neutral"
    DISENGAGED = "disengaged"
    RESISTANT = "resistant"

class ResponsePattern(Enum):
    """Patterns in prospect responses"""
    POSITIVE_SIGNAL = "positive_signal"
    INTEREST_INDICATOR = "interest_indicator"
    CONCERN_EXPRESSED = "concern_expressed"
    OBJECTION_RAISED = "objection_raised"
    BUYING_SIGNAL = "buying_signal"
    STALLING_TACTIC = "stalling_tactic"
    INFORMATION_REQUEST = "information_request"
    COMPETITOR_MENTION = "competitor_mention"

class AdaptationStrategy(Enum):
    """Script adaptation strategies"""
    ACCELERATE = "accelerate"
    SLOW_DOWN = "slow_down"
    REDIRECT = "redirect"
    DEEPEN = "deepen"
    SIMPLIFY = "simplify"
    PERSONALIZE = "personalize"
    DEMONSTRATE = "demonstrate"
    REASSURE = "reassure"

@dataclass
class BehavioralCue:
    """Individual behavioral cue detection"""
    cue_type: str
    confidence: float
    indicators: List[str]
    context: str
    timestamp: datetime
    impact_level: str

@dataclass
class EngagementMetrics:
    """Engagement measurement metrics"""
    attention_score: float
    interaction_frequency: float
    response_quality: float
    question_engagement: float
    objection_rate: float
    buying_signal_strength: float

@dataclass
class ConversationFlow:
    """Current conversation flow state"""
    current_stage: str
    stage_progress: float
    next_optimal_stage: str
    flow_velocity: float
    resistance_points: List[str]
    acceleration_opportunities: List[str]

@dataclass
class ScriptAdaptation:
    """Script adaptation recommendation"""
    adaptation_type: str
    confidence: float
    reasoning: str
    recommended_content: str
    tone_adjustment: str
    pacing_change: str
    focus_areas: List[str]
    success_probability: float

@dataclass
class ConversationContext:
    """Enhanced conversation context for adaptation"""
    conversation_id: str
    participant_profile: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    behavioral_patterns: Dict[str, Any]
    engagement_timeline: List[Tuple[datetime, EngagementMetrics]]
    adaptation_history: List[ScriptAdaptation]
    success_indicators: List[str]


class BehavioralAnalyzer:
    """Analyzes behavioral cues from prospect responses"""
    
    def __init__(self):
        self.nlp_models = {}
        self.behavioral_patterns = self._load_behavioral_patterns()
        self.engagement_indicators = self._load_engagement_indicators()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize behavioral analysis models"""
        try:
            # Initialize intent classification model
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            # Initialize sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy for linguistic analysis
            try:
                self.nlp_models['en'] = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy English model not available")
            
            logger.info("Behavioral analyzer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing behavioral models: {e}")
    
    def _load_behavioral_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load behavioral pattern recognition rules"""
        return {
            'positive_signals': {
                'keywords': ['interested', 'sounds good', 'tell me more', 'that makes sense', 'I like that'],
                'phrases': [
                    r'that sounds? (good|great|interesting|promising)',
                    r'i\'d like to (know|learn|hear) more',
                    r'what (would|does) that (mean|look like)',
                    r'how (would|does|can) that work'
                ],
                'indicators': ['questions_about_details', 'timeline_inquiries', 'pricing_questions']
            },
            'buying_signals': {
                'keywords': ['when', 'how long', 'next steps', 'implementation', 'contract', 'agreement'],
                'phrases': [
                    r'when (can|could|would) we (start|begin|implement)',
                    r'what are the next steps',
                    r'how long (does|would) (it|this) take',
                    r'what (would|does) the (process|timeline) look like'
                ],
                'indicators': ['timeline_focus', 'process_questions', 'decision_maker_involvement']
            },
            'objections': {
                'keywords': ['but', 'however', 'concern', 'worried', 'expensive', 'cost', 'budget'],
                'phrases': [
                    r'but what about',
                    r'i\'m (concerned|worried) about',
                    r'that (seems|sounds) (expensive|costly)',
                    r'we don\'t have (the )?budget'
                ],
                'indicators': ['price_resistance', 'feature_concerns', 'timing_issues']
            },
            'stalling_tactics': {
                'keywords': ['think about it', 'discuss', 'team', 'later', 'maybe'],
                'phrases': [
                    r'(let me|i need to) think about (it|this)',
                    r'i\'ll (discuss|talk) (to|with) (my|the) team',
                    r'maybe (we can|we should) (talk|discuss) later',
                    r'i\'m not ready to (decide|commit)'
                ],
                'indicators': ['delay_requests', 'team_consultation', 'information_gathering']
            },
            'disengagement': {
                'keywords': ['okay', 'sure', 'fine', 'whatever', 'i guess'],
                'phrases': [
                    r'^(okay|ok|sure|fine)\.?$',
                    r'i guess so',
                    r'if you say so',
                    r'whatever works'
                ],
                'indicators': ['short_responses', 'lack_of_questions', 'passive_agreement']
            }
        }
    
    def _load_engagement_indicators(self) -> Dict[str, List[str]]:
        """Load engagement indicator patterns"""
        return {
            'high_engagement': [
                'asks_detailed_questions',
                'provides_specific_examples',
                'shares_personal_experiences',
                'requests_demonstrations',
                'discusses_timeline',
                'mentions_budget_availability'
            ],
            'moderate_engagement': [
                'asks_clarifying_questions',
                'provides_general_feedback',
                'shows_mild_interest',
                'requests_more_information'
            ],
            'low_engagement': [
                'gives_short_responses',
                'avoids_specific_questions',
                'changes_subject',
                'shows_distraction_signs'
            ]
        }
    
    async def analyze_behavioral_cues(self, message: str, conversation_context: Dict[str, Any]) -> List[BehavioralCue]:
        """Analyze behavioral cues from prospect message"""
        try:
            cues = []
            
            # Analyze for different behavioral patterns
            for pattern_type, pattern_data in self.behavioral_patterns.items():
                detected_cues = await self._detect_pattern_cues(message, pattern_type, pattern_data)
                cues.extend(detected_cues)
            
            # Analyze linguistic features
            linguistic_cues = await self._analyze_linguistic_features(message)
            cues.extend(linguistic_cues)
            
            # Analyze response timing and length
            timing_cues = await self._analyze_response_characteristics(message, conversation_context)
            cues.extend(timing_cues)
            
            return cues
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral cues: {e}")
            return []
    
    async def _detect_pattern_cues(self, message: str, pattern_type: str, pattern_data: Dict[str, Any]) -> List[BehavioralCue]:
        """Detect specific behavioral pattern cues"""
        cues = []
        message_lower = message.lower()
        
        try:
            # Keyword detection
            keyword_matches = []
            for keyword in pattern_data.get('keywords', []):
                if keyword in message_lower:
                    keyword_matches.append(keyword)
            
            # Phrase pattern detection
            phrase_matches = []
            for phrase_pattern in pattern_data.get('phrases', []):
                if re.search(phrase_pattern, message_lower):
                    phrase_matches.append(phrase_pattern)
            
            # Calculate confidence based on matches
            total_indicators = len(pattern_data.get('keywords', [])) + len(pattern_data.get('phrases', []))
            matches = len(keyword_matches) + len(phrase_matches)
            confidence = min(matches / max(total_indicators * 0.3, 1), 1.0) if total_indicators > 0 else 0.0
            
            if confidence > 0.3:  # Threshold for detection
                impact_level = 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low'
                
                cue = BehavioralCue(
                    cue_type=pattern_type,
                    confidence=confidence,
                    indicators=keyword_matches + phrase_matches,
                    context=message,
                    timestamp=datetime.now(),
                    impact_level=impact_level
                )
                cues.append(cue)
                
        except Exception as e:
            logger.error(f"Error detecting pattern cues: {e}")
        
        return cues
    
    async def _analyze_linguistic_features(self, message: str) -> List[BehavioralCue]:
        """Analyze linguistic features for behavioral cues"""
        cues = []
        
        try:
            if 'en' not in self.nlp_models:
                return cues
            
            nlp = self.nlp_models['en']
            doc = nlp(message)
            
            # Question density analysis
            question_count = message.count('?')
            word_count = len(message.split())
            question_density = question_count / max(word_count, 1)
            
            if question_density > 0.1:  # High question density
                cue = BehavioralCue(
                    cue_type='high_question_engagement',
                    confidence=min(question_density * 5, 1.0),
                    indicators=[f'{question_count} questions in {word_count} words'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='medium'
                )
                cues.append(cue)
            
            # Uncertainty markers
            uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'not sure', 'i think']
            uncertainty_count = sum(1 for word in uncertainty_words if word in message.lower())
            
            if uncertainty_count > 0:
                confidence = min(uncertainty_count * 0.3, 1.0)
                cue = BehavioralCue(
                    cue_type='uncertainty_expressed',
                    confidence=confidence,
                    indicators=[f'{uncertainty_count} uncertainty markers'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='medium'
                )
                cues.append(cue)
            
            # Enthusiasm markers
            enthusiasm_markers = ['!', 'amazing', 'fantastic', 'great', 'excellent', 'perfect']
            enthusiasm_count = sum(1 for marker in enthusiasm_markers if marker in message.lower())
            exclamation_count = message.count('!')
            
            if enthusiasm_count > 0 or exclamation_count > 0:
                confidence = min((enthusiasm_count * 0.4 + exclamation_count * 0.2), 1.0)
                cue = BehavioralCue(
                    cue_type='enthusiasm_displayed',
                    confidence=confidence,
                    indicators=[f'{enthusiasm_count} enthusiasm words, {exclamation_count} exclamations'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='high'
                )
                cues.append(cue)
                
        except Exception as e:
            logger.error(f"Error analyzing linguistic features: {e}")
        
        return cues
    
    async def _analyze_response_characteristics(self, message: str, conversation_context: Dict[str, Any]) -> List[BehavioralCue]:
        """Analyze response timing and characteristics"""
        cues = []
        
        try:
            # Message length analysis
            word_count = len(message.split())
            
            if word_count < 5:  # Very short response
                cue = BehavioralCue(
                    cue_type='short_response',
                    confidence=0.8,
                    indicators=[f'Only {word_count} words'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='medium'
                )
                cues.append(cue)
            elif word_count > 50:  # Very detailed response
                cue = BehavioralCue(
                    cue_type='detailed_response',
                    confidence=0.7,
                    indicators=[f'{word_count} words - detailed engagement'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='high'
                )
                cues.append(cue)
            
            # Response complexity
            sentence_count = len([s for s in message.split('.') if s.strip()])
            complexity_score = sentence_count / max(word_count / 10, 1)
            
            if complexity_score > 1.5:
                cue = BehavioralCue(
                    cue_type='complex_response',
                    confidence=0.6,
                    indicators=[f'{sentence_count} sentences, complexity score: {complexity_score:.2f}'],
                    context=message,
                    timestamp=datetime.now(),
                    impact_level='medium'
                )
                cues.append(cue)
                
        except Exception as e:
            logger.error(f"Error analyzing response characteristics: {e}")
        
        return cues


class EngagementTracker:
    """Tracks and measures prospect engagement levels"""
    
    def __init__(self):
        self.engagement_history = {}
        self.engagement_thresholds = self._set_engagement_thresholds()
    
    def _set_engagement_thresholds(self) -> Dict[str, float]:
        """Set thresholds for engagement level classification"""
        return {
            'highly_engaged': 0.8,
            'engaged': 0.6,
            'neutral': 0.4,
            'disengaged': 0.2,
            'resistant': 0.0
        }
    
    async def calculate_engagement_metrics(self, 
                                        conversation_id: str,
                                        behavioral_cues: List[BehavioralCue],
                                        conversation_context: Dict[str, Any]) -> EngagementMetrics:
        """Calculate comprehensive engagement metrics"""
        try:
            # Attention score based on response quality and length
            attention_score = await self._calculate_attention_score(behavioral_cues, conversation_context)
            
            # Interaction frequency
            interaction_frequency = await self._calculate_interaction_frequency(conversation_id, conversation_context)
            
            # Response quality
            response_quality = await self._calculate_response_quality(behavioral_cues)
            
            # Question engagement
            question_engagement = await self._calculate_question_engagement(behavioral_cues)
            
            # Objection rate
            objection_rate = await self._calculate_objection_rate(behavioral_cues)
            
            # Buying signal strength
            buying_signal_strength = await self._calculate_buying_signal_strength(behavioral_cues)
            
            return EngagementMetrics(
                attention_score=attention_score,
                interaction_frequency=interaction_frequency,
                response_quality=response_quality,
                question_engagement=question_engagement,
                objection_rate=objection_rate,
                buying_signal_strength=buying_signal_strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return EngagementMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    
    async def _calculate_attention_score(self, behavioral_cues: List[BehavioralCue], context: Dict[str, Any]) -> float:
        """Calculate attention score based on behavioral cues"""
        # Base attention from response characteristics
        attention_indicators = [cue for cue in behavioral_cues if cue.cue_type in ['detailed_response', 'complex_response']]
        disengagement_indicators = [cue for cue in behavioral_cues if cue.cue_type in ['short_response', 'disengagement']]
        
        attention_boost = sum(cue.confidence * 0.3 for cue in attention_indicators)
        attention_penalty = sum(cue.confidence * 0.4 for cue in disengagement_indicators)
        
        base_score = 0.6
        final_score = base_score + attention_boost - attention_penalty
        
        return max(min(final_score, 1.0), 0.0)
    
    async def _calculate_interaction_frequency(self, conversation_id: str, context: Dict[str, Any]) -> float:
        """Calculate interaction frequency score"""
        # Simple implementation - in real scenario would track actual interaction timing
        message_count = context.get('message_count', 1)
        duration_minutes = context.get('duration_minutes', 1)
        
        frequency = message_count / max(duration_minutes, 1)
        normalized_frequency = min(frequency / 2.0, 1.0)  # Normalize to 0-1 scale
        
        return normalized_frequency
    
    async def _calculate_response_quality(self, behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate response quality score"""
        quality_indicators = [
            'positive_signals', 'buying_signals', 'high_question_engagement',
            'enthusiasm_displayed', 'detailed_response'
        ]
        
        quality_cues = [cue for cue in behavioral_cues if cue.cue_type in quality_indicators]
        
        if not quality_cues:
            return 0.4  # Neutral quality
        
        quality_score = sum(cue.confidence for cue in quality_cues) / len(quality_cues)
        return min(quality_score, 1.0)
    
    async def _calculate_question_engagement(self, behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate question engagement score"""
        question_cues = [cue for cue in behavioral_cues if 'question' in cue.cue_type]
        
        if not question_cues:
            return 0.3  # Low question engagement
        
        engagement_score = sum(cue.confidence for cue in question_cues) / len(question_cues)
        return min(engagement_score, 1.0)
    
    async def _calculate_objection_rate(self, behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate objection rate (higher is worse for engagement)"""
        objection_cues = [cue for cue in behavioral_cues if cue.cue_type in ['objections', 'stalling_tactics']]
        
        if not objection_cues:
            return 0.0  # No objections
        
        objection_score = sum(cue.confidence for cue in objection_cues) / len(objection_cues)
        return min(objection_score, 1.0)
    
    async def _calculate_buying_signal_strength(self, behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate buying signal strength"""
        buying_cues = [cue for cue in behavioral_cues if cue.cue_type == 'buying_signals']
        
        if not buying_cues:
            return 0.0  # No buying signals
        
        buying_score = sum(cue.confidence for cue in buying_cues) / len(buying_cues)
        return min(buying_score, 1.0)
    
    async def determine_engagement_level(self, metrics: EngagementMetrics) -> EngagementLevel:
        """Determine overall engagement level"""
        # Weighted engagement calculation
        overall_score = (
            metrics.attention_score * 0.25 +
            metrics.interaction_frequency * 0.15 +
            metrics.response_quality * 0.25 +
            metrics.question_engagement * 0.15 +
            (1 - metrics.objection_rate) * 0.1 +  # Lower objection rate is better
            metrics.buying_signal_strength * 0.1
        )
        
        # Classify based on thresholds
        if overall_score >= self.engagement_thresholds['highly_engaged']:
            return EngagementLevel.HIGHLY_ENGAGED
        elif overall_score >= self.engagement_thresholds['engaged']:
            return EngagementLevel.ENGAGED
        elif overall_score >= self.engagement_thresholds['neutral']:
            return EngagementLevel.NEUTRAL
        elif overall_score >= self.engagement_thresholds['disengaged']:
            return EngagementLevel.DISENGAGED
        else:
            return EngagementLevel.RESISTANT


class ConversationFlowManager:
    """Manages conversation flow and stage transitions"""
    
    def __init__(self):
        self.stage_transitions = self._define_stage_transitions()
        self.stage_indicators = self._define_stage_indicators()
    
    def _define_stage_transitions(self) -> Dict[str, Dict[str, float]]:
        """Define valid stage transitions and their probabilities"""
        return {
            ConversationStage.OPENING.value: {
                ConversationStage.RAPPORT_BUILDING.value: 0.8,
                ConversationStage.DISCOVERY.value: 0.2
            },
            ConversationStage.RAPPORT_BUILDING.value: {
                ConversationStage.DISCOVERY.value: 0.7,
                ConversationStage.PRESENTATION.value: 0.3
            },
            ConversationStage.DISCOVERY.value: {
                ConversationStage.PRESENTATION.value: 0.6,
                ConversationStage.OBJECTION_HANDLING.value: 0.3,
                ConversationStage.CLOSING.value: 0.1
            },
            ConversationStage.PRESENTATION.value: {
                ConversationStage.OBJECTION_HANDLING.value: 0.4,
                ConversationStage.CLOSING.value: 0.4,
                ConversationStage.DISCOVERY.value: 0.2
            },
            ConversationStage.OBJECTION_HANDLING.value: {
                ConversationStage.PRESENTATION.value: 0.4,
                ConversationStage.CLOSING.value: 0.3,
                ConversationStage.DISCOVERY.value: 0.3
            },
            ConversationStage.CLOSING.value: {
                ConversationStage.FOLLOW_UP.value: 0.6,
                ConversationStage.OBJECTION_HANDLING.value: 0.4
            }
        }
    
    def _define_stage_indicators(self) -> Dict[str, List[str]]:
        """Define indicators for each conversation stage"""
        return {
            ConversationStage.OPENING.value: [
                'greeting', 'introduction', 'agenda_setting', 'permission_asking'
            ],
            ConversationStage.RAPPORT_BUILDING.value: [
                'personal_connection', 'small_talk', 'common_ground', 'trust_building'
            ],
            ConversationStage.DISCOVERY.value: [
                'needs_assessment', 'problem_identification', 'current_situation', 'pain_points'
            ],
            ConversationStage.PRESENTATION.value: [
                'solution_presentation', 'benefit_explanation', 'feature_demonstration', 'value_proposition'
            ],
            ConversationStage.OBJECTION_HANDLING.value: [
                'concern_addressing', 'objection_response', 'clarification', 'reassurance'
            ],
            ConversationStage.CLOSING.value: [
                'decision_request', 'next_steps', 'commitment_seeking', 'agreement_confirmation'
            ],
            ConversationStage.FOLLOW_UP.value: [
                'action_items', 'timeline_confirmation', 'contact_scheduling', 'relationship_maintenance'
            ]
        }
    
    async def analyze_conversation_flow(self, 
                                      conversation_history: List[Dict[str, Any]],
                                      current_behavioral_cues: List[BehavioralCue]) -> ConversationFlow:
        """Analyze current conversation flow and recommend transitions"""
        try:
            # Determine current stage
            current_stage = await self._determine_current_stage(conversation_history, current_behavioral_cues)
            
            # Calculate stage progress
            stage_progress = await self._calculate_stage_progress(current_stage, conversation_history)
            
            # Determine optimal next stage
            next_optimal_stage = await self._determine_next_stage(current_stage, current_behavioral_cues)
            
            # Calculate flow velocity
            flow_velocity = await self._calculate_flow_velocity(conversation_history)
            
            # Identify resistance points
            resistance_points = await self._identify_resistance_points(current_behavioral_cues)
            
            # Identify acceleration opportunities
            acceleration_opportunities = await self._identify_acceleration_opportunities(current_behavioral_cues)
            
            return ConversationFlow(
                current_stage=current_stage,
                stage_progress=stage_progress,
                next_optimal_stage=next_optimal_stage,
                flow_velocity=flow_velocity,
                resistance_points=resistance_points,
                acceleration_opportunities=acceleration_opportunities
            )
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            return ConversationFlow(
                current_stage=ConversationStage.DISCOVERY.value,
                stage_progress=0.5,
                next_optimal_stage=ConversationStage.PRESENTATION.value,
                flow_velocity=0.5,
                resistance_points=[],
                acceleration_opportunities=[]
            )
    
    async def _determine_current_stage(self, 
                                     conversation_history: List[Dict[str, Any]],
                                     behavioral_cues: List[BehavioralCue]) -> str:
        """Determine current conversation stage"""
        # Analyze recent conversation content for stage indicators
        recent_messages = conversation_history[-5:] if conversation_history else []
        
        stage_scores = {}
        for stage in ConversationStage:
            stage_scores[stage.value] = 0.0
        
        # Score based on content analysis
        for message in recent_messages:
            content = message.get('content', '').lower()
            for stage, indicators in self.stage_indicators.items():
                for indicator in indicators:
                    if indicator.replace('_', ' ') in content:
                        stage_scores[stage] += 1.0
        
        # Boost scores based on behavioral cues
        for cue in behavioral_cues:
            if cue.cue_type == 'buying_signals':
                stage_scores[ConversationStage.CLOSING.value] += cue.confidence * 2
            elif cue.cue_type == 'objections':
                stage_scores[ConversationStage.OBJECTION_HANDLING.value] += cue.confidence * 2
            elif cue.cue_type == 'positive_signals':
                stage_scores[ConversationStage.PRESENTATION.value] += cue.confidence
        
        # Return stage with highest score
        return max(stage_scores.items(), key=lambda x: x[1])[0]
    
    async def _calculate_stage_progress(self, current_stage: str, conversation_history: List[Dict[str, Any]]) -> float:
        """Calculate progress through current stage"""
        # Simple implementation based on conversation length in current stage
        # In real implementation, would track stage-specific milestones
        
        stage_messages = 0
        for message in reversed(conversation_history):
            if message.get('stage') == current_stage:
                stage_messages += 1
            else:
                break
        
        # Progress based on message count (arbitrary scaling)
        expected_messages_per_stage = 5
        progress = min(stage_messages / expected_messages_per_stage, 1.0)
        
        return progress
    
    async def _determine_next_stage(self, current_stage: str, behavioral_cues: List[BehavioralCue]) -> str:
        """Determine optimal next stage based on behavioral cues"""
        # Get possible transitions
        possible_transitions = self.stage_transitions.get(current_stage, {})
        
        if not possible_transitions:
            return current_stage  # Stay in current stage
        
        # Adjust transition probabilities based on behavioral cues
        adjusted_probabilities = possible_transitions.copy()
        
        for cue in behavioral_cues:
            if cue.cue_type == 'buying_signals' and ConversationStage.CLOSING.value in adjusted_probabilities:
                adjusted_probabilities[ConversationStage.CLOSING.value] *= (1 + cue.confidence)
            elif cue.cue_type == 'objections' and ConversationStage.OBJECTION_HANDLING.value in adjusted_probabilities:
                adjusted_probabilities[ConversationStage.OBJECTION_HANDLING.value] *= (1 + cue.confidence)
            elif cue.cue_type == 'positive_signals' and ConversationStage.PRESENTATION.value in adjusted_probabilities:
                adjusted_probabilities[ConversationStage.PRESENTATION.value] *= (1 + cue.confidence * 0.5)
        
        # Return stage with highest adjusted probability
        return max(adjusted_probabilities.items(), key=lambda x: x[1])[0]
    
    async def _calculate_flow_velocity(self, conversation_history: List[Dict[str, Any]]) -> float:
        """Calculate conversation flow velocity"""
        if len(conversation_history) < 3:
            return 0.5  # Default moderate velocity
        
        # Calculate stage changes over time
        stages = [msg.get('stage', 'unknown') for msg in conversation_history[-10:]]
        unique_stages = len(set(stages))
        
        # Velocity based on stage diversity (more stages = faster flow)
        velocity = min(unique_stages / 5.0, 1.0)  # Normalize to 0-1
        
        return velocity
    
    async def _identify_resistance_points(self, behavioral_cues: List[BehavioralCue]) -> List[str]:
        """Identify points of resistance in conversation flow"""
        resistance_points = []
        
        for cue in behavioral_cues:
            if cue.cue_type == 'objections' and cue.confidence > 0.6:
                resistance_points.append(f"Strong objection detected: {cue.indicators}")
            elif cue.cue_type == 'stalling_tactics' and cue.confidence > 0.5:
                resistance_points.append(f"Stalling behavior: {cue.indicators}")
            elif cue.cue_type == 'disengagement' and cue.confidence > 0.7:
                resistance_points.append(f"Disengagement signs: {cue.indicators}")
        
        return resistance_points
    
    async def _identify_acceleration_opportunities(self, behavioral_cues: List[BehavioralCue]) -> List[str]:
        """Identify opportunities to accelerate conversation flow"""
        opportunities = []
        
        for cue in behavioral_cues:
            if cue.cue_type == 'buying_signals' and cue.confidence > 0.6:
                opportunities.append(f"Strong buying signal: {cue.indicators}")
            elif cue.cue_type == 'positive_signals' and cue.confidence > 0.7:
                opportunities.append(f"High positivity: {cue.indicators}")
            elif cue.cue_type == 'high_question_engagement' and cue.confidence > 0.6:
                opportunities.append(f"High engagement: {cue.indicators}")
        
        return opportunities


class ScriptAdaptationEngine:
    """Core engine for dynamic script adaptation"""
    
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.engagement_tracker = EngagementTracker()
        self.flow_manager = ConversationFlowManager()
        self.adaptation_templates = self._load_adaptation_templates()
        self.conversation_contexts: Dict[str, ConversationContext] = {}
    
    async def initialize(self):
        """Initialize the script adaptation engine"""
        logger.info("Initializing Dynamic Script Adaptation Engine...")
        # Additional initialization if needed
        logger.info("Dynamic Script Adaptation Engine initialized successfully")
    
    def _load_adaptation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load script adaptation templates"""
        return {
            AdaptationStrategy.ACCELERATE.value: {
                'description': 'Speed up conversation flow toward closing',
                'triggers': ['buying_signals', 'high_engagement', 'positive_momentum'],
                'content_adjustments': [
                    'Focus on next steps',
                    'Present clear call-to-action',
                    'Create urgency appropriately'
                ],
                'tone': 'confident_decisive',
                'pacing': 'faster'
            },
            AdaptationStrategy.SLOW_DOWN.value: {
                'description': 'Reduce pace to address concerns',
                'triggers': ['confusion_detected', 'information_overload', 'hesitation'],
                'content_adjustments': [
                    'Break down complex concepts',
                    'Check for understanding',
                    'Allow processing time'
                ],
                'tone': 'patient_supportive',
                'pacing': 'slower'
            },
            AdaptationStrategy.REDIRECT.value: {
                'description': 'Redirect conversation to more productive path',
                'triggers': ['objections', 'resistance', 'off_topic'],
                'content_adjustments': [
                    'Acknowledge concerns',
                    'Pivot to value proposition',
                    'Refocus on benefits'
                ],
                'tone': 'diplomatic_refocusing',
                'pacing': 'steady'
            },
            AdaptationStrategy.DEEPEN.value: {
                'description': 'Go deeper into specific topics of interest',
                'triggers': ['specific_questions', 'detail_requests', 'technical_interest'],
                'content_adjustments': [
                    'Provide detailed explanations',
                    'Share technical specifications',
                    'Offer demonstrations'
                ],
                'tone': 'expert_informative',
                'pacing': 'thorough'
            },
            AdaptationStrategy.PERSONALIZE.value: {
                'description': 'Make conversation more personal and relevant',
                'triggers': ['personal_examples_shared', 'rapport_building', 'trust_signals'],
                'content_adjustments': [
                    'Use personal examples',
                    'Reference their specific situation',
                    'Connect to their values'
                ],
                'tone': 'personal_relatable',
                'pacing': 'conversational'
            },
            AdaptationStrategy.REASSURE.value: {
                'description': 'Provide reassurance and build confidence',
                'triggers': ['uncertainty', 'fear_signals', 'risk_concerns'],
                'content_adjustments': [
                    'Share success stories',
                    'Provide guarantees',
                    'Address risk mitigation'
                ],
                'tone': 'confident_reassuring',
                'pacing': 'measured'
            }
        }
    
    async def analyze_and_adapt(self, 
                              conversation_id: str,
                              participant_message: str,
                              conversation_history: List[Dict[str, Any]],
                              current_script_content: str) -> ScriptAdaptation:
        """Main method to analyze conversation and adapt script"""
        try:
            # Get or create conversation context
            context = await self._get_conversation_context(conversation_id, conversation_history)
            
            # Analyze behavioral cues
            behavioral_cues = await self.behavioral_analyzer.analyze_behavioral_cues(
                participant_message, 
                {'message_count': len(conversation_history), 'duration_minutes': 10}  # Simplified context
            )
            
            # Calculate engagement metrics
            engagement_metrics = await self.engagement_tracker.calculate_engagement_metrics(
                conversation_id, behavioral_cues, {}
            )
            
            # Analyze conversation flow
            conversation_flow = await self.flow_manager.analyze_conversation_flow(
                conversation_history, behavioral_cues
            )
            
            # Determine optimal adaptation strategy
            adaptation_strategy = await self._determine_adaptation_strategy(
                behavioral_cues, engagement_metrics, conversation_flow
            )
            
            # Generate adapted content
            adapted_content = await self._generate_adapted_content(
                adaptation_strategy, current_script_content, context, behavioral_cues
            )
            
            # Calculate success probability
            success_probability = await self._calculate_success_probability(
                adaptation_strategy, engagement_metrics, behavioral_cues
            )
            
            # Update conversation context
            await self._update_conversation_context(
                context, behavioral_cues, engagement_metrics, conversation_flow
            )
            
            adaptation = ScriptAdaptation(
                adaptation_type=adaptation_strategy,
                confidence=await self._calculate_adaptation_confidence(behavioral_cues),
                reasoning=await self._generate_adaptation_reasoning(behavioral_cues, engagement_metrics),
                recommended_content=adapted_content,
                tone_adjustment=self.adaptation_templates[adaptation_strategy]['tone'],
                pacing_change=self.adaptation_templates[adaptation_strategy]['pacing'],
                focus_areas=self.adaptation_templates[adaptation_strategy]['content_adjustments'],
                success_probability=success_probability
            )
            
            # Store adaptation in context
            context.adaptation_history.append(adaptation)
            
            return adaptation
            
        except Exception as e:
            logger.error(f"Error in script adaptation: {e}")
            return ScriptAdaptation(
                adaptation_type=AdaptationStrategy.SLOW_DOWN.value,
                confidence=0.5,
                reasoning="Error in adaptation analysis - using safe default",
                recommended_content=current_script_content,
                tone_adjustment="supportive",
                pacing_change="steady",
                focus_areas=["maintain_conversation"],
                success_probability=0.5
            )
    
    async def _get_conversation_context(self, conversation_id: str, conversation_history: List[Dict[str, Any]]) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                participant_profile={},
                conversation_history=conversation_history,
                behavioral_patterns={},
                engagement_timeline=[],
                adaptation_history=[],
                success_indicators=[]
            )
        
        context = self.conversation_contexts[conversation_id]
        context.conversation_history = conversation_history  # Update with latest history
        return context
    
    async def _determine_adaptation_strategy(self,
                                           behavioral_cues: List[BehavioralCue],
                                           engagement_metrics: EngagementMetrics,
                                           conversation_flow: ConversationFlow) -> str:
        """Determine optimal adaptation strategy"""
        
        # Score each strategy based on current conditions
        strategy_scores = {}
        for strategy in AdaptationStrategy:
            strategy_scores[strategy.value] = 0.0
        
        # Score based on behavioral cues
        for cue in behavioral_cues:
            if cue.cue_type == 'buying_signals':
                strategy_scores[AdaptationStrategy.ACCELERATE.value] += cue.confidence * 2
            elif cue.cue_type == 'objections':
                strategy_scores[AdaptationStrategy.REDIRECT.value] += cue.confidence * 1.5
                strategy_scores[AdaptationStrategy.REASSURE.value] += cue.confidence
            elif cue.cue_type == 'uncertainty_expressed':
                strategy_scores[AdaptationStrategy.SLOW_DOWN.value] += cue.confidence
                strategy_scores[AdaptationStrategy.REASSURE.value] += cue.confidence * 0.5
            elif cue.cue_type == 'detailed_response':
                strategy_scores[AdaptationStrategy.DEEPEN.value] += cue.confidence
            elif cue.cue_type == 'enthusiasm_displayed':
                strategy_scores[AdaptationStrategy.PERSONALIZE.value] += cue.confidence
                strategy_scores[AdaptationStrategy.ACCELERATE.value] += cue.confidence * 0.5
            elif cue.cue_type == 'disengagement':
                strategy_scores[AdaptationStrategy.REDIRECT.value] += cue.confidence
                strategy_scores[AdaptationStrategy.PERSONALIZE.value] += cue.confidence * 0.7
        
        # Adjust based on engagement level
        overall_engagement = (
            engagement_metrics.attention_score + 
            engagement_metrics.response_quality + 
            engagement_metrics.question_engagement
        ) / 3
        
        if overall_engagement > 0.7:
            strategy_scores[AdaptationStrategy.ACCELERATE.value] *= 1.5
        elif overall_engagement < 0.4:
            strategy_scores[AdaptationStrategy.SLOW_DOWN.value] *= 1.3
            strategy_scores[AdaptationStrategy.REDIRECT.value] *= 1.2
        
        # Consider conversation flow
        if len(conversation_flow.acceleration_opportunities) > 0:
            strategy_scores[AdaptationStrategy.ACCELERATE.value] *= 1.2
        if len(conversation_flow.resistance_points) > 0:
            strategy_scores[AdaptationStrategy.REASSURE.value] *= 1.3
        
        # Return strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    async def _generate_adapted_content(self,
                                      strategy: str,
                                      original_content: str,
                                      context: ConversationContext,
                                      behavioral_cues: List[BehavioralCue]) -> str:
        """Generate adapted content based on strategy"""
        
        template = self.adaptation_templates.get(strategy, {})
        adjustments = template.get('content_adjustments', [])
        
        # Start with original content
        adapted_content = original_content
        
        # Apply strategy-specific modifications
        if strategy == AdaptationStrategy.ACCELERATE.value:
            adapted_content += "\n\nBased on your positive response, I think we should move forward quickly. What would you like to do as the next step?"
            
        elif strategy == AdaptationStrategy.SLOW_DOWN.value:
            adapted_content = "Let me slow down and make sure I'm explaining this clearly. " + adapted_content
            adapted_content += " Does this make sense so far? Should I elaborate on any part?"
            
        elif strategy == AdaptationStrategy.REDIRECT.value:
            adapted_content = "I understand your concern. Let me address that by focusing on how this specifically helps you. " + adapted_content
            
        elif strategy == AdaptationStrategy.DEEPEN.value:
            adapted_content += "\n\nSince you're interested in the details, let me share more specific information about how this works..."
            
        elif strategy == AdaptationStrategy.PERSONALIZE.value:
            # Look for personal context in behavioral cues
            adapted_content = "Given your specific situation, " + adapted_content.lower()
            adapted_content += " This is particularly relevant for someone in your position."
            
        elif strategy == AdaptationStrategy.REASSURE.value:
            adapted_content = "I completely understand your concerns, and many of our successful clients felt the same way initially. " + adapted_content
            adapted_content += " We have guarantees in place to ensure your success."
        
        return adapted_content
    
    async def _calculate_success_probability(self,
                                           strategy: str,
                                           engagement_metrics: EngagementMetrics,
                                           behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate probability of adaptation success"""
        
        base_probability = 0.5
        
        # Boost based on engagement
        engagement_boost = (
            engagement_metrics.attention_score * 0.2 +
            engagement_metrics.response_quality * 0.2 +
            engagement_metrics.buying_signal_strength * 0.3
        )
        
        # Penalty for objections
        objection_penalty = engagement_metrics.objection_rate * 0.3
        
        # Strategy-specific adjustments
        strategy_multiplier = 1.0
        if strategy == AdaptationStrategy.ACCELERATE.value:
            # Success depends on buying signals
            strategy_multiplier = 1.0 + engagement_metrics.buying_signal_strength * 0.5
        elif strategy == AdaptationStrategy.REASSURE.value:
            # Success depends on addressing concerns effectively
            strategy_multiplier = 1.2  # Generally effective for reassurance
        
        probability = (base_probability + engagement_boost - objection_penalty) * strategy_multiplier
        return max(min(probability, 1.0), 0.1)
    
    async def _calculate_adaptation_confidence(self, behavioral_cues: List[BehavioralCue]) -> float:
        """Calculate confidence in adaptation decision"""
        if not behavioral_cues:
            return 0.3  # Low confidence without clear signals
        
        # Average confidence from behavioral cues
        avg_confidence = sum(cue.confidence for cue in behavioral_cues) / len(behavioral_cues)
        
        # Boost confidence if multiple cues point in same direction
        high_confidence_cues = [cue for cue in behavioral_cues if cue.confidence > 0.6]
        consensus_boost = len(high_confidence_cues) * 0.1
        
        final_confidence = min(avg_confidence + consensus_boost, 1.0)
        return final_confidence
    
    async def _generate_adaptation_reasoning(self,
                                           behavioral_cues: List[BehavioralCue],
                                           engagement_metrics: EngagementMetrics) -> str:
        """Generate human-readable reasoning for adaptation"""
        
        reasoning_parts = []
        
        # Summarize key behavioral cues
        if behavioral_cues:
            cue_summary = ", ".join([f"{cue.cue_type} (confidence: {cue.confidence:.1f})" 
                                   for cue in behavioral_cues[:3]])  # Top 3 cues
            reasoning_parts.append(f"Key behavioral signals: {cue_summary}")
        
        # Summarize engagement
        overall_engagement = (engagement_metrics.attention_score + engagement_metrics.response_quality) / 2
        if overall_engagement > 0.7:
            reasoning_parts.append("High engagement level detected")
        elif overall_engagement < 0.4:
            reasoning_parts.append("Low engagement requires intervention")
        
        # Specific recommendations
        if engagement_metrics.buying_signal_strength > 0.5:
            reasoning_parts.append("Strong buying signals suggest readiness to advance")
        if engagement_metrics.objection_rate > 0.5:
            reasoning_parts.append("Objections need to be addressed")
        
        return ". ".join(reasoning_parts) if reasoning_parts else "Standard conversation flow adaptation"
    
    async def _update_conversation_context(self,
                                         context: ConversationContext,
                                         behavioral_cues: List[BehavioralCue],
                                         engagement_metrics: EngagementMetrics,
                                         conversation_flow: ConversationFlow):
        """Update conversation context with latest analysis"""
        
        # Update engagement timeline
        context.engagement_timeline.append((datetime.now(), engagement_metrics))
        
        # Keep only recent engagement history
        cutoff_time = datetime.now() - timedelta(minutes=30)
        context.engagement_timeline = [
            (timestamp, metrics) for timestamp, metrics in context.engagement_timeline
            if timestamp > cutoff_time
        ]
        
        # Update behavioral patterns
        for cue in behavioral_cues:
            pattern_key = cue.cue_type
            if pattern_key not in context.behavioral_patterns:
                context.behavioral_patterns[pattern_key] = []
            context.behavioral_patterns[pattern_key].append({
                'timestamp': cue.timestamp.isoformat(),
                'confidence': cue.confidence,
                'indicators': cue.indicators
            })
        
        # Update success indicators
        if engagement_metrics.buying_signal_strength > 0.6:
            context.success_indicators.append(f"Strong buying signals at {datetime.now().isoformat()}")
        if engagement_metrics.response_quality > 0.7:
            context.success_indicators.append(f"High quality responses at {datetime.now().isoformat()}")
    
    async def get_adaptation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get insights about conversation adaptations"""
        context = self.conversation_contexts.get(conversation_id)
        if not context:
            return {'error': 'Conversation not found'}
        
        return {
            'conversation_id': conversation_id,
            'adaptation_history': [asdict(adaptation) for adaptation in context.adaptation_history],
            'behavioral_patterns': context.behavioral_patterns,
            'engagement_trends': await self._analyze_engagement_trends(context),
            'success_indicators': context.success_indicators,
            'recommendations': await self._generate_conversation_recommendations(context)
        }
    
    async def _analyze_engagement_trends(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze engagement trends over time"""
        if len(context.engagement_timeline) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate trend in overall engagement
        recent_engagement = []
        for timestamp, metrics in context.engagement_timeline[-5:]:  # Last 5 measurements
            overall_score = (metrics.attention_score + metrics.response_quality + metrics.question_engagement) / 3
            recent_engagement.append(overall_score)
        
        if len(recent_engagement) >= 2:
            trend = 'improving' if recent_engagement[-1] > recent_engagement[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'overall_trend': trend,
            'current_engagement': recent_engagement[-1] if recent_engagement else 0.5,
            'engagement_volatility': np.std(recent_engagement) if len(recent_engagement) > 1 else 0.0,
            'peak_engagement': max(recent_engagement) if recent_engagement else 0.5
        }
    
    async def _generate_conversation_recommendations(self, context: ConversationContext) -> List[str]:
        """Generate recommendations for conversation improvement"""
        recommendations = []
        
        # Analyze recent adaptations
        if context.adaptation_history:
            recent_adaptations = context.adaptation_history[-3:]
            adaptation_types = [adapt.adaptation_type for adapt in recent_adaptations]
            
            # Check for repeated adaptations
            if len(set(adaptation_types)) < len(adaptation_types):
                recommendations.append("Repeated adaptation strategies detected - consider alternative approaches")
        
        # Analyze behavioral patterns
        frequent_patterns = {}
        for pattern, occurrences in context.behavioral_patterns.items():
            if len(occurrences) > 2:  # Frequent pattern
                frequent_patterns[pattern] = len(occurrences)
        
        if 'objections' in frequent_patterns:
            recommendations.append("Frequent objections detected - consider addressing root concerns")
        
        if 'buying_signals' in frequent_patterns:
            recommendations.append("Multiple buying signals detected - opportunity to accelerate closing")
        
        # Analyze success indicators
        if len(context.success_indicators) > 3:
            recommendations.append("Strong success indicators - maintain current approach")
        elif len(context.success_indicators) < 1:
            recommendations.append("Limited success indicators - consider strategy adjustment")
        
        return recommendations


# Global dynamic script adaptation engine instance
script_adaptation_engine = ScriptAdaptationEngine()