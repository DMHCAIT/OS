"""
Advanced Emotion Detection System for Conversation Intelligence
Real-time emotional intelligence with sentiment analysis, empathy responses, and emotional state tracking
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

# Advanced emotion detection libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sentence_transformers import SentenceTransformer
    import librosa
    import soundfile as sf
    from scipy import signal
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass

# Text analysis
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import spacy
except ImportError:
    pass

logger = logging.getLogger(__name__)

class EmotionCategory(Enum):
    """Primary emotion categories"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

class EmotionalState(Enum):
    """Emotional states for sales conversations"""
    ENTHUSIASTIC = "enthusiastic"
    INTERESTED = "interested"
    NEUTRAL = "neutral"
    SKEPTICAL = "skeptical"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXCITED = "excited"
    WORRIED = "worried"
    CONFIDENT = "confident"
    DEFENSIVE = "defensive"

class SentimentPolarity(Enum):
    """Sentiment polarity levels"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    SLIGHTLY_POSITIVE = "slightly_positive"
    NEUTRAL = "neutral"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class EmotionAnalysisResult:
    """Comprehensive emotion analysis result"""
    primary_emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]
    sentiment_polarity: str
    sentiment_score: float
    emotional_state: str
    intensity: float
    stability: float
    authenticity_score: float

@dataclass
class VoiceEmotionFeatures:
    """Voice-based emotion features"""
    pitch_mean: float
    pitch_variance: float
    energy_level: float
    speaking_rate: float
    voice_quality: str
    tone_stability: float
    emotional_markers: List[str]

@dataclass
class TextEmotionFeatures:
    """Text-based emotion features"""
    sentiment_words: List[str]
    emotional_phrases: List[str]
    question_density: float
    exclamation_usage: float
    uncertainty_markers: List[str]
    confidence_indicators: List[str]

@dataclass
class EmotionalContext:
    """Emotional context and history"""
    conversation_id: str
    participant_id: str
    emotion_timeline: List[Tuple[datetime, EmotionAnalysisResult]]
    emotional_patterns: Dict[str, Any]
    trigger_events: List[Dict[str, Any]]
    adaptation_history: List[Dict[str, Any]]

@dataclass
class EmpathyResponse:
    """Empathy-driven response recommendation"""
    response_type: str
    empathy_level: str
    recommended_phrases: List[str]
    tone_adjustment: str
    emotional_validation: str
    next_action: str
    sensitivity_notes: str


class VoiceEmotionAnalyzer:
    """Analyzes emotions from voice audio"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.window_size = 1024
        self.hop_length = 512
        self.emotion_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize voice emotion models"""
        try:
            # Initialize any pre-trained voice emotion models
            # Placeholder for actual model initialization
            self.emotion_models['voice_classifier'] = None
            logger.info("Voice emotion analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing voice emotion models: {e}")
    
    async def analyze_voice_emotions(self, audio_data: np.ndarray) -> VoiceEmotionFeatures:
        """Analyze emotions from voice audio"""
        try:
            # Extract acoustic features
            features = await self._extract_acoustic_features(audio_data)
            
            # Analyze emotional markers
            emotional_markers = await self._detect_voice_emotional_markers(features)
            
            return VoiceEmotionFeatures(
                pitch_mean=features.get('pitch_mean', 0.0),
                pitch_variance=features.get('pitch_variance', 0.0),
                energy_level=features.get('energy_level', 0.0),
                speaking_rate=features.get('speaking_rate', 0.0),
                voice_quality=features.get('voice_quality', 'normal'),
                tone_stability=features.get('tone_stability', 0.5),
                emotional_markers=emotional_markers
            )
            
        except Exception as e:
            logger.error(f"Error in voice emotion analysis: {e}")
            return VoiceEmotionFeatures(
                pitch_mean=0.0,
                pitch_variance=0.0,
                energy_level=0.0,
                speaking_rate=0.0,
                voice_quality='unknown',
                tone_stability=0.5,
                emotional_markers=[]
            )
    
    async def _extract_acoustic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract acoustic features from audio"""
        try:
            # Fundamental frequency (pitch)
            f0 = librosa.yin(audio_data, fmin=80, fmax=400)
            pitch_mean = np.mean(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
            pitch_variance = np.var(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
            
            # Energy features
            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            energy_level = np.mean(rms)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            
            # Zero crossing rate (indicates voicing)
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)[0]
            
            # Speaking rate estimation
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
            speaking_rate = len(onset_frames) / (len(audio_data) / self.sample_rate)
            
            # Voice quality indicators
            voice_quality = 'normal'
            if np.mean(zcr) > 0.1:
                voice_quality = 'breathy'
            elif pitch_variance > 1000:
                voice_quality = 'shaky'
            elif energy_level < 0.01:
                voice_quality = 'weak'
            
            # Tone stability
            tone_stability = 1.0 / (1.0 + pitch_variance / 1000)
            
            return {
                'pitch_mean': float(pitch_mean),
                'pitch_variance': float(pitch_variance),
                'energy_level': float(energy_level),
                'speaking_rate': float(speaking_rate),
                'voice_quality': voice_quality,
                'tone_stability': float(tone_stability),
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'zcr': float(np.mean(zcr))
            }
            
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            return {}
    
    async def _detect_voice_emotional_markers(self, features: Dict[str, float]) -> List[str]:
        """Detect emotional markers in voice features"""
        markers = []
        
        try:
            # High pitch variance indicates excitement or stress
            if features.get('pitch_variance', 0) > 1500:
                markers.append('high_arousal')
            
            # Low energy might indicate sadness or fatigue
            if features.get('energy_level', 0) < 0.02:
                markers.append('low_energy')
            
            # Fast speaking rate might indicate excitement or nervousness
            if features.get('speaking_rate', 0) > 3.0:
                markers.append('rapid_speech')
            
            # Breathy voice quality might indicate nervousness
            if features.get('voice_quality') == 'breathy':
                markers.append('nervousness')
            
            # Low tone stability might indicate uncertainty
            if features.get('tone_stability', 0.5) < 0.3:
                markers.append('uncertainty')
                
        except Exception as e:
            logger.error(f"Error detecting voice emotional markers: {e}")
        
        return markers


class TextEmotionAnalyzer:
    """Analyzes emotions from text using NLP"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_classifier = None
        self.nlp_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize text emotion analysis models"""
        try:
            # Initialize NLTK sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize emotion classification models
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Load spaCy model for linguistic features
            try:
                self.nlp_models['en'] = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy English model not available")
            
            logger.info("Text emotion analyzer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing text emotion models: {e}")
    
    async def analyze_text_emotions(self, text: str) -> EmotionAnalysisResult:
        """Comprehensive text emotion analysis"""
        try:
            # Basic sentiment analysis
            sentiment_scores = await self._analyze_sentiment(text)
            
            # Emotion classification
            emotion_scores = await self._classify_emotions(text)
            
            # Extract text features
            text_features = await self._extract_text_features(text)
            
            # Determine primary emotion
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            emotion_confidence = emotion_scores[primary_emotion]
            
            # Determine sentiment polarity
            sentiment_polarity = await self._determine_sentiment_polarity(sentiment_scores['compound'])
            
            # Determine emotional state
            emotional_state = await self._determine_emotional_state(emotion_scores, text_features)
            
            # Calculate intensity and stability
            intensity = await self._calculate_emotion_intensity(emotion_scores, text_features)
            stability = await self._calculate_emotion_stability(text_features)
            
            # Calculate authenticity score
            authenticity_score = await self._calculate_authenticity(text, emotion_scores, text_features)
            
            return EmotionAnalysisResult(
                primary_emotion=primary_emotion,
                emotion_confidence=emotion_confidence,
                emotion_scores=emotion_scores,
                sentiment_polarity=sentiment_polarity,
                sentiment_score=sentiment_scores['compound'],
                emotional_state=emotional_state,
                intensity=intensity,
                stability=stability,
                authenticity_score=authenticity_score
            )
            
        except Exception as e:
            logger.error(f"Error in text emotion analysis: {e}")
            return EmotionAnalysisResult(
                primary_emotion="neutral",
                emotion_confidence=0.5,
                emotion_scores={"neutral": 0.5},
                sentiment_polarity="neutral",
                sentiment_score=0.0,
                emotional_state="neutral",
                intensity=0.5,
                stability=0.5,
                authenticity_score=0.5
            )
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods"""
        try:
            # NLTK VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment (as backup)
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine scores
            return {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    async def _classify_emotions(self, text: str) -> Dict[str, float]:
        """Classify emotions using transformer models"""
        try:
            if not self.emotion_classifier:
                return {"neutral": 1.0}
            
            # Get emotion predictions
            emotions = self.emotion_classifier(text)
            
            # Convert to emotion scores dictionary
            emotion_scores = {}
            for emotion in emotions[0]:  # emotions is a list of lists
                emotion_scores[emotion['label'].lower()] = emotion['score']
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion classification: {e}")
            return {"neutral": 1.0}
    
    async def _extract_text_features(self, text: str) -> TextEmotionFeatures:
        """Extract emotion-relevant text features"""
        try:
            # Sentiment words
            sentiment_words = await self._extract_sentiment_words(text)
            
            # Emotional phrases
            emotional_phrases = await self._extract_emotional_phrases(text)
            
            # Question density
            question_count = text.count('?')
            question_density = question_count / len(text.split()) if text.split() else 0
            
            # Exclamation usage
            exclamation_count = text.count('!')
            exclamation_usage = exclamation_count / len(text.split()) if text.split() else 0
            
            # Uncertainty markers
            uncertainty_markers = await self._find_uncertainty_markers(text)
            
            # Confidence indicators
            confidence_indicators = await self._find_confidence_indicators(text)
            
            return TextEmotionFeatures(
                sentiment_words=sentiment_words,
                emotional_phrases=emotional_phrases,
                question_density=question_density,
                exclamation_usage=exclamation_usage,
                uncertainty_markers=uncertainty_markers,
                confidence_indicators=confidence_indicators
            )
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return TextEmotionFeatures([], [], 0.0, 0.0, [], [])
    
    async def _extract_sentiment_words(self, text: str) -> List[str]:
        """Extract words with strong sentiment"""
        positive_words = ['excellent', 'amazing', 'fantastic', 'great', 'wonderful', 'perfect', 'love', 'excited']
        negative_words = ['terrible', 'awful', 'horrible', 'hate', 'disappointed', 'frustrated', 'angry', 'worried']
        
        words = text.lower().split()
        sentiment_words = []
        
        for word in words:
            if word in positive_words or word in negative_words:
                sentiment_words.append(word)
        
        return sentiment_words
    
    async def _extract_emotional_phrases(self, text: str) -> List[str]:
        """Extract phrases with emotional content"""
        emotional_patterns = [
            r'i feel \w+',
            r'i\'m (so|very|really) \w+',
            r'that (makes me|is) \w+',
            r'i\'m (excited|worried|concerned|happy|frustrated)',
            r'this is (amazing|terrible|perfect|awful)',
        ]
        
        emotional_phrases = []
        for pattern in emotional_patterns:
            matches = re.findall(pattern, text.lower())
            emotional_phrases.extend(matches)
        
        return emotional_phrases
    
    async def _find_uncertainty_markers(self, text: str) -> List[str]:
        """Find markers of uncertainty in text"""
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain', 'not sure', 'i think']
        words = text.lower()
        
        found_markers = []
        for marker in uncertainty_words:
            if marker in words:
                found_markers.append(marker)
        
        return found_markers
    
    async def _find_confidence_indicators(self, text: str) -> List[str]:
        """Find indicators of confidence in text"""
        confidence_words = ['definitely', 'absolutely', 'certainly', 'sure', 'confident', 'positive', 'convinced']
        words = text.lower()
        
        found_indicators = []
        for indicator in confidence_words:
            if indicator in words:
                found_indicators.append(indicator)
        
        return found_indicators
    
    async def _determine_sentiment_polarity(self, compound_score: float) -> str:
        """Determine sentiment polarity from compound score"""
        if compound_score >= 0.6:
            return SentimentPolarity.VERY_POSITIVE.value
        elif compound_score >= 0.2:
            return SentimentPolarity.POSITIVE.value
        elif compound_score >= 0.05:
            return SentimentPolarity.SLIGHTLY_POSITIVE.value
        elif compound_score >= -0.05:
            return SentimentPolarity.NEUTRAL.value
        elif compound_score >= -0.2:
            return SentimentPolarity.SLIGHTLY_NEGATIVE.value
        elif compound_score >= -0.6:
            return SentimentPolarity.NEGATIVE.value
        else:
            return SentimentPolarity.VERY_NEGATIVE.value
    
    async def _determine_emotional_state(self, emotion_scores: Dict[str, float], text_features: TextEmotionFeatures) -> str:
        """Determine overall emotional state"""
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Map emotions to sales-relevant states
        emotion_to_state = {
            'joy': EmotionalState.ENTHUSIASTIC.value,
            'surprise': EmotionalState.INTERESTED.value,
            'fear': EmotionalState.WORRIED.value,
            'anger': EmotionalState.FRUSTRATED.value,
            'sadness': EmotionalState.CONFUSED.value,
            'anticipation': EmotionalState.EXCITED.value,
            'trust': EmotionalState.CONFIDENT.value,
            'disgust': EmotionalState.DEFENSIVE.value
        }
        
        # Consider uncertainty markers
        if text_features.uncertainty_markers:
            return EmotionalState.SKEPTICAL.value
        
        # Consider confidence indicators
        if text_features.confidence_indicators:
            return EmotionalState.CONFIDENT.value
        
        return emotion_to_state.get(primary_emotion, EmotionalState.NEUTRAL.value)
    
    async def _calculate_emotion_intensity(self, emotion_scores: Dict[str, float], text_features: TextEmotionFeatures) -> float:
        """Calculate emotion intensity"""
        # Base intensity from highest emotion score
        max_emotion_score = max(emotion_scores.values())
        
        # Boost from exclamation marks
        exclamation_boost = min(text_features.exclamation_usage * 0.3, 0.3)
        
        # Boost from strong sentiment words
        sentiment_boost = min(len(text_features.sentiment_words) * 0.1, 0.2)
        
        intensity = max_emotion_score + exclamation_boost + sentiment_boost
        return min(intensity, 1.0)
    
    async def _calculate_emotion_stability(self, text_features: TextEmotionFeatures) -> float:
        """Calculate emotion stability"""
        # High question density might indicate instability
        question_penalty = text_features.question_density * 0.3
        
        # Uncertainty markers reduce stability
        uncertainty_penalty = len(text_features.uncertainty_markers) * 0.1
        
        # Confidence indicators increase stability
        confidence_boost = len(text_features.confidence_indicators) * 0.1
        
        stability = 0.7 - question_penalty - uncertainty_penalty + confidence_boost
        return max(min(stability, 1.0), 0.0)
    
    async def _calculate_authenticity(self, text: str, emotion_scores: Dict[str, float], text_features: TextEmotionFeatures) -> float:
        """Calculate authenticity score of expressed emotions"""
        # Base authenticity from emotion confidence
        base_authenticity = max(emotion_scores.values())
        
        # Check for emotional phrase consistency
        phrase_consistency = len(text_features.emotional_phrases) * 0.1
        
        # Check for mixed signals (reduce authenticity)
        if text_features.sentiment_words and text_features.uncertainty_markers:
            mixed_signal_penalty = 0.2
        else:
            mixed_signal_penalty = 0.0
        
        authenticity = base_authenticity + phrase_consistency - mixed_signal_penalty
        return max(min(authenticity, 1.0), 0.0)


class EmpathyEngine:
    """Generates empathetic responses based on emotional analysis"""
    
    def __init__(self):
        self.empathy_templates = self._load_empathy_templates()
        self.response_strategies = self._load_response_strategies()
    
    def _load_empathy_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load empathy response templates"""
        return {
            'acknowledgment': {
                'frustrated': [
                    "I can understand why that would be frustrating.",
                    "That does sound like a challenging situation.",
                    "I hear the frustration in your voice."
                ],
                'worried': [
                    "I can sense your concern about this.",
                    "It's natural to have concerns about such important decisions.",
                    "I understand why this would be worrying for you."
                ],
                'excited': [
                    "I can feel your excitement about this opportunity!",
                    "Your enthusiasm is wonderful to hear.",
                    "It's great to see you so energetic about this."
                ],
                'confused': [
                    "I can see this might be confusing.",
                    "Let me help clarify this for you.",
                    "I understand this can be a lot to process."
                ]
            },
            'validation': {
                'frustrated': [
                    "Your frustration is completely valid.",
                    "Anyone would feel the same way in your situation.",
                    "You have every right to feel this way."
                ],
                'worried': [
                    "Your concerns are completely understandable.",
                    "It's smart to think carefully about this.",
                    "Being cautious shows good judgment."
                ],
                'excited': [
                    "Your excitement is infectious!",
                    "It's wonderful to see such enthusiasm.",
                    "Your positive energy is amazing."
                ]
            },
            'support': {
                'frustrated': [
                    "Let's work together to find a solution.",
                    "I'm here to help make this easier for you.",
                    "We can definitely address these concerns."
                ],
                'worried': [
                    "I'm here to answer any questions you have.",
                    "Let's go through this step by step together.",
                    "I'll make sure you have all the information you need."
                ],
                'confused': [
                    "Let me break this down in a simpler way.",
                    "I'll walk you through this step by step.",
                    "Don't worry, we'll figure this out together."
                ]
            }
        }
    
    def _load_response_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load response strategies for different emotional states"""
        return {
            'frustrated': {
                'approach': 'solution_focused',
                'tone': 'calm_supportive',
                'pace': 'slower',
                'validation_level': 'high',
                'next_steps': ['acknowledge_issue', 'propose_solution', 'offer_support']
            },
            'worried': {
                'approach': 'reassurance_focused',
                'tone': 'confident_gentle',
                'pace': 'moderate',
                'validation_level': 'medium',
                'next_steps': ['address_concerns', 'provide_evidence', 'offer_guarantees']
            },
            'excited': {
                'approach': 'momentum_building',
                'tone': 'enthusiastic_professional',
                'pace': 'matching_energy',
                'validation_level': 'high',
                'next_steps': ['channel_excitement', 'move_forward', 'maintain_enthusiasm']
            },
            'confused': {
                'approach': 'clarification_focused',
                'tone': 'patient_educational',
                'pace': 'slower',
                'validation_level': 'medium',
                'next_steps': ['simplify_explanation', 'check_understanding', 'provide_examples']
            },
            'skeptical': {
                'approach': 'evidence_based',
                'tone': 'professional_credible',
                'pace': 'measured',
                'validation_level': 'medium',
                'next_steps': ['provide_proof', 'share_testimonials', 'offer_trial']
            },
            'defensive': {
                'approach': 'non_confrontational',
                'tone': 'respectful_understanding',
                'pace': 'calm',
                'validation_level': 'high',
                'next_steps': ['de_escalate', 'find_common_ground', 'rebuild_trust']
            }
        }
    
    async def generate_empathy_response(self, emotion_analysis: EmotionAnalysisResult, context: str = "") -> EmpathyResponse:
        """Generate empathetic response based on emotion analysis"""
        try:
            emotional_state = emotion_analysis.emotional_state
            primary_emotion = emotion_analysis.primary_emotion
            
            # Get response strategy
            strategy = self.response_strategies.get(
                emotional_state, 
                self.response_strategies['confused']  # Default strategy
            )
            
            # Select appropriate empathy phrases
            acknowledgment_phrases = self._select_empathy_phrases('acknowledgment', emotional_state)
            validation_phrases = self._select_empathy_phrases('validation', emotional_state)
            support_phrases = self._select_empathy_phrases('support', emotional_state)
            
            # Determine empathy level
            empathy_level = await self._determine_empathy_level(emotion_analysis)
            
            # Generate tone adjustment
            tone_adjustment = strategy['tone']
            
            # Create emotional validation
            emotional_validation = await self._create_emotional_validation(emotion_analysis)
            
            # Determine next action
            next_action = await self._determine_next_action(strategy, emotion_analysis)
            
            # Generate sensitivity notes
            sensitivity_notes = await self._generate_sensitivity_notes(emotion_analysis, emotional_state)
            
            return EmpathyResponse(
                response_type=strategy['approach'],
                empathy_level=empathy_level,
                recommended_phrases=acknowledgment_phrases + validation_phrases + support_phrases,
                tone_adjustment=tone_adjustment,
                emotional_validation=emotional_validation,
                next_action=next_action,
                sensitivity_notes=sensitivity_notes
            )
            
        except Exception as e:
            logger.error(f"Error generating empathy response: {e}")
            return EmpathyResponse(
                response_type="supportive",
                empathy_level="medium",
                recommended_phrases=["I understand your perspective."],
                tone_adjustment="calm_professional",
                emotional_validation="Your feelings are valid.",
                next_action="continue_conversation",
                sensitivity_notes="Maintain professional empathy."
            )
    
    def _select_empathy_phrases(self, phrase_type: str, emotional_state: str) -> List[str]:
        """Select appropriate empathy phrases"""
        templates = self.empathy_templates.get(phrase_type, {})
        phrases = templates.get(emotional_state, templates.get('confused', ["I understand."]))
        
        # Return 1-2 phrases to avoid overwhelming
        return phrases[:2]
    
    async def _determine_empathy_level(self, emotion_analysis: EmotionAnalysisResult) -> str:
        """Determine appropriate empathy level"""
        intensity = emotion_analysis.intensity
        authenticity = emotion_analysis.authenticity_score
        
        if intensity > 0.7 and authenticity > 0.6:
            return "high"
        elif intensity > 0.4 or authenticity > 0.5:
            return "medium"
        else:
            return "low"
    
    async def _create_emotional_validation(self, emotion_analysis: EmotionAnalysisResult) -> str:
        """Create emotional validation message"""
        emotional_state = emotion_analysis.emotional_state
        
        validation_messages = {
            'frustrated': "Your frustration is completely understandable.",
            'worried': "It's natural to have concerns about this.",
            'excited': "Your enthusiasm is wonderful!",
            'confused': "This can be a lot to take in.",
            'skeptical': "Healthy skepticism shows good judgment.",
            'defensive': "I respect your position on this."
        }
        
        return validation_messages.get(emotional_state, "I understand how you're feeling.")
    
    async def _determine_next_action(self, strategy: Dict[str, Any], emotion_analysis: EmotionAnalysisResult) -> str:
        """Determine the next recommended action"""
        next_steps = strategy.get('next_steps', ['continue_conversation'])
        
        # Select first step as primary next action
        primary_action = next_steps[0] if next_steps else 'continue_conversation'
        
        # Adjust based on emotion intensity
        if emotion_analysis.intensity > 0.8:
            if primary_action == 'move_forward':
                return 'proceed_carefully'
            elif primary_action == 'continue_conversation':
                return 'address_emotion_first'
        
        return primary_action
    
    async def _generate_sensitivity_notes(self, emotion_analysis: EmotionAnalysisResult, emotional_state: str) -> str:
        """Generate sensitivity notes for the conversation"""
        notes = []
        
        if emotion_analysis.intensity > 0.7:
            notes.append("High emotional intensity detected - proceed with extra care")
        
        if emotion_analysis.authenticity_score < 0.5:
            notes.append("Emotional authenticity uncertain - watch for mixed signals")
        
        if emotional_state in ['frustrated', 'defensive']:
            notes.append("Avoid confrontational language")
        
        if emotional_state == 'worried':
            notes.append("Provide reassurance and concrete information")
        
        if emotional_state == 'confused':
            notes.append("Speak slowly and check understanding frequently")
        
        return "; ".join(notes) if notes else "Maintain professional empathy"


class EmotionConversationManager:
    """Manages emotion-aware conversations"""
    
    def __init__(self):
        self.voice_analyzer = VoiceEmotionAnalyzer()
        self.text_analyzer = TextEmotionAnalyzer()
        self.empathy_engine = EmpathyEngine()
        self.emotional_contexts: Dict[str, EmotionalContext] = {}
        self.emotion_history_window = timedelta(minutes=30)
    
    async def initialize(self):
        """Initialize the emotion conversation manager"""
        logger.info("Initializing Emotion Conversation Manager...")
        # Additional initialization if needed
        logger.info("Emotion Conversation Manager initialized successfully")
    
    async def analyze_conversation_emotions(self,
                                          conversation_id: str,
                                          participant_id: str,
                                          text: str,
                                          audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive emotion analysis for conversation"""
        try:
            # Text emotion analysis
            text_emotion = await self.text_analyzer.analyze_text_emotions(text)
            
            # Voice emotion analysis (if audio provided)
            voice_features = None
            if audio_data is not None:
                voice_features = await self.voice_analyzer.analyze_voice_emotions(audio_data)
            
            # Combine analyses
            combined_analysis = await self._combine_emotion_analyses(text_emotion, voice_features)
            
            # Update emotional context
            await self._update_emotional_context(conversation_id, participant_id, combined_analysis)
            
            # Generate empathy response
            empathy_response = await self.empathy_engine.generate_empathy_response(combined_analysis)
            
            # Get conversation insights
            insights = await self._generate_conversation_insights(conversation_id, participant_id)
            
            return {
                'conversation_id': conversation_id,
                'participant_id': participant_id,
                'text_analysis': asdict(text_emotion),
                'voice_features': asdict(voice_features) if voice_features else None,
                'combined_analysis': asdict(combined_analysis),
                'empathy_response': asdict(empathy_response),
                'conversation_insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in conversation emotion analysis: {e}")
            return {
                'conversation_id': conversation_id,
                'participant_id': participant_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _combine_emotion_analyses(self, 
                                      text_emotion: EmotionAnalysisResult,
                                      voice_features: Optional[VoiceEmotionFeatures]) -> EmotionAnalysisResult:
        """Combine text and voice emotion analyses"""
        if not voice_features:
            return text_emotion
        
        try:
            # Weight text and voice analyses
            text_weight = 0.7
            voice_weight = 0.3
            
            # Adjust emotion confidence based on voice features
            adjusted_confidence = text_emotion.emotion_confidence
            
            # Voice markers can boost or reduce confidence
            if 'high_arousal' in voice_features.emotional_markers:
                if text_emotion.primary_emotion in ['anger', 'excitement', 'joy']:
                    adjusted_confidence = min(adjusted_confidence + 0.2, 1.0)
            
            if 'low_energy' in voice_features.emotional_markers:
                if text_emotion.primary_emotion in ['sadness', 'neutral']:
                    adjusted_confidence = min(adjusted_confidence + 0.1, 1.0)
            
            # Adjust intensity based on voice energy
            voice_intensity_boost = voice_features.energy_level * 0.3
            adjusted_intensity = min(text_emotion.intensity + voice_intensity_boost, 1.0)
            
            # Adjust stability based on tone stability
            adjusted_stability = (text_emotion.stability + voice_features.tone_stability) / 2
            
            return EmotionAnalysisResult(
                primary_emotion=text_emotion.primary_emotion,
                emotion_confidence=adjusted_confidence,
                emotion_scores=text_emotion.emotion_scores,
                sentiment_polarity=text_emotion.sentiment_polarity,
                sentiment_score=text_emotion.sentiment_score,
                emotional_state=text_emotion.emotional_state,
                intensity=adjusted_intensity,
                stability=adjusted_stability,
                authenticity_score=text_emotion.authenticity_score
            )
            
        except Exception as e:
            logger.error(f"Error combining emotion analyses: {e}")
            return text_emotion
    
    async def _update_emotional_context(self,
                                      conversation_id: str,
                                      participant_id: str,
                                      emotion_analysis: EmotionAnalysisResult):
        """Update emotional context for conversation"""
        try:
            # Get or create emotional context
            if conversation_id not in self.emotional_contexts:
                self.emotional_contexts[conversation_id] = EmotionalContext(
                    conversation_id=conversation_id,
                    participant_id=participant_id,
                    emotion_timeline=[],
                    emotional_patterns={},
                    trigger_events=[],
                    adaptation_history=[]
                )
            
            context = self.emotional_contexts[conversation_id]
            
            # Add to timeline
            context.emotion_timeline.append((datetime.now(), emotion_analysis))
            
            # Keep only recent history
            cutoff_time = datetime.now() - self.emotion_history_window
            context.emotion_timeline = [
                (timestamp, analysis) for timestamp, analysis in context.emotion_timeline
                if timestamp > cutoff_time
            ]
            
            # Update emotional patterns
            await self._update_emotional_patterns(context)
            
        except Exception as e:
            logger.error(f"Error updating emotional context: {e}")
    
    async def _update_emotional_patterns(self, context: EmotionalContext):
        """Update emotional patterns in conversation"""
        if not context.emotion_timeline:
            return
        
        # Calculate emotion distribution
        emotions = [analysis.primary_emotion for _, analysis in context.emotion_timeline]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate average intensity
        intensities = [analysis.intensity for _, analysis in context.emotion_timeline]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        
        # Calculate emotion stability
        stabilities = [analysis.stability for _, analysis in context.emotion_timeline]
        avg_stability = sum(stabilities) / len(stabilities) if stabilities else 0
        
        # Update patterns
        context.emotional_patterns.update({
            'dominant_emotions': emotion_counts,
            'average_intensity': avg_intensity,
            'average_stability': avg_stability,
            'emotion_changes': len(set(emotions)),
            'last_update': datetime.now().isoformat()
        })
    
    async def _generate_conversation_insights(self, conversation_id: str, participant_id: str) -> Dict[str, Any]:
        """Generate insights about conversation emotions"""
        context = self.emotional_contexts.get(conversation_id)
        if not context or not context.emotion_timeline:
            return {'insights': 'Insufficient data for insights'}
        
        insights = {
            'emotional_journey': await self._analyze_emotional_journey(context),
            'engagement_level': await self._assess_engagement_level(context),
            'empathy_opportunities': await self._identify_empathy_opportunities(context),
            'conversation_health': await self._assess_conversation_health(context)
        }
        
        return insights
    
    async def _analyze_emotional_journey(self, context: EmotionalContext) -> Dict[str, Any]:
        """Analyze the emotional journey through the conversation"""
        if len(context.emotion_timeline) < 2:
            return {'status': 'insufficient_data'}
        
        emotions = [analysis.primary_emotion for _, analysis in context.emotion_timeline]
        intensities = [analysis.intensity for _, analysis in context.emotion_timeline]
        
        return {
            'starting_emotion': emotions[0],
            'current_emotion': emotions[-1],
            'emotion_progression': emotions,
            'intensity_trend': 'increasing' if intensities[-1] > intensities[0] else 'decreasing',
            'emotional_stability': 'stable' if len(set(emotions[-3:])) == 1 else 'variable'
        }
    
    async def _assess_engagement_level(self, context: EmotionalContext) -> str:
        """Assess participant engagement level"""
        if not context.emotion_timeline:
            return 'unknown'
        
        recent_emotions = [analysis.primary_emotion for _, analysis in context.emotion_timeline[-3:]]
        avg_intensity = sum(analysis.intensity for _, analysis in context.emotion_timeline[-3:]) / len(context.emotion_timeline[-3:])
        
        positive_emotions = ['joy', 'anticipation', 'trust']
        engaged_emotions = [emotion for emotion in recent_emotions if emotion in positive_emotions]
        
        if avg_intensity > 0.6 and engaged_emotions:
            return 'highly_engaged'
        elif avg_intensity > 0.4:
            return 'moderately_engaged'
        elif 'sadness' in recent_emotions or 'anger' in recent_emotions:
            return 'disengaged'
        else:
            return 'neutral'
    
    async def _identify_empathy_opportunities(self, context: EmotionalContext) -> List[str]:
        """Identify opportunities for empathetic responses"""
        opportunities = []
        
        if not context.emotion_timeline:
            return opportunities
        
        recent_analysis = context.emotion_timeline[-1][1]
        
        if recent_analysis.intensity > 0.6:
            opportunities.append('high_intensity_response_needed')
        
        if recent_analysis.emotional_state in ['frustrated', 'worried', 'confused']:
            opportunities.append('supportive_response_recommended')
        
        if recent_analysis.authenticity_score < 0.5:
            opportunities.append('probe_for_real_concerns')
        
        return opportunities
    
    async def _assess_conversation_health(self, context: EmotionalContext) -> Dict[str, Any]:
        """Assess overall conversation health"""
        if not context.emotion_timeline:
            return {'status': 'insufficient_data'}
        
        patterns = context.emotional_patterns
        
        # Check for positive trend
        recent_sentiments = [analysis.sentiment_score for _, analysis in context.emotion_timeline[-3:]]
        sentiment_trend = 'improving' if len(recent_sentiments) > 1 and recent_sentiments[-1] > recent_sentiments[0] else 'declining'
        
        # Overall health score
        health_score = 0.0
        if patterns.get('average_intensity', 0) > 0.3:
            health_score += 0.3
        if patterns.get('average_stability', 0) > 0.5:
            health_score += 0.3
        if sentiment_trend == 'improving':
            health_score += 0.4
        
        return {
            'health_score': health_score,
            'sentiment_trend': sentiment_trend,
            'conversation_status': 'healthy' if health_score > 0.6 else 'needs_attention',
            'recommendations': await self._get_health_recommendations(health_score, patterns)
        }
    
    async def _get_health_recommendations(self, health_score: float, patterns: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving conversation health"""
        recommendations = []
        
        if health_score < 0.4:
            recommendations.append('Consider shifting conversation approach')
            recommendations.append('Increase empathy and active listening')
        
        if patterns.get('average_intensity', 0) > 0.8:
            recommendations.append('Conversation is highly emotional - proceed with care')
        
        if patterns.get('emotion_changes', 0) > 5:
            recommendations.append('Participant emotions are unstable - provide reassurance')
        
        return recommendations


# Global emotion conversation manager instance
emotion_manager = EmotionConversationManager()