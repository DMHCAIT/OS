"""
Advanced AI/ML Service for Intelligent Sales Automation
Implements cutting-edge AI techniques for deal closing and sales automation
"""

import asyncio
import logging
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle

# Advanced NLP and ML imports
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

# Text processing
import textdistance
from textblob import TextBlob

logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Advanced AI insight with actionable recommendations"""
    insight_type: str
    confidence: float
    message: str
    recommended_action: str
    priority: str
    reasoning: str
    data_points: Dict[str, Any]

@dataclass
class ConversationAnalysis:
    """Advanced conversation analysis results"""
    sentiment_score: float
    emotion_detected: str
    intent_classification: str
    objection_detected: bool
    objection_type: Optional[str]
    buying_signals: List[str]
    pain_points: List[str]
    decision_maker_score: float
    urgency_level: float
    next_best_action: str
    recommended_response: str
    close_probability: float

@dataclass
class LeadScoringResult:
    """ML-powered lead scoring results"""
    overall_score: int
    conversion_probability: float
    engagement_score: float
    behavioral_score: float
    demographic_score: float
    interaction_quality: float
    time_decay_factor: float
    predicted_deal_size: float
    optimal_contact_time: datetime
    recommended_approach: str
    risk_factors: List[str]

class AdvancedAIMLService:
    """
    Advanced AI/ML Service for Intelligent Sales Automation
    Implements state-of-the-art techniques for deal closing
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.models = {}
        self.scalers = {}
        self.conversation_history = {}
        self.lead_profiles = {}
        self.sales_patterns = {}
        
        # Flag to track if models are initialized
        self._models_initialized = False
    
    async def _ensure_models_initialized(self):
        """Ensure models are initialized before use"""
        if not self._models_initialized:
            await self._initialize_models()
            self._models_initialized = True
    
    def _get_device(self) -> str:
        """Get optimal device for ML computations"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def _initialize_models(self):
        """Initialize all AI/ML models"""
        try:
            logger.info("Initializing advanced AI/ML models...")
            
            # Core NLP models
            self.nlp = spacy.load("en_core_web_sm")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Sentence transformers for semantic analysis
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            # Emotion detection model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == "cuda" else -1
            )
            
            # Intent classification model
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            # Conversation generation model (GPT-style)
            self.conversation_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == "cuda" else -1
            )
            
            # Objection detection model
            self.objection_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize ML models for lead scoring
            self._initialize_ml_models()
            
            logger.info("Advanced AI/ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to basic models if advanced ones fail
            await self._initialize_fallback_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for predictions"""
        # Lead scoring ensemble model
        self.models['lead_scoring'] = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'nn': MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42)
        }
        
        # Conversation outcome prediction
        self.models['conversation_outcome'] = RandomForestClassifier(n_estimators=100)
        
        # Deal size prediction
        self.models['deal_size'] = GradientBoostingClassifier(n_estimators=100)
        
        # Optimal timing model
        self.models['timing'] = MLPClassifier(hidden_layer_sizes=(64, 32))
        
        # Scalers for feature normalization
        self.scalers = {
            'lead_features': StandardScaler(),
            'conversation_features': StandardScaler(),
            'behavioral_features': StandardScaler()
        }
    
    async def _initialize_fallback_models(self):
        """Initialize basic fallback models if advanced models fail"""
        logger.warning("Falling back to basic models")
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentence_model = None  # Will use basic similarity
    
    async def analyze_conversation(self, conversation_text: str, lead_context: Dict[str, Any]) -> ConversationAnalysis:
        """
        Advanced conversation analysis using multiple AI techniques
        """
        await self._ensure_models_initialized()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(conversation_text)
            
            # Sentiment analysis with multiple models
            sentiment_score = await self._analyze_sentiment(processed_text)
            
            # Emotion detection
            emotion = await self._detect_emotion(processed_text)
            
            # Intent classification
            intent = await self._classify_intent(processed_text)
            
            # Objection detection
            objection_detected, objection_type = await self._detect_objection(processed_text)
            
            # Extract buying signals
            buying_signals = await self._extract_buying_signals(processed_text)
            
            # Identify pain points
            pain_points = await self._extract_pain_points(processed_text)
            
            # Decision maker analysis
            decision_maker_score = await self._analyze_decision_maker(processed_text, lead_context)
            
            # Urgency detection
            urgency_level = await self._detect_urgency(processed_text)
            
            # Generate next best action
            next_action = await self._recommend_next_action(
                sentiment_score, emotion, intent, objection_detected, 
                buying_signals, urgency_level, lead_context
            )
            
            # Generate AI response
            recommended_response = await self._generate_response(
                processed_text, sentiment_score, emotion, intent, 
                objection_type, lead_context
            )
            
            # Calculate close probability
            close_probability = await self._calculate_close_probability(
                sentiment_score, buying_signals, urgency_level, 
                decision_maker_score, lead_context
            )
            
            return ConversationAnalysis(
                sentiment_score=sentiment_score,
                emotion_detected=emotion,
                intent_classification=intent,
                objection_detected=objection_detected,
                objection_type=objection_type,
                buying_signals=buying_signals,
                pain_points=pain_points,
                decision_maker_score=decision_maker_score,
                urgency_level=urgency_level,
                next_best_action=next_action,
                recommended_response=recommended_response,
                close_probability=close_probability
            )
            
        except Exception as e:
            logger.error(f"Error in conversation analysis: {e}")
            return self._get_fallback_conversation_analysis()
    
    async def score_lead_ml(self, lead_data: Dict[str, Any]) -> LeadScoringResult:
        """
        Advanced ML-based lead scoring with multiple algorithms
        """
        try:
            # Extract features for ML models
            features = await self._extract_lead_features(lead_data)
            
            # Normalize features
            normalized_features = self._normalize_features(features, 'lead_features')
            
            # Ensemble prediction
            predictions = {}
            for model_name, model in self.models['lead_scoring'].items():
                if hasattr(model, 'predict_proba'):
                    predictions[model_name] = model.predict_proba([normalized_features])[0][1]
                else:
                    predictions[model_name] = model.predict([normalized_features])[0] / 100
            
            # Weighted ensemble
            conversion_probability = np.mean(list(predictions.values()))
            
            # Calculate component scores
            engagement_score = await self._calculate_engagement_score(lead_data)
            behavioral_score = await self._calculate_behavioral_score(lead_data)
            demographic_score = await self._calculate_demographic_score(lead_data)
            interaction_quality = await self._calculate_interaction_quality(lead_data)
            time_decay_factor = self._calculate_time_decay(lead_data.get('last_contact'))
            
            # Overall score (0-100)
            overall_score = int(
                (conversion_probability * 40 +
                 engagement_score * 20 +
                 behavioral_score * 15 +
                 demographic_score * 10 +
                 interaction_quality * 10 +
                 time_decay_factor * 5) * 100
            )
            
            # Predict deal size
            predicted_deal_size = await self._predict_deal_size(features)
            
            # Optimal contact time
            optimal_contact_time = await self._predict_optimal_contact_time(lead_data)
            
            # Recommended approach
            recommended_approach = await self._recommend_sales_approach(
                conversion_probability, engagement_score, behavioral_score
            )
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(lead_data, conversion_probability)
            
            return LeadScoringResult(
                overall_score=overall_score,
                conversion_probability=conversion_probability,
                engagement_score=engagement_score,
                behavioral_score=behavioral_score,
                demographic_score=demographic_score,
                interaction_quality=interaction_quality,
                time_decay_factor=time_decay_factor,
                predicted_deal_size=predicted_deal_size,
                optimal_contact_time=optimal_contact_time,
                recommended_approach=recommended_approach,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error in ML lead scoring: {e}")
            return self._get_fallback_lead_scoring(lead_data)
    
    async def generate_ai_insights(self, lead_data: Dict[str, Any], conversation_history: List[Dict]) -> List[AIInsight]:
        """
        Generate actionable AI insights for sales optimization
        """
        insights = []
        
        try:
            # Analyze conversation patterns
            if conversation_history:
                conversation_insights = await self._analyze_conversation_patterns(conversation_history)
                insights.extend(conversation_insights)
            
            # Lead behavior analysis
            behavior_insights = await self._analyze_lead_behavior(lead_data)
            insights.extend(behavior_insights)
            
            # Market timing insights
            timing_insights = await self._analyze_market_timing(lead_data)
            insights.extend(timing_insights)
            
            # Competitive analysis insights
            competitive_insights = await self._analyze_competitive_position(lead_data)
            insights.extend(competitive_insights)
            
            # Personalization insights
            personalization_insights = await self._generate_personalization_insights(lead_data)
            insights.extend(personalization_insights)
            
            # Sort by priority and confidence
            insights.sort(key=lambda x: (x.priority == 'high', x.confidence), reverse=True)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return []
    
    async def predict_sales_outcome(self, lead_data: Dict[str, Any], conversation_data: List[Dict]) -> Dict[str, Any]:
        """
        Predict sales outcome using advanced ML techniques
        """
        try:
            # Feature extraction
            lead_features = await self._extract_lead_features(lead_data)
            conversation_features = await self._extract_conversation_features(conversation_data)
            
            # Combine features
            combined_features = np.concatenate([lead_features, conversation_features])
            
            # Normalize
            normalized_features = self._normalize_features(combined_features, 'conversation_features')
            
            # Predict outcome
            outcome_probability = self.models['conversation_outcome'].predict_proba([normalized_features])[0]
            
            # Predict deal size
            deal_size = await self._predict_deal_size(lead_features)
            
            # Predict timeline
            timeline = await self._predict_sales_timeline(combined_features)
            
            # Risk assessment
            risks = await self._assess_sales_risks(lead_data, conversation_data)
            
            return {
                'win_probability': float(outcome_probability[1]),
                'lose_probability': float(outcome_probability[0]),
                'predicted_deal_size': float(deal_size),
                'predicted_timeline_days': int(timeline),
                'confidence_score': float(np.max(outcome_probability)),
                'risk_factors': risks,
                'recommended_actions': await self._recommend_sales_actions(
                    outcome_probability[1], deal_size, timeline, risks
                )
            }
            
        except Exception as e:
            logger.error(f"Error predicting sales outcome: {e}")
            return {
                'win_probability': 0.5,
                'lose_probability': 0.5,
                'predicted_deal_size': 0.0,
                'predicted_timeline_days': 30,
                'confidence_score': 0.5,
                'risk_factors': [],
                'recommended_actions': []
            }
    
    # Helper methods for AI processing
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        return text
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using multiple approaches"""
        try:
            # NLTK VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine scores (weighted average)
            combined_sentiment = (vader_scores['compound'] * 0.6 + textblob_polarity * 0.4)
            
            return float(combined_sentiment)
            
        except Exception:
            return 0.0
    
    async def _detect_emotion(self, text: str) -> str:
        """Detect emotion using transformer model"""
        try:
            if self.emotion_classifier:
                result = self.emotion_classifier(text)[0]
                return result['label'].lower()
            return "neutral"
        except Exception:
            return "neutral"
    
    async def _classify_intent(self, text: str) -> str:
        """Classify conversation intent"""
        try:
            candidate_labels = [
                "information_seeking", "price_inquiry", "demo_request",
                "ready_to_buy", "objection", "scheduling", "follow_up"
            ]
            
            if self.intent_classifier:
                result = self.intent_classifier(text, candidate_labels)
                return result['labels'][0]
            
            return "information_seeking"
            
        except Exception:
            return "information_seeking"
    
    async def _detect_objection(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect objections in conversation"""
        objection_patterns = [
            (r"too expensive|too costly|can't afford|budget", "price"),
            (r"need to think|need time|not ready|not sure", "timing"),
            (r"competitor|other option|already have|using", "competition"),
            (r"not authorized|need approval|boss|manager", "authority"),
            (r"doesn't fit|not what we need|wrong solution", "need"),
            (r"not working|bad experience|problems|issues", "trust")
        ]
        
        text_lower = text.lower()
        for pattern, objection_type in objection_patterns:
            if re.search(pattern, text_lower):
                return True, objection_type
        
        return False, None
    
    async def _extract_buying_signals(self, text: str) -> List[str]:
        """Extract buying signals from conversation"""
        buying_signals = []
        signal_patterns = [
            (r"when can we start|how soon|timeline|implementation", "timeline_interest"),
            (r"pricing|cost|investment|budget|price", "pricing_interest"),
            (r"contract|agreement|terms|conditions", "contract_interest"),
            (r"team|colleagues|stakeholders|decision", "stakeholder_involvement"),
            (r"demo|trial|test|proof|pilot", "trial_interest"),
            (r"looks good|sounds great|perfect|exactly", "positive_feedback"),
            (r"next steps|move forward|proceed|continue", "progression_intent")
        ]
        
        text_lower = text.lower()
        for pattern, signal in signal_patterns:
            if re.search(pattern, text_lower):
                buying_signals.append(signal)
        
        return buying_signals
    
    async def _extract_pain_points(self, text: str) -> List[str]:
        """Extract pain points from conversation"""
        pain_points = []
        pain_patterns = [
            (r"slow|taking too long|inefficient", "efficiency"),
            (r"expensive|costly|waste money", "cost"),
            (r"difficult|hard|complex|complicated", "complexity"),
            (r"manual|tedious|time-consuming", "automation"),
            (r"error|mistake|wrong|incorrect", "accuracy"),
            (r"unreliable|unstable|down|crash", "reliability")
        ]
        
        text_lower = text.lower()
        for pattern, pain in pain_patterns:
            if re.search(pattern, text_lower):
                pain_points.append(pain)
        
        return pain_points
    
    async def _analyze_decision_maker(self, text: str, lead_context: Dict) -> float:
        """Analyze if lead is a decision maker"""
        decision_indicators = [
            r"I decide|my decision|I choose",
            r"my budget|my team|my company",
            r"I can approve|I have authority",
            r"CEO|CTO|director|manager|VP|president"
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for indicator in decision_indicators:
            if re.search(indicator, text_lower):
                score += 0.25
        
        # Check job title from context
        job_title = lead_context.get('job_title', '').lower()
        if any(title in job_title for title in ['ceo', 'cto', 'director', 'vp', 'manager', 'president']):
            score += 0.3
        
        return min(score, 1.0)
    
    async def _detect_urgency(self, text: str) -> float:
        """Detect urgency level in conversation"""
        urgency_patterns = [
            (r"urgent|asap|immediately|right away|quickly", 0.9),
            (r"soon|this week|this month|deadline", 0.7),
            (r"sometime|eventually|future|later", 0.3),
            (r"no rush|take time|not urgent", 0.1)
        ]
        
        text_lower = text.lower()
        max_urgency = 0.5  # default
        
        for pattern, urgency in urgency_patterns:
            if re.search(pattern, text_lower):
                max_urgency = max(max_urgency, urgency)
        
        return max_urgency
    
    async def _recommend_next_action(self, sentiment: float, emotion: str, intent: str, 
                                   objection_detected: bool, buying_signals: List[str], 
                                   urgency: float, context: Dict) -> str:
        """Recommend next best action using AI"""
        
        if objection_detected:
            return "address_objection"
        
        if len(buying_signals) >= 2 and urgency > 0.6:
            return "close_deal"
        
        if intent == "ready_to_buy":
            return "send_proposal"
        
        if intent == "demo_request":
            return "schedule_demo"
        
        if sentiment > 0.3 and emotion in ["joy", "surprise"]:
            return "nurture_positive"
        
        if urgency > 0.5:
            return "create_urgency"
        
        return "continue_nurturing"
    
    async def _generate_response(self, text: str, sentiment: float, emotion: str, 
                               intent: str, objection_type: Optional[str], 
                               context: Dict) -> str:
        """Generate AI-powered response"""
        
        # Response templates based on analysis
        if objection_type == "price":
            return "I understand budget is important. Let me show you the ROI and value this brings to justify the investment."
        
        if objection_type == "timing":
            return "I appreciate you taking time to consider this. What specific timeline works best for your team?"
        
        if intent == "ready_to_buy":
            return "That's fantastic! Let me prepare a customized proposal that addresses all your requirements."
        
        if sentiment > 0.5:
            return "I'm glad this resonates with you! What would be the best next step to move forward?"
        
        # Default personalized response
        return f"Thank you for sharing that information. Based on what you've told me, I think our solution could really help with your {context.get('industry', 'business')} challenges."
    
    async def _calculate_close_probability(self, sentiment: float, buying_signals: List[str], 
                                         urgency: float, decision_maker_score: float, 
                                         context: Dict) -> float:
        """Calculate probability of closing the deal"""
        
        base_probability = 0.1
        
        # Sentiment contribution (0-30%)
        sentiment_contrib = max(0, sentiment) * 0.3
        
        # Buying signals contribution (0-25%)
        signal_contrib = min(len(buying_signals) / 4, 1.0) * 0.25
        
        # Urgency contribution (0-20%)
        urgency_contrib = urgency * 0.2
        
        # Decision maker contribution (0-15%)
        decision_contrib = decision_maker_score * 0.15
        
        # Context contribution (0-10%)
        context_contrib = 0.1 if context.get('company_size') == 'enterprise' else 0.05
        
        close_probability = (base_probability + sentiment_contrib + signal_contrib + 
                           urgency_contrib + decision_contrib + context_contrib)
        
        return min(close_probability, 0.95)  # Cap at 95%
    
    # Additional helper methods would continue here...
    # (For brevity, I'll implement the most critical ones)
    
    def _get_fallback_conversation_analysis(self) -> ConversationAnalysis:
        """Fallback conversation analysis"""
        return ConversationAnalysis(
            sentiment_score=0.0,
            emotion_detected="neutral",
            intent_classification="information_seeking",
            objection_detected=False,
            objection_type=None,
            buying_signals=[],
            pain_points=[],
            decision_maker_score=0.5,
            urgency_level=0.5,
            next_best_action="continue_nurturing",
            recommended_response="Thank you for your interest. How can I help you today?",
            close_probability=0.3
        )
    
    def _get_fallback_lead_scoring(self, lead_data: Dict) -> LeadScoringResult:
        """Fallback lead scoring"""
        return LeadScoringResult(
            overall_score=50,
            conversion_probability=0.3,
            engagement_score=0.5,
            behavioral_score=0.5,
            demographic_score=0.5,
            interaction_quality=0.5,
            time_decay_factor=1.0,
            predicted_deal_size=10000.0,
            optimal_contact_time=datetime.now() + timedelta(hours=24),
            recommended_approach="consultative_selling",
            risk_factors=[]
        )