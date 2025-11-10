"""
AI Conversation Engine
Advanced conversational AI for sales automation and customer engagement
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, AutoModelForSequenceClassification
)
import torch
import numpy as np
from textblob import TextBlob
import spacy
from collections import defaultdict, deque

# Load spacy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

logger = logging.getLogger(__name__)

class ConversationStage(Enum):
    """Different stages of sales conversation"""
    INITIAL_CONTACT = "initial_contact"
    DISCOVERY = "discovery" 
    NEEDS_ANALYSIS = "needs_analysis"
    SOLUTION_PRESENTATION = "solution_presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"
    POST_SALE = "post_sale"

class MessageType(Enum):
    """Types of conversation messages"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    SYSTEM = "system"
    AI_GENERATED = "ai_generated"

class SentimentType(Enum):
    """Sentiment classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class ConversationMessage:
    """Individual conversation message"""
    id: str
    timestamp: datetime
    message_type: MessageType
    content: str
    sender: str
    recipient: str
    sentiment: Optional[SentimentType] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = None
    confidence_score: float = 0.0
    response_time_seconds: Optional[int] = None

@dataclass
class ConversationContext:
    """Conversation context and state"""
    conversation_id: str
    lead_id: str
    current_stage: ConversationStage
    messages: List[ConversationMessage]
    key_insights: Dict[str, Any]
    pain_points: List[str]
    identified_needs: List[str]
    budget_range: Optional[str] = None
    decision_timeline: Optional[str] = None
    decision_makers: List[str] = None
    competitors_mentioned: List[str] = None
    objections_raised: List[str] = None
    next_actions: List[str] = None
    conversation_score: float = 0.0
    engagement_level: str = "low"

@dataclass
class AIResponse:
    """AI-generated response"""
    content: str
    confidence_score: float
    suggested_stage: ConversationStage
    follow_up_actions: List[str]
    detected_opportunities: List[str]
    risk_flags: List[str]
    personalization_elements: Dict[str, str]

class AIConversationEngine:
    """
    Advanced AI Conversation Engine for Sales Automation
    Integrates multiple AI models for intelligent conversation handling
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize models
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Conversation memory
        self.conversation_memory = defaultdict(deque)
        self.conversation_contexts = {}
        
        # Pattern recognition
        self.objection_patterns = {}
        self.buying_signals = {}
        self.intent_classifiers = {}
        
        # Response templates
        self.response_templates = {}
        
        # Initialize components
        self._initialize_models()
        self._initialize_patterns()
        self._initialize_templates()
    
    def _initialize_models(self):
        """Initialize AI models for conversation processing"""
        try:
            # Sentiment analysis pipeline
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Intent classification pipeline
            self.pipelines['intent'] = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium"
            )
            
            # Emotion detection pipeline
            self.pipelines['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            
            # Question answering pipeline
            self.pipelines['qa'] = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
            
            # Text generation model (fallback to GPT-2 if OpenAI not available)
            if not self.openai_api_key:
                self.tokenizers['generator'] = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.models['generator'] = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _initialize_patterns(self):
        """Initialize conversation patterns for recognition"""
        
        # Objection patterns
        self.objection_patterns = {
            'price': [
                r'too expensive', r'cost.*much', r'budget.*tight', r'cheaper.*option',
                r'price.*high', r'afford', r'money.*issue', r'financial.*constraint'
            ],
            'timing': [
                r'not.*right.*time', r'busy.*now', r'later.*year', r'timing.*bad',
                r'wait.*until', r'delay', r'postpone', r'schedule.*full'
            ],
            'authority': [
                r'need.*approval', r'boss.*decide', r'team.*decision', r'not.*authority',
                r'check.*with', r'discuss.*internally', r'committee.*review'
            ],
            'need': [
                r'don.*need', r'working.*fine', r'current.*solution', r'satisfied.*with',
                r'no.*problem', r'unnecessary', r'waste.*time', r'not.*priority'
            ],
            'trust': [
                r'never.*heard', r'not.*sure.*about', r'proof.*works', r'references',
                r'testimonials', r'track.*record', r'skeptical', r'doubt.*works'
            ]
        }
        
        # Buying signals
        self.buying_signals = {
            'interest': [
                r'tell.*more', r'interested.*in', r'sounds.*good', r'like.*idea',
                r'want.*know', r'learn.*more', r'explore.*options', r'consider'
            ],
            'urgency': [
                r'need.*soon', r'urgent', r'asap', r'quickly', r'immediate',
                r'right.*away', r'time.*sensitive', r'deadline'
            ],
            'budget': [
                r'what.*cost', r'price.*range', r'budget.*for', r'afford.*to',
                r'investment', r'spend.*on', r'roi', r'return.*investment'
            ],
            'decision': [
                r'ready.*to', r'want.*to.*buy', r'move.*forward', r'next.*step',
                r'sign.*up', r'get.*started', r'proceed.*with', r'commitment'
            ]
        }
        
        # Intent patterns
        self.intent_classifiers = {
            'information_request': [
                r'what.*is', r'how.*does', r'can.*you.*tell', r'explain',
                r'describe', r'details.*about', r'more.*info', r'clarify'
            ],
            'pricing_inquiry': [
                r'how.*much', r'what.*cost', r'price', r'pricing', r'quote',
                r'estimate', r'budget', r'fee', r'charge'
            ],
            'demo_request': [
                r'demo', r'demonstration', r'show.*me', r'see.*it.*work',
                r'trial', r'test.*drive', r'preview', r'sample'
            ],
            'meeting_request': [
                r'meet', r'call', r'schedule', r'appointment', r'discussion',
                r'talk.*about', r'get.*together', r'conference'
            ]
        }
    
    def _initialize_templates(self):
        """Initialize response templates for different scenarios"""
        
        self.response_templates = {
            'initial_contact': {
                'cold_outreach': [
                    "Hi {name}, I noticed {company} has been growing rapidly in {industry}. "
                    "I'd love to show you how companies like yours are {value_prop}. "
                    "Would you be open to a brief conversation?",
                    
                    "Hello {name}, congratulations on {recent_achievement}! "
                    "I work with {industry} companies to {solution_brief}. "
                    "Could we schedule a quick 15-minute call?"
                ],
                'warm_introduction': [
                    "Hi {name}, {referrer_name} mentioned you might be interested in {solution_area}. "
                    "I'd love to learn more about {company}'s goals and share how we've helped "
                    "similar companies achieve {outcome}.",
                    
                    "Hello {name}, following up on our connection at {event}. "
                    "As discussed, I'd like to show you how {value_prop}. "
                    "When would be a good time for a brief call?"
                ]
            },
            
            'discovery': {
                'pain_point_questions': [
                    "What's your biggest challenge with {current_process}?",
                    "How is {current_situation} impacting your {business_metric}?",
                    "If you could wave a magic wand and fix one thing about {area}, what would it be?",
                    "Tell me about a typical {process} - where do you see the most friction?"
                ],
                'situation_questions': [
                    "Help me understand your current {solution_area} setup.",
                    "What tools are you currently using for {function}?",
                    "How does your team typically handle {process}?",
                    "What's working well with your current approach?"
                ]
            },
            
            'objection_handling': {
                'price': [
                    "I understand budget is a consideration. Let's look at the ROI - "
                    "our clients typically see {roi_metric} within {timeframe}. "
                    "What if we could show a path to {value_outcome}?",
                    
                    "Price is definitely important. Can you help me understand "
                    "what budget range you're working with? That way I can show "
                    "options that make sense for {company}."
                ],
                'timing': [
                    "I appreciate you being upfront about timing. What would need to change "
                    "for this to become a priority? Is it a matter of {factor1} or {factor2}?",
                    
                    "Timing makes sense. What if I could show you a solution that "
                    "actually saves time in implementation? Would {timeframe} work better?"
                ]
            },
            
            'closing': {
                'trial_close': [
                    "Based on what you've shared, it sounds like {solution} could help "
                    "with {pain_point}. What questions do you have about moving forward?",
                    
                    "If we can deliver {desired_outcome} within {timeline}, "
                    "what would the next step look like on your end?"
                ],
                'direct_close': [
                    "It sounds like we're aligned on the value. I'd love to get "
                    "{company} started. What does your approval process look like?",
                    
                    "Perfect! Let's get the paperwork started. I'll send over "
                    "the proposal today. When could you review it?"
                ]
            }
        }
    
    async def analyze_message(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Comprehensive analysis of incoming message
        """
        analysis = {
            'sentiment': await self._analyze_sentiment(message),
            'emotion': await self._analyze_emotion(message),
            'intent': await self._classify_intent(message),
            'entities': await self._extract_entities(message),
            'objections': await self._detect_objections(message),
            'buying_signals': await self._detect_buying_signals(message),
            'engagement_level': await self._assess_engagement(message, context),
            'conversation_stage': await self._determine_stage(message, context),
            'key_insights': await self._extract_insights(message),
            'next_actions': await self._suggest_actions(message, context)
        }
        
        return analysis
    
    async def _analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """Analyze sentiment of message"""
        try:
            if 'sentiment' in self.pipelines:
                result = self.pipelines['sentiment'](message)[0]
                
                # Map labels to our sentiment types
                label_mapping = {
                    'LABEL_0': SentimentType.NEGATIVE,
                    'LABEL_1': SentimentType.NEUTRAL, 
                    'LABEL_2': SentimentType.POSITIVE,
                    'NEGATIVE': SentimentType.NEGATIVE,
                    'NEUTRAL': SentimentType.NEUTRAL,
                    'POSITIVE': SentimentType.POSITIVE
                }
                
                sentiment_type = label_mapping.get(result['label'], SentimentType.NEUTRAL)
                
                # Use TextBlob as backup for polarity score
                blob = TextBlob(message)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.5:
                    sentiment_type = SentimentType.VERY_POSITIVE
                elif polarity < -0.5:
                    sentiment_type = SentimentType.VERY_NEGATIVE
                
                return {
                    'type': sentiment_type,
                    'confidence': result['score'],
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            else:
                # Fallback to TextBlob
                blob = TextBlob(message)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.3:
                    sentiment_type = SentimentType.POSITIVE
                elif polarity < -0.3:
                    sentiment_type = SentimentType.NEGATIVE
                else:
                    sentiment_type = SentimentType.NEUTRAL
                
                return {
                    'type': sentiment_type,
                    'confidence': 0.7,
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'type': SentimentType.NEUTRAL,
                'confidence': 0.5,
                'polarity': 0.0,
                'subjectivity': 0.5
            }
    
    async def _analyze_emotion(self, message: str) -> Dict[str, Any]:
        """Analyze emotional content of message"""
        try:
            if 'emotion' in self.pipelines:
                result = self.pipelines['emotion'](message)[0]
                return {
                    'primary_emotion': result['label'],
                    'confidence': result['score'],
                    'all_emotions': result if isinstance(result, list) else [result]
                }
            else:
                return {
                    'primary_emotion': 'neutral',
                    'confidence': 0.5,
                    'all_emotions': []
                }
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': []
            }
    
    async def _classify_intent(self, message: str) -> Dict[str, Any]:
        """Classify the intent of the message"""
        intents = {}
        message_lower = message.lower()
        
        # Pattern-based intent detection
        for intent_type, patterns in self.intent_classifiers.items():
            confidence = 0.0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    confidence = max(confidence, 0.8)
            
            if confidence > 0:
                intents[intent_type] = confidence
        
        # Determine primary intent
        if intents:
            primary_intent = max(intents, key=intents.get)
            primary_confidence = intents[primary_intent]
        else:
            primary_intent = 'general_inquiry'
            primary_confidence = 0.3
        
        return {
            'primary_intent': primary_intent,
            'confidence': primary_confidence,
            'all_intents': intents
        }
    
    async def _extract_entities(self, message: str) -> Dict[str, List]:
        """Extract named entities from message"""
        entities = {}
        
        if nlp:
            doc = nlp(message)
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Extract business-specific entities
        business_entities = self._extract_business_entities(message)
        entities.update(business_entities)
        
        return entities
    
    def _extract_business_entities(self, message: str) -> Dict[str, List]:
        """Extract business-specific entities"""
        entities = {}
        
        # Budget/money patterns
        money_patterns = [
            r'\$[\d,]+(?:\.\d{2})?[kKmMbB]?',
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|k|K|million|M|billion|B)',
            r'budget.*?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                if 'MONEY' not in entities:
                    entities['MONEY'] = []
                entities['MONEY'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Timeline patterns
        timeline_patterns = [
            r'\d+\s*(?:days?|weeks?|months?|years?)',
            r'(?:next|this)\s+(?:week|month|quarter|year)',
            r'by\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)',
            r'Q[1-4]'
        ]
        
        for pattern in timeline_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                if 'TIMELINE' not in entities:
                    entities['TIMELINE'] = []
                entities['TIMELINE'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    async def _detect_objections(self, message: str) -> List[Dict[str, Any]]:
        """Detect objections in the message"""
        objections = []
        message_lower = message.lower()
        
        for objection_type, patterns in self.objection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    objections.append({
                        'type': objection_type,
                        'confidence': 0.8,
                        'pattern_matched': pattern
                    })
                    break  # Only count each objection type once
        
        return objections
    
    async def _detect_buying_signals(self, message: str) -> List[Dict[str, Any]]:
        """Detect buying signals in the message"""
        buying_signals = []
        message_lower = message.lower()
        
        for signal_type, patterns in self.buying_signals.items():
            confidence = 0.0
            matched_pattern = None
            
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    confidence = 0.8
                    matched_pattern = pattern
                    break
            
            if confidence > 0:
                buying_signals.append({
                    'type': signal_type,
                    'confidence': confidence,
                    'pattern_matched': matched_pattern
                })
        
        return buying_signals
    
    async def _assess_engagement(self, message: str, context: ConversationContext) -> str:
        """Assess engagement level of the message"""
        engagement_score = 0
        
        # Message length factor
        word_count = len(message.split())
        if word_count > 50:
            engagement_score += 3
        elif word_count > 20:
            engagement_score += 2
        elif word_count > 10:
            engagement_score += 1
        
        # Question asking (shows interest)
        question_count = message.count('?')
        engagement_score += min(question_count * 2, 4)
        
        # Specific details mentioned
        if re.search(r'\d+', message):  # Numbers mentioned
            engagement_score += 1
        
        if len(context.messages) > 1:
            # Response time factor
            last_msg = context.messages[-2]
            if hasattr(last_msg, 'timestamp'):
                response_time = (datetime.now() - last_msg.timestamp).total_seconds()
                if response_time < 3600:  # Less than 1 hour
                    engagement_score += 2
                elif response_time < 86400:  # Less than 1 day
                    engagement_score += 1
        
        # Convert to engagement level
        if engagement_score >= 6:
            return 'very_high'
        elif engagement_score >= 4:
            return 'high'
        elif engagement_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    async def _determine_stage(self, message: str, context: ConversationContext) -> ConversationStage:
        """Determine the current conversation stage"""
        message_lower = message.lower()
        
        # Stage indicators
        stage_indicators = {
            ConversationStage.INITIAL_CONTACT: [
                'hello', 'hi', 'introduction', 'nice to meet', 'first time'
            ],
            ConversationStage.DISCOVERY: [
                'tell me about', 'what do you', 'how do you', 'current process',
                'challenges', 'problems', 'pain points'
            ],
            ConversationStage.NEEDS_ANALYSIS: [
                'need', 'require', 'looking for', 'want', 'goals', 'objectives'
            ],
            ConversationStage.SOLUTION_PRESENTATION: [
                'demo', 'show me', 'how does it work', 'features', 'capabilities'
            ],
            ConversationStage.OBJECTION_HANDLING: [
                'but', 'however', 'concern', 'worry', 'not sure', 'doubt'
            ],
            ConversationStage.CLOSING: [
                'next steps', 'move forward', 'ready to', 'sign up', 'get started',
                'proposal', 'contract', 'agreement'
            ]
        }
        
        # Check for stage indicators
        stage_scores = {}
        for stage, indicators in stage_indicators.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            if score > 0:
                stage_scores[stage] = score
        
        # Determine stage based on context and indicators
        if stage_scores:
            suggested_stage = max(stage_scores, key=stage_scores.get)
        else:
            # Default progression logic
            if len(context.messages) <= 2:
                suggested_stage = ConversationStage.INITIAL_CONTACT
            elif context.current_stage == ConversationStage.INITIAL_CONTACT:
                suggested_stage = ConversationStage.DISCOVERY
            else:
                suggested_stage = context.current_stage
        
        return suggested_stage
    
    async def _extract_insights(self, message: str) -> Dict[str, Any]:
        """Extract key insights from the message"""
        insights = {}
        
        # Budget insights
        budget_mentions = re.findall(r'\$[\d,]+(?:\.\d{2})?[kKmMbB]?', message)
        if budget_mentions:
            insights['budget_mentioned'] = budget_mentions
        
        # Timeline insights
        timeline_mentions = re.findall(
            r'\d+\s*(?:days?|weeks?|months?|years?)|(?:next|this)\s+(?:week|month|quarter|year)',
            message, re.IGNORECASE
        )
        if timeline_mentions:
            insights['timeline_mentioned'] = timeline_mentions
        
        # Competitor mentions
        competitor_keywords = ['competitor', 'alternative', 'other vendor', 'comparison']
        for keyword in competitor_keywords:
            if keyword in message.lower():
                insights['competitor_discussion'] = True
                break
        
        # Decision maker indicators
        authority_keywords = ['boss', 'manager', 'team', 'committee', 'board', 'decision maker']
        for keyword in authority_keywords:
            if keyword in message.lower():
                insights['decision_maker_involvement'] = True
                break
        
        return insights
    
    async def _suggest_actions(self, message: str, context: ConversationContext) -> List[str]:
        """Suggest next actions based on message analysis"""
        actions = []
        message_lower = message.lower()
        
        # Action suggestions based on content
        if any(word in message_lower for word in ['demo', 'show', 'see it work']):
            actions.append("Schedule product demonstration")
        
        if any(word in message_lower for word in ['price', 'cost', 'budget']):
            actions.append("Provide pricing information")
            actions.append("Calculate ROI for customer")
        
        if any(word in message_lower for word in ['timeline', 'when', 'how long']):
            actions.append("Discuss implementation timeline")
        
        if any(word in message_lower for word in ['team', 'boss', 'approval']):
            actions.append("Identify decision makers")
            actions.append("Prepare stakeholder presentation")
        
        if any(word in message_lower for word in ['next step', 'move forward']):
            actions.append("Send proposal")
            actions.append("Schedule follow-up meeting")
        
        # Default actions if none specific
        if not actions:
            actions.append("Continue discovery conversation")
            actions.append("Ask clarifying questions")
        
        return actions
    
    async def generate_response(self, message: str, context: ConversationContext,
                              lead_data: Dict[str, Any] = None) -> AIResponse:
        """
        Generate intelligent AI response to customer message
        """
        try:
            # Analyze the incoming message
            analysis = await self.analyze_message(message, context)
            
            # Update context based on analysis
            context.current_stage = analysis['conversation_stage']
            context.engagement_level = analysis['engagement_level']
            
            # Generate response based on current stage and analysis
            if self.openai_api_key:
                response_content = await self._generate_openai_response(message, analysis, context, lead_data)
            else:
                response_content = await self._generate_local_response(message, analysis, context, lead_data)
            
            # Create AI response object
            ai_response = AIResponse(
                content=response_content,
                confidence_score=0.8,
                suggested_stage=analysis['conversation_stage'],
                follow_up_actions=analysis['next_actions'],
                detected_opportunities=self._identify_opportunities(analysis),
                risk_flags=self._identify_risks(analysis),
                personalization_elements=self._get_personalization_elements(lead_data)
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response()
    
    async def _generate_openai_response(self, message: str, analysis: Dict[str, Any], 
                                      context: ConversationContext, 
                                      lead_data: Dict[str, Any]) -> str:
        """Generate response using OpenAI GPT models"""
        
        # Build conversation history for context
        conversation_history = []
        for msg in context.messages[-5:]:  # Last 5 messages for context
            role = "user" if msg.message_type == MessageType.INBOUND else "assistant"
            conversation_history.append({
                "role": role,
                "content": msg.content
            })
        
        # Add current message
        conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Build system prompt
        system_prompt = self._build_system_prompt(analysis, context, lead_data)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_history
                ],
                temperature=0.7,
                max_tokens=500,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return await self._generate_local_response(message, analysis, context, lead_data)
    
    def _build_system_prompt(self, analysis: Dict[str, Any], 
                           context: ConversationContext,
                           lead_data: Dict[str, Any]) -> str:
        """Build system prompt for AI response generation"""
        
        company_name = lead_data.get('company', 'the company') if lead_data else 'the company'
        lead_name = lead_data.get('name', 'the prospect') if lead_data else 'the prospect'
        stage = context.current_stage.value
        
        base_prompt = f"""You are an expert AI sales assistant helping with a conversation with {lead_name} from {company_name}. 

Current conversation stage: {stage}
Lead engagement level: {context.engagement_level}
Detected sentiment: {analysis.get('sentiment', {}).get('type', 'neutral')}

Key conversation insights:
- Pain points identified: {context.pain_points}
- Objections detected: {[obj['type'] for obj in analysis.get('objections', [])]}
- Buying signals: {[signal['type'] for signal in analysis.get('buying_signals', [])]}

Your goals:
1. Build rapport and trust
2. Understand their needs and challenges  
3. Present relevant solutions
4. Handle objections professionally
5. Move the conversation forward toward a positive outcome

Guidelines:
- Be conversational and professional
- Ask open-ended questions to understand their situation
- Provide specific value propositions relevant to their industry
- Address concerns directly and honestly
- Use the prospect's name and company naturally
- Keep responses concise but substantive (2-4 sentences max)
- Always end with a question or clear next step

Respond to their message in a way that advances the sales conversation:"""
        
        return base_prompt
    
    async def _generate_local_response(self, message: str, analysis: Dict[str, Any],
                                     context: ConversationContext, 
                                     lead_data: Dict[str, Any]) -> str:
        """Generate response using local models and templates"""
        
        stage = context.current_stage
        objections = analysis.get('objections', [])
        buying_signals = analysis.get('buying_signals', [])
        
        # Handle objections first
        if objections:
            objection_type = objections[0]['type']
            if objection_type in self.response_templates['objection_handling']:
                template = np.random.choice(self.response_templates['objection_handling'][objection_type])
                response = self._personalize_template(template, lead_data)
                return response
        
        # Handle buying signals
        if buying_signals:
            if stage in [ConversationStage.SOLUTION_PRESENTATION, ConversationStage.CLOSING]:
                template = np.random.choice(self.response_templates['closing']['trial_close'])
                response = self._personalize_template(template, lead_data)
                return response
        
        # Stage-specific responses
        if stage == ConversationStage.INITIAL_CONTACT:
            template = np.random.choice(self.response_templates['initial_contact']['cold_outreach'])
            response = self._personalize_template(template, lead_data)
        elif stage == ConversationStage.DISCOVERY:
            template = np.random.choice(self.response_templates['discovery']['pain_point_questions'])
            response = self._personalize_template(template, lead_data)
        else:
            response = "Thank you for your message. I'd love to learn more about your specific needs and how we might be able to help. Could you tell me more about your current situation?"
        
        return response
    
    def _personalize_template(self, template: str, lead_data: Dict[str, Any]) -> str:
        """Personalize response template with lead data"""
        if not lead_data:
            return template
        
        # Simple template substitution
        personalizations = {
            'name': lead_data.get('name', 'there'),
            'company': lead_data.get('company', 'your company'),
            'industry': lead_data.get('industry', 'your industry'),
            'value_prop': 'increase efficiency and reduce costs',
            'solution_brief': 'optimize operations',
            'outcome': 'significant ROI',
            'business_metric': 'productivity',
            'timeframe': '3-6 months',
            'roi_metric': '25% cost reduction'
        }
        
        try:
            return template.format(**personalizations)
        except KeyError:
            return template
    
    def _identify_opportunities(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify sales opportunities from conversation analysis"""
        opportunities = []
        
        buying_signals = analysis.get('buying_signals', [])
        for signal in buying_signals:
            if signal['type'] == 'urgency':
                opportunities.append("Urgent need identified - accelerate process")
            elif signal['type'] == 'budget':
                opportunities.append("Budget discussion initiated - qualify amount")
            elif signal['type'] == 'decision':
                opportunities.append("Decision intent detected - prepare proposal")
        
        if analysis.get('engagement_level') in ['high', 'very_high']:
            opportunities.append("High engagement - schedule deeper conversation")
        
        return opportunities
    
    def _identify_risks(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risks from conversation analysis"""
        risks = []
        
        objections = analysis.get('objections', [])
        for objection in objections:
            risks.append(f"Objection raised: {objection['type']}")
        
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('type') in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]:
            risks.append("Negative sentiment detected")
        
        if analysis.get('engagement_level') == 'low':
            risks.append("Low engagement - risk of disqualification")
        
        insights = analysis.get('key_insights', {})
        if insights.get('competitor_discussion'):
            risks.append("Competitor discussion - differentiation needed")
        
        return risks
    
    def _get_personalization_elements(self, lead_data: Dict[str, Any]) -> Dict[str, str]:
        """Get personalization elements for response"""
        if not lead_data:
            return {}
        
        return {
            'name': lead_data.get('name', ''),
            'company': lead_data.get('company', ''),
            'title': lead_data.get('job_title', ''),
            'industry': lead_data.get('industry', ''),
            'location': lead_data.get('location', '')
        }
    
    def _get_fallback_response(self) -> AIResponse:
        """Fallback response when AI generation fails"""
        return AIResponse(
            content="Thank you for your message. I'd love to learn more about your needs and how we can help. Could you tell me more about your current situation?",
            confidence_score=0.5,
            suggested_stage=ConversationStage.DISCOVERY,
            follow_up_actions=["Continue discovery conversation"],
            detected_opportunities=[],
            risk_flags=["AI generation failed - manual review needed"],
            personalization_elements={}
        )

# Global instance
ai_conversation_engine = AIConversationEngine()