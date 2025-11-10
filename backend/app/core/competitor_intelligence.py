"""
Competitor Intelligence System
Automatic competitor mention detection with strategic response recommendations and competitive analysis
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
from collections import defaultdict, Counter

# NLP and ML libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from fuzzywuzzy import fuzz, process
except ImportError:
    pass

logger = logging.getLogger(__name__)

class CompetitorMentionType(Enum):
    """Types of competitor mentions"""
    DIRECT_MENTION = "direct_mention"
    INDIRECT_REFERENCE = "indirect_reference"
    FEATURE_COMPARISON = "feature_comparison"
    PRICE_COMPARISON = "price_comparison"
    EXPERIENCE_SHARING = "experience_sharing"
    EVALUATION_PROCESS = "evaluation_process"

class CompetitorSentiment(Enum):
    """Sentiment toward competitors"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class ResponseStrategy(Enum):
    """Strategic response approaches"""
    ACKNOWLEDGE_AND_DIFFERENTIATE = "acknowledge_and_differentiate"
    REDIRECT_TO_VALUE = "redirect_to_value"
    PROVIDE_COMPARISON = "provide_comparison"
    ADDRESS_CONCERNS = "address_concerns"
    MINIMIZE_AND_REFOCUS = "minimize_and_refocus"
    LEVERAGE_WEAKNESS = "leverage_weakness"

@dataclass
class CompetitorProfile:
    """Competitor profile with key information"""
    name: str
    aliases: List[str]
    market_position: str
    key_strengths: List[str]
    key_weaknesses: List[str]
    pricing_position: str
    target_segments: List[str]
    differentiation_points: List[str]

@dataclass
class CompetitorMention:
    """Individual competitor mention detection"""
    competitor_name: str
    mention_type: str
    confidence: float
    context_text: str
    sentiment_toward_competitor: str
    specific_aspects_mentioned: List[str]
    timestamp: datetime
    response_urgency: str

@dataclass
class CompetitiveAnalysis:
    """Comprehensive competitive analysis"""
    detected_competitors: List[str]
    mention_details: List[CompetitorMention]
    competitive_landscape: Dict[str, Any]
    threat_assessment: Dict[str, float]
    opportunity_analysis: Dict[str, Any]
    positioning_recommendations: List[str]

@dataclass
class StrategicResponse:
    """Strategic response to competitor mention"""
    response_strategy: str
    confidence: float
    recommended_talking_points: List[str]
    positioning_message: str
    differentiation_focus: List[str]
    response_tone: str
    next_steps: List[str]
    competitive_advantages: List[str]

@dataclass
class CompetitiveBattlecard:
    """Competitive battlecard for specific competitor"""
    competitor_name: str
    positioning_statement: str
    key_differentiators: List[str]
    objection_responses: Dict[str, str]
    proof_points: List[str]
    competitive_traps: List[str]
    win_loss_factors: Dict[str, List[str]]


class CompetitorDetectionEngine:
    """Core engine for detecting competitor mentions"""
    
    def __init__(self):
        self.competitor_profiles = self._load_competitor_profiles()
        self.competitor_keywords = self._build_competitor_keywords()
        self.detection_patterns = self._build_detection_patterns()
        self.nlp_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models for competitor detection"""
        try:
            # Initialize named entity recognition
            self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
            
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Initialize sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy for linguistic analysis
            try:
                self.nlp_models['en'] = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy English model not available")
            
            logger.info("Competitor detection engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing competitor detection models: {e}")
    
    def _load_competitor_profiles(self) -> Dict[str, CompetitorProfile]:
        """Load competitor profiles and intelligence"""
        # This would typically be loaded from a database or configuration file
        return {
            'salesforce': CompetitorProfile(
                name='Salesforce',
                aliases=['salesforce', 'sf', 'sales force', 'crm giant'],
                market_position='market_leader',
                key_strengths=['brand recognition', 'ecosystem', 'customization', 'integrations'],
                key_weaknesses=['complexity', 'high cost', 'steep learning curve', 'over-engineering'],
                pricing_position='premium',
                target_segments=['enterprise', 'large_companies'],
                differentiation_points=['simplicity', 'cost_effectiveness', 'ease_of_use', 'faster_implementation']
            ),
            'hubspot': CompetitorProfile(
                name='HubSpot',
                aliases=['hubspot', 'hub spot', 'hs'],
                market_position='strong_challenger',
                key_strengths=['inbound marketing', 'free tier', 'user friendly', 'integrated platform'],
                key_weaknesses=['limited customization', 'reporting limitations', 'enterprise features', 'scalability'],
                pricing_position='mid_market',
                target_segments=['smb', 'mid_market', 'marketing_focused'],
                differentiation_points=['advanced_ai', 'better_enterprise_features', 'superior_analytics']
            ),
            'pipedrive': CompetitorProfile(
                name='Pipedrive',
                aliases=['pipedrive', 'pipe drive'],
                market_position='niche_player',
                key_strengths=['simplicity', 'visual pipeline', 'affordable', 'sales focus'],
                key_weaknesses=['limited features', 'basic reporting', 'minimal automation', 'integrations'],
                pricing_position='budget',
                target_segments=['small_business', 'sales_teams'],
                differentiation_points=['advanced_automation', 'comprehensive_features', 'ai_capabilities']
            ),
            'microsoft': CompetitorProfile(
                name='Microsoft Dynamics',
                aliases=['dynamics', 'microsoft dynamics', 'ms dynamics', 'microsoft crm'],
                market_position='established_player',
                key_strengths=['microsoft integration', 'enterprise features', 'office 365', 'familiar interface'],
                key_weaknesses=['complexity', 'customization challenges', 'user experience', 'modern features'],
                pricing_position='premium',
                target_segments=['enterprise', 'microsoft_shops'],
                differentiation_points=['modern_interface', 'ai_first_approach', 'better_ux', 'faster_setup']
            )
        }
    
    def _build_competitor_keywords(self) -> Dict[str, List[str]]:
        """Build comprehensive keyword lists for each competitor"""
        keywords = {}
        
        for comp_key, profile in self.competitor_profiles.items():
            # Combine name, aliases, and common variations
            comp_keywords = [profile.name.lower()]
            comp_keywords.extend([alias.lower() for alias in profile.aliases])
            
            # Add common variations
            name_parts = profile.name.lower().split()
            comp_keywords.extend(name_parts)
            
            # Add acronyms
            if len(name_parts) > 1:
                acronym = ''.join([word[0] for word in name_parts])
                comp_keywords.append(acronym)
            
            keywords[comp_key] = list(set(comp_keywords))  # Remove duplicates
        
        return keywords
    
    def _build_detection_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for detecting competitor mentions"""
        return {
            'direct_mention': [
                r'\b(?:we use|using|have|tried|considering)\s+(\w+)',
                r'\b(\w+)\s+(?:is our|was our|is the)\s+(?:crm|system|platform)',
                r'\bcompared to\s+(\w+)',
                r'\b(\w+)\s+(?:vs|versus)\s+',
                r'\bcurrently (?:on|with|using)\s+(\w+)'
            ],
            'feature_comparison': [
                r'\b(\w+)\s+(?:has|offers|provides|includes)\s+',
                r'\bwith\s+(\w+)\s+(?:we can|you can|it\'s possible)',
                r'\b(\w+)\'s\s+(?:feature|functionality|capability)',
                r'\bunlike\s+(\w+)',
                r'\bbetter than\s+(\w+)'
            ],
            'price_comparison': [
                r'\b(\w+)\s+(?:costs?|pricing|price|expensive|cheap)',
                r'\bpaying\s+(\w+)',
                r'\b(\w+)\s+(?:is|was)\s+(?:too\s+)?(?:expensive|costly|cheap)',
                r'\bbudget for\s+(\w+)',
                r'\b(\w+)\s+subscription'
            ],
            'evaluation': [
                r'\bevaluating\s+(\w+)',
                r'\blooking at\s+(\w+)',
                r'\bconsidering\s+(\w+)',
                r'\bdemoed?\s+(\w+)',
                r'\btrial of\s+(\w+)'
            ]
        }
    
    async def detect_competitor_mentions(self, text: str) -> List[CompetitorMention]:
        """Detect competitor mentions in text"""
        try:
            mentions = []
            text_lower = text.lower()
            
            # Direct keyword matching
            keyword_mentions = await self._detect_keyword_mentions(text, text_lower)
            mentions.extend(keyword_mentions)
            
            # Pattern-based detection
            pattern_mentions = await self._detect_pattern_mentions(text, text_lower)
            mentions.extend(pattern_mentions)
            
            # Named entity recognition
            ner_mentions = await self._detect_ner_mentions(text)
            mentions.extend(ner_mentions)
            
            # Fuzzy matching for variations
            fuzzy_mentions = await self._detect_fuzzy_mentions(text, text_lower)
            mentions.extend(fuzzy_mentions)
            
            # Remove duplicates and merge similar mentions
            merged_mentions = await self._merge_similar_mentions(mentions)
            
            # Enhance with sentiment and context analysis
            enhanced_mentions = []
            for mention in merged_mentions:
                enhanced_mention = await self._enhance_mention_analysis(mention, text)
                enhanced_mentions.append(enhanced_mention)
            
            return enhanced_mentions
            
        except Exception as e:
            logger.error(f"Error detecting competitor mentions: {e}")
            return []
    
    async def _detect_keyword_mentions(self, text: str, text_lower: str) -> List[CompetitorMention]:
        """Detect mentions using direct keyword matching"""
        mentions = []
        
        for comp_key, keywords in self.competitor_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the actual position and context
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = re.finditer(pattern, text_lower)
                    
                    for match in matches:
                        start, end = match.span()
                        
                        # Get surrounding context
                        context_start = max(0, start - 50)
                        context_end = min(len(text), end + 50)
                        context = text[context_start:context_end]
                        
                        mention = CompetitorMention(
                            competitor_name=self.competitor_profiles[comp_key].name,
                            mention_type=CompetitorMentionType.DIRECT_MENTION.value,
                            confidence=0.8,  # High confidence for direct keyword match
                            context_text=context,
                            sentiment_toward_competitor='neutral',  # Will be enhanced later
                            specific_aspects_mentioned=[],
                            timestamp=datetime.now(),
                            response_urgency='medium'
                        )
                        mentions.append(mention)
        
        return mentions
    
    async def _detect_pattern_mentions(self, text: str, text_lower: str) -> List[CompetitorMention]:
        """Detect mentions using regex patterns"""
        mentions = []
        
        for pattern_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    # Extract the potential competitor name
                    if match.groups():
                        potential_competitor = match.group(1)
                        
                        # Check if this matches any known competitor
                        competitor_match = await self._match_to_known_competitor(potential_competitor)
                        
                        if competitor_match:
                            comp_key, confidence = competitor_match
                            
                            mention = CompetitorMention(
                                competitor_name=self.competitor_profiles[comp_key].name,
                                mention_type=self._pattern_to_mention_type(pattern_type),
                                confidence=confidence * 0.7,  # Slightly lower confidence for pattern matching
                                context_text=match.group(0),
                                sentiment_toward_competitor='neutral',
                                specific_aspects_mentioned=[],
                                timestamp=datetime.now(),
                                response_urgency='medium'
                            )
                            mentions.append(mention)
        
        return mentions
    
    async def _detect_ner_mentions(self, text: str) -> List[CompetitorMention]:
        """Detect competitor mentions using named entity recognition"""
        mentions = []
        
        try:
            if not hasattr(self, 'ner_pipeline'):
                return mentions
            
            # Extract named entities
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                if entity['entity_group'] in ['ORG', 'PRODUCT']:  # Organization or product entities
                    entity_text = entity['word'].lower()
                    
                    # Check if this entity matches a known competitor
                    competitor_match = await self._match_to_known_competitor(entity_text)
                    
                    if competitor_match:
                        comp_key, confidence = competitor_match
                        
                        mention = CompetitorMention(
                            competitor_name=self.competitor_profiles[comp_key].name,
                            mention_type=CompetitorMentionType.DIRECT_MENTION.value,
                            confidence=confidence * entity['score'],  # Combine NER confidence with match confidence
                            context_text=entity['word'],
                            sentiment_toward_competitor='neutral',
                            specific_aspects_mentioned=[],
                            timestamp=datetime.now(),
                            response_urgency='medium'
                        )
                        mentions.append(mention)
                        
        except Exception as e:
            logger.error(f"Error in NER detection: {e}")
        
        return mentions
    
    async def _detect_fuzzy_mentions(self, text: str, text_lower: str) -> List[CompetitorMention]:
        """Detect competitor mentions using fuzzy string matching"""
        mentions = []
        
        try:
            words = text_lower.split()
            
            for comp_key, keywords in self.competitor_keywords.items():
                for keyword in keywords:
                    if len(keyword) > 3:  # Only fuzzy match longer keywords
                        for word in words:
                            # Use fuzzy matching
                            ratio = fuzz.ratio(keyword, word)
                            
                            if ratio > 80:  # High similarity threshold
                                mention = CompetitorMention(
                                    competitor_name=self.competitor_profiles[comp_key].name,
                                    mention_type=CompetitorMentionType.INDIRECT_REFERENCE.value,
                                    confidence=(ratio / 100.0) * 0.6,  # Lower confidence for fuzzy matches
                                    context_text=word,
                                    sentiment_toward_competitor='neutral',
                                    specific_aspects_mentioned=[],
                                    timestamp=datetime.now(),
                                    response_urgency='low'
                                )
                                mentions.append(mention)
                                
        except Exception as e:
            logger.error(f"Error in fuzzy detection: {e}")
        
        return mentions
    
    async def _match_to_known_competitor(self, text: str) -> Optional[Tuple[str, float]]:
        """Match text to known competitor with confidence score"""
        text_lower = text.lower().strip()
        
        for comp_key, keywords in self.competitor_keywords.items():
            for keyword in keywords:
                if keyword == text_lower:
                    return comp_key, 1.0  # Exact match
                elif keyword in text_lower or text_lower in keyword:
                    return comp_key, 0.8  # Partial match
        
        # Use fuzzy matching for close matches
        all_keywords = []
        keyword_to_competitor = {}
        
        for comp_key, keywords in self.competitor_keywords.items():
            for keyword in keywords:
                all_keywords.append(keyword)
                keyword_to_competitor[keyword] = comp_key
        
        try:
            best_match = process.extractOne(text_lower, all_keywords, score_cutoff=70)
            if best_match:
                matched_keyword, score = best_match[0], best_match[1]
                comp_key = keyword_to_competitor[matched_keyword]
                return comp_key, score / 100.0
        except:
            pass
        
        return None
    
    def _pattern_to_mention_type(self, pattern_type: str) -> str:
        """Convert pattern type to mention type"""
        mapping = {
            'feature_comparison': CompetitorMentionType.FEATURE_COMPARISON.value,
            'price_comparison': CompetitorMentionType.PRICE_COMPARISON.value,
            'evaluation': CompetitorMentionType.EVALUATION_PROCESS.value
        }
        return mapping.get(pattern_type, CompetitorMentionType.DIRECT_MENTION.value)
    
    async def _merge_similar_mentions(self, mentions: List[CompetitorMention]) -> List[CompetitorMention]:
        """Merge similar competitor mentions to avoid duplicates"""
        if not mentions:
            return mentions
        
        merged = []
        processed = set()
        
        for i, mention in enumerate(mentions):
            if i in processed:
                continue
            
            # Find similar mentions
            similar_indices = [i]
            for j, other_mention in enumerate(mentions[i+1:], i+1):
                if (mention.competitor_name == other_mention.competitor_name and
                    abs((mention.timestamp - other_mention.timestamp).total_seconds()) < 10):  # Within 10 seconds
                    similar_indices.append(j)
                    processed.add(j)
            
            if len(similar_indices) > 1:
                # Merge mentions
                best_confidence = max(mentions[idx].confidence for idx in similar_indices)
                combined_context = ' | '.join(mentions[idx].context_text for idx in similar_indices)
                
                merged_mention = CompetitorMention(
                    competitor_name=mention.competitor_name,
                    mention_type=mention.mention_type,
                    confidence=best_confidence,
                    context_text=combined_context,
                    sentiment_toward_competitor=mention.sentiment_toward_competitor,
                    specific_aspects_mentioned=mention.specific_aspects_mentioned,
                    timestamp=mention.timestamp,
                    response_urgency=mention.response_urgency
                )
                merged.append(merged_mention)
            else:
                merged.append(mention)
            
            processed.add(i)
        
        return merged
    
    async def _enhance_mention_analysis(self, mention: CompetitorMention, full_text: str) -> CompetitorMention:
        """Enhance mention with sentiment and aspect analysis"""
        try:
            # Sentiment analysis
            sentiment = await self._analyze_competitor_sentiment(mention.context_text, full_text)
            mention.sentiment_toward_competitor = sentiment
            
            # Extract specific aspects mentioned
            aspects = await self._extract_mentioned_aspects(mention.context_text, full_text, mention.competitor_name)
            mention.specific_aspects_mentioned = aspects
            
            # Determine response urgency
            urgency = await self._determine_response_urgency(mention, full_text)
            mention.response_urgency = urgency
            
        except Exception as e:
            logger.error(f"Error enhancing mention analysis: {e}")
        
        return mention
    
    async def _analyze_competitor_sentiment(self, context: str, full_text: str) -> str:
        """Analyze sentiment toward competitor"""
        try:
            if not hasattr(self, 'sentiment_analyzer'):
                return CompetitorSentiment.NEUTRAL.value
            
            # Analyze sentiment of the context
            sentiment_result = self.sentiment_analyzer(context)
            sentiment_label = sentiment_result[0]['label'].lower()
            confidence = sentiment_result[0]['score']
            
            # Map to our sentiment categories
            if 'positive' in sentiment_label:
                if confidence > 0.8:
                    return CompetitorSentiment.POSITIVE.value
                else:
                    return CompetitorSentiment.NEUTRAL.value
            elif 'negative' in sentiment_label:
                if confidence > 0.8:
                    return CompetitorSentiment.NEGATIVE.value
                else:
                    return CompetitorSentiment.NEUTRAL.value
            else:
                return CompetitorSentiment.NEUTRAL.value
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return CompetitorSentiment.NEUTRAL.value
    
    async def _extract_mentioned_aspects(self, context: str, full_text: str, competitor_name: str) -> List[str]:
        """Extract specific aspects mentioned about competitor"""
        aspects = []
        context_lower = context.lower()
        
        # Common aspect keywords
        aspect_keywords = {
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'budget', 'pricing', 'fee'],
            'features': ['feature', 'functionality', 'capability', 'tool', 'option'],
            'usability': ['easy', 'difficult', 'user-friendly', 'interface', 'ux', 'experience'],
            'performance': ['fast', 'slow', 'performance', 'speed', 'reliable'],
            'support': ['support', 'help', 'service', 'customer service'],
            'integration': ['integrate', 'connect', 'api', 'compatibility'],
            'reporting': ['report', 'analytics', 'dashboard', 'insights'],
            'customization': ['customize', 'configure', 'flexible', 'adaptable']
        }
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                aspects.append(aspect)
        
        return aspects
    
    async def _determine_response_urgency(self, mention: CompetitorMention, full_text: str) -> str:
        """Determine urgency of response needed"""
        urgency_indicators = {
            'high': ['switching', 'moving to', 'decided on', 'signed with', 'contract'],
            'medium': ['considering', 'evaluating', 'looking at', 'comparing'],
            'low': ['heard of', 'mentioned', 'aware of']
        }
        
        context_lower = mention.context_text.lower()
        full_text_lower = full_text.lower()
        
        for urgency_level, indicators in urgency_indicators.items():
            if any(indicator in context_lower or indicator in full_text_lower for indicator in indicators):
                return urgency_level
        
        # Default based on mention type
        if mention.mention_type == CompetitorMentionType.EVALUATION_PROCESS.value:
            return 'high'
        elif mention.mention_type in [CompetitorMentionType.FEATURE_COMPARISON.value, CompetitorMentionType.PRICE_COMPARISON.value]:
            return 'medium'
        else:
            return 'low'


class CompetitiveResponseEngine:
    """Engine for generating strategic responses to competitor mentions"""
    
    def __init__(self):
        self.competitor_profiles = {}  # Will be populated from detection engine
        self.battlecards = self._load_battlecards()
        self.response_templates = self._load_response_templates()
    
    def set_competitor_profiles(self, profiles: Dict[str, CompetitorProfile]):
        """Set competitor profiles from detection engine"""
        self.competitor_profiles = profiles
    
    def _load_battlecards(self) -> Dict[str, CompetitiveBattlecard]:
        """Load competitive battlecards"""
        return {
            'salesforce': CompetitiveBattlecard(
                competitor_name='Salesforce',
                positioning_statement="While Salesforce is a powerful platform, our solution offers the same enterprise capabilities with significantly better usability and faster implementation.",
                key_differentiators=[
                    "70% faster implementation time",
                    "Intuitive interface requires minimal training",
                    "AI-first approach with built-in intelligence",
                    "Transparent pricing with no hidden fees",
                    "Superior customer success support"
                ],
                objection_responses={
                    "salesforce_brand": "Salesforce has great brand recognition, but what really matters is which solution will get your team productive fastest and deliver the best ROI.",
                    "salesforce_features": "Salesforce does have many features, but most companies use less than 20% of them. Our solution focuses on the features that actually drive results.",
                    "salesforce_ecosystem": "While Salesforce has a large app ecosystem, most integrations are complex and costly. Our native integrations are simpler and more reliable."
                },
                proof_points=[
                    "Customer implementation 70% faster on average",
                    "95% user adoption rate vs industry average of 40%",
                    "ROI achieved 6 months sooner than Salesforce implementations"
                ],
                competitive_traps=[
                    "Ask about their experience with Salesforce complexity",
                    "Inquire about their timeline for seeing ROI",
                    "Discuss their team's technical expertise level"
                ],
                win_loss_factors={
                    'win_factors': ['simplicity', 'speed_of_value', 'cost_effectiveness'],
                    'loss_factors': ['brand_perception', 'extensive_customization_needs']
                }
            )
        }
    
    def _load_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response strategy templates"""
        return {
            ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value: {
                'approach': 'acknowledge_then_differentiate',
                'tone': 'respectful_confident',
                'structure': [
                    'acknowledge_competitor_strengths',
                    'transition_to_differentiation',
                    'provide_specific_advantages',
                    'reinforce_value_proposition'
                ]
            },
            ResponseStrategy.REDIRECT_TO_VALUE.value: {
                'approach': 'minimize_then_refocus',
                'tone': 'consultative_helpful',
                'structure': [
                    'brief_acknowledgment',
                    'redirect_to_customer_needs',
                    'focus_on_value_delivery',
                    'demonstrate_superiority'
                ]
            },
            ResponseStrategy.PROVIDE_COMPARISON.value: {
                'approach': 'direct_comparison',
                'tone': 'factual_confident',
                'structure': [
                    'acknowledge_comparison_request',
                    'present_factual_comparison',
                    'highlight_key_advantages',
                    'offer_proof_points'
                ]
            },
            ResponseStrategy.ADDRESS_CONCERNS.value: {
                'approach': 'empathetic_problem_solving',
                'tone': 'understanding_helpful',
                'structure': [
                    'acknowledge_concerns',
                    'provide_reassurance',
                    'address_specific_issues',
                    'offer_solutions'
                ]
            },
            ResponseStrategy.LEVERAGE_WEAKNESS.value: {
                'approach': 'strategic_weakness_focus',
                'tone': 'consultative_insightful',
                'structure': [
                    'acknowledge_competitor_position',
                    'introduce_concern_questions',
                    'highlight_alternative_approach',
                    'demonstrate_advantage'
                ]
            }
        }
    
    async def generate_strategic_response(self, mentions: List[CompetitorMention], conversation_context: Dict[str, Any]) -> List[StrategicResponse]:
        """Generate strategic responses for competitor mentions"""
        responses = []
        
        for mention in mentions:
            try:
                response = await self._generate_single_response(mention, conversation_context)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for {mention.competitor_name}: {e}")
        
        return responses
    
    async def _generate_single_response(self, mention: CompetitorMention, context: Dict[str, Any]) -> StrategicResponse:
        """Generate strategic response for single competitor mention"""
        
        # Determine response strategy
        strategy = await self._determine_response_strategy(mention, context)
        
        # Get battlecard for competitor
        battlecard = self._get_battlecard(mention.competitor_name)
        
        # Generate talking points
        talking_points = await self._generate_talking_points(mention, strategy, battlecard)
        
        # Create positioning message
        positioning_message = await self._create_positioning_message(mention, strategy, battlecard)
        
        # Determine differentiation focus
        differentiation_focus = await self._determine_differentiation_focus(mention, battlecard)
        
        # Generate next steps
        next_steps = await self._generate_next_steps(mention, strategy)
        
        # Extract competitive advantages
        competitive_advantages = await self._extract_competitive_advantages(mention, battlecard)
        
        # Calculate confidence
        confidence = await self._calculate_response_confidence(mention, strategy, battlecard)
        
        return StrategicResponse(
            response_strategy=strategy,
            confidence=confidence,
            recommended_talking_points=talking_points,
            positioning_message=positioning_message,
            differentiation_focus=differentiation_focus,
            response_tone=self.response_templates[strategy]['tone'],
            next_steps=next_steps,
            competitive_advantages=competitive_advantages
        )
    
    async def _determine_response_strategy(self, mention: CompetitorMention, context: Dict[str, Any]) -> str:
        """Determine optimal response strategy"""
        
        # Strategy based on mention type
        if mention.mention_type == CompetitorMentionType.FEATURE_COMPARISON.value:
            return ResponseStrategy.PROVIDE_COMPARISON.value
        elif mention.mention_type == CompetitorMentionType.PRICE_COMPARISON.value:
            return ResponseStrategy.REDIRECT_TO_VALUE.value
        elif mention.mention_type == CompetitorMentionType.EXPERIENCE_SHARING.value:
            if mention.sentiment_toward_competitor == CompetitorSentiment.NEGATIVE.value:
                return ResponseStrategy.LEVERAGE_WEAKNESS.value
            else:
                return ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value
        elif mention.mention_type == CompetitorMentionType.EVALUATION_PROCESS.value:
            return ResponseStrategy.PROVIDE_COMPARISON.value
        
        # Strategy based on sentiment
        if mention.sentiment_toward_competitor == CompetitorSentiment.VERY_POSITIVE.value:
            return ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value
        elif mention.sentiment_toward_competitor in [CompetitorSentiment.NEGATIVE.value, CompetitorSentiment.VERY_NEGATIVE.value]:
            return ResponseStrategy.LEVERAGE_WEAKNESS.value
        
        # Strategy based on urgency
        if mention.response_urgency == 'high':
            return ResponseStrategy.PROVIDE_COMPARISON.value
        
        # Default strategy
        return ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value
    
    def _get_battlecard(self, competitor_name: str) -> Optional[CompetitiveBattlecard]:
        """Get battlecard for competitor"""
        competitor_key = competitor_name.lower().replace(' ', '').replace('-', '')
        return self.battlecards.get(competitor_key)
    
    async def _generate_talking_points(self, mention: CompetitorMention, strategy: str, battlecard: Optional[CompetitiveBattlecard]) -> List[str]:
        """Generate talking points for response"""
        talking_points = []
        
        if not battlecard:
            # Generic talking points
            talking_points = [
                f"I understand you're familiar with {mention.competitor_name}",
                "Let me share how our approach differs and might be more suitable for your needs",
                "We focus on delivering faster time-to-value and better user experience"
            ]
            return talking_points
        
        # Strategy-specific talking points
        if strategy == ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value:
            talking_points.extend([
                f"{mention.competitor_name} is certainly a well-known solution in the market",
                "Here's how our approach differs in key ways that matter to companies like yours",
                *battlecard.key_differentiators[:3]
            ])
        
        elif strategy == ResponseStrategy.PROVIDE_COMPARISON.value:
            talking_points.extend([
                "I'm happy to provide a detailed comparison",
                f"Both solutions have their merits, but here's where we excel",
                *battlecard.proof_points
            ])
        
        elif strategy == ResponseStrategy.LEVERAGE_WEAKNESS.value:
            talking_points.extend([
                "I understand you've had some experience with them",
                "Many of our clients came from similar situations",
                "Let me show you how we specifically address those common challenges"
            ])
        
        # Add aspect-specific points
        for aspect in mention.specific_aspects_mentioned:
            if aspect == 'pricing' and 'cost_effectiveness' in battlecard.key_differentiators:
                talking_points.append("Our transparent pricing model eliminates surprise costs")
            elif aspect == 'usability' and any('user' in diff.lower() for diff in battlecard.key_differentiators):
                talking_points.append("User adoption is typically 2-3x higher with our intuitive interface")
        
        return talking_points[:5]  # Limit to 5 key points
    
    async def _create_positioning_message(self, mention: CompetitorMention, strategy: str, battlecard: Optional[CompetitiveBattlecard]) -> str:
        """Create core positioning message"""
        
        if battlecard and battlecard.positioning_statement:
            return battlecard.positioning_statement
        
        # Generic positioning based on strategy
        positioning_messages = {
            ResponseStrategy.ACKNOWLEDGE_AND_DIFFERENTIATE.value: f"While {mention.competitor_name} has its strengths, our solution is specifically designed to address the challenges that companies like yours face with faster implementation and better results.",
            
            ResponseStrategy.REDIRECT_TO_VALUE.value: f"Rather than comparing features, let's focus on what will deliver the best outcome for your business - faster ROI, higher user adoption, and measurable results.",
            
            ResponseStrategy.PROVIDE_COMPARISON.value: f"I can provide a detailed comparison with {mention.competitor_name}. What you'll find is that while both solutions have their place, ours consistently delivers better results in the areas that matter most to your business.",
            
            ResponseStrategy.LEVERAGE_WEAKNESS.value: f"I understand your experience with {mention.competitor_name}. We specifically designed our solution to address the common challenges that companies face with traditional platforms."
        }
        
        return positioning_messages.get(strategy, f"Our solution offers a modern alternative to {mention.competitor_name} with better outcomes.")
    
    async def _determine_differentiation_focus(self, mention: CompetitorMention, battlecard: Optional[CompetitiveBattlecard]) -> List[str]:
        """Determine key areas to focus differentiation on"""
        if not battlecard:
            return ['ease_of_use', 'implementation_speed', 'roi_acceleration']
        
        focus_areas = []
        
        # Map mentioned aspects to differentiation areas
        aspect_mapping = {
            'pricing': 'cost_effectiveness',
            'features': 'feature_superiority',
            'usability': 'user_experience',
            'performance': 'performance_advantages',
            'support': 'customer_success',
            'integration': 'integration_simplicity'
        }
        
        # Focus on mentioned aspects
        for aspect in mention.specific_aspects_mentioned:
            if aspect in aspect_mapping:
                focus_areas.append(aspect_mapping[aspect])
        
        # Add key differentiators from battlecard
        focus_areas.extend(battlecard.key_differentiators[:3])
        
        return list(set(focus_areas))  # Remove duplicates
    
    async def _generate_next_steps(self, mention: CompetitorMention, strategy: str) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        
        if mention.response_urgency == 'high':
            next_steps.extend([
                "Schedule detailed comparison session",
                "Provide ROI analysis specific to your situation",
                "Arrange reference call with similar customer"
            ])
        elif mention.response_urgency == 'medium':
            next_steps.extend([
                "Share detailed competitive comparison document",
                "Demonstrate key differentiating features",
                "Provide trial access for hands-on evaluation"
            ])
        else:
            next_steps.extend([
                "Send additional information about our approach",
                "Schedule follow-up to discuss specific needs",
                "Provide relevant case studies"
            ])
        
        return next_steps
    
    async def _extract_competitive_advantages(self, mention: CompetitorMention, battlecard: Optional[CompetitiveBattlecard]) -> List[str]:
        """Extract relevant competitive advantages"""
        if not battlecard:
            return ["Faster implementation", "Better user experience", "Superior support"]
        
        advantages = []
        
        # Include key differentiators
        advantages.extend(battlecard.key_differentiators)
        
        # Add proof points as advantages
        for proof_point in battlecard.proof_points:
            if not any(proof_point.lower() in adv.lower() for adv in advantages):
                advantages.append(proof_point)
        
        return advantages[:5]  # Limit to top 5
    
    async def _calculate_response_confidence(self, mention: CompetitorMention, strategy: str, battlecard: Optional[CompetitiveBattlecard]) -> float:
        """Calculate confidence in response strategy"""
        base_confidence = 0.6
        
        # Boost confidence if we have a good battlecard
        if battlecard:
            base_confidence += 0.2
        
        # Boost confidence if mention has specific aspects we can address
        if mention.specific_aspects_mentioned:
            base_confidence += 0.1
        
        # Adjust based on mention confidence
        confidence_boost = mention.confidence * 0.2
        
        # Strategy-specific adjustments
        if strategy == ResponseStrategy.LEVERAGE_WEAKNESS.value and mention.sentiment_toward_competitor == CompetitorSentiment.NEGATIVE.value:
            base_confidence += 0.1
        
        final_confidence = min(base_confidence + confidence_boost, 1.0)
        return final_confidence


class CompetitorIntelligenceManager:
    """Main manager for competitor intelligence system"""
    
    def __init__(self):
        self.detection_engine = CompetitorDetectionEngine()
        self.response_engine = CompetitiveResponseEngine()
        self.competitive_history: Dict[str, List[Dict[str, Any]]] = {}
        
    async def initialize(self):
        """Initialize the competitor intelligence manager"""
        logger.info("Initializing Competitor Intelligence System...")
        
        # Share competitor profiles between engines
        self.response_engine.set_competitor_profiles(self.detection_engine.competitor_profiles)
        
        logger.info("Competitor Intelligence System initialized successfully")
    
    async def analyze_competitive_conversation(self, 
                                            conversation_id: str,
                                            message: str,
                                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive competitive analysis of conversation"""
        try:
            # Detect competitor mentions
            competitor_mentions = await self.detection_engine.detect_competitor_mentions(message)
            
            if not competitor_mentions:
                return {
                    'conversation_id': conversation_id,
                    'competitive_mentions': [],
                    'strategic_responses': [],
                    'competitive_landscape': {},
                    'recommendations': ['No competitor mentions detected'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate strategic responses
            strategic_responses = await self.response_engine.generate_strategic_response(
                competitor_mentions, conversation_context
            )
            
            # Analyze competitive landscape
            competitive_landscape = await self._analyze_competitive_landscape(
                competitor_mentions, conversation_id
            )
            
            # Generate recommendations
            recommendations = await self._generate_competitive_recommendations(
                competitor_mentions, strategic_responses, competitive_landscape
            )
            
            # Update competitive history
            await self._update_competitive_history(conversation_id, competitor_mentions, strategic_responses)
            
            return {
                'conversation_id': conversation_id,
                'competitive_mentions': [asdict(mention) for mention in competitor_mentions],
                'strategic_responses': [asdict(response) for response in strategic_responses],
                'competitive_landscape': competitive_landscape,
                'recommendations': recommendations,
                'analysis_summary': await self._generate_analysis_summary(competitor_mentions, strategic_responses),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in competitive conversation analysis: {e}")
            return {
                'conversation_id': conversation_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_competitive_landscape(self, 
                                           mentions: List[CompetitorMention],
                                           conversation_id: str) -> Dict[str, Any]:
        """Analyze the competitive landscape"""
        landscape = {
            'competitors_mentioned': [],
            'competitive_intensity': 0.0,
            'threat_levels': {},
            'opportunity_areas': []
        }
        
        # Extract competitors
        competitors = list(set(mention.competitor_name for mention in mentions))
        landscape['competitors_mentioned'] = competitors
        
        # Calculate competitive intensity
        total_confidence = sum(mention.confidence for mention in mentions)
        high_urgency_count = sum(1 for mention in mentions if mention.response_urgency == 'high')
        landscape['competitive_intensity'] = min((total_confidence + high_urgency_count) / len(mentions), 1.0)
        
        # Assess threat levels
        for competitor in competitors:
            competitor_mentions = [m for m in mentions if m.competitor_name == competitor]
            avg_confidence = sum(m.confidence for m in competitor_mentions) / len(competitor_mentions)
            high_urgency = any(m.response_urgency == 'high' for m in competitor_mentions)
            
            threat_level = avg_confidence
            if high_urgency:
                threat_level += 0.3
            
            landscape['threat_levels'][competitor] = min(threat_level, 1.0)
        
        # Identify opportunity areas
        aspect_counts = Counter()
        for mention in mentions:
            aspect_counts.update(mention.specific_aspects_mentioned)
        
        # Top mentioned aspects are opportunity areas
        landscape['opportunity_areas'] = [aspect for aspect, count in aspect_counts.most_common(3)]
        
        return landscape
    
    async def _generate_competitive_recommendations(self,
                                                 mentions: List[CompetitorMention],
                                                 responses: List[StrategicResponse],
                                                 landscape: Dict[str, Any]) -> List[str]:
        """Generate actionable competitive recommendations"""
        recommendations = []
        
        # High-urgency situations
        high_urgency_mentions = [m for m in mentions if m.response_urgency == 'high']
        if high_urgency_mentions:
            recommendations.append(f"URGENT: {len(high_urgency_mentions)} high-priority competitive situations require immediate attention")
        
        # Competitive intensity recommendations
        if landscape['competitive_intensity'] > 0.7:
            recommendations.append("High competitive intensity detected - consider bringing in competitive specialist")
        
        # Threat-specific recommendations
        for competitor, threat_level in landscape['threat_levels'].items():
            if threat_level > 0.8:
                recommendations.append(f"High threat from {competitor} - deploy competitive battlecard immediately")
        
        # Opportunity recommendations
        if landscape['opportunity_areas']:
            top_area = landscape['opportunity_areas'][0]
            recommendations.append(f"Focus differentiation messaging on {top_area} - most frequently mentioned concern")
        
        # Response confidence recommendations
        low_confidence_responses = [r for r in responses if r.confidence < 0.6]
        if low_confidence_responses:
            recommendations.append("Some competitive responses have low confidence - consider additional preparation")
        
        # Strategy recommendations
        strategy_counts = Counter(r.response_strategy for r in responses)
        most_common_strategy = strategy_counts.most_common(1)[0][0]
        recommendations.append(f"Primary strategy: {most_common_strategy.replace('_', ' ').title()}")
        
        return recommendations
    
    async def _update_competitive_history(self,
                                        conversation_id: str,
                                        mentions: List[CompetitorMention],
                                        responses: List[StrategicResponse]):
        """Update competitive history for analysis"""
        if conversation_id not in self.competitive_history:
            self.competitive_history[conversation_id] = []
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'mentions': [asdict(mention) for mention in mentions],
            'responses': [asdict(response) for response in responses],
            'competitors': list(set(mention.competitor_name for mention in mentions))
        }
        
        self.competitive_history[conversation_id].append(history_entry)
        
        # Keep only recent history (last 50 entries)
        if len(self.competitive_history[conversation_id]) > 50:
            self.competitive_history[conversation_id] = self.competitive_history[conversation_id][-50:]
    
    async def _generate_analysis_summary(self,
                                       mentions: List[CompetitorMention],
                                       responses: List[StrategicResponse]) -> str:
        """Generate concise analysis summary"""
        if not mentions:
            return "No competitive mentions detected in this conversation."
        
        competitors = list(set(mention.competitor_name for mention in mentions))
        total_mentions = len(mentions)
        high_urgency = sum(1 for mention in mentions if mention.response_urgency == 'high')
        avg_confidence = sum(response.confidence for response in responses) / len(responses) if responses else 0.0
        
        summary_parts = [
            f"Detected {total_mentions} competitive mention{'s' if total_mentions != 1 else ''} involving {len(competitors)} competitor{'s' if len(competitors) != 1 else ''}: {', '.join(competitors)}"
        ]
        
        if high_urgency > 0:
            summary_parts.append(f"{high_urgency} high-urgency situation{'s' if high_urgency != 1 else ''} requiring immediate attention")
        
        summary_parts.append(f"Strategic response confidence: {avg_confidence:.1%}")
        
        return ". ".join(summary_parts)
    
    async def get_competitive_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get competitive insights for conversation"""
        history = self.competitive_history.get(conversation_id, [])
        
        if not history:
            return {'insights': 'No competitive history available for this conversation'}
        
        # Analyze patterns
        all_competitors = []
        all_strategies = []
        urgency_levels = []
        
        for entry in history:
            all_competitors.extend(entry['competitors'])
            for response in entry['responses']:
                all_strategies.append(response['response_strategy'])
            for mention in entry['mentions']:
                urgency_levels.append(mention['response_urgency'])
        
        # Generate insights
        competitor_frequency = Counter(all_competitors)
        strategy_frequency = Counter(all_strategies)
        urgency_distribution = Counter(urgency_levels)
        
        insights = {
            'conversation_id': conversation_id,
            'competitive_history_length': len(history),
            'most_mentioned_competitors': dict(competitor_frequency.most_common(3)),
            'most_used_strategies': dict(strategy_frequency.most_common(3)),
            'urgency_distribution': dict(urgency_distribution),
            'competitive_trend': await self._analyze_competitive_trend(history),
            'recommendations': await self._generate_historical_recommendations(history)
        }
        
        return insights
    
    async def _analyze_competitive_trend(self, history: List[Dict[str, Any]]) -> str:
        """Analyze trend in competitive mentions over time"""
        if len(history) < 2:
            return 'insufficient_data'
        
        recent_entries = history[-3:] if len(history) >= 3 else history
        earlier_entries = history[:-3] if len(history) >= 6 else []
        
        if not earlier_entries:
            return 'stable'
        
        recent_avg = sum(len(entry['mentions']) for entry in recent_entries) / len(recent_entries)
        earlier_avg = sum(len(entry['mentions']) for entry in earlier_entries) / len(earlier_entries)
        
        if recent_avg > earlier_avg * 1.2:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _generate_historical_recommendations(self, history: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on historical patterns"""
        recommendations = []
        
        if len(history) > 5:
            # Frequent competitive discussions
            recommendations.append("High frequency of competitive discussions - consider proactive competitive positioning")
        
        # Check for repeated competitors
        all_competitors = []
        for entry in history:
            all_competitors.extend(entry['competitors'])
        
        competitor_counts = Counter(all_competitors)
        for competitor, count in competitor_counts.most_common(2):
            if count > 2:
                recommendations.append(f"Frequent mentions of {competitor} - ensure battlecard is up-to-date")
        
        # Check for urgency patterns
        recent_urgency = []
        for entry in history[-3:]:
            for mention in entry['mentions']:
                recent_urgency.append(mention['response_urgency'])
        
        if recent_urgency.count('high') > len(recent_urgency) / 2:
            recommendations.append("High urgency patterns detected - competitive pressure increasing")
        
        return recommendations


# Global competitor intelligence manager instance
competitor_intelligence = CompetitorIntelligenceManager()