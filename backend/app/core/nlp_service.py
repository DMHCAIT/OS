"""
Advanced NLP Service for Lead Management System
Provides text analysis, entity extraction, semantic search, and sentiment analysis
"""

import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import textdistance
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TextAnalysisResult:
    """Result of comprehensive text analysis"""
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # positive, negative, neutral
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[str]
    language: str
    readability_score: float
    urgency_indicators: List[str]
    intent_classification: str
    confidence: float


@dataclass
class SemanticSearchResult:
    """Result of semantic search"""
    text: str
    score: float
    metadata: Dict[str, Any]


class AdvancedNLPService:
    """Advanced NLP service using spaCy, NLTK, and SentenceTransformers"""
    
    def __init__(self):
        self._initialize_models()
        self._load_custom_patterns()
        
    def _initialize_models(self):
        """Initialize all NLP models and resources"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
                # Use blank model as fallback
                self.nlp = spacy.blank("en")
            
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize NLTK sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get stopwords
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            raise
    
    def _load_custom_patterns(self):
        """Load custom patterns for business-specific analysis"""
        
        # Intent classification patterns
        self.intent_patterns = {
            'pricing_inquiry': [
                'price', 'cost', 'budget', 'expensive', 'cheap', 'affordable',
                'pricing', 'fees', 'rates', 'quote', 'estimate'
            ],
            'demo_request': [
                'demo', 'demonstration', 'show', 'see', 'preview', 'trial',
                'test', 'example', 'walkthrough'
            ],
            'technical_inquiry': [
                'technical', 'integration', 'api', 'setup', 'configuration',
                'requirements', 'compatibility', 'specifications'
            ],
            'support_request': [
                'help', 'support', 'problem', 'issue', 'bug', 'error',
                'assistance', 'question', 'trouble'
            ],
            'feature_inquiry': [
                'feature', 'functionality', 'capability', 'can it', 'does it',
                'ability', 'function', 'option'
            ],
            'competitor_comparison': [
                'competitor', 'alternative', 'versus', 'compared to', 'better than',
                'different from', 'similar to'
            ]
        }
        
        # Urgency indicators
        self.urgency_patterns = [
            'urgent', 'asap', 'immediately', 'right away', 'emergency',
            'critical', 'time sensitive', 'deadline', 'rush', 'priority',
            'today', 'tomorrow', 'this week', 'end of month', 'quarter'
        ]
        
        # Business value indicators
        self.value_patterns = [
            'roi', 'return on investment', 'save money', 'reduce costs',
            'increase revenue', 'efficiency', 'productivity', 'streamline',
            'automate', 'scale', 'growth', 'competitive advantage'
        ]
        
        # Objection patterns
        self.objection_patterns = {
            'price': ['expensive', 'costly', 'budget', 'afford', 'cheap'],
            'timing': ['not now', 'later', 'busy', 'time', 'schedule'],
            'authority': ['decision', 'manager', 'boss', 'team', 'approval'],
            'need': ['not need', 'satisfied', 'current solution', 'working fine'],
            'trust': ['proven', 'references', 'track record', 'experience']
        }
    
    def analyze_text(self, text: str, context: Dict[str, Any] = None) -> TextAnalysisResult:
        """Comprehensive text analysis"""
        
        if not text or not text.strip():
            return TextAnalysisResult(
                sentiment_score=0.0,
                sentiment_label="neutral",
                entities=[],
                keywords=[],
                topics=[],
                language="en",
                readability_score=0.0,
                urgency_indicators=[],
                intent_classification="unknown",
                confidence=0.0
            )
        
        # Clean text
        clean_text = self._clean_text(text)
        
        # Sentiment analysis
        sentiment_score, sentiment_label = self._analyze_sentiment(clean_text)
        
        # Entity extraction
        entities = self._extract_entities(clean_text)
        
        # Keyword extraction
        keywords = self._extract_keywords(clean_text)
        
        # Topic modeling
        topics = self._extract_topics(clean_text)
        
        # Language detection
        language = self._detect_language(clean_text)
        
        # Readability assessment
        readability_score = self._calculate_readability(clean_text)
        
        # Urgency detection
        urgency_indicators = self._detect_urgency(clean_text)
        
        # Intent classification
        intent_classification, confidence = self._classify_intent(clean_text)
        
        return TextAnalysisResult(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            entities=entities,
            keywords=keywords,
            topics=topics,
            language=language,
            readability_score=readability_score,
            urgency_indicators=urgency_indicators,
            intent_classification=intent_classification,
            confidence=confidence
        )
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using NLTK VADER"""
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        # Determine label
        if compound_score >= 0.05:
            label = "positive"
        elif compound_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return compound_score, label
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, 'score', 1.0)
            })
        
        # Custom business entity extraction
        business_entities = self._extract_business_entities(text)
        entities.extend(business_entities)
        
        return entities
    
    def _extract_business_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract business-specific entities"""
        
        entities = []
        text_lower = text.lower()
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "EMAIL",
                "description": "Email address",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })
        
        # Phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "PHONE",
                "description": "Phone number",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Company indicators
        company_indicators = ['inc', 'corp', 'llc', 'ltd', 'company', 'corporation']
        for indicator in company_indicators:
            pattern = rf'\b\w+\s+{indicator}\b'
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": text[match.start():match.end()],
                    "label": "COMPANY",
                    "description": "Company name",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TF-IDF and spaCy"""
        
        doc = self.nlp(text)
        
        # Extract important tokens
        keywords = []
        for token in doc:
            if (token.is_alpha and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                keywords.append(token.lemma_.lower())
        
        # Count frequency and return top keywords
        keyword_freq = Counter(keywords)
        return [word for word, freq in keyword_freq.most_common(10)]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using pattern matching"""
        
        topics = []
        text_lower = text.lower()
        
        # Business topics
        topic_patterns = {
            'pricing': ['price', 'cost', 'budget', 'pricing', 'fee'],
            'product': ['product', 'feature', 'functionality', 'capability'],
            'support': ['support', 'help', 'assistance', 'service'],
            'integration': ['integration', 'api', 'setup', 'implementation'],
            'demo': ['demo', 'trial', 'preview', 'demonstration'],
            'competition': ['competitor', 'alternative', 'comparison']
        }
        
        for topic, patterns in topic_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                topics.append(topic)
        
        return topics
    
    def _detect_language(self, text: str) -> str:
        """Detect language (simplified implementation)"""
        
        # Use spaCy's language detection (if available)
        # For now, assume English
        return "en"
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (approximation)
        syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllables / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, score / 100.0))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        
        if word[0] in vowels:
            count += 1
        
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        
        if word.endswith("e"):
            count -= 1
        
        if count == 0:
            count += 1
        
        return count
    
    def _detect_urgency(self, text: str) -> List[str]:
        """Detect urgency indicators in text"""
        
        urgency_found = []
        text_lower = text.lower()
        
        for pattern in self.urgency_patterns:
            if pattern in text_lower:
                urgency_found.append(pattern)
        
        return urgency_found
    
    def _classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the text"""
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score / len(patterns)
        
        if not intent_scores:
            return "general_inquiry", 0.5
        
        # Get the intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def semantic_search(self, query: str, texts: List[str], top_k: int = 5) -> List[SemanticSearchResult]:
        """Perform semantic search across a collection of texts"""
        
        if not texts:
            return []
        
        # Encode query and texts
        query_embedding = self.sentence_model.encode([query])
        text_embeddings = self.sentence_model.encode(texts)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, text_embeddings.T)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold
                results.append(SemanticSearchResult(
                    text=texts[idx],
                    score=float(similarities[idx]),
                    metadata={"index": idx}
                ))
        
        return results
    
    def extract_objections(self, text: str) -> List[Dict[str, Any]]:
        """Extract and classify objections from text"""
        
        objections = []
        text_lower = text.lower()
        
        for objection_type, patterns in self.objection_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    objections.append({
                        "type": objection_type,
                        "indicator": pattern,
                        "confidence": 0.8,
                        "context": text  # Could extract surrounding context
                    })
        
        return objections
    
    def text_similarity(self, text1: str, text2: str, method: str = "jaccard") -> float:
        """Calculate text similarity using various methods"""
        
        if method == "jaccard":
            return textdistance.jaccard.normalized_similarity(text1, text2)
        elif method == "levenshtein":
            return textdistance.levenshtein.normalized_similarity(text1, text2)
        elif method == "cosine":
            return textdistance.cosine.normalized_similarity(text1, text2)
        elif method == "semantic":
            return self.semantic_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        
        return self.sentence_model.encode(texts)
    
    def cluster_texts(self, texts: List[str], n_clusters: int = None) -> Dict[str, Any]:
        """Cluster texts using embeddings (simplified without sklearn)"""
        
        if len(texts) < 2:
            return {"clusters": [0] * len(texts), "centers": [], "score": 0.0}
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Simple clustering based on similarity threshold
        if n_clusters is None:
            n_clusters = min(5, len(texts) // 2)
        
        # Simplified clustering algorithm
        clusters = [0] * len(texts)
        cluster_id = 0
        used = [False] * len(texts)
        
        for i in range(len(texts)):
            if used[i]:
                continue
                
            # Start new cluster
            used[i] = True
            clusters[i] = cluster_id
            
            # Find similar texts
            for j in range(i + 1, len(texts)):
                if used[j]:
                    continue
                    
                # Calculate similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity > 0.7:  # Similarity threshold
                    used[j] = True
                    clusters[j] = cluster_id
            
            cluster_id += 1
            if cluster_id >= n_clusters:
                break
        
        return {
            "clusters": clusters,
            "centers": [],  # Simplified - no cluster centers
            "score": 0.8,   # Simplified - fixed score
            "n_clusters": max(clusters) + 1 if clusters else 0
        }
    
    def analyze_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze a conversation thread"""
        
        if not messages:
            return {}
        
        # Combine all messages
        all_text = " ".join([msg.get("content", "") for msg in messages])
        
        # Basic analysis
        analysis = self.analyze_text(all_text)
        
        # Conversation-specific metrics
        conversation_analysis = {
            "overall_analysis": analysis.__dict__,
            "message_count": len(messages),
            "avg_message_length": len(all_text) / len(messages) if messages else 0,
            "sentiment_progression": [],
            "topic_evolution": [],
            "objections_timeline": []
        }
        
        # Analyze sentiment progression
        for msg in messages:
            if msg.get("content"):
                sentiment_score, _ = self._analyze_sentiment(msg["content"])
                conversation_analysis["sentiment_progression"].append({
                    "timestamp": msg.get("timestamp"),
                    "sentiment": sentiment_score,
                    "speaker": msg.get("speaker", "unknown")
                })
        
        # Analyze objections over time
        for i, msg in enumerate(messages):
            if msg.get("content"):
                objections = self.extract_objections(msg["content"])
                if objections:
                    conversation_analysis["objections_timeline"].append({
                        "message_index": i,
                        "timestamp": msg.get("timestamp"),
                        "objections": objections
                    })
        
        return conversation_analysis


# Global instance
nlp_service = AdvancedNLPService()