"""
NLP API endpoints for text analysis and semantic operations
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.core.nlp_service import nlp_service, TextAnalysisResult, SemanticSearchResult
from app.core.security import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis"""
    sentiment_score: float
    sentiment_label: str
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[str]
    language: str
    readability_score: float
    urgency_indicators: List[str]
    intent_classification: str
    confidence: float


class SimilarityRequest(BaseModel):
    """Request model for text similarity"""
    text1: str = Field(..., min_length=1, max_length=5000)
    text2: str = Field(..., min_length=1, max_length=5000)
    method: str = Field("semantic", description="Similarity method: jaccard, levenshtein, cosine, semantic")


class SimilarityResponse(BaseModel):
    """Response model for text similarity"""
    similarity: float
    method: str


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., min_length=1, max_length=1000)
    texts: List[str] = Field(..., min_items=1, max_items=100)
    top_k: int = Field(5, ge=1, le=20)


class SemanticSearchResponseItem(BaseModel):
    """Individual search result"""
    text: str
    score: float
    metadata: Dict[str, Any]


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search"""
    results: List[SemanticSearchResponseItem]
    query: str
    total_searched: int


class ConversationMessage(BaseModel):
    """Message in a conversation"""
    content: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1)
    timestamp: Optional[str] = None


class ConversationAnalysisRequest(BaseModel):
    """Request model for conversation analysis"""
    messages: List[ConversationMessage] = Field(..., min_items=1, max_items=100)


class ConversationAnalysisResponse(BaseModel):
    """Response model for conversation analysis"""
    overall_analysis: Dict[str, Any]
    message_count: int
    avg_message_length: float
    sentiment_progression: List[Dict[str, Any]]
    topic_evolution: List[str]
    objections_timeline: List[Dict[str, Any]]


class ClusteringRequest(BaseModel):
    """Request model for text clustering"""
    texts: List[str] = Field(..., min_items=2, max_items=100)
    n_clusters: Optional[int] = Field(None, ge=2, le=10)


class ClusteringResponse(BaseModel):
    """Response model for text clustering"""
    clusters: List[int]
    centers: List[List[float]]
    score: float
    n_clusters: int


class ObjectionDetectionRequest(BaseModel):
    """Request model for objection detection"""
    text: str = Field(..., min_length=1, max_length=5000)


class ObjectionDetectionResponse(BaseModel):
    """Response model for objection detection"""
    objections: List[Dict[str, Any]]
    total_found: int


@router.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform comprehensive text analysis including sentiment, entities, keywords, and intent
    """
    try:
        logger.info(f"Analyzing text for user {current_user.email}")
        
        # Perform analysis
        result = nlp_service.analyze_text(request.text, request.context)
        
        return TextAnalysisResponse(
            sentiment_score=result.sentiment_score,
            sentiment_label=result.sentiment_label,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            language=result.language,
            readability_score=result.readability_score,
            urgency_indicators=result.urgency_indicators,
            intent_classification=result.intent_classification,
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail="Text analysis failed")


@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(
    request: SimilarityRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate similarity between two texts using various methods
    """
    try:
        logger.info(f"Calculating similarity for user {current_user.email}")
        
        # Validate method
        valid_methods = ["jaccard", "levenshtein", "cosine", "semantic"]
        if request.method not in valid_methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Must be one of: {valid_methods}"
            )
        
        # Calculate similarity
        similarity = nlp_service.text_similarity(
            request.text1, 
            request.text2, 
            request.method
        )
        
        return SimilarityResponse(
            similarity=similarity,
            method=request.method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail="Similarity calculation failed")


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform semantic search across a collection of texts
    """
    try:
        logger.info(f"Performing semantic search for user {current_user.email}")
        
        # Perform search
        results = nlp_service.semantic_search(
            request.query, 
            request.texts, 
            request.top_k
        )
        
        # Convert results
        response_results = [
            SemanticSearchResponseItem(
                text=result.text,
                score=result.score,
                metadata=result.metadata
            )
            for result in results
        ]
        
        return SemanticSearchResponse(
            results=response_results,
            query=request.query,
            total_searched=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail="Semantic search failed")


@router.post("/analyze-conversation", response_model=ConversationAnalysisResponse)
async def analyze_conversation(
    request: ConversationAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze a conversation thread for sentiment progression, topics, and objections
    """
    try:
        logger.info(f"Analyzing conversation for user {current_user.email}")
        
        # Convert to dict format
        messages = [
            {
                "content": msg.content,
                "speaker": msg.speaker,
                "timestamp": msg.timestamp
            }
            for msg in request.messages
        ]
        
        # Perform analysis
        analysis = nlp_service.analyze_conversation(messages)
        
        return ConversationAnalysisResponse(
            overall_analysis=analysis.get("overall_analysis", {}),
            message_count=analysis.get("message_count", 0),
            avg_message_length=analysis.get("avg_message_length", 0.0),
            sentiment_progression=analysis.get("sentiment_progression", []),
            topic_evolution=analysis.get("topic_evolution", []),
            objections_timeline=analysis.get("objections_timeline", [])
        )
        
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        raise HTTPException(status_code=500, detail="Conversation analysis failed")


@router.post("/cluster-texts", response_model=ClusteringResponse)
async def cluster_texts(
    request: ClusteringRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Cluster texts using semantic embeddings
    """
    try:
        logger.info(f"Clustering texts for user {current_user.email}")
        
        # Perform clustering
        result = nlp_service.cluster_texts(request.texts, request.n_clusters)
        
        return ClusteringResponse(
            clusters=result["clusters"],
            centers=result["centers"],
            score=result["score"],
            n_clusters=result["n_clusters"]
        )
        
    except Exception as e:
        logger.error(f"Error clustering texts: {e}")
        raise HTTPException(status_code=500, detail="Text clustering failed")


@router.post("/detect-objections", response_model=ObjectionDetectionResponse)
async def detect_objections(
    request: ObjectionDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Detect and classify objections in text
    """
    try:
        logger.info(f"Detecting objections for user {current_user.email}")
        
        # Detect objections
        objections = nlp_service.extract_objections(request.text)
        
        return ObjectionDetectionResponse(
            objections=objections,
            total_found=len(objections)
        )
        
    except Exception as e:
        logger.error(f"Error detecting objections: {e}")
        raise HTTPException(status_code=500, detail="Objection detection failed")


@router.get("/health")
async def nlp_health_check():
    """
    Health check for NLP service
    """
    try:
        # Test basic functionality
        test_analysis = nlp_service.analyze_text("This is a test message.")
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "test_result": {
                "sentiment": test_analysis.sentiment_label,
                "entities_count": len(test_analysis.entities),
                "keywords_count": len(test_analysis.keywords)
            }
        }
        
    except Exception as e:
        logger.error(f"NLP health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "models_loaded": False
        }


@router.get("/info")
async def nlp_info():
    """
    Get information about available NLP capabilities
    """
    return {
        "capabilities": [
            "sentiment_analysis",
            "entity_extraction",
            "keyword_extraction",
            "topic_modeling",
            "intent_classification",
            "urgency_detection",
            "objection_detection",
            "semantic_similarity",
            "semantic_search",
            "text_clustering",
            "conversation_analysis"
        ],
        "supported_languages": ["en"],
        "similarity_methods": ["jaccard", "levenshtein", "cosine", "semantic"],
        "intent_types": [
            "pricing_inquiry",
            "demo_request", 
            "technical_inquiry",
            "support_request",
            "feature_inquiry",
            "competitor_comparison",
            "general_inquiry"
        ],
        "objection_types": ["price", "timing", "authority", "need", "trust"]
    }