# 🧠 Advanced NLP Service Documentation

## Overview

The Advanced NLP Service provides comprehensive text analysis capabilities for the Lead Management System, including sentiment analysis, entity extraction, intent classification, and semantic search. This service enhances lead qualification, conversation analysis, and customer interaction understanding.

## Features

### 🎯 Core NLP Capabilities

1. **Sentiment Analysis**
   - Real-time sentiment scoring (-1 to +1 scale)
   - Emotion classification (positive/negative/neutral)
   - VADER sentiment analysis for social media text

2. **Entity Extraction**
   - Named Entity Recognition (NER) using spaCy
   - Business-specific entity detection (emails, phones, companies)
   - Custom entity patterns for CRM data

3. **Intent Classification** 
   - Automatic categorization of customer inquiries
   - Support for: pricing, demo requests, technical questions, support, features, competition
   - Confidence scoring for classification accuracy

4. **Keyword & Topic Extraction**
   - TF-IDF based keyword extraction
   - Topic modeling for conversation themes
   - Business value indicator detection

5. **Semantic Search**
   - Vector-based similarity search using SentenceTransformers
   - Knowledge base search with relevance scoring
   - Support for multiple similarity methods

6. **Conversation Analysis**
   - Multi-turn conversation sentiment tracking
   - Objection detection and classification
   - Topic evolution analysis

## Libraries Used

### Primary NLP Libraries

- **spaCy** (3.7.2+): Industrial-strength NLP with fast tokenization and NER
- **NLTK** (3.8.2+): Classic NLP utilities and sentiment analysis
- **SentenceTransformers** (2.2.2+): State-of-the-art sentence embeddings
- **textdistance** (4.6.0+): Multiple text similarity algorithms

### Supporting Libraries

- **NumPy**: Numerical computations for embeddings
- **scikit-learn**: Clustering and machine learning utilities
- **pandas**: Data manipulation for NLP results

## API Endpoints

### Text Analysis

```http
POST /api/v1/nlp/analyze-text
```

Comprehensive text analysis including sentiment, entities, keywords, and intent.

**Request:**
```json
{
    "text": "I'm interested in your pricing but need a demo first",
    "context": {
        "source": "email",
        "lead_stage": "consideration"
    }
}
```

**Response:**
```json
{
    "sentiment_score": 0.6,
    "sentiment_label": "positive",
    "entities": [
        {
            "text": "pricing",
            "label": "TOPIC",
            "confidence": 0.9
        }
    ],
    "keywords": ["interested", "pricing", "demo"],
    "topics": ["pricing", "demo"],
    "language": "en",
    "readability_score": 0.8,
    "urgency_indicators": [],
    "intent_classification": "pricing_inquiry",
    "confidence": 0.85
}
```

### Semantic Search

```http
POST /api/v1/nlp/semantic-search
```

Search through text collections using semantic similarity.

**Request:**
```json
{
    "query": "How much does it cost?",
    "texts": [
        "Pricing starts at $99/month for up to 1000 contacts",
        "We offer a 14-day free trial with full access",
        "Custom enterprise plans available"
    ],
    "top_k": 3
}
```

**Response:**
```json
{
    "results": [
        {
            "text": "Pricing starts at $99/month for up to 1000 contacts",
            "score": 0.89,
            "metadata": {"index": 0}
        },
        {
            "text": "Custom enterprise plans available", 
            "score": 0.72,
            "metadata": {"index": 2}
        }
    ],
    "query": "How much does it cost?",
    "total_searched": 3
}
```

### Conversation Analysis

```http
POST /api/v1/nlp/analyze-conversation
```

Analyze entire conversation threads for sentiment progression and objections.

**Request:**
```json
{
    "messages": [
        {
            "content": "Tell me about your pricing",
            "speaker": "prospect",
            "timestamp": "2024-01-15T10:00:00Z"
        },
        {
            "content": "Our plans start at $99/month",
            "speaker": "sales_rep", 
            "timestamp": "2024-01-15T10:01:00Z"
        }
    ]
}
```

**Response:**
```json
{
    "overall_analysis": {
        "sentiment_score": 0.3,
        "intent_classification": "pricing_inquiry"
    },
    "message_count": 2,
    "avg_message_length": 25.5,
    "sentiment_progression": [
        {
            "timestamp": "2024-01-15T10:00:00Z",
            "sentiment": 0.2,
            "speaker": "prospect"
        }
    ],
    "objections_timeline": []
}
```

### Objection Detection

```http
POST /api/v1/nlp/detect-objections
```

Detect and classify customer objections.

**Request:**
```json
{
    "text": "This looks expensive for our small team budget"
}
```

**Response:**
```json
{
    "objections": [
        {
            "type": "price",
            "indicator": "expensive",
            "confidence": 0.9,
            "context": "This looks expensive for our small team budget"
        }
    ],
    "total_found": 1
}
```

## Integration Examples

### Lead Scoring Enhancement

```python
from app.core.nlp_service import nlp_service

async def analyze_lead_communication(lead_id: str, message: str):
    """Analyze lead communication for scoring"""
    
    # Perform NLP analysis
    analysis = nlp_service.analyze_text(message)
    
    # Extract insights for lead scoring
    insights = {
        "sentiment": analysis.sentiment_score,
        "intent": analysis.intent_classification,
        "urgency": len(analysis.urgency_indicators),
        "topics": analysis.topics,
        "objections": nlp_service.extract_objections(message)
    }
    
    # Update lead score based on insights
    return insights
```

### Voice Call Analysis

```python
async def analyze_call_transcript(call_id: str, transcript: str):
    """Analyze voice call transcript"""
    
    # Split transcript by speaker
    messages = parse_call_transcript(transcript)
    
    # Analyze conversation
    conversation_analysis = nlp_service.analyze_conversation(messages)
    
    # Extract actionable insights
    return {
        "sentiment_trend": conversation_analysis["sentiment_progression"],
        "objections_raised": conversation_analysis["objections_timeline"],
        "topics_discussed": conversation_analysis["overall_analysis"]["topics"],
        "next_actions": generate_next_actions(conversation_analysis)
    }
```

### Knowledge Base Search

```python
async def search_sales_knowledge(query: str):
    """Search sales knowledge base semantically"""
    
    # Load knowledge base
    knowledge_items = await load_knowledge_base()
    texts = [item["content"] for item in knowledge_items]
    
    # Perform semantic search
    results = nlp_service.semantic_search(query, texts, top_k=5)
    
    # Return relevant information
    return [
        {
            "content": result.text,
            "relevance": result.score,
            "source": knowledge_items[result.metadata["index"]]["source"]
        }
        for result in results
    ]
```

## Configuration

### Environment Variables

```bash
# NLP Service Configuration
NLP_MODEL_PATH=/path/to/models
SPACY_MODEL=en_core_web_sm
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
NLP_CACHE_SIZE=1000
```

### Model Downloads

The setup script automatically downloads required models:

```bash
# spaCy English model
python -m spacy download en_core_web_sm

# NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Performance Optimization

### Caching Strategy

- Embedding caching for frequently analyzed texts
- Model lazy loading to reduce startup time
- Batch processing for multiple text analysis

### Memory Management

- Model sharing across requests
- Efficient tokenization with spaCy
- Optimized embedding storage

## Error Handling

### Common Issues

1. **Missing Models**: Automatically downloads on first use
2. **Memory Limits**: Configurable batch sizes for large texts
3. **Language Detection**: Fallback to English for unsupported languages

### Monitoring

```python
# Health check endpoint
GET /api/v1/nlp/health

# Response
{
    "status": "healthy",
    "models_loaded": true,
    "test_result": {
        "sentiment": "positive",
        "entities_count": 2,
        "keywords_count": 5
    }
}
```

## Advanced Features

### Custom Entity Recognition

```python
# Add custom business entities
nlp_service.add_custom_entity_pattern("PRODUCT", ["CRM", "lead management", "sales tool"])
```

### Multilingual Support

```python
# Extend to other languages
nlp_service.add_language_model("es", "es_core_news_sm")
```

### Custom Intent Classification

```python
# Add new intent categories
nlp_service.add_intent_pattern("refund_request", ["refund", "money back", "cancel subscription"])
```

## Usage Examples

### Demo Script

Run the demonstration script to see all capabilities:

```bash
python scripts/demo_nlp.py
```

### Interactive Testing

```python
from app.core.nlp_service import nlp_service

# Quick text analysis
result = nlp_service.analyze_text("I love your product but it's too expensive")
print(f"Sentiment: {result.sentiment_label}")
print(f"Intent: {result.intent_classification}")
print(f"Objections: {nlp_service.extract_objections(text)}")
```

## Best Practices

1. **Text Preprocessing**: Clean text before analysis for better accuracy
2. **Context Provision**: Include relevant context for better intent classification
3. **Batch Processing**: Use batch operations for multiple texts
4. **Result Caching**: Cache results for identical texts
5. **Regular Updates**: Keep models updated for better performance

## Troubleshooting

### Installation Issues

```bash
# If spaCy model download fails
python -m spacy download en_core_web_sm --user

# If NLTK data download fails
python -c "import nltk; nltk.download('all')"

# If sentence-transformers fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Performance Issues

- Reduce batch size for memory constraints
- Use CPU-only mode for development
- Enable model caching for production

### Accuracy Issues

- Provide more context in requests
- Use domain-specific training data
- Fine-tune models for specific use cases

## Future Enhancements

1. **Custom Model Training**: Train models on domain-specific data
2. **Real-time Processing**: Stream processing for live conversations
3. **Advanced Analytics**: Trend analysis and pattern recognition
4. **Multi-modal Analysis**: Support for audio and image analysis
5. **Federated Learning**: Privacy-preserving model improvements