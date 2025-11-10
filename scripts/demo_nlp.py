#!/usr/bin/env python3
"""
Advanced NLP Service Demonstration Script
Shows how to use the enhanced text analysis capabilities
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the backend directory to the Python path
sys.path.append('/Users/rubeenakhan/Desktop/OS/backend')

# Note: This demo shows the API structure. 
# Actual usage requires installing dependencies: pip install -r requirements.txt


async def demo_text_analysis():
    """Demonstrate text analysis capabilities"""
    
    print("=" * 60)
    print("🧠 ADVANCED NLP SERVICE DEMONSTRATION")
    print("=" * 60)
    
    # Sample lead communication texts
    sample_texts = [
        {
            "title": "Pricing Inquiry with Urgency",
            "text": "Hi, I'm very interested in your product but need to know about pricing ASAP. We have a deadline next week and need to make a decision quickly. Can you send me a quote today?",
            "context": {"source": "email", "lead_stage": "consideration"}
        },
        {
            "title": "Technical Integration Question",
            "text": "Our development team is evaluating your API. We need to integrate with Salesforce and have specific security requirements. Do you support OAuth 2.0 and can we get sandbox access for testing?",
            "context": {"source": "support_ticket", "lead_stage": "evaluation"}
        },
        {
            "title": "Competitive Comparison",
            "text": "We're currently using HubSpot but looking for alternatives. How does your solution compare in terms of features and pricing? We're particularly interested in better analytics and reporting capabilities.",
            "context": {"source": "sales_call", "lead_stage": "comparison"}
        },
        {
            "title": "Budget Objection",
            "text": "Your solution looks great but it's quite expensive for our small team. We don't have a huge budget right now. Are there any more affordable options or payment plans available?",
            "context": {"source": "demo_feedback", "lead_stage": "negotiation"}
        },
        {
            "title": "Feature Request",
            "text": "We love the core functionality but really need mobile app support and offline capabilities. When will these features be available? This is crucial for our field team.",
            "context": {"source": "feature_request", "lead_stage": "consideration"}
        }
    ]
    
    # Note: In actual implementation, you would initialize the NLP service
    # from app.core.nlp_service import nlp_service
    
    print("\n📊 TEXT ANALYSIS RESULTS:")
    print("-" * 40)
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n{i}. {sample['title']}")
        print(f"Text: {sample['text'][:100]}...")
        
        # Simulated analysis results (what the real service would return)
        simulated_results = simulate_analysis(sample['text'], sample['title'])
        
        print(f"📈 Sentiment: {simulated_results['sentiment_label']} ({simulated_results['sentiment_score']:.2f})")
        print(f"🎯 Intent: {simulated_results['intent_classification']}")
        print(f"⚡ Urgency: {', '.join(simulated_results['urgency_indicators']) if simulated_results['urgency_indicators'] else 'None'}")
        print(f"🏷️  Topics: {', '.join(simulated_results['topics'])}")
        print(f"🔑 Keywords: {', '.join(simulated_results['keywords'][:5])}")
        
        if simulated_results['entities']:
            print(f"🏢 Entities: {', '.join([e['text'] for e in simulated_results['entities'][:3]])}")
        
        if simulated_results['objections']:
            print(f"🚫 Objections: {', '.join([o['type'] for o in simulated_results['objections']])}")


def simulate_analysis(text, title):
    """Simulate NLP analysis results for demonstration"""
    
    # This simulates what the actual NLP service would return
    results = {
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "intent_classification": "general_inquiry",
        "urgency_indicators": [],
        "topics": [],
        "keywords": [],
        "entities": [],
        "objections": []
    }
    
    text_lower = text.lower()
    
    # Sentiment analysis simulation
    positive_words = ["great", "love", "interested", "excellent", "good"]
    negative_words = ["expensive", "problem", "issue", "difficult", "concern"]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        results["sentiment_score"] = 0.6
        results["sentiment_label"] = "positive"
    elif neg_count > pos_count:
        results["sentiment_score"] = -0.4
        results["sentiment_label"] = "negative"
    else:
        results["sentiment_score"] = 0.1
        results["sentiment_label"] = "neutral"
    
    # Intent classification
    if "pricing" in title.lower() or "price" in text_lower or "cost" in text_lower:
        results["intent_classification"] = "pricing_inquiry"
    elif "technical" in title.lower() or "api" in text_lower or "integration" in text_lower:
        results["intent_classification"] = "technical_inquiry"
    elif "demo" in text_lower or "trial" in text_lower:
        results["intent_classification"] = "demo_request"
    elif "feature" in text_lower or "functionality" in text_lower:
        results["intent_classification"] = "feature_inquiry"
    elif "compare" in text_lower or "alternative" in text_lower:
        results["intent_classification"] = "competitor_comparison"
    
    # Urgency detection
    urgency_words = ["asap", "urgent", "quickly", "deadline", "today", "immediately"]
    results["urgency_indicators"] = [word for word in urgency_words if word in text_lower]
    
    # Topic extraction
    topic_mapping = {
        "pricing": ["price", "cost", "budget", "expensive", "affordable"],
        "integration": ["api", "integrate", "salesforce", "oauth"],
        "features": ["feature", "functionality", "capability", "mobile", "offline"],
        "comparison": ["compare", "alternative", "hubspot", "versus"]
    }
    
    for topic, keywords in topic_mapping.items():
        if any(keyword in text_lower for keyword in keywords):
            results["topics"].append(topic)
    
    # Keywords (simplified)
    import re
    words = re.findall(r'\b\w+\b', text_lower)
    common_words = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"])
    keywords = [word for word in words if len(word) > 3 and word not in common_words]
    results["keywords"] = list(set(keywords))[:10]
    
    # Entity simulation
    if "salesforce" in text_lower:
        results["entities"].append({"text": "Salesforce", "label": "ORG"})
    if "hubspot" in text_lower:
        results["entities"].append({"text": "HubSpot", "label": "ORG"})
    if "oauth" in text_lower:
        results["entities"].append({"text": "OAuth 2.0", "label": "TECH"})
    
    # Objection detection
    if "expensive" in text_lower or "budget" in text_lower:
        results["objections"].append({"type": "price", "indicator": "expensive"})
    if "deadline" in text_lower or "time" in text_lower:
        results["objections"].append({"type": "timing", "indicator": "deadline"})
    
    return results


async def demo_semantic_search():
    """Demonstrate semantic search capabilities"""
    
    print("\n\n🔍 SEMANTIC SEARCH DEMONSTRATION:")
    print("-" * 40)
    
    # Sample knowledge base
    knowledge_base = [
        "Our CRM system integrates with Salesforce, HubSpot, and Pipedrive through REST APIs",
        "Pricing starts at $99/month for up to 1000 contacts with premium features",
        "We offer a 14-day free trial with full access to all features",
        "Our mobile app is available for iOS and Android with offline sync capabilities",
        "Advanced analytics include lead scoring, conversion tracking, and ROI analysis",
        "Customer support is available 24/7 via chat, email, and phone",
        "Data security includes SOC2 compliance, encryption at rest, and GDPR compliance",
        "Custom integrations can be built using our webhook system and API",
        "Bulk import supports CSV, Excel, and direct database connections",
        "Automated workflows can trigger emails, tasks, and follow-up reminders"
    ]
    
    # Sample queries
    queries = [
        "What integrations do you support?",
        "How much does it cost?",
        "Do you have a mobile app?",
        "What security features are available?",
        "Can I try it for free?"
    ]
    
    print("Knowledge Base:")
    for i, kb_item in enumerate(knowledge_base[:3], 1):
        print(f"{i}. {kb_item}")
    print("... (7 more items)")
    
    print("\nSemantic Search Results:")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        # Simulate semantic search results
        results = simulate_semantic_search(query, knowledge_base)
        for i, result in enumerate(results[:2], 1):
            print(f"  {i}. (Score: {result['score']:.3f}) {result['text']}")


def simulate_semantic_search(query, knowledge_base):
    """Simulate semantic search results"""
    
    # Simple keyword-based simulation (real implementation uses embeddings)
    query_words = set(query.lower().split())
    results = []
    
    for text in knowledge_base:
        text_words = set(text.lower().split())
        overlap = len(query_words.intersection(text_words))
        score = overlap / len(query_words) if query_words else 0
        
        # Add some context-based scoring
        if "integration" in query.lower() and "integrate" in text.lower():
            score += 0.3
        if "cost" in query.lower() and ("pricing" in text.lower() or "$" in text):
            score += 0.3
        if "mobile" in query.lower() and "mobile" in text.lower():
            score += 0.4
        if "security" in query.lower() and ("security" in text.lower() or "encryption" in text.lower()):
            score += 0.3
        if "free" in query.lower() and ("trial" in text.lower() or "free" in text.lower()):
            score += 0.4
        
        if score > 0:
            results.append({"text": text, "score": min(score, 1.0)})
    
    return sorted(results, key=lambda x: x["score"], reverse=True)


async def demo_conversation_analysis():
    """Demonstrate conversation analysis"""
    
    print("\n\n💬 CONVERSATION ANALYSIS DEMONSTRATION:")
    print("-" * 40)
    
    # Sample sales conversation
    conversation = [
        {"speaker": "prospect", "content": "Hi, I'm interested in learning more about your CRM solution."},
        {"speaker": "sales_rep", "content": "Great! I'd be happy to help. What's your biggest challenge with your current system?"},
        {"speaker": "prospect", "content": "We're using spreadsheets right now and it's getting messy. We need better organization."},
        {"speaker": "sales_rep", "content": "I understand. Our CRM can definitely help with that. How many contacts are you managing?"},
        {"speaker": "prospect", "content": "About 500 contacts. But we're growing fast and expect to double that this year."},
        {"speaker": "sales_rep", "content": "Perfect timing then. Our system scales easily. Would you like to see a demo?"},
        {"speaker": "prospect", "content": "Yes, but I'm concerned about the cost. We're a small startup with limited budget."},
        {"speaker": "sales_rep", "content": "I understand budget concerns. We have flexible pricing options. Let me show you our startup plan."},
        {"speaker": "prospect", "content": "That sounds good. When can we schedule the demo?"},
        {"speaker": "sales_rep", "content": "How about tomorrow at 2 PM? I can show you exactly how it would work for your use case."}
    ]
    
    print("Conversation Summary:")
    print(f"Messages: {len(conversation)}")
    
    # Analyze sentiment progression
    print("\nSentiment Progression:")
    for i, msg in enumerate(conversation):
        sentiment = simulate_sentiment(msg["content"])
        print(f"  {i+1}. {msg['speaker']}: {sentiment['label']} ({sentiment['score']:+.2f})")
    
    # Detect objections
    objections = []
    for i, msg in enumerate(conversation):
        if msg["speaker"] == "prospect" and ("cost" in msg["content"].lower() or "budget" in msg["content"].lower()):
            objections.append({"message": i+1, "type": "price", "content": msg["content"]})
    
    if objections:
        print(f"\nObjections Detected:")
        for obj in objections:
            print(f"  Message {obj['message']}: {obj['type']} objection")
            print(f"    '{obj['content']}'")
    
    # Overall analysis
    all_text = " ".join([msg["content"] for msg in conversation])
    overall_sentiment = simulate_sentiment(all_text)
    
    print(f"\nOverall Analysis:")
    print(f"  Sentiment: {overall_sentiment['label']} ({overall_sentiment['score']:+.2f})")
    print(f"  Objections: {len(objections)} found")
    print(f"  Outcome: Positive - Demo scheduled")


def simulate_sentiment(text):
    """Simulate sentiment analysis"""
    positive_words = ["great", "good", "interested", "perfect", "yes", "sounds good"]
    negative_words = ["concerned", "problem", "issue", "expensive", "limited", "messy"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {"score": 0.5, "label": "positive"}
    elif neg_count > pos_count:
        return {"score": -0.3, "label": "negative"}
    else:
        return {"score": 0.1, "label": "neutral"}


async def demo_api_usage():
    """Demonstrate API usage examples"""
    
    print("\n\n🚀 API USAGE EXAMPLES:")
    print("-" * 40)
    
    api_examples = [
        {
            "endpoint": "POST /api/v1/nlp/analyze-text",
            "description": "Analyze any text for sentiment, entities, intent",
            "request": {
                "text": "I'm interested in your pricing but need a demo first",
                "context": {"source": "email", "lead_stage": "consideration"}
            }
        },
        {
            "endpoint": "POST /api/v1/nlp/semantic-search", 
            "description": "Search knowledge base semantically",
            "request": {
                "query": "How much does it cost?",
                "texts": ["Pricing starts at $99/month", "Free trial available"],
                "top_k": 5
            }
        },
        {
            "endpoint": "POST /api/v1/nlp/analyze-conversation",
            "description": "Analyze entire conversation threads",
            "request": {
                "messages": [
                    {"speaker": "prospect", "content": "Tell me about pricing"},
                    {"speaker": "sales_rep", "content": "Our plans start at $99/month"}
                ]
            }
        },
        {
            "endpoint": "POST /api/v1/nlp/detect-objections",
            "description": "Detect and classify objections",
            "request": {
                "text": "This looks expensive for our small team budget"
            }
        }
    ]
    
    for example in api_examples:
        print(f"\n{example['endpoint']}")
        print(f"Description: {example['description']}")
        print(f"Request: {json.dumps(example['request'], indent=2)}")


async def main():
    """Main demonstration function"""
    
    print("🚀 Starting Advanced NLP Service Demonstration...")
    print("\nNote: This demo shows capabilities and API structure.")
    print("To run with real data, install dependencies: pip install -r requirements.txt")
    print("Then install spaCy model: python -m spacy download en_core_web_sm")
    
    await demo_text_analysis()
    await demo_semantic_search() 
    await demo_conversation_analysis()
    await demo_api_usage()
    
    print("\n\n✅ DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("🎯 Key Benefits of Advanced NLP Integration:")
    print("  • Automatic lead intent classification")
    print("  • Real-time sentiment analysis") 
    print("  • Intelligent objection detection")
    print("  • Semantic search for knowledge base")
    print("  • Conversation quality scoring")
    print("  • Entity extraction for data enrichment")
    print("  • Topic modeling for trend analysis")
    print("  • Multi-language support (extensible)")
    print("\n🚀 Ready for production deployment!")
    print("Run: ./scripts/setup.sh && ./scripts/start-dev.sh")


if __name__ == "__main__":
    asyncio.run(main())