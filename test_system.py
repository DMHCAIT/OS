#!/usr/bin/env python3
"""
Test script to demonstrate the AI Lead Management System capabilities
"""

import requests
import json
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8001"

def test_api_health():
    """Test API health endpoint"""
    print("🏥 Testing API Health...")
    
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"📋 Services: {data['services']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_nlp_demo():
    """Test NLP demo endpoint"""
    print("\n🧠 Testing NLP Demo...")
    
    response = requests.get(f"{BASE_URL}/nlp/demo")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Demo Text: {data['demo_text'][:50]}...")
        
        analysis = data['analysis']
        print(f"📊 Sentiment: {analysis['sentiment']['label']} ({analysis['sentiment']['score']:.2f})")
        print(f"🎯 Intent: {analysis['intent']}")
        print(f"⚡ Urgency: {', '.join(analysis['urgency_indicators'])}")
        print(f"🏷️  Topics: {', '.join(analysis['topics'])}")
        print(f"🔑 Keywords: {', '.join(analysis['keywords'])}")
        
        if analysis['entities']:
            print(f"🏢 Entities: {', '.join([e['text'] for e in analysis['entities']])}")
        
        return True
    else:
        print(f"❌ NLP demo failed: {response.status_code}")
        return False

def test_text_analysis():
    """Test custom text analysis"""
    print("\n📝 Testing Custom Text Analysis...")
    
    # Test different types of lead communications
    test_texts = [
        "I need pricing information for 100 users urgently - can you call me today?",
        "Your product looks great but it's too expensive for our budget",
        "We're currently using HubSpot but looking for alternatives",
        "Can you schedule a demo for next week? Our team is very interested",
        "I have some technical questions about your API integration"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Analyzing: '{text[:40]}...'")
        
        response = requests.post(
            f"{BASE_URL}/nlp/analyze",
            params={"text": text}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   📈 Sentiment: {data['sentiment_label']} ({data['sentiment_score']:+.2f})")
            print(f"   🎯 Intent: {data['intent']}")
            print(f"   ⚡ Urgency: {', '.join(data['urgency_indicators']) if data['urgency_indicators'] else 'None'}")
            print(f"   🏷️  Topics: {', '.join(data['topics'])}")
        else:
            print(f"   ❌ Analysis failed: {response.status_code}")

def test_similarity():
    """Test text similarity"""
    print("\n🔍 Testing Text Similarity...")
    
    text_pairs = [
        ("I need pricing information", "Can you send me your rates?"),
        ("Schedule a demo please", "I want to see a product demonstration"),
        ("Your software is expensive", "I love your features but can't afford it"),
        ("Integration with Salesforce", "Connect to our CRM system")
    ]
    
    for i, (text1, text2) in enumerate(text_pairs, 1):
        print(f"\n{i}. Comparing:")
        print(f"   Text 1: '{text1}'")
        print(f"   Text 2: '{text2}'")
        
        response = requests.post(
            f"{BASE_URL}/nlp/similarity",
            params={"text1": text1, "text2": text2}
        )
        
        if response.status_code == 200:
            data = response.json()
            similarity = data['semantic_similarity']
            interpretation = data['interpretation']
            print(f"   📊 Similarity: {similarity:.3f} ({interpretation})")
        else:
            print(f"   ❌ Similarity test failed: {response.status_code}")

def main():
    """Main test function"""
    print("🚀 AI Lead Management System - API Tests")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("❌ API is not responding. Please check if the server is running.")
        return
    
    # Test NLP capabilities
    test_nlp_demo()
    test_text_analysis()
    test_similarity()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("🌐 Visit http://localhost:8001/docs for interactive API documentation")
    print("🧠 Visit http://localhost:8001/nlp/demo for NLP demonstration")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the server. Please make sure it's running on http://localhost:8001")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")